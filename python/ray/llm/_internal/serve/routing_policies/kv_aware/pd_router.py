"""Sequential P/D coordination for the direct-streaming ingress router."""

import base64
import json
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Dict, List

from fastapi import HTTPException, Request

from ray.llm._internal.serve.core.configs.openai_api_models import (
    ChatCompletionRequest,
    CompletionRequest,
)
from ray.llm._internal.serve.core.ingress.utils import _sanitize_chat_completion_request
from ray.llm._internal.serve.core.protocol import (
    RawRequestInfo,
)
from ray.llm._internal.serve.routing_policies.kv_aware.constants import (
    KV_TRANSFER_PARAMS_HEADER,
)
from ray.serve._private.common import DeploymentID
from ray.serve._private.constants import (
    RAY_SERVE_INGRESS_REQUEST_ROUTER_OPT_HEADERS_FIELD,
    SERVE_INGRESS_ROUTER_REQUEST_PATH_HEADER,
)
from ray.serve._private.http_util import session_id_from_headers
from ray.serve.context import _get_serve_request_context
from ray.serve.handle import DeploymentHandle

# The tracker imports optional Dynamo code; only its type is needed here.
if TYPE_CHECKING:
    from ray.llm._internal.serve.routing_policies.kv_aware.kv_token_tracker import (
        KVTokenTracker,
    )

DecodeRouter = Callable[[SimpleNamespace, List[int], str], Awaitable[Dict[str, Any]]]


class PDRequestCoordinator:
    """Run prefill before selecting decode for a direct-streaming request.

    Each ingress replica owns one coordinator with separate P and D trackers.
    The prefill handle selects and invokes P. After P finishes, the decode
    callback selects D and stages tokens. The returned routing response carries
    KV transfer metadata to HAProxy; D streams through HAProxy to the client.
    """

    def __init__(
        self,
        prefill: DeploymentHandle,
        prefill_kv_token_tracker: "KVTokenTracker",
        decode_kv_token_tracker: "KVTokenTracker",
        tokenizer: Any,
        route_decode: DecodeRouter,
        router_destination: DeploymentID,
        get_multiplexed_model_id: Callable[[Dict[str, Any]], str],
    ) -> None:
        self.prefill = prefill
        self.prefill_kv_token_tracker = prefill_kv_token_tracker
        self.decode_kv_token_tracker = decode_kv_token_tracker
        self.tokenizer = tokenizer
        self.route_decode = route_decode
        self.router_destination = router_destination
        self.get_multiplexed_model_id = get_multiplexed_model_id

    async def route(self, request: Request, body: bytes) -> Dict[str, Any]:
        from ray.llm._internal.serve.routing_policies.kv_aware.vllm.tokenizer import (
            TokenizeError,
        )

        if "x-body-truncated" in request.headers:
            raise HTTPException(413, "P/D routing requires the complete request body")
        try:
            payload = json.loads(body)
            path = request.headers[SERVE_INGRESS_ROUTER_REQUEST_PATH_HEADER].split(
                "?", 1
            )[0]
            cls = (
                ChatCompletionRequest
                if path == "/v1/chat/completions"
                else CompletionRequest
            )
            parsed = cls.model_validate(payload)
            if isinstance(parsed, ChatCompletionRequest):
                parsed = _sanitize_chat_completion_request(parsed)
            if isinstance(parsed, CompletionRequest) and (
                not isinstance(parsed.prompt, str) or parsed.prompt_embeds is not None
            ):
                raise ValueError("KV-aware P/D requires a single string prompt")
            if parsed.kv_transfer_params is not None:
                raise ValueError("kv_transfer_params is managed by the P/D router")
            if isinstance(parsed, ChatCompletionRequest):
                for message in parsed.messages:
                    content = message.get("content")
                    if isinstance(content, list) and any(
                        not isinstance(part, dict)
                        or part.get("type") != "text"
                        or part.get("uuid") is not None
                        for part in content
                    ):
                        raise ValueError(
                            "KV-aware P/D currently supports text-only prompts"
                        )
            token_ids = await self.tokenizer.tokenize(
                parsed.model_dump(), request_path=path
            )
            if not token_ids:
                raise ValueError("KV-aware P/D requires a single tokenizable prompt")
        except TokenizeError as exc:
            raise HTTPException(exc.status_code, exc.message) from exc
        except (ValueError, TypeError) as exc:
            raise HTTPException(400, str(exc)) from exc

        request_id = _get_serve_request_context()._internal_request_id
        raw_info = RawRequestInfo.from_starlette_request(request)
        prefill = self.prefill
        multiplexed_model_id = self.get_multiplexed_model_id(parsed.model_dump())
        if multiplexed_model_id:
            prefill = prefill.options(multiplexed_model_id=multiplexed_model_id)
        session_id = session_id_from_headers(request.headers)
        if session_id:
            prefill = prefill.options(session_id=session_id)
        prefill_response = None
        try:
            prefill_response = prefill.prefill.remote(
                parsed,
                raw_info,
                request_token_ids=token_ids,
                routing_request_id=request_id,
                router_destination=self.router_destination,
            )
            params: Dict[str, Any] = await prefill_response
            await self.prefill_kv_token_tracker.release_request(request_id)
            metadata = base64.b64encode(
                json.dumps(params, separators=(",", ":")).encode()
            ).decode("ascii")
            response = await self.route_decode(
                SimpleNamespace(**parsed.model_dump()), token_ids, request_id
            )
            response[RAY_SERVE_INGRESS_REQUEST_ROUTER_OPT_HEADERS_FIELD][
                KV_TRANSFER_PARAMS_HEADER
            ] = metadata
            return response
        except BaseException:
            if prefill_response is not None:
                prefill_response.cancel()
            # The trackers own cleanup tasks so delayed Rust frees cannot hold
            # the HTTP decision past its deadline.
            for tracker in (
                self.prefill_kv_token_tracker,
                self.decode_kv_token_tracker,
            ):
                tracker._schedule(tracker.release_request(request_id))
            raise
