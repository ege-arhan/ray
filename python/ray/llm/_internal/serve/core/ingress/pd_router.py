"""Application-level router for a single KV-aware prefill/decode model."""

import asyncio
import hashlib
import os
import secrets
import sys
from types import SimpleNamespace
from typing import List
from urllib.parse import urlsplit

from fastapi import HTTPException, Request
from starlette.responses import JSONResponse

from ray import serve
from ray.llm._internal.serve.core.ingress.router import LLMRouter
from ray.llm._internal.serve.routing_policies.kv_aware.constants import (
    KV_TOKEN_KEY_HEADER,
)
from ray.llm._internal.serve.routing_policies.kv_aware.kv_token_tracker import (
    build_kv_token_tracker,
)
from ray.llm._internal.serve.routing_policies.kv_aware.pd_router import (
    PDRequestCoordinator,
)
from ray.serve._private.constants import (
    RAY_SERVE_HAPROXY_INGRESS_REQUEST_ROUTER_TIMEOUT_S,
    SERVE_INGRESS_ROUTER_REQUEST_PATH_HEADER,
    SERVE_ROUTER_APPLICATION_DIRECT_RESPONSE_HEADER,
    SERVE_ROUTER_APPLICATION_ORIGINAL_METHOD_HEADER,
)

if sys.version_info >= (3, 11):
    from asyncio import timeout
else:
    from async_timeout import timeout


@serve._router_application
class LLMPDRouter(LLMRouter):
    """Own two local trackers; run P through Ray and route HTTP directly to D."""

    async def __init__(
        self,
        backend_application: str,
        prefill_deployment: str,
        decode_deployment: str,
        prefill_config,
        decode_config,
    ):
        self._routing_timeout_s = (
            0.8 * RAY_SERVE_HAPROXY_INGRESS_REQUEST_ROUTER_TIMEOUT_S
        )
        api_keys = decode_config.get_engine_config().frontend_kwargs.get("api_key")
        if not api_keys:
            actor_options = (
                decode_config.deployment_config.get("ray_actor_options") or {}
            )
            runtime_env = {
                **(actor_options.get("runtime_env") or {}),
                **(decode_config.runtime_env or {}),
            }
            env_vars = runtime_env.get("env_vars") or {}
            key = env_vars.get("VLLM_API_KEY", os.getenv("VLLM_API_KEY"))
            api_keys = [key] if key else []
        if isinstance(api_keys, str):
            api_keys = [api_keys]
        self._api_key_hashes = [
            hashlib.sha256(key.encode()).digest() for key in api_keys if key
        ]
        self.backend_application = backend_application
        self.prefill_config = prefill_config
        # No backend Application objects are bound into this application.
        self.prefill_server = serve.get_deployment_handle(
            prefill_deployment, app_name=backend_application
        )
        self.decode_server = serve.get_deployment_handle(
            decode_deployment, app_name=backend_application
        )
        await super().__init__(
            self.decode_server,
            decode_config,
            base_model_id=decode_config.model_id if decode_config.lora_config else None,
        )
        # Both trackers are registered before either handle's router initializes.
        self.prefill_server._init(_run_router_in_separate_loop=False)
        self.coordinator = PDRequestCoordinator(
            self.prefill_server,
            self.prefill_kv_token_tracker,
            self.decode_kv_token_tracker,
            self._tokenizer,
            self.route_decode,
            serve.get_replica_context().replica_id.deployment_id,
            self._get_multiplexed_model_id,
        )

    def create_token_trackers(self, server, llm_config):
        self.prefill_kv_token_tracker = build_kv_token_tracker(
            self.prefill_config, self.prefill_server.deployment_id
        )
        self.decode_kv_token_tracker = build_kv_token_tracker(
            llm_config, server.deployment_id
        )
        return {
            self.prefill_server.deployment_id: self.prefill_kv_token_tracker,
            server.deployment_id: self.decode_kv_token_tracker,
        }

    async def route_request(self, request: Request):
        # Tokenization, P execution, and D selection share one decision deadline.
        try:
            async with timeout(self._routing_timeout_s):
                return await self._route_control(request)
        except asyncio.TimeoutError:
            status_code, detail = 503, "P/D routing deadline exceeded"
        except HTTPException as exc:
            status_code, detail = exc.status_code, exc.detail
        return JSONResponse(
            {"detail": detail},
            status_code=status_code,
            headers={SERVE_ROUTER_APPLICATION_DIRECT_RESPONSE_HEADER: "1"},
        )

    async def _route_control(self, request: Request):
        method = request.headers.get(SERVE_ROUTER_APPLICATION_ORIGINAL_METHOD_HEADER)
        path = urlsplit(
            request.headers.get(SERVE_INGRESS_ROUTER_REQUEST_PATH_HEADER, "")
        ).path
        if not method or not path:
            raise HTTPException(400, "Missing original request method or path")
        if method == "POST" and path in (
            "/v1/messages",
            "/v1/responses",
            "/v1/chat/completions/batch",
            "/invocations",
        ):
            raise HTTPException(
                400,
                "KV-aware P/D supports chat/completions and completions generation only",
            )
        if method == "POST" and path in ("/v1/chat/completions", "/v1/completions"):
            if self._api_key_hashes:
                scheme, _, key = request.headers.get("authorization", "").partition(" ")
                digest = hashlib.sha256(key.encode()).digest()
                if scheme.lower() != "bearer" or not any(
                    secrets.compare_digest(digest, expected)
                    for expected in self._api_key_hashes
                ):
                    raise HTTPException(401, "Unauthorized")
            return await self.coordinator.route(request, await request.body())
        _, _, replica_id, _ = await self._pick_replica(self.decode_server)
        return {"application": self.backend_application, "replica_id": replica_id}

    async def route_decode(
        self, routing_payload: SimpleNamespace, token_ids: List[int], request_id: str
    ):
        multiplexed_model_id = self._get_multiplexed_model_id(vars(routing_payload))
        handle = self._get_handle(multiplexed_model_id)
        _, _, replica_id, token_endpoint = await self._pick_replica(
            handle, routing_payload, token_ids, routing_request_id=request_id
        )
        self._push_prompt_tokens(
            token_endpoint=token_endpoint,
            replica_id=replica_id,
            request_token_ids=token_ids,
            token_key=request_id,
        )
        headers = {KV_TOKEN_KEY_HEADER: request_id}
        if multiplexed_model_id:
            from ray.serve._private.constants import SERVE_MULTIPLEXED_MODEL_ID

            headers[SERVE_MULTIPLEXED_MODEL_ID] = multiplexed_model_id
        return {
            "application": self.backend_application,
            "replica_id": replica_id,
            "request_headers": headers,
        }
