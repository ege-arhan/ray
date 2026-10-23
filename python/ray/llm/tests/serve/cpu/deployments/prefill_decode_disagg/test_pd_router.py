"""Validate the public two-application graph and unsupported configurations."""

import sys

import pytest


def kv_config(**engine_overrides):
    from ray.llm._internal.serve.core.configs.llm_config import LLMConfig

    return LLMConfig(
        model_loading_config={"model_id": "model"},
        deployment_config={
            "num_replicas": 1,
            "request_router_config": {
                "request_router_class": "ray.serve.llm.request_router.KVAwareRouter"
            },
        },
        engine_kwargs={
            "kv_transfer_config": {
                "kv_connector": "NixlConnector",
                "kv_role": "kv_both",
            },
            **engine_overrides,
        },
    )


def test_two_application_graphs(monkeypatch):
    from ray.llm._internal.serve.serving_patterns.prefill_decode import builder
    from ray.serve._private import constants
    from ray.serve.deployment import Application

    monkeypatch.setattr(builder, "RAY_SERVE_LLM_ENABLE_DIRECT_STREAMING", True)
    monkeypatch.setattr(constants, "RAY_SERVE_ENABLE_HA_PROXY", True)
    # The obsolete FORWARD_BODY flag is deliberately false.
    monkeypatch.setattr(
        constants, "RAY_SERVE_INGRESS_REQUEST_ROUTER_FORWARD_BODY", False
    )
    config = kv_config()
    args = {"prefill_config": config, "decode_config": config}
    backend = builder.build_pd_openai_app(args)
    assert backend._bound_deployment.name == "Decode:model"
    prefill = backend._bound_deployment.init_kwargs["prefill_server"]
    assert prefill._bound_deployment.name == "Prefill:model"
    assert backend._ingress_request_router is None
    router = builder.build_pd_router_app(
        {
            **args,
            "backend_application": "pd-backend",
            "prefill_deployment": "Prefill:model",
            "decode_deployment": "Decode:model",
        }
    )
    bound = router._bound_deployment
    assert bound.name == "LLMRouter"
    assert all(
        not isinstance(value, Application) for value in bound.init_kwargs.values()
    )
    assert (
        bound.init_kwargs["prefill_config"].experimental_configs["pd_routing_stage"]
        == "prefill"
    )
    assert (
        bound.init_kwargs["decode_config"].experimental_configs["pd_routing_stage"]
        == "decode"
    )
    assert "pd_routing_stage" not in config.experimental_configs


@pytest.mark.parametrize("kind", ["dp", "moriio", "nested_moriio", "cold", "partial"])
def test_builder_restrictions(monkeypatch, kind):
    from ray.llm._internal.serve.serving_patterns.prefill_decode import builder
    from ray.serve._private import constants

    monkeypatch.setattr(builder, "RAY_SERVE_LLM_ENABLE_DIRECT_STREAMING", True)
    monkeypatch.setattr(constants, "RAY_SERVE_ENABLE_HA_PROXY", True)
    p, d = kv_config(), kv_config()
    if kind == "dp":
        p.engine_kwargs["data_parallel_size"] = 2
    elif kind in ("moriio", "nested_moriio"):
        connector = {"kv_connector": "MoRIIOConnector", "kv_role": "kv_both"}
        if kind == "nested_moriio":
            connector = {
                "kv_connector": "MultiConnector",
                "kv_role": "kv_both",
                "kv_connector_extra_config": {"connectors": [connector]},
            }
        p.engine_kwargs["kv_transfer_config"] = connector
    elif kind == "cold":
        d.deployment_config["num_replicas"] = 0
    else:
        d.deployment_config.pop("request_router_config")
    with pytest.raises(ValueError):
        builder.build_pd_openai_app({"prefill_config": p, "decode_config": d})


@pytest.mark.parametrize("autoscaling", [None, {}, {"min_replicas": 1}, "typed"])
def test_default_pools_stay_warm(monkeypatch, autoscaling):
    from ray.llm._internal.serve.serving_patterns.prefill_decode import builder
    from ray.serve._private import constants
    from ray.serve.config import AutoscalingConfig

    monkeypatch.setattr(builder, "RAY_SERVE_LLM_ENABLE_DIRECT_STREAMING", True)
    monkeypatch.setattr(constants, "RAY_SERVE_ENABLE_HA_PROXY", True)
    config = kv_config()
    config.deployment_config.pop("num_replicas")
    config.deployment_config["autoscaling_config"] = (
        AutoscalingConfig(min_replicas=1) if autoscaling == "typed" else autoscaling
    )
    args = builder.PDServingArgs(prefill_config=config, decode_config=config)
    assert builder._configure_pd_kv_aware(args)
    for stage in (args.prefill_config, args.decode_config):
        assert stage.deployment_config["autoscaling_config"]["min_replicas"] == 1


if __name__ == "__main__":
    sys.exit(pytest.main(["-v", __file__]))
