"""The registry exposes a histogram for adapter-swap latency with a
`method` label that matches what the runtime emits (unpatch|reload)."""
from __future__ import annotations

from fastapi.testclient import TestClient


def test_registry_has_adapter_swap_metric() -> None:
    import src.serving.full_pipeline_server as fps

    _reg, metrics = fps._build_registry()
    assert "adapter_swap" in metrics, "registry missing adapter_swap entry"


def test_metrics_endpoint_exposes_swap_histogram(monkeypatch) -> None:
    """The /metrics endpoint must include kiki_adapter_swap_seconds in its
    scrape output (Prometheus HELP/TYPE lines are emitted even before the
    first observation)."""
    import src.serving.full_pipeline_server as fps

    class _Fake:
        def __init__(self, *_a, **_k) -> None:
            pass

        def apply(self, _a) -> None:
            pass

        def generate(self, *_a, **_k) -> str:
            return "ok"

        def route(self, _q):
            return [("coding", 0.9)]

        def recall(self, *_a, **_k):
            return []

        def write(self, *_a, **_k):
            pass

        async def arbitrate(self, cs):
            return cs[0], {}

        async def check(self, t, ctx=None):
            return t, {}

    monkeypatch.setattr(fps, "_build_runtime", lambda cfg, **_k: _Fake())
    monkeypatch.setattr(fps, "_build_meta_router", lambda cfg: _Fake())
    monkeypatch.setattr(fps, "_build_aeon", lambda cfg: _Fake())
    monkeypatch.setattr(fps, "_build_negotiator", lambda cfg: _Fake())
    monkeypatch.setattr(fps, "_build_antibias", lambda cfg: _Fake())

    client = TestClient(fps.make_app(fps.FullPipelineConfig.defaults()))
    r = client.get("/metrics")
    assert r.status_code == 200
    body = r.text
    assert "kiki_adapter_swap_seconds" in body


def test_initial_apply_observes_method_initial(monkeypatch) -> None:
    """First adapter load (when _current_adapter is None) must be
    observed with method="initial" so dashboards distinguish cold
    boot latency from steady-state swaps."""
    import sys
    import types

    from prometheus_client import CollectorRegistry, Histogram

    from src.serving.full_pipeline_server import _MLXRuntimeAdapter

    # Stub mlx_lm.tuner.utils.load_adapters so apply() does not blow
    # up on the fake model we inject below.
    fake_tuner_utils = types.ModuleType("mlx_lm.tuner.utils")
    fake_tuner_utils.load_adapters = lambda model, _path: model
    monkeypatch.setitem(sys.modules, "mlx_lm.tuner.utils", fake_tuner_utils)

    reg = CollectorRegistry()
    h = Histogram(
        "kiki_adapter_swap_seconds",
        "test",
        ["method"],
        registry=reg,
    )

    # Construct without __init__ so we skip real mlx_load + eager_materialize.
    adapter = object.__new__(_MLXRuntimeAdapter)
    adapter._base_model_path = "/dev/null"
    adapter._adapters_root = "/tmp"
    adapter._swap_metric = h
    adapter._current_adapter = None
    adapter._model = object()
    adapter._tokenizer = object()

    adapter.apply(["python"])

    samples = {
        (s.name, s.labels.get("method", "")): s.value
        for metric in reg.collect()
        for s in metric.samples
    }
    assert samples.get(("kiki_adapter_swap_seconds_count", "initial"), 0.0) == 1.0
    assert samples.get(("kiki_adapter_swap_seconds_count", "unpatch"), 0.0) == 0.0
    assert samples.get(("kiki_adapter_swap_seconds_count", "reload"), 0.0) == 0.0
    assert adapter._current_adapter == "python"
