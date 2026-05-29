"""Prometheus metric registration + monotonicity tests."""
from __future__ import annotations

import pytest

prom = pytest.importorskip("prometheus_client")
metrics = pytest.importorskip("stt_streaming.metrics")

# Names required by the operator's metrics poller (spec §4.2).
REQUIRED_COUNTERS = {
    "asr_requests_total",
    "asr_duration_ms_sum",
    "asr_duration_ms_count",
    "asr_audio_ms_total",
    "asr_bytes_received_total",
    "asr_ttft_ms_sum",
    "asr_ttft_ms_count",
}
REQUIRED_GAUGES = {"asr_inflight"}


def _sample_value(name: str, labels: dict | None = None) -> float:
    """Look up a single sample value from the default registry."""
    labels = labels or {}
    for family in prom.REGISTRY.collect():
        for sample in family.samples:
            if sample.name == name and all(
                sample.labels.get(k) == v for k, v in labels.items()
            ):
                return sample.value
    return 0.0


def test_counter_names_match_spec():
    names = set()
    for family in prom.REGISTRY.collect():
        names.add(family.name)
        for sample in family.samples:
            names.add(sample.name)
    # Counter family names in prometheus_client lose the "_total" suffix;
    # accept either form.
    for required in REQUIRED_COUNTERS:
        assert required in names or required.removesuffix("_total") in names, (
            f"missing required counter: {required}"
        )
    for required in REQUIRED_GAUGES:
        assert required in names, f"missing required gauge: {required}"


def test_counters_monotonic_increase():
    before = _sample_value("asr_requests_total", {"status": "ok"})
    metrics.record_request_finished(status="ok", duration_ms=123, audio_ms=1000)
    after = _sample_value("asr_requests_total", {"status": "ok"})
    assert after == before + 1
    assert after >= before  # monotonic


def test_inflight_gauge_inc_dec():
    base = _sample_value("asr_inflight")
    metrics.inflight_inc()
    assert _sample_value("asr_inflight") == base + 1
    metrics.inflight_dec()
    assert _sample_value("asr_inflight") == base
