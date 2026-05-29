from __future__ import annotations

from prometheus_client import Counter, Gauge, generate_latest

# Names match spec §4.2 exactly — the dispatcher's metrics poller is name-sensitive.
_requests_total = Counter(
    "asr_requests_total",
    "Total streaming sessions accepted (success + error)",
    ["status"],
)
_duration_ms_sum = Counter(
    "asr_duration_ms_sum",
    "Sum of end-to-end session durations in milliseconds",
)
_duration_ms_count = Counter(
    "asr_duration_ms_count",
    "Number of completed sessions (matches asr_requests_total{status=\"ok\"})",
)
_audio_ms_total = Counter(
    "asr_audio_ms_total",
    "Total milliseconds of audio transcribed",
)
_bytes_received_total = Counter(
    "asr_bytes_received_total",
    "Total bytes received over WS audio frames",
)
_inflight = Gauge(
    "asr_inflight",
    "Current open streaming sessions",
)
_ttft_ms_sum = Counter(
    "asr_ttft_ms_sum",
    "Sum of time-to-first-partial across sessions (ms)",
)
_ttft_ms_count = Counter(
    "asr_ttft_ms_count",
    "Number of sessions that emitted at least one partial",
)

# Pre-create label instances so they appear in scrape output before any traffic.
_requests_total.labels(status="ok")
_requests_total.labels(status="error")
_requests_total.labels(status="timeout")


def record_request_ok(
    duration_ms: float,
    audio_ms: float,
    bytes_received: int,
    ttft_ms: float | None,
) -> None:
    _requests_total.labels(status="ok").inc()
    _duration_ms_sum.inc(duration_ms)
    _duration_ms_count.inc()
    _audio_ms_total.inc(audio_ms)
    _bytes_received_total.inc(bytes_received)
    if ttft_ms is not None:
        _ttft_ms_sum.inc(ttft_ms)
        _ttft_ms_count.inc()


def record_request_error() -> None:
    _requests_total.labels(status="error").inc()


def record_request_timeout() -> None:
    _requests_total.labels(status="timeout").inc()


def record_request_finished(status: str = "ok", duration_ms: float = 0.0, audio_ms: float = 0.0,
                            bytes_received: int = 0, ttft_ms: float = 0.0) -> None:
    if status == "ok":
        record_request_ok(duration_ms=duration_ms, audio_ms=audio_ms,
                          bytes_received=bytes_received, ttft_ms=ttft_ms)
    elif status == "timeout":
        record_request_timeout()
    else:
        record_request_error()


def inflight_inc() -> None:
    _inflight.inc()


def inflight_dec() -> None:
    _inflight.dec()


def export_metrics() -> bytes:
    return generate_latest()


def get_inflight() -> int:
    return int(_inflight._value.get())


# Aliases used by ws.py (workflow agents diverged on naming — keep both APIs working)
def inc_inflight() -> None: _inflight.inc()
def dec_inflight() -> None: _inflight.dec()
def inc_request(status: str = "ok") -> None: _requests_total.labels(status=status).inc()
def add_audio_ms(ms: float) -> None: _audio_ms_total.inc(max(0.0, ms))
def add_bytes_received(n: int) -> None: _bytes_received_total.inc(max(0, n))
def observe_duration_ms(ms: float) -> None:
    _duration_ms_sum.inc(max(0.0, ms))
    _duration_ms_count.inc()
def observe_ttft_ms(ms: float) -> None:
    _ttft_ms_sum.inc(max(0.0, ms))
    _ttft_ms_count.inc()
