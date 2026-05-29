"""Protocol message validation tests.

These exercise the JSON-schema validators in ``stt_streaming.proto``.
They are pure-Python: no torch / nemo imports so they run in GPU-less CI.
"""
from __future__ import annotations

import pytest

pydantic = pytest.importorskip("pydantic")

# Import lazily so collection still works if the package is not yet installed.
proto = pytest.importorskip("stt_streaming.proto")


def test_valid_start_message_defaults():
    msg = proto.parse_client_message(
        {"type": "start", "sample_rate": 16000, "encoding": "pcm_s16le"}
    )
    assert msg.type == "start"
    assert msg.sample_rate == 16000
    assert msg.encoding == "pcm_s16le"
    assert msg.language in ("auto", None) or isinstance(msg.language, str)
    assert msg.enable_partials is True
    assert msg.vad_events is False


def test_invalid_sample_rate_rejected():
    with pytest.raises(pydantic.ValidationError):
        proto.parse_client_message(
            {"type": "start", "sample_rate": 8000, "encoding": "pcm_s16le"}
        )


def test_invalid_encoding_rejected():
    with pytest.raises(pydantic.ValidationError):
        proto.parse_client_message(
            {"type": "start", "sample_rate": 16000, "encoding": "opus"}
        )


def test_commit_message_parses():
    msg = proto.parse_client_message({"type": "commit"})
    assert msg.type == "commit"


def test_close_message_parses():
    msg = proto.parse_client_message({"type": "close"})
    assert msg.type == "close"


def test_ping_with_ts():
    msg = proto.parse_client_message({"type": "ping", "ts": 1748528521234})
    assert msg.type == "ping"
    assert msg.ts == 1748528521234
