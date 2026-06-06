"""End-to-end protocol test for the WS /v1/stream endpoint.

Drives the *real* FastAPI app and ws session handler with a stubbed model + VAD
(so it runs on CPU-only CI with no torch/nemo). Verifies the full streaming
handshake over real PCM16 frames: ready -> partial(s) -> final -> clean 1000
close, plus that a short trailing frame does not abort the session.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402
from starlette.websockets import WebSocketDisconnect  # noqa: E402

model_mod = pytest.importorskip("stt_streaming.model")
import stt_streaming.server as server  # noqa: E402

API_KEY = "test_key_local_only"
SR = 16000


class _StubModel:
    """Deterministic stand-in: transcript length tracks audio length, so a
    correctly-buffered (non-duplicated) stream yields stable, growing text."""

    def __init__(self, *a, **k):
        self.model_dir = "stub"

    def reset_state(self):
        return model_mod.InferState()

    def transcribe_window(self, audio):
        if audio is None or audio.shape[0] == 0:
            return ""
        words = int(audio.shape[0] / SR / 0.2)  # ~1 token per 200 ms of audio
        return " ".join(["hello"] * words)

    def try_transcribe_window(self, audio):
        # The stub is never "busy"; mirror the real best-effort partial path.
        return self.transcribe_window(audio)

    def transcribe_chunk(self, audio, state):
        text = self.transcribe_window(audio)
        state.last_text = text
        return text, state

    def finalize(self, state):
        return state.last_text

    def warm_up(self):
        pass


class _StubVad:
    def reset_state(self):
        return None

    def is_speech(self, chunk, state=None, sample_rate=SR):
        return True, state


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setenv("ASR_API_KEY", API_KEY)
    monkeypatch.setenv("ASR_PARTIAL_INTERVAL_MS", "0")  # emit on every speech frame
    monkeypatch.setattr(server, "ParakeetModel", _StubModel)
    monkeypatch.setattr(server, "load_vad", lambda: _StubVad())
    with TestClient(server.app) as c:
        yield c


def _speech_pcm(seconds: float) -> bytes:
    n = int(SR * seconds)
    t = np.arange(n) / SR
    wave = (0.2 * np.sin(2 * np.pi * 220 * t) * 32767).astype("<i2")
    return wave.tobytes()


def _drain(ws):
    """Receive until the server closes; return (messages, close_code)."""
    msgs = []
    try:
        while True:
            msgs.append(json.loads(ws.receive_text()))
    except WebSocketDisconnect as e:
        return msgs, e.code


def test_full_stream_handshake(client):
    pcm = _speech_pcm(1.5)
    chunk = 320 * 2  # 20 ms frames
    with client.websocket_connect("/v1/stream", headers={"x-api-key": API_KEY}) as ws:
        ws.send_text(json.dumps({
            "type": "start", "sample_rate": SR,
            "encoding": "pcm_s16le", "enable_partials": True,
        }))
        ready = json.loads(ws.receive_text())
        assert ready["type"] == "ready"
        assert ready["sample_rate"] == SR

        for i in range(0, len(pcm), chunk):
            ws.send_bytes(pcm[i:i + chunk])
        ws.send_text(json.dumps({"type": "close"}))

        msgs, code = _drain(ws)

    types = [m["type"] for m in msgs]
    assert "partial" in types, f"no partial emitted: {types}"
    finals = [m for m in msgs if m["type"] == "final"]
    assert finals, f"no final emitted: {types}"
    assert finals[-1]["text"], "final transcript was empty"
    assert code == 1000, f"expected clean 1000 close, got {code}"


def test_short_trailing_frame_does_not_abort(client):
    """A real-time client's final remainder chunk can be < 80 samples; the
    session must keep going and still finalize cleanly."""
    pcm = _speech_pcm(0.5) + b"\x01\x00" * 10  # 10-sample (20-byte) tail
    chunk = 320 * 2
    with client.websocket_connect("/v1/stream", headers={"x-api-key": API_KEY}) as ws:
        ws.send_text(json.dumps({"type": "start", "sample_rate": SR, "encoding": "pcm_s16le"}))
        assert json.loads(ws.receive_text())["type"] == "ready"
        for i in range(0, len(pcm), chunk):
            ws.send_bytes(pcm[i:i + chunk])
        ws.send_text(json.dumps({"type": "close"}))
        msgs, code = _drain(ws)

    assert code == 1000, f"short tail aborted the session: code={code}, msgs={msgs}"
    assert any(m["type"] == "final" for m in msgs)


def test_bad_api_key_rejected(client):
    # Auth is checked before accept(), so the upgrade itself closes 4401.
    with pytest.raises(WebSocketDisconnect) as exc:
        with client.websocket_connect("/v1/stream", headers={"x-api-key": "wrong"}):
            pass
    assert exc.value.code == 4401


def test_odd_frame_rejected(client):
    with client.websocket_connect("/v1/stream", headers={"x-api-key": API_KEY}) as ws:
        ws.send_text(json.dumps({"type": "start", "sample_rate": SR, "encoding": "pcm_s16le"}))
        assert json.loads(ws.receive_text())["type"] == "ready"
        ws.send_bytes(b"\x00\x00\x00")  # 3 bytes — unaligned PCM16
        msgs, code = _drain(ws)
    assert code == 4400
    assert any(m.get("type") == "error" for m in msgs)
