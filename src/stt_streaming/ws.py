"""WebSocket session handler for /v1/stream."""
from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
from fastapi import WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

from . import metrics
from .model import SAMPLE_RATE, InferState, ParakeetModel

logger = logging.getLogger(__name__)

# Frame size bounds in samples (spec §5.2)
MIN_FRAME_SAMPLES = 80
MAX_FRAME_SAMPLES = 1600

# Close codes per spec §5.6
CLOSE_NORMAL = 1000
CLOSE_INTERNAL = 1011
CLOSE_BAD_REQUEST = 4400
CLOSE_UNAUTHORIZED = 4401
CLOSE_FRAME_TOO_LARGE = 4413
CLOSE_TOO_MANY = 4429

_START_TIMEOUT_S = 5.0


@dataclass
class WSSession:
    session_id: str
    language: str = "auto"
    sample_rate: int = SAMPLE_RATE
    enable_partials: bool = True
    vad_events: bool = False
    metadata: dict = field(default_factory=dict)

    model_state: Optional[InferState] = None
    vad_state: Any = None

    last_partial_ts: float = 0.0
    last_partial_text: str = ""
    audio_ms_consumed: int = 0
    bytes_received: int = 0
    utterance_buffer: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    utterance_start_ms: int = 0
    silence_run_ms: int = 0

    session_start_monotonic: float = field(default_factory=time.monotonic)
    first_partial_emitted: bool = False
    ttfp_recorded: bool = False


def _now_ms_since(start: float) -> int:
    return int((time.monotonic() - start) * 1000)


def _pcm16_to_float32(buf: bytes) -> np.ndarray:
    arr = np.frombuffer(buf, dtype="<i2")
    return (arr.astype(np.float32) / 32768.0).copy()


async def _send_json(ws: WebSocket, payload: dict) -> None:
    if ws.client_state == WebSocketState.CONNECTED:
        await ws.send_text(json.dumps(payload, separators=(",", ":")))


async def _send_error(ws: WebSocket, code: str, message: str) -> None:
    await _send_json(ws, {"type": "error", "code": code, "message": message})


async def _close(ws: WebSocket, code: int, reason: str = "") -> None:
    try:
        if ws.client_state != WebSocketState.DISCONNECTED:
            await ws.close(code=code, reason=reason)
    except Exception:
        pass


def _validate_start(msg: dict) -> Optional[str]:
    if msg.get("type") != "start":
        return "first message must be 'start'"
    sr = msg.get("sample_rate", SAMPLE_RATE)
    if sr != SAMPLE_RATE:
        return f"sample_rate must be {SAMPLE_RATE}"
    enc = msg.get("encoding", "pcm_s16le")
    if enc != "pcm_s16le":
        return "encoding must be pcm_s16le"
    return None


async def handle_session(
    websocket: WebSocket,
    model: ParakeetModel,
    vad: Any,
    config: Any,
) -> None:
    """Run a full WS session. Auth must have been checked before .accept()."""
    # Concurrency cap
    if metrics.get_inflight() >= getattr(config, "max_concurrent", 32):
        await websocket.close(code=CLOSE_TOO_MANY, reason="too many concurrent sessions")
        metrics.inc_request("error")
        return

    api_key = getattr(config, "api_key", None)
    header_key = websocket.headers.get("x-api-key")
    if api_key and header_key != api_key:
        await websocket.close(code=CLOSE_UNAUTHORIZED, reason="unauthorized")
        metrics.inc_request("error")
        return

    await websocket.accept()
    metrics.inc_inflight()
    status = "ok"
    sess: Optional[WSSession] = None

    try:
        # Receive start with timeout
        try:
            raw = await asyncio.wait_for(websocket.receive_text(), timeout=_START_TIMEOUT_S)
        except asyncio.TimeoutError:
            await _send_error(websocket, "bad_request", "start message timeout")
            await _close(websocket, CLOSE_BAD_REQUEST, "start timeout")
            status = "timeout"
            return
        except WebSocketDisconnect:
            status = "error"
            return

        try:
            start = json.loads(raw)
        except json.JSONDecodeError:
            await _send_error(websocket, "bad_request", "start is not JSON")
            await _close(websocket, CLOSE_BAD_REQUEST, "bad start")
            status = "error"
            return

        err = _validate_start(start)
        if err:
            await _send_error(websocket, "bad_request", err)
            await _close(websocket, CLOSE_BAD_REQUEST, err)
            status = "error"
            return

        sess = WSSession(
            session_id=start.get("session_id") or str(uuid.uuid4()),
            language=start.get("language", "auto"),
            sample_rate=start.get("sample_rate", SAMPLE_RATE),
            enable_partials=start.get("enable_partials", True),
            vad_events=start.get("vad_events", False),
            metadata=start.get("metadata") or {},
            model_state=model.reset_state(),
            vad_state=vad.reset_state() if vad is not None else None,
        )

        await _send_json(websocket, {
            "type": "ready",
            "session_id": sess.session_id,
            "model": getattr(config, "model_name", model.model_dir),
            "language": sess.language,
            "sample_rate": sess.sample_rate,
        })

        await _session_loop(websocket, model, vad, config, sess)

    except WebSocketDisconnect:
        status = "ok" if sess is not None else "error"
    except Exception:
        logger.exception("session crashed")
        await _send_error(websocket, "internal", "internal error")
        await _close(websocket, CLOSE_INTERNAL, "internal")
        status = "error"
    finally:
        metrics.dec_inflight()
        metrics.inc_request(status)
        if sess is not None:
            dur_ms = _now_ms_since(sess.session_start_monotonic)
            metrics.observe_duration_ms(dur_ms)
            metrics.add_audio_ms(sess.audio_ms_consumed)
            metrics.add_bytes_received(sess.bytes_received)


async def _session_loop(
    ws: WebSocket,
    model: ParakeetModel,
    vad: Any,
    config: Any,
    sess: WSSession,
) -> None:
    partial_interval_s = getattr(config, "partial_interval_ms", 200) / 1000.0
    silence_commit_ms = getattr(config, "internal_silence_ms", 800)
    max_session_s = getattr(config, "max_session_seconds", 1800)

    while True:
        if time.monotonic() - sess.session_start_monotonic > max_session_s:
            await _finalize_utterance(ws, model, sess, force=True)
            await _close(ws, CLOSE_NORMAL, "session max duration")
            return

        try:
            msg = await ws.receive()
        except WebSocketDisconnect:
            await _finalize_utterance(ws, model, sess, force=True)
            return

        msg_type = msg.get("type")
        if msg_type == "websocket.disconnect":
            await _finalize_utterance(ws, model, sess, force=True)
            return

        # Binary audio frame
        if "bytes" in msg and msg["bytes"] is not None:
            frame = msg["bytes"]
            sample_count = len(frame) // 2
            if len(frame) % 2 != 0 or sample_count < MIN_FRAME_SAMPLES:
                await _send_error(ws, "audio_format_error", "frame too small or unaligned")
                await _close(ws, CLOSE_BAD_REQUEST, "bad frame")
                return
            if sample_count > MAX_FRAME_SAMPLES:
                await _send_error(ws, "audio_format_error", "frame too large")
                await _close(ws, CLOSE_FRAME_TOO_LARGE, "frame too large")
                return

            sess.bytes_received += len(frame)
            chunk = _pcm16_to_float32(frame)
            sess.utterance_buffer = np.concatenate([sess.utterance_buffer, chunk])
            chunk_ms = int(sample_count * 1000 / SAMPLE_RATE)
            sess.audio_ms_consumed += chunk_ms

            # VAD on this chunk
            is_speech = True
            if vad is not None:
                try:
                    is_speech, sess.vad_state = vad.is_speech(chunk, sess.vad_state)
                except Exception:
                    is_speech = True

            if is_speech:
                if sess.silence_run_ms > 0 and sess.vad_events:
                    await _send_json(ws, {
                        "type": "vad_speech",
                        "audio_ms_consumed": sess.audio_ms_consumed,
                    })
                sess.silence_run_ms = 0
            else:
                sess.silence_run_ms += chunk_ms
                if sess.vad_events:
                    await _send_json(ws, {
                        "type": "vad_silence",
                        "audio_ms_consumed": sess.audio_ms_consumed,
                        "silence_ms": sess.silence_run_ms,
                    })

            # Partial throttling
            now = time.monotonic()
            if (
                sess.enable_partials
                and is_speech
                and (now - sess.last_partial_ts) >= partial_interval_s
                and sess.utterance_buffer.shape[0] > 0
            ):
                await _emit_partial(ws, model, sess)
                sess.last_partial_ts = now

            # Silence commit
            if sess.silence_run_ms >= silence_commit_ms and sess.utterance_buffer.shape[0] > 0:
                await _finalize_utterance(ws, model, sess)

            continue

        # Text control frame
        if "text" in msg and msg["text"] is not None:
            try:
                payload = json.loads(msg["text"])
            except json.JSONDecodeError:
                await _send_error(ws, "bad_request", "not JSON")
                continue

            ptype = payload.get("type")
            if ptype == "commit":
                await _finalize_utterance(ws, model, sess, force=True)
            elif ptype == "close":
                await _finalize_utterance(ws, model, sess, force=True)
                await _close(ws, CLOSE_NORMAL, "client close")
                return
            elif ptype == "ping":
                await _send_json(ws, {"type": "pong", "ts": payload.get("ts")})
            else:
                await _send_error(ws, "bad_request", f"unknown message type: {ptype}")


async def _emit_partial(ws: WebSocket, model: ParakeetModel, sess: WSSession) -> None:
    loop = asyncio.get_running_loop()
    audio = sess.utterance_buffer
    state = sess.model_state
    try:
        text, new_state = await loop.run_in_executor(
            None, model.transcribe_chunk, audio, state
        )
    except Exception:
        logger.exception("transcribe_chunk failed")
        return
    sess.model_state = new_state
    if not text or text == sess.last_partial_text:
        return
    sess.last_partial_text = text
    if not sess.ttfp_recorded:
        ttfp_ms = _now_ms_since(sess.session_start_monotonic)
        metrics.observe_ttft_ms(ttfp_ms)
        sess.ttfp_recorded = True
    await _send_json(ws, {
        "type": "partial",
        "text": text,
        "since_session_start_ms": _now_ms_since(sess.session_start_monotonic),
        "audio_ms_consumed": sess.audio_ms_consumed,
    })


async def _finalize_utterance(
    ws: WebSocket, model: ParakeetModel, sess: WSSession, force: bool = False
) -> None:
    if sess.utterance_buffer.shape[0] == 0 and not sess.last_partial_text:
        return
    loop = asyncio.get_running_loop()
    try:
        text = await loop.run_in_executor(None, model.finalize, sess.model_state)
    except Exception:
        logger.exception("finalize failed")
        text = sess.last_partial_text

    if not text and not force:
        return

    end_ms = sess.audio_ms_consumed
    await _send_json(ws, {
        "type": "final",
        "text": text or "",
        "utterance_start_ms": sess.utterance_start_ms,
        "utterance_end_ms": end_ms,
        "audio_ms_consumed": end_ms,
        "confidence": None,
        "language_detected": sess.language,
    })

    # Reset utterance state
    sess.utterance_buffer = np.zeros(0, dtype=np.float32)
    sess.utterance_start_ms = end_ms
    sess.silence_run_ms = 0
    sess.last_partial_text = ""
    sess.model_state = model.reset_state()
