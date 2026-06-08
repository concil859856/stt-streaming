#!/usr/bin/env python3
"""One-port test harness + turn-detection ensembler for the mic page.

Serves mic_test.html and bridges the browser to THREE upstreams, fusing them
into turn-end events the UI can render:

  browser PCM ─┬─▶ stt-streaming  /v1/stream        ─▶ partial/final transcripts + Silero vad_events
               └─▶ turn-detection /v1/smart-turn     ─▶ p_end_of_turn from audio/prosody
  stt text    ───▶ turn-detection /v1/turn-detector  ─▶ p_end_of_turn from semantics

Fusion — "is the user's turn over?" is decided only at a *pause* (you don't end a
turn mid-word just because the text momentarily looks complete):

  once the turn has speech AND it's been silent for >= ENDPOINT_MIN_MS:
    * if Smart-Turn prosody p >= AUDIO_FIRE          -> turn_end (reason: audio)
    * elif Turn-Detector confidence >= TEXT_FIRE      -> turn_end (reason: text)
    * elif silence >= SILENCE_BACKSTOP_MS (Silero)    -> turn_end (reason: silence)

The browser only needs ONE forwarded port (open http://localhost:8080/) and no
keys — the proxy injects them, read from the running pod containers.

    python3 examples/mic_proxy.py --port 8080
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import time

import uvicorn
import websockets
from starlette.applications import Starlette
from starlette.responses import FileResponse
from starlette.routing import Route, WebSocketRoute

HERE = os.path.dirname(os.path.abspath(__file__))

STT_BACKEND = os.environ.get("STT_BACKEND", "ws://localhost:8117/v1/stream")
TD_BASE = os.environ.get("TD_BASE", "ws://localhost:8125")  # /v1/smart-turn, /v1/turn-detector

# Fusion knobs (env-overridable).
ENDPOINT_MIN_MS = int(os.environ.get("ENDPOINT_MIN_MS", "600"))     # min pause before deciding
AUDIO_FIRE = float(os.environ.get("AUDIO_FIRE", "0.85"))           # Smart-Turn prosody fire
TEXT_FIRE = float(os.environ.get("TEXT_FIRE", "1.0"))             # Turn-Detector confidence fire
SILENCE_BACKSTOP_MS = int(os.environ.get("SILENCE_BACKSTOP_MS", "4000"))


def _key_from_container(container: str, var: str, env_override: str) -> str:
    key = os.environ.get(env_override, "")
    if key:
        return key
    import subprocess
    try:
        out = subprocess.run(
            ["docker", "inspect", container,
             "--format", "{{range .Config.Env}}{{println .}}{{end}}"],
            capture_output=True, text=True, timeout=10,
        ).stdout
        for line in out.splitlines():
            if line.startswith(var + "="):
                return line.split("=", 1)[1]
    except Exception:
        pass
    return ""


STT_API_KEY = _key_from_container(
    os.environ.get("STT_CONTAINER", "vocence-asr_streaming_rt-6"), "ASR_API_KEY", "STT_API_KEY")
TD_API_KEY = _key_from_container(
    os.environ.get("TD_CONTAINER", "vocence-turn_detection-13"), "TD_API_KEY", "TD_API_KEY")


async def index(request):
    return FileResponse(os.path.join(HERE, "mic_test.html"))


def _hdr(key):
    return [("X-API-Key", key)] if key else []


async def _browser_send(browser, obj):
    try:  # browser is a Starlette WebSocket
        await browser.send_text(json.dumps(obj))
    except Exception:
        pass


async def _up_send(up, obj):
    try:  # up is a `websockets` client connection
        await up.send(json.dumps(obj))
    except Exception:
        pass


async def bridge(browser):
    try:
        await _bridge(browser)
    except Exception:
        import traceback
        traceback.print_exc()
        try:
            await browser.close(code=1011)
        except Exception:
            pass


async def _bridge(browser):
    await browser.accept()
    try:
        first = await browser.receive_text()
        json.loads(first)
    except Exception:
        await browser.close(code=1002)
        return

    try:
        up_stt = await websockets.connect(STT_BACKEND, additional_headers=_hdr(STT_API_KEY), max_size=2 ** 21)
        up_audio = await websockets.connect(TD_BASE + "/v1/smart-turn", additional_headers=_hdr(TD_API_KEY))
        up_text = await websockets.connect(TD_BASE + "/v1/turn-detector", additional_headers=_hdr(TD_API_KEY))
    except Exception as e:
        await _browser_send(browser, {"type": "error", "code": "upstream", "message": str(e)})
        await browser.close(code=1011)
        return

    await up_stt.send(json.dumps({
        "type": "start", "language": "auto", "sample_rate": 16000,
        "encoding": "pcm_s16le", "enable_partials": True, "vad_events": True}))
    await up_audio.send(json.dumps({
        "type": "start", "sample_rate": 16000, "encoding": "pcm_s16le",
        "window_ms": 4000, "emit_every_ms": 150}))
    await up_text.send(json.dumps({"type": "start", "history": [], "language": "en"}))

    stt_ready = json.loads(await up_stt.recv())
    await up_audio.recv()
    await up_text.recv()
    await _browser_send(browser, {
        "type": "ready", "model": stt_ready.get("model"),
        "silence_backstop_ms": SILENCE_BACKSTOP_MS, "endpoint_min_ms": ENDPOINT_MIN_MS})

    # ---- shared turn state ----
    state = {
        "turn_finals": [],     # committed STT finals in the current turn
        "partial": "",         # current interim text
        "turn_active": False,  # speech has occurred in the current turn
        "last_voice": time.monotonic(),
        "p_audio": 0.0,        # latest Smart-Turn prosody probability
        "p_text": 0.0,         # latest Turn-Detector probability
        "conf_text": 0.0,      # latest Turn-Detector cross-language confidence
        "firing": False,
    }
    closed = asyncio.Event()

    def turn_text():
        parts = [p for p in state["turn_finals"] if p]
        if state["partial"]:
            parts.append(state["partial"])
        return " ".join(parts).strip()

    async def feed_text_model():
        txt = turn_text()
        if txt:
            await _up_send(up_text, {"type": "token", "text": txt})

    async def fire_turn_end(reason):
        if state["firing"] or not state["turn_active"]:
            return
        final_text = turn_text()
        if not final_text:
            # Nothing was actually said this turn — reset quietly, don't emit.
            state["turn_active"] = False
            state["last_voice"] = time.monotonic()
            return
        state["firing"] = True
        await _browser_send(browser, {
            "type": "turn_end", "reason": reason, "text": final_text,
            "p_audio": round(state["p_audio"], 3), "p_text": round(state["p_text"], 3)})
        await _up_send(up_text, {"type": "commit", "content": final_text})
        await _up_send(up_audio, {"type": "reset"})
        state["turn_finals"] = []
        state["partial"] = ""
        state["turn_active"] = False
        state["p_audio"] = state["p_text"] = state["conf_text"] = 0.0
        state["last_voice"] = time.monotonic()
        state["firing"] = False

    def mark_voice():
        # Real-time speech activity resets the silence clock and opens a turn.
        state["turn_active"] = True
        state["last_voice"] = time.monotonic()

    async def from_browser():
        try:
            while True:
                msg = await browser.receive()
                if msg["type"] == "websocket.disconnect":
                    break
                if msg.get("bytes") is not None:
                    b = msg["bytes"]
                    await up_stt.send(b)
                    await up_audio.send(b)
                elif msg.get("text") is not None:
                    obj = json.loads(msg["text"])
                    if obj.get("type") == "close":
                        await fire_turn_end("close")
                        break
        except Exception:
            pass
        finally:
            closed.set()

    async def from_stt():
        try:
            async for raw in up_stt:
                if isinstance(raw, bytes):
                    continue
                m = json.loads(raw)
                t = m.get("type")
                if t == "partial":
                    state["partial"] = m.get("text", "")
                    mark_voice()
                    await _browser_send(browser, {"type": "partial", "text": state["partial"]})
                    await feed_text_model()
                elif t == "final":
                    # NB: do NOT mark_voice — a final arrives *after* the pause,
                    # so it must not reset the silence clock.
                    if m.get("text"):
                        state["turn_finals"].append(m["text"])
                    state["partial"] = ""
                    await _browser_send(browser, {"type": "final", "text": m.get("text", "")})
                    await feed_text_model()
                elif t == "vad_speech":
                    mark_voice()
        except Exception:
            pass
        finally:
            closed.set()

    async def from_audio():
        try:
            async for raw in up_audio:
                m = json.loads(raw)
                if m.get("type") == "probability":
                    state["p_audio"] = m.get("p_end_of_turn", 0.0)
                    await _browser_send(browser, {"type": "eou", "source": "audio", "p": state["p_audio"]})
        except Exception:
            pass

    async def from_text():
        try:
            async for raw in up_text:
                m = json.loads(raw)
                if m.get("type") == "probability":
                    state["p_text"] = m.get("p_end_of_turn", 0.0)
                    state["conf_text"] = m.get("confidence", 0.0) or 0.0
                    await _browser_send(browser, {
                        "type": "eou", "source": "text", "p": state["p_text"], "confidence": state["conf_text"]})
        except Exception:
            pass

    async def decision_watch():
        # The fusion lives here: only decide at a pause.
        try:
            while not closed.is_set():
                await asyncio.sleep(0.12)
                if not state["turn_active"]:
                    continue
                silence_ms = int((time.monotonic() - state["last_voice"]) * 1000)
                await _browser_send(browser, {"type": "silence", "ms": silence_ms})
                if silence_ms < ENDPOINT_MIN_MS:
                    continue
                if state["p_audio"] >= AUDIO_FIRE:
                    await fire_turn_end("audio")
                elif state["conf_text"] >= TEXT_FIRE:
                    await fire_turn_end("text")
                elif silence_ms >= SILENCE_BACKSTOP_MS:
                    await fire_turn_end("silence")
        except Exception:
            pass

    tasks = [asyncio.create_task(c) for c in
             (from_browser(), from_stt(), from_audio(), from_text(), decision_watch())]
    await closed.wait()
    for t in tasks:
        t.cancel()
    for up in (up_stt, up_audio, up_text):
        try:
            await up.close()
        except Exception:
            pass
    try:
        await browser.close()
    except Exception:
        pass


app = Starlette(routes=[Route("/", index), WebSocketRoute("/v1/stream", bridge)])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8080)
    args = p.parse_args()
    print(f"stt={STT_BACKEND} (key {'ok' if STT_API_KEY else 'MISSING'})", flush=True)
    print(f"turn-detection={TD_BASE} (key {'ok' if TD_API_KEY else 'MISSING'})", flush=True)
    print(f"fusion: endpoint_min={ENDPOINT_MIN_MS}ms audio_fire={AUDIO_FIRE} "
          f"text_fire={TEXT_FIRE} silence_backstop={SILENCE_BACKSTOP_MS}ms", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning", access_log=False)


if __name__ == "__main__":
    main()
