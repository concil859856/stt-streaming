#!/usr/bin/env python3
"""One-WAV smoke test for a running stt-streaming pod.

Streams a 16 kHz mono PCM16 WAV file to ``ws://host:port/v1/stream`` in
20 ms chunks, prints every partial / final message, and asserts that the
WebSocket closes cleanly with code 1000.

Usage::

    python scripts/smoke_test.py \
        --url ws://localhost:8114/v1/stream \
        --api-key test_key_local_only \
        --wav tests/fixtures/librispeech_short.wav
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import wave

import websockets

CHUNK_SAMPLES = 320  # 20 ms at 16 kHz
SAMPLE_RATE = 16000


def _read_wav_pcm16(path: str) -> bytes:
    with wave.open(path, "rb") as wf:
        if wf.getnchannels() != 1:
            raise SystemExit(f"WAV must be mono; got {wf.getnchannels()} channels")
        if wf.getsampwidth() != 2:
            raise SystemExit(f"WAV must be 16-bit; got {8 * wf.getsampwidth()}-bit")
        if wf.getframerate() != SAMPLE_RATE:
            raise SystemExit(f"WAV must be {SAMPLE_RATE} Hz; got {wf.getframerate()}")
        return wf.readframes(wf.getnframes())


async def _run(url: str, api_key: str, wav: str) -> int:
    pcm = _read_wav_pcm16(wav)
    chunk_bytes = CHUNK_SAMPLES * 2

    saw_partial = False
    saw_final = False
    close_code: int | None = None

    headers = [("X-API-Key", api_key)]
    async with websockets.connect(url, additional_headers=headers) as ws:
        await ws.send(json.dumps({
            "type": "start",
            "language": "auto",
            "sample_rate": SAMPLE_RATE,
            "encoding": "pcm_s16le",
            "enable_partials": True,
        }))

        ready = json.loads(await ws.recv())
        if ready.get("type") != "ready":
            raise SystemExit(f"expected ready, got: {ready}")
        print(f"[ready] session={ready.get('session_id')} model={ready.get('model')}")

        async def _send_audio():
            for i in range(0, len(pcm), chunk_bytes):
                await ws.send(pcm[i : i + chunk_bytes])
                # Pace at real-time so the server's VAD / partial cadence works
                # as it would for a live mic source.
                await asyncio.sleep(0.02)
            await ws.send(json.dumps({"type": "close"}))

        async def _recv_loop():
            nonlocal saw_partial, saw_final
            try:
                async for raw in ws:
                    if isinstance(raw, bytes):
                        continue
                    msg = json.loads(raw)
                    t = msg.get("type")
                    if t == "partial":
                        saw_partial = True
                        print(f"[partial] {msg.get('text')!r}")
                    elif t == "final":
                        saw_final = True
                        print(f"[final]   {msg.get('text')!r} "
                              f"(conf={msg.get('confidence')})")
                    elif t == "error":
                        print(f"[error]   {msg.get('code')}: {msg.get('message')}",
                              file=sys.stderr)
                    elif t == "pong":
                        pass
                    else:
                        print(f"[?]       {msg}")
            except websockets.ConnectionClosed:
                pass

        send_task = asyncio.create_task(_send_audio())
        recv_task = asyncio.create_task(_recv_loop())
        await asyncio.gather(send_task, recv_task)
        close_code = ws.close_code

    assert close_code == 1000, f"expected close code 1000, got {close_code}"
    assert saw_final, "did not receive any final transcript"
    if not saw_partial:
        print("[warn] no partial messages received (enable_partials was true)",
              file=sys.stderr)
    print("[ok] smoke test passed")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", required=True, help="ws://host:port/v1/stream")
    p.add_argument("--api-key", required=True)
    p.add_argument("--wav", required=True, help="Path to a 16 kHz mono PCM16 WAV")
    args = p.parse_args()
    return asyncio.run(_run(args.url, args.api_key, args.wav))


if __name__ == "__main__":
    raise SystemExit(main())
