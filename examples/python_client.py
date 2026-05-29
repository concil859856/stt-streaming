#!/usr/bin/env python3
"""USAGE EXAMPLE: minimal reference Python client for stt-streaming.

This file exists to show, end to end, how a third-party application speaks
the WebSocket protocol described in stt-streaming-implementation.md §5.
It is intentionally small (no fancy reconnect, no streaming output buffer,
no metrics) so the wire protocol is obvious.

Two input modes:

    # Stream a WAV file at real-time pace
    python examples/python_client.py --wav my_speech.wav \
        --url ws://localhost:8114/v1/stream --api-key KEY

    # Stream from the default microphone (requires sounddevice)
    python examples/python_client.py --mic \
        --url ws://localhost:8114/v1/stream --api-key KEY
"""
from __future__ import annotations

import argparse
import asyncio
import json
import wave

import websockets

SAMPLE_RATE = 16000
CHUNK_SAMPLES = 320  # 20 ms at 16 kHz — recommended by the spec


async def _stream(url: str, api_key: str, frame_iter) -> None:
    # The X-API-Key header gates the WS upgrade. Wrong/missing key -> close 4401.
    async with websockets.connect(
        url, additional_headers=[("X-API-Key", api_key)]
    ) as ws:
        # 1. Send the single mandatory `start` message. Sample rate MUST be
        #    16000 and encoding MUST be pcm_s16le; anything else closes 4400.
        await ws.send(json.dumps({
            "type": "start",
            "language": "auto",
            "sample_rate": SAMPLE_RATE,
            "encoding": "pcm_s16le",
            "enable_partials": True,
        }))

        # 2. Wait for the `ready` reply, which echoes the server-assigned
        #    session id and the resolved model name.
        ready = json.loads(await ws.recv())
        print(f"[ready] {ready}")

        async def _send_audio():
            # 3. Stream raw PCM bytes as WS binary frames. The server accepts
            #    any frame between 80 and 1600 samples — we use the
            #    recommended 320 (20 ms) chunks.
            for frame in frame_iter:
                await ws.send(frame)
            # 4. Tell the server we're done; it will flush a final transcript.
            await ws.send(json.dumps({"type": "close"}))

        async def _recv_msgs():
            # 5. Receive partials / finals until the server closes the WS.
            try:
                async for raw in ws:
                    if isinstance(raw, bytes):
                        continue
                    msg = json.loads(raw)
                    if msg["type"] == "partial":
                        print(f"  ... {msg['text']}")
                    elif msg["type"] == "final":
                        print(f"  >>> {msg['text']}")
                    elif msg["type"] == "error":
                        print(f"  !!! {msg.get('code')}: {msg.get('message')}")
            except websockets.ConnectionClosed:
                pass

        await asyncio.gather(_send_audio(), _recv_msgs())


def _wav_frames(path: str):
    """Yield 20-ms PCM16 chunks from a 16 kHz mono WAV at real-time pace."""
    chunk_bytes = CHUNK_SAMPLES * 2
    with wave.open(path, "rb") as wf:
        assert wf.getframerate() == SAMPLE_RATE and wf.getnchannels() == 1
        data = wf.readframes(wf.getnframes())
    for i in range(0, len(data), chunk_bytes):
        yield data[i : i + chunk_bytes]


async def _mic_frames(duration_sec: float):
    """Yield 20-ms PCM16 chunks captured from the default microphone."""
    import sounddevice as sd  # optional dep; only needed for --mic
    q: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_event_loop()

    def _cb(indata, frames, time_info, status):
        loop.call_soon_threadsafe(q.put_nowait, bytes(indata))

    with sd.RawInputStream(samplerate=SAMPLE_RATE, channels=1,
                           dtype="int16", blocksize=CHUNK_SAMPLES, callback=_cb):
        end = loop.time() + duration_sec
        while loop.time() < end:
            yield await q.get()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--url", required=True)
    p.add_argument("--api-key", required=True)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--wav", help="Path to a 16 kHz mono PCM16 WAV file")
    src.add_argument("--mic", action="store_true", help="Capture from microphone")
    p.add_argument("--mic-seconds", type=float, default=10.0)
    args = p.parse_args()

    async def _go():
        if args.wav:
            await _stream(args.url, args.api_key, _wav_frames(args.wav))
        else:
            async def _agen():
                async for f in _mic_frames(args.mic_seconds):
                    yield f
            # Bridge async-iter to sync-iter expected by _stream.
            frames: list[bytes] = []
            async for f in _mic_frames(args.mic_seconds):
                frames.append(f)
            await _stream(args.url, args.api_key, iter(frames))

    asyncio.run(_go())


if __name__ == "__main__":
    main()
