#!/usr/bin/env python3
"""One-port test harness for the mic page.

Serves mic_test.html AND reverse-proxies its WebSocket to the real stt-streaming
pod, injecting the X-API-Key header. So the browser only needs ONE forwarded
port and no API key in the page — open http://localhost:8080/ and talk.

    STT_BACKEND=ws://localhost:8117/v1/stream STT_API_KEY=<key> \
        python3 examples/mic_proxy.py --port 8080
"""
from __future__ import annotations

import argparse
import asyncio
import os

import uvicorn
import websockets
from starlette.applications import Starlette
from starlette.responses import FileResponse
from starlette.routing import Route, WebSocketRoute

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.environ.get("STT_BACKEND", "ws://localhost:8117/v1/stream")


def _resolve_api_key() -> str:
    """Use STT_API_KEY if set; otherwise read ASR_API_KEY straight out of the
    running pod container so you don't have to handle the secret yourself."""
    key = os.environ.get("STT_API_KEY", "")
    if key:
        return key
    import subprocess
    container = os.environ.get("STT_CONTAINER", "vocence-asr_streaming_rt-6")
    try:
        out = subprocess.run(
            ["docker", "inspect", container,
             "--format", "{{range .Config.Env}}{{println .}}{{end}}"],
            capture_output=True, text=True, timeout=10,
        ).stdout
        for line in out.splitlines():
            if line.startswith("ASR_API_KEY="):
                return line.split("=", 1)[1]
    except Exception:
        pass
    return ""


API_KEY = _resolve_api_key()


async def index(request):
    return FileResponse(os.path.join(HERE, "mic_test.html"))


async def proxy(ws):
    await ws.accept()
    headers = [("X-API-Key", API_KEY)] if API_KEY else []
    try:
        upstream = await websockets.connect(BACKEND, additional_headers=headers, max_size=2 ** 21)
    except Exception as e:
        await ws.close(code=1011, reason=f"backend connect failed: {e}")
        return

    async def client_to_backend():
        try:
            while True:
                msg = await ws.receive()
                if msg["type"] == "websocket.disconnect":
                    break
                if msg.get("bytes") is not None:
                    await upstream.send(msg["bytes"])
                elif msg.get("text") is not None:
                    await upstream.send(msg["text"])
        except Exception:
            pass
        finally:
            await upstream.close()

    async def backend_to_client():
        try:
            async for m in upstream:
                if isinstance(m, bytes):
                    await ws.send_bytes(m)
                else:
                    await ws.send_text(m)
        except Exception:
            pass
        finally:
            try:
                await ws.close()
            except Exception:
                pass

    await asyncio.gather(client_to_backend(), backend_to_client())


app = Starlette(routes=[Route("/", index), WebSocketRoute("/v1/stream", proxy)])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8080)
    args = p.parse_args()
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info", access_log=True)


if __name__ == "__main__":
    main()
