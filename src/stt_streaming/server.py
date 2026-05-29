"""FastAPI entrypoint for stt-streaming."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Header, HTTPException, Request, WebSocket
from fastapi.responses import JSONResponse, Response

from . import healthz, metrics, ws as ws_module
from .config import load_config
from .model import ParakeetModel
from .vad import load_vad

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model + VAD, warm up, mark health ok."""
    config = load_config()
    logging.basicConfig(level=getattr(logging, config.log_level.upper(), logging.INFO))
    healthz.set_status("warming")

    logger.info("Loading model")
    model = ParakeetModel(config.model_dir)
    logger.info("Loading VAD")
    vad = load_vad()
    logger.info("Warming up")
    model.warm_up()
    healthz.set_status("ok")

    app.state.config = config
    app.state.model = model
    app.state.vad = vad
    logger.info("Ready on port %d", config.port)
    try:
        yield
    finally:
        healthz.set_status("degraded")
        logger.info("Shutting down")


app = FastAPI(title="stt-streaming", lifespan=lifespan)


def _check_api_key(request_key: str | None, config) -> None:
    if config.api_key and request_key != config.api_key:
        raise HTTPException(status_code=401, detail={"error": "unauthorized"})


@app.get("/healthz")
async def get_healthz(request: Request, x_api_key: str | None = Header(default=None)):
    _check_api_key(x_api_key, request.app.state.config)
    return JSONResponse(healthz.get_health_state(request.app.state.config))


@app.get("/metrics")
async def get_metrics(request: Request, x_api_key: str | None = Header(default=None)):
    _check_api_key(x_api_key, request.app.state.config)
    return Response(metrics.export_metrics(), media_type="text/plain; version=0.0.4")


@app.websocket("/v1/stream")
async def stream(websocket: WebSocket):
    config = websocket.app.state.config
    model = websocket.app.state.model
    vad = websocket.app.state.vad
    await ws_module.handle_session(websocket, model, vad, config)


def main() -> None:
    config = load_config()
    uvicorn.run(
        "stt_streaming.server:app",
        host="0.0.0.0",
        port=config.port,
        log_level=config.log_level.lower(),
        ws_max_size=2 * 1024 * 1024,
    )


if __name__ == "__main__":
    main()
