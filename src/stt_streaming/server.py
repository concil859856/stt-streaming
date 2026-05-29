"""FastAPI entrypoint for stt-streaming."""
from __future__ import annotations

import asyncio
import io
import logging
import time
from contextlib import asynccontextmanager

import numpy as np
import soundfile as sf
import uvicorn
from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile, WebSocket
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


@app.post("/v1/transcribe")
async def transcribe_file(
    request: Request,
    audio: UploadFile = File(...),
    language: str | None = Form(None),
    x_api_key: str | None = Header(default=None),
):
    """Batch transcription — accepts any audio file ffmpeg/libsndfile can read."""
    _check_api_key(x_api_key, request.app.state.config)
    raw = await audio.read()
    if not raw:
        raise HTTPException(status_code=400, detail={"error": "empty audio"})
    if len(raw) > 100 * 1024 * 1024:
        raise HTTPException(status_code=413, detail={"error": "audio too large (max 100 MB)"})

    started = time.perf_counter()
    try:
        pcm, sr = _decode_audio_to_16k_mono(raw)
    except Exception as e:
        raise HTTPException(status_code=400, detail={"error": f"audio decode failed: {e}"})

    model: ParakeetModel = request.app.state.model
    try:
        text = await asyncio.get_event_loop().run_in_executor(None, model._run_transcribe, pcm)
    except Exception as e:
        logger.exception("transcribe failed")
        raise HTTPException(status_code=500, detail={"error": "internal", "message": str(e)})

    latency_ms = int((time.perf_counter() - started) * 1000)
    audio_ms = int(len(pcm) * 1000 / 16000)
    metrics.record_request_finished(status="ok", duration_ms=latency_ms, audio_ms=audio_ms,
                                    bytes_received=len(raw))
    return JSONResponse({
        "text": text.strip(),
        "language": language,
        "audio_ms": audio_ms,
        "latency_ms": latency_ms,
        "model": healthz._state.model,
    })


def _decode_audio_to_16k_mono(raw: bytes) -> tuple[np.ndarray, int]:
    """Decode any soundfile/ffmpeg-readable audio to 16 kHz mono float32."""
    buf = io.BytesIO(raw)
    try:
        data, sr = sf.read(buf, dtype="float32", always_2d=False)
    except Exception:
        buf.seek(0)
        import librosa
        data, sr = librosa.load(buf, sr=None, mono=False)
    if data.ndim > 1:
        data = data.mean(axis=-1) if data.shape[-1] <= 8 else data.mean(axis=0)
    if sr != 16000:
        import librosa
        data = librosa.resample(data.astype(np.float32), orig_sr=sr, target_sr=16000)
        sr = 16000
    return data.astype(np.float32), sr


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
