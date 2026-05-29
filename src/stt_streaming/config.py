from __future__ import annotations

import os
from dataclasses import dataclass


def _as_bool(v: str | None) -> bool:
    if v is None:
        return False
    return v.strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class Config:
    api_key: str
    model: str
    model_dir: str
    port: int
    max_concurrent: int
    internal_silence_ms: int
    partial_interval_ms: int
    max_session_seconds: int
    log_level: str
    log_payloads: bool


def load_config() -> Config:
    # Refuse to start without an API key — no anonymous mode (spec §4.4).
    api_key = os.environ.get("ASR_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("ASR_API_KEY is required; refusing to start without it")

    return Config(
        api_key=api_key,
        model=os.environ.get("ASR_MODEL", "nvidia/parakeet-tdt-0.6b-v3"),
        model_dir=os.environ.get("ASR_MODEL_DIR", "/cache/models/parakeet"),
        port=int(os.environ.get("ASR_PORT", "8117")),
        max_concurrent=int(os.environ.get("ASR_MAX_CONCURRENT", "32")),
        internal_silence_ms=int(os.environ.get("ASR_INTERNAL_SILENCE_MS", "800")),
        partial_interval_ms=int(os.environ.get("ASR_PARTIAL_INTERVAL_MS", "200")),
        max_session_seconds=int(os.environ.get("ASR_MAX_SESSION_SECONDS", "1800")),
        log_level=os.environ.get("ASR_LOG_LEVEL", "info").lower(),
        log_payloads=_as_bool(os.environ.get("ASR_LOG_PAYLOADS", "0")),
    )
