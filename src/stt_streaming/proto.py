from __future__ import annotations

from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, Field, ValidationError, field_validator


# ---- client -> server ----

class StartMessage(BaseModel):
    type: Literal["start"]
    session_id: Optional[str] = None
    language: str = "auto"
    sample_rate: int = 16000
    encoding: str = "pcm_s16le"
    enable_partials: bool = True
    vad_events: bool = False
    metadata: Optional[Dict[str, Any]] = None

    @field_validator("sample_rate")
    @classmethod
    def _only_16k(cls, v: int) -> int:
        if v != 16000:
            raise ValueError("sample_rate must be 16000")
        return v

    @field_validator("encoding")
    @classmethod
    def _only_pcm16(cls, v: str) -> str:
        if v != "pcm_s16le":
            raise ValueError("encoding must be 'pcm_s16le'")
        return v


class CommitMessage(BaseModel):
    type: Literal["commit"]


class CloseMessage(BaseModel):
    type: Literal["close"]


class PingMessage(BaseModel):
    type: Literal["ping"]
    ts: Optional[int] = None


# ---- server -> client ----

class ReadyMessage(BaseModel):
    type: Literal["ready"] = "ready"
    session_id: str
    model: str
    language: str
    sample_rate: int


class PartialMessage(BaseModel):
    type: Literal["partial"] = "partial"
    text: str
    since_session_start_ms: int
    audio_ms_consumed: int


class FinalMessage(BaseModel):
    type: Literal["final"] = "final"
    text: str
    utterance_start_ms: int
    utterance_end_ms: int
    audio_ms_consumed: int
    confidence: float
    language_detected: Optional[str] = None


class VadSpeechMessage(BaseModel):
    type: Literal["vad_speech"] = "vad_speech"
    audio_ms_consumed: int


class VadSilenceMessage(BaseModel):
    type: Literal["vad_silence"] = "vad_silence"
    audio_ms_consumed: int
    silence_ms: int


class ErrorMessage(BaseModel):
    type: Literal["error"] = "error"
    code: str
    message: str


class PongMessage(BaseModel):
    type: Literal["pong"] = "pong"
    ts: Optional[int] = None


def validate_start(data: dict) -> StartMessage:
    if not isinstance(data, dict):
        raise ValueError("start message must be a JSON object")
    if data.get("type") != "start":
        raise ValueError("first message must have type='start'")
    try:
        return StartMessage(**data)
    except ValidationError as e:
        first = e.errors()[0]
        loc = ".".join(str(p) for p in first.get("loc", ()))
        raise ValueError(f"invalid start message: {loc}: {first.get('msg')}") from e


_CLIENT_TYPES = {
    "start": StartMessage,
    "commit": CommitMessage,
    "close": CloseMessage,
    "ping": PingMessage,
}


def parse_client_message(data: dict):
    if not isinstance(data, dict):
        raise ValueError("client message must be a JSON object")
    t = data.get("type")
    cls = _CLIENT_TYPES.get(t)
    if cls is None:
        raise ValueError(f"unknown client message type: {t!r}")
    return cls(**data)
