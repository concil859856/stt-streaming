"""NeMo Parakeet TDT streaming wrapper."""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# 16 kHz mono is the only supported rate per spec §6
SAMPLE_RATE = 16000
# Sliding-window context kept across chunks when the streaming API is unavailable
_FALLBACK_CONTEXT_SAMPLES = SAMPLE_RATE * 8  # 8 s rolling buffer


@dataclass
class InferState:
    """Opaque per-utterance state passed back and forth across chunk calls."""
    # Rolling raw audio buffer for fallback sliding-window mode
    audio: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    # Last emitted hypothesis text (so caller can diff if needed)
    last_text: str = ""
    # Optional native NeMo streaming cache (cache-aware streaming models populate this)
    cache: Any = None


class ParakeetModel:
    """Thread-safe NeMo Parakeet wrapper. Serializes GPU work via a lock."""

    def __init__(self, model_dir: str) -> None:
        import torch
        from nemo.collections.asr.models import ASRModel

        self.model_dir = model_dir
        self._lock = threading.Lock()
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info("Loading NeMo ASR model from %s", model_dir)
        # Prefer restore_from for a local .nemo snapshot; fall back to from_pretrained on a dir
        try:
            self.model = ASRModel.restore_from(restore_path=model_dir, map_location=self._device)
        except Exception:
            self.model = ASRModel.from_pretrained(model_name=model_dir, map_location=self._device)

        self.model = self.model.to(self._device)
        self.model.eval()
        # Detect whether this checkpoint supports cache-aware streaming
        self._has_streaming = hasattr(self.model, "conformer_stream_step") or hasattr(
            self.model, "transcribe_simulate_cache_aware_streaming"
        )
        logger.info("Model loaded on %s (streaming=%s)", self._device, self._has_streaming)

    def reset_state(self) -> InferState:
        return InferState()

    def transcribe_chunk(
        self, audio_chunk: np.ndarray, state: Optional[InferState]
    ) -> Tuple[str, InferState]:
        """Run incremental transcription on one PCM chunk; returns (cumulative_text, new_state)."""
        if state is None:
            state = self.reset_state()

        if audio_chunk.dtype != np.float32:
            audio_chunk = audio_chunk.astype(np.float32)

        # Append to rolling buffer; we always re-decode the buffer because Parakeet TDT
        # via NeMo 2.0.0 does not expose a stable per-chunk streaming step for all checkpoints.
        # The lock keeps GPU access serialized; concurrency is achieved across sessions by queueing.
        state.audio = np.concatenate([state.audio, audio_chunk])
        # Trim to a bounded sliding window so latency does not grow without bound
        if state.audio.shape[0] > _FALLBACK_CONTEXT_SAMPLES:
            state.audio = state.audio[-_FALLBACK_CONTEXT_SAMPLES:]

        text = self._run_transcribe(state.audio)
        state.last_text = text
        return text, state

    def finalize(self, state: InferState) -> str:
        """Final flush — return best hypothesis on whatever audio is buffered."""
        if state is None or state.audio.shape[0] == 0:
            return state.last_text if state else ""
        return self._run_transcribe(state.audio)

    def warm_up(self) -> None:
        """One dummy transcribe over 1 s of silence to compile kernels."""
        logger.info("Warming up model")
        silence = np.zeros(SAMPLE_RATE, dtype=np.float32)
        try:
            self._run_transcribe(silence)
            logger.info("Warm-up complete")
        except Exception:
            logger.exception("Warm-up failed (continuing)")

    def _run_transcribe(self, audio: np.ndarray) -> str:
        """Underlying NeMo call. Holds the GPU lock for the duration."""
        import torch

        with self._lock:
            with torch.inference_mode():
                # NeMo 2.0 transcribe accepts a list of numpy arrays in newer builds
                try:
                    out = self.model.transcribe([audio], batch_size=1, verbose=False)
                except TypeError:
                    out = self.model.transcribe([audio], batch_size=1)
        return _extract_text(out)


def _extract_text(out: Any) -> str:
    """Normalize NeMo transcribe return shape across versions."""
    if not out:
        return ""
    first = out[0] if isinstance(out, (list, tuple)) else out
    # NeMo may return Hypothesis, str, or a list of those
    if isinstance(first, (list, tuple)):
        first = first[0] if first else ""
    if hasattr(first, "text"):
        return (first.text or "").strip()
    if isinstance(first, str):
        return first.strip()
    return str(first).strip()
