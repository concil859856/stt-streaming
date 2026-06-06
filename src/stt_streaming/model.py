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

        logger.info("Loading NeMo ASR model: %s (device=%s)", model_dir, self._device)
        # Two paths:
        #  - local .nemo file path (restore_from)
        #  - HF model id like "nvidia/parakeet-tdt-0.6b-v3" (from_pretrained downloads to HF_HOME)
        import os
        if os.path.exists(model_dir) and os.path.isfile(model_dir):
            self.model = ASRModel.restore_from(restore_path=model_dir, map_location=self._device)
        else:
            model_id = os.environ.get("ASR_MODEL", "nvidia/parakeet-tdt-0.6b-v3")
            self.model = ASRModel.from_pretrained(model_name=model_id, map_location=self._device)

        self.model = self.model.to(self._device)
        self.model.eval()
        # Detect whether this checkpoint supports cache-aware streaming
        self._has_streaming = hasattr(self.model, "conformer_stream_step") or hasattr(
            self.model, "transcribe_simulate_cache_aware_streaming"
        )
        logger.info("Model loaded on %s (streaming=%s)", self._device, self._has_streaming)

    def reset_state(self) -> InferState:
        return InferState()

    def transcribe_window(self, audio: np.ndarray) -> str:
        """Stateless re-decode of a PCM window; returns the transcript.

        Parakeet TDT via NeMo does not expose a stable per-chunk streaming step
        for all checkpoints, so we re-decode the (bounded) cumulative utterance
        window each call. The caller owns the audio buffer; this method keeps no
        audio state of its own, which is what avoids double-counting.
        """
        if audio is None or audio.shape[0] == 0:
            return ""
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        # Bound the window so latency does not grow without limit on long utterances.
        if audio.shape[0] > _FALLBACK_CONTEXT_SAMPLES:
            audio = audio[-_FALLBACK_CONTEXT_SAMPLES:]
        return self._run_transcribe(audio)

    def transcribe_chunk(
        self, audio_window: np.ndarray, state: Optional[InferState]
    ) -> Tuple[str, InferState]:
        """Re-decode the current utterance window; returns (cumulative_text, new_state).

        ``audio_window`` is the full cumulative PCM for the utterance so far — the
        caller (ws session) owns that buffer. We re-decode it rather than appending
        to per-chunk state, so audio the caller has already buffered is never
        counted twice. The GPU lock in ``_run_transcribe`` serializes device work.
        """
        if state is None:
            state = self.reset_state()
        text = self.transcribe_window(audio_window)
        state.last_text = text
        return text, state

    def finalize(self, state: InferState) -> str:
        """Final flush — best hypothesis on the last decoded window.

        Prefer :meth:`transcribe_window` on the caller's live buffer; this is kept
        for callers that only hold the opaque state.
        """
        if state is None:
            return ""
        if state.audio.shape[0] == 0:
            return state.last_text
        return self.transcribe_window(state.audio)

    def warm_up(self) -> None:
        """One dummy transcribe over 1 s of silence to compile kernels."""
        logger.info("Warming up model")
        silence = np.zeros(SAMPLE_RATE, dtype=np.float32)
        try:
            self._run_transcribe(silence)
            logger.info("Warm-up complete")
        except Exception:
            logger.exception("Warm-up failed (continuing)")

    def _bound(self, audio: np.ndarray) -> Optional[np.ndarray]:
        if audio is None or audio.shape[0] == 0:
            return None
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        if audio.shape[0] > _FALLBACK_CONTEXT_SAMPLES:
            audio = audio[-_FALLBACK_CONTEXT_SAMPLES:]
        return audio

    def try_transcribe_window(self, audio: np.ndarray) -> Optional[str]:
        """Best-effort partial decode: returns the transcript, or ``None`` if the
        GPU is already busy.

        Each NeMo ``transcribe`` call costs a fixed ~55 ms (Python/dataloader
        overhead, independent of window length), so the lock-serialized model
        caps out around ~18 calls/s. Partials are disposable — if we blocked
        here, ticks from many concurrent sessions would queue without bound and
        starve the event loop (uvicorn's WS keepalive then drops the socket).
        Skipping a busy tick keeps the loop responsive; the next frame retries.
        """
        bounded = self._bound(audio)
        if bounded is None:
            return ""
        if not self._lock.acquire(blocking=False):
            return None
        try:
            return self._infer(bounded)
        finally:
            self._lock.release()

    def _run_transcribe(self, audio: np.ndarray) -> str:
        """Underlying NeMo call. Holds the GPU lock for the duration (blocking)."""
        with self._lock:
            return self._infer(audio)

    def _infer(self, audio: np.ndarray) -> str:
        """Run NeMo transcribe on one window. Caller must hold ``self._lock``."""
        import torch

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
