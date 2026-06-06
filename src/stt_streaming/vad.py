from __future__ import annotations

import logging
import threading
from typing import List, Tuple

import numpy as np

log = logging.getLogger(__name__)

# Silero v5 consumes fixed 512-sample windows at 16 kHz (256 @ 8 kHz). Feeding it
# isolated 20 ms WS frames returns *no* speech ever — it needs streaming context —
# which is why per-frame get_speech_timestamps() silently gated out every partial.
_WINDOW_SAMPLES = 512
# Hysteresis thresholds: enter speech high, leave low, so brief dips mid-word do
# not bounce the gate and prematurely commit utterances.
_SPEECH_ON = 0.5
_SPEECH_OFF = 0.35


class SileroVAD:
    def __init__(self) -> None:
        self._model = None
        self._get_speech_timestamps = None
        # The silero model carries its LSTM state on the object itself, so a single
        # shared instance is not safe across concurrent sessions. We serialize the
        # restore->infer->save critical section and stash each session's state in
        # the per-session dict returned by reset_state().
        self._lock = threading.Lock()
        try:
            from silero_vad import load_silero_vad, get_speech_timestamps  # type: ignore

            self._model = load_silero_vad()
            self._get_speech_timestamps = get_speech_timestamps
        except Exception as e:  # pragma: no cover - exercised only without silero installed
            log.warning("silero-vad unavailable (%s); falling back to RMS energy threshold", e)

    def reset_state(self) -> dict:
        # buf: leftover samples not yet forming a full 512 window
        # triggered: current speech/silence latch (hysteresis)
        # silero: this session's silero internals — recurrent _state plus the
        #   64-sample _context carried between windows (None until first window)
        return {"buf": np.zeros(0, dtype=np.float32), "triggered": False, "silero": None}

    def is_speech(self, audio_chunk: np.ndarray, state=None, sample_rate: int = 16000):
        if isinstance(state, int):
            sample_rate = state
            state = None
        if state is None:
            state = self.reset_state()
        if self._model is None:
            return _rms_is_speech(audio_chunk), state
        try:
            triggered = self._silero_stream(audio_chunk, state, sample_rate)
        except Exception as e:
            log.warning("silero streaming VAD failed (%s); using RMS fallback", e)
            return _rms_is_speech(audio_chunk), state
        return triggered, state

    def _silero_stream(self, audio_chunk: np.ndarray, state: dict, sample_rate: int) -> bool:
        """Run silero over freshly-completed 512-sample windows, carrying this
        session's recurrent state across calls. Concurrency-safe: the shared
        model's state is swapped in/out under a lock per call."""
        import torch

        buf = np.concatenate([state["buf"], _to_float32(audio_chunk)])
        max_prob: float | None = None
        with self._lock:
            # Load this session's silero internals into the shared model.
            if state["silero"] is None:
                self._model.reset_states()
            else:
                self._model._state = state["silero"]["state"]
                self._model._context = state["silero"]["context"]
                self._model._last_sr = sample_rate
                self._model._last_batch_size = 1
            with torch.no_grad():
                while buf.shape[0] >= _WINDOW_SAMPLES:
                    window = buf[:_WINDOW_SAMPLES]
                    buf = buf[_WINDOW_SAMPLES:]
                    p = float(self._model(torch.from_numpy(window.copy()), sample_rate).item())
                    max_prob = p if max_prob is None else max(max_prob, p)
            # Save this session's internals before another session borrows the model.
            st = getattr(self._model, "_state", None)
            ctx = getattr(self._model, "_context", None)
            state["silero"] = {
                "state": st.clone() if st is not None else None,
                "context": ctx.clone() if ctx is not None else None,
            }

        state["buf"] = buf
        if max_prob is not None:
            if max_prob >= _SPEECH_ON:
                state["triggered"] = True
            elif max_prob < _SPEECH_OFF:
                state["triggered"] = False
        # No full window yet (sub-32 ms chunk): hold the previous latch.
        return state["triggered"]

    def process_stream(self, audio_buffer: np.ndarray) -> List[Tuple[int, int, bool]]:
        sample_rate = 16000
        if self._model is None:
            return _rms_segments(audio_buffer, sample_rate)
        try:
            import torch

            tensor = torch.from_numpy(_to_float32(audio_buffer))
            ts = self._get_speech_timestamps(
                tensor, self._model, sampling_rate=sample_rate, return_seconds=False
            )
            out: List[Tuple[int, int, bool]] = []
            cursor = 0
            total = len(audio_buffer)
            for seg in ts:
                start = int(seg["start"])
                end = int(seg["end"])
                if start > cursor:
                    out.append((_samples_to_ms(cursor, sample_rate), _samples_to_ms(start, sample_rate), False))
                out.append((_samples_to_ms(start, sample_rate), _samples_to_ms(end, sample_rate), True))
                cursor = end
            if cursor < total:
                out.append((_samples_to_ms(cursor, sample_rate), _samples_to_ms(total, sample_rate), False))
            return out
        except Exception as e:
            log.warning("silero VAD stream inference failed (%s); using RMS fallback", e)
            return _rms_segments(audio_buffer, sample_rate)


def _to_float32(audio: np.ndarray) -> np.ndarray:
    if audio.dtype == np.float32:
        return audio
    if audio.dtype == np.int16:
        return (audio.astype(np.float32) / 32768.0)
    return audio.astype(np.float32)


def _rms_is_speech(audio: np.ndarray) -> bool:
    a = _to_float32(audio)
    if a.size == 0:
        return False
    rms = float(np.sqrt(np.mean(a * a)))
    return rms > 0.01


def _samples_to_ms(n: int, sr: int) -> int:
    return int(round(n * 1000 / sr))


def _rms_segments(audio: np.ndarray, sample_rate: int) -> List[Tuple[int, int, bool]]:
    # Fixed 30 ms windows — cheap fallback, not as accurate as Silero but stable.
    win = max(1, int(sample_rate * 0.03))
    a = _to_float32(audio)
    n = len(a)
    out: List[Tuple[int, int, bool]] = []
    i = 0
    cur_speech = None
    cur_start = 0
    while i < n:
        chunk = a[i : i + win]
        is_sp = bool(np.sqrt(np.mean(chunk * chunk)) > 0.01) if chunk.size else False
        if cur_speech is None:
            cur_speech = is_sp
            cur_start = i
        elif is_sp != cur_speech:
            out.append((_samples_to_ms(cur_start, sample_rate), _samples_to_ms(i, sample_rate), cur_speech))
            cur_speech = is_sp
            cur_start = i
        i += win
    if cur_speech is not None:
        out.append((_samples_to_ms(cur_start, sample_rate), _samples_to_ms(n, sample_rate), cur_speech))
    return out


def load_vad() -> SileroVAD:
    return SileroVAD()
