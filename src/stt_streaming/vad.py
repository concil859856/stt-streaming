from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np

log = logging.getLogger(__name__)


class SileroVAD:
    def __init__(self) -> None:
        self._model = None
        self._get_speech_timestamps = None
        try:
            from silero_vad import load_silero_vad, get_speech_timestamps  # type: ignore

            self._model = load_silero_vad()
            self._get_speech_timestamps = get_speech_timestamps
        except Exception as e:  # pragma: no cover - exercised only without silero installed
            log.warning("silero-vad unavailable (%s); falling back to RMS energy threshold", e)

    def is_speech(self, audio_chunk: np.ndarray, sample_rate: int = 16000) -> bool:
        if self._model is None:
            return _rms_is_speech(audio_chunk)
        try:
            import torch  # local import — only needed when silero is loaded

            tensor = torch.from_numpy(_to_float32(audio_chunk))
            ts = self._get_speech_timestamps(tensor, self._model, sampling_rate=sample_rate)
            return len(ts) > 0
        except Exception as e:
            log.warning("silero VAD chunk inference failed (%s); using RMS fallback", e)
            return _rms_is_speech(audio_chunk)

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
