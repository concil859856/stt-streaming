"""Regression tests for the streaming re-decode buffering.

The model must NOT keep its own copy of the audio that the ws session already
buffers. A previous bug appended the caller's full cumulative buffer to per-call
state on every partial, so the Nth partial re-decoded ~N times the audio
(quadratic growth + garbled transcripts). These tests pin the stateless
re-decode contract without importing torch/nemo.
"""
from __future__ import annotations

import threading

import numpy as np
import pytest

model_mod = pytest.importorskip("stt_streaming.model")
ParakeetModel = model_mod.ParakeetModel
SAMPLE_RATE = model_mod.SAMPLE_RATE


def _fake_model() -> ParakeetModel:
    """A ParakeetModel without the (torch/nemo) __init__, with _run_transcribe
    replaced by one that reports how many samples it was actually handed."""
    m = ParakeetModel.__new__(ParakeetModel)
    m._lock = threading.Lock()
    m.seen_lengths = []

    def _run(audio):
        m.seen_lengths.append(int(audio.shape[0]))
        return f"{audio.shape[0]}"

    m._run_transcribe = _run  # type: ignore[attr-defined]
    return m


def test_transcribe_chunk_does_not_accumulate():
    m = _fake_model()
    state = m.reset_state()

    # Simulate the ws session: it owns a growing utterance buffer and passes the
    # *whole* buffer to the model on each partial tick.
    buf = np.zeros(0, dtype=np.float32)
    frame = np.ones(320, dtype=np.float32)  # 20 ms
    for _ in range(5):
        buf = np.concatenate([buf, frame])
        text, state = m.transcribe_chunk(buf, state)
        assert text == str(buf.shape[0])

    # The model must have decoded exactly the buffer length each time — never a
    # multiple of it. After 5 frames the buffer is 1600 samples.
    assert m.seen_lengths == [320, 640, 960, 1280, 1600]


def test_transcribe_window_bounds_long_audio():
    m = _fake_model()
    huge = np.ones(SAMPLE_RATE * 20, dtype=np.float32)  # 20 s
    text = m.transcribe_window(huge)
    # Trimmed to the 8 s fallback window.
    assert int(text) == model_mod._FALLBACK_CONTEXT_SAMPLES == SAMPLE_RATE * 8


def test_transcribe_window_empty_is_noop():
    m = _fake_model()
    assert m.transcribe_window(np.zeros(0, dtype=np.float32)) == ""
    assert m.seen_lengths == []  # never touched the GPU path
