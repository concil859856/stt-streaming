"""Audio frame size + encoding validation tests.

The server accepts WS binary frames in the range 80..1600 samples (5..100 ms
at 16 kHz). PCM is 16-bit signed little-endian mono, so byte length == 2 *
sample count. These tests check the validator that the WS handler runs on
each incoming binary frame.
"""
from __future__ import annotations

import pytest

audio = pytest.importorskip("stt_streaming.audio")

MIN_SAMPLES = 80
MAX_SAMPLES = 1600
BYTES_PER_SAMPLE = 2  # pcm_s16le mono


def _frame_of_samples(n: int) -> bytes:
    return b"\x00\x00" * n


def _validate(frame: bytes):
    """Helper: validate a frame by its byte count."""
    return audio.validate_frame_bytes(len(frame))


def test_frame_size_lower_bound():
    assert _validate(_frame_of_samples(MIN_SAMPLES)) is True


def test_frame_size_upper_bound():
    assert _validate(_frame_of_samples(MAX_SAMPLES)) is True


def test_frame_too_large_rejected():
    with pytest.raises(audio.AudioFrameTooLarge):
        _validate(_frame_of_samples(MAX_SAMPLES + 1))


def test_frame_too_small_rejected():
    with pytest.raises(audio.AudioFrameTooSmall):
        _validate(_frame_of_samples(MIN_SAMPLES - 1))


def test_odd_byte_count_rejected():
    # PCM s16le frames must be an even number of bytes.
    with pytest.raises(audio.AudioFormatError):
        audio.validate_frame_bytes(2 * MIN_SAMPLES + 1)
