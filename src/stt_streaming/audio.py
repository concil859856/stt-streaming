from __future__ import annotations

MIN_SAMPLES = 80
MAX_SAMPLES = 1600
BYTES_PER_SAMPLE = 2  # pcm_s16le mono


class AudioFormatError(ValueError):
    pass


class AudioFrameTooLarge(AudioFormatError):
    pass


class AudioFrameTooSmall(AudioFormatError):
    pass


def validate_frame_bytes(n: int) -> bool:
    if n % BYTES_PER_SAMPLE != 0:
        raise AudioFormatError(f"frame must be a multiple of {BYTES_PER_SAMPLE} bytes (pcm_s16le)")
    samples = n // BYTES_PER_SAMPLE
    if samples < MIN_SAMPLES:
        raise AudioFrameTooSmall(f"frame {samples} samples < {MIN_SAMPLES} min")
    if samples > MAX_SAMPLES:
        raise AudioFrameTooLarge(f"frame {samples} samples > {MAX_SAMPLES} max")
    return True
