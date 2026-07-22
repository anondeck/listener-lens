from __future__ import annotations

import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .util import sha256_file


@dataclass(frozen=True)
class DecodedPcm16Wav:
    path: Path
    samples: np.ndarray
    channels: int
    sample_width_bytes: int
    sample_rate_hz: int
    decoded_sample_count: int
    duration_s: float
    clipped_sample_count: int
    sha256: str

    def metadata(self) -> dict[str, int | float | str]:
        return {
            "path": str(self.path),
            "sha256": self.sha256,
            "channels": self.channels,
            "sample_width_bytes": self.sample_width_bytes,
            "sample_rate_hz": self.sample_rate_hz,
            "decoded_sample_count": self.decoded_sample_count,
            "duration_s": self.duration_s,
            "clipped_sample_count": self.clipped_sample_count,
        }


def decode_pcm16_mono(path: Path) -> DecodedPcm16Wav:
    """Decode canonical research WAVs and derive metadata from decoded PCM."""
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        sample_width = handle.getsampwidth()
        sample_rate = handle.getframerate()
        compression = handle.getcomptype()
        advertised_frames = handle.getnframes()
        raw = handle.readframes(advertised_frames)
    if channels != 1 or sample_width != 2 or sample_rate <= 0 or compression != "NONE":
        raise ValueError("Expected uncompressed mono 16-bit PCM WAV audio")
    samples = np.frombuffer(raw, dtype="<i2").copy()
    decoded_count = int(samples.size)
    widened = samples.astype(np.int32, copy=False)
    clipped = int(np.count_nonzero(np.abs(widened) >= 32767))
    return DecodedPcm16Wav(
        path=path,
        samples=samples,
        channels=channels,
        sample_width_bytes=sample_width,
        sample_rate_hz=sample_rate,
        decoded_sample_count=decoded_count,
        duration_s=decoded_count / sample_rate,
        clipped_sample_count=clipped,
        sha256=sha256_file(path),
    )
