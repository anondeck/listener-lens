from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

from earshift_bakeoff.pcm import decode_pcm16_mono


def test_pcm_metadata_uses_decoded_samples(tmp_path: Path) -> None:
    path = tmp_path / "short.wav"
    samples = np.array([0, 100, -100, 32767, -32768], dtype="<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(10)
        handle.writeframes(samples.tobytes())
    decoded = decode_pcm16_mono(path)
    assert decoded.decoded_sample_count == 5
    assert decoded.duration_s == 0.5
    assert decoded.clipped_sample_count == 2
    assert decoded.samples.tolist() == samples.tolist()
