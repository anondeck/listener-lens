from __future__ import annotations

import math
import struct
import wave
from pathlib import Path

import pytest

from earshift_bakeoff.runtime_audio import analyze_prosody_fingerprint


def write_contoured_tone(path: Path, *, rising: bool) -> None:
    sample_rate = 24_000
    duration_s = 1.2
    phase = 0.0
    frames: list[bytes] = []
    for index in range(round(sample_rate * duration_s)):
        fraction = index / (sample_rate * duration_s)
        f0 = 170 + (80 * fraction if rising else 80 * (1 - fraction))
        amplitude = 3_500 + 7_000 * math.sin(math.pi * fraction) ** 2
        phase += 2 * math.pi * f0 / sample_rate
        frames.append(struct.pack("<h", round(amplitude * math.sin(phase))))
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(b"".join(frames))


def test_prosody_fingerprint_is_fixed_size_and_tracks_relative_contour(
    tmp_path: Path,
) -> None:
    rising_path = tmp_path / "rising.wav"
    falling_path = tmp_path / "falling.wav"
    write_contoured_tone(rising_path, rising=True)
    write_contoured_tone(falling_path, rising=False)

    rising = analyze_prosody_fingerprint(rising_path)
    falling = analyze_prosody_fingerprint(falling_path)

    assert rising.version == falling.version == "prosody-fingerprint-v1"
    assert rising.bin_count == falling.bin_count == 32
    assert len(rising.energy_contour_db) == len(rising.pitch_contour_semitones) == 32
    assert rising.pitch_contour_semitones[4] < rising.pitch_contour_semitones[-5]
    assert falling.pitch_contour_semitones[4] > falling.pitch_contour_semitones[-5]
    assert rising.median_f0_hz == pytest.approx(falling.median_f0_hz, rel=0.08)
    assert rising.voiced_fraction > 0.8
    assert rising.energy_span_db > 3
