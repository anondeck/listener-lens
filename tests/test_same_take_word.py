from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

from earshift_bakeoff.pcm import decode_pcm16_mono
from earshift_bakeoff.same_take_word import (
    ACTIVE_RELATIVE_DB,
    ANCHOR_FAMILY_COSINE_MIN,
    SHIFT_FAMILY_COSINE_MIN,
    WORD_SOURCES,
    _raised_cosine_splice,
    decoded_active_bounds,
)


def test_word_sources_and_cross_family_thresholds_are_frozen() -> None:
    assert [source.token for source in WORD_SOURCES] == ["vap", "vihp"]
    assert ACTIVE_RELATIVE_DB == -35.0
    assert ANCHOR_FAMILY_COSINE_MIN == 0.75
    assert SHIFT_FAMILY_COSINE_MIN == 0.75


def test_decoded_active_bounds_are_sample_indexed(tmp_path: Path) -> None:
    rate = 1_000
    samples = np.zeros(rate, dtype="<i2")
    samples[200:600] = 10_000
    path = tmp_path / "active.wav"
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(samples.tobytes())
    bounds = decoded_active_bounds(decode_pcm16_mono(path))
    assert bounds["start_sample"] <= 200
    assert bounds["end_sample_exclusive"] >= 600
    assert bounds["sample_count"] == (
        bounds["end_sample_exclusive"] - bounds["start_sample"]
    )


def test_splice_taper_is_symmetric_and_has_identity_boundaries() -> None:
    weights = _raised_cosine_splice(100, 20)
    assert weights[0] == 0
    assert weights[-1] == 0
    assert np.allclose(weights, weights[::-1])
    assert np.all(weights[20:80] == 1)
