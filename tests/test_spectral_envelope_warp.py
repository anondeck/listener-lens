from __future__ import annotations

import numpy as np
import pytest

from earshift_bakeoff.spectral_envelope_warp import (
    FormantWarpSpec,
    spectral_envelope_warp,
)


def _signal() -> np.ndarray:
    sample_rate = 24_000
    time = np.arange(sample_rate, dtype=np.float64) / sample_rate
    values = sum(
        np.sin(2.0 * np.pi * frequency * time) / index
        for index, frequency in enumerate(range(120, 3_000, 120), start=1)
    )
    values *= 0.35 / np.max(np.abs(values))
    return np.rint(values * 32767.0).astype(np.int16)


def _spec() -> FormantWarpSpec:
    return FormantWarpSpec(
        start_sample=8_000,
        end_sample_exclusive=12_000,
        source_f1_hz=600.0,
        source_f2_hz=1_700.0,
        target_f1_hz=720.0,
        target_f2_hz=1_850.0,
    )


def test_zero_strength_is_bit_exact_identity() -> None:
    source = _signal()
    result = spectral_envelope_warp(
        source, (_spec(),), sample_rate_hz=24_000, strength=0.0
    )
    assert np.array_equal(result.pcm, source)
    assert result.metrics["identity"] is True
    assert not np.any(result.weights)


def test_shift_is_local_finite_equal_length_and_unclipped() -> None:
    source = _signal()
    result = spectral_envelope_warp(
        source, (_spec(),), sample_rate_hz=24_000, strength=1.0
    )
    assert result.pcm.shape == source.shape
    assert result.metrics["finite"] is True
    assert result.metrics["clipped_sample_count"] == 0
    assert result.metrics["outside_windows_bit_exact"] is True
    assert np.array_equal(
        result.pcm[result.weights == 0.0], source[result.weights == 0.0]
    )
    assert np.any(result.pcm[result.weights > 0.0] != source[result.weights > 0.0])
    assert max(abs(value) for value in result.metrics["rms_db_change_by_window"]) < 0.5


def test_rejects_overlapping_windows_and_nonmonotonic_targets() -> None:
    source = _signal()
    overlap = FormantWarpSpec(
        start_sample=11_000,
        end_sample_exclusive=13_000,
        source_f1_hz=600.0,
        source_f2_hz=1_700.0,
        target_f1_hz=720.0,
        target_f2_hz=1_850.0,
    )
    with pytest.raises(ValueError, match="overlap"):
        spectral_envelope_warp(
            source, (_spec(), overlap), sample_rate_hz=24_000, strength=1.0
        )
    invalid = FormantWarpSpec(
        start_sample=8_000,
        end_sample_exclusive=12_000,
        source_f1_hz=600.0,
        source_f2_hz=1_700.0,
        target_f1_hz=1_900.0,
        target_f2_hz=1_800.0,
    )
    with pytest.raises(ValueError, match="invalid"):
        spectral_envelope_warp(source, (invalid,), sample_rate_hz=24_000, strength=1.0)
