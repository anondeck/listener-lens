from __future__ import annotations

import numpy as np

from earshift_bakeoff.same_take_corrective import (
    design_complementary_lowpass,
    low_band_log_spectral_distance_db,
    solve_rms_gain,
)


def test_frozen_filter_meets_response_gate_and_is_symmetric() -> None:
    coefficients, record = design_complementary_lowpass(24_000)
    assert record["passed"] is True
    assert np.allclose(coefficients, coefficients[::-1])
    assert abs(float(np.sum(coefficients)) - 1.0) < 1e-12


def test_rms_gain_solver_matches_reference_after_taper() -> None:
    original = np.linspace(-2000, 2000, 200, dtype=np.float64)
    component = original * 1.7
    residual = np.zeros_like(original)
    taper = np.ones_like(original)
    gain = solve_rms_gain(original, component, residual, taper)
    assert gain is not None
    result = original + taper * (gain * component + residual - original)
    assert abs(np.sqrt(np.mean(result**2)) / np.sqrt(np.mean(original**2)) - 1) < 1e-12


def test_identity_log_spectral_distance_is_zero() -> None:
    time = np.arange(1_920) / 24_000
    samples = np.sin(2 * np.pi * 1_200 * time)
    assert low_band_log_spectral_distance_db(samples, samples, 24_000) < 1e-12
