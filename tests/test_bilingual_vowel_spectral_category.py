from __future__ import annotations

import numpy as np
import pytest

from earshift_bakeoff.bilingual_vowel_spectral_category import (
    DEFAULT_FEATURE_CONFIG,
    aggregate_spectral_cell,
    apply_robust_feature_scaler,
    classify_spectral_endpoint,
    fit_robust_feature_scaler,
    spectral_trajectory_feature,
)


def _tone(frequency_hz: float, *, amplitude: float = 12_000.0) -> np.ndarray:
    count = 2_400
    time = np.arange(count, dtype=np.float64) / 24_000.0
    return np.rint(amplitude * np.sin(2 * np.pi * frequency_hz * time)).astype(np.int16)


def test_spectral_feature_is_deterministic_finite_and_amplitude_stable() -> None:
    first = spectral_trajectory_feature(
        _tone(300), start_sample=0, end_sample_exclusive=2_400
    )
    second = spectral_trajectory_feature(
        _tone(300), start_sample=0, end_sample_exclusive=2_400
    )
    quieter = spectral_trajectory_feature(
        _tone(300, amplitude=6_000), start_sample=0, end_sample_exclusive=2_400
    )

    assert first == second
    assert first["feature_size"] == (
        DEFAULT_FEATURE_CONFIG.cepstral_coefficient_count
        * len(DEFAULT_FEATURE_CONFIG.temporal_sample_fractions)
    )
    assert first["frame_count"] == 8
    assert np.isfinite(first["feature"]).all()
    assert np.allclose(first["feature"], quieter["feature"], atol=2e-3)


def test_spectral_feature_rejects_invalid_or_silent_intervals() -> None:
    samples = _tone(300)
    with pytest.raises(ValueError, match="outside"):
        spectral_trajectory_feature(
            samples, start_sample=-1, end_sample_exclusive=1_200
        )
    with pytest.raises(ValueError, match="shorter"):
        spectral_trajectory_feature(samples, start_sample=0, end_sample_exclusive=599)
    with pytest.raises(ValueError, match="silent"):
        spectral_trajectory_feature(
            np.zeros(1_200, dtype=np.int16),
            start_sample=0,
            end_sample_exclusive=1_200,
        )


def test_robust_scaler_is_finite_and_applies_its_floor() -> None:
    scaler = fit_robust_feature_scaler([[1.0, 2.0], [1.0, 4.0], [1.0, 6.0]])
    transformed = apply_robust_feature_scaler([1.05, 4.0], scaler)

    assert scaler["feature_size"] == 2
    assert scaler["observation_count"] == 3
    assert scaler["scale"][0] == DEFAULT_FEATURE_CONFIG.robust_scale_floor
    assert np.allclose(transformed, [1.0, 0.0])


def test_spectral_endpoint_distinguishes_exact_directional_and_wrong_way() -> None:
    source = [0.0, 0.0]
    target = [2.0, 0.0]

    exact = classify_spectral_endpoint(
        source_anchor=source,
        target_anchor=target,
        neutral=[0.0, 0.0],
        lens=[1.2, 0.0],
    )
    directional = classify_spectral_endpoint(
        source_anchor=source,
        target_anchor=target,
        neutral=[0.0, 0.0],
        lens=[0.6, 0.0],
    )
    wrong = classify_spectral_endpoint(
        source_anchor=source,
        target_anchor=target,
        neutral=[0.0, 0.0],
        lens=[-1.0, 0.0],
    )
    identity = classify_spectral_endpoint(
        source_anchor=source,
        target_anchor=target,
        neutral=[0.0, 0.0],
        lens=[0.0, 0.0],
    )

    assert exact["classification"] == "exact_category_pass"
    assert directional["classification"] == "directional_only_pass"
    assert wrong["classification"] == identity["classification"] == "fail"


def test_cell_aggregation_requires_anchor_validation_and_every_candidate() -> None:
    exact = classify_spectral_endpoint(
        source_anchor=[0.0, 0.0],
        target_anchor=[2.0, 0.0],
        neutral=[0.0, 0.0],
        lens=[1.2, 0.0],
    )
    directional = classify_spectral_endpoint(
        source_anchor=[0.0, 0.0],
        target_anchor=[2.0, 0.0],
        neutral=[0.0, 0.0],
        lens=[0.6, 0.0],
    )
    fail = classify_spectral_endpoint(
        source_anchor=[0.0, 0.0],
        target_anchor=[2.0, 0.0],
        neutral=[0.0, 0.0],
        lens=[0.1, 0.0],
    )

    passed = aggregate_spectral_cell(
        natural_anchor_records=[exact, exact, exact, fail],
        candidate_records=[directional] * 4,
    )
    incomplete = aggregate_spectral_cell(
        natural_anchor_records=[exact, exact, exact, fail],
        candidate_records=[directional, directional, directional, fail],
    )
    invalid_anchor = aggregate_spectral_cell(
        natural_anchor_records=[exact, exact, fail, fail],
        candidate_records=[exact] * 4,
    )

    assert passed["classification"] == "directional_only_pass"
    assert incomplete["classification"] == "fail"
    assert invalid_anchor["classification"] == "anchor_validation_fail"
