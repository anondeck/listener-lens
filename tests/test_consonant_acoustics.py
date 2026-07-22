from __future__ import annotations

import numpy as np
import pytest

from earshift_bakeoff.consonant_acoustics import (
    SampleInterval,
    consonant_acoustic_metrics,
    decoder_column_interval,
    descriptive_distance,
    expanded_interval,
)


def test_decoder_columns_map_to_exact_sample_grid() -> None:
    interval = decoder_column_interval(
        (1, 2, 3, 4),
        (1, 2),
        sample_count=6_000,
        sample_rate_hz=24_000,
    )

    assert interval.start_sample == 600
    assert interval.end_sample_exclusive == 3_600
    assert interval.start_s == pytest.approx(0.025)
    assert interval.end_s == pytest.approx(0.15)


def test_decoder_columns_reject_noncontiguous_or_fractional_layout() -> None:
    with pytest.raises(ValueError, match="contiguous"):
        decoder_column_interval(
            (1, 2, 3, 4),
            (1, 3),
            sample_count=6_000,
            sample_rate_hz=24_000,
        )
    with pytest.raises(ValueError, match="integral"):
        decoder_column_interval(
            (1, 2),
            (1,),
            sample_count=1_001,
            sample_rate_hz=24_000,
        )


def test_expanded_interval_is_bounded_by_pcm() -> None:
    interval = SampleInterval(100, 200, 1_000)

    actual = expanded_interval(interval, context_ms=150.0, sample_count=300)

    assert actual == SampleInterval(0, 300, 1_000)


def test_acoustic_metrics_separate_periodic_and_noisy_intervals() -> None:
    sample_rate = 24_000
    seconds = 0.1
    time = np.arange(round(sample_rate * seconds)) / sample_rate
    periodic = np.rint(np.sin(2 * np.pi * 200 * time) * 12_000).astype("<i2")
    rng = np.random.default_rng(7)
    noisy = np.rint(rng.normal(0, 4_000, periodic.size)).astype("<i2")
    interval = SampleInterval(0, periodic.size, sample_rate)

    periodic_metrics = consonant_acoustic_metrics(periodic, interval)
    noisy_metrics = consonant_acoustic_metrics(noisy, interval)

    assert periodic_metrics["periodicity"] > noisy_metrics["periodicity"]
    assert noisy_metrics["spectral_flatness"] > periodic_metrics["spectral_flatness"]
    assert descriptive_distance(periodic_metrics, noisy_metrics) > 0.1


def test_descriptive_distance_is_zero_for_identical_measurements() -> None:
    values = np.arange(-500, 500, dtype="<i2")
    interval = SampleInterval(0, values.size, 24_000)
    metrics = consonant_acoustic_metrics(values, interval)

    assert descriptive_distance(metrics, metrics) == pytest.approx(0.0)
