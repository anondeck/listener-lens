from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from earshift_bakeoff.prosody_acoustics import (
    interval_summary,
    measure_question_component,
    measure_stress_component,
    measure_stress_unit_component,
)


def _frames(
    duration_s: float, pitch_for_time: object, rms_for_time: object
) -> list[dict[str, float]]:
    return [
        {
            "time_s": time,
            "pitch_hz": float(pitch_for_time(time)),
            "rms": float(rms_for_time(time)),
        }
        for time in np.arange(0.0025, duration_s, 0.005)
    ]


def test_interval_summary_trims_edges_and_reports_voicing() -> None:
    frames = _frames(0.1, lambda _: 200.0, lambda _: 0.5)

    result = interval_summary(frames, 0.0, 0.1, retain_fraction=0.8)

    assert result["measurement_frame_count"] == 16
    assert result["voiced_frame_fraction"] == 1.0
    assert result["median_pitch_hz"] == 200.0
    assert result["median_rms"] == 0.5


def test_stress_measurement_requires_duration_and_rms_directions() -> None:
    render = SimpleNamespace(
        stress_interventions=(
            {
                "promoted_vowel_column": 1,
                "demoted_vowel_column": 2,
            },
        ),
        neutral_durations=(0, 2, 2, 2),
        lens_durations=(0, 3, 1, 2),
        neutral_pcm=np.zeros(3600, dtype="<i2"),
        lens_pcm=np.zeros(3600, dtype="<i2"),
    )
    neutral = _frames(0.15, lambda _: 200.0, lambda _: 1.0)
    lens = _frames(
        0.15,
        lambda _: 200.0,
        lambda time: 1.3 if time < 0.075 else 0.7,
    )

    result = measure_stress_component(
        render,
        neutral,
        lens,
        minimum_frames=3,
        minimum_duration_delta_ms=20.0,
        minimum_promoted_rms_ratio=1.1,
        maximum_demoted_rms_ratio=0.9,
    )

    assert result["gate_pass"] is True
    assert result["occurrences"][0]["roles"]["promoted"]["duration_delta_ms"] == 25.0
    assert result["occurrences"][0]["roles"]["demoted"][
        "duration_delta_ms"
    ] == pytest.approx(-25.0)


def test_question_measurement_distinguishes_rise_fall_from_statement_fall() -> None:
    render = SimpleNamespace(
        target_intervals=({"start_s": 0.0, "end_s": 0.3},),
    )
    neutral = _frames(
        0.3,
        lambda time: 100.0 if time < 0.1 else (120.0 if time < 0.2 else 90.0),
        lambda _: 1.0,
    )
    lens = _frames(
        0.3,
        lambda time: 100.0 if time < 0.1 else (98.0 if time < 0.2 else 80.0),
        lambda _: 1.0,
    )

    result = measure_question_component(
        render,
        neutral,
        lens,
        minimum_frames=5,
        minimum_voiced_fraction=0.5,
        minimum_neutral_rise_ratio=1.05,
        maximum_neutral_end_to_peak_ratio=0.9,
        maximum_lens_end_to_start_ratio=0.9,
        maximum_lens_middle_to_start_ratio=1.05,
    )

    assert result["gate_pass"] is True
    assert all(result["checks"].values())


def test_stress_unit_measurement_accepts_marker_frame_donation() -> None:
    render = SimpleNamespace(
        stress_interventions=(
            {
                "promoted_marker_column": 1,
                "promoted_vowel_column": 2,
                "demoted_marker_column": 4,
                "demoted_vowel_column": 5,
                "duration_donor_kind": "demoted_stress_marker",
            },
        ),
        neutral_durations=(1, 1, 1, 1, 2, 1, 2),
        lens_durations=(1, 1, 2, 1, 1, 1, 2),
        neutral_pcm=np.zeros(5400, dtype="<i2"),
        lens_pcm=np.zeros(5400, dtype="<i2"),
    )
    neutral = _frames(0.225, lambda _: 200.0, lambda _: 1.0)
    lens = _frames(0.225, lambda _: 200.0, lambda _: 1.0)

    result = measure_stress_unit_component(
        render,
        neutral,
        lens,
        minimum_frames=3,
        minimum_duration_delta_ms=20.0,
        minimum_promoted_rms_ratio=0.9,
        maximum_demoted_rms_ratio=1.1,
    )

    assert result["gate_pass"] is True
    assert result["occurrences"][0]["roles"]["promoted"][
        "stress_unit_duration_delta_ms"
    ] == pytest.approx(25.0)
    assert result["occurrences"][0]["roles"]["demoted"][
        "stress_unit_duration_delta_ms"
    ] == pytest.approx(-25.0)
