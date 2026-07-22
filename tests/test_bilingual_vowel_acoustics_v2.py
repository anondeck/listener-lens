from __future__ import annotations

from earshift_bakeoff.bilingual_vowel_acoustics_v2 import (
    classify_stress_core_endpoint,
    measurement_mode,
    stress_core_measurement,
)


def _frames():
    return [
        {
            "time_s": index * 0.005,
            "f1_hz": 500.0 + index,
            "f2_hz": 1700.0 + index * 2,
            "f3_hz": 2600.0 + index,
        }
        for index in range(41)
    ]


def test_measurement_mode_separates_core_points_and_diphthong_trajectories() -> None:
    assert measurement_mode("æ", "ɛ") == "monophthong_core"
    assert measurement_mode("o", "O") == "diphthong_core_trajectory"
    assert measurement_mode("õ", "Õ") == "diphthong_core_trajectory"


def test_stress_core_measurement_retains_monophthong_and_diphthong_features() -> None:
    mono = stress_core_measurement(
        _frames(), start_s=0.0, end_s=0.2, mode="monophthong_core"
    )
    diphthong = stress_core_measurement(
        _frames(), start_s=0.0, end_s=0.2, mode="diphthong_core_trajectory"
    )

    assert mono["measurable"] is True
    assert len(mono["feature_bark"]) == 2
    assert len(mono["rhoticity_gap_bark"]) == 1
    assert diphthong["measurable"] is True
    assert len(diphthong["feature_bark"]) == 4
    assert len(diphthong["rhoticity_gap_bark"]) == 2


def test_v2_endpoint_classifier_separates_directional_and_exact_movement() -> None:
    source = [5.0, 10.0]
    target = [7.0, 12.0]
    too_small = classify_stress_core_endpoint(
        source_anchor=source,
        target_anchor=target,
        neutral=source,
        lens=[5.6, 10.6],
    )
    exact = classify_stress_core_endpoint(
        source_anchor=source,
        target_anchor=target,
        neutral=source,
        lens=target,
    )

    assert too_small["classification"] == "directional_only_pass"
    assert too_small["movement_gate_pass"] is True
    assert too_small["exact_movement_gate_pass"] is False
    assert exact["classification"] == "exact_category_pass"
