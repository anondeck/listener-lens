from __future__ import annotations

from earshift_bakeoff.bilingual_vowel_acoustics import (
    classify_vowel_endpoint,
    trajectory_measurement,
)


def _frames(offset: float = 0.0):
    return [
        {
            "time_s": index * 0.005,
            "f1_hz": 500.0 + offset + index,
            "f2_hz": 1700.0 + offset + index * 2,
            "f3_hz": 2600.0,
        }
        for index in range(21)
    ]


def test_trajectory_measurement_retains_three_plausible_bins() -> None:
    measured = trajectory_measurement(_frames(), start_s=0.0, end_s=0.1)

    assert measured["measurable"] is True
    assert measured["retention_pass"] is True
    assert measured["plausibility_pass"] is True
    assert len(measured["feature_bark"]) == 6
    assert len(measured["rhoticity_gap_bark"]) == 3
    assert all(row["valid_frame_count"] >= 2 for row in measured["bins"])


def test_endpoint_classifier_distinguishes_exact_directional_and_fail() -> None:
    source = [5.0, 10.0, 5.1, 10.1, 5.2, 10.2]
    target = [6.0, 11.0, 6.1, 11.1, 6.2, 11.2]
    exact = classify_vowel_endpoint(
        source_anchor=source,
        target_anchor=target,
        neutral=source,
        lens=target,
    )
    directional = classify_vowel_endpoint(
        source_anchor=source,
        target_anchor=target,
        neutral=[4.8, 9.8, 4.9, 9.9, 5.0, 10.0],
        lens=[5.2, 10.2, 5.3, 10.3, 5.4, 10.4],
    )
    failed = classify_vowel_endpoint(
        source_anchor=source,
        target_anchor=target,
        neutral=source,
        lens=[4.0, 9.0, 4.1, 9.1, 4.2, 9.2],
    )

    assert exact["classification"] == "exact_category_pass"
    assert directional["classification"] == "directional_only_pass"
    assert failed["classification"] == "fail"


def test_endpoint_classifier_supports_rhoticity_gap_vectors() -> None:
    exact = classify_vowel_endpoint(
        source_anchor=[1.0, 1.1, 1.2],
        target_anchor=[1.8, 1.9, 2.0],
        neutral=[1.0, 1.1, 1.2],
        lens=[1.8, 1.9, 2.0],
        minimum_anchor_separation=0.08,
        minimum_controlled_movement=0.06,
    )

    assert exact["classification"] == "exact_category_pass"
