from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from earshift_bakeoff.kokoro_typed_diagnostic import (
    _repeated_alignment,
    anchor_precedence,
    candidate_route_decision,
    localization_report,
    rescore_attribution,
    summarize_frame_table,
)
from earshift_bakeoff.kokoro_typed_diagnostic_protocol import (
    REPEATED_MEASUREMENT_COLUMNS,
    REPEATED_NEUTRAL_PHONEMES,
    REPEATED_WORD_COLUMNS,
)


def test_window_summaries_use_centered_regions_of_one_frame_table() -> None:
    rows = [
        {"time_s": index / 100, "f1_hz": 700.0 + index, "f2_hz": 1800.0}
        for index in range(101)
    ]
    interval = {"start_s": 0.0, "end_s": 1.0}
    forty = summarize_frame_table(rows, interval, 5500, 40)
    fifty = summarize_frame_table(rows, interval, 5500, 50)
    sixty = summarize_frame_table(rows, interval, 5500, 60)
    assert (forty["window_start_s"], forty["window_end_s"]) == (0.3, 0.7)
    assert (fifty["window_start_s"], fifty["window_end_s"]) == (0.25, 0.75)
    assert (sixty["window_start_s"], sixty["window_end_s"]) == (0.2, 0.8)
    assert forty["middle_frame_count"] < fifty["middle_frame_count"]
    assert fifty["middle_frame_count"] < sixty["middle_frame_count"]
    assert all(row["valid_f1_f2_fraction"] == 1.0 for row in (forty, fifty, sixty))


def test_zero_delta_localization_is_invalid_not_a_perfect_pass() -> None:
    zero = np.zeros(100, dtype=np.float64)
    result = localization_report(
        zero,
        zero,
        [{"start_sample": 40, "end_sample_exclusive": 60}],
        sample_rate_hz=100,
    )
    assert result["sample_count_equal"] is True
    assert result["total_difference_energy_positive"] is False
    assert result["inside_difference_energy_fraction"] == 0.0
    assert result["pass"] is False


def test_localization_still_uses_squared_difference_energy() -> None:
    neutral = np.zeros(100, dtype=np.float64)
    lens = np.zeros(100, dtype=np.float64)
    lens[40:60] = 2.0
    result = localization_report(
        neutral,
        lens,
        [{"start_sample": 40, "end_sample_exclusive": 60}],
        sample_rate_hz=100,
    )
    assert result["inside_difference_energy_fraction"] == 1.0
    assert result["pass"] is True


def test_anchor_precedence_is_exact_and_fail_closed() -> None:
    assert (
        anchor_precedence(
            measurement_valid=False, medial_valid=True, phrase_final_valid=True
        )
        == "anchor_measurement_inconclusive"
    )
    assert (
        anchor_precedence(
            measurement_valid=True, medial_valid=True, phrase_final_valid=True
        )
        == "rescore"
    )
    assert (
        anchor_precedence(
            measurement_valid=True, medial_valid=True, phrase_final_valid=False
        )
        == "phrase_final_reference_not_realized"
    )
    assert (
        anchor_precedence(
            measurement_valid=True, medial_valid=False, phrase_final_valid=True
        )
        == "unexpected_medial_reference_failure"
    )
    assert (
        anchor_precedence(
            measurement_valid=True, medial_valid=False, phrase_final_valid=False
        )
        == "reference_geometry_invalid"
    )


def _cells(*, transported: bool, endpoint: bool, threshold: bool, local: bool):
    return {
        "transported_endpoints__transported_threshold": {"pass": transported},
        "local_endpoints__transported_threshold": {"pass": endpoint},
        "transported_endpoints__local_threshold": {"pass": threshold},
        "local_endpoints__local_threshold": {"pass": local},
    }


def test_rescore_attribution_uses_mechanical_claim_vocabulary() -> None:
    endpoint = rescore_attribution(
        _cells(transported=False, endpoint=True, threshold=False, local=True)
    )
    assert endpoint == {
        "calibration_claim": "transported_calibration_mechanically_sufficient_for_this_fixture",
        "mechanical_attribution": "transported_endpoint_geometry_implicated",
    }
    threshold = rescore_attribution(
        _cells(transported=False, endpoint=False, threshold=True, local=True)
    )
    assert threshold["mechanical_attribution"] == (
        "transported_magnitude_threshold_implicated"
    )
    mixed = rescore_attribution(
        _cells(transported=False, endpoint=False, threshold=False, local=True)
    )
    assert mixed["mechanical_attribution"] == (
        "mixed_transported_endpoint_and_threshold_calibration"
    )
    both = rescore_attribution(
        _cells(transported=False, endpoint=True, threshold=True, local=True)
    )
    assert both["mechanical_attribution"] == [
        "transported_endpoint_geometry_implicated",
        "transported_magnitude_threshold_implicated",
    ]


def test_candidate_route_advances_only_valid_lens_repairable_failure() -> None:
    base = {
        "complete_pass": False,
        "measurement_valid": True,
        "neutral_source_pass": True,
        "lens_repairable_failure": True,
        "output_gate": {
            "checks": {
                "runtime_integrity": True,
                "exact_state_contract": True,
                "sample_count_equal": True,
                "localization_at_least_0_80": True,
            }
        },
    }
    assert candidate_route_decision(base, has_later_span=True) == (
        "advance_to_next_span"
    )
    assert candidate_route_decision(base, has_later_span=False) == (
        "bounded_controlled_span_route_failed"
    )
    assert (
        candidate_route_decision(
            {**base, "measurement_valid": False}, has_later_span=True
        )
        == "diagnostic_inconclusive_measurement_or_instrument_failure"
    )
    assert (
        candidate_route_decision(
            {**base, "neutral_source_pass": False}, has_later_span=True
        )
        == "diagnostic_stopped_neutral_source_reference_failure"
    )
    assert (
        candidate_route_decision({**base, "complete_pass": True}, has_later_span=True)
        == "select_for_unseen_confirmation"
    )


def test_candidate_route_splits_integrity_and_localization_stops() -> None:
    base = {
        "complete_pass": False,
        "measurement_valid": True,
        "neutral_source_pass": True,
        "lens_repairable_failure": False,
        "output_gate": {
            "checks": {
                "runtime_integrity": True,
                "exact_state_contract": True,
                "sample_count_equal": True,
                "localization_at_least_0_80": True,
            }
        },
    }
    integrity = {
        **base,
        "output_gate": {
            "checks": {
                **base["output_gate"]["checks"],
                "exact_state_contract": False,
            }
        },
    }
    assert candidate_route_decision(integrity, has_later_span=True) == (
        "diagnostic_inconclusive_runtime_or_integrity_failure"
    )
    localization = {
        **base,
        "output_gate": {
            "checks": {
                **base["output_gate"]["checks"],
                "localization_at_least_0_80": False,
            }
        },
    }
    assert candidate_route_decision(localization, has_later_span=True) == (
        "candidate_localization_gate_failed"
    )


def test_anchor_alignment_uses_each_sides_own_duration_map() -> None:
    model = SimpleNamespace(
        vocab={
            symbol: index + 1
            for index, symbol in enumerate(set(REPEATED_NEUTRAL_PHONEMES))
        }
    )
    duration_count = len(REPEATED_NEUTRAL_PHONEMES) + 2
    durations = tuple(range(1, duration_count + 1))
    total_frames = sum(durations)
    result = _repeated_alignment(
        model=model,
        phonemes=REPEATED_NEUTRAL_PHONEMES,
        target_symbol="æ",
        durations=durations,
        sample_count=total_frames * 600,
    )
    assert result["own_predicted_durations"] is True
    assert result["own_alignment"] is True
    assert [
        tuple(row["measurement_interval"]["columns"])
        for row in result["target_occurrences"]
    ] == list(REPEATED_MEASUREMENT_COLUMNS)
    assert [
        tuple(row["target_word_interval"]["columns"])
        for row in result["target_occurrences"]
    ] == list(REPEATED_WORD_COLUMNS)
