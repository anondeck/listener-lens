from __future__ import annotations

import numpy as np

from earshift_bakeoff.kokoro_output_domain_splice import (
    EXPECTED_FIXTURES,
    TAPER_SAMPLES,
    boundary_artifact_report,
    outcome_for,
    output_domain_splice,
    protocol_record,
    raised_cosine_weights,
)
from earshift_bakeoff.kokoro_typed_diagnostic import localization_report


def _window(start: int, end: int) -> dict[str, int]:
    return {"start_sample": start, "end_sample_exclusive": end}


def test_frozen_protocol_binds_exactly_one_candidate_and_two_parent_fixtures() -> None:
    protocol = protocol_record()
    assert protocol["status"] == "frozen_before_candidate_artifacts"
    assert protocol["intervention_name"] == "output-domain splice"
    assert protocol["scope"]["candidate_count"] == 1
    assert protocol["scope"]["candidate_wav_count"] == 2
    assert protocol["scope"]["api_calls"] == 0
    assert tuple(row["fixture_id"] for row in protocol["fixtures"]) == EXPECTED_FIXTURES
    assert [
        row["baseline_localization"]["inside_difference_energy_fraction"]
        for row in protocol["fixtures"]
    ] == [0.9532788320705685, 0.7255241266369296]
    assert protocol["candidate"]["taper"]["samples_each_edge"] == TAPER_SAMPLES


def test_output_domain_splice_is_deterministic_and_local() -> None:
    size = 4 * TAPER_SAMPLES
    neutral = np.arange(size, dtype=np.int16)
    lens = neutral + 1000
    windows = [_window(TAPER_SAMPLES // 2, size - TAPER_SAMPLES // 2)]
    first, weights = output_domain_splice(neutral, lens, windows)
    second, repeated_weights = output_domain_splice(neutral, lens, windows)
    assert np.array_equal(first, second)
    assert np.array_equal(weights, repeated_weights)
    assert weights[windows[0]["start_sample"]] == 0.0
    assert weights[windows[0]["end_sample_exclusive"] - 1] == 0.0
    outside = weights == 0.0
    interior = weights == 1.0
    assert np.array_equal(first[outside], neutral[outside])
    assert interior.any()
    assert np.array_equal(first[interior], lens[interior])


def test_raised_cosine_rejects_parameter_search() -> None:
    windows = [_window(0, 4 * TAPER_SAMPLES)]
    try:
        raised_cosine_weights(4 * TAPER_SAMPLES, windows, taper_samples=20)
    except ValueError as exc:
        assert "frozen 10 ms taper" in str(exc)
    else:  # pragma: no cover - assertion clarity
        raise AssertionError("non-frozen taper unexpectedly accepted")


def test_boundary_metric_reports_zero_outer_delta_step() -> None:
    size = 5 * TAPER_SAMPLES
    x = np.arange(size, dtype=np.float64)
    neutral = np.rint(2000 * np.sin(x / 11.0)).astype(np.int16)
    lens = np.rint(2200 * np.sin(x / 11.0 + 0.03)).astype(np.int16)
    windows = [_window(TAPER_SAMPLES, 4 * TAPER_SAMPLES)]
    candidate, _ = output_domain_splice(neutral, lens, windows)
    report = boundary_artifact_report(neutral, lens, candidate, windows)
    assert report["maximum_edge_delta_step_pcm"] == 0.0
    assert len(report["boundaries"]) == 2


def test_localization_gate_fails_closed_for_identical_and_mismatched_pcm() -> None:
    neutral = np.zeros(1000, dtype=np.int16)
    lens = neutral.copy()
    interval = [{"start_sample": 200, "end_sample_exclusive": 400}]
    assert localization_report(neutral, lens, interval)["pass"] is False
    assert localization_report(neutral, lens[:-1], interval)["pass"] is False


def test_outcome_table_is_exhaustive_and_keeps_success_nonproduction() -> None:
    assert outcome_for([True, True], True) == (
        "candidate_succeeds_both_known_fixtures",
        "eligible_for_one_unseen_confirmation_no_product_integration",
    )
    assert outcome_for([True, False], True)[1] == (
        "coverage_gated_path_is_product_candidate"
    )
    assert outcome_for([False, False], False)[1] == (
        "close_kokoro_product_remediation_for_build_week"
    )
