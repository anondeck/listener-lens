from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from earshift_bakeoff.config import ROOT, sha256_json
from earshift_bakeoff.kokoro_output_domain_splice import (
    ADJUDICATION_FILE,
    ANALYSIS_FILE,
    EXPECTED_FIXTURES,
    PROTOCOL_FILE,
    _read_pcm16,
    output_domain_splice,
    parent_dir,
    run_dir,
)
from earshift_bakeoff.util import sha256_file


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_result_is_bound_to_frozen_protocol_and_contains_one_candidate() -> None:
    protocol = _load(run_dir() / PROTOCOL_FILE)
    analysis = _load(run_dir() / ANALYSIS_FILE)
    unhashed = {key: value for key, value in analysis.items() if key != "analysis_sha256"}
    assert analysis["analysis_sha256"] == sha256_json(unhashed)
    assert analysis["protocol_sha256"] == protocol["protocol_sha256"]
    assert analysis["candidate_count"] == 1
    assert analysis["candidate_wav_count"] == 2
    assert analysis["api_calls"] == 0
    assert analysis["model_decodes"] == 0
    assert tuple(row["fixture_id"] for row in analysis["fixtures"]) == EXPECTED_FIXTURES


def test_candidate_wavs_recompute_byte_identically_from_bound_parents() -> None:
    protocol = _load(run_dir() / PROTOCOL_FILE)
    for fixture in protocol["fixtures"]:
        neutral, neutral_rate = _read_pcm16(
            parent_dir() / fixture["neutral"]["relative_path"]
        )
        lens, lens_rate = _read_pcm16(parent_dir() / fixture["lens"]["relative_path"])
        assert neutral_rate == lens_rate == 24_000
        expected, _ = output_domain_splice(
            neutral, lens, fixture["splice_windows"]
        )
        actual, actual_rate = _read_pcm16(
            run_dir() / "audio" / f"{fixture['fixture_id']}__candidate.wav"
        )
        assert actual_rate == 24_000
        assert np.array_equal(actual, expected)


def test_frozen_outcome_is_coverage_gated_path_not_splice_promotion() -> None:
    analysis = _load(run_dir() / ANALYSIS_FILE)
    assert analysis["classification"] == (
        "candidate_fails_runtime_gate_cheap_and_fail_closed"
    )
    assert analysis["recommendation"] == "coverage_gated_path_is_product_candidate"
    assert analysis["eligible_for_one_unseen_confirmation"] is False
    assert analysis["production_integration_authorized"] is False
    assert analysis["runtime_localization_gate"] == {
        "cheap_and_fail_closed": True,
        "pass": True,
    }


def test_adjudication_corrects_only_the_descriptive_gate_semantics() -> None:
    adjudication = _load(run_dir() / ADJUDICATION_FILE)
    unhashed = {
        key: value
        for key, value in adjudication.items()
        if key != "adjudication_sha256"
    }
    assert adjudication["adjudication_sha256"] == sha256_json(unhashed)
    assert adjudication["raw_classification_preserved"] == (
        "candidate_fails_runtime_gate_cheap_and_fail_closed"
    )
    assert adjudication["classification"] == (
        "candidate_succeeds_both_known_fixtures"
    )
    assert adjudication["recommendation"] == (
        "eligible_for_one_unseen_confirmation_no_product_integration"
    )
    assert adjudication["eligible_for_one_unseen_confirmation"] is True
    assert adjudication["production_integration_authorized"] is False
    assert all(
        row["adjudicated_candidate_pass"] for row in adjudication["fixtures"]
    )
    assert all(
        not row["acoustic_boolean_degradations"]
        for row in adjudication["fixtures"]
    )
    improvements = [
        path
        for row in adjudication["fixtures"]
        for path in row["acoustic_boolean_improvements"]
    ]
    assert improvements
    assert all("window=40" in path for path in improvements)


def test_both_candidates_fix_localization_and_pass_integrity_and_click_metrics() -> None:
    analysis = _load(run_dir() / ANALYSIS_FILE)
    rows = {row["fixture_id"]: row for row in analysis["fixtures"]}
    assert rows["new-repeated-phrase-final"]["untouched_baseline"][
        "inside_difference_energy_fraction"
    ] == 0.9532788320705685
    assert rows["independent-phrase-final-only"]["untouched_baseline"][
        "inside_difference_energy_fraction"
    ] == 0.7255241266369296
    for row in rows.values():
        assert row["localization"]["inside_difference_energy_fraction"] == 1.0
        assert row["localization"]["outside_rms_pcm"] == 0.0
        assert row["localization"]["pass"] is True
        assert row["integrity"]["pass"] is True
        assert row["boundary_artifact"]["pass"] is True
        assert row["boundary_artifact"]["maximum_edge_delta_step_pcm"] == 0.0
        assert row["localization_runtime_benchmark"]["pass"] is True


def test_only_failure_is_frozen_descriptive_signature_preservation() -> None:
    analysis = _load(run_dir() / ANALYSIS_FILE)
    first, second = analysis["fixtures"]
    assert first["acoustic"]["primary_gate_pass"] is True
    assert second["acoustic"]["primary_gate_pass"] is True
    assert second["acoustic"]["all_existing_gate_booleans_preserved"] is True
    assert second["candidate_pass"] is True
    assert first["acoustic"]["all_existing_gate_booleans_preserved"] is False
    baseline_family = first["acoustic"]["baseline_gate_signature"]["40"][
        "occurrences"
    ][0]["families"]["6000"]
    candidate_family = first["acoustic"]["candidate_gate_signature"]["40"][
        "occurrences"
    ][0]["families"]["6000"]
    assert baseline_family["checks"]["direction_cosine_at_least_0_50"] is False
    assert candidate_family["checks"]["direction_cosine_at_least_0_50"] is True
    assert first["candidate_pass"] is False


def test_candidate_hashes_match_recorded_files() -> None:
    analysis = _load(run_dir() / ANALYSIS_FILE)
    for row in analysis["fixtures"]:
        path = run_dir() / row["candidate"]["relative_path"]
        assert path.is_relative_to(ROOT)
        assert sha256_file(path) == row["candidate"]["wav_sha256"]
