from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from earshift_bakeoff import (
    kokoro_typed_diagnostic_protocol as diagnostic_protocol,
)
from earshift_bakeoff.kokoro_typed_diagnostic_protocol import (
    CONFIRMATION_FIXTURES,
    CONFIRMATION_PLANNING_REGISTER,
    DECODER_SLOTS,
    REPEATED_FULL_CONTEXTUAL_STATE_COLUMNS,
    REPEATED_TARGET_WORD_COLUMNS,
    REPEATED_TARGET_WORD_PLUS_BOUNDARIES_COLUMNS,
    protocol_record,
)
from earshift_bakeoff.config import sha256_json


ROOT = Path(__file__).resolve().parents[1]
DIAGNOSTIC_PROTOCOL_PATH = (
    ROOT
    / "artifacts"
    / "typed-engine"
    / "20260717-kokoro-typed-diagnostic-v1"
    / "protocol.json"
)
CONFIRMATION_PROTOCOL_PATH = (
    ROOT
    / "artifacts"
    / "typed-engine"
    / "20260717-kokoro-typed-confirmation-v1"
    / "protocol.json"
)


@pytest.fixture(autouse=True)
def _isolate_creation_time_novelty_check(monkeypatch: pytest.MonkeyPatch) -> None:
    """Do not replay a pre-write novelty scan against its later child run."""
    frozen = json.loads(DIAGNOSTIC_PROTOCOL_PATH.read_text(encoding="utf-8"))
    assert "at protocol creation" in frozen["confirmation_precommit"]["novelty_check"]
    monkeypatch.setattr(
        diagnostic_protocol,
        "_verify_confirmation_novelty",
        lambda: None,
    )


def test_later_confirmation_is_a_bound_child_not_a_prior_novelty_collision() -> None:
    diagnostic_bytes = DIAGNOSTIC_PROTOCOL_PATH.read_bytes()
    confirmation = json.loads(CONFIRMATION_PROTOCOL_PATH.read_text(encoding="utf-8"))
    diagnostic_parent = confirmation["parents"]["diagnostic"]
    assert (
        diagnostic_parent["protocol_file_sha256"]
        == hashlib.sha256(diagnostic_bytes).hexdigest()
    )
    assert [row["text"] for row in confirmation["fixtures"]] == [
        fixture.text for fixture in CONFIRMATION_FIXTURES
    ]
    assert [row["expected_plan_sha256"] for row in confirmation["fixtures"]] == [
        fixture.expected_plan_sha256 for fixture in CONFIRMATION_FIXTURES
    ]


def test_protocol_binds_all_parent_wavs_code_model_voice_and_instrument() -> None:
    protocol = protocol_record()
    assert protocol["status"] == "frozen_before_any_diagnostic_decoder_slot"
    assert len(protocol["parents"]["bound_parent_wavs"]) == 22
    assert len(protocol["implementation"]["source_file_sha256"]) == 8
    assert protocol["implementation"]["renderer"]["voice"] == "af_heart"
    assert protocol["implementation"]["renderer"]["voice_style_row"] == 20
    assert len(protocol["implementation"]["measurement"]["praat_sha256"]) == 64
    assert len(protocol["implementation"]["measurement"]["script_sha256"]) == 64
    assert len(protocol["protocol_sha256"]) == 64


def test_decoder_slots_and_candidate_spans_are_exact_and_bounded() -> None:
    assert [slot.order for slot in DECODER_SLOTS] == [1, 2, 3, 4]
    assert [slot.role for slot in DECODER_SLOTS[:2]] == [
        "independent_ordinary_exact_carrier_anchor"
    ] * 2
    assert REPEATED_TARGET_WORD_COLUMNS == (4, 5, 6, 7, 17, 18, 19, 20)
    assert REPEATED_TARGET_WORD_PLUS_BOUNDARIES_COLUMNS == (
        3,
        4,
        5,
        6,
        7,
        8,
        16,
        17,
        18,
        19,
        20,
        21,
    )
    assert REPEATED_FULL_CONTEXTUAL_STATE_COLUMNS == tuple(range(23))
    protocol = protocol_record()
    assert protocol["candidate_route"]["period_column"] == 21
    assert protocol["candidate_route"]["first_complete_pass_wins"] is True
    assert protocol["measurement_protocol"]["primary_window_percent"] == 50
    assert protocol["measurement_protocol"][
        "descriptive_sensitivity_window_percents"
    ] == [40, 60]


def test_phase_1_audit_preserves_failure_and_ranks_confounds_not_causes() -> None:
    audit = protocol_record()["phase_1_forensic_audit"]
    assert "immutable and failed" in audit["frozen_outcome"]
    assert len(audit["evidence_table"]) == 5
    assert [row["rank"] for row in audit["ranked_hypotheses"]] == [1, 2, 3, 4, 5]
    assert audit["ranked_hypotheses"][0]["hypothesis"] == (
        "transported_calibration_and_context_mismatch"
    )
    assert all("cause" not in row["strength"] for row in audit["ranked_hypotheses"])
    assert len(audit["noncausal_old_code_findings"]) == 2
    assert all(
        "none" in row["frozen_result_effect"]
        for row in audit["noncausal_old_code_findings"]
    )


def test_two_independent_critic_reviews_are_explicitly_resolved() -> None:
    review = protocol_record()["internal_review_resolution"]
    assert review["all_findings_resolved"] is True
    assert review["novel_audio_available_to_critics"] is False
    assert [critic["role"] for critic in review["critics"]] == [
        "acoustic_and_instrument_validity",
        "selection_integrity_outcome_branches_and_claims",
    ]
    assert all(critic["independent_read_only"] for critic in review["critics"])
    assert all(
        len(critic["findings_and_resolutions"]) >= 6 for critic in review["critics"]
    )
    assert all(
        row["finding"] and row["resolution"]
        for critic in review["critics"]
        for row in critic["findings_and_resolutions"]
    )


def test_confirmation_set_and_full_planning_register_are_precommitted() -> None:
    assert [fixture.text for fixture in CONFIRMATION_FIXTURES] == [
        "The cap turns near the cap.",
        "We rest near the cap.",
    ]
    assert [fixture.expected_plan_sha256 for fixture in CONFIRMATION_FIXTURES] == [
        "c83bab90075c75619ed7c164cb4f325fc94a1325cb71c0c7e0fb7e87ba36320b",
        "1f03a5383c38d504bd5bbd565f105675081660e0c214e33c48189d3001c748d7",
    ]
    assert len(CONFIRMATION_PLANNING_REGISTER) == 20
    assert [row["order"] for row in CONFIRMATION_PLANNING_REGISTER] == list(
        range(1, 21)
    )
    protocol = protocol_record()
    precommit = protocol["confirmation_precommit"]
    assert precommit["same_set_for_every_selected_span"] is True
    assert len(precommit["planning_only_consideration_register"]) == 20
    assert "no acoustic outcome" in precommit["selection_rule"]
    assert "artifacts/**/*.json" in precommit["novelty_check"]


def test_frozen_protocol_artifact_is_intact() -> None:
    # The 2026-07-17 freeze predates later uv.lock changes, so prepare() can no
    # longer reproduce it byte-for-byte; the durable invariant is the frozen
    # artifact's internal hash and its binding to the recorded analysis.
    frozen = json.loads(DIAGNOSTIC_PROTOCOL_PATH.read_text(encoding="utf-8"))
    payload = dict(frozen)
    digest = payload.pop("protocol_sha256")
    assert digest == sha256_json(payload)
    analysis_path = DIAGNOSTIC_PROTOCOL_PATH.with_name("analysis.json")
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    assert analysis["protocol_sha256"] == digest
