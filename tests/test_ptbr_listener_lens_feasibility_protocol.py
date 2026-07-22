from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

import earshift_bakeoff.ptbr_listener_lens_feasibility_protocol as protocol_module
from earshift_bakeoff.portuguese_carrier_planner_v1 import (
    PortugueseCarrierPlan,
    PortugueseCarrierWord,
    PortugueseGateReceipt,
)
from earshift_bakeoff.ptbr_listener_lens_feasibility_protocol import (
    EXPECTED_BASE_PLAN_SHA256,
    EXPECTED_ACOUSTIC_REPORT_SHA256,
    EXPECTED_EN_GATE_PROTOCOL_SHA256,
    EXPECTED_EN_GATE_RECEIPT_SHA256,
    EXPECTED_EVIDENCE_SHA256,
    EXPECTED_PROFILE_SHA256,
    EXPECTED_PT_GATE_PROTOCOL_SHA256,
    EXPECTED_PT_GATE_RECEIPT_SHA256,
    EXPECTED_RESOLUTION_SHA256,
    EXPECTED_SELECTION_REPORT_SHA256,
    EXPECTED_VOICE_SCREEN_SUMMARY_SHA256,
    MAX_DECODER_CALLS,
    RESPONSE_FILENAME,
    RULE_ID,
    SECONDARY_UNRENDERED_RULE_ID,
    TECHNICAL_PROBE_VOICE_ID,
    ReciprocalFeasibilityProtocolError,
    _load_profile_and_evidence,
    _load_voice_screen_summary,
    _validate_raw_report,
    _validate_review_snapshot,
    assert_independent_review_resolved,
    build_reciprocal_profile_phone_plan,
    protocol_record,
    render_manifest,
    validate_independent_review_chain,
    write_frozen_protocol,
)


class _MandatoryPhoneGate:
    def __init__(self, collisions: set[str | tuple[str, str]] | None = None) -> None:
        self.collisions = collisions or set()
        self.calls: list[tuple[str, str]] = []

    def phone_match(self, language: str, phone: str) -> bool:
        self.calls.append((language, phone))
        return phone in self.collisions or (language, phone) in self.collisions


class _NativePositiveIndex:
    def __init__(self, collisions: set[str] | None = None) -> None:
        self.collisions = collisions or set()
        self.calls: list[str] = []

    def phone_match(self, phone: str) -> bool:
        self.calls.append(phone)
        return phone in self.collisions


def _base_plan(*, source_phone: str = "avˈɔ") -> PortugueseCarrierPlan:
    gate = PortugueseGateReceipt(
        mandatory_written_espeak_gate_pass=True,
        mandatory_gate_language="pt",
        mandatory_espeak_voice="pt-br",
        native_index_scope="partial_positive_only_index",
        native_positive_only_gate_pass=True,
        native_negative_used_for_clearance=False,
        isolated_candidates_checked=2,
        adjacency_pairs_checked=1,
        candidate_attempts=2,
        candidate_rejection_counts={},
        exact_native_phrase_plan=True,
        model_representable=True,
    )
    words = (
        PortugueseCarrierWord(
            word_index=0,
            source="foi",
            source_phone="fˈoj",
            carrier_role="content",
            carrier_surface="bado",
            carrier_phone="bˈɔd",
            target_occurrence_count=0,
            candidate_attempt=0,
        ),
        PortugueseCarrierWord(
            word_index=1,
            source="avó",
            source_phone=source_phone,
            carrier_role="content",
            carrier_surface="pleras",
            carrier_phone="plˈeɾæs",
            target_occurrence_count=1,
            candidate_attempt=0,
        ),
    )
    return PortugueseCarrierPlan(
        planner_version=1,
        language_id="pt-BR",
        kokoro_lang_code="p",
        voice_id="pf_dora",
        candidate_enabled=False,
        production_route_available=False,
        normalized_text="Foi avó.",
        source_phonemes=f"fˈoj {source_phone}.",
        carrier_script="bado pleras.",
        carrier_phonemes="bˈɔd plˈeɾæs.",
        target_phone="ɔ",
        target_analysis_scope="isolated_native_kokoro_word_predictions",
        target_word_indexes=(1,),
        target_word_count=1,
        target_occurrence_count=1,
        target_available=True,
        words=words,
        gate_receipt=gate,
        plan_sha256=EXPECTED_BASE_PLAN_SHA256,
    )


def _vocab() -> frozenset[str]:
    return frozenset(" bdoˈplɔɑeɾæs.fjv")


def test_profile_layer_sanitizes_incidental_symbols_and_assigns_exact_target() -> None:
    mandatory = _MandatoryPhoneGate()
    native = _NativePositiveIndex()
    english = _NativePositiveIndex()

    plan = build_reciprocal_profile_phone_plan(
        _base_plan(),
        model_vocab=_vocab(),
        mandatory_gate=mandatory,  # type: ignore[arg-type]
        native_index=native,
        english_kokoro_index=english,
        english_kokoro_compatible_symbols=_vocab(),
    )

    assert plan.neutral_phonemes == "bˈod plˈɔɾæs."
    assert plan.lens_phonemes == "bˈod plˈɑɾæs."
    assert plan.source_alignment_phonemes == "bˈod plˈeɾæs."
    assert plan.neutral_phonemes.count("ɔ") == 1
    assert plan.neutral_phonemes.count("ɑ") == 0
    assert plan.lens_phonemes.count("ɑ") == 1
    assert plan.lens_phonemes.count("ɔ") == 0
    assert len(plan.target_occurrences) == 1
    occurrence = plan.target_occurrences[0]
    assert occurrence.source_word_index == 1
    assert occurrence.stress_model_column + 1 == occurrence.model_column
    assert plan.neutral_phonemes[occurrence.profile_character_index] == "ɔ"
    assert plan.lens_phonemes[occurrence.profile_character_index] == "ɑ"
    assert plan.equal_model_token_count == len(plan.neutral_phonemes)
    assert (
        plan.derived_phone_gate_receipt.portuguese_native_negative_used_for_clearance
        is False
    )
    english_receipt = (
        plan.derived_phone_gate_receipt.supplemental_english_listener_collision
    )
    assert (
        english_receipt.clearance_role == "supplemental_rejection_only_never_clearance"
    )
    assert english_receipt.negative_used_for_portuguese_clearance is False
    assert english_receipt.no_exact_positive_among_compatible_comparisons is True
    assert english_receipt.sequence_count == 6
    assert english_receipt.espeak_compatible_sequence_count == 2
    assert english_receipt.espeak_incompatible_sequence_count == 4
    assert len(mandatory.calls) == 8
    assert len(native.calls) == 6
    assert len(english.calls) == 6


def test_profile_layer_rejects_unstressed_source_target() -> None:
    with pytest.raises(ReciprocalFeasibilityProtocolError, match="stressed source"):
        build_reciprocal_profile_phone_plan(
            _base_plan(source_phone="avɔ"),
            model_vocab=_vocab(),
            mandatory_gate=_MandatoryPhoneGate(),  # type: ignore[arg-type]
            native_index=_NativePositiveIndex(),
            english_kokoro_index=_NativePositiveIndex(),
            english_kokoro_compatible_symbols=_vocab(),
        )


def test_profile_layer_rejects_a_derived_exact_phone_collision() -> None:
    with pytest.raises(ReciprocalFeasibilityProtocolError, match="written/eSpeak"):
        build_reciprocal_profile_phone_plan(
            _base_plan(),
            model_vocab=_vocab(),
            mandatory_gate=_MandatoryPhoneGate({"plˈɔɾæs"}),  # type: ignore[arg-type]
            native_index=_NativePositiveIndex(),
            english_kokoro_index=_NativePositiveIndex(),
            english_kokoro_compatible_symbols=_vocab(),
        )


def test_profile_layer_rejects_exact_supplemental_english_positive() -> None:
    with pytest.raises(ReciprocalFeasibilityProtocolError, match="English-listener"):
        build_reciprocal_profile_phone_plan(
            _base_plan(),
            model_vocab=_vocab(),
            mandatory_gate=_MandatoryPhoneGate({("en", "bˈod")}),  # type: ignore[arg-type]
            native_index=_NativePositiveIndex(),
            english_kokoro_index=_NativePositiveIndex(),
            english_kokoro_compatible_symbols=_vocab(),
        )


def test_supplemental_english_kokoro_incompatibility_is_recorded_not_cleared() -> None:
    plan = build_reciprocal_profile_phone_plan(
        _base_plan(),
        model_vocab=_vocab(),
        mandatory_gate=_MandatoryPhoneGate(),  # type: ignore[arg-type]
        native_index=_NativePositiveIndex(),
        english_kokoro_index=_NativePositiveIndex(),
        english_kokoro_compatible_symbols=_vocab() - {"ɾ"},
    )

    receipt = plan.derived_phone_gate_receipt.supplemental_english_listener_collision
    incompatible = [
        row for row in receipt.sequences if not row.kokoro_exact_comparison_compatible
    ]
    assert receipt.kokoro_incompatible_sequence_count == len(incompatible) > 0
    assert all(row.kokoro_positive_match is None for row in incompatible)
    assert all(row.kokoro_incompatibility_reason for row in incompatible)
    assert receipt.negative_used_for_portuguese_clearance is False


def test_manifest_is_exactly_two_independent_anchors_and_one_triplet() -> None:
    manifest = render_manifest()

    assert len(manifest) == MAX_DECODER_CALLS == 5
    assert [row["order"] for row in manifest] == [1, 2, 3, 4, 5]
    assert [row["slot_id"] for row in manifest] == [
        "ordinary-anchor-neutral",
        "ordinary-anchor-lens",
        "controlled-neutral",
        "controlled-identity",
        "controlled-lens",
    ]
    assert sum(row["mode"] == "ordinary_context_local_anchor" for row in manifest) == 2
    assert RESPONSE_FILENAME == "ptbr-to-ae-listener-lens-v1-response.json"
    assert TECHNICAL_PROBE_VOICE_ID == "pf_dora"


def test_only_strongest_committed_rule_is_eligible_for_this_chain() -> None:
    profile, evidence = _load_profile_and_evidence()
    rules = {row["rule_id"]: row for row in profile["rules"]}
    candidates = {row["id"]: row for row in evidence["candidate_contrasts"]}

    assert (rules[RULE_ID]["source_phone"], rules[RULE_ID]["lens_phone"]) == (
        "ɔ",
        "ɑ",
    )
    assert candidates[RULE_ID]["observed_response"]["response_share"] == 0.72
    assert rules[SECONDARY_UNRENDERED_RULE_ID]["enabled"] is False


def test_protocol_draft_serializes_review_collision_and_window_boundaries(
    tmp_path: Path,
) -> None:
    protocol = protocol_record(approval_paths=_approval_paths(tmp_path))

    review = protocol["independent_protocol_review"]
    assert review["status"] == "awaiting_repeated_independent_rechecks"
    basis = review["pending_review_basis"]
    assert basis["status"] == "awaiting_repeated_independent_rechecks"
    assert basis["received_report_count"] == 2
    assert basis["resolved_finding_count"] == 15
    assert review["received_recheck_count"] == 0
    assert review["freeze_authorized"] is False
    windows = protocol["automatic_gate"]["measurement_window_policy"]
    assert windows == {
        "primary_fraction": 0.5,
        "primary_relative_bounds": [0.25, 0.75],
        "exploratory_fractions": [],
        "compute_middle_40_percent": False,
        "compute_middle_60_percent": False,
        "window_selection": "none",
    }
    english = protocol["gates"]["supplemental_english_listener_collision_evidence"]
    assert english["negative_used_for_portuguese_clearance"] is False
    assert english["receipt"]["clearance_role"] == (
        "supplemental_rejection_only_never_clearance"
    )
    assert english["receipt"]["no_exact_positive_among_compatible_comparisons"] is True


def test_review_chain_validates_reports_and_resolution_but_blocks_freeze(
    tmp_path: Path,
) -> None:
    review = validate_independent_review_chain()
    assert review["status"] == "awaiting_repeated_independent_rechecks"
    assert review["received_report_count"] == 2
    assert review["resolved_finding_count"] == 15
    assert review["required_recheck_count"] == 2
    assert review["self_authorization_permitted"] is False

    approval_paths = _approval_paths(tmp_path)
    current = protocol_record(approval_paths=approval_paths)
    with pytest.raises(ReciprocalFeasibilityProtocolError, match="pending two exact"):
        assert_independent_review_resolved(current, approval_paths=approval_paths)
    destination = tmp_path / "protocol.json"
    with pytest.raises(ReciprocalFeasibilityProtocolError, match="pending two exact"):
        write_frozen_protocol(
            current,
            destination,
            approval_paths=approval_paths,
        )
    assert not destination.exists()


def test_production_approval_transition_is_closed_and_current() -> None:
    approval_paths = protocol_module.default_authorization_approval_paths()
    paths = (
        approval_paths.selection_recheck,
        approval_paths.acoustic_recheck,
        approval_paths.final_approval_resolution,
    )
    present = [path.is_file() for path in paths]
    assert all(present) or not any(present)

    protocol = protocol_record()
    review = protocol["independent_protocol_review"]
    if all(present):
        assert protocol["status"] == "independent_rechecks_approved_ready_for_freeze"
        assert review["status"] == "authorized_for_protocol_freeze"
        assert review["received_recheck_count"] == 2
        assert review["all_findings_independently_rechecked"] is True
        assert review["freeze_authorized"] is True
    else:
        assert protocol["status"] == (
            "authorization_gate_resolved_awaiting_repeated_rechecks"
        )
        assert review["status"] == "awaiting_repeated_independent_rechecks"
        assert review["received_recheck_count"] == 0
        assert review["all_findings_independently_rechecked"] is False
        assert review["freeze_authorized"] is False


def test_review_chain_rejects_duplicate_or_stale_report_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot = _validate_review_snapshot()
    report = json.loads(
        protocol_module.SELECTION_REPORT_PATH.read_text(encoding="utf-8")
    )
    report["reviewed_snapshot"]["semantic_sha256"] = "0" * 64
    stale = tmp_path / "stale.json"
    stale.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ReciprocalFeasibilityProtocolError, match="snapshot binding"):
        _validate_raw_report(
            path=stale,
            expected_sha256=protocol_module.sha256_file(stale),
            expected_report_id=report["report_id"],
            expected_role=report["reviewer_role"],
            snapshot=snapshot,
        )

    duplicate = json.loads(
        protocol_module.SELECTION_REPORT_PATH.read_text(encoding="utf-8")
    )
    monkeypatch.setattr(protocol_module, "_validate_review_snapshot", lambda: snapshot)
    monkeypatch.setattr(
        protocol_module,
        "_validate_raw_report",
        lambda **kwargs: duplicate,
    )
    with pytest.raises(ReciprocalFeasibilityProtocolError, match="distinct report_id"):
        validate_independent_review_chain()


def _write_approval_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _approval_paths(root: Path) -> protocol_module.AuthorizationApprovalPaths:
    research = root / "docs" / "research"
    return protocol_module.AuthorizationApprovalPaths(
        selection_recheck=(research / protocol_module.SELECTION_RECHECK_V2_PATH.name),
        acoustic_recheck=(research / protocol_module.ACOUSTIC_RECHECK_V2_PATH.name),
        final_approval_resolution=(
            research / protocol_module.FINAL_APPROVAL_RESOLUTION_PATH.name
        ),
    )


def _approval_payloads(
    root: Path,
) -> tuple[
    protocol_module.AuthorizationApprovalPaths,
    dict[str, object],
    dict[str, object],
    dict[str, object],
    dict[str, object],
]:
    paths = _approval_paths(root)
    pending = protocol_record(approval_paths=paths)
    subject = pending["independent_protocol_review"]["pending_subject"]
    gate_binding = {
        "path": str(
            protocol_module.AUTHORIZATION_GATE_RESOLUTION_PATH.relative_to(
                protocol_module.Paths().root
            )
        ),
        "sha256": protocol_module.EXPECTED_AUTHORIZATION_GATE_RESOLUTION_SHA256,
    }
    original_binding = {
        "path": str(
            protocol_module.RESOLUTION_PATH.relative_to(protocol_module.Paths().root)
        ),
        "sha256": EXPECTED_RESOLUTION_SHA256,
    }

    def recheck(
        *,
        role: str,
        reviewer_id: str,
        recheck_id: str,
        report_path: Path,
        report_id: str,
        report_sha256: str,
    ) -> dict[str, object]:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        return {
            "schema_version": 2,
            "status": "immutable_independent_authorization_recheck",
            "recheck_id": recheck_id,
            "reviewer_id": reviewer_id,
            "reviewer_role": role,
            "source_report": {
                "path": str(report_path.relative_to(protocol_module.Paths().root)),
                "report_id": report_id,
                "sha256": report_sha256,
            },
            "original_resolution": original_binding,
            "authorization_gate_resolution": gate_binding,
            "reviewed_pending_subject": subject,
            "finding_reviews": [
                {
                    "finding_id": item["finding_id"],
                    "verdict": "approve",
                    "resolved": True,
                }
                for item in report["findings"]
            ],
            "residual_findings": [],
            "new_blockers": [],
            "verdict": "approve",
            "freeze_authorized": True,
        }

    selection = recheck(
        role="selection-integrity-claims-and-human-review",
        reviewer_id=protocol_module.SELECTION_RECHECK_V2_REVIEWER_ID,
        recheck_id="track-d-selection-claims-independent-recheck-v2",
        report_path=protocol_module.SELECTION_REPORT_PATH,
        report_id="track-d-selection-claims-independent-report-v1",
        report_sha256=EXPECTED_SELECTION_REPORT_SHA256,
    )
    acoustic = recheck(
        role="acoustic-instrument-alignment-and-thresholds",
        reviewer_id=protocol_module.ACOUSTIC_RECHECK_V2_REVIEWER_ID,
        recheck_id="track-d-acoustic-instrument-independent-recheck-v2",
        report_path=protocol_module.ACOUSTIC_REPORT_PATH,
        report_id="track-d-acoustic-instrument-independent-report-v1",
        report_sha256=EXPECTED_ACOUSTIC_REPORT_SHA256,
    )
    _write_approval_json(paths.selection_recheck, selection)
    _write_approval_json(paths.acoustic_recheck, acoustic)
    final = {
        "schema_version": 1,
        "status": "two_independent_approvals_bound",
        "resolution_id": (
            "track-d-reciprocal-feasibility-final-approval-resolution-v1"
        ),
        "authorization_gate_resolution": gate_binding,
        "reviewed_pending_subject": subject,
        "rechecks": [
            {
                "path": str(
                    protocol_module.SELECTION_RECHECK_V2_PATH.relative_to(
                        protocol_module.Paths().root
                    )
                ),
                "sha256": protocol_module.sha256_file(paths.selection_recheck),
                "reviewer_id": protocol_module.SELECTION_RECHECK_V2_REVIEWER_ID,
                "reviewer_role": "selection-integrity-claims-and-human-review",
            },
            {
                "path": str(
                    protocol_module.ACOUSTIC_RECHECK_V2_PATH.relative_to(
                        protocol_module.Paths().root
                    )
                ),
                "sha256": protocol_module.sha256_file(paths.acoustic_recheck),
                "reviewer_id": protocol_module.ACOUSTIC_RECHECK_V2_REVIEWER_ID,
                "reviewer_role": "acoustic-instrument-alignment-and-thresholds",
            },
        ],
        "all_findings_approved": True,
        "freeze_authorized": True,
    }
    _write_approval_json(paths.final_approval_resolution, final)
    return paths, pending, selection, acoustic, final


def _rewrite_approval_set(
    paths: protocol_module.AuthorizationApprovalPaths,
    selection: dict[str, object],
    acoustic: dict[str, object],
    final: dict[str, object],
) -> None:
    _write_approval_json(paths.selection_recheck, selection)
    _write_approval_json(paths.acoustic_recheck, acoustic)
    receipts = final["rechecks"]
    assert isinstance(receipts, list)
    receipts[0]["sha256"] = protocol_module.sha256_file(paths.selection_recheck)
    receipts[1]["sha256"] = protocol_module.sha256_file(paths.acoustic_recheck)
    _write_approval_json(paths.final_approval_resolution, final)


def _commit_approval_set(repo: Path) -> None:
    subprocess.run(["git", "-C", str(repo), "init", "-q"], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.email", "test@example.invalid"],
        check=True,
    )
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "Test"], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "commit", "-q", "-m", "bind approvals"],
        check=True,
    )


def test_authorization_gate_two_exact_approvals_unlock_writer(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    paths, pending, selection, acoustic, final = _approval_payloads(repo)
    _rewrite_approval_set(paths, selection, acoustic, final)
    _commit_approval_set(repo)

    authorized = protocol_record(approval_paths=paths)
    review = authorized["independent_protocol_review"]
    assert review["status"] == "authorized_for_protocol_freeze"
    assert review["received_recheck_count"] == 2
    assert review["freeze_authorized"] is True
    assert (
        review["pending_subject"]
        == pending["independent_protocol_review"]["pending_subject"]
    )

    destination = tmp_path / "frozen" / "protocol.json"
    write_frozen_protocol(
        authorized,
        destination,
        approval_paths=paths,
        repository=repo,
    )
    assert protocol_module.stable_json(
        json.loads(destination.read_text(encoding="utf-8"))
    ) == protocol_module.stable_json(authorized)


def test_authorization_writer_rejects_untracked_symlink_aliases(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    targets = [repo / f"approval-{index}.json" for index in range(3)]
    for index, target in enumerate(targets):
        target.write_text(json.dumps({"approval": index}) + "\n", encoding="utf-8")
    _commit_approval_set(repo)
    paths = _approval_paths(repo)
    for alias, target in zip(
        (
            paths.selection_recheck,
            paths.acoustic_recheck,
            paths.final_approval_resolution,
        ),
        targets,
        strict=True,
    ):
        alias.parent.mkdir(parents=True, exist_ok=True)
        alias.symlink_to(target)
    review = {
        "rechecks": [
            {"sha256": protocol_module.sha256_file(target)} for target in targets[:2]
        ],
        "final_approval_resolution": {
            "sha256": protocol_module.sha256_file(targets[2])
        },
    }

    with pytest.raises(ReciprocalFeasibilityProtocolError, match="symlink"):
        protocol_module._verify_authorization_inputs_at_head(
            approval_paths=paths,
            review=review,
            repository=repo,
        )


def test_authorization_gate_missing_pair_pending_and_one_file_hard_fails(
    tmp_path: Path,
) -> None:
    paths = _approval_paths(tmp_path)
    pending = protocol_record(approval_paths=paths)
    assert pending["independent_protocol_review"]["received_recheck_count"] == 0
    paths.selection_recheck.parent.mkdir(parents=True)
    paths.selection_recheck.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ReciprocalFeasibilityProtocolError, match="only part"):
        protocol_record(approval_paths=paths)


def test_authorization_gate_rejects_duplicate_reviewer_or_role(
    tmp_path: Path,
) -> None:
    paths, _, selection, acoustic, final = _approval_payloads(tmp_path)
    acoustic["reviewer_id"] = selection["reviewer_id"]
    acoustic["reviewer_role"] = selection["reviewer_role"]
    receipts = final["rechecks"]
    assert isinstance(receipts, list)
    receipts[1]["reviewer_id"] = selection["reviewer_id"]
    receipts[1]["reviewer_role"] = selection["reviewer_role"]
    _rewrite_approval_set(paths, selection, acoustic, final)
    with pytest.raises(ReciprocalFeasibilityProtocolError, match="whole-file|stale"):
        protocol_record(approval_paths=paths)


@pytest.mark.parametrize(
    "mutation", ["extra-key", "missing-finding", "numeric-resolved", "reject"]
)
def test_authorization_gate_rejects_extra_keys_missing_finding_and_wrong_verdict(
    tmp_path: Path,
    mutation: str,
) -> None:
    paths, _, selection, acoustic, final = _approval_payloads(tmp_path)
    if mutation == "extra-key":
        selection["unexpected"] = True
    elif mutation == "missing-finding":
        reviews = selection["finding_reviews"]
        assert isinstance(reviews, list)
        reviews.pop()
    elif mutation == "numeric-resolved":
        reviews = selection["finding_reviews"]
        assert isinstance(reviews, list)
        reviews[0]["resolved"] = 1
    else:
        selection["verdict"] = "reject"
    _rewrite_approval_set(paths, selection, acoustic, final)
    with pytest.raises(ReciprocalFeasibilityProtocolError, match="keys|incomplete"):
        protocol_record(approval_paths=paths)


@pytest.mark.parametrize(
    "mutation",
    ["subject", "source-report", "original-resolution", "gate-resolution", "edit"],
)
def test_authorization_gate_rejects_stale_bindings_and_edited_whole_file(
    tmp_path: Path,
    mutation: str,
) -> None:
    paths, _, selection, acoustic, final = _approval_payloads(tmp_path)
    if mutation == "edit":
        paths.selection_recheck.write_text(
            paths.selection_recheck.read_text(encoding="utf-8") + "\n",
            encoding="utf-8",
        )
    else:
        field = {
            "subject": "reviewed_pending_subject",
            "source-report": "source_report",
            "original-resolution": "original_resolution",
            "gate-resolution": "authorization_gate_resolution",
        }[mutation]
        binding = selection[field]
        assert isinstance(binding, dict)
        hash_key = "protocol_sha256" if mutation == "subject" else "sha256"
        binding[hash_key] = "0" * 64
        _rewrite_approval_set(paths, selection, acoustic, final)
    with pytest.raises(
        ReciprocalFeasibilityProtocolError, match="whole-file|stale|incomplete"
    ):
        protocol_record(approval_paths=paths)


def test_authorized_transition_preserves_canonical_scientific_subject(
    tmp_path: Path,
) -> None:
    paths, pending, selection, acoustic, final = _approval_payloads(tmp_path)
    _rewrite_approval_set(paths, selection, acoustic, final)
    authorized = protocol_record(approval_paths=paths)

    assert protocol_module.canonical_pending_protocol_payload(authorized) == (
        protocol_module.canonical_pending_protocol_payload(pending)
    )
    assert (
        authorized["independent_protocol_review"]["pending_subject"]
        == pending["independent_protocol_review"]["pending_subject"]
    )
    for field in (
        "question",
        "claim_boundary",
        "evidence",
        "fixture",
        "renderer",
        "gates",
        "automatic_gate",
        "review_contract",
    ):
        assert authorized[field] == pending[field]


def test_writer_refuses_every_preeligibility_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol = {"protocol_sha256": "a" * 64}
    monkeypatch.setattr(
        protocol_module,
        "assert_independent_review_resolved",
        lambda value, **kwargs: {
            "rechecks": [{"sha256": "a" * 64}, {"sha256": "b" * 64}],
            "final_approval_resolution": {"sha256": "c" * 64},
        },
    )
    monkeypatch.setattr(
        protocol_module, "_verify_authorization_inputs_at_head", lambda **kwargs: {}
    )
    for name in (
        "render-attempt.json",
        "render-records.json",
        "analysis.json",
        "audio",
        "public",
        "private",
        "review-generation-failure.json",
    ):
        root = tmp_path / name.replace(".", "_")
        destination = root / "protocol.json"
        stale = destination.parent / name
        if "." in name:
            stale.parent.mkdir(parents=True, exist_ok=True)
            stale.write_text("stale", encoding="utf-8")
        else:
            stale.mkdir(parents=True)
        with pytest.raises(ReciprocalFeasibilityProtocolError, match="evidence exists"):
            write_frozen_protocol(protocol, destination)


def test_forged_review_booleans_cannot_authorize_freeze() -> None:
    forged = {
        "independent_protocol_review": {
            "status": "resolved",
            "received_report_count": 2,
            "freeze_authorized": True,
        }
    }
    with pytest.raises(ReciprocalFeasibilityProtocolError, match="exact recomputed"):
        assert_independent_review_resolved(forged)


def test_frozen_protocol_verifier_rejects_exact_pending_protocol(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    approval_paths = _approval_paths(tmp_path / "missing-approvals")
    monkeypatch.setattr(
        protocol_module,
        "default_authorization_approval_paths",
        lambda: approval_paths,
    )
    pending = protocol_record(approval_paths=approval_paths)
    monkeypatch.setattr(protocol_module, "run_dir", lambda: tmp_path)
    _write_approval_json(tmp_path / "protocol.json", pending)

    with pytest.raises(ReciprocalFeasibilityProtocolError, match="pending two exact"):
        protocol_module.verify_frozen_protocol()


def test_english_espeak_queries_only_audited_compatible_sequences() -> None:
    mandatory = _MandatoryPhoneGate()
    plan = build_reciprocal_profile_phone_plan(
        _base_plan(),
        model_vocab=_vocab(),
        mandatory_gate=mandatory,  # type: ignore[arg-type]
        native_index=_NativePositiveIndex(),
        english_kokoro_index=_NativePositiveIndex(),
        english_kokoro_compatible_symbols=_vocab(),
    )
    receipt = plan.derived_phone_gate_receipt.supplemental_english_listener_collision
    english_calls = [phone for language, phone in mandatory.calls if language == "en"]
    compatible = [
        row for row in receipt.sequences if row.espeak_exact_comparison_compatible
    ]
    incompatible = [
        row for row in receipt.sequences if not row.espeak_exact_comparison_compatible
    ]
    assert english_calls == ["bˈod", "bˈod"]
    assert len(compatible) == receipt.espeak_compatible_sequence_count == 2
    assert len(incompatible) == receipt.espeak_incompatible_sequence_count == 4
    assert all(row.espeak_positive_match is None for row in incompatible)
    assert all(row.espeak_incompatibility_reason for row in incompatible)


def test_compatible_english_positive_rejects_for_both_indexes() -> None:
    with pytest.raises(ReciprocalFeasibilityProtocolError, match="English-listener"):
        build_reciprocal_profile_phone_plan(
            _base_plan(),
            model_vocab=_vocab(),
            mandatory_gate=_MandatoryPhoneGate({("en", "bˈod")}),  # type: ignore[arg-type]
            native_index=_NativePositiveIndex(),
            english_kokoro_index=_NativePositiveIndex(),
            english_kokoro_compatible_symbols=_vocab(),
        )
    with pytest.raises(ReciprocalFeasibilityProtocolError, match="English-listener"):
        build_reciprocal_profile_phone_plan(
            _base_plan(),
            model_vocab=_vocab(),
            mandatory_gate=_MandatoryPhoneGate(),  # type: ignore[arg-type]
            native_index=_NativePositiveIndex(),
            english_kokoro_index=_NativePositiveIndex({"bˈod"}),
            english_kokoro_compatible_symbols=_vocab(),
        )


def test_frozen_evidence_gate_and_rule_bindings_are_exact() -> None:
    profile, evidence = _load_profile_and_evidence()
    assert protocol_module.sha256_file(protocol_module.EVIDENCE_PATH) == (
        EXPECTED_EVIDENCE_SHA256
    )
    assert protocol_module.sha256_file(protocol_module.PROFILE_PATH) == (
        EXPECTED_PROFILE_SHA256
    )
    assert protocol_module.sha256_file(
        protocol_module.PT_GATE_ROOT / "protocol.json"
    ) == (EXPECTED_PT_GATE_PROTOCOL_SHA256)
    assert (
        protocol_module.sha256_file(
            protocol_module.PT_GATE_ROOT / "full-index-receipt.json"
        )
        == EXPECTED_PT_GATE_RECEIPT_SHA256
    )
    assert (
        protocol_module.sha256_file(
            protocol_module.ENGLISH_KOKORO_GATE_ROOT / "protocol.json"
        )
        == EXPECTED_EN_GATE_PROTOCOL_SHA256
    )
    assert (
        protocol_module.sha256_file(
            protocol_module.ENGLISH_KOKORO_GATE_ROOT / "full-index-receipt.json"
        )
        == EXPECTED_EN_GATE_RECEIPT_SHA256
    )
    primary = next(
        item for item in evidence["candidate_contrasts"] if item["id"] == RULE_ID
    )
    assert primary["confidence_tier"] == "tier-1-direct-majority-small-sample"
    assert primary["source_ids"] == ["D1"]
    rules = {item["rule_id"]: item for item in profile["rules"]}
    assert rules[SECONDARY_UNRENDERED_RULE_ID] == {
        "rule_id": SECONDARY_UNRENDERED_RULE_ID,
        "priority": 2,
        "enabled": False,
        "source_phone": "a",
        "neutral_phone": "a",
        "lens_phone": "æ",
        "target_scope": "exact stressed /a/ occurrences only; never unstressed /ɐ/",
        "evidence_tier": "tier-2-direct-weak-majority-small-sample",
        "stage": "eligible-for-future-disabled-acoustic-feasibility-only",
        "exact_target_coverage_required": True,
        "ordinary_context_local_anchors_required": True,
        "listener_validation_complete": False,
    }


def test_voice_screen_binding_is_pending_no_selection_and_nontransferable() -> None:
    summary = _load_voice_screen_summary()
    assert protocol_module.sha256_file(protocol_module.VOICE_SCREEN_SUMMARY_PATH) == (
        EXPECTED_VOICE_SCREEN_SUMMARY_SHA256
    )
    assert summary["status"] == "pending-human-review"
    assert summary["voice_selection_performed"] is False
    protocol = protocol_record()
    boundary = protocol["voice_screen_boundary"]
    assert boundary["technical_probe_is_selected_voice"] is False
    assert boundary["selected_voice_requires_separate_freeze"] is True
    assert "nontransferable" in protocol["renderer"]["voice_result_transfer"]


def test_claim_and_threshold_boundary_is_narrow_and_nonperceptual() -> None:
    protocol = protocol_record()
    assert protocol["claim_boundary"]["automatic_pass_proves"] == (
        "a local median-F1/F2 shift plus a localized waveform difference on this "
        "one fixed technical probe"
    )
    assert protocol["automatic_gate"]["threshold_interpretation"] == (
        "engineering_nonperceptual_criteria_only"
    )
    assert protocol["automatic_gate"]["minimum_anchor_distance_bark"] == 0.25


def test_resolution_hash_is_exact_and_awaiting_recheck() -> None:
    assert protocol_module.sha256_file(protocol_module.RESOLUTION_PATH) == (
        EXPECTED_RESOLUTION_SHA256
    )
    resolution = json.loads(protocol_module.RESOLUTION_PATH.read_text(encoding="utf-8"))
    assert resolution["status"] == "awaiting_independent_recheck"
    assert resolution["freeze_authorized"] is False
    assert len(resolution["finding_resolutions"]) == 15
    assert [item["sha256"] for item in resolution["reports"]] == [
        EXPECTED_SELECTION_REPORT_SHA256,
        EXPECTED_ACOUSTIC_REPORT_SHA256,
    ]
