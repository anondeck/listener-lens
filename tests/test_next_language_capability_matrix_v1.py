from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = (
    ROOT
    / "artifacts"
    / "language-expansion"
    / "20260717-next-language-capability-matrix-v1"
)

COMPONENT_KEYS = (
    "voices",
    "g2p",
    "gate",
    "phoneme_representation",
    "planner",
    "listener_profile_evidence",
    "fixtures",
    "acoustic_validation",
    "human_review",
)


def _load(name: str) -> dict[str, Any]:
    return json.loads((ARTIFACT / name).read_text(encoding="utf-8"))


def _stable_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def test_matrix_validates_against_checked_in_draft_2020_12_schema() -> None:
    schema = _load("schema.json")
    matrix = _load("capability-matrix.json")

    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(matrix)


def test_status_vocabularies_are_closed_and_every_language_is_disabled() -> None:
    matrix = _load("capability-matrix.json")
    vocabulary = matrix["status_vocabulary"]

    assert set(vocabulary["component_statuses"]) == {
        "verified_local",
        "verified_partial_positive_only",
        "verified_structural_probe_only",
        "declared_unverified",
        "blocked_by_variant_mismatch",
        "requires_new_evidence",
        "missing",
    }
    assert set(vocabulary["support_statuses"]) == {
        "foundation_only_disabled",
        "structural_path_only_disabled",
        "catalog_and_g2p_probe_only_disabled",
    }
    assert set(vocabulary["architecture_assessments"]) == {
        "portable",
        "portable_with_language_adapter",
        "english_bound",
        "evidence_chain_specific",
    }
    assert set(vocabulary["difficulty_levels"]) == {"medium", "high"}
    assert set(vocabulary["failure_severities"]) == {
        "blocking",
        "high",
        "medium",
    }

    languages = {row["language_id"]: row for row in matrix["language_capabilities"]}
    assert set(languages) == {"pt-BR", "es", "fr-FR", "it-IT"}
    assert matrix["all_candidates_enabled"] is False
    for language in languages.values():
        assert language["enabled"] is False
        assert language["production_support_claimed"] is False
        assert language["feature_flag"]["enabled"] is False
        assert language["feature_flag"]["routing_status"] == "not_reachable"
        assert language["support_status"] in vocabulary["support_statuses"]
        assert (
            language["estimated_implementation_difficulty"]
            in vocabulary["difficulty_levels"]
        )
        for key in COMPONENT_KEYS:
            assert language[key]["status"] in vocabulary["component_statuses"]

    assert languages["pt-BR"]["feature_flag"] == {
        "status": "defined_disabled",
        "name": "PORTUGUESE_RENDERER_CANDIDATE_ENABLED",
        "enabled": False,
        "routing_status": "not_reachable",
    }
    assert all(
        languages[language_id]["feature_flag"]["status"] == "not_defined"
        for language_id in ("es", "fr-FR", "it-IT")
    )


def test_evidence_and_failure_references_are_closed() -> None:
    matrix = _load("capability-matrix.json")
    evidence_ids = [row["id"] for row in matrix["evidence"]]
    failure_ids = [row["id"] for row in matrix["failure_modes"]]
    language_ids = {row["language_id"] for row in matrix["language_capabilities"]}

    assert len(evidence_ids) == len(set(evidence_ids))
    assert len(failure_ids) == len(set(failure_ids))

    evidence_set = set(evidence_ids)
    failure_set = set(failure_ids)
    for component in matrix["architecture_components"]:
        assert set(component["evidence_refs"]) <= evidence_set
        assert (
            component["assessment"]
            in matrix["status_vocabulary"]["architecture_assessments"]
        )
    for language in matrix["language_capabilities"]:
        assert set(language["blocking_failure_mode_refs"]) <= failure_set
        for key in COMPONENT_KEYS:
            assert set(language[key]["evidence_refs"]) <= evidence_set
    for failure in matrix["failure_modes"]:
        assert set(failure["applies_to"]) <= language_ids
        assert set(failure["evidence_refs"]) <= evidence_set
        assert failure["severity"] in matrix["status_vocabulary"]["failure_severities"]


def test_probe_receipts_and_language_boundaries_are_exact() -> None:
    matrix = _load("capability-matrix.json")
    evidence = {row["id"]: row for row in matrix["evidence"]}
    languages = {row["language_id"]: row for row in matrix["language_capabilities"]}

    probe_rows = [
        {"language_id": row["language_id"], **row["g2p"]["probe"]}
        for row in matrix["language_capabilities"]
    ]
    assert _stable_sha256(probe_rows) == evidence["EV-G2P-STRUCTURAL-PROBES"]["sha256"]

    spanish_gate = languages["es"]["gate"]
    assert spanish_gate["status"] == "blocked_by_variant_mismatch"
    assert spanish_gate["predicted_phone_variant"] == "es-419"
    assert languages["es"]["g2p"]["renderer_espeak_variant"] == "es"
    assert spanish_gate["renderer_variant_matches_gate"] is False
    assert (
        sum(not row["equal"] for row in spanish_gate["variant_comparison_probe"]) == 3
    )
    assert (
        _stable_sha256(spanish_gate["variant_comparison_probe"])
        == evidence["EV-ES-VARIANT-PROBE"]["sha256"]
    )

    portuguese_gate = languages["pt-BR"]["gate"]
    assert portuguese_gate["native_index_status"] == "partial_positive_only_index"
    assert portuguese_gate["negative_lookup_can_clear"] is False
    assert portuguese_gate["renderer_variant_matches_gate"] is True
    assert languages["pt-BR"]["planner"]["status"] == "verified_local"
    assert languages["pt-BR"]["phoneme_representation"]["status"] == ("verified_local")
    assert languages["pt-BR"]["fixtures"]["status"] == "verified_local"
    assert evidence["EV-PT-PLANNER"]["sha256"] == (
        "0353221b2f5d9363b2b4b4a845a5c7af1c11521797b4973981e1f34723c23ba9"
    )
    assert evidence["EV-PT-PLANNER-TESTS"]["sha256"] == (
        "88a318c7ebf05be94c67bcfc539d201be47cb0364220775ac2fe2ccae2e08e74"
    )
    assert evidence["EV-PT-PLANNER-CHECKER"]["sha256"] == (
        "8bd55cfade0864ac4950e6e798b385dd46bc89b994330525c4fb72a372ffda08"
    )

    for language_id in ("fr-FR", "it-IT"):
        language = languages[language_id]
        assert language["gate"]["status"] == "missing"
        assert language["voices"]["status"] == "declared_unverified"
        assert language["voices"]["local_pack_verification"] == (
            "no_declared_packs_verified"
        )
        assert language["estimated_implementation_difficulty"] == "high"


def test_matrix_makes_no_support_acoustic_or_perceptual_claim() -> None:
    matrix = _load("capability-matrix.json")
    assert matrix["overall_status"] == "architecture_assessment_only_disabled"
    assert matrix["claim_limits"] == {
        "language_support_claimed": False,
        "acoustic_quality_claimed": False,
        "perceptual_effect_claimed": False,
        "production_routing_changed": False,
        "api_calls_made": 0,
        "audio_renders_made": 0,
    }
    assert all(
        language["acoustic_validation"]["status"] == "missing"
        for language in matrix["language_capabilities"]
    )
    assert all(
        language["listener_profile_evidence"]["status"]
        in {"missing", "requires_new_evidence"}
        for language in matrix["language_capabilities"]
    )
