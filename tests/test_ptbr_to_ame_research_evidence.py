from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "artifacts" / "research" / "20260717-ptbr-to-ame-listener-evidence-v1"


def _load(name: str) -> dict[str, Any]:
    return json.loads((RUN_DIR / name).read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_evidence_chain_keeps_perception_production_and_inference_separate() -> None:
    evidence = _load("evidence.json")

    assert evidence["schema_version"] == 1
    assert evidence["decision"]["status"] == "research-only-disabled"
    assert evidence["decision"]["production_approved_mappings"] == []
    assert set(evidence["evidence_layers"]) == {
        "documented_perceptual_findings",
        "production_or_acoustic_findings",
        "derived_engineering_approximations",
        "unsupported_intuitions",
    }

    sources = {source["id"]: source for source in evidence["sources"]}
    assert sources["D1"]["evidence_class"] == "direct-target-listener-perception"
    assert sources["P1"]["evidence_class"] == "production-or-acoustic"
    assert sources["P2"]["evidence_class"] == "reverse-direction-accent-perception"
    assert sources["D1"]["doi"] == "10.3390/languages3030037"


def test_candidate_ranking_preserves_direct_percentages_and_rejections() -> None:
    evidence = _load("evidence.json")
    candidates = {
        candidate["id"]: candidate for candidate in evidence["candidate_contrasts"]
    }

    strongest = candidates["bp-open-mid-back-to-ce-low-back"]
    assert (strongest["source_phone"], strongest["listener_phone"]) == ("ɔ", "ɑ")
    assert strongest["observed_response"]["response_share"] == 0.72
    assert strongest["decision"] == "selected-for-disabled-acoustic-feasibility"

    secondary = candidates["bp-low-central-to-ce-near-front-low"]
    assert (secondary["source_phone"], secondary["listener_phone"]) == ("a", "æ")
    assert secondary["observed_response"]["response_share"] == 0.54

    ambiguous = candidates["bp-close-mid-front-to-ce-near-high-front"]
    assert ambiguous["decision"] == "hold-ambiguous"
    assert ambiguous["observed_response"]["primary_response_share"] == 0.34
    assert ambiguous["observed_response"]["competing_response_phone"] == "ɛ"
    assert ambiguous["observed_response"]["competing_response_share"] == 0.31

    rejected = candidates["bp-close-mid-back-to-ce-high-back"]
    assert (rejected["source_phone"], rejected["listener_phone"]) == ("o", "u")
    assert rejected["decision"] == "rejected"


def test_research_profile_is_inert_and_bound_to_frozen_evidence() -> None:
    profile = _load("research-profile.json")
    evidence_path = RUN_DIR / profile["evidence"]["artifact"]

    assert profile["status"] == "disabled-research-only"
    assert profile["enabled"] is False
    assert profile["production_approved"] is False
    assert profile["feature_flag"] == {
        "name": "RECIPROCAL_LISTENER_LENS_RESEARCH_ENABLED",
        "required_value_for_research_route": "true",
        "default": False,
        "public_enablement_authorized": False,
    }
    assert all(rule["enabled"] is False for rule in profile["rules"])
    assert all(
        rule["listener_validation_complete"] is False for rule in profile["rules"]
    )
    assert profile["plan_status"]["render_authorized_by_this_file"] is False
    assert profile["evidence"]["sha256"] == _sha256(evidence_path)


def test_json_schema_contract_covers_decision_critical_sections() -> None:
    schema = _load("evidence.schema.json")

    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["additionalProperties"] is False
    assert {
        "decision",
        "sources",
        "evidence_layers",
        "candidate_contrasts",
        "evidence_boundary",
    }.issubset(schema["required"])
    assert (
        schema["properties"]["decision"]["properties"]["production_approved_mappings"][
            "maxItems"
        ]
        == 0
    )
