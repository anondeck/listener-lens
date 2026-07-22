from __future__ import annotations

import hashlib
import json
from pathlib import Path

from earshift_bakeoff.config import ROOT, stable_json


AUDIT = (
    ROOT
    / "artifacts"
    / "product-matrix"
    / "20260717-bilingual-product-rule-isolation-audit-v1"
    / "results.json"
)
MANIFEST = (
    ROOT
    / "artifacts"
    / "product-matrix"
    / "20260717-bilingual-product-isolated-acoustic-manifest-v1"
    / "manifest.json"
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _semantic_hash(payload: dict) -> str:
    semantic = dict(payload)
    semantic.pop("record_sha256", None)
    return hashlib.sha256(stable_json(semantic).encode("utf-8")).hexdigest()


def test_original_rule_isolation_limit_is_preserved_without_reclassifying_audio() -> None:
    audit = _load(AUDIT)

    assert audit["record_sha256"] == _semantic_hash(audit)
    assert audit["classification"] == (
        "rule_isolation_incomplete_acoustic_fixtures_require_isolated_plans"
    )
    assert audit["slot_count"] == 280
    assert audit["isolated_slot_count"] == 192
    assert audit["coactivated_slot_count"] == 88
    assert audit["families"]["vowel"]["coactivated_count"] == 76
    assert audit["families"]["consonant"]["coactivated_count"] == 12
    assert audit["families"]["insertion"]["coactivated_count"] == 0
    assert audit["families"]["prosody"]["coactivated_count"] == 0
    assert audit["api_calls_made"] == 0
    assert audit["audio_renders_made"] == 0
    assert audit["production_enabled"] is False


def test_isolated_acoustic_manifest_has_exactly_one_rule_in_every_slot() -> None:
    manifest = _load(MANIFEST)
    slots = manifest["slots"]

    assert manifest["record_sha256"] == _semantic_hash(manifest)
    assert manifest["classification"] == "all_acoustic_slots_atomically_isolated"
    assert manifest["logical_slot_count"] == 280
    assert manifest["isolated_plan_pass_count"] == 280
    assert manifest["isolated_plan_fail_count"] == 0
    assert manifest["isolated_plan_gate_yield"] == 1.0
    assert len({row["logical_slot_id"] for row in slots}) == 280
    assert all(row["status"] == "pass" for row in slots)
    assert all(row["isolated_plan_gate_pass"] is True for row in slots)
    assert all(
        row["isolated_active_changed_rule_ids"] == [row["rule_id"]]
        for row in slots
    )
    assert all(row["product_enabled"] is False for row in slots)
    assert manifest["api_calls_made"] == 0
    assert manifest["audio_renders_made"] == 0
    assert manifest["production_enabled"] is False
