from __future__ import annotations

import hashlib
import json

from earshift_bakeoff.config import ROOT, stable_json


MANIFEST = (
    ROOT
    / "artifacts"
    / "product-matrix"
    / "20260717-bilingual-product-v8-vowel-manifest"
    / "manifest.json"
)


def _semantic_hash(payload: dict) -> str:
    semantic = dict(payload)
    semantic.pop("record_sha256", None)
    return hashlib.sha256(stable_json(semantic).encode("utf-8")).hexdigest()


def test_all_240_v8_vowel_slots_are_structurally_frozen() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    slots = manifest["slots"]

    assert manifest["record_sha256"] == _semantic_hash(manifest)
    assert manifest["classification"] == (
        "all_v8_vowel_slots_frozen_with_stress_context"
    )
    assert manifest["logical_slot_count"] == manifest["pass_count"] == 240
    assert manifest["fail_count"] == 0
    assert manifest["stress_context_slot_count"] == 240
    assert len({row["logical_slot_id"] for row in slots}) == 240
    assert len({row["cell_id"] for row in slots}) == 80
    assert all(row["v8_active_changed_rule_ids"] == [row["rule_id"]] for row in slots)
    assert all(row["v8_plan_sha256"] != row["source_v7_plan_sha256"] for row in slots)
    assert all(
        set(row["vowel_unit_columns"]).issubset(row["vowel_state_columns"])
        for row in slots
    )
    assert all(row["stress_context_added"] is True for row in slots)


def test_v8_manifest_keeps_every_product_cell_disabled() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    assert manifest["api_calls_made"] == 0
    assert manifest["audio_renders_made"] == 0
    assert manifest["production_enabled"] is False
    assert all(row["product_enabled"] is False for row in manifest["slots"])
