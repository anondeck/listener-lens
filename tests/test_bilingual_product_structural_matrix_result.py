from __future__ import annotations

import hashlib
import json
from pathlib import Path

from earshift_bakeoff.bilingual_product_matrix import (
    load_bilingual_product_matrix,
)
from earshift_bakeoff.config import ROOT, stable_json


MANIFEST = (
    ROOT
    / "artifacts"
    / "product-matrix"
    / "20260717-bilingual-product-matrix-v1"
    / "manifest.json"
)
RESULT = (
    ROOT
    / "artifacts"
    / "product-matrix"
    / "20260717-bilingual-product-structural-matrix-v1"
    / "results.json"
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _semantic_hash(payload: dict) -> str:
    semantic = dict(payload)
    semantic.pop("record_sha256", None)
    return hashlib.sha256(stable_json(semantic).encode("utf-8")).hexdigest()


def test_matrix_manifest_and_structural_result_are_hash_bound() -> None:
    manifest = _load(MANIFEST)
    result = _load(RESULT)

    assert manifest["record_sha256"] == _semantic_hash(manifest)
    assert result["record_sha256"] == _semantic_hash(result)
    assert result["source_manifest_sha256"] == hashlib.sha256(
        stable_json(manifest["validation_manifest"]).encode("utf-8")
    ).hexdigest()


def test_every_changed_voice_rule_context_passed_the_structural_gates() -> None:
    matrix = load_bilingual_product_matrix()
    result = _load(RESULT)
    outcomes = result["outcomes"]

    assert result["classification"] == "all_structural_slots_pass"
    assert result["planner_slot_count"] == 280
    assert result["planner_pass_count"] == 280
    assert result["planner_fail_count"] == 0
    assert result["planner_gate_yield"] == 1.0
    assert result["api_calls_made"] == 0
    assert result["audio_renders_made"] == 0
    assert len({row["logical_slot_id"] for row in outcomes}) == 280
    assert all(row["status"] == "pass" for row in outcomes)
    assert all(row["target_rule_occurrence_count"] >= 1 for row in outcomes)
    assert all(row["matrix_product_ready"] is False for row in outcomes)

    expected_cells = {
        (cell.profile_id, cell.voice_id, cell.rule_id)
        for cell in matrix.cells
        if cell.changed
    }
    actual_cells = {
        (row["profile_id"], row["voice_id"], row["rule_id"])
        for row in outcomes
    }
    assert actual_cells == expected_cells
