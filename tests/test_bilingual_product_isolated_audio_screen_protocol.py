from __future__ import annotations

import json

from earshift_bakeoff.config import ROOT
from earshift_bakeoff.util import sha256_file


PROTOCOL = ROOT / "rules" / "bilingual-product-isolated-audio-screen-v1.json"
MANIFEST = (
    ROOT
    / "artifacts"
    / "product-matrix"
    / "20260717-bilingual-product-isolated-acoustic-manifest-v1"
    / "manifest.json"
)


def _load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_isolated_audio_protocol_binds_all_280_atomic_slots() -> None:
    protocol = _load(PROTOCOL)
    manifest = _load(MANIFEST)
    binding = protocol["isolated_manifest_binding"]

    assert protocol["status"] == "frozen_before_first_isolated_audio_render"
    assert binding["logical_slot_count"] == 280
    assert binding["sha256"] == sha256_file(MANIFEST)
    assert binding["record_sha256"] == manifest["record_sha256"]
    assert manifest["isolated_plan_pass_count"] == 280
    assert all(
        row["isolated_active_changed_rule_ids"] == [row["rule_id"]]
        for row in manifest["slots"]
    )


def test_isolated_audio_protocol_is_zero_api_nonpromotional_and_one_pass() -> None:
    protocol = _load(PROTOCOL)
    renderer = protocol["renderer_policy"]
    classification = protocol["classification_policy"]

    assert renderer["exactly_one_active_changed_rule_required"] is True
    assert renderer["common_rng_required"] is True
    assert renderer["render_every_slot_once"] is True
    assert renderer["replacement_slots_allowed"] is False
    assert renderer["selective_rerender_allowed"] is False
    assert renderer["api_calls_allowed"] == 0
    assert classification["product_promotion_allowed"] is False
    assert classification["human_qc"] == "not_generated_by_this_screen"
    assert protocol["production_enabled"] is False


def test_isolated_audio_protocol_binds_every_execution_source() -> None:
    protocol = _load(PROTOCOL)
    for binding in protocol["source_bindings"]:
        assert sha256_file(ROOT / binding["path"]) == binding["sha256"]
