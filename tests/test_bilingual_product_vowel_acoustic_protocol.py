from __future__ import annotations

import json

from earshift_bakeoff.config import ROOT
from earshift_bakeoff.util import sha256_file


PROTOCOL = ROOT / "rules" / "bilingual-product-vowel-acoustic-screen-v1.json"
MANIFEST = (
    ROOT
    / "artifacts"
    / "product-matrix"
    / "20260717-bilingual-product-isolated-acoustic-manifest-v1"
    / "manifest.json"
)
ISOLATED_AUDIO = (
    ROOT
    / "artifacts"
    / "product-matrix"
    / "20260717-bilingual-product-isolated-audio-screen-v1"
    / "results.json"
)


def _load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_vowel_protocol_covers_every_isolated_changed_vowel_slot() -> None:
    protocol = _load(PROTOCOL)
    manifest = _load(MANIFEST)
    vowel_slots = [slot for slot in manifest["slots"] if slot["family"] == "vowel"]

    assert protocol["status"] == (
        "frozen_before_first_vowel_anchor_render_or_measurement"
    )
    assert len(vowel_slots) == protocol["scope"]["logical_slot_count"] == 240
    assert len({slot["cell_id"] for slot in vowel_slots}) == 80
    assert len({slot["rule_id"] for slot in vowel_slots}) == 40
    assert all(
        slot["isolated_active_changed_rule_ids"] == [slot["rule_id"]]
        for slot in vowel_slots
    )
    assert protocol["source_data_bindings"]["isolated_manifest_sha256"] == sha256_file(
        MANIFEST
    )
    assert protocol["source_data_bindings"][
        "isolated_audio_result_sha256"
    ] == sha256_file(ISOLATED_AUDIO)


def test_vowel_protocol_requires_cross_ceiling_cross_context_success() -> None:
    protocol = _load(PROTOCOL)
    gates = protocol["analysis_gates"]
    aggregation = protocol["aggregation_policy"]

    assert protocol["instrument"]["formant_ceilings_hz"] == [5000, 5500, 6000]
    assert gates["measurement_retention"]["all_four_conditions_required"] is True
    assert gates["analysis_family"]["exact_requires_exact_at_every_ceiling"] is True
    assert gates["analysis_family"]["one_ceiling_cannot_rescue_another"] is True
    assert (
        aggregation["cell_exact_requires_all_three_slots_and_four_occurrences_exact"]
        is True
    )
    assert aggregation["missing_or_excluded_measurement_fails_the_cell"] is True
    assert aggregation["product_promotion_allowed"] is False


def test_vowel_protocol_preserves_nasal_rhotic_and_no_rerender_limits() -> None:
    protocol = _load(PROTOCOL)
    anchors = protocol["natural_anchor_policy"]

    assert anchors["maximum_anchor_render_count"] == 480
    assert anchors["replacement_anchor_renders_allowed"] is False
    assert anchors["selective_rerender_allowed"] is False
    assert anchors["api_calls_allowed"] == 0
    assert "nasality-preservation gate" in protocol["claim_limits"]["nasal_vowel"]
    assert (
        protocol["analysis_gates"]["rhoticity_endpoint"][
            "base_vowel_and_rhoticity_components_must_both_pass"
        ]
        is True
    )
    assert protocol["production_enabled"] is False
    for binding in protocol["source_bindings"]:
        assert sha256_file(ROOT / binding["path"]) == binding["sha256"]
