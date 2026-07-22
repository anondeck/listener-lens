from __future__ import annotations

import json

from earshift_bakeoff.config import ROOT
from earshift_bakeoff.util import sha256_file


PROTOCOL = ROOT / "rules" / "bilingual-vowel-word-context-screen-v1.json"
V8_PROTOCOL = ROOT / "rules" / "bilingual-product-v8-vowel-acoustic-screen.json"
V8_RESULT = (
    ROOT
    / "artifacts"
    / "product-matrix"
    / "20260717-bilingual-product-v8-vowel-acoustic-screen"
    / "results.json"
)


def _load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_word_context_protocol_selects_only_anchor_adequate_v8_failures() -> None:
    protocol = _load(PROTOCOL)
    v8 = _load(V8_RESULT)
    cell_ids = set(protocol["scope"]["cell_ids_in_order"])

    assert protocol["status"] == "frozen_before_first_word_context_matrix_render"
    assert len(cell_ids) == protocol["scope"]["voice_rule_cell_count"] == 16
    assert protocol["scope"]["logical_slot_count"] == 48
    assert all(
        cell["classification"] == "fail"
        for cell in v8["cell_summaries"]
        if cell["cell_id"] in cell_ids
    )
    assert protocol["parent_bindings"]["v8_result_sha256"] == sha256_file(V8_RESULT)
    assert protocol["parent_bindings"]["v8_protocol_sha256"] == sha256_file(V8_PROTOCOL)


def test_word_context_protocol_inherits_v8_instrument_and_thresholds() -> None:
    protocol = _load(PROTOCOL)
    v8 = _load(V8_PROTOCOL)
    inherited = protocol["instrument_and_gates"]

    assert inherited["formant_ceilings_hz"] == v8["instrument"]["formant_ceilings_hz"]
    assert inherited["measurement_retention"] == v8["analysis_gates"][
        "measurement_retention"
    ] | {"all_four_conditions_required": True}
    inherited_base = inherited["base_vowel"]
    v8_base = v8["analysis_gates"]["base_vowel_endpoint"]
    assert (
        inherited_base["minimum_anchor_separation_bark_rms"]
        == v8_base["minimum_anchor_separation_bark_rms"]
    )
    assert (
        inherited_base["minimum_controlled_movement_bark_rms"]
        == v8_base["minimum_controlled_movement_bark_rms"]
    )
    assert (
        inherited_base["minimum_direction_cosine"]
        == v8_base["minimum_direction_cosine"]
    )


def test_word_context_protocol_is_zero_api_single_candidate_and_nonpromotional() -> (
    None
):
    protocol = _load(PROTOCOL)
    candidate = protocol["candidate_intervention"]

    assert candidate["candidate_order"] == ["target_word_state_plus_excitation"]
    assert candidate["replacement_slots_allowed"] is False
    assert candidate["selective_rerender_allowed"] is False
    assert candidate["api_calls_allowed"] == 0
    assert protocol["aggregation_policy"]["product_promotion_allowed"] is False
    assert protocol["production_enabled"] is False
    for binding in protocol["source_bindings"]:
        assert sha256_file(ROOT / binding["path"]) == binding["sha256"]
