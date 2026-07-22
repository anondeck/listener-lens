from __future__ import annotations

import json

from earshift_bakeoff.config import ROOT
from earshift_bakeoff.util import sha256_file


PROTOCOL = ROOT / "rules" / "bilingual-vowel-adaptive-strength-screen-v1.json"
FULL_RESULT = (
    ROOT
    / "artifacts"
    / "product-matrix"
    / "20260717-bilingual-vowel-full-context-screen-v1"
    / "results.json"
)


def _load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_adaptive_protocol_selects_all_remaining_full_context_failures() -> None:
    protocol = _load(PROTOCOL)
    result = _load(FULL_RESULT)
    expected = sorted(
        row["cell_id"]
        for row in result["cell_summaries"]
        if row["candidate_classification"] == "fail"
    )

    assert protocol["scope"]["cell_ids_in_order"] == expected
    assert len(expected) == protocol["scope"]["voice_rule_cell_count"] == 12
    assert protocol["scope"]["logical_slot_count"] == 36
    assert protocol["scope"]["candidate_render_set_count"] == 216
    assert protocol["parent_bindings"]["full_result_sha256"] == sha256_file(FULL_RESULT)


def test_adaptive_protocol_freezes_strength_order_and_selection() -> None:
    candidate = _load(PROTOCOL)["candidate_intervention"]

    assert candidate["strength_order"] == [1.0, 0.75, 1.25, 0.5, 1.5, 2.0]
    assert "first exact-category" in candidate["occurrence_selection"]
    assert "first directional-only" in candidate["occurrence_selection"]
    assert candidate["selection_uses_listening"] is False
    assert candidate["selective_rerender_allowed"] is False
    assert candidate["replacement_slots_allowed"] is False
    assert candidate["api_calls_allowed"] == 0


def test_adaptive_protocol_is_hash_bound_and_nonpromotional() -> None:
    protocol = _load(PROTOCOL)

    assert protocol["production_enabled"] is False
    assert protocol["aggregation_policy"]["product_promotion_allowed"] is False
    assert protocol["instrument_and_gates"]["all_three_ceilings_required"] is True
    for binding in protocol["source_bindings"]:
        assert sha256_file(ROOT / binding["path"]) == binding["sha256"]
