from __future__ import annotations

import json

from earshift_bakeoff.config import ROOT
from earshift_bakeoff.util import sha256_file


PROTOCOL = ROOT / "rules" / "bilingual-vowel-full-context-screen-v1.json"
WORD_RESULT = (
    ROOT
    / "artifacts"
    / "product-matrix"
    / "20260717-bilingual-vowel-word-context-screen-v1"
    / "results.json"
)


def _load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_full_context_protocol_selects_exactly_the_remaining_parent_failures() -> None:
    protocol = _load(PROTOCOL)
    word = _load(WORD_RESULT)
    expected = sorted(
        row["cell_id"]
        for row in word["cell_summaries"]
        if row["candidate_classification"] == "fail"
    )

    assert protocol["status"] == "frozen_before_first_full_context_matrix_render"
    assert protocol["scope"]["cell_ids_in_order"] == expected
    assert len(expected) == protocol["scope"]["voice_rule_cell_count"] == 13
    assert protocol["scope"]["logical_slot_count"] == 39
    assert protocol["parent_bindings"]["word_result_sha256"] == sha256_file(WORD_RESULT)


def test_full_context_protocol_separates_state_from_neutral_excitation() -> None:
    candidate = _load(PROTOCOL)["candidate_intervention"]

    assert candidate["text_state"] == (
        "decode every column of the complete lens-conditioned text-encoder state"
    )
    assert "neutral" in candidate["duration_and_alignment"]
    assert "neutral-conditioned" in candidate["f0_and_noise"]
    assert candidate["candidate_order"] == ["complete_context_state_neutral_excitation"]
    assert candidate["replacement_slots_allowed"] is False
    assert candidate["selective_rerender_allowed"] is False
    assert candidate["api_calls_allowed"] == 0


def test_full_context_protocol_is_hash_bound_and_nonpromotional() -> None:
    protocol = _load(PROTOCOL)

    assert protocol["aggregation_policy"]["product_promotion_allowed"] is False
    assert protocol["production_enabled"] is False
    for binding in protocol["source_bindings"]:
        assert sha256_file(ROOT / binding["path"]) == binding["sha256"]
