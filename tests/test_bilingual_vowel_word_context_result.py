from __future__ import annotations

from collections import Counter
import hashlib
import json

from earshift_bakeoff.config import ROOT, stable_json


RESULT = (
    ROOT
    / "artifacts"
    / "product-matrix"
    / "20260717-bilingual-vowel-word-context-screen-v1"
    / "results.json"
)


def _load() -> dict:
    return json.loads(RESULT.read_text(encoding="utf-8"))


def _semantic_hash(payload: dict) -> str:
    semantic = dict(payload)
    semantic.pop("record_sha256", None)
    return hashlib.sha256(stable_json(semantic).encode("utf-8")).hexdigest()


def test_word_context_result_is_complete_zero_api_and_nonpromotional() -> None:
    result = _load()

    assert result["record_sha256"] == _semantic_hash(result)
    assert result["classification"] == (
        "word_context_candidate_screen_complete_no_product_promotion"
    )
    assert result["candidate_cell_count"] == 16
    assert result["logical_slot_count"] == result["measured_slot_count"] == 48
    assert result["error_slot_count"] == 0
    assert result["api_calls_made"] == result["replacement_slots_used"] == 0
    assert result["production_enabled"] is False
    assert all(row["status"] == "measured" for row in result["outcomes"])
    assert all(row["product_enabled"] is False for row in result["cell_summaries"])


def test_word_context_result_records_three_rescues_without_hiding_failures() -> None:
    result = _load()
    cells = result["cell_summaries"]

    assert result["cell_classification_counts"] == {
        "exact_category_pass": 2,
        "directional_only_pass": 1,
        "fail": 13,
    }
    assert result["rescued_cell_count"] == 3
    assert result["automatic_human_qc_eligible_cell_count"] == 3
    rescued = {
        (row["voice_id"], row["rule_id"]): row["candidate_classification"]
        for row in cells
        if row["candidate_classification"] != "fail"
    }
    assert rescued == {
        ("af_heart", "enpt.back_unrounded_u"): "directional_only_pass",
        ("am_michael", "enpt.ih_i"): "exact_category_pass",
        ("pm_alex", "pten.front_rounded_face"): "exact_category_pass",
    }
    assert Counter(row["candidate_classification"] for row in cells)["fail"] == 13


def test_word_context_result_preserves_v8_neutral_and_universal_integrity() -> None:
    outcomes = _load()["outcomes"]

    assert all(row["neutral_pcm_reused_from_v8"] for row in outcomes)
    assert all(row["verification"]["integrity_pass"] for row in outcomes)
    assert all(row["verification"]["neutral_identity_bit_exact"] for row in outcomes)
    assert all(row["verification"]["outside_splice_exact_neutral"] for row in outcomes)
    assert all(
        row["verification"]["full_weight_interior_exact_lens"] for row in outcomes
    )


def test_word_context_result_keeps_repeated_context_as_primary_failure() -> None:
    outcomes = _load()["outcomes"]
    counts = {
        context: Counter(
            row["candidate_classification"]
            for row in outcomes
            if row["context"] == context
        )
        for context in {
            "phrase_medial_continuous_speech",
            "phrase_final_new_context",
            "repeated_multi_target",
        }
    }

    assert counts["phrase_final_new_context"]["fail"] == 1
    assert counts["phrase_medial_continuous_speech"]["fail"] == 4
    assert counts["repeated_multi_target"]["fail"] == 10
