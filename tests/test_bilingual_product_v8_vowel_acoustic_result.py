from __future__ import annotations

from collections import Counter
import hashlib
import json

from earshift_bakeoff.config import ROOT, stable_json


RESULT = (
    ROOT
    / "artifacts"
    / "product-matrix"
    / "20260717-bilingual-product-v8-vowel-acoustic-screen"
    / "results.json"
)


def _load() -> dict:
    return json.loads(RESULT.read_text(encoding="utf-8"))


def _semantic_hash(payload: dict) -> str:
    semantic = dict(payload)
    semantic.pop("record_sha256", None)
    return hashlib.sha256(stable_json(semantic).encode("utf-8")).hexdigest()


def test_v8_vowel_result_is_complete_zero_api_and_not_promoted() -> None:
    result = _load()

    assert result["record_sha256"] == _semantic_hash(result)
    assert result["classification"] == (
        "v8_vowel_acoustic_screen_complete_no_product_promotion"
    )
    assert result["logical_slot_count"] == result["measured_slot_count"] == 240
    assert result["error_slot_count"] == 0
    assert result["voice_rule_cell_count"] == 80
    assert result["v8_render_set_count"] == 240
    assert result["reused_anchor_pair_count"] == 240
    assert result["api_calls_made"] == result["replacement_slots_used"] == 0
    assert result["production_enabled"] is False
    assert all(row["status"] == "measured" for row in result["outcomes"])
    assert all(row["product_enabled"] is False for row in result["cell_summaries"])


def test_v8_vowel_result_preserves_integrity_and_frozen_anchor_bindings() -> None:
    outcomes = _load()["outcomes"]

    assert all(row["neutral_pcm_unchanged_from_v7"] for row in outcomes)
    assert all(row["v1_natural_anchor_reuse_pass"] for row in outcomes)
    assert all(row["verification"]["integrity_pass"] for row in outcomes)
    assert all(row["verification"]["neutral_identity_bit_exact"] for row in outcomes)
    assert all(row["verification"]["outside_splice_exact_neutral"] for row in outcomes)
    assert all(
        row["verification"]["full_weight_interior_exact_lens"] for row in outcomes
    )
    assert all(row["verification"]["boundary_metrics_pass"] for row in outcomes)
    assert all(row["verification"]["localization_pass"] for row in outcomes)
    assert all(
        set(row["vowel_unit_columns"]) < set(row["vowel_state_columns"])
        for row in outcomes
    )


def test_v8_vowel_result_records_mixed_voice_local_acoustic_evidence() -> None:
    result = _load()
    cells = result["cell_summaries"]

    assert result["cell_classification_counts"] == {
        "exact_category_pass": 12,
        "directional_only_pass": 9,
        "fail": 59,
    }
    assert result["automatic_human_qc_eligible_cell_count"] == 19
    expected = {
        "af_heart": (
            {"exact_category_pass": 4, "directional_only_pass": 3, "fail": 12},
            7,
        ),
        "am_michael": (
            {"exact_category_pass": 3, "directional_only_pass": 2, "fail": 14},
            5,
        ),
        "pm_alex": (
            {"exact_category_pass": 2, "directional_only_pass": 4, "fail": 15},
            4,
        ),
        "pf_dora": ({"exact_category_pass": 3, "fail": 18}, 3),
    }
    for voice_id, (counts, eligible) in expected.items():
        rows = [row for row in cells if row["voice_id"] == voice_id]
        assert dict(Counter(row["classification"] for row in rows)) == counts
        assert sum(row["automatic_human_qc_eligible"] for row in rows) == eligible

    nasal_passes = [
        row
        for row in cells
        if row["classification"] != "fail"
        and "nasality_unvalidated" in row["claim_limit"]
    ]
    assert len(nasal_passes) == 2
    assert not any(row["automatic_human_qc_eligible"] for row in nasal_passes)


def test_v8_vowel_result_keeps_context_failures_visible() -> None:
    result = _load()
    analyses = [
        classification
        for outcome in result["outcomes"]
        for ceiling in outcome["analysis_by_formant_ceiling"]
        for classification in ceiling["occurrence_classifications"]
    ]
    contexts = {
        context: Counter(
            row["classification"]
            for row in result["outcomes"]
            if row["context"] == context
        )
        for context in {
            "phrase_medial_continuous_speech",
            "phrase_final_new_context",
            "repeated_multi_target",
        }
    }

    assert len(analyses) == 960
    assert Counter(row["classification"] for row in analyses) == {
        "exact_category_pass": 627,
        "directional_only_pass": 63,
        "fail": 219,
        "measurement_exclusion": 51,
    }
    assert contexts["repeated_multi_target"]["fail"] == 50
    assert contexts["phrase_medial_continuous_speech"]["fail"] == 34
    assert contexts["phrase_final_new_context"]["fail"] == 23
