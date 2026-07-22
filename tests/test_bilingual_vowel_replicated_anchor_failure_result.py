from __future__ import annotations

from collections import Counter
import hashlib
import json

from earshift_bakeoff.config import ROOT, stable_json


RESULT = (
    ROOT
    / "artifacts"
    / "product-matrix"
    / "20260718-bilingual-vowel-replicated-anchor-failure-screen-v1"
    / "results.json"
)


def _load() -> dict:
    return json.loads(RESULT.read_text(encoding="utf-8"))


def _semantic_hash(payload: dict) -> str:
    semantic = dict(payload)
    semantic.pop("record_sha256", None)
    return hashlib.sha256(stable_json(semantic).encode("utf-8")).hexdigest()


def test_failure_screen_is_complete_reproduced_and_nonpromotional() -> None:
    result = _load()

    assert result["record_sha256"] == _semantic_hash(result)
    assert result["classification"] == (
        "eligible_failure_cells_evaluated_no_product_promotion"
    )
    assert result["production_enabled"] is False
    assert result["api_calls_made"] == 0
    assert result["new_candidate_decoder_renders_made"] == 0
    assert result["anchor_reproduction_decoder_render_count"] == 864
    assert result["anchor_reproduction_pcm_pass_count"] == 864
    assert result["anchor_reproduction_feature_count"] == 864
    assert result["anchor_reproduction_feature_pass_count"] == 864
    assert result["eligible_failure_cell_count"] == 23
    assert result["eligible_failure_slot_count"] == 69
    assert result["eligible_failure_occurrence_count"] == 92
    assert all(row["product_enabled"] is False for row in result["cell_summaries"])


def test_failure_screen_records_nineteen_context_matched_candidates() -> None:
    result = _load()

    assert result["cell_classification_counts"] == {
        "exact_category_pass": 15,
        "directional_only_pass": 4,
        "fail": 4,
    }
    assert result["new_context_matched_candidate_pass_count"] == 19
    assert result["new_oral_blind_qc_queue_count"] == 17
    assert result["nasal_candidate_pass_pending_nasality_count"] == 2
    assert Counter(row["voice_id"] for row in result["cell_summaries"]) == {
        "af_heart": 5,
        "am_michael": 7,
        "pf_dora": 7,
        "pm_alex": 4,
    }

    failures = {
        (row["voice_id"], row["rule_id"])
        for row in result["cell_summaries"]
        if row["replicated_anchor"]["classification"] == "fail"
    }
    assert failures == {
        ("af_heart", "enpt.reduced_schwa_a"),
        ("af_heart", "enpt.schwa_reduced_a"),
        ("am_michael", "enpt.schwa_reduced_a"),
        ("pf_dora", "pten.ao_aa"),
    }


def test_failure_screen_keeps_invalid_anchor_and_nasal_claims_out_of_qc() -> None:
    result = _load()

    assert result["ineligible_failure_cell"]["cell_id"] == (
        "pt-BR-to-en-US-listener-v2::pm_alex::vowel::pten.nasal_a_schwa"
    )
    nasal_passes = [
        row
        for row in result["cell_summaries"]
        if row["nasality_gate_required"]
        and row["replicated_anchor"]["directional_pass"]
    ]
    assert len(nasal_passes) == 2
    assert all(not row["automatic_blind_qc_eligible"] for row in nasal_passes)


def test_adaptive_candidates_are_integral_and_remeasured() -> None:
    result = _load()
    adaptive = [
        row
        for row in result["outcomes"]
        if row["candidate_rung"] == "adaptive_strength"
    ]

    assert len(adaptive) == 6
    assert result["retained_adaptive_composite_wav_count"] == 6
    assert all(row["candidate_integrity"]["integrity_pass"] for row in adaptive)
    assert all(
        occurrence["composite_remeasurement"]["classification"]
        == occurrence["aggregate"]["classification"]
        for row in adaptive
        for occurrence in row["occurrences"]
    )
    assert sum(result["adaptive_selection_strength_counts"].values()) == 8
