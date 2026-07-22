from __future__ import annotations

import hashlib
import json

from earshift_bakeoff.config import ROOT, stable_json


RESULT = (
    ROOT
    / "artifacts"
    / "product-matrix"
    / "20260717-bilingual-vowel-adaptive-strength-screen-v1"
    / "results.json"
)


def _load() -> dict:
    return json.loads(RESULT.read_text(encoding="utf-8"))


def _semantic_hash(payload: dict) -> str:
    semantic = dict(payload)
    semantic.pop("record_sha256", None)
    return hashlib.sha256(stable_json(semantic).encode("utf-8")).hexdigest()


def test_adaptive_result_is_complete_zero_api_and_nonpromotional() -> None:
    result = _load()

    assert result["record_sha256"] == _semantic_hash(result)
    assert result["classification"] == (
        "adaptive_strength_screen_complete_no_product_promotion"
    )
    assert result["candidate_cell_count"] == 12
    assert result["logical_slot_count"] == result["measured_slot_count"] == 36
    assert result["candidate_render_set_count"] == 216
    assert result["error_slot_count"] == 0
    assert result["api_calls_made"] == result["replacement_slots_used"] == 0
    assert result["production_enabled"] is False
    assert all(row["status"] == "measured" for row in result["outcomes"])
    assert all(row["product_enabled"] is False for row in result["cell_summaries"])


def test_adaptive_result_records_six_voice_local_rescues() -> None:
    result = _load()

    assert result["cell_classification_counts"] == {
        "exact_category_pass": 3,
        "directional_only_pass": 3,
        "fail": 6,
    }
    assert result["rescued_cell_count"] == 6
    assert result["automatic_human_qc_eligible_cell_count"] == 6
    rescued = {
        (row["voice_id"], row["rule_id"]): row["candidate_classification"]
        for row in result["cell_summaries"]
        if row["candidate_classification"] != "fail"
    }
    assert rescued == {
        ("af_heart", "enpt.aa_a"): "exact_category_pass",
        ("af_heart", "enpt.ae_eh"): "exact_category_pass",
        ("af_heart", "enpt.back_mid_o"): "directional_only_pass",
        ("af_heart", "enpt.nurse_eh"): "exact_category_pass",
        ("pm_alex", "pten.ao_aa"): "directional_only_pass",
        ("pf_dora", "pten.central_high_kit"): "directional_only_pass",
    }


def test_every_strength_preserves_neutral_and_universal_integrity() -> None:
    candidates = [
        candidate
        for outcome in _load()["outcomes"]
        for candidate in outcome["strength_candidates"]
    ]

    assert len(candidates) == 216
    assert all(candidate["neutral_pcm_reused_from_v8"] for candidate in candidates)
    assert all(candidate["verification"]["integrity_pass"] for candidate in candidates)
    assert all(
        candidate["verification"]["outside_splice_exact_neutral"]
        for candidate in candidates
    )
    assert all(
        candidate["verification"]["full_weight_interior_exact_lens"]
        for candidate in candidates
    )


def test_adaptive_composites_are_remeasured_and_unresolved_stay_failed() -> None:
    result = _load()
    complete = [row for row in result["outcomes"] if row["selection_complete"]]
    unresolved = [row for row in result["outcomes"] if not row["selection_complete"]]

    assert len(complete) == 30
    assert len(unresolved) == result["unresolved_occurrence_count"] == 6
    assert all(row["adaptive_verification"]["integrity_pass"] for row in complete)
    assert all(
        row["adaptive_verification"]["localization_fraction"] == 1.0 for row in complete
    )
    assert all(row["candidate_classification"] == "fail" for row in unresolved)
    assert all(row["adaptive_audio"] is None for row in unresolved)
    assert sum(result["selection_strength_counts"].values()) == 42
    assert result["selection_strength_counts"]["strength-100"] == 30
    assert (
        sum(
            count
            for label, count in result["selection_strength_counts"].items()
            if label != "strength-100"
        )
        == 12
    )
