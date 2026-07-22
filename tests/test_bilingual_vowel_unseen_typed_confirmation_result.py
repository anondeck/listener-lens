from __future__ import annotations

from collections import Counter
import hashlib
import json

from earshift_bakeoff.config import ROOT, stable_json


RESULT = (
    ROOT
    / "artifacts"
    / "product-matrix"
    / "20260718-bilingual-vowel-unseen-typed-confirmation-v1"
    / "results.json"
)


def _load() -> dict:
    return json.loads(RESULT.read_text(encoding="utf-8"))


def _semantic_hash(payload: dict) -> str:
    semantic = dict(payload)
    semantic.pop("record_sha256", None)
    return hashlib.sha256(stable_json(semantic).encode("utf-8")).hexdigest()


def test_unseen_typed_confirmation_is_complete_integral_and_nonpromotional() -> None:
    result = _load()

    assert result["record_sha256"] == _semantic_hash(result)
    assert result["record_sha256"] == (
        "14add28fe961d14ba2541bb20dc143d26f7fb9e93d0aa149a3bdf2a0a0c328a4"
    )
    assert result["classification"] == (
        "unseen_typed_confirmation_complete_no_product_promotion"
    )
    assert result["production_enabled"] is False
    assert result["api_calls_made"] == 0
    assert result["oral_candidate_cell_count"] == 28
    assert result["logical_slot_count"] == 84
    assert result["target_occurrence_count"] == 112
    assert result["candidate_render_set_count"] == 174
    assert result["natural_anchor_decoder_render_count"] == 672
    assert result["retained_wav_count"] == 168
    assert all(row["product_enabled"] is False for row in result["cell_summaries"])


def test_unseen_typed_confirmation_records_eighteen_qc_candidates() -> None:
    result = _load()

    assert result["cell_classification_counts"] == {
        "exact_category_pass": 12,
        "directional_only_pass": 6,
        "fail": 7,
        "anchor_validation_fail": 3,
    }
    assert result["unseen_automatic_pass_count"] == 18
    assert result["blind_human_qc_queue_count"] == 18
    assert result["anchor_valid_cell_count"] == 25

    passes = {
        (row["voice_id"], row["rule_id"])
        for row in result["cell_summaries"]
        if row["unseen_automatic_pass"]
    }
    assert passes == {
        ("af_heart", "enpt.aa_a"),
        ("af_heart", "enpt.ae_eh"),
        ("af_heart", "enpt.ah_a"),
        ("af_heart", "enpt.goat_o"),
        ("af_heart", "enpt.nurse_eh"),
        ("af_heart", "enpt.uh_u"),
        ("am_michael", "enpt.aa_a"),
        ("am_michael", "enpt.ae_eh"),
        ("am_michael", "enpt.ah_a"),
        ("am_michael", "enpt.ih_i"),
        ("am_michael", "enpt.nurse_eh"),
        ("am_michael", "enpt.reduced_i_i"),
        ("am_michael", "enpt.reduced_schwa_a"),
        ("am_michael", "enpt.uh_u"),
        ("pf_dora", "pten.final_e_i"),
        ("pf_dora", "pten.o_goat"),
        ("pm_alex", "pten.ao_aa"),
        ("pm_alex", "pten.e_ih"),
    }
    assert Counter(voice for voice, _ in passes) == {
        "af_heart": 6,
        "am_michael": 8,
        "pf_dora": 2,
        "pm_alex": 2,
    }


def test_unseen_typed_confirmation_preserves_all_failures() -> None:
    result = _load()

    nonpasses = {
        (row["voice_id"], row["rule_id"]): row["replicated_anchor"][
            "classification"
        ]
        for row in result["cell_summaries"]
        if not row["unseen_automatic_pass"]
    }
    assert nonpasses == {
        ("af_heart", "enpt.ih_i"): "fail",
        ("af_heart", "enpt.reduced_i_i"): "fail",
        ("am_michael", "enpt.goat_o"): "fail",
        ("pf_dora", "pten.a_ae"): "anchor_validation_fail",
        ("pf_dora", "pten.e_ih"): "fail",
        ("pf_dora", "pten.final_a_schwa"): "fail",
        ("pm_alex", "pten.a_ae"): "anchor_validation_fail",
        ("pm_alex", "pten.final_a_schwa"): "anchor_validation_fail",
        ("pm_alex", "pten.final_e_i"): "fail",
        ("pm_alex", "pten.o_goat"): "fail",
    }


def test_unseen_typed_confirmation_has_clean_controls_and_integrity() -> None:
    result = _load()

    assert result["candidate_integrity_slot_pass_count"] == 84
    assert result["identity_negative_control_count"] == 112
    assert result["identity_negative_control_false_positive_count"] == 0
    assert result["measurement_excluded_occurrence_count"] == 0
    assert result["measurement_excluded_slot_count"] == 0
    assert result["adaptive_selection_strength_counts"] == {"strength-100": 24}
    assert all(row["candidate_integrity"]["integrity_pass"] for row in result["outcomes"])
    assert all(
        row["identity_negative_control_false_positive_count"] == 0
        for row in result["outcomes"]
    )
