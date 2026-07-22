from __future__ import annotations

import hashlib
import json

from earshift_bakeoff.config import ROOT, stable_json


RESULT = (
    ROOT
    / "artifacts"
    / "product-matrix"
    / "20260717-bilingual-vowel-spectral-category-screen-v1"
    / "results.json"
)


def _load() -> dict:
    return json.loads(RESULT.read_text(encoding="utf-8"))


def _semantic_hash(payload: dict) -> str:
    semantic = dict(payload)
    semantic.pop("record_sha256", None)
    return hashlib.sha256(stable_json(semantic).encode("utf-8")).hexdigest()


def test_spectral_result_is_complete_zero_api_and_nonpromotional() -> None:
    result = _load()

    assert result["record_sha256"] == _semantic_hash(result)
    assert result["classification"] == (
        "spectral_instrument_sanity_fail_no_product_promotion"
    )
    assert result["voice_rule_cell_count"] == 80
    assert result["occurrence_count"] == 320
    assert result["feature_extraction_error_count"] == 0
    assert result["api_calls_made"] == result["new_audio_renders_made"] == 0
    assert result["production_enabled"] is False
    assert all(row["product_enabled"] is False for row in result["cell_summaries"])


def test_spectral_instrument_fails_frozen_reference_sanity_gate() -> None:
    sanity = _load()["instrument_sanity"]

    assert sanity["frozen_v8_reference_pass_cell_count"] == 21
    assert sanity["minimum_reference_concordant_cell_count"] == 16
    assert sanity["reference_concordant_cell_count"] == 11
    assert sanity["pass"] is False
    assert sanity["identity_negative_control_count"] == 320
    assert sanity["identity_negative_control_false_positive_count"] == 0


def test_spectral_candidates_remain_descriptive_after_instrument_failure() -> None:
    result = _load()

    assert result["cell_classification_counts"] == {
        "anchor_validation_fail": 48,
        "directional_only_pass": 4,
        "exact_category_pass": 20,
        "fail": 8,
    }
    assert result["typed_core"]["cell_count"] == 36
    assert result["typed_core"]["spectral_anchor_validated_count"] == 14
    assert result["typed_core"]["spectral_candidate_pass_count"] == 8
    assert result["typed_core"]["new_spectral_candidate_count_against_frozen_v8"] == 6
    assert "does not rewrite" in result["typed_core"]["interpretation"]
