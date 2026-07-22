from __future__ import annotations

import hashlib
import json

from earshift_bakeoff.config import ROOT, stable_json


RESULT = (
    ROOT
    / "artifacts"
    / "product-matrix"
    / "20260717-bilingual-product-vowel-acoustic-screen-v1"
    / "results.json"
)


def _semantic_hash(payload: dict) -> str:
    semantic = dict(payload)
    semantic.pop("record_sha256", None)
    return hashlib.sha256(stable_json(semantic).encode("utf-8")).hexdigest()


def test_vowel_v1_result_is_complete_and_remains_a_frozen_failure() -> None:
    result = json.loads(RESULT.read_text(encoding="utf-8"))

    assert result["record_sha256"] == _semantic_hash(result)
    assert result["classification"] == (
        "vowel_acoustic_screen_complete_no_product_promotion"
    )
    assert result["logical_slot_count"] == result["measured_slot_count"] == 240
    assert result["measurement_error_slot_count"] == 0
    assert result["voice_rule_cell_count"] == 80
    assert result["cell_classification_counts"] == {
        "exact_category_pass": 0,
        "directional_only_pass": 0,
        "fail": 80,
    }
    assert result["automatic_human_qc_eligible_cell_count"] == 0
    assert result["natural_anchor_render_count"] == 480
    assert result["api_calls_made"] == 0
    assert result["replacement_renders_used"] == 0
    assert result["production_enabled"] is False


def test_vowel_v1_failure_is_dominated_by_measurement_exclusion() -> None:
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    outcomes = result["outcomes"]
    analyses = [
        classification
        for outcome in outcomes
        for ceiling in outcome["analysis_by_formant_ceiling"]
        for classification in ceiling["occurrence_classifications"]
    ]
    measurements = [
        measurement
        for outcome in outcomes
        for ceiling in outcome["analysis_by_formant_ceiling"]
        for condition in ceiling["measurements"].values()
        for measurement in condition
    ]

    assert len(analyses) == 960
    assert (
        sum(row["classification"] == "measurement_exclusion" for row in analyses) == 790
    )
    assert len(measurements) == 3840
    assert sum(row["retention_pass"] for row in measurements) == 3807
    assert sum(row["plausibility_pass"] for row in measurements) == 972
    assert sum(row["measurable"] for row in measurements) == 945
    assert all(
        outcome["source_anchor_pcm_identical_to_controlled_neutral"]
        for outcome in outcomes
    )
    assert all(outcome["anchor_integrity_pass"] for outcome in outcomes)


def test_vowel_v1_protocol_does_not_allow_post_hoc_rescue() -> None:
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    protocol = result["protocol"]

    assert protocol["aggregation_policy"]["product_promotion_allowed"] is False
    assert (
        protocol["analysis_gates"]["analysis_family"][
            "one_ceiling_cannot_rescue_another"
        ]
        is True
    )
    assert (
        protocol["natural_anchor_policy"]["replacement_anchor_renders_allowed"] is False
    )
    assert result["cell_summaries"]
    assert all(row["product_enabled"] is False for row in result["cell_summaries"])
