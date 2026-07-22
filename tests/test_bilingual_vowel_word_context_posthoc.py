from __future__ import annotations

import json

from earshift_bakeoff.config import ROOT
from earshift_bakeoff.util import sha256_file


RECORD = (
    ROOT
    / "artifacts"
    / "product-matrix"
    / "20260717-bilingual-vowel-word-context-screen-v1"
    / "posthoc-mechanism-correction.json"
)


def test_posthoc_correction_binds_frozen_records_without_reclassification() -> None:
    record = json.loads(RECORD.read_text(encoding="utf-8"))
    bound = record["bound_records"]

    assert record["status"] == "posthoc_interpretation_correction_results_unchanged"
    assert bound["v8_result_sha256"] == sha256_file(
        ROOT
        / "artifacts"
        / "product-matrix"
        / "20260717-bilingual-product-v8-vowel-acoustic-screen"
        / "results.json"
    )
    assert bound["word_context_result_sha256"] == sha256_file(
        RECORD.parent / "results.json"
    )
    assert record["unchanged_results"]["reclassification_allowed"] is False
    assert record["unchanged_results"]["product_promotion_allowed"] is False


def test_posthoc_correction_names_actual_state_and_excitation_delta() -> None:
    record = json.loads(RECORD.read_text(encoding="utf-8"))
    corrected = record["corrected_interpretation"]

    assert "complete target-word lens text state" in corrected["v8"]
    assert "true experimental delta" in corrected["word_context_candidate"]
    assert "target-conditioned word excitation" in corrected["word_context_candidate"]
    assert "complete contextual lens text state" in corrected["next_hypothesis"]
