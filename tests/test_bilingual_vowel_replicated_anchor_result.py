from __future__ import annotations

import hashlib
import json

from earshift_bakeoff.config import ROOT, stable_json


RUN_DIR = (
    ROOT
    / "artifacts"
    / "product-matrix"
    / "20260717-bilingual-vowel-replicated-anchor-calibration-v1"
)
RESULT = RUN_DIR / "results.json"
ERRATUM = RUN_DIR / "accounting-erratum.json"
BASELINE_SEED = "20260716"


def _load(path=RESULT) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _semantic_hash(payload: dict) -> str:
    semantic = dict(payload)
    semantic.pop("record_sha256", None)
    return hashlib.sha256(stable_json(semantic).encode("utf-8")).hexdigest()


def test_replicated_anchor_result_passes_without_product_promotion() -> None:
    result = _load()

    assert result["record_sha256"] == _semantic_hash(result)
    assert result["classification"] == (
        "context_matched_anchor_instrument_pass_no_failure_cells_evaluated"
    )
    assert result["instrument_pass"] is True
    assert result["production_enabled"] is False
    assert result["api_calls_made"] == 0
    assert result["retained_new_wav_count"] == 0
    assert result["new_local_decoder_render_count"] == 864
    assert all(row["product_enabled"] is False for row in result["cell_summaries"])


def test_replicated_anchor_result_passes_every_frozen_sanity_gate() -> None:
    result = _load()

    assert result["baseline_parity_condition_count"] == 216
    assert result["baseline_parity_pass"] is True
    assert result["reference_candidate_cell_count"] == 12
    assert result["reference_concordant_cell_count"] == 12
    assert result["minimum_reference_concordant_cell_count"] == 10
    assert result["identity_negative_control_count"] == 48
    assert result["identity_negative_control_false_positive_count"] == 0
    assert len(result["reference_concordant_cell_ids"]) == 12
    assert result["nonconcordant_reference_cell_ids"] == []


def test_replicated_anchor_result_keeps_failure_cells_out_of_calibration() -> None:
    result = _load()
    references = [row for row in result["cell_summaries"] if row["reference_rung"]]
    failures = [row for row in result["cell_summaries"] if not row["reference_rung"]]

    assert len(references) == 12
    assert all(
        row["replicated_anchor"]["candidate_occurrence_count"] == 4
        for row in references
    )
    assert len(failures) == 24
    assert all(
        row["replicated_anchor"]["candidate_occurrence_count"] == 0 for row in failures
    )
    assert all(
        row["replicated_anchor"]["classification"] == "anchors_only" for row in failures
    )
    invalid = [
        row["cell_id"]
        for row in result["cell_summaries"]
        if not row["replicated_anchor"]["all_anchor_occurrences_valid"]
    ]
    assert invalid == ["pt-BR-to-en-US-listener-v2::pm_alex::vowel::pten.nasal_a_schwa"]
    assert result["anchor_valid_core_cell_count"] == 35


def test_accounting_erratum_separates_training_from_baseline_integrity() -> None:
    result = _load()
    erratum = _load(ERRATUM)
    training = [
        receipt
        for slot in result["slot_receipts"]
        for side in ("source_seed_audio", "target_seed_audio")
        for seed, receipt in slot[side].items()
        if seed != BASELINE_SEED
    ]
    baseline = [
        receipt
        for slot in result["slot_receipts"]
        for side in ("source_seed_audio", "target_seed_audio")
        for seed, receipt in slot[side].items()
        if seed == BASELINE_SEED
    ]

    assert len(training) == erratum["correct_training_only_condition_count"] == 648
    assert sum(row["finite_nonempty_unclipped"] for row in training) == 648
    assert len(baseline) == erratum["baseline_parity_condition_count"] == 216
    assert sum(row["finite_nonempty_unclipped"] for row in baseline) == 216
    assert result["training_seed_integrity_pass_count"] == 864
    assert erratum["classification_effect"] == "none"
    assert erratum["result_rewritten"] is False
