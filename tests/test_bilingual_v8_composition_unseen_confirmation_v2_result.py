from __future__ import annotations

import hashlib
import json
from pathlib import Path
import wave

import pytest

from earshift_bakeoff.config import ROOT, stable_json
from earshift_bakeoff.util import sha256_file


RUN_DIR = (
    ROOT
    / "artifacts"
    / "product-matrix"
    / "20260718-bilingual-v8-composition-unseen-confirmation-v2"
)
RESULT = RUN_DIR / "results.json"
PROTOCOL = (
    ROOT / "rules" / "bilingual-v8-composition-unseen-confirmation-v2.json"
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _semantic_hash(payload: dict) -> str:
    semantic = dict(payload)
    semantic.pop("record_sha256", None)
    return hashlib.sha256(stable_json(semantic).encode("utf-8")).hexdigest()


def test_unseen_composition_result_is_complete_failed_and_nonpromotional() -> None:
    result = _load(RESULT)

    assert sha256_file(RESULT) == (
        "6afcd959d5fc95c2668d2232ce3b461db185c83b0f4d238b3969367ba84cf2a3"
    )
    assert result["record_sha256"] == _semantic_hash(result)
    assert result["record_sha256"] == (
        "4ee292c546702f4cb016eb8248f57c97b08fbd997561ebb47c89a45543c72a3b"
    )
    assert result["protocol_sha256"] == sha256_file(PROTOCOL)
    assert result["classification"] == (
        "unseen_v8_composition_automatic_failed_preserve_exact_result"
    )
    assert result["production_enabled"] is False
    assert result["api_calls_made"] == 0
    assert result["fixture_count"] == result["render_set_count"] == 3
    assert result["automatic_pass_count"] == 2
    assert result["selected_rule_occurrence_count"] == 16
    assert result["shared_natural_decoder_render_count"] == 18


def test_unseen_composition_result_preserves_exact_failure_mechanism() -> None:
    result = _load(RESULT)
    fixtures = {row["fixture_id"]: row for row in result["fixtures"]}

    assert {fixture_id: row["automatic_pass"] for fixture_id, row in fixtures.items()} == {
        "heart_unseen_continuous": False,
        "michael_unseen_repeated": True,
        "dora_unseen_phrase_final": True,
    }
    assert all(row["contract_pass"] for row in fixtures.values())
    assert all(row["execution_error"] is None for row in fixtures.values())
    assert all(row["render_integrity"]["integrity_pass"] for row in fixtures.values())
    assert all(
        row["render_integrity"]["neutral_identity_bit_exact"]
        and row["render_integrity"]["outside_splice_exact_neutral"]
        and row["render_integrity"]["full_weight_interior_exact_lens"]
        and row["render_integrity"]["boundary_metrics_pass"]
        and row["render_integrity"]["localization_fraction"] == 1.0
        for row in fixtures.values()
    )
    assert all(
        row["acoustic"]["identity_false_positive_count"] == 0
        for row in fixtures.values()
    )

    heart = {
        row["rule_id"]: row
        for row in fixtures["heart_unseen_continuous"]["acoustic"]["cells"]
    }
    michael = {
        row["rule_id"]: row
        for row in fixtures["michael_unseen_repeated"]["acoustic"]["cells"]
    }
    dora = {
        row["rule_id"]: row
        for row in fixtures["dora_unseen_phrase_final"]["acoustic"]["cells"]
    }
    assert heart["enpt.ah_a"]["classification"] == "exact_category_pass"
    assert heart["enpt.uh_u"]["classification"] == "fail"
    assert {row["rule_id"]: row["classification"] for row in michael.values()} == {
        "enpt.aa_a": "directional_only_pass",
        "enpt.ae_eh": "exact_category_pass",
        "enpt.ah_a": "exact_category_pass",
    }
    assert {row["rule_id"]: row["classification"] for row in dora.values()} == {
        "pten.final_e_i": "directional_only_pass",
        "pten.o_goat": "exact_category_pass",
    }

    failed = next(
        occurrence
        for occurrence in heart["enpt.uh_u"]["occurrences"]
        if not occurrence["aggregate"]["directional_pass"]
    )
    assert failed["occurrence_index"] == 3
    candidate = failed["candidate"]
    assert candidate["direction_gate_pass"] is True
    assert candidate["directional_movement_gate_pass"] is True
    assert candidate["lens_endpoint_gate_pass"] is True
    assert candidate["target_gain_gate_pass"] is False
    assert candidate["direction_cosine"] == pytest.approx(0.5863295192415352)
    assert candidate["controlled_movement_fraction_of_anchor"] == pytest.approx(
        1.254087513394471
    )
    assert candidate["lens_target_distance_scaled_rms"] == pytest.approx(
        0.6479526535965936
    )
    assert candidate["neutral_target_distance_scaled_rms"] == pytest.approx(
        0.6435894801609547
    )


def test_unseen_composition_audio_receipts_bind_every_retained_wav() -> None:
    result = _load(RESULT)

    for fixture in result["fixtures"]:
        for condition in ("neutral", "lens"):
            receipt = fixture["audio"][condition]
            path = RUN_DIR / receipt["relative_path"]
            assert sha256_file(path) == receipt["wav_sha256"]
            with wave.open(str(path), "rb") as handle:
                assert handle.getnchannels() == 1
                assert handle.getsampwidth() == 2
                assert handle.getframerate() == 24_000
                assert handle.getnframes() == receipt["sample_count"]
                pcm = handle.readframes(handle.getnframes())
            assert hashlib.sha256(pcm).hexdigest() == receipt["pcm_sha256"]
