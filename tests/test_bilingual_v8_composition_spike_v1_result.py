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
    / "20260718-bilingual-v8-composition-spike-v1"
)
RESULT = RUN_DIR / "results.json"
PROTOCOL = ROOT / "rules" / "bilingual-v8-composition-spike-v1.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _semantic_hash(payload: dict) -> str:
    semantic = dict(payload)
    semantic.pop("record_sha256", None)
    return hashlib.sha256(stable_json(semantic).encode("utf-8")).hexdigest()


def test_v8_composition_result_is_complete_immutable_and_nonpromotional() -> None:
    result = _load(RESULT)

    assert sha256_file(RESULT) == (
        "020f086da838c948312d3b88be0f59b09bf78e32a6b6e8f3b89f5cdf8e28d7d6"
    )
    assert result["record_sha256"] == _semantic_hash(result)
    assert result["record_sha256"] == (
        "a43c659106c4cf0637af5e5d707c0342ad04e8731584bd9aa10743ad80a99058"
    )
    assert result["protocol_sha256"] == sha256_file(PROTOCOL)
    assert result["classification"] == (
        "exploratory_v8_composition_incomplete_preserve_exact_failure"
    )
    assert result["production_enabled"] is False
    assert result["api_calls_made"] == 0
    assert result["fixture_count"] == result["render_set_count"] == 3
    assert result["automatic_pass_count"] == 2
    assert result["shared_natural_decoder_render_count"] == 18


def test_v8_composition_result_preserves_the_exact_mixed_outcome() -> None:
    result = _load(RESULT)
    fixtures = {row["fixture_id"]: row for row in result["fixtures"]}

    assert {fixture_id: row["automatic_pass"] for fixture_id, row in fixtures.items()} == {
        "heart_two_v8_rules": True,
        "michael_three_v8_rules": True,
        "dora_two_v8_rules": False,
    }
    assert all(row["contract_pass"] for row in fixtures.values())
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
        for row in fixtures["heart_two_v8_rules"]["acoustic"]["cells"]
    }
    michael = {
        row["rule_id"]: row
        for row in fixtures["michael_three_v8_rules"]["acoustic"]["cells"]
    }
    dora = {
        row["rule_id"]: row
        for row in fixtures["dora_two_v8_rules"]["acoustic"]["cells"]
    }
    assert {rule_id: row["classification"] for rule_id, row in heart.items()} == {
        "enpt.ah_a": "exact_category_pass",
        "enpt.uh_u": "exact_category_pass",
    }
    assert {rule_id: row["classification"] for rule_id, row in michael.items()} == {
        "enpt.aa_a": "exact_category_pass",
        "enpt.ae_eh": "exact_category_pass",
        "enpt.ah_a": "exact_category_pass",
    }
    assert dora["pten.final_e_i"]["classification"] == "directional_only_pass"
    assert dora["pten.o_goat"]["classification"] == "fail"
    failed = dora["pten.o_goat"]["occurrences"][0]["candidate"]
    assert failed["controlled_movement_fraction_of_anchor"] == pytest.approx(
        0.48273369591260895
    )
    assert failed["direction_cosine"] == pytest.approx(0.28137707252934)
    assert failed["minimum_direction_cosine"] == 0.5


def test_v8_composition_audio_receipts_bind_every_retained_wav() -> None:
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

