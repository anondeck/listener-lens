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
    / "20260718-bilingual-v8-factorized-composition-v2"
)
RESULT = RUN_DIR / "results.json"
PROTOCOL = ROOT / "rules" / "bilingual-v8-factorized-composition-v2.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _semantic_hash(payload: dict) -> str:
    semantic = dict(payload)
    semantic.pop("record_sha256", None)
    return hashlib.sha256(stable_json(semantic).encode("utf-8")).hexdigest()


def test_factorized_v2_result_is_complete_immutable_and_nonpromotional() -> None:
    result = _load(RESULT)

    assert sha256_file(RESULT) == (
        "a565882cd566d63c0896c28da175e5bce89bd11790194d0c0be2229471155142"
    )
    assert result["record_sha256"] == _semantic_hash(result)
    assert result["record_sha256"] == (
        "edc4875cbf54189e90eabdb8b8c9b451c1f7f93e25ccf93b893cb74e70963679"
    )
    assert result["protocol_sha256"] == sha256_file(PROTOCOL)
    assert result["baseline_result_sha256"] == (
        "020f086da838c948312d3b88be0f59b09bf78e32a6b6e8f3b89f5cdf8e28d7d6"
    )
    assert result["classification"] == (
        "factorized_composition_incomplete_preserve_exact_failure"
    )
    assert result["production_enabled"] is False
    assert result["api_calls_made"] == 0
    assert result["fixture_count"] == result["render_set_count"] == 3
    assert result["automatic_pass_count"] == 2
    assert result["shared_natural_decoder_render_count"] == 18


def test_factorized_v2_preserves_two_passes_and_the_exact_dora_failure() -> None:
    result = _load(RESULT)
    fixtures = {row["fixture_id"]: row for row in result["fixtures"]}

    assert {fixture_id: row["automatic_pass"] for fixture_id, row in fixtures.items()} == {
        "heart_two_v8_rules": True,
        "michael_three_v8_rules": True,
        "dora_two_v8_rules": False,
    }
    assert all(row["contract_pass"] for row in fixtures.values())
    assert all(row["neutral_baseline_bit_exact"] for row in fixtures.values())
    assert all(row["lens_changed_from_v1"] for row in fixtures.values())
    assert all(row["factorization"]["contract_pass"] for row in fixtures.values())
    assert all(row["render_integrity"]["integrity_pass"] for row in fixtures.values())
    assert all(
        row["acoustic"]["identity_false_positive_count"] == 0
        for row in fixtures.values()
    )

    dora = {
        row["rule_id"]: row
        for row in fixtures["dora_two_v8_rules"]["acoustic"]["cells"]
    }
    assert dora["pten.final_e_i"]["classification"] == "directional_only_pass"
    assert dora["pten.o_goat"]["classification"] == "fail"
    failed = dora["pten.o_goat"]["occurrences"][0]["candidate"]
    assert failed["direction_cosine"] == pytest.approx(0.27508503987523025)
    assert failed["controlled_movement_fraction_of_anchor"] == pytest.approx(
        0.48618040796099915
    )
    assert failed["minimum_direction_cosine"] == 0.5


def test_factorized_v2_audio_receipts_bind_every_retained_wav() -> None:
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

