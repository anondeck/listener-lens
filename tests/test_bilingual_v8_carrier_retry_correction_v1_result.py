from __future__ import annotations

import hashlib
import json
import wave

from earshift_bakeoff.config import ROOT, stable_json
from earshift_bakeoff.util import sha256_file


RUN_DIR = (
    ROOT
    / "artifacts"
    / "product-matrix"
    / "20260718-bilingual-v8-carrier-retry-correction-v1"
)
RESULT = RUN_DIR / "results.json"
PROTOCOL = ROOT / "rules" / "bilingual-v8-carrier-retry-correction-v1.json"


def _result() -> dict:
    return json.loads(RESULT.read_text(encoding="utf-8"))


def _semantic_hash(payload: dict) -> str:
    semantic = dict(payload)
    semantic.pop("record_sha256", None)
    return hashlib.sha256(stable_json(semantic).encode("utf-8")).hexdigest()


def test_carrier_retry_result_is_first_round_pass_and_nonpromotional() -> None:
    result = _result()

    assert sha256_file(RESULT) == (
        "e1aebcac898e70cccf9bb6d77ae8800b0338b4688abddf75b5963c427bd88ee7"
    )
    assert result["record_sha256"] == _semantic_hash(result)
    assert result["record_sha256"] == (
        "d032d8466663b19bde8640739741ca00f7221b7723b78e8e35cb39a22e936ea4"
    )
    assert result["protocol_sha256"] == sha256_file(PROTOCOL)
    assert result["classification"] == (
        "known_failure_carrier_retry_pass_eligible_fresh_unseen_confirmation"
    )
    assert result["attempt_count"] == 1
    assert result["attempted_rounds"] == [1]
    assert result["selected_round"] == 1
    assert result["production_enabled"] is False
    assert result["api_calls_made"] == 0
    assert result["human_review_generated"] is False
    assert result["fresh_unseen_confirmation_required"] is True


def test_carrier_retry_result_passes_every_occurrence_and_control() -> None:
    attempt = _result()["attempts"][0]

    assert attempt["automatic_pass"] is True
    assert attempt["minimum_attempt"] == attempt["selected_attempt"] == 4
    assert attempt["plan"]["retried_word"] == {
        "word_index": 3,
        "source_casefold": "took",
        "source_phone": "tˈʊk",
        "neutral_phone": "kˈʊtp",
        "lens_phone": "kˈutp",
        "candidate_attempt": 4,
    }
    assert attempt["render_integrity"]["integrity_pass"] is True
    assert attempt["render_integrity"]["neutral_identity_bit_exact"] is True
    assert attempt["render_integrity"]["localization_fraction"] == 1.0
    assert attempt["acoustic"]["pass"] is True
    assert attempt["acoustic"]["identity_false_positive_count"] == 0
    occurrences = [
        row
        for cell in attempt["acoustic"]["cells"]
        for row in cell["occurrences"]
    ]
    assert len(occurrences) == 5
    assert all(
        row["aggregate"]["classification"] == "exact_category_pass"
        and row["candidate"]["target_gain_gate_pass"]
        for row in occurrences
    )
    repaired = next(row for row in occurrences if row["occurrence_index"] == 3)
    assert repaired["candidate"]["direction_cosine"] > 0.9
    assert repaired["candidate"]["lens_target_distance_scaled_rms"] < (
        repaired["candidate"]["neutral_target_distance_scaled_rms"]
    )


def test_carrier_retry_selected_wavs_match_receipts() -> None:
    result = _result()

    for receipt in result["selected_audio"].values():
        path = RUN_DIR / receipt["relative_path"]
        assert sha256_file(path) == receipt["wav_sha256"]
        with wave.open(str(path), "rb") as handle:
            assert handle.getnchannels() == 1
            assert handle.getsampwidth() == 2
            assert handle.getframerate() == 24_000
            assert handle.getnframes() == receipt["sample_count"]
            payload = handle.readframes(handle.getnframes())
        assert hashlib.sha256(payload).hexdigest() == receipt["pcm_sha256"]
