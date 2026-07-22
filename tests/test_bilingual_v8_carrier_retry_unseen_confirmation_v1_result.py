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
    / "20260718-bilingual-v8-carrier-retry-unseen-confirmation-v1"
)
RESULT = RUN_DIR / "results.json"
PROTOCOL = (
    ROOT / "rules" / "bilingual-v8-carrier-retry-unseen-confirmation-v1.json"
)


def _result() -> dict:
    return json.loads(RESULT.read_text(encoding="utf-8"))


def _semantic_hash(payload: dict) -> str:
    semantic = dict(payload)
    semantic.pop("record_sha256", None)
    return hashlib.sha256(stable_json(semantic).encode("utf-8")).hexdigest()


def test_adaptive_carrier_unseen_result_is_three_of_three_with_two_rescues() -> None:
    result = _result()

    assert sha256_file(RESULT) == (
        "c61ef59c2ef64f71485cf336cef98f190425c1fd406c596c841d0667718f8af0"
    )
    assert result["record_sha256"] == _semantic_hash(result)
    assert result["record_sha256"] == (
        "f359dd474aefe47fe1767809d9a53949a05cabc3275308dbb85db5dbfcbf611d"
    )
    assert result["protocol_sha256"] == sha256_file(PROTOCOL)
    assert result["classification"] == (
        "unseen_carrier_retry_algorithm_automatic_pass_pending_human_qc"
    )
    assert result["fixture_count"] == result["automatic_pass_count"] == 3
    assert result["rescued_fixture_count"] == 2
    assert result["total_attempt_count"] == 5
    assert result["api_calls_made"] == 0
    assert result["production_enabled"] is False
    assert result["human_review_generated"] is False


def test_adaptive_carrier_unseen_preserves_exact_failure_and_rescue_paths() -> None:
    fixtures = {row["fixture_id"]: row for row in _result()["fixtures"]}

    assert {
        fixture_id: (row["automatic_pass"], row["rescued_after_retry"])
        for fixture_id, row in fixtures.items()
    } == {
        "heart_adaptive_unseen": (True, False),
        "michael_adaptive_unseen": (True, True),
        "dora_adaptive_unseen": (True, True),
    }
    assert fixtures["heart_adaptive_unseen"]["selected_round_index"] == 0
    assert fixtures["michael_adaptive_unseen"]["selected_round_index"] == 1
    assert fixtures["dora_adaptive_unseen"]["selected_round_index"] == 1

    michael = fixtures["michael_adaptive_unseen"]["attempts"]
    dora = fixtures["dora_adaptive_unseen"]["attempts"]
    assert michael[0]["failed_mapping_keys"] == [
        {
            "source_casefold": "got",
            "source_phone": "ɡɑt",
            "carrier_role": "content",
            "profile_id": "en-US-to-pt-BR-listener-v2",
        }
    ]
    assert michael[1]["retry_specs"] == [
        {
            "source_casefold": "got",
            "source_phone": "ɡɑt",
            "carrier_role": "content",
            "minimum_attempt": 1,
        }
    ]
    assert dora[0]["failed_mapping_keys"] == [
        {
            "source_casefold": "leve",
            "source_phone": "lˈɛvy",
            "carrier_role": "content",
            "profile_id": "pt-BR-to-en-US-listener-v2",
        }
    ]
    assert dora[1]["retry_specs"] == [
        {
            "source_casefold": "leve",
            "source_phone": "lˈɛvy",
            "carrier_role": "content",
            "minimum_attempt": 1,
        }
    ]


def test_adaptive_carrier_selected_attempts_pass_every_automatic_control() -> None:
    result = _result()

    assert sum(
        sum(row["selected_rule_occurrences"].values())
        for row in result["fixtures"]
    ) == 17
    for fixture in result["fixtures"]:
        selected = next(
            attempt
            for attempt in fixture["attempts"]
            if attempt["round_index"] == fixture["selected_round_index"]
        )
        assert selected["automatic_pass"] is True
        assert selected["render_integrity"]["integrity_pass"] is True
        assert selected["render_integrity"]["neutral_identity_bit_exact"] is True
        assert selected["render_integrity"]["localization_fraction"] == 1.0
        assert selected["acoustic"]["pass"] is True
        assert selected["acoustic"]["identity_false_positive_count"] == 0
        assert all(cell["pass"] for cell in selected["acoustic"]["cells"])


def test_adaptive_carrier_selected_wavs_match_receipts() -> None:
    result = _result()

    for fixture in result["fixtures"]:
        for receipt in fixture["selected_audio"].values():
            path = RUN_DIR / receipt["relative_path"]
            assert sha256_file(path) == receipt["wav_sha256"]
            with wave.open(str(path), "rb") as handle:
                assert handle.getnchannels() == 1
                assert handle.getsampwidth() == 2
                assert handle.getframerate() == 24_000
                assert handle.getnframes() == receipt["sample_count"]
                payload = handle.readframes(handle.getnframes())
            assert hashlib.sha256(payload).hexdigest() == receipt["pcm_sha256"]
