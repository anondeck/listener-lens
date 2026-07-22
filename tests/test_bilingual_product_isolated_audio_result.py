from __future__ import annotations

import hashlib
import json
from pathlib import Path
import wave

from earshift_bakeoff.bilingual_product_audio_state import (
    load_bilingual_isolated_audio_state,
)
from earshift_bakeoff.bilingual_product_matrix import load_bilingual_product_matrix
from earshift_bakeoff.config import ROOT, stable_json
from earshift_bakeoff.util import sha256_file


RUN_DIR = (
    ROOT
    / "artifacts"
    / "product-matrix"
    / "20260717-bilingual-product-isolated-audio-screen-v1"
)
RESULT = RUN_DIR / "results.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _semantic_hash(payload: dict) -> str:
    semantic = dict(payload)
    semantic.pop("record_sha256", None)
    return hashlib.sha256(stable_json(semantic).encode("utf-8")).hexdigest()


def test_all_280_atomically_isolated_audio_slots_pass_universal_integrity() -> None:
    result = _load(RESULT)
    outcomes = result["outcomes"]

    assert result["record_sha256"] == _semantic_hash(result)
    assert result["classification"] == (
        "all_isolated_slots_universal_integrity_pass_family_acoustics_pending"
    )
    assert result["slot_count"] == 280
    assert result["isolated_universal_integrity_pass_count"] == 280
    assert result["isolated_universal_integrity_fail_count"] == 0
    assert result["isolated_universal_integrity_yield"] == 1.0
    assert result["audio_render_sets_made"] == 280
    assert result["api_calls_made"] == 0
    assert result["replacement_slots_used"] == 0
    assert len({row["logical_slot_id"] for row in outcomes}) == 280
    assert all(
        row["status"] == "isolated_universal_integrity_pass" for row in outcomes
    )
    assert all(
        row["isolated_active_changed_rule_ids"] == [row["rule_id"]]
        for row in outcomes
    )
    assert all(row["product_enabled"] is False for row in outcomes)
    assert result["production_enabled"] is False


def test_every_isolated_audio_file_matches_the_bound_wav_and_pcm_hashes() -> None:
    result = _load(RESULT)
    for outcome in result["outcomes"]:
        for role in ("neutral", "lens"):
            record = outcome["audio"][role]
            path = RUN_DIR / record["relative_path"]
            assert sha256_file(path) == record["wav_sha256"]
            with wave.open(str(path), "rb") as handle:
                assert handle.getnchannels() == 1
                assert handle.getsampwidth() == 2
                assert handle.getframerate() == 24_000
                frames = handle.readframes(handle.getnframes())
                assert handle.getnframes() == record["sample_count"]
            assert hashlib.sha256(frames).hexdigest() == record["pcm_sha256"]


def test_isolated_audio_state_preserves_family_acoustic_and_product_boundaries() -> None:
    matrix = load_bilingual_product_matrix()
    state = load_bilingual_isolated_audio_state(
        matrix_version=matrix.matrix_version,
        matrix_sha256=matrix.matrix_sha256,
    )

    assert state["family_counts"] == {
        "vowel": 240,
        "consonant": 32,
        "insertion": 4,
        "prosody": 4,
    }
    assert state["voice_counts"] == {
        "af_heart": 66,
        "am_michael": 66,
        "pm_alex": 74,
        "pf_dora": 74,
    }
    assert state["family_acoustic_validation_status"] == "pending"
    assert state["human_validation_status"] == (
        "pending_automatic_family_acoustics"
    )
    assert state["production_enabled"] is False
