from __future__ import annotations

import hashlib
import json
from pathlib import Path
import wave

from earshift_bakeoff.bilingual_product_audio_state import (
    load_bilingual_audio_integrity_state,
)
from earshift_bakeoff.bilingual_product_matrix import (
    load_bilingual_product_matrix,
)
from earshift_bakeoff.config import ROOT, stable_json
from earshift_bakeoff.util import sha256_file


RUN_DIR = (
    ROOT
    / "artifacts"
    / "product-matrix"
    / "20260717-bilingual-product-audio-integrity-screen-v1"
)
RESULT = RUN_DIR / "results.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _semantic_hash(payload: dict) -> str:
    semantic = dict(payload)
    semantic.pop("record_sha256", None)
    return hashlib.sha256(stable_json(semantic).encode("utf-8")).hexdigest()


def test_all_changed_voice_rule_cells_passed_universal_audio_integrity() -> None:
    result = _load(RESULT)
    outcomes = result["outcomes"]

    assert result["record_sha256"] == _semantic_hash(result)
    assert result["classification"] == (
        "all_cells_universal_integrity_pass_family_acoustics_pending"
    )
    assert result["slot_count"] == 98
    assert result["universal_integrity_pass_count"] == 98
    assert result["universal_integrity_fail_count"] == 0
    assert result["universal_integrity_yield"] == 1.0
    assert result["api_calls_made"] == 0
    assert result["audio_render_sets_made"] == 98
    assert result["family_acoustic_classification_status"] == "pending"
    assert result["production_enabled"] is False
    assert len(outcomes) == 98
    assert len({row["cell_id"] for row in outcomes}) == 98
    assert all(row["status"] == "universal_integrity_pass" for row in outcomes)
    assert all(
        row["family_acoustic_status"] == "not_classified_by_integrity_screen"
        for row in outcomes
    )
    assert all(row["product_enabled"] is False for row in outcomes)


def test_every_saved_audio_file_matches_its_frozen_pcm_and_wav_hash() -> None:
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


def test_audio_integrity_state_binds_result_without_promoting_cells() -> None:
    matrix = load_bilingual_product_matrix()
    state = load_bilingual_audio_integrity_state(
        matrix_version=matrix.matrix_version,
        matrix_sha256=matrix.matrix_sha256,
    )

    assert state["slot_count"] == 98
    assert state["universal_integrity_pass_count"] == 98
    assert state["family_counts"] == {
        "vowel": 80,
        "consonant": 12,
        "insertion": 2,
        "prosody": 4,
    }
    assert state["voice_counts"] == {
        "af_heart": 23,
        "am_michael": 23,
        "pm_alex": 26,
        "pf_dora": 26,
    }
    assert state["family_acoustic_validation_status"] == "pending"
    assert state["production_enabled"] is False
