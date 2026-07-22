from __future__ import annotations

import json
from pathlib import Path

from earshift_bakeoff.util import sha256_file


RUN_DIR = (
    Path(__file__).resolve().parents[1]
    / "artifacts"
    / "typed-engine"
    / "20260716-kokoro-typed-replication-v1"
)


def test_one_pass_render_has_exact_manifest_and_runtime_integrity() -> None:
    payload = json.loads((RUN_DIR / "render-records.json").read_text(encoding="utf-8"))
    assert payload["triplets_rendered"] == 3
    assert payload["logical_wav_outputs"] == 9
    assert payload["api_calls_made"] == 0
    assert payload["paid_calls_made"] == 0
    assert payload["all_runtime_gates_pass"] is True
    assert payload["one_pass_stopping_rule_satisfied"] is True
    assert len(list((RUN_DIR / "audio").glob("*.wav"))) == 9
    for record in payload["records"]:
        assert record["runtime_pass"] is True
        assert record["neutral_identity_bit_identical"] is True
        assert (
            record["audio"]["neutral"]["pcm_sha256"]
            == record["audio"]["identity"]["pcm_sha256"]
        )
        for audio in record["audio"].values():
            path = RUN_DIR / audio["relative_path"]
            assert sha256_file(path) == audio["wav_sha256"]


def test_frozen_result_is_one_context_sensitive_acoustic_failure() -> None:
    result = json.loads((RUN_DIR / "analysis.json").read_text(encoding="utf-8"))
    assert result["classification"] == "automatic_replication_failed_no_promotion"
    assert result["automatic_replication_pass"] is False
    assert result["target_occurrence_count"] == 4
    fixtures = {row["fixture_id"]: row for row in result["fixtures"]}
    assert fixtures["single-target"]["automatic_replication_pass"] is True
    assert fixtures["rhythm-punctuation-weak"]["automatic_replication_pass"] is True
    repeated = fixtures["multi-target-repeated"]
    assert repeated["runtime_pass"] is True
    assert repeated["localization"]["pass"] is True
    assert repeated["acoustic_pass"] is False
    assert [row["pass"] for row in repeated["target_occurrences"]] == [True, False]
    failed = repeated["target_occurrences"][1]["classification"]["families"]
    assert failed["5500"]["direction_cosine"] >= 0.5
    assert failed["5750"]["direction_cosine"] >= 0.5
    assert failed["6000"]["direction_cosine"] >= 0.5
    assert failed["5500"]["lens_category_pass"] is False
    assert failed["5750"]["lens_category_pass"] is False
    assert failed["6000"]["lens_category_pass"] is True
    assert failed["6000"]["magnitude_bark"] < failed["6000"]["threshold_bark"]


def test_failed_automatic_gate_did_not_open_blind_review() -> None:
    html = (RUN_DIR / "review.html").read_text(encoding="utf-8")
    assert "listener review was not opened" in html
    assert not (RUN_DIR / "blind-key.json").exists()
    assert not (RUN_DIR / "review-manifest.json").exists()
    assert not (RUN_DIR / "manual-result.json").exists()
