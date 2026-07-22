from __future__ import annotations

import json

from earshift_bakeoff.config import Paths
from earshift_bakeoff.portuguese_kokoro_gate import RUN_ID
from earshift_bakeoff.util import sha256_file


def test_frozen_portuguese_index_result_is_partial_positive_only() -> None:
    run_dir = Paths().artifacts / "portuguese" / RUN_ID
    protocol = json.loads((run_dir / "protocol.json").read_text(encoding="utf-8"))
    receipt = json.loads(
        (run_dir / "full-index-receipt.json").read_text(encoding="utf-8")
    )
    assert receipt["protocol_sha256"] == protocol["protocol_sha256"]
    assert receipt["status"] == "partial_positive_only_index"
    assert receipt["sample_repeatable"] is True
    assert receipt["challenge_pass"] is True
    assert receipt["counts"] == {
        "covered_words": 255_881,
        "database_rows": 255_881,
        "input_words": 262_151,
        "uncovered_words": 6_270,
        "unique_phone_hashes": 238_702,
    }
    assert receipt["coverage_rate"] == 0.9760824868110364
    assert receipt["database_sha256"] == (
        "ee63da3ebd3bd73eaa50beffc083a0d098e9150576b48ea0a4e9c3778805f89e"
    )
    assert receipt["contains_plaintext_words_or_phones"] is False
    assert receipt["api_calls_made"] == 0
    assert receipt["audio_renders_made"] == 0
    assert sha256_file(run_dir / "full-index-receipt.json") == (
        "d24c48a8dff31631e3d204312506177f6298ce333a86a33863832f0fd059f991"
    )
