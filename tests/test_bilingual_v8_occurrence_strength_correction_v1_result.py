from __future__ import annotations

import hashlib
import json

from earshift_bakeoff.config import ROOT, stable_json
from earshift_bakeoff.util import sha256_file


RUN_DIR = (
    ROOT
    / "artifacts"
    / "product-matrix"
    / "20260718-bilingual-v8-occurrence-strength-correction-v1"
)
RESULT = RUN_DIR / "results.json"
PROTOCOL = ROOT / "rules" / "bilingual-v8-occurrence-strength-correction-v1.json"


def _result() -> dict:
    return json.loads(RESULT.read_text(encoding="utf-8"))


def _semantic_hash(payload: dict) -> str:
    semantic = dict(payload)
    semantic.pop("record_sha256", None)
    return hashlib.sha256(stable_json(semantic).encode("utf-8")).hexdigest()


def test_occurrence_strength_result_is_complete_failed_and_nonpromotional() -> None:
    result = _result()

    assert sha256_file(RESULT) == (
        "7cd009a1d7236640efa6cb96dc6bed30e31be21509f2610c7886c7a407f70f33"
    )
    assert result["record_sha256"] == _semantic_hash(result)
    assert result["record_sha256"] == (
        "beaab9a76f758620bb9ef9ca66fff44ca90334fa73a6e7313602f96f6a27c5c7"
    )
    assert result["protocol_sha256"] == sha256_file(PROTOCOL)
    assert result["classification"] == (
        "known_failure_occurrence_correction_failed_preserve_parent_failure"
    )
    assert result["production_enabled"] is False
    assert result["api_calls_made"] == 0
    assert result["human_review_generated"] is False
    assert result["fresh_unseen_confirmation_required"] is False
    assert result["selected_strength"] is None
    assert result["selected_audio"] is None
    assert result["attempt_count"] == 5
    assert result["attempted_strengths"] == [0.75, 1.25, 0.5, 1.5, 2.0]


def test_occurrence_strength_preserves_baseline_and_every_unaffected_window() -> None:
    result = _result()

    assert result["baseline_binding_pass"] is True
    assert result["baseline_acoustic"]["pass"] is False
    assert result["baseline_equivalence"]["pass"] is True
    assert result["baseline_equivalence"]["neutral_pcm_exact"] is True
    assert result["baseline_equivalence"]["identity_pcm_exact"] is True
    assert result["baseline_equivalence"]["full_lens_pcm_exact"] is True
    assert all(
        attempt["neutral_control_pass"]
        and attempt["diagnostics"]["baseline_lens_exact_outside_failed_window"]
        and attempt["diagnostics"]["verification"]["integrity_pass"]
        and attempt["diagnostics"]["localization"][
            "inside_difference_energy_fraction"
        ]
        == 1.0
        for attempt in result["attempts"]
    )

    for attempt in result["attempts"]:
        occurrences = {
            row["occurrence_index"]: row
            for cell in attempt["acoustic"]["cells"]
            for row in cell["occurrences"]
        }
        assert {
            index: row["aggregate"]["classification"]
            for index, row in occurrences.items()
            if index != 3
        } == {
            0: "exact_category_pass",
            1: "exact_category_pass",
            2: "exact_category_pass",
            4: "exact_category_pass",
        }
        failed = occurrences[3]
        assert failed["aggregate"]["classification"] == "fail"
        assert failed["candidate"]["target_gain_gate_pass"] is False
        assert attempt["automatic_pass"] is False
