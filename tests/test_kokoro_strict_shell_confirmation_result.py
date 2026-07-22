from __future__ import annotations

import json

from earshift_bakeoff.config import sha256_json
from earshift_bakeoff.kokoro_strict_shell_confirmation import (
    ANALYSIS_FILE,
    NEW_FIXTURE_ID,
    RECORDS_FILE,
    REUSED_FIXTURE_IDS,
    run_dir,
)


def _load(name: str) -> dict:
    return json.loads((run_dir() / name).read_text(encoding="utf-8"))


def test_strict_shell_aggregate_passes_and_builds_one_blind_review() -> None:
    analysis = _load(ANALYSIS_FILE)
    records = _load(RECORDS_FILE)
    assert analysis["classification"] == (
        "strict_shell_aggregate_automatic_pass_pending_human_qc"
    )
    assert analysis["automatic_pass"] is True
    assert analysis["pending_human_review"] is True
    assert analysis["reused_fixture_ids"] == list(REUSED_FIXTURE_IDS)
    assert records["decoder_attempt_count"] == 3
    assert records["status"] == "render_complete"
    assert (run_dir() / "review.html").is_file()
    assert _load("review-manifest.json")["trial_count"] == 6
    unhashed = {
        key: value for key, value in analysis.items() if key != "analysis_sha256"
    }
    assert analysis["analysis_sha256"] == sha256_json(unhashed)


def test_new_medial_fixture_passes_every_primary_and_integrity_gate() -> None:
    result = _load(ANALYSIS_FILE)["new_fixture"]
    assert result["fixture_id"] == NEW_FIXTURE_ID
    assert result["automatic_pass"] is True
    assert all(result["automatic_checks"].values())
    assert result["phrase_medial_edge_gate"]["pass"] is True
    assert result["boundary_artifact"]["pass"] is True
    assert result["boundary_artifact"]["maximum_edge_delta_step_pcm"] == 0.0
    assert result["acoustic"]["primary_gate_pass"] is True
    assert all(
        result["acoustic"]["windows"][window]["pass"]
        for window in ("40", "50", "60")
    )
    assert result["spliced_localization"][
        "inside_difference_energy_fraction"
    ] == 1.0
    assert result["localization_runtime_benchmark"]["pass"] is True


def test_review_manifest_exposes_no_condition_or_fixture_identity() -> None:
    manifest = _load("review-manifest.json")
    assert manifest["hidden_fields_absent"] is True
    for trial in manifest["public_trials"]:
        assert set(trial) == {
            "trial_id",
            "duration_s",
            "target_intervals",
            "sides",
        }
        assert all(set(side) == {"side", "audio"} for side in trial["sides"])
