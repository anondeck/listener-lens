from __future__ import annotations

import json

from earshift_bakeoff.config import sha256_json
from earshift_bakeoff.kokoro_validated_shell_confirmation import (
    ANALYSIS_FILE,
    RECORDS_FILE,
    run_dir,
)


def _load(name: str) -> dict:
    return json.loads((run_dir() / name).read_text(encoding="utf-8"))


def test_validated_shell_run_remains_frozen_failed_without_review() -> None:
    analysis = _load(ANALYSIS_FILE)
    records = _load(RECORDS_FILE)
    assert analysis["classification"] == "validated_shell_automatic_failed"
    assert analysis["automatic_pass"] is False
    assert analysis["pending_human_review"] is False
    assert records["status"] == "render_complete"
    assert records["decoder_attempt_count"] == 9
    assert not (run_dir() / "review.html").exists()
    unhashed = {
        key: value for key, value in analysis.items() if key != "analysis_sha256"
    }
    assert analysis["analysis_sha256"] == sha256_json(unhashed)


def test_exact_shell_cells_pass_and_extended_shell_cell_fails_only_acoustics() -> None:
    rows = {row["fixture_id"]: row for row in _load(ANALYSIS_FILE)["fixtures"]}
    assert rows["phrase-final-validated-shell"]["automatic_pass"] is True
    assert rows["multiple-repeated-target"]["automatic_pass"] is True
    medial = rows["phrase-medial-continuous"]
    assert medial["automatic_pass"] is False
    assert medial["phrase_medial_edge_gate"]["pass"] is True
    assert medial["automatic_checks"]["primary_50_acoustic_gate"] is False
    assert all(
        value is True
        for key, value in medial["automatic_checks"].items()
        if key != "primary_50_acoustic_gate"
    )


def test_extended_shell_failure_is_neutral_anchor_sanity_not_direction() -> None:
    medial = next(
        row
        for row in _load(ANALYSIS_FILE)["fixtures"]
        if row["fixture_id"] == "phrase-medial-continuous"
    )
    families = medial["acoustic"]["windows"]["50"]["occurrences"][0]["families"]
    assert families["5500"]["pass"] is True
    for ceiling in ("5750", "6000"):
        checks = families[ceiling]["checks"]
        assert checks["neutral_nearer_local_ae"] is False
        assert checks["lens_nearer_local_eh"] is True
        assert checks["direction_cosine_at_least_0_50"] is True
        assert checks["magnitude_at_least_local_threshold"] is True
