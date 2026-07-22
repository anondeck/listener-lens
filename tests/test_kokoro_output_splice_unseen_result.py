from __future__ import annotations

import json

from earshift_bakeoff.config import sha256_json
from earshift_bakeoff.kokoro_output_splice_unseen import (
    ANALYSIS_FILE,
    RECORDS_FILE,
    run_dir,
)


def _load(name: str) -> dict:
    return json.loads((run_dir() / name).read_text(encoding="utf-8"))


def test_unseen_result_is_frozen_failure_with_all_nine_decodes() -> None:
    records = _load(RECORDS_FILE)
    analysis = _load(ANALYSIS_FILE)
    assert records["status"] == "render_complete"
    assert records["decoder_attempt_count"] == 9
    assert all(row["status"] == "complete" for row in records["slots"])
    assert analysis["classification"] == "unseen_output_splice_automatic_failed"
    assert analysis["automatic_pass"] is False
    assert analysis["pending_human_review"] is False
    assert not (run_dir() / "review.html").exists()
    unhashed = {
        key: value for key, value in analysis.items() if key != "analysis_sha256"
    }
    assert analysis["analysis_sha256"] == sha256_json(unhashed)


def test_splice_engine_passes_while_upstream_acoustic_contexts_fail() -> None:
    analysis = _load(ANALYSIS_FILE)
    for fixture in analysis["fixtures"]:
        checks = fixture["automatic_checks"]
        assert checks["runtime_and_exact_pcm_integrity"] is True
        assert checks["boundary_click_metrics"] is True
        assert checks["localization_at_least_0_80"] is True
        assert checks["localization_runtime_cheap_fail_closed"] is True
        assert checks["primary_50_acoustic_gate"] is False
        assert fixture["spliced_localization"][
            "inside_difference_energy_fraction"
        ] == 1.0
        assert fixture["spliced_localization"]["expected_by_construction"] is True


def test_phrase_medial_fixture_also_fails_frozen_boundary_placement() -> None:
    analysis = _load(ANALYSIS_FILE)
    rows = {row["fixture_id"]: row for row in analysis["fixtures"]}
    edge = rows["phrase-medial-continuous"]["phrase_medial_edge_gate"]
    assert edge["start_inside_preceding_word"] is False
    assert edge["end_inside_following_word"] is True
    assert edge["pass"] is False


def test_descriptive_windows_are_reported_but_do_not_rescue_failure() -> None:
    analysis = _load(ANALYSIS_FILE)
    phrase_final = next(
        row
        for row in analysis["fixtures"]
        if row["fixture_id"] == "phrase-final-new-context"
    )
    assert phrase_final["acoustic"]["windows"]["50"]["pass"] is False
    assert phrase_final["acoustic"]["windows"]["60"]["pass"] is True
    assert phrase_final["automatic_pass"] is False
    assert analysis["descriptive_40_60_windows_cannot_change_outcome"] is True
