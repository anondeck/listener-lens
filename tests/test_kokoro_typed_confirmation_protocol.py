from __future__ import annotations

import pytest

from earshift_bakeoff import kokoro_typed_confirmation_protocol as confirmation_protocol
from earshift_bakeoff.kokoro_typed_confirmation_protocol import (
    REVIEW_RESPONSE_FILENAME,
    blinded_trial_plan,
    protocol_record,
)


def test_protocol_binds_diagnostic_decision_and_immutable_failed_parent() -> None:
    protocol = protocol_record()
    diagnostic = protocol["parents"]["diagnostic"]
    failed = protocol["parents"]["frozen_failed_replication_v1"]
    assert diagnostic["classification"] == (
        "transported_calibration_mechanically_sufficient_for_this_fixture"
    )
    assert diagnostic["selected_span"] == "target-word"
    assert diagnostic["mechanical_attribution"] == (
        "mixed_transported_endpoint_and_threshold_calibration"
    )
    assert diagnostic["confirmation_eligible"] is True
    assert (
        len(
            [
                row
                for row in diagnostic["bound_output_files"]
                if row["relative_path"].endswith(".wav")
            ]
        )
        == 2
    )
    assert failed["classification"] == "automatic_replication_failed_no_promotion"
    assert failed["preservation"] == "immutable_failed_result_not_reclassified"
    assert len(failed["bound_wavs"]) == 9


def test_exact_two_fixtures_and_six_one_attempt_slots_are_frozen() -> None:
    protocol = protocol_record()
    fixtures = protocol["fixtures"]
    assert [row["fixture_id"] for row in fixtures] == [
        "new-repeated-phrase-final",
        "independent-phrase-final-only",
    ]
    assert [row["text"] for row in fixtures] == [
        "The cap turns near the cap.",
        "We rest near the cap.",
    ]
    assert [row["expected_plan_sha256"] for row in fixtures] == [
        "c83bab90075c75619ed7c164cb4f325fc94a1325cb71c0c7e0fb7e87ba36320b",
        "1f03a5383c38d504bd5bbd565f105675081660e0c214e33c48189d3001c748d7",
    ]
    assert [row["anchor_occurrence_map"] for row in fixtures] == [[0, 1], [1]]
    manifest = protocol["render_manifest"]
    assert len(manifest) == 6
    assert [row["order"] for row in manifest] == list(range(1, 7))
    assert [row["role"] for row in manifest] == [
        "neutral",
        "identity",
        "lens",
    ] * 2
    assert all(row["selected_span"] == "target-word" for row in manifest)
    assert all(row["one_attempt_no_retry"] for row in manifest)


def test_gate_windows_review_layout_and_commit_barrier_are_exact() -> None:
    protocol = protocol_record()
    gate = protocol["automatic_gate"]
    assert gate["primary_window_percent"] == 50
    assert gate["descriptive_window_percents"] == [40, 60]
    assert protocol["implementation"]["measurement"]["ceiling_hz_family"] == [
        5500,
        5750,
        6000,
    ]
    review = protocol["blind_review"]
    assert review["only_after_automatic_pass"] is True
    assert review["response_filename"] == REVIEW_RESPONSE_FILENAME
    assert review["frozen_layout"] == blinded_trial_plan()
    assert review["trial_count"] == 4
    tracked = protocol["implementation"]["committed_before_render"][
        "tracked_clean_paths"
    ]
    assert "src/earshift_bakeoff/kokoro_typed_confirmation.py" in tracked
    assert "scripts/run_kokoro_typed_confirmation_v1.py" in tracked
    assert (
        "artifacts/typed-engine/20260717-kokoro-typed-confirmation-v1/protocol.json"
        in tracked
    )
    assert (
        "artifacts/typed-engine/20260717-kokoro-typed-diagnostic-v1/analysis.json"
        in tracked
    )


def test_blind_layout_is_repeatable_and_covers_both_branches() -> None:
    first = blinded_trial_plan()
    assert first == blinded_trial_plan()
    assert [row["trial_id"] for row in first] == [
        f"comparison-{index:02d}" for index in range(1, 5)
    ]
    assert {row["condition"] for row in first} == {
        "identity-catch",
        "lens-candidate",
    }
    for fixture_id in {
        "new-repeated-phrase-final",
        "independent-phrase-final-only",
    }:
        rows = [row for row in first if row["fixture_id"] == fixture_id]
        assert {row["condition"] for row in rows} == {
            "identity-catch",
            "lens-candidate",
        }
        assert all(set(row["side_roles"]) == {"A", "B"} for row in rows)


def test_prepare_refuses_outputs_that_predate_the_freeze(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(confirmation_protocol, "run_dir", lambda: tmp_path)
    monkeypatch.setattr(
        confirmation_protocol,
        "protocol_record",
        lambda: {"run_id": "test", "protocol_sha256": "a" * 64},
    )
    (tmp_path / "audio").mkdir()
    with pytest.raises(RuntimeError, match="output exists before"):
        confirmation_protocol.prepare()
    assert not (tmp_path / "protocol.json").exists()
