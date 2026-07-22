from __future__ import annotations

from pathlib import Path

from earshift_bakeoff.kokoro_stronger_span_qc import (
    CANDIDATES,
    CUE_END_S,
    CUE_START_S,
    EXCLUDED,
    IDENTITY_SLOT_ID,
    NEUTRAL_SLOT_ID,
    PARENT_RUN_ID,
    RUN_ID,
    SELECTION_ORDER,
    _review_html,
    blinded_layout,
    prepare,
    protocol_record,
    public_review_manifest,
)
from earshift_bakeoff.util import sha256_file


def test_protocol_is_zero_cost_and_hash_bound_to_frozen_v4() -> None:
    protocol = protocol_record()
    assert protocol["parent"]["run_id"] == PARENT_RUN_ID
    assert protocol["parent"]["classification_is_immutable"] is True
    assert protocol["scope"]["api_calls"] == 0
    assert protocol["scope"]["new_renders"] == 0
    assert protocol["scope"]["audio_edits"] == 0
    assert protocol["scope"]["can_reclassify_parent_v4"] is False
    for key in (
        "protocol_sha256",
        "protocol_file_sha256",
        "records_sha256",
        "summary_sha256",
        "manual_result_sha256",
    ):
        assert len(protocol["parent"][key]) == 64
    assert len(protocol["protocol_sha256"]) == 64


def test_candidate_set_exclusions_order_and_metrics_are_frozen() -> None:
    protocol = protocol_record()
    assert tuple(protocol["candidate_set"]["selection_order"]) == SELECTION_ORDER
    assert (
        tuple(
            (row["candidate_id"], row["lens_slot_id"])
            for row in protocol["candidate_set"]["candidates"]
        )
        == CANDIDATES
    )
    assert tuple(protocol["candidate_set"]["excluded"]) == EXCLUDED
    candidates = [
        row
        for row in protocol["source_wavs"]
        if row["role"] == "eligible-stronger-candidate"
    ]
    assert len(candidates) == 3
    for row in candidates:
        assert row["automatic_acoustic_pass"] is True
        assert row["known_descriptive_metrics"]["selection_authority"] is False
        assert row["known_descriptive_metrics"]["threshold"] is None
        assert (
            0
            < row["known_descriptive_metrics"]["inside_difference_energy_fraction"]
            < 1
        )
        assert row["known_descriptive_metrics"]["outside_rms_pcm"] > 0


def test_layout_is_deterministic_complete_and_uses_invariant_cue() -> None:
    protocol = protocol_record()
    layout = blinded_layout(protocol)
    assert layout == blinded_layout(protocol)
    assert {row["logical_id"] for row in layout} == {
        *SELECTION_ORDER,
        "identity-control",
    }
    for trial in layout:
        assert trial["cue_start_s"] == CUE_START_S
        assert trial["cue_end_s"] == CUE_END_S
        assert [side["side"] for side in trial["sides"]] == ["A", "B"]
        slots = {side["source_slot_id"] for side in trial["sides"]}
        assert NEUTRAL_SLOT_ID in slots
        if trial["logical_id"] == "identity-control":
            assert slots == {NEUTRAL_SLOT_ID, IDENTITY_SLOT_ID}


def test_public_review_hides_key_and_collects_complete_frozen_schema() -> None:
    protocol = protocol_record()
    public = public_review_manifest(protocol)
    html = _review_html(public, protocol["protocol_sha256"])
    assert len(public) == 4
    for hidden in (
        "common-neutral",
        "target-word",
        "full-contextual-state",
        "stress-plus-target",
        "target-only",
        "source_slot_id",
        "blind-key",
    ):
        assert hidden not in html
    for field in (
        "naturalness",
        "delivery",
        "meaning",
        "artifact",
        "difference_strength",
        "direction",
        "confidence",
        "interference",
        "notes",
        "replay_count",
    ):
        assert field in html
    assert "TARGET NOW" in html
    assert "String.fromCharCode(10)" in html


def test_prepare_uses_hash_verified_symlinks_without_audio_modification() -> None:
    run_dir = (
        Path(__file__).resolve().parents[1] / "artifacts" / "phoneme-renderer" / RUN_ID
    )
    evidence_paths = (run_dir / "response.json", run_dir / "manual-result.json")
    evidence_before = {
        path.name: sha256_file(path) if path.exists() else None
        for path in evidence_paths
    }
    protocol = prepare()
    assert protocol["run_id"] == RUN_ID
    assert {
        path.name: sha256_file(path) if path.exists() else None
        for path in evidence_paths
    } == evidence_before
    layout = blinded_layout(protocol)
    for trial in layout:
        for side in trial["sides"]:
            path = run_dir / side["opaque_audio_relative_path"]
            assert path.is_symlink()
            assert sha256_file(path) == side["audio_sha256"]
