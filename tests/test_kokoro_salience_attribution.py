from __future__ import annotations

from earshift_bakeoff.kokoro_salience_attribution import (
    PARENT_RUN_ID,
    _review_html,
    blinded_layout,
    protocol_record,
    public_review_manifest,
)


def test_protocol_is_one_zero_cost_session_bound_to_frozen_v4() -> None:
    protocol = protocol_record()
    assert protocol["parent"]["run_id"] == PARENT_RUN_ID
    assert protocol["parent"]["classification_is_immutable"] is True
    assert protocol["scope"]["sessions"] == 1
    assert protocol["scope"]["api_calls"] == 0
    assert protocol["scope"]["new_renders"] == 0
    assert protocol["scope"]["new_audio_edits"] == 0
    assert len(protocol["logical_comparisons"]) == 3
    assert protocol["decision_mapping"]["clear_correct_direction"].startswith("difference strength >=5")
    assert protocol["decision_mapping"]["anchors_unclear"].startswith("the full-context anchor comparison")
    assert len(protocol["protocol_sha256"]) == 64


def test_frozen_identity_sources_are_bit_identical() -> None:
    protocol = protocol_record()
    sources = {row["slot_id"]: row for row in protocol["source_wavs"]}
    neutral = sources["common-neutral"]
    identity = sources["common-neutral-identity"]
    assert neutral["audio_sha256"] == identity["audio_sha256"]
    assert neutral["decoded_sample_count"] == identity["decoded_sample_count"]


def test_blinding_is_deterministic_complete_and_side_invariant() -> None:
    protocol = protocol_record()
    first = blinded_layout(protocol)
    assert first == blinded_layout(protocol)
    assert {row["logical_id"] for row in first} == {
        "identity-control",
        "selected-lens",
        "full-context-anchors",
    }
    for trial in first:
        assert [side["side"] for side in trial["sides"]] == ["A", "B"]
        assert trial["cue_start_s"] < trial["cue_end_s"]


def test_public_review_hides_conditions_and_collects_every_measure() -> None:
    protocol = protocol_record()
    public = public_review_manifest(protocol)
    html = _review_html(public, protocol["protocol_sha256"])
    assert len(public) == 3
    assert "common-neutral" not in html
    assert "stress-plus-target" not in html
    assert "anchor-full-carrier" not in html
    assert "difference_strength" in html
    assert "category_judgment" in html
    assert "confidence" in html
    assert "replay_count" in html
    assert "interference" in html
    assert "artifact" in html
    assert "TARGET NOW" in html
    assert "String.fromCharCode(10)" in html
