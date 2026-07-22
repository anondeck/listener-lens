from __future__ import annotations

from earshift_bakeoff.audio_delexicalization_probe import (
    _content_audit,
    build_manifest,
    protocol_record,
)


def test_probe_is_bounded_and_reuses_frozen_audio() -> None:
    protocol = protocol_record()
    assert len(protocol["protocol_sha256"]) == 64
    assert protocol["limits"] == {
        "logical_slots": 3,
        "maximum_successful_audio_returns": 3,
        "maximum_estimated_cost_usd": 0.10,
    }
    assert [slot.model for slot in build_manifest()] == [
        "gpt-audio-1.5",
        "gpt-realtime-2.1",
        "gpt-realtime-2.1",
    ]


def test_generative_content_audit_rejects_source_and_real_words() -> None:
    result = _content_audit("what a great day.")
    assert result["source_overlap_pass"] is False
    assert result["dictionary_homophone_adjacency_pass"] is False
    assert result["contract_pass"] is False
