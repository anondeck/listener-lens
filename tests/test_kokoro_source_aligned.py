from __future__ import annotations

from earshift_bakeoff.kokoro_source_aligned import (
    CARRIER_LENS,
    CARRIER_NEUTRAL,
    SOURCE_PHONEMES,
    TARGET_RAW_INDEX,
    anchor_manifest,
    isomorphism_report,
    phone_gate_report,
    protocol_record,
    stress_span_alignment,
)


def test_carrier_is_token_isomorphic_and_lens_changes_one_vowel() -> None:
    report = isomorphism_report()
    assert report["pass"] is True
    assert len(SOURCE_PHONEMES) == len(CARRIER_NEUTRAL) == len(CARRIER_LENS)
    assert report["neutral_lens_differences"] == [
        {"index": TARGET_RAW_INDEX, "neutral": "æ", "lens": "ɛ"}
    ]


def test_carrier_phone_and_adjacency_gates_pass() -> None:
    report = phone_gate_report()
    assert report["neutral"]["pass"] is True
    assert report["lens"]["pass"] is True


def test_anchor_manifest_is_fixed_at_ten_slots() -> None:
    manifest = anchor_manifest()
    assert len(manifest) == 10
    assert [slot.request_order for slot in manifest] == list(range(9, 19))
    assert sum(slot.vowel_label == "ae" for slot in manifest) == 3
    assert sum(slot.vowel_label == "eh" for slot in manifest) == 3


def test_protocol_records_zero_api_and_shared_state_contract() -> None:
    protocol = protocol_record()
    assert protocol["renderer"]["api_calls"] == 0
    assert len(protocol["manifest"]) == 18
    assert protocol["carrier"]["isomorphism"]["pass"] is True
    assert len(protocol["protocol_sha256"]) == 64


def test_stress_span_alignment_includes_stress_and_vowel_frames() -> None:
    vocab = {symbol: index for index, symbol in enumerate("hˈæd")}
    result = stress_span_alignment(
        vocab=vocab,
        phonemes="hˈæd",
        pred_dur=[10, 2, 5, 3, 4, 6],
        sample_count=30 * 600,
        target_symbol="æ",
    )
    assert result["stress_duration_frames"] == 5
    assert result["target_duration_frames"] == 3
    assert result["start_sample"] == 12 * 600
    assert result["end_sample_exclusive"] == 20 * 600
