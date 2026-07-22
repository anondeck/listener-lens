from __future__ import annotations

from earshift_bakeoff.kokoro_phoneme_spike import (
    MODEL_HASHES,
    SCHWA_WEAK_LENS,
    SCHWA_WEAK_NEUTRAL,
    _one_symbol_difference,
    _phone_gate_report,
    build_manifest,
    duration_alignment,
    protocol_record,
)


def test_kokoro_spike_is_zero_api_and_hash_bound() -> None:
    protocol = protocol_record()
    assert len(protocol["protocol_sha256"]) == 64
    assert protocol["renderer"]["api_calls"] == 0
    assert protocol["renderer"]["model_hashes"] == MODEL_HASHES
    assert len(build_manifest()) == 6


def test_neutral_and_lens_differ_only_at_target_vowel() -> None:
    result = _one_symbol_difference(SCHWA_WEAK_NEUTRAL, SCHWA_WEAK_LENS)
    assert result["pass"] is True
    assert len(result["differences"]) == 1
    assert result["differences"][0]["neutral"] == "æ"
    assert result["differences"][0]["lens"] == "ɛ"


def test_phoneme_plan_is_gate_clean_in_isolation_and_adjacency() -> None:
    report = _phone_gate_report()
    assert report["neutral"]["pass"] is True
    assert report["lens"]["pass"] is True


def test_duration_alignment_maps_unique_target_and_boundary_tokens() -> None:
    vocab = {symbol: index for index, symbol in enumerate("abæ.", start=1)}
    result = duration_alignment(
        vocab=vocab,
        phonemes="abæ.",
        pred_dur=[1, 2, 3, 4, 5, 1],
        sample_count=16 * 600,
        target_symbol="æ",
    )
    assert result["duration_index_with_boundary"] == 3
    assert result["target_duration_frames"] == 4
    assert result["samples_per_duration_frame"] == 600
    assert result["start_sample"] == 6 * 600
    assert result["end_sample_exclusive"] == 10 * 600
