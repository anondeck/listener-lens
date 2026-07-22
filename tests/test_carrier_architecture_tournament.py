from __future__ import annotations

from earshift_bakeoff.carrier_architecture_tournament import (
    RUN_ID,
    SOURCE_TEXT,
    build_manifest,
    compare_prosody,
    prepare_tournament,
    protocol_record,
)
from earshift_bakeoff.runtime_audio import AudioTiming, ProsodyFingerprint


def timing(duration: float = 2.0) -> AudioTiming:
    return AudioTiming(
        duration_s=duration,
        sample_rate_hz=24_000,
        decoded_sample_count=round(duration * 24_000),
        clipped_fraction=0.0,
        utterance_duration_s=duration,
        estimated_syllables_per_second=5.0,
        interior_pause_count=0,
        interior_pause_s=0.0,
        interior_pauses=(),
    )


def prosody(invert: bool = False) -> ProsodyFingerprint:
    sign = -1 if invert else 1
    return ProsodyFingerprint(
        version="prosody-fingerprint-v1",
        bin_count=32,
        frame_count=80,
        energy_contour_db=tuple(sign * (index - 15.5) / 4 for index in range(32)),
        pitch_contour_semitones=tuple(sign * (index - 15.5) / 8 for index in range(32)),
        median_f0_hz=220.0,
        voiced_fraction=0.75,
        energy_span_db=8.0,
    )


def test_protocol_freezes_two_architectures_and_exact_dependency_manifest() -> None:
    protocol = protocol_record()
    assert protocol["run_id"] == RUN_ID
    assert protocol["source_text"] == SOURCE_TEXT
    assert len(protocol["protocol_sha256"]) == 64
    assert [item["carrier_token_count"] for item in protocol["architectures"]] == [10, 5]
    assert [slot.slot_id for slot in build_manifest()] == [
        "source-anchor-1",
        "per-word-neutral-1",
        "prosodic-neutral-1",
        "per-word-neutral-2",
        "prosodic-neutral-2",
        "per-word-identity-1",
        "prosodic-identity-1",
        "per-word-lens-1",
        "prosodic-lens-1",
    ]
    assert protocol["limits"]["maximum_successfully_returned_audio"] == 9
    assert protocol["limits"]["maximum_estimated_cost_usd"] == 0.15


def test_reference_match_accepts_identity_and_rejects_inverted_contours() -> None:
    identity = compare_prosody(timing(), prosody(), timing(), prosody())
    assert identity["eligible"] is True
    assert identity["score"] == 0.0

    inverted = compare_prosody(timing(), prosody(), timing(), prosody(invert=True))
    assert inverted["eligible"] is False
    assert "energy_correlation" in inverted["reasons"]
    assert "pitch_correlation" in inverted["reasons"]


def test_saved_protocol_reloads_without_tuple_list_false_drift() -> None:
    assert prepare_tournament()["protocol_sha256"] == protocol_record()["protocol_sha256"]
