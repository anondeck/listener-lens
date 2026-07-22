from __future__ import annotations

from earshift_bakeoff.realtime_transport_correction import (
    MAX_COST_USD,
    build_manifest,
    protocol_record,
)


def test_transport_correction_changes_only_required_pcm_rate() -> None:
    protocol = protocol_record()
    assert protocol["only_amendment"]["session_audio_output_format"] == {
        "type": "audio/pcm",
        "rate": 24_000,
    }
    assert len(build_manifest()) == 2
    assert all(slot.model == "gpt-realtime-2.1" for slot in build_manifest())
    assert protocol["limits"] == {
        "logical_slots": 2,
        "maximum_successful_audio_returns": 2,
        "maximum_estimated_cost_usd": MAX_COST_USD,
    }
