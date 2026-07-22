from __future__ import annotations

import base64
import json

from earshift_bakeoff.native_realtime_probe import (
    GROUP_SYLLABLE_PATTERN,
    ProbeSlot,
    _extract_transcript,
    _realtime_cost,
    _realtime_usage,
    _safe_realtime_event,
    build_manifest,
    protocol_record,
)


def test_native_probe_is_bounded_and_transport_explicit() -> None:
    protocol = protocol_record()
    assert len(protocol["protocol_sha256"]) == 64
    assert protocol["limits"] == {
        "logical_slots": 3,
        "maximum_successful_audio_returns": 3,
        "maximum_estimated_cost_usd": 0.10,
    }
    assert [slot.transport for slot in build_manifest()] == [
        "chat_completions",
        "realtime_websocket",
        "realtime_websocket",
    ]
    assert protocol["structured_content_gate"]["predicted_group_syllables"] == list(GROUP_SYLLABLE_PATTERN)
    assert protocol["request_payloads"]["structured_delex"]["target"] == {
        "word_number": 4,
        "global_syllable": 8,
        "phonetic_target": "first syllable /bævd/; second syllable reduced /ə/",
    }
    assert protocol["realtime_transport"]["turn_detection"] is None


def test_realtime_usage_and_cost_use_audio_and_text_buckets() -> None:
    raw = {
        "input_tokens": 100,
        "output_tokens": 50,
        "input_token_details": {"text_tokens": 20, "audio_tokens": 80, "cached_tokens": 0},
        "output_token_details": {"text_tokens": 10, "audio_tokens": 40},
    }
    assert _realtime_usage(raw)["input_audio_tokens"] == 80
    expected = (20 * 4 + 80 * 32 + 10 * 24 + 40 * 64) / 1_000_000
    assert _realtime_cost(raw) == round(expected, 8)


def test_transcript_falls_back_to_response_content() -> None:
    event = {
        "response": {
            "output": [{"content": [{"type": "audio", "transcript": "nava deh."}]}]
        }
    }
    assert _extract_transcript(event) == "nava deh."


def test_safe_event_never_retains_audio_delta() -> None:
    encoded = base64.b64encode(b"secret-audio").decode("ascii")
    safe = _safe_realtime_event({"type": "response.output_audio.delta", "delta": encoded})
    assert safe == {"type": "response.output_audio.delta"}
    assert encoded not in json.dumps(safe)
