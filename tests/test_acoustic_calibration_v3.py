from __future__ import annotations

import base64
import io
import math
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from earshift_bakeoff.acoustic_calibration_v3 import (
    F1_ORDER,
    F2_ORDER,
    _render_confirmation_slot,
    build_confirmation_manifest,
    confirmation_protocol_record,
    evaluate_internal_coherence,
    load_v2_records,
)


def _tone_wav_bytes() -> bytes:
    sample_rate = 24000
    silence = np.zeros(round(0.15 * sample_rate))
    time = np.arange(round(0.70 * sample_rate)) / sample_rate
    token = 0.4 * np.sin(2 * math.pi * 220 * time)
    token += 0.2 * np.sin(2 * math.pi * 880 * time)
    samples = np.concatenate((silence, token, silence))
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes((samples * 32767).astype("<i2").tobytes())
    return buffer.getvalue()


def test_confirmation_manifest_is_exactly_the_frozen_38_slots() -> None:
    manifest = build_confirmation_manifest()

    assert len(manifest) == 38
    assert len({item.slot_id for item in manifest}) == 38
    assert sum(item.kind == "reference" for item in manifest) == 2
    assert sum(item.kind == "contrast" for item in manifest) == 36
    assert {item.rule_id for item in manifest if item.kind == "contrast"} == {
        "ptbr.vowel.ih_to_i",
        "ptbr.vowel.ae_to_eh",
    }
    assert {item.shell for item in manifest if item.kind == "contrast"} == {
        "z_V_f",
        "k_V_sh",
        "v_V_p",
    }
    assert build_confirmation_manifest() == manifest

    for item in manifest:
        if item.kind == "reference":
            assert item.token == "bat"
            assert item.take in {3, 4}
            continue
        neutral_start, neutral_end = item.neutral_character_span or (0, 0)
        lens_start, lens_end = item.lens_character_span or (0, 0)
        if item.rule_id == "ptbr.vowel.ih_to_i":
            neutral, lens = "ih", "ee"
        else:
            neutral, lens = "a", "eh"
        pair_token = item.token
        if item.side == "neutral":
            assert pair_token[neutral_start:neutral_end] == neutral
        else:
            assert pair_token[lens_start:lens_end] == lens


def test_confirmation_protocol_hash_is_frozen() -> None:
    assert confirmation_protocol_record()["protocol_sha256"] == (
        "359e73f466adf896b5b92950c83811fda7aa76b0ff0254d2a6fbee2132dda67c"
    )


def _coherent_reference_records() -> list[dict]:
    f1 = {"i": 300, "u": 360, "ih": 600, "uh": 680, "eh": 880, "ae": 1060}
    f2 = {"u": 1250, "uh": 1300, "ae": 1550, "eh": 2000, "ih": 2200, "i": 2900}
    records = []
    for category in F1_ORDER:
        for take in (1, 2):
            records.append(
                {
                    "status": "ok",
                    "source_exact_token_match": True,
                    "audio_integrity_pass": True,
                    "stimulus": {
                        "kind": "reference",
                        "reference_category": category,
                        "take": take,
                    },
                    "analysis": {
                        "valid_formant_frame_count": 10,
                        "f1_hz": f1[category] + take - 1.5,
                        "f2_hz": f2[category] + take - 1.5,
                    },
                    "exclusion_reasons": [],
                }
            )
    return records


def test_internal_instrument_requires_ordering_and_broad_plausibility() -> None:
    records = _coherent_reference_records()
    passed = evaluate_internal_coherence(records)
    assert passed["passed"]
    assert passed["f1_order_low_to_high"] == list(F1_ORDER)
    assert passed["f2_order_low_to_high"] == list(F2_ORDER)

    for record in records:
        if record["stimulus"]["reference_category"] == "u":
            record["analysis"]["f1_hz"] = 200
    failed = evaluate_internal_coherence(records)
    assert not failed["passed"]
    assert not failed["f1_order_pass"]


def test_existing_66_pass_the_exploratory_internal_instrument_gate() -> None:
    records = load_v2_records()
    references = [r for r in records if r["stimulus"]["kind"] == "reference"]

    result = evaluate_internal_coherence(references)

    assert result["passed"]
    assert result["categories"]["ae"]["eligible_take_count"] == 2


def test_confirmation_renderer_uses_frozen_audio_contract(tmp_path: Path) -> None:
    stimulus = build_confirmation_manifest()[0]
    audio = SimpleNamespace(
        data=base64.b64encode(_tone_wav_bytes()).decode("ascii"),
        transcript=stimulus.token,
    )
    completion = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(audio=audio))],
        _request_id="req_v3",
        model="gpt-audio-1.5",
        usage={
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "completion_tokens_details": {"audio_tokens": 40},
        },
    )

    class Completions:
        def create(self, **kwargs):
            assert kwargs["model"] == "gpt-audio-1.5"
            assert kwargs["modalities"] == ["text", "audio"]
            assert kwargs["audio"] == {"voice": "marin", "format": "wav"}
            assert kwargs["store"] is False
            return completion

    client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    output = tmp_path / "v3.wav"

    record = _render_confirmation_slot(
        client=client,
        stimulus=stimulus,
        request_order=1,
        audio_path=output,
    )

    assert record["status"] == "ok"
    assert record["request_id"] == "req_v3"
    assert record["transcript_check"]["exact_token_match"]
    assert record["audio_integrity_pass"]
    assert output.is_file()
