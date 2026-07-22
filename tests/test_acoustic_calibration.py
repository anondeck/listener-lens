from __future__ import annotations

import base64
import io
import json
import math
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from earshift_bakeoff.acoustic_calibration import (
    CALIBRATION_DEVELOPER_PROMPT,
    RULE_SPECS,
    _render_slot,
    analyze_wav,
    build_calibration_messages,
    build_manifest,
    classify_calibration,
    estimated_cost_usd,
    exclusion_reasons,
    protocol_record,
    run_acoustic_calibration,
)


def _tone_wav_bytes(f1: float = 500, f2: float = 1500) -> bytes:
    sample_rate = 24000
    silence = np.zeros(round(0.15 * sample_rate))
    time = np.arange(round(0.70 * sample_rate)) / sample_rate
    token = 0.45 * np.sin(2 * math.pi * f1 * time)
    token += 0.22 * np.sin(2 * math.pi * f2 * time)
    samples = np.concatenate((silence, token, silence))
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes((np.clip(samples, -0.99, 0.99) * 32767).astype("<i2").tobytes())
    return buffer.getvalue()


def test_frozen_manifest_has_exact_inventory_and_hash() -> None:
    manifest = build_manifest()

    assert len(manifest) == 66
    assert len({item.slot_id for item in manifest}) == 66
    assert sum(item.kind == "reference" for item in manifest) == 12
    assert sum(item.kind == "contrast" for item in manifest) == 54
    assert protocol_record()["protocol_sha256"] == (
        "2860bb2b01f898aaed37fa62a27ec40b0c43a62bc68db36246c6ddddd750f748"
    )
    assert build_manifest() == manifest


def test_calibration_payload_keeps_token_as_json_data() -> None:
    stimulus = build_manifest()[0]
    messages = build_calibration_messages(stimulus)

    assert messages[0] == {
        "role": "developer",
        "content": CALIBRATION_DEVELOPER_PROMPT,
    }
    payload = json.loads(messages[1]["content"])
    assert payload["script"] == stimulus.token
    assert payload["task"] == "verbatim_isolated_calibration_render"
    assert stimulus.token not in messages[0]["content"]


def test_midpoint_lpc_recovers_synthetic_formants(tmp_path: Path) -> None:
    path = tmp_path / "synthetic-vowel.wav"
    path.write_bytes(_tone_wav_bytes())

    analysis = analyze_wav(path)

    assert analysis["duration_s"] == 1.0
    assert analysis["valid_formant_frame_fraction"] == 1.0
    assert analysis["f1_hz"] == pytest.approx(500, abs=30)
    assert analysis["f2_hz"] == pytest.approx(1500, abs=40)


def test_frozen_exclusions_are_mechanical() -> None:
    analysis = {
        "decoded_sample_count": 100,
        "duration_s": 2.51,
        "clipped_fraction": 0.001,
        "active_duration_s": 0.099,
        "valid_formant_frame_count": 4,
        "valid_formant_frame_fraction": 0.59,
    }

    reasons = exclusion_reasons(status="ok", transcript_exact=False, analysis=analysis)

    assert reasons == [
        "provider_transcript_mismatch",
        "duration_outside_0.25_to_2.50_s",
        "clipped_fraction_at_least_0.001",
        "missing_or_short_active_interval",
        "fewer_than_5_valid_formant_frames",
        "fewer_than_60_percent_valid_formant_frames",
    ]


def test_render_slot_records_usage_and_audio_without_retries(tmp_path: Path) -> None:
    stimulus = build_manifest()[0]
    audio = SimpleNamespace(
        data=base64.b64encode(_tone_wav_bytes()).decode("ascii"),
        transcript=stimulus.token,
    )
    completion = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(audio=audio))],
        _request_id="req_calibration",
        model="gpt-audio-1.5",
        usage={
            "prompt_tokens": 100,
            "prompt_tokens_details": {"audio_tokens": 0},
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
    output = tmp_path / "slot.wav"

    record = _render_slot(
        client=client,
        stimulus=stimulus,
        request_order=1,
        audio_path=output,
    )

    assert record["status"] == "ok"
    assert record["request_id"] == "req_calibration"
    assert record["transcript_check"]["exact_token_match"]
    assert record["exclusion_reasons"] == []
    assert record["estimated_cost_usd"] == pytest.approx(0.00291)
    assert output.is_file()


ANCHORS = {
    "ih": np.array([5.0, 11.0]),
    "i": np.array([3.0, 14.0]),
    "ae": np.array([8.0, 10.0]),
    "eh": np.array([6.5, 11.0]),
    "uh": np.array([5.0, 8.0]),
    "u": np.array([3.5, 7.0]),
}


def _synthetic_records(lens_fraction: float) -> list[dict]:
    rule_by_id = {rule["rule_id"]: rule for rule in RULE_SPECS}
    shell_offsets = {
        "n_V_sh": np.array([0.0, 0.0]),
        "z_V_f": np.array([0.04, -0.03]),
        "v_V_m": np.array([-0.03, 0.04]),
    }
    records = []
    for stimulus in build_manifest():
        jitter = np.array([(stimulus.take - 1.5) * 0.01, (1.5 - stimulus.take) * 0.01])
        if stimulus.kind == "reference":
            point = ANCHORS[stimulus.reference_category] + jitter
        else:
            rule = rule_by_id[stimulus.rule_id]
            source = ANCHORS[rule["source_category"]]
            target = ANCHORS[rule["target_category"]]
            fraction = 0.0 if stimulus.side == "neutral" else lens_fraction
            point = source + fraction * (target - source)
            point = point + shell_offsets[stimulus.shell] + jitter
        records.append(
            {
                "stimulus": {
                    **stimulus.__dict__,
                },
                "analysis": {
                    "f1_bark": float(point[0]),
                    "f2_bark": float(point[1]),
                },
                "exclusion_reasons": [],
            }
        )
    return records


def test_classifier_distinguishes_exact_directional_and_fail() -> None:
    exact = classify_calibration(_synthetic_records(lens_fraction=1.0))
    directional = classify_calibration(_synthetic_records(lens_fraction=0.30))
    failed = classify_calibration([])

    assert {result["outcome"] for result in exact["rules"].values()} == {
        "exact-category pass"
    }
    assert {result["outcome"] for result in directional["rules"].values()} == {
        "directional-only pass"
    }
    assert {result["outcome"] for result in failed["rules"].values()} == {"fail"}


def test_cost_uses_separate_text_and_audio_token_rates() -> None:
    usage = {
        "prompt_tokens": 100,
        "prompt_tokens_details": {"audio_tokens": 20},
        "completion_tokens": 50,
        "completion_tokens_details": {"audio_tokens": 40},
    }

    assert estimated_cost_usd(usage) == pytest.approx(0.00350)


def test_full_fake_run_uses_66_slots_once_and_is_resumable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import earshift_bakeoff.acoustic_calibration as calibration

    calls = 0

    class Completions:
        def create(self, **kwargs):
            nonlocal calls
            calls += 1
            token = json.loads(kwargs["messages"][-1]["content"])["script"]
            audio = SimpleNamespace(
                data=base64.b64encode(_tone_wav_bytes()).decode("ascii"),
                transcript=token,
            )
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(audio=audio))],
                _request_id=f"req-{calls}",
                model="gpt-audio-1.5",
                usage={
                    "prompt_tokens": 100,
                    "completion_tokens": 50,
                    "completion_tokens_details": {"audio_tokens": 40},
                },
            )

    fake_paths = SimpleNamespace(
        artifacts=tmp_path,
        run_dir=lambda run_id: tmp_path / "runs" / run_id,
    )
    monkeypatch.setattr(calibration, "Paths", lambda: fake_paths)
    client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))

    first = run_acoustic_calibration("fake-fixed-run", client=client)
    second = run_acoustic_calibration("fake-fixed-run", client=client)

    assert calls == 66
    assert first["logical_request_slots"] == 66
    assert first["completed_records"] == 66
    assert first["successful_requests"] == 66
    assert second == first
    run_dir = tmp_path / "acoustic-calibration" / "fake-fixed-run"
    assert len(list((run_dir / "slots").glob("*.json"))) == 66
    assert (run_dir / "results.csv").is_file()
    assert (run_dir / "analysis.json").is_file()
