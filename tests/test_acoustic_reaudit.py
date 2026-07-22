from __future__ import annotations

import io
import math
import wave
from pathlib import Path

import numpy as np
import pytest

from earshift_bakeoff.acoustic_calibration import build_manifest
from earshift_bakeoff.acoustic_reaudit import (
    HILLENBRAND_FEMALE_RANGES,
    MAXIMUM_FORMANT_HZ,
    MAX_NUMBER_OF_FORMANTS,
    PRE_EMPHASIS_FROM_HZ,
    TIME_STEP_S,
    VOWEL_CENTER_END_FRACTION,
    VOWEL_CENTER_START_FRACTION,
    WINDOW_LENGTH_S,
    analyze_wav_praat,
    validate_hillenbrand_anchors,
)


def _tone_wav_bytes() -> bytes:
    sample_rate = 24000
    silence = np.zeros(round(0.15 * sample_rate))
    time = np.arange(round(0.70 * sample_rate)) / sample_rate
    token = 0.45 * np.sin(2 * math.pi * 200 * time)
    samples = np.concatenate((silence, token, silence))
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes((samples * 32767).astype("<i2").tobytes())
    return buffer.getvalue()


def test_praat_burg_protocol_parameters_are_frozen() -> None:
    assert TIME_STEP_S == 0.005
    assert MAX_NUMBER_OF_FORMANTS == 5.0
    assert MAXIMUM_FORMANT_HZ == 5500.0
    assert WINDOW_LENGTH_S == 0.025
    assert PRE_EMPHASIS_FROM_HZ == 50.0
    assert (VOWEL_CENTER_START_FRACTION, VOWEL_CENTER_END_FRACTION) == (0.25, 0.75)


def test_praat_analysis_uses_middle_half_and_median(
    tmp_path: Path, monkeypatch
) -> None:
    path = tmp_path / "vowel.wav"
    path.write_bytes(_tone_wav_bytes())

    class FakeFormant:
        def xs(self):
            return np.arange(0.0025, 1.0, 0.005)

        def get_value_at_time(self, number, time):
            return (500 if number == 1 else 1500) + 100 * time

        def get_bandwidth_at_time(self, number, time):
            return 100.0

    class FakeSound:
        def __init__(self, samples, sampling_frequency):
            self.samples = samples
            self.sampling_frequency = sampling_frequency

        def to_formant_burg(self, **kwargs):
            assert kwargs == {
                "time_step": 0.005,
                "max_number_of_formants": 5.0,
                "maximum_formant": 5500.0,
                "window_length": 0.025,
                "pre_emphasis_from": 50.0,
            }
            return FakeFormant()

    import earshift_bakeoff.acoustic_reaudit as reaudit

    monkeypatch.setattr(reaudit.parselmouth, "Sound", FakeSound)
    analysis = analyze_wav_praat(path)

    measured_width = analysis["midpoint_end_s"] - analysis["midpoint_start_s"]
    assert measured_width == pytest.approx(analysis["active_duration_s"] * 0.5)
    midpoint = (analysis["midpoint_start_s"] + analysis["midpoint_end_s"]) / 2
    assert analysis["f1_hz"] == pytest.approx(500 + 100 * midpoint)
    assert analysis["f2_hz"] == pytest.approx(1500 + 100 * midpoint)
    assert analysis["valid_formant_frame_fraction"] == 1.0


def _anchor_records(outside: str | None = None) -> list[dict]:
    records = []
    for stimulus in build_manifest():
        if stimulus.kind != "reference":
            continue
        ranges = HILLENBRAND_FEMALE_RANGES[stimulus.reference_category]
        f1 = sum(ranges["f1"]) / 2
        f2 = sum(ranges["f2"]) / 2
        if outside == stimulus.reference_category:
            f2 = ranges["f2"][1] + 1
        records.append(
            {
                "stimulus": stimulus.__dict__,
                "analysis": {"f1_hz": f1, "f2_hz": f2},
                "exclusion_reasons": [],
            }
        )
    return records


def test_instrument_requires_all_six_anchor_medians_in_range() -> None:
    passed = validate_hillenbrand_anchors(_anchor_records())
    failed = validate_hillenbrand_anchors(_anchor_records(outside="ih"))

    assert passed["passed"]
    assert not failed["passed"]
    assert failed["failed_categories"] == ["ih"]
