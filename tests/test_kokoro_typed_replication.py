from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from earshift_bakeoff.kokoro_typed_replication import (
    _review_html,
    alignment_record,
    classify_measurements,
    localization_report,
    merge_sample_intervals,
)
from earshift_bakeoff.kokoro_typed_replication_protocol import protocol_record


def test_sample_interval_union_is_deterministic_and_clipped() -> None:
    assert merge_sample_intervals(((-2, 3), (2, 6), (8, 12), (6, 8), (20, 30)), 12) == (
        (0, 12),
    )


def test_alignment_maps_target_offset_to_stress_vowel_and_complete_word() -> None:
    phones = "zɪ ʒˈæʒ."
    model = SimpleNamespace(
        vocab={symbol: index + 1 for index, symbol in enumerate(set(phones))}
    )
    words = (
        SimpleNamespace(neutral_phone="zɪ", lens_phone="zɪ", target_offsets=()),
        SimpleNamespace(neutral_phone="ʒˈæʒ", lens_phone="ʒˈɛʒ", target_offsets=(2,)),
    )
    plan = SimpleNamespace(
        source_phonemes=phones,
        neutral_phonemes=phones,
        words=words,
        target_word_indexes=(1,),
        target_occurrence_count=1,
    )
    durations = (1,) * (len(phones) + 2)
    result = alignment_record(
        model=model,
        plan=plan,
        durations=durations,
        sample_count=len(durations) * 600,
    )
    occurrence = result["target_occurrences"][0]
    assert result["samples_per_alignment_frame"] == 600
    assert occurrence["measurement_interval"]["columns"] == [5, 6]
    assert occurrence["target_word_interval"]["columns"] == [4, 5, 6, 7]


def test_localization_gate_uses_squared_difference_energy() -> None:
    neutral = np.zeros(100, dtype=np.float64)
    lens = np.zeros(100, dtype=np.float64)
    lens[40:60] = 1.0
    passed = localization_report(
        neutral,
        lens,
        [{"start_sample": 40, "end_sample_exclusive": 60}],
        sample_rate_hz=100,
    )
    failed = localization_report(
        neutral,
        lens,
        [{"start_sample": 0, "end_sample_exclusive": 5}],
        sample_rate_hz=100,
    )
    assert passed["inside_difference_energy_fraction"] == 1.0
    assert passed["pass"] is True
    assert failed["inside_difference_energy_fraction"] == 0.0
    assert failed["pass"] is False


def test_acoustic_classifier_requires_categories_direction_and_magnitude() -> None:
    geometry = protocol_record()["replication_only_acoustic_gate"]["anchor_geometry"]
    neutral: dict[str, dict[str, float | bool]] = {}
    lens: dict[str, dict[str, float | bool]] = {}
    for key, anchor in geometry["families"].items():
        neutral[key] = {
            "f1_bark": anchor["full_ae_bark"][0],
            "f2_bark": anchor["full_ae_bark"][1],
            "plausibility_pass": True,
        }
        lens[key] = {
            "f1_bark": anchor["full_eh_bark"][0],
            "f2_bark": anchor["full_eh_bark"][1],
            "plausibility_pass": True,
        }
    assert classify_measurements(neutral, lens, geometry)["pass"] is True
    assert classify_measurements(lens, neutral, geometry)["pass"] is False


def test_public_review_hides_fixture_text_plan_roles_and_conditions() -> None:
    public = [
        {
            "trial_id": "comparison-01",
            "duration_s": 1.0,
            "target_intervals": [{"start_s": 0.2, "end_s": 0.4}],
            "sides": [
                {"side": "A", "audio": "review-audio/one-a.wav"},
                {"side": "B", "audio": "review-audio/one-b.wav"},
            ],
        }
    ]
    html = _review_html(public, "a" * 64)
    for hidden in (
        "The map rests",
        "zhazh",
        "zhehzh",
        "single-target",
        "identity-catch",
        "lens-candidate",
        "side_roles",
    ):
        assert hidden not in html
    for field in (
        "naturalness",
        "delivery",
        "meaning",
        "artifact",
        "difference_strength",
        "category_judgment",
        "confidence",
        "interference",
        "replay_count",
    ):
        assert field in html
    assert "TARGET NOW" in html


def test_protocol_freeze_precedes_novel_replication_audio() -> None:
    protocol = protocol_record()
    assert protocol["status"] == "frozen_before_any_novel_replication_fixture_audio"
