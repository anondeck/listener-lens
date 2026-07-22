from __future__ import annotations

from earshift_bakeoff.same_take import MAX_FORMANTS_FAMILY, align_vowel_core, bark


def test_same_take_family_is_frozen() -> None:
    assert MAX_FORMANTS_FAMILY == (5.0, 5.5, 6.0)


def test_bark_is_monotonic() -> None:
    assert bark(300) < bark(700) < bark(2500)


def test_alignment_snaps_a_voiced_core_to_samples() -> None:
    frames = []
    for index in range(100):
        time = (index + 0.5) * 0.005
        valid = 0.20 <= time <= 0.32
        frames.append(
            {
                "time_s": time,
                "f1_hz": 700.0 if valid else None,
                "f2_hz": 1700.0 if valid else None,
                "f3_hz": 2800.0 if valid else None,
                "f4_hz": 3900.0 if valid else None,
                "pitch_hz": 200.0 if valid else None,
                "rms": 1.0 - abs(time - 0.26) if valid else 0.0,
            }
        )
    result = align_vowel_core(
        frames,
        word_start_s=0.10,
        word_end_s=0.50,
        search_fraction=(0.15, 0.60),
        sample_rate_hz=24_000,
    )
    assert result["sample_count"] == 2400
    assert result["start_sample"] < result["end_sample_exclusive"]

