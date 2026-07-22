from __future__ import annotations

import numpy as np
import pytest

from earshift_bakeoff.kokoro_synthesis import (
    KokoroSynthesisError,
    PairPlan,
    _project_f0,
    _project_noise,
    _validate_plan,
    pcm16_bytes,
    target_word_columns,
)


class FakeModel:
    vocab = {symbol: index for index, symbol in enumerate(" abcdefghæɛˈ.,", start=1)}


def test_target_word_columns_merge_repeated_and_overlapping_requests() -> None:
    model = FakeModel()
    phonemes = "bˈæd bˈæd, fˈæd."

    assert target_word_columns(model, phonemes, (0, 1, 1)) == (
        1,
        2,
        3,
        4,
        6,
        7,
        8,
        9,
    )


def test_plan_rejects_changes_outside_complete_target_word() -> None:
    model = FakeModel()
    plan = PairPlan(
        source_phonemes="bˈæd fˈæd.",
        neutral_phonemes="bˈæd fˈæd.",
        lens_phonemes="bˈɛd fˈɛd.",
        target_word_indexes=(0,),
    )

    with pytest.raises(KokoroSynthesisError, match="escape"):
        _validate_plan(model, plan)


def test_plan_rejects_unknown_symbols_instead_of_silently_dropping_them() -> None:
    model = FakeModel()
    plan = PairPlan(
        source_phonemes="bˈæd ☃",
        neutral_phonemes="bˈæd ☃",
        lens_phonemes="bˈɛd ☃",
        target_word_indexes=(0,),
    )

    with pytest.raises(KokoroSynthesisError, match="unsupported"):
        _validate_plan(model, plan)


def test_pcm16_conversion_is_finite_clipped_and_deterministic() -> None:
    audio = np.asarray([-2.0, -0.5, 0.0, 0.5, 2.0])

    assert np.frombuffer(pcm16_bytes(audio), dtype="<i2").tolist() == [
        -32767,
        -16384,
        0,
        16384,
        32767,
    ]
    with pytest.raises(KokoroSynthesisError, match="finite"):
        pcm16_bytes(np.asarray([np.nan]))


class FakeProjection:
    def __init__(self, torch: object, channels: int) -> None:
        self.weight = torch.arange(1, channels + 1, dtype=torch.float32).view(
            1, channels, 1
        )
        self.bias = torch.tensor([0.25], dtype=torch.float32)


def test_frozen_projection_schedules_are_repeatable() -> None:
    torch = pytest.importorskip("torch")
    values = torch.arange(2 * 204, dtype=torch.float32).view(1, 2, 204) / 100
    module = FakeProjection(torch, channels=2)

    f0_first = _project_f0(module, values, torch)
    f0_second = _project_f0(module, values, torch)
    noise_first = _project_noise(module, values, torch)
    noise_second = _project_noise(module, values, torch)

    assert f0_first.shape == (1, 204)
    assert torch.equal(f0_first, f0_second)
    assert torch.equal(noise_first, noise_second)
