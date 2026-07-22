from __future__ import annotations

import numpy as np
import pytest

from earshift_bakeoff.controlled_vowel_state_strength import (
    CONTROLLED_VOWEL_STATE_STRENGTH_VERSION,
    interpolate_context_state,
)
from earshift_bakeoff.kokoro_synthesis import KokoroSynthesisError


def test_state_strength_interpolation_has_frozen_endpoints() -> None:
    neutral = np.array([[[1.0, 2.0], [3.0, 4.0]]])
    lens = np.array([[[3.0, 6.0], [7.0, 12.0]]])

    assert np.array_equal(interpolate_context_state(neutral, lens, 1.0), lens)
    assert np.array_equal(
        interpolate_context_state(neutral, lens, 0.5),
        np.array([[[2.0, 4.0], [5.0, 8.0]]]),
    )
    assert np.array_equal(
        interpolate_context_state(neutral, lens, 1.5),
        np.array([[[4.0, 8.0], [9.0, 16.0]]]),
    )


@pytest.mark.parametrize("strength", [0.0, -1.0, float("inf"), float("nan")])
def test_state_strength_rejects_invalid_values(strength: float) -> None:
    values = np.zeros((1, 1, 2))

    with pytest.raises(KokoroSynthesisError):
        interpolate_context_state(values, values, strength)


def test_state_strength_version_is_explicit() -> None:
    assert CONTROLLED_VOWEL_STATE_STRENGTH_VERSION == (
        "controlled-vowel-state-strength-v1"
    )
