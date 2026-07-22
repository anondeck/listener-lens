from __future__ import annotations

import pytest

from earshift_bakeoff.controlled_vowel_synthesis_v2 import (
    vowel_stress_context_columns,
)
from earshift_bakeoff.kokoro_synthesis import KokoroSynthesisError


def test_stressed_vowel_replaces_stress_and_complete_nasal_unit() -> None:
    neutral = tuple("bˈãz bˈæz")
    lens = tuple("bˈæ̃z bˈɛz")

    assert vowel_stress_context_columns(neutral, lens, (3, 4, 9), (3, 9)) == (
        2,
        3,
        4,
        8,
        9,
    )


def test_unstressed_vowel_does_not_absorb_unrelated_context() -> None:
    neutral = tuple("baz")
    lens = tuple("bɛz")

    assert vowel_stress_context_columns(neutral, lens, (2,), (2,)) == (2,)


def test_vowel_expansion_cannot_cross_a_word_boundary() -> None:
    with pytest.raises(KokoroSynthesisError, match="word boundary"):
        vowel_stress_context_columns(tuple("b a"), tuple("b ɛ"), (2,), (2,))


def test_unchanged_vowel_unit_cannot_create_an_intervention() -> None:
    with pytest.raises(KokoroSynthesisError, match="no changed model column"):
        vowel_stress_context_columns(tuple("bˈaz"), tuple("bˈaz"), (3,), ())
