from __future__ import annotations

import pytest

from earshift_bakeoff.kokoro_strict_shell import StrictShellPlanner
from earshift_bakeoff.kokoro_typed_engine import KokoroTypedEngineError
from earshift_bakeoff.kokoro_validated_shell import (
    VALIDATED_LENS_SHELL,
    VALIDATED_NEUTRAL_SHELL,
)


def test_strict_shell_accepts_exact_target_words_and_repetition() -> None:
    planner = StrictShellPlanner.load()
    plan = planner.plan("We place one cap beside one cap.")
    target_words = [plan.words[index] for index in plan.target_word_indexes]
    assert len(target_words) == 2
    assert {word.neutral_phone for word in target_words} == {VALIDATED_NEUTRAL_SHELL}
    assert {word.lens_phone for word in target_words} == {VALIDATED_LENS_SHELL}


def test_strict_shell_fails_closed_for_extended_target_words() -> None:
    with pytest.raises(KokoroTypedEngineError) as caught:
        StrictShellPlanner.load().plan("We drift slowly past quiet fields.")
    assert caught.value.code == "strict_shell_unsupported_target_word"
