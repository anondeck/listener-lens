from __future__ import annotations

from earshift_bakeoff.kokoro_validated_shell import (
    VALIDATED_LENS_SHELL,
    VALIDATED_NEUTRAL_SHELL,
    ValidatedShellPlanner,
)


def test_validated_shell_planner_preserves_length_and_repetition() -> None:
    plan = ValidatedShellPlanner.load().plan("The rabbit follows the rabbit.")
    target_words = [plan.words[index] for index in plan.target_word_indexes]
    assert len(target_words) == 2
    assert target_words[0].neutral_phone == target_words[1].neutral_phone
    assert target_words[0].lens_phone == target_words[1].lens_phone
    for word in target_words:
        target = word.target_offsets[0]
        assert word.neutral_phone[target - 2 : target + 2] == VALIDATED_NEUTRAL_SHELL
        assert word.lens_phone[target - 2 : target + 2] == VALIDATED_LENS_SHELL
        assert len(word.neutral_phone) == len(word.source_phone)
        assert len(word.lens_phone) == len(word.source_phone)


def test_validated_shell_still_uses_complete_global_gates() -> None:
    plan = ValidatedShellPlanner.load().plan("We drift slowly past quiet fields.")
    assert plan.gate_summary.espeak_gate_pass is True
    assert plan.gate_summary.kokoro_phone_gate_pass is True
    assert plan.gate_summary.exact_plan_representable is True
    assert plan.target_occurrence_count == 1
