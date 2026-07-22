from __future__ import annotations

from types import SimpleNamespace

import pytest

from earshift_bakeoff.bilingual_vowel_engine import BilingualVowelEngineError
from earshift_bakeoff.bilingual_vowel_occurrence_strength import (
    OccurrenceStrengthSpec,
    occurrence_strength_columns,
)


def _occurrence(
    *, rule_id: str, source: str = "ʊ", target: str = "u", changed: bool = True
) -> SimpleNamespace:
    return SimpleNamespace(
        rule_id=rule_id,
        source=source,
        target=target,
        changed=changed,
    )


def _word(*occurrences: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(
        neutral_phone="tʊk",
        vowel_occurrences=occurrences,
        consonant_occurrences=(),
        prosody_occurrences=(),
        insertion_occurrences=(),
    )


def _plan(*words: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(
        neutral_phonemes=" ".join(word.neutral_phone for word in words),
        words=words,
        target_word_indexes=tuple(range(len(words))),
    )


def test_occurrence_strength_maps_only_the_bound_target_word() -> None:
    first = _word(_occurrence(rule_id="enpt.uh_u"))
    second = _word(_occurrence(rule_id="enpt.uh_u"))
    plan = _plan(first, second)
    model = SimpleNamespace(vocab={symbol: index for index, symbol in enumerate(" tʊk")})

    strengths = occurrence_strength_columns(
        model=model,
        plan=plan,
        specs=(
            OccurrenceStrengthSpec(
                rule_id="enpt.uh_u",
                rule_occurrence_ordinal=1,
                expected_occurrence_index=1,
                expected_word_index=1,
                expected_source="ʊ",
                expected_target="u",
                strength=0.75,
            ),
        ),
    )

    assert strengths == {5: 0.75, 6: 0.75, 7: 0.75}


def test_occurrence_strength_fails_closed_on_binding_drift() -> None:
    plan = _plan(_word(_occurrence(rule_id="enpt.uh_u")))
    model = SimpleNamespace(vocab={symbol: index for index, symbol in enumerate(" tʊk")})

    with pytest.raises(
        BilingualVowelEngineError, match="no longer matches its frozen binding"
    ):
        occurrence_strength_columns(
            model=model,
            plan=plan,
            specs=(
                OccurrenceStrengthSpec(
                    rule_id="enpt.uh_u",
                    rule_occurrence_ordinal=0,
                    expected_occurrence_index=0,
                    expected_word_index=1,
                    expected_source="ʊ",
                    expected_target="u",
                    strength=0.75,
                ),
            ),
        )


def test_occurrence_strength_rejects_a_word_with_multiple_changed_segments() -> None:
    plan = _plan(
        _word(
            _occurrence(rule_id="enpt.uh_u"),
            _occurrence(rule_id="enpt.ah_a", source="ʌ", target="a"),
        )
    )
    model = SimpleNamespace(vocab={symbol: index for index, symbol in enumerate(" tʊk")})

    with pytest.raises(
        BilingualVowelEngineError, match="no longer matches its frozen binding"
    ):
        occurrence_strength_columns(
            model=model,
            plan=plan,
            specs=(
                OccurrenceStrengthSpec(
                    rule_id="enpt.uh_u",
                    rule_occurrence_ordinal=0,
                    expected_occurrence_index=0,
                    expected_word_index=0,
                    expected_source="ʊ",
                    expected_target="u",
                    strength=0.75,
                ),
            ),
        )
