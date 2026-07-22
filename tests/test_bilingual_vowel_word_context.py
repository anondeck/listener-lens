from __future__ import annotations

from dataclasses import replace

import pytest

from earshift_bakeoff.bilingual_vowel_engine import (
    BilingualVowelEngineError,
    BilingualVowelPlan,
    CoverageReport,
)
from earshift_bakeoff.bilingual_vowel_word_context import (
    BilingualVowelWordContextRuntime,
    VOWEL_WORD_CONTEXT_CANDIDATE_VERSION,
)


def _coverage(**changes) -> CoverageReport:
    base = CoverageReport(
        source_vowel_occurrences=1,
        mapped_vowel_occurrences=1,
        changed_vowel_occurrences=1,
        identity_vowel_occurrences=0,
        directly_observed_occurrences=0,
        derived_or_structural_occurrences=1,
        acoustically_validated_changed_occurrences=0,
        pending_acoustic_changed_occurrences=1,
        changed_word_count=1,
        rules_used=("vowel",),
        source_consonant_occurrences=2,
        mapped_consonant_occurrences=0,
        changed_consonant_occurrences=0,
        identity_consonant_occurrences=0,
        directly_observed_consonant_occurrences=0,
        derived_or_structural_consonant_occurrences=0,
        acoustically_validated_changed_consonant_occurrences=0,
        pending_acoustic_changed_consonant_occurrences=0,
        consonant_rules_used=(),
        changed_prosody_occurrences=0,
        acoustically_validated_changed_prosody_occurrences=0,
        pending_acoustic_changed_prosody_occurrences=0,
        prosody_rules_used=(),
        changed_insertion_occurrences=0,
        acoustically_validated_changed_insertion_occurrences=0,
        pending_acoustic_changed_insertion_occurrences=0,
        insertion_rules_used=(),
    )
    return replace(base, **changes)


def test_word_context_candidate_version_is_explicit() -> None:
    assert VOWEL_WORD_CONTEXT_CANDIDATE_VERSION == (
        "vowel-word-context-plus-excitation-v1"
    )


def test_word_context_candidate_rejects_non_atomic_rule_families() -> None:
    plan = object.__new__(BilingualVowelPlan)
    object.__setattr__(plan, "coverage", _coverage(changed_consonant_occurrences=1))
    object.__setattr__(plan, "active_prosody_rule_ids", ())

    with pytest.raises(BilingualVowelEngineError) as error:
        BilingualVowelWordContextRuntime._require_atomic_vowel_plan(plan)

    assert error.value.code == "non_atomic_vowel_candidate"


def test_word_context_candidate_accepts_one_isolated_vowel_family() -> None:
    plan = object.__new__(BilingualVowelPlan)
    object.__setattr__(plan, "coverage", _coverage())
    object.__setattr__(plan, "active_prosody_rule_ids", ())

    BilingualVowelWordContextRuntime._require_atomic_vowel_plan(plan)
