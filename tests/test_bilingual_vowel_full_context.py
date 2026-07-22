from __future__ import annotations

from earshift_bakeoff.bilingual_vowel_full_context import (
    VOWEL_FULL_CONTEXT_CANDIDATE_VERSION,
)
from earshift_bakeoff.controlled_vowel_full_context import (
    CONTROLLED_VOWEL_FULL_CONTEXT_VERSION,
)


def test_full_context_candidate_versions_are_explicit() -> None:
    assert CONTROLLED_VOWEL_FULL_CONTEXT_VERSION == ("controlled-vowel-full-context-v1")
    assert VOWEL_FULL_CONTEXT_CANDIDATE_VERSION == (
        "vowel-full-context-neutral-excitation-v1"
    )
