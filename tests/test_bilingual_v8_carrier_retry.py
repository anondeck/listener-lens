from __future__ import annotations

from collections import Counter
from types import SimpleNamespace

import pytest

from earshift_bakeoff.bilingual_v8_carrier_retry import (
    BilingualListenerPlannerV8CarrierRetry,
    CarrierRetrySpec,
)
from earshift_bakeoff.bilingual_vowel_engine import (
    BilingualVowelEngineError,
    CarrierAssignment,
    MappingKey,
)


def _assignment(attempt: int) -> CarrierAssignment:
    return CarrierAssignment(
        neutral_surface=f"n{attempt}",
        lens_surface=f"l{attempt}",
        neutral_phone=f"n{attempt}",
        lens_phone=f"l{attempt}",
        vowel_occurrences=(),
        consonant_occurrences=(),
        prosody_occurrences=(),
        insertion_occurrences=(),
        candidate_attempt=attempt,
        inserted_consonant_count=0,
    )


def _planner(minimums: dict[MappingKey, int]) -> SimpleNamespace:
    planner = SimpleNamespace(
        minimum_attempts=minimums,
        profile={"source_gate_language": "en"},
        phone_indexes=(),
    )
    planner._candidate = lambda key, attempt: _assignment(attempt)
    planner._isolated_reasons = lambda candidate: set()
    planner.nonce_checker = SimpleNamespace(
        check=lambda current, language, previous: SimpleNamespace(accepted=True)
    )
    return planner


def test_retry_resolver_starts_bound_mapping_at_frozen_minimum() -> None:
    first = MappingKey("one", "wʌn", "content", "profile")
    second = MappingKey("took", "tʊk", "content", "profile")
    planner = _planner({second: 4})

    assignments, attempts, adjacency = (
        BilingualListenerPlannerV8CarrierRetry._resolve_assignments(
            planner, (first, second), Counter()
        )
    )

    assert assignments[first].candidate_attempt == 0
    assert assignments[second].candidate_attempt == 4
    assert attempts == 2
    assert adjacency == 1


def test_retry_resolver_preserves_repeated_mapping_invariant() -> None:
    repeated = MappingKey("took", "tʊk", "content", "profile")
    planner = _planner({repeated: 7})

    assignments, _, adjacency = (
        BilingualListenerPlannerV8CarrierRetry._resolve_assignments(
            planner, (repeated, repeated), Counter()
        )
    )

    assert assignments[repeated].candidate_attempt == 7
    assert len(assignments) == 1
    assert adjacency == 1


def test_retry_resolver_fails_when_bound_mapping_is_absent() -> None:
    requested = MappingKey("took", "tʊk", "content", "profile")
    actual = MappingKey("book", "bʊk", "content", "profile")
    planner = _planner({requested: 4})

    with pytest.raises(BilingualVowelEngineError, match="absent"):
        BilingualListenerPlannerV8CarrierRetry._resolve_assignments(
            planner, (actual,), Counter()
        )


def test_retry_spec_normalizes_case_and_phone() -> None:
    spec = CarrierRetrySpec(
        source_casefold="TOOK",
        source_phone="tʊk",
        carrier_role="content",
        minimum_attempt=4,
    )

    assert spec.mapping_key("profile") == MappingKey(
        "took", "tʊk", "content", "profile"
    )
