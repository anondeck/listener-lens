from __future__ import annotations

import pytest

from earshift_bakeoff.bilingual_vowel_replicated_anchors import (
    TRAINING_SEEDS,
    aggregate_replicated_anchor_cell,
    aggregate_replicated_anchor_occurrence,
    validate_seed_order,
)


def _record(label: str, *, cosine: float = 1.0) -> dict:
    return {
        "classification": label,
        "directional_pass": label in ("exact_category_pass", "directional_only_pass"),
        "exact_category_pass": label == "exact_category_pass",
        "direction_cosine": cosine,
    }


def test_seed_order_is_unique_bounded_and_stable() -> None:
    assert validate_seed_order(TRAINING_SEEDS) == TRAINING_SEEDS
    with pytest.raises(ValueError, match="unique"):
        validate_seed_order((1, 1))
    with pytest.raises(ValueError, match="unique"):
        validate_seed_order((-1,))
    with pytest.raises(ValueError, match="unique"):
        validate_seed_order((True,))


def test_occurrence_requires_two_exact_seed_pairs_and_no_reversal() -> None:
    exact = _record("exact_category_pass")
    fail = _record("fail", cosine=0.2)
    candidate = _record("directional_only_pass")

    passed = aggregate_replicated_anchor_occurrence(
        natural_seed_records=[exact, exact, fail], candidate_record=candidate
    )
    reversed_pair = aggregate_replicated_anchor_occurrence(
        natural_seed_records=[exact, exact, _record("fail", cosine=-0.1)],
        candidate_record=candidate,
    )
    insufficient = aggregate_replicated_anchor_occurrence(
        natural_seed_records=[exact, fail, fail], candidate_record=candidate
    )

    assert passed["classification"] == "directional_only_pass"
    assert passed["anchor_validation_pass"] is True
    assert reversed_pair["classification"] == "anchor_validation_fail"
    assert insufficient["classification"] == "anchor_validation_fail"


def test_cell_requires_every_anchor_and_candidate_occurrence() -> None:
    exact = _record("exact_category_pass")
    natural = [exact, exact, exact]
    occurrence = aggregate_replicated_anchor_occurrence(
        natural_seed_records=natural, candidate_record=exact
    )
    anchor_only = aggregate_replicated_anchor_occurrence(
        natural_seed_records=natural, candidate_record=None
    )

    assert aggregate_replicated_anchor_cell([occurrence] * 4)["classification"] == (
        "exact_category_pass"
    )
    assert aggregate_replicated_anchor_cell([anchor_only] * 4)["classification"] == (
        "anchors_only"
    )
