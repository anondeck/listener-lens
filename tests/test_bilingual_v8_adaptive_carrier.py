from __future__ import annotations

from types import SimpleNamespace

from earshift_bakeoff.bilingual_v8_adaptive_carrier import (
    failed_vowel_mapping_keys,
)


def _plan() -> SimpleNamespace:
    words = (
        SimpleNamespace(
            source="Books",
            source_phone="bʊks",
            carrier_role="content",
        ),
        SimpleNamespace(
            source="Books",
            source_phone="bʊks",
            carrier_role="content",
        ),
    )
    return SimpleNamespace(words=words, profile_id="profile")


def _render(*, integrity: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        verification=SimpleNamespace(integrity_pass=integrity),
        alignment={
            "target_occurrences": [
                {"occurrence_index": 0, "word_index": 0, "segment_type": "vowel"},
                {"occurrence_index": 1, "word_index": 1, "segment_type": "vowel"},
            ]
        },
    )


def _acoustic(*, identity_false_positives: int = 0) -> dict:
    return {
        "integrity_pass": True,
        "identity_false_positive_count": identity_false_positives,
        "cells": [
            {
                "occurrences": [
                    {
                        "occurrence_index": 0,
                        "aggregate": {"directional_pass": False},
                    },
                    {
                        "occurrence_index": 1,
                        "aggregate": {"directional_pass": False},
                    },
                ]
            }
        ],
    }


def test_failed_repeated_occurrences_collapse_to_one_mapping_key() -> None:
    keys = failed_vowel_mapping_keys(
        plan=_plan(), render=_render(), acoustic=_acoustic()
    )

    assert len(keys) == 1
    assert keys[0].source_casefold == "books"
    assert keys[0].source_phone == "bʊks"


def test_identity_false_positive_makes_failure_nonretryable() -> None:
    assert (
        failed_vowel_mapping_keys(
            plan=_plan(),
            render=_render(),
            acoustic=_acoustic(identity_false_positives=1),
        )
        == ()
    )


def test_integrity_failure_makes_failure_nonretryable() -> None:
    assert (
        failed_vowel_mapping_keys(
            plan=_plan(), render=_render(integrity=False), acoustic=_acoustic()
        )
        == ()
    )
