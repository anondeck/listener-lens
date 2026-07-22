from __future__ import annotations

import pytest

from earshift_bakeoff.listener_lens import (
    LENS_RULES_PATH,
    ListenerLensEngine,
    ListenerLensError,
    NonceDecision,
)
from earshift_bakeoff.prosodic_carrier import (
    PROSODIC_CARRIER_VERSION,
    build_prosodic_carrier,
)


class FakeAnalyzer:
    IPA = {
        "the": "ðə",
        "cat": "kæt",
        "is": "ɪz",
        "good": "ɡʊd",
        "what": "wʌt",
        "a": "eɪ",
        "great": "ɡreɪt",
        "day": "deɪ",
        "it": "ɪt",
        "to": "tu",
        "catch": "kætʃ",
        "some": "sʌm",
        "sun": "sʌn",
        "that": "ðæt",
    }

    def phonemize_words(self, words, voice: str) -> list[str]:
        assert voice == "en-us"
        return [self.IPA[word.casefold()] for word in words]


class PermissiveChecker:
    @property
    def enabled(self) -> bool:
        return True

    def check(
        self, surface: str, language: str, previous_surface: str | None
    ) -> NonceDecision:
        assert language == "en"
        return NonceDecision(True, f"/{surface}/", None)

    def accepts(
        self, surface: str, language: str, previous_surface: str | None
    ) -> tuple[bool, str]:
        decision = self.check(surface, language, previous_surface)
        return decision.accepted, decision.predicted_ipa


def transformed(text: str):
    return ListenerLensEngine(
        rules_path=LENS_RULES_PATH,
        analyzer=FakeAnalyzer(),
        nonce_checker=PermissiveChecker(),
    ).transform(text, "en-to-pt-BR-vowel-lens")


def test_smoke_carrier_becomes_five_accentual_groups_without_losing_syllables() -> None:
    source = transformed("What a great day it is to catch some sun.")
    grouped = build_prosodic_carrier(source, PermissiveChecker())

    assert grouped.version == PROSODIC_CARRIER_VERSION == 1
    assert [group.source_word_indices for group in grouped.groups] == [
        (0, 1),
        (2,),
        (3, 4, 5, 6),
        (7, 8),
        (9,),
    ]
    assert grouped.source_word_count == 10
    assert grouped.group_count == 5
    assert grouped.total_syllables == sum(word.syllables for word in source.words)
    assert len(grouped.slots) == len(source.slots) == 1
    assert len(grouped.neutral_script.split()) == 5
    assert len(grouped.lens_script.split()) == 5
    assert grouped.neutral_script.endswith(".")
    assert grouped.lens_script.endswith(".")


def test_punctuation_forces_a_group_boundary_and_is_preserved() -> None:
    source = transformed("The cat, it is good.")
    grouped = build_prosodic_carrier(source, PermissiveChecker())

    assert [group.source_word_indices for group in grouped.groups] == [(0, 1), (2, 3, 4)]
    assert ", " in grouped.neutral_script
    assert grouped.neutral_script.endswith(".")
    assert grouped.lens_script.endswith(".")


def test_repeated_source_mappings_remain_identical_inside_different_groups() -> None:
    source = transformed("The cat the cat.")
    grouped = build_prosodic_carrier(source, PermissiveChecker())

    assert [group.source_word_indices for group in grouped.groups] == [(0, 1, 2), (3,)]
    assert source.words[0].neutral_surface == source.words[2].neutral_surface
    assert source.words[1].neutral_surface == source.words[3].neutral_surface
    weak_surface = source.words[0].neutral_surface
    content_surface = source.words[1].neutral_surface
    assert grouped.groups[0].neutral_surface.startswith(weak_surface + content_surface)
    assert grouped.groups[0].neutral_surface.endswith(weak_surface)
    assert grouped.groups[1].neutral_surface == content_surface


def test_rule_bearing_function_word_remains_its_own_content_head() -> None:
    source = transformed("That cat.")
    grouped = build_prosodic_carrier(source, PermissiveChecker())

    assert [group.source_word_indices for group in grouped.groups] == [(0,), (1,)]
    assert len(grouped.slots) == 2


class RejectCompositeChecker(PermissiveChecker):
    def check(
        self, surface: str, language: str, previous_surface: str | None
    ) -> NonceDecision:
        if previous_surface is None and len(surface) > 8:
            return NonceDecision(False, f"/{surface}/", "written_word")
        return super().check(surface, language, previous_surface)


def test_composite_opacity_failure_is_closed_not_silently_ungrouped() -> None:
    source = transformed("The cat is good.")

    with pytest.raises(ListenerLensError, match="prosodic group"):
        build_prosodic_carrier(source, RejectCompositeChecker())
