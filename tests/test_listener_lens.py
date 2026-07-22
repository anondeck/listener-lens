from __future__ import annotations

from pathlib import Path

import pytest

from earshift_bakeoff.listener_lens import (
    LENS_RULES_PATH,
    TRANSFORM_ALGORITHM_VERSION,
    ListenerLensCache,
    ListenerLensEngine,
    ListenerLensError,
    ListenerLensService,
    NonceDecision,
)


class FakeAnalyzer:
    IPA = {
        "the": "ðə",
        "kit": "kɪt",
        "cat": "kæt",
        "book": "bʊk",
        "is": "ɪz",
        "good": "ɡʊd",
        "today": "tədˈeɪ",
        "day": "deɪ",
        "and": "ænd",
        "in": "ɪn",
        "back": "bæk",
        "that": "ðæt",
        "what": "wʌt",
        "a": "eɪ",
        "great": "ɡreɪt",
        "it": "ɪt",
        "to": "tu",
        "catch": "kætʃ",
        "some": "sʌm",
        "sun": "sʌn",
    }

    def phonemize_words(self, words, voice: str) -> list[str]:
        assert voice == "en-us"
        return [self.IPA[word.casefold()] for word in words]


class RecordingNonceChecker:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str | None]] = []

    @property
    def enabled(self) -> bool:
        return True

    def accepts(
        self, surface: str, language: str, previous_surface: str | None
    ) -> tuple[bool, str]:
        self.calls.append((surface, language, previous_surface))
        return True, f"/{surface}/"


def engine(checker: RecordingNonceChecker | None = None) -> ListenerLensEngine:
    return ListenerLensEngine(
        rules_path=LENS_RULES_PATH,
        analyzer=FakeAnalyzer(),
        nonce_checker=checker or RecordingNonceChecker(),
    )


def test_applies_only_declared_vowel_collapses() -> None:
    result = engine().transform(
        "The kit cat book is good.", "en-to-pt-BR-vowel-lens"
    )

    assert [rule.rule_id for rule in result.applied_rules] == [
        "ptbr.vowel.ae_to_eh"
    ]
    assert [rule.occurrences for rule in result.applied_rules] == [1]
    assert [word.listener_ipa for word in result.words] == [
        "ðə",
        "kɪt",
        "kɛt",
        "bʊk",
        "ɪz",
        "ɡʊd",
    ]
    assert len(result.slots) == 1
    assert result.comparison_available is True
    assert result.neutral_script != result.lens_script
    assert result.api_calls_made == 0
    assert result.renderer_status == "endpoint_implemented_pending_live_smoke"
    assert "some_listener_rules_excluded_after_calibration" in result.warnings

    cat = next(word for word in result.words if word.source.casefold() == "cat")
    slot = cat.slots[0]
    assert slot.neutral_grapheme == "a"
    assert slot.lens_grapheme == "eh"
    assert slot.neutral_character_span[1] - slot.neutral_character_span[0] == 1
    assert slot.lens_character_span[1] - slot.lens_character_span[0] == 2


def _masked_surface(surface: str, slots, side: str) -> str:
    masked = list(surface)
    for slot in reversed(slots):
        start, end = (
            slot.neutral_character_span
            if side == "neutral"
            else slot.lens_character_span
        )
        masked[start:end] = ["•"]
    return "".join(masked)


def test_carrier_versions_differ_only_inside_declared_vowel_slots() -> None:
    result = engine().transform(
        "The kit cat book is good.", "en-to-pt-BR-vowel-lens"
    )

    for word in result.words:
        assert _masked_surface(word.neutral_surface, word.slots, "neutral") == _masked_surface(
            word.lens_surface, word.slots, "lens"
        )
        for slot in word.slots:
            neutral_start, neutral_end = slot.neutral_character_span
            lens_start, lens_end = slot.lens_character_span
            assert (
                word.neutral_surface[neutral_start:neutral_end]
                == slot.neutral_grapheme
            )
            assert word.lens_surface[lens_start:lens_end] == slot.lens_grapheme
        if not word.slots:
            assert word.neutral_surface == word.lens_surface

    assert result.words[0].source.casefold() == "the"
    assert result.words[0].neutral_surface == result.words[0].lens_surface
    assert all(
        word.neutral_surface.casefold() != word.source.casefold()
        for word in result.words
    )
    source_punctuation = "".join(
        character for character in result.original_text if not character.isalpha()
    )
    assert source_punctuation == "".join(
        character for character in result.neutral_script if not character.isalpha()
    )
    assert source_punctuation == "".join(
        character for character in result.lens_script if not character.isalpha()
    )


def test_nonce_output_is_deterministic_and_gate_checked() -> None:
    checker = RecordingNonceChecker()
    first = engine(checker).transform("The cat is good.", "en-to-pt-BR-vowel-lens")
    second = engine(RecordingNonceChecker()).transform(
        "The cat is good.", "en-to-pt-BR-vowel-lens"
    )

    assert first.neutral_script == second.neutral_script
    assert first.lens_script == second.lens_script
    assert first.cache_key == second.cache_key
    assert [word.carrier_role for word in first.words] == [
        "weak", "content", "weak", "content"
    ]
    assert first.weak_form_report.eligible_word_count == 2
    assert first.weak_form_report.eligible_mapping_count == 2
    assert first.weak_form_report.selected_mapping_count == 2
    assert first.weak_form_report.candidate_attempt_count == 2
    assert first.weak_form_report.candidate_gate_yield == 1.0
    assert first.weak_form_report.rejection_reason_counts == {}
    assert len(checker.calls) == 11
    assert all(language == "en" for _, language, _ in checker.calls)
    assert sum(previous is None for _, _, previous in checker.calls) == 6
    assert sum(previous is not None for _, _, previous in checker.calls) == 5
    assert first.neutral_script.casefold() != first.original_text.casefold()


class RejectFirstLensChecker(RecordingNonceChecker):
    def accepts(
        self, surface: str, language: str, previous_surface: str | None
    ) -> tuple[bool, str]:
        self.calls.append((surface, language, previous_surface))
        return len(self.calls) != 2, f"/{surface}/"


def test_rejecting_one_variant_regenerates_the_pair_together() -> None:
    checker = RejectFirstLensChecker()
    result = engine(checker).transform("Cat book.", "en-to-pt-BR-vowel-lens")
    first_word = result.words[0]

    assert first_word.pair_generation_attempt == 1
    assert checker.calls[2][0] == first_word.neutral_surface
    assert checker.calls[3][0] == first_word.lens_surface


def test_no_supported_rule_returns_no_audio_comparison() -> None:
    result = engine().transform("The day today.", "en-to-pt-BR-vowel-lens")

    assert result.comparison_available is False
    assert result.neutral_script == result.lens_script
    assert result.slots == []
    assert "no_supported_listener_rule" in result.warnings


@pytest.mark.parametrize(
    "text, message",
    [
        ("", "Enter one or two"),
        ("Only", "between 2 and"),
        ("Call 911 now.", "numbers"),
        ("Use GPT today.", "Acronyms"),
        ("One. Two. Three.", "at most 2"),
    ],
)
def test_rejects_inputs_outside_bounded_prototype(text: str, message: str) -> None:
    with pytest.raises(ListenerLensError, match=message):
        engine().transform(text, "en-to-pt-BR-vowel-lens")


def test_cache_prevents_duplicate_transformation(tmp_path: Path) -> None:
    service = ListenerLensService(
        engine=engine(), cache=ListenerLensCache(tmp_path / "cache")
    )

    first = service.transform("The cat is good.", "en-to-pt-BR-vowel-lens")
    second = service.transform("The cat is good.", "en-to-pt-BR-vowel-lens")

    assert first["runtime"] == {"cache_hit": False, "api_calls_made": 0}
    assert second["runtime"] == {"cache_hit": True, "api_calls_made": 0}
    assert first["cache_key"] == second["cache_key"]
    assert len(list((tmp_path / "cache").glob("*.json"))) == 1


def test_carrier_v5_reuses_mapping_for_repeated_source_words() -> None:
    result = engine().transform("The cat the cat.", "en-to-pt-BR-vowel-lens")

    assert TRANSFORM_ALGORITHM_VERSION == 5
    assert result.words[0].neutral_surface == result.words[2].neutral_surface
    assert result.words[0].lens_surface == result.words[2].lens_surface
    assert result.words[1].neutral_surface == result.words[3].neutral_surface
    assert result.words[1].lens_surface == result.words[3].lens_surface
    assert result.words[0].carrier_role == result.words[2].carrier_role == "weak"


def test_rule_bearing_function_word_is_forced_to_content() -> None:
    result = engine().transform("That cat.", "en-to-pt-BR-vowel-lens")

    assert [word.carrier_role for word in result.words] == ["content", "content"]
    assert result.words[0].slots
    assert result.words[0].neutral_surface != result.words[0].lens_surface
    assert result.weak_form_report.eligible_word_count == 0


class RejectFirstWeakCandidateChecker(RecordingNonceChecker):
    rejected = False

    def check(
        self, surface: str, language: str, previous_surface: str | None
    ) -> NonceDecision:
        self.calls.append((surface, language, previous_surface))
        if previous_surface is None and not self.rejected:
            self.rejected = True
            return NonceDecision(False, "", "written_word")
        return NonceDecision(True, f"/{surface}/", None)


def test_weak_candidate_report_preserves_rejection_reason_and_bounded_retry() -> None:
    result = engine(RejectFirstWeakCandidateChecker()).transform(
        "The cat.", "en-to-pt-BR-vowel-lens"
    )

    report = result.weak_form_report
    assert report.eligible_word_count == 1
    assert report.eligible_mapping_count == 1
    assert report.selected_mapping_count == 1
    assert report.candidate_attempt_count == 2
    assert report.rejected_attempt_count == 1
    assert report.rejection_reason_counts == {"written_word": 1}
    assert [attempt.outcome for attempt in report.attempts[:2]] == [
        "rejected", "accepted"
    ]


def test_current_smoke_roles_are_transform_owned_and_slot_safe() -> None:
    result = engine().transform(
        "What a great day it is to catch some sun.",
        "en-to-pt-BR-vowel-lens",
    )

    assert [word.carrier_role for word in result.words] == [
        "content", "weak", "content", "content", "weak",
        "weak", "weak", "content", "weak", "content",
    ]
    assert result.words[7].slots
    assert result.words[7].carrier_role == "content"
    assert all(not word.slots for word in result.words if word.carrier_role == "weak")


class AlternatingPronunciationAnalyzer:
    def phonemize_words(self, words, voice: str) -> list[str]:
        assert [word.casefold() for word in words] == ["read", "read"]
        assert voice == "en-us"
        return ["rɪd", "rɛd"]


def test_mapping_key_includes_pronunciation_and_rule_signature() -> None:
    checker = RecordingNonceChecker()
    result = ListenerLensEngine(
        rules_path=LENS_RULES_PATH,
        analyzer=AlternatingPronunciationAnalyzer(),
        nonce_checker=checker,
    ).transform("Read read.", "en-to-pt-BR-vowel-lens")

    assert result.words[0].source.casefold() == result.words[1].source.casefold()
    assert result.words[0].source_ipa != result.words[1].source_ipa
    assert len([call for call in checker.calls if call[2] is None]) == 4


class RejectFirstAdjacencyChecker(RecordingNonceChecker):
    rejected = False

    def accepts(
        self, surface: str, language: str, previous_surface: str | None
    ) -> tuple[bool, str]:
        self.calls.append((surface, language, previous_surface))
        if previous_surface is not None and not self.rejected:
            self.rejected = True
            return False, f"/{surface}/"
        return True, f"/{surface}/"


def test_global_adjacency_conflict_reresolves_all_implicated_mappings() -> None:
    checker = RejectFirstAdjacencyChecker()
    result = engine(checker).transform("The cat the cat.", "en-to-pt-BR-vowel-lens")

    assert result.words[0].pair_generation_attempt == 1
    assert result.words[1].pair_generation_attempt == 1
    assert result.words[0].neutral_surface == result.words[2].neutral_surface
    assert result.words[1].neutral_surface == result.words[3].neutral_surface


class AlwaysRejectAdjacencyChecker(RecordingNonceChecker):
    def accepts(
        self, surface: str, language: str, previous_surface: str | None
    ) -> tuple[bool, str]:
        self.calls.append((surface, language, previous_surface))
        return previous_surface is None, f"/{surface}/"


def test_bounded_adjacency_failure_never_breaks_repeated_word_invariant() -> None:
    with pytest.raises(ListenerLensError, match="preserving repeated"):
        engine(AlwaysRejectAdjacencyChecker()).transform(
            "The cat the cat.", "en-to-pt-BR-vowel-lens"
        )


def test_policy_keeps_live_renderer_disabled() -> None:
    status = ListenerLensService(
        engine=engine(), cache=ListenerLensCache(Path("/tmp/unused-listener-lens-test"))
    ).status()

    assert status["runtime_policy"]["api_renderer_enabled"] is False
    assert (
        status["runtime_policy"]["renderer_status"]
        == "endpoint_implemented_pending_live_smoke"
    )
    assert status["production_renderer"] == "gpt-audio-1.5"
    assert status["production_renderer_contract"] == "json-flow-v2"
