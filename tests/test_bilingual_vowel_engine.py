from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import unicodedata

import pytest

from earshift_bakeoff.bilingual_vowel_engine import (
    BILINGUAL_ENGINE_VERSION,
    BilingualVowelEngineError,
    BilingualVowelPlanner,
    SourceAnalysis,
    SourceWord,
    load_profiles,
)
from earshift_bakeoff.listener_lens import NonceDecision


class _NonceChecker:
    enabled = True

    def check(
        self, surface: str, language: str, previous_surface: str | None
    ) -> NonceDecision:
        assert surface.isascii() and surface.isalpha()
        assert language in {"en", "pt"}
        return NonceDecision(True, "nˈɑns", None)


class _PhoneIndex:
    def phone_match(self, phone: str) -> bool:
        return False


@dataclass
class _Adapter:
    language_id: str
    words: tuple[tuple[str, str], ...]
    punctuation: str = "."

    def analyze(self, normalized_text: str) -> SourceAnalysis:
        separators = [""] + [" "] * (len(self.words) - 1) + [self.punctuation]
        source_words = tuple(
            SourceWord(index, source, unicodedata.normalize("NFD", phone))
            for index, (source, phone) in enumerate(self.words)
        )
        source_phone = "".join(
            value
            for index, word in enumerate(source_words)
            for value in (word.phone, separators[index + 1])
        )
        return SourceAnalysis(
            language_id=self.language_id,
            normalized_text=normalized_text,
            source_phonemes=source_phone,
            words=source_words,
            phone_separators=tuple(separators),
        )


def _vocab(profiles: dict[str, dict]) -> set[str]:
    symbols = set(' ;:,.!?—…"()“”ˈˌːʰʲ̃')
    symbols.update(
        "AIOSQTWYᵊaeiouyɑɐɒæɔəɚɛɜɨɪɯʊʌᵻɤøœ"
    )
    symbols.update(
        "bcdfghjklmnpqrstvwxyzɖðʤʥʦʧʨɟɡŋɲɳɴɸθɹɾɻɽʁʂʃʈʋʎʒʔʝɕɗçβɣχɥɰʣ"
    )
    for profile in profiles.values():
        for rule in profile["vowel_rules"]:
            symbols.update(unicodedata.normalize("NFD", rule["source"]))
            symbols.update(unicodedata.normalize("NFD", rule["target"]))
    return symbols


def _planner(
    profile: dict,
    words: tuple[tuple[str, str], ...],
    *,
    punctuation: str = ".",
) -> BilingualVowelPlanner:
    profiles = load_profiles()
    return BilingualVowelPlanner(
        profile=profile,
        adapter=_Adapter(profile["source_language"], words, punctuation),
        model_vocab=_vocab(profiles),
        nonce_checker=_NonceChecker(),
        phone_indexes=(_PhoneIndex(),),
    )


def test_profiles_cover_both_directions_with_separate_rule_evidence() -> None:
    profiles = load_profiles()

    assert set(profiles) == {
        "en-US-to-pt-BR-vowels-v1",
        "pt-BR-to-en-US-vowels-v1",
    }
    assert profiles["en-US-to-pt-BR-vowels-v1"]["source_language"] == "en-US"
    assert profiles["pt-BR-to-en-US-vowels-v1"]["source_language"] == "pt-BR"
    assert {
        rule["evidence_tier"]
        for profile in profiles.values()
        for rule in profile["vowel_rules"]
    } >= {"direct_assimilation", "derived_nearest_listener_category"}


def test_selected_voice_is_language_safe_and_part_of_the_plan_hash() -> None:
    profiles = load_profiles()
    english = profiles["en-US-to-pt-BR-vowels-v1"]
    words = (("Cat", "kˈæt"), ("mat", "mˈæt"))

    heart = _planner({**english, "voice_id": "af_heart"}, words).plan("Cat mat.")
    michael = _planner({**english, "voice_id": "am_michael"}, words).plan(
        "Cat mat."
    )

    assert heart.voice_id == "af_heart"
    assert michael.voice_id == "am_michael"
    assert heart.voice_registry_version == michael.voice_registry_version
    assert heart.voice_registry_sha256 == michael.voice_registry_sha256
    assert heart.plan_sha256 != michael.plan_sha256

    with pytest.raises(BilingualVowelEngineError) as exc_info:
        _planner({**english, "voice_id": "pm_alex"}, words)
    assert exc_info.value.code == "unsupported_product_voice"


def test_portuguese_product_voices_share_rules_but_not_plan_identity() -> None:
    portuguese = load_profiles()["pt-BR-to-en-US-vowels-v1"]
    words = (("pato", "pˈatʊ"), ("bonito", "bˈonitʊ"))

    alex = _planner({**portuguese, "voice_id": "pm_alex"}, words).plan(
        "pato bonito."
    )
    dora = _planner({**portuguese, "voice_id": "pf_dora"}, words).plan(
        "pato bonito."
    )

    assert alex.voice_id == "pm_alex"
    assert dora.voice_id == "pf_dora"
    assert alex.plan_sha256 != dora.plan_sha256


@pytest.mark.parametrize(
    "profile_id,source_word",
    [
        ("en-US-to-pt-BR-vowels-v1", "Polysyllabic"),
        ("pt-BR-to-en-US-vowels-v1", "Polissilábica"),
    ],
)
def test_every_profile_vowel_rule_is_consumed_without_a_coverage_hole(
    profile_id: str, source_word: str
) -> None:
    profile = load_profiles()[profile_id]
    # Consonant separators prevent two neighboring one-symbol rules from being
    # misread as an unintended composite source.
    phone = "bˈ" + "d".join(rule["source"] for rule in profile["vowel_rules"]) + "k"
    planner = _planner(profile, ((source_word, phone),))

    plan = planner.plan(source_word + ".")

    assert plan.engine_version == BILINGUAL_ENGINE_VERSION
    assert plan.coverage.source_vowel_occurrences == len(profile["vowel_rules"])
    assert plan.coverage.mapped_vowel_occurrences == len(profile["vowel_rules"])
    assert set(plan.coverage.rules_used) == {
        rule["id"] for rule in profile["vowel_rules"]
    }
    assert plan.coverage.changed_vowel_occurrences > 0
    assert plan.pair_plan() is not None


def test_repetition_punctuation_and_multiple_vowels_are_preserved() -> None:
    profile = load_profiles()["en-US-to-pt-BR-vowels-v1"]
    planner = _planner(
        profile,
        (("Catamaran", "kˈætəmæɹən"), ("catamaran", "kˈætəmæɹən")),
        punctuation="!",
    )

    plan = planner.plan("Catamaran catamaran!")

    assert plan.neutral_script.endswith("!")
    assert plan.lens_script.endswith("!")
    assert plan.words[0].neutral_phone == plan.words[1].neutral_phone
    assert plan.words[0].lens_phone == plan.words[1].lens_phone
    assert plan.words[0].neutral_surface == plan.words[1].neutral_surface
    assert plan.words[0].lens_surface == plan.words[1].lens_surface
    assert plan.coverage.source_vowel_occurrences == 8
    assert plan.coverage.changed_vowel_occurrences == 8
    assert plan.target_word_indexes == (0, 1)
    assert plan.gates.repeated_word_invariant_pass is True


def test_single_vowel_function_word_gets_nonsyllabic_padding_not_vowel_erasure() -> None:
    profile = load_profiles()["en-US-to-pt-BR-vowels-v1"]
    planner = _planner(profile, (("a", "ɐ"),))

    plan = planner.plan("a.")

    word = plan.words[0]
    assert word.inserted_consonant_count == 3
    assert word.neutral_phone.count("ɐ") == 1
    assert word.lens_phone.count("ɐ") == 1
    assert sum(character in {"a", "e", "i", "o", "u"} for character in word.neutral_surface) >= 1


def test_unknown_source_phone_fails_closed_instead_of_claiming_full_coverage() -> None:
    profile = load_profiles()["en-US-to-pt-BR-vowels-v1"]
    planner = _planner(profile, (("odd", "b☃d"),))

    with pytest.raises(BilingualVowelEngineError) as exc_info:
        planner.plan("odd.")

    assert exc_info.value.code == "unsupported_source_phone"


def test_no_changed_categories_disables_comparison() -> None:
    profile = load_profiles()["en-US-to-pt-BR-vowels-v1"]
    planner = _planner(profile, (("beet", "bˈit"),))

    plan = planner.plan("beet.")

    assert plan.coverage.source_vowel_occurrences == 1
    assert plan.coverage.identity_vowel_occurrences == 1
    assert plan.comparison_available is False
    assert plan.target_word_indexes == ()
    assert plan.pair_plan() is None


def test_plans_are_deterministic_under_concurrent_requests() -> None:
    profile = load_profiles()["pt-BR-to-en-US-vowels-v1"]
    planner = _planner(
        profile,
        (("pato", "pˈatʊ"), ("viu", "vˈiʊ"), ("pato", "pˈatʊ")),
    )
    expected = planner.plan("pato viu pato.").plan_sha256

    with ThreadPoolExecutor(max_workers=4) as executor:
        actual = list(executor.map(lambda _: planner.plan("pato viu pato.").plan_sha256, range(8)))

    assert actual == [expected] * 8
