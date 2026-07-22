from __future__ import annotations

from dataclasses import dataclass
import unicodedata

from earshift_bakeoff.bilingual_listener_engine import (
    BILINGUAL_LISTENER_ENGINE_VERSION,
    BilingualListenerPlanner,
    load_listener_profiles,
)
from earshift_bakeoff.bilingual_vowel_engine import SourceAnalysis, SourceWord
from earshift_bakeoff.listener_lens import NonceDecision
from earshift_bakeoff.prosody_component import prosody_only_lens_phonemes


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
    symbols.update("AIOSQTWYᵊaeiouyɑɐɒæɔəɚɛɜɨɪɯʊʌᵻɤøœ")
    symbols.update("bcdfghjklmnpqrstvwxyzɖðʤʥʦʧʨɟɡŋɲɳɴɸθɹɾɻɽʁʂʃʈʋʎʒʔʝɕɗçβɣχɥɰʣ")
    for profile in profiles.values():
        for family in ("vowel_rules", "consonant_rules"):
            for rule in profile.get(family, ()):
                symbols.update(unicodedata.normalize("NFD", rule["source"]))
                symbols.update(unicodedata.normalize("NFD", rule["target"]))
    return symbols


def _planner(
    profile: dict, words: tuple[tuple[str, str], ...]
) -> BilingualListenerPlanner:
    profiles = load_listener_profiles()
    return BilingualListenerPlanner(
        profile=profile,
        adapter=_Adapter(profile["source_language"], words),
        model_vocab=_vocab(profiles),
        nonce_checker=_NonceChecker(),
        phone_indexes=(_PhoneIndex(),),
    )


def test_v2_profiles_layer_consonant_insertion_and_prosody_rules_on_vowels() -> None:
    profiles = load_listener_profiles()

    assert set(profiles) == {
        "en-US-to-pt-BR-listener-v2",
        "pt-BR-to-en-US-listener-v2",
    }
    assert all(
        profile["engine_version"] == BILINGUAL_LISTENER_ENGINE_VERSION
        for profile in profiles.values()
    )
    assert all(profile["vowel_rules"] for profile in profiles.values())
    assert all(profile["consonant_rules"] for profile in profiles.values())
    assert all(profile["prosody_rules"] for profile in profiles.values())


def test_listener_plans_preserve_selected_voice_in_derived_hash() -> None:
    profile = load_listener_profiles()["en-US-to-pt-BR-listener-v2"]
    words = (("think", "θˈɪŋk"), ("cat", "kˈæt"))

    heart = _planner({**profile, "voice_id": "af_heart"}, words).plan(
        "think cat."
    )
    michael = _planner({**profile, "voice_id": "am_michael"}, words).plan(
        "think cat."
    )

    assert heart.voice_id == "af_heart"
    assert michael.voice_id == "am_michael"
    assert heart.plan_sha256 != michael.plan_sha256


def test_english_interdentals_use_source_categories_only_on_neutral_side() -> None:
    profile = load_listener_profiles()["en-US-to-pt-BR-listener-v2"]
    planner = _planner(
        profile,
        (("think", "θˈɪŋk"), ("this", "ðɪs"), ("think", "θˈɪŋk")),
    )

    plan = planner.plan("think this think.")

    assert plan.engine_version == BILINGUAL_LISTENER_ENGINE_VERSION
    assert plan.coverage.changed_consonant_occurrences == 3
    assert plan.coverage.directly_observed_consonant_occurrences >= 3
    assert {
        rule for rule in plan.coverage.consonant_rules_used if rule.startswith("enpt.")
    } == {
        "enpt.theta_t",
        "enpt.eth_d",
    }
    assert "θ" in plan.words[0].neutral_phone
    assert "θ" not in plan.words[0].lens_phone
    assert "t" in plan.words[0].lens_phone
    assert "ð" in plan.words[1].neutral_phone
    assert "d" in plan.words[1].lens_phone
    assert plan.words[0].neutral_phone == plan.words[2].neutral_phone
    assert plan.words[0].lens_phone == plan.words[2].lens_phone
    assert plan.target_word_indexes == (0, 1, 2)


def test_portuguese_changed_consonants_are_separate_direct_and_derived_rules() -> None:
    profile = load_listener_profiles()["pt-BR-to-en-US-listener-v2"]
    planner = _planner(
        profile,
        (
            ("filha", "fˈiljæ"),
            ("minha", "mˌiɲæ"),
            ("rato", "xˈatʊ"),
            ("cara", "kˈaɾæ"),
        ),
    )

    plan = planner.plan("filha minha rato cara.")

    assert plan.coverage.changed_consonant_occurrences == 4
    assert "lj" in plan.words[0].neutral_phone
    assert "jj" in plan.words[0].lens_phone
    assert "ɲ" in plan.words[1].neutral_phone
    assert "n" in plan.words[1].lens_phone
    assert "x" in plan.words[2].neutral_phone
    assert "h" in plan.words[2].lens_phone
    assert "ɾ" in plan.words[3].neutral_phone
    assert "T" in plan.words[3].lens_phone
    direct = [
        occurrence
        for word in plan.words
        for occurrence in word.consonant_occurrences
        if occurrence.changed and occurrence.evidence_tier.startswith("direct_")
    ]
    derived = [
        occurrence
        for word in plan.words
        for occurrence in word.consonant_occurrences
        if occurrence.changed and not occurrence.evidence_tier.startswith("direct_")
    ]
    assert len(direct) == 1
    assert len(derived) == 3


def test_tap_rule_is_context_bounded() -> None:
    profile = load_listener_profiles()["pt-BR-to-en-US-listener-v2"]
    planner = _planner(profile, (("rta", "ɾtˈa"),))

    plan = planner.plan("rta.")

    assert all(
        occurrence.rule_id != "pten.tap_flap"
        for occurrence in plan.words[0].consonant_occurrences
    )


def test_english_initial_secondary_stress_bias_is_structure_bounded() -> None:
    profile = load_listener_profiles()["en-US-to-pt-BR-listener-v2"]
    planner = _planner(
        profile,
        (("banana", "bˌanənˈa"), ("normal", "nˈɔɹməl")),
    )

    plan = planner.plan("banana normal.")

    biased = plan.words[0]
    assert biased.neutral_phone.count("ˌ") == biased.lens_phone.count("ˌ") == 1
    assert biased.neutral_phone.count("ˈ") == biased.lens_phone.count("ˈ") == 1
    assert biased.neutral_phone.index("ˌ") == biased.lens_phone.index("ˈ")
    assert biased.neutral_phone.index("ˈ") == biased.lens_phone.index("ˌ")
    assert len(biased.prosody_occurrences) == 2
    assert plan.words[1].prosody_occurrences == ()
    assert plan.coverage.changed_prosody_occurrences == 2
    assert plan.coverage.prosody_rules_used == ("enpt.lexical_stress_initial_bias",)


def test_prosody_component_holds_every_segment_constant() -> None:
    profile = load_listener_profiles()["en-US-to-pt-BR-listener-v2"]
    planner = _planner(
        profile,
        (
            ("today", "tədˈA"),
            ("independence", "ˌɪndəpˈɛndᵊns"),
            ("matters", "mˈæTəɹz"),
        ),
    )

    plan = planner.plan("today independence matters.")
    lens, target_indexes = prosody_only_lens_phonemes(plan)

    assert target_indexes == (1,)
    assert lens != plan.neutral_phonemes
    assert lens.replace("ˈ", "").replace("ˌ", "") == plan.neutral_phonemes.replace(
        "ˈ", ""
    ).replace("ˌ", "")
    assert lens != plan.lens_phonemes


def test_epenthesis_uses_latent_slots_only_after_illegal_bp_codas() -> None:
    profile = load_listener_profiles()["en-US-to-pt-BR-listener-v2"]
    planner = _planner(
        profile,
        (("ask", "ˈæsk"), ("cat", "kˈæt"), ("open", "ˈOpən")),
    )

    plan = planner.plan("ask cat open.")
    eligible = planner.insertion_eligibility(plan)

    assert {(row["word_index"], row["context"]) for row in eligible} == {
        (0, "word_final_obstruent"),
        (1, "word_final_obstruent"),
    }
    # /s/ is a legal BP coda, so /sk/ does not receive an internal vowel.
    assert len(plan.words[0].insertion_occurrences) == 1
    for word in plan.words[:2]:
        occurrence = word.insertion_occurrences[0]
        assert (
            word.neutral_phone[occurrence.phone_offset]
            == word.neutral_phone[occurrence.phone_offset - 1]
        )
        assert word.lens_phone[occurrence.phone_offset] == "i"
    assert plan.words[2].insertion_occurrences == ()
    assert plan.coverage.changed_insertion_occurrences == 2
    assert plan.coverage.pending_acoustic_changed_insertion_occurrences == 2
    assert all(
        row["quality_status"] == "coarticulation_conditioning_required"
        for row in eligible
    )
    assert all(
        row["architecture_status"] == "controlled_latent_slot_v2_pending_validation"
        for row in eligible
    )
    assert all(len(word.neutral_phone) == len(word.lens_phone) for word in plan.words)


def test_stress_alignment_survives_an_earlier_latent_insertion_slot() -> None:
    profile = load_listener_profiles()["en-US-to-pt-BR-listener-v2"]
    planner = _planner(profile, (("abstract", "bˌaktɹˈakt"),))

    plan = planner.plan("abstract.")
    word = plan.words[0]

    assert word.insertion_occurrences
    assert len(word.prosody_occurrences) == 2
    for occurrence in word.prosody_occurrences:
        assert word.neutral_phone[occurrence.phone_offset] == occurrence.source
        assert word.lens_phone[occurrence.phone_offset] == occurrence.target
