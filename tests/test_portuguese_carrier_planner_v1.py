from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor

import pytest

from earshift_bakeoff.kokoro_specs import VOICE_SPECS_BY_ID
from earshift_bakeoff.listener_lens import NonceDecision
from earshift_bakeoff.portuguese_carrier_planner_v1 import (
    KOKORO_LANG_CODE,
    LANGUAGE_ID,
    PORTUGUESE_RENDERER_CANDIDATE_ENABLED,
    PORTUGUESE_KOKORO_CONSONANT_SYMBOLS,
    PORTUGUESE_KOKORO_VOWEL_SYMBOLS,
    PORTUGUESE_NASAL_COMPONENTS,
    PORTUGUESE_SMOKE_FIXTURE_ID_V1,
    PORTUGUESE_SMOKE_TARGET_PHONE_V1,
    PORTUGUESE_SMOKE_TEXT_V1,
    PRODUCTION_ROUTE_AVAILABLE,
    TARGET_ANALYSIS_SCOPE,
    PortugueseCarrierPlannerError,
    PortugueseCarrierPlannerV1,
    plan_portuguese_smoke_fixture_v1,
    portuguese_smoke_screening_receipt_v1,
)


_WORD_RE = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)?", flags=re.UNICODE)
_VOCAB = frozenset(
    " ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,!?;:'’ˈˌ̃õũɐɛɔæɪʊəɡŋɾʃʒxɲʎʧʤ"
)


class _G2P:
    language_id = LANGUAGE_ID
    kokoro_lang_code = KOKORO_LANG_CODE
    voice_id = "pf_dora"

    def __init__(self, phones: dict[str, str]) -> None:
        self.phones = {word.casefold(): phone for word, phone in phones.items()}

    def phonemize_words(self, words) -> list[str]:
        return [self.phones.get(word.casefold(), "bˈa") for word in words]

    def phonemize_phrase(self, text: str) -> str:
        word_phones = self.phonemize_words(_WORD_RE.findall(text))
        punctuation = "".join(character for character in text if character in ".,!?;:")
        return " ".join(word_phones) + punctuation


class _MandatoryGate:
    enabled = True

    def __init__(self, *, reject: bool = False) -> None:
        self.reject = reject
        self.calls: list[tuple[str, str, str | None]] = []

    def check(
        self, surface: str, language: str, previous_surface: str | None
    ) -> NonceDecision:
        self.calls.append((surface, language, previous_surface))
        if self.reject:
            return NonceDecision(False, "bˈa", "predicted_homophone")
        return NonceDecision(True, "bˈa", None)


class _NegativeNativeIndex:
    scope = "partial_positive_only_index"

    def phone_match(self, phone: str) -> bool:
        return False


class _OnePositiveNativeIndex:
    scope = "partial_positive_only_index"

    def __init__(self) -> None:
        self.used = False

    def phone_match(self, phone: str) -> bool:
        if not self.used:
            self.used = True
            return True
        return False


def _planner(
    phones: dict[str, str],
    *,
    mandatory_gate: _MandatoryGate | None = None,
    native_index: _NegativeNativeIndex | _OnePositiveNativeIndex | None = None,
) -> PortugueseCarrierPlannerV1:
    return PortugueseCarrierPlannerV1(
        voice_spec=VOICE_SPECS_BY_ID["pf_dora"],
        g2p=_G2P(phones),
        model_vocab=_VOCAB,
        mandatory_gate=mandatory_gate or _MandatoryGate(),
        native_positive_index=native_index or _NegativeNativeIndex(),
    )


def test_repeated_target_word_has_identical_mapping_and_preserves_punctuation() -> None:
    planner = _planner({"avó": "avˈɔ"})

    plan = planner.plan("Avó, avó!", target_phone="ɔ")

    assert plan.target_word_indexes == (0, 1)
    assert plan.target_word_count == 2
    assert plan.target_occurrence_count == 2
    assert plan.words[0].carrier_surface == plan.words[1].carrier_surface
    assert plan.words[0].carrier_phone == plan.words[1].carrier_phone
    assert re.sub(_WORD_RE, "", plan.carrier_script) == ", !"
    assert plan.carrier_script.endswith("!")


def test_weak_words_use_portuguese_weak_carrier_class() -> None:
    planner = _planner({"a": "a", "casa": "kˈazæ", "de": "ʤi", "ana": "ˈɐ̃næ"})

    plan = planner.plan("A casa de Ana.", target_phone="ɔ")

    assert [word.carrier_role for word in plan.words] == [
        "weak",
        "content",
        "weak",
        "content",
    ]
    assert all(word.carrier_surface.isalpha() for word in plan.words)


def test_multiple_targets_inside_one_word_are_counted_exactly() -> None:
    planner = _planner({"vovó": "vˈɔvɔ", "chegou": "ʃegˈo"})

    plan = planner.plan("Vovó chegou.", target_phone="ɔ")

    assert plan.target_word_indexes == (0,)
    assert plan.target_word_count == 1
    assert plan.target_occurrence_count == 2
    assert plan.words[0].target_occurrence_count == 2


def test_no_target_is_reported_without_a_listener_lens_claim() -> None:
    planner = _planner({"a": "a", "menina": "menˈinæ", "viu": "vˈiʊ"})

    plan = planner.plan("A menina viu.", target_phone="ɔ")

    assert plan.target_available is False
    assert plan.target_word_indexes == ()
    assert plan.target_occurrence_count == 0
    assert plan.candidate_enabled is False
    assert plan.production_route_available is False


def test_negative_native_index_never_overrides_mandatory_gate_rejection() -> None:
    planner = _planner(
        {"avó": "avˈɔ", "chegou": "ʃegˈo"},
        mandatory_gate=_MandatoryGate(reject=True),
        native_index=_NegativeNativeIndex(),
    )

    with pytest.raises(PortugueseCarrierPlannerError) as exc_info:
        planner.plan("Avó chegou.", target_phone="ɔ")

    assert exc_info.value.code == "candidate_search_exhausted"


def test_positive_native_index_can_only_add_a_rejection() -> None:
    planner = _planner(
        {"avó": "avˈɔ", "chegou": "ʃegˈo"},
        native_index=_OnePositiveNativeIndex(),
    )

    plan = planner.plan("Avó chegou.", target_phone="ɔ")

    assert (
        plan.gate_receipt.candidate_rejection_counts[
            "native_v1_positive_predicted_homophone"
        ]
        == 1
    )
    assert plan.gate_receipt.native_negative_used_for_clearance is False
    assert plan.gate_receipt.native_index_scope == "partial_positive_only_index"


def test_planner_is_deterministic_under_identical_and_mixed_concurrency() -> None:
    planner = _planner(
        {
            "avó": "avˈɔ",
            "chegou": "ʃegˈo",
            "a": "a",
            "menina": "menˈinæ",
            "viu": "vˈiʊ",
        }
    )
    cases = (
        ("Avó chegou.", "ɔ"),
        ("A menina viu.", "ɔ"),
        ("Avó chegou.", "ɔ"),
    )
    expected = [
        planner.plan(text, target_phone=phone).plan_sha256 for text, phone in cases
    ]

    with ThreadPoolExecutor(max_workers=3) as executor:
        actual = list(
            executor.map(
                lambda item: planner.plan(item[0], target_phone=item[1]).plan_sha256,
                cases,
            )
        )

    assert actual == expected


def test_explicit_portuguese_voice_and_disabled_candidate_are_invariant() -> None:
    with pytest.raises(PortugueseCarrierPlannerError) as exc_info:
        PortugueseCarrierPlannerV1(
            voice_spec=VOICE_SPECS_BY_ID["af_heart"],
            g2p=_G2P({}),
            model_vocab=_VOCAB,
            mandatory_gate=_MandatoryGate(),
            native_positive_index=_NegativeNativeIndex(),
        )

    assert exc_info.value.code == "voice_language_mismatch"
    assert PORTUGUESE_RENDERER_CANDIDATE_ENABLED is False
    assert PRODUCTION_ROUTE_AVAILABLE is False


def test_explicit_renderer_inventory_covers_pinned_portuguese_behaviors() -> None:
    assert {"ɐ̃", "õ", "ũ", "ʊ̃"} <= PORTUGUESE_NASAL_COMPONENTS
    assert {"ɾ", "x", "ɲ", "ʧ", "ʤ"} <= PORTUGUESE_KOKORO_CONSONANT_SYMBOLS
    assert {"æ", "ʊ", "y", "ɔ", "ɛ", "A", "I", "W", "Y"} <= (
        PORTUGUESE_KOKORO_VOWEL_SYMBOLS
    )


def test_stable_smoke_factory_emits_exact_gate_receipt() -> None:
    fixture_phones = {
        "a": "a",
        "avó": "avˈɔ",
        "comprou": "kõprˈo",
        "pão": "pˈɐ̃ʊ̃",
        "e": "i",
        "tia": "ʧˈiæ",
        "chamou": "ʃamˈo",
        "filha": "fˈiljæ",
    }
    planner = _planner(fixture_phones)

    plan = plan_portuguese_smoke_fixture_v1(planner)
    receipt = portuguese_smoke_screening_receipt_v1(planner)

    assert plan.normalized_text == PORTUGUESE_SMOKE_TEXT_V1
    assert plan.target_phone == PORTUGUESE_SMOKE_TARGET_PHONE_V1
    assert plan.target_analysis_scope == TARGET_ANALYSIS_SCOPE
    assert receipt["fixture_id"] == PORTUGUESE_SMOKE_FIXTURE_ID_V1
    assert receipt["carrier_script"] == plan.carrier_script
    assert receipt["carrier_phonemes"] == plan.carrier_phonemes
    assert receipt["gate_receipt"]["mandatory_written_espeak_gate_pass"] is True
    assert receipt["candidate_enabled"] is False
    assert receipt["production_route_available"] is False
