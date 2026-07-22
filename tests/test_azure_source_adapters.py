from __future__ import annotations

import pytest

from earshift_bakeoff.azure_source_adapters import (
    ESPEAK_LANGUAGE_BY_LOCALE,
    EspeakSourceAdapter,
)
from earshift_bakeoff.bilingual_vowel_engine import BilingualVowelEngineError

SAMPLES = {
    "es-ES": ("El gato duerme en la casa", 6),
    "fr-FR": ("Le chat dort dans la maison", 6),
    "de-DE": ("Die Katze schläft im Haus", 5),
    "it-IT": ("Il gatto dorme nella casa", 5),
}


@pytest.mark.parametrize("locale", sorted(SAMPLES))
def test_adapter_aligns_one_phone_group_per_word(locale: str) -> None:
    text, expected_words = SAMPLES[locale]
    analysis = EspeakSourceAdapter.load(locale).analyze(text)
    assert analysis.language_id == locale
    assert len(analysis.words) == expected_words
    assert [word.source for word in analysis.words] == text.split()
    assert all(word.phone for word in analysis.words)
    # The separator plan must rebuild the phrase phones exactly, which is what
    # keeps a word's lens phones from drifting onto its neighbour.
    assert (
        analysis.compose([word.phone for word in analysis.words])
        == analysis.source_phonemes
    )


def test_accented_forms_stay_single_source_words() -> None:
    analysis = EspeakSourceAdapter.load("de-DE").analyze("Die Katze schläft")
    assert [word.source for word in analysis.words] == ["Die", "Katze", "schläft"]


def test_cross_word_context_is_preserved_by_phrase_phonemisation() -> None:
    # French nasal vowels survive because the whole phrase is phonemised in
    # one call rather than word by word.
    analysis = EspeakSourceAdapter.load("fr-FR").analyze("dans la maison")
    combining_tilde = "̃"
    assert any(combining_tilde in word.phone for word in analysis.words)


def test_unregistered_locale_fails_closed() -> None:
    with pytest.raises(BilingualVowelEngineError) as caught:
        EspeakSourceAdapter.load("ja-JP")
    assert caught.value.code == "unsupported_source_language"


def test_every_registered_locale_loads() -> None:
    for locale in ESPEAK_LANGUAGE_BY_LOCALE:
        assert EspeakSourceAdapter.load(locale).language_id == locale
