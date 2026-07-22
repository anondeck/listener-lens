"""Generic espeak-ng source adapters for Azure listener-lens locales.

The Kokoro-era adapters are bespoke per language: English rides misaki's
English G2P, Brazilian Portuguese rides a hand-built native G2P tied to a
Kokoro voice. Neither generalises, and the Azure lane does not need them to —
it needs per-word phones and nothing else.

`EspeakSourceAdapter` supplies exactly that for any espeak-ng language, so a
new listener-lens locale costs a registry entry rather than a new adapter.
Alignment is fail-closed in the same way as the Portuguese phrase adapter: if
espeak does not return one phone group per source word, the request raises
rather than silently mispairing a word with another word's phones.

Cross-word context is preserved because the whole sentence is phonemised in
one call; only the alignment is per word.
"""

from __future__ import annotations

import re
import threading
import unicodedata
from dataclasses import dataclass
from typing import Any

from .bilingual_vowel_engine import (
    _PHONE_WORD_RE,
    BilingualVowelEngineError,
    SourceAnalysis,
    SourceWord,
)

# Word pattern with unicode letter classes, so accented forms ("schläft",
# "duerme", "perché") count as single source words.
#
# Combining marks have to count as word-internal, not as boundaries. In the
# Indic scripts a vowel sign or virama is category Mn/Mc rather than a letter,
# so a letters-only class cut नमस्ते into नमस + त and reported five words as
# ten — which failed the one-phone-group-per-word check for every Devanagari,
# Telugu and Gujarati source. Latin behaviour is unchanged: this only widens
# the class, and it additionally keeps decomposed (NFD) accented forms whole.
_MARK_CHARS = "".join(
    chr(code)
    for code in range(0x0300, 0x1B00)
    if unicodedata.category(chr(code)) in {"Mn", "Mc"}
)
_WORD_BODY = rf"(?:[^\W\d_]|[{re.escape(_MARK_CHARS)}])"
_LATIN_WORD_RE = re.compile(
    rf"{_WORD_BODY}+(?:['’]{_WORD_BODY}+)?", re.UNICODE
)

# locale -> espeak-ng language code. Every entry here has an
# azure-locale-feasibility-v1 receipt showing the locale's neural voice both
# accepts and honours an IPA phoneme override.
ESPEAK_LANGUAGE_BY_LOCALE = {
    "es-ES": "es",
    "es-MX": "es-419",
    "fr-FR": "fr-fr",
    "de-DE": "de",
    "it-IT": "it",
    "el-GR": "el",
    "ru-RU": "ru",
    "hi-IN": "hi",
    "nl-NL": "nl",
    "pl-PL": "pl",
    "tr-TR": "tr",
    "sv-SE": "sv",
    "uk-UA": "uk",
    "id-ID": "id",
    "cs-CZ": "cs",
    "ro-RO": "ro",
    "hu-HU": "hu",
    "nb-NO": "nb",
    "pt-PT": "pt",
    "ca-ES": "ca",
    "hr-HR": "hr",
    "sk-SK": "sk",
    "sl-SI": "sl",
    "bg-BG": "bg",
    "ms-MY": "ms",
    "mr-IN": "mr",
    "te-IN": "te",
    "gu-IN": "gu",
}


# Repairs applied to one locale's raw phone string before punctuation is
# stripped, because the repair depends on the punctuation.
#
# espeak-gu emits the Gujarati retroflex lateral ળ as "r." — an r plus a
# syllable separator — so every word carrying it was mispronounced with an r
# on BOTH sides of the pair, the neutral track included. Marathi and Telugu
# render the same letter correctly, so this is one broken mapping rather than
# a family weakness.
#
# The separator is what makes it recoverable: espeak writes the real Gujarati
# rhotic ર as ɾ and never as a bare r. Measured over 18 ળ words and 16 ર
# words, every r was followed by the separator and every rhotic was ɾ, with
# no bare r anywhere — so "r." identifies ળ unambiguously.
_PHONE_REPAIRS: dict[str, tuple[tuple[str, str], ...]] = {
    "gu-IN": (("r.", "ɭ"),),
    # espeak lowers Turkish ü (/y/) to ø in many words; real Turkish ö surfaces
    # as œ, never ø, so ø is unambiguously a mis-rendered ü. Repairing it here —
    # before lens rules match and before the azure-ipa map — fixes BOTH the
    # neutral track and the lens rules, which key on the raw adapter symbol.
    # Verified: ~50 ü/ö words, every ø came from a ü, zero from ö or elsewhere.
    "tr-TR": (("ø", "y"),),
}


@dataclass
class EspeakSourceAdapter:
    """Per-word phones for one espeak-ng language."""

    language_id: str
    espeak_language: str
    g2p: Any
    _lock: threading.RLock

    @classmethod
    def load(cls, language_id: str) -> EspeakSourceAdapter:
        try:
            espeak_language = ESPEAK_LANGUAGE_BY_LOCALE[language_id]
        except KeyError as exc:
            raise BilingualVowelEngineError(
                "unsupported_source_language",
                f"No espeak language is registered for {language_id!r}.",
            ) from exc
        from misaki.espeak import EspeakG2P

        return cls(
            language_id=language_id,
            espeak_language=espeak_language,
            g2p=EspeakG2P(language=espeak_language),
            _lock=threading.RLock(),
        )

    def _per_word_matches(self, source_words: list[str]) -> tuple[re.Match[str], ...]:
        """One phone group per word, phonemized one word at a time.

        Fails closed exactly as the phrase path does: a word the G2P cannot
        turn into a single group still raises rather than being dropped or
        guessed at.
        """

        groups: list[str] = []
        for source in source_words:
            with self._lock:
                raw, _ = self.g2p(source)
            phone = unicodedata.normalize("NFD", raw).replace(".", "")
            found = tuple(_PHONE_WORD_RE.finditer(phone))
            if len(found) != 1:
                raise BilingualVowelEngineError(
                    "word_alignment_drift",
                    f"{self.language_id} G2P did not return one phone group for "
                    f"{source!r}.",
                )
            groups.append(found[0].group())
        return tuple(_PHONE_WORD_RE.finditer(" ".join(groups)))

    def analyze(self, normalized_text: str) -> SourceAnalysis:
        with self._lock:
            raw_phone, _ = self.g2p(normalized_text)
        # espeak marks a syllable boundary with "." — not a phone, and never
        # renderable in a ph attribute. It also breaks word alignment: the
        # Gujarati retroflex lateral ળ comes back as "r.", which split શાળા
        # into two phone groups and failed the one-group-per-word check for
        # every Gujarati direction. Dropping it before alignment is safe
        # because the lane never consumes syllable structure.
        #
        # espeak echoes sentence punctuation into the phone string generally,
        # not just that separator: Spanish "¿Tienes la llave?" comes back as
        # "¿tjˈenes ..." with the inverted mark bound to the first word, which
        # then failed as an unmapped adapter symbol — so every Spanish
        # question written the way Spanish actually writes questions was
        # unrenderable. No Unicode punctuation is ever a phone (the stress and
        # length marks are modifier letters, and diacritics are combining
        # marks), so the whole category goes.
        repaired = unicodedata.normalize("NFD", raw_phone)
        for wrong, right in _PHONE_REPAIRS.get(self.language_id, ()):
            repaired = repaired.replace(wrong, right)
        source_phonemes = "".join(
            char
            for char in repaired
            if not unicodedata.category(char).startswith("P")
        )
        matches = tuple(_PHONE_WORD_RE.finditer(source_phonemes))
        source_words = _LATIN_WORD_RE.findall(normalized_text)
        if not source_words:
            raise BilingualVowelEngineError(
                "g2p_empty", f"{self.language_id} input contained no source words."
            )
        if len(matches) != len(source_words):
            # espeak binds a clitic to its host: Slovak "pri ústí" comes back
            # as one group prˈiuːstʲiː, Slovene "V kožuščku" as ukɔʒˈuːʃʧku,
            # Bulgarian "че пухът" as ʧepˈuxət. That is correct phonology and
            # wrong for a lane that substitutes per word, and because most
            # sentences in those languages contain a monosyllabic function
            # word it made three locales unrenderable on ordinary input.
            #
            # Re-phonemize word by word instead. This gives up cross-word
            # coarticulation, which the lane never consumed anyway, and it can
            # only run where the phrase call already failed closed — so no
            # direction that renders today can change.
            matches = self._per_word_matches(source_words)
            source_phonemes = " ".join(match.group() for match in matches)
        words = tuple(
            SourceWord(index, source, match.group())
            for index, (source, match) in enumerate(
                zip(source_words, matches, strict=True)
            )
        )
        if any(not word.phone for word in words):
            raise BilingualVowelEngineError(
                "unpronounceable_source_word",
                f"{self.language_id} G2P returned an empty source word.",
            )
        separators: list[str] = []
        cursor = 0
        for match in matches:
            separators.append(source_phonemes[cursor : match.start()])
            cursor = match.end()
        separators.append(source_phonemes[cursor:])
        analysis = SourceAnalysis(
            language_id=self.language_id,
            normalized_text=normalized_text,
            source_phonemes=source_phonemes,
            words=words,
            phone_separators=tuple(separators),
        )
        if analysis.compose([word.phone for word in words]) != source_phonemes:
            raise BilingualVowelEngineError(
                "source_plan_drift",
                f"{self.language_id} source phone plan is incomplete.",
            )
        return analysis
