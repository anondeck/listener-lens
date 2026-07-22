from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import re
import threading
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Protocol, Sequence

import numpy as np

from .config import ROOT, stable_json
from .kokoro_gate_bridge import KokoroGateIndex
from .kokoro_specs import VOICE_SPECS_BY_ID
from .kokoro_synthesis import (
    CONFIG_FILE,
    KOKORO_VERSION,
    MAX_PHONEME_CHARACTERS,
    MODEL_FILE,
    MODEL_REPO,
    SAMPLE_RATE_HZ,
    PairPlan,
    ParityRender,
    KokoroSynthesisRuntime,
    _INFERENCE_LOCK,
    _filtered_symbols,
    _word_column_spans,
    pcm16_bytes,
    verify_model_files,
)
from .listener_lens import DatabaseNonceChecker, NonceDecision
from .portuguese_carrier_planner_v1 import (
    NativePortugueseKokoroG2P,
    PORTUGUESE_WEAK_WORDS,
    PortuguesePositiveOnlyIndexV1,
)
from .product_voices import ProductVoiceError, load_product_voice_registry
from .util import sha256_file


BILINGUAL_ENGINE_VERSION = 2
BILINGUAL_RULES_VERSION = 1
BILINGUAL_RULES_PATH = ROOT / "rules" / "bilingual-vowel-lenses.json"
MAX_CANDIDATE_ATTEMPTS = 512
MAX_RESOLUTION_ROUNDS = 64

_WORD_PATTERNS = {
    "en-US": re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)?"),
    "pt-BR": re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)?", re.UNICODE),
}
_INPUT_PUNCTUATION = frozenset("'’.,!?;: ")
_PHONE_PUNCTUATION = frozenset(';:,.!?—…"()“” ')
_PHONE_WORD_RE = re.compile(r'[^ ;:,.!?—…"()“”]+')
_STRUCTURAL_SYMBOLS = frozenset("ˈˌːʰʲ")
_COMBINING_TILDE = "̃"

_ENGLISH_WEAK_WORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "but",
        "or",
        "nor",
        "for",
        "so",
        "yet",
        "as",
        "at",
        "by",
        "from",
        "in",
        "of",
        "on",
        "per",
        "than",
        "to",
        "with",
        "am",
        "are",
        "be",
        "been",
        "being",
        "is",
        "was",
        "were",
        "do",
        "does",
        "did",
        "have",
        "has",
        "had",
        "can",
        "could",
        "may",
        "might",
        "must",
        "shall",
        "should",
        "will",
        "would",
        "he",
        "her",
        "him",
        "his",
        "i",
        "it",
        "its",
        "me",
        "my",
        "our",
        "she",
        "their",
        "them",
        "they",
        "us",
        "we",
        "you",
        "your",
        "some",
        "any",
        "this",
        "that",
        "these",
        "those",
    }
)

# The carrier changes consonant identity while preserving a narrow phonetic class.
# This is an engineering opacity device, not a listener-perception claim.
_CONSONANT_CLASSES: tuple[tuple[str, ...], ...] = (
    ("p", "t", "k", "c", "q", "ʔ", "ʈ"),
    ("b", "d", "ɡ", "ɖ", "ɟ"),
    ("f", "s", "ʃ", "θ", "h", "x", "ç", "χ", "ɕ", "ʂ", "ɸ", "S"),
    ("v", "z", "ʒ", "ð", "β", "ɣ", "ʝ", "ʁ", "ʋ"),
    ("m", "n", "ŋ", "ɲ", "ɳ", "ɴ"),
    ("l", "ʎ"),
    ("r", "ɹ", "ɾ", "ɻ", "ɽ", "T"),
    ("w", "j", "ɥ", "ɰ"),
    ("ʧ", "ʦ", "ʨ"),
    ("ʤ", "ʣ", "ʥ"),
)
_CONSONANT_CLASS_BY_SYMBOL = {
    symbol: values for values in _CONSONANT_CLASSES for symbol in values
}
_PRESERVED_GLIDES = frozenset(("w", "j", "ɥ", "ɰ"))
_BP_LEGAL_CODA_CATEGORIES = frozenset(
    ("s", "z", "ʃ", "ʒ", "r", "ɹ", "ɾ", "l", "w", "j", "m", "n", "ŋ")
)
_EPENTHESIS_OBSTRUENTS = frozenset(
    ("p", "b", "t", "d", "k", "ɡ", "f", "v", "s", "z", "ʃ", "ʒ", "θ", "ð", "ʧ", "ʤ")
)
_COMMON_OPACITY_SYMBOLS = {
    "en-US": frozenset(
        (
            "p",
            "t",
            "k",
            "b",
            "d",
            "ɡ",
            "f",
            "s",
            "ʃ",
            "θ",
            "h",
            "v",
            "z",
            "ʒ",
            "ð",
            "m",
            "n",
            "ŋ",
            "l",
            "r",
            "ɹ",
            "w",
            "j",
            "ʧ",
            "ʤ",
        )
    ),
    "pt-BR": frozenset(
        (
            "p",
            "t",
            "k",
            "b",
            "d",
            "ɡ",
            "f",
            "s",
            "ʃ",
            "x",
            "v",
            "z",
            "ʒ",
            "m",
            "n",
            "ŋ",
            "ɲ",
            "l",
            "ʎ",
            "r",
            "ɾ",
            "w",
            "j",
            "ʧ",
            "ʤ",
        )
    ),
}
_BP_LEGAL_PADDING_CODAS = ("l", "m", "n", "s", "z", "ʃ", "ʒ")
_INSERTED_ONSETS = ("b", "d", "f", "l", "m", "n", "p", "s", "t", "v", "z")
_INSERTED_CODAS = ("b", "d", "f", "k", "l", "m", "n", "p", "s", "t", "v", "z", "ʃ", "ʒ")

_SPELLING = {
    "A": "ay",
    "I": "eye",
    "O": "oh",
    "Q": "aw",
    "W": "ow",
    "Y": "oy",
    "a": "ah",
    "e": "eh",
    "i": "ee",
    "o": "oh",
    "u": "oo",
    "y": "ee",
    "ɑ": "ah",
    "ɐ": "uh",
    "ɒ": "ah",
    "æ": "a",
    "ɔ": "aw",
    "ə": "uh",
    "ɚ": "er",
    "ɛ": "eh",
    "ɜ": "er",
    "ɨ": "ih",
    "ɪ": "ih",
    "ɯ": "oo",
    "ʊ": "uu",
    "ʌ": "uh",
    "ᵊ": "uh",
    "ᵻ": "ih",
    "ɤ": "oh",
    "ø": "eh",
    "œ": "eh",
    "b": "b",
    "c": "k",
    "d": "d",
    "f": "f",
    "h": "h",
    "j": "y",
    "k": "k",
    "l": "l",
    "m": "m",
    "n": "n",
    "p": "p",
    "q": "k",
    "r": "r",
    "s": "s",
    "t": "t",
    "v": "v",
    "w": "w",
    "x": "kh",
    "z": "z",
    "ɖ": "d",
    "ð": "dh",
    "ʤ": "j",
    "ʥ": "j",
    "ʦ": "ch",
    "ʧ": "ts",
    "ʨ": "ch",
    "ɟ": "g",
    "ɡ": "g",
    "ŋ": "ng",
    "ɲ": "ny",
    "ɳ": "n",
    "ɴ": "n",
    "ɸ": "f",
    "θ": "th",
    "ɹ": "r",
    "ɾ": "r",
    "ɻ": "r",
    "ɽ": "r",
    "ʁ": "r",
    "ʂ": "sh",
    "ʃ": "sh",
    "ʈ": "t",
    "ʋ": "v",
    "ʎ": "ly",
    "ʒ": "zh",
    "ʔ": "k",
    "ʝ": "y",
    "ɕ": "sh",
    "ɗ": "zh",
    "ç": "hy",
    "β": "v",
    "ɣ": "gh",
    "χ": "kh",
    "ɥ": "y",
    "ɰ": "w",
    "ʣ": "dz",
    "S": "sh",
    "T": "r",
}


class BilingualVowelEngineError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class NonceChecker(Protocol):
    @property
    def enabled(self) -> bool: ...

    def check(
        self, surface: str, language: str, previous_surface: str | None
    ) -> NonceDecision: ...


class PhoneIndex(Protocol):
    def phone_match(self, phone: str) -> bool: ...


@dataclass(frozen=True)
class VowelRule:
    rule_id: str
    source: str
    target: str
    evidence_tier: str
    acoustic_status: str
    source_ids: tuple[str, ...]

    @property
    def changed(self) -> bool:
        return self.source != self.target

    @property
    def directly_observed(self) -> bool:
        return self.evidence_tier.startswith("direct_")


@dataclass(frozen=True)
class ConsonantRule:
    rule_id: str
    source: str
    target: str
    contexts: tuple[str, ...]
    evidence_tier: str
    acoustic_status: str
    source_ids: tuple[str, ...]

    @property
    def changed(self) -> bool:
        return self.source != self.target

    @property
    def directly_observed(self) -> bool:
        return self.evidence_tier.startswith("direct_")


@dataclass(frozen=True)
class SourceWord:
    word_index: int
    source: str
    phone: str


@dataclass(frozen=True)
class SourceAnalysis:
    language_id: str
    normalized_text: str
    source_phonemes: str
    words: tuple[SourceWord, ...]
    phone_separators: tuple[str, ...]

    def compose(self, phones: Sequence[str]) -> str:
        if (
            len(phones) != len(self.words)
            or len(self.phone_separators) != len(self.words) + 1
        ):
            raise BilingualVowelEngineError(
                "source_alignment_drift", "Source word/phone alignment changed."
            )
        chunks = [self.phone_separators[0]]
        for index, phone in enumerate(phones):
            chunks.extend((phone, self.phone_separators[index + 1]))
        return "".join(chunks)


class SourceAdapter(Protocol):
    language_id: str

    def analyze(self, normalized_text: str) -> SourceAnalysis: ...


@dataclass(frozen=True)
class VowelOccurrence:
    source: str
    target: str
    rule_id: str
    evidence_tier: str
    acoustic_status: str
    changed: bool
    phone_offset: int
    phone_length: int


@dataclass(frozen=True)
class ConsonantOccurrence:
    source: str
    target: str
    rule_id: str
    evidence_tier: str
    acoustic_status: str
    changed: bool
    phone_offset: int
    phone_length: int


@dataclass(frozen=True)
class ProsodyOccurrence:
    source: str
    target: str
    rule_id: str
    evidence_tier: str
    acoustic_status: str
    changed: bool
    phone_offset: int
    phone_length: int
    measurement_phone_offset: int
    measurement_phone_length: int


@dataclass(frozen=True)
class InsertionOccurrence:
    neutral_placeholder: str
    target: str
    rule_id: str
    evidence_tier: str
    acoustic_status: str
    context: str
    changed: bool
    phone_offset: int
    phone_length: int

    @property
    def source(self) -> str:
        """Neutral-side model token used to reserve the insertion slot."""

        return self.neutral_placeholder


@dataclass(frozen=True)
class _PendingInsertion:
    insert_after: int
    context: str


@dataclass(frozen=True)
class MappingKey:
    source_casefold: str
    source_phone: str
    carrier_role: str
    profile_id: str


@dataclass(frozen=True)
class CarrierAssignment:
    neutral_surface: str
    lens_surface: str
    neutral_phone: str
    lens_phone: str
    vowel_occurrences: tuple[VowelOccurrence, ...]
    consonant_occurrences: tuple[ConsonantOccurrence, ...]
    prosody_occurrences: tuple[ProsodyOccurrence, ...]
    insertion_occurrences: tuple[InsertionOccurrence, ...]
    candidate_attempt: int
    inserted_consonant_count: int


@dataclass(frozen=True)
class BilingualCarrierWord:
    word_index: int
    source: str
    source_phone: str
    carrier_role: str
    neutral_surface: str
    lens_surface: str
    neutral_phone: str
    lens_phone: str
    vowel_occurrences: tuple[VowelOccurrence, ...]
    consonant_occurrences: tuple[ConsonantOccurrence, ...]
    prosody_occurrences: tuple[ProsodyOccurrence, ...]
    insertion_occurrences: tuple[InsertionOccurrence, ...]
    candidate_attempt: int
    inserted_consonant_count: int


@dataclass(frozen=True)
class CoverageReport:
    source_vowel_occurrences: int
    mapped_vowel_occurrences: int
    changed_vowel_occurrences: int
    identity_vowel_occurrences: int
    directly_observed_occurrences: int
    derived_or_structural_occurrences: int
    acoustically_validated_changed_occurrences: int
    pending_acoustic_changed_occurrences: int
    changed_word_count: int
    rules_used: tuple[str, ...]
    source_consonant_occurrences: int
    mapped_consonant_occurrences: int
    changed_consonant_occurrences: int
    identity_consonant_occurrences: int
    directly_observed_consonant_occurrences: int
    derived_or_structural_consonant_occurrences: int
    acoustically_validated_changed_consonant_occurrences: int
    pending_acoustic_changed_consonant_occurrences: int
    consonant_rules_used: tuple[str, ...]
    changed_prosody_occurrences: int
    acoustically_validated_changed_prosody_occurrences: int
    pending_acoustic_changed_prosody_occurrences: int
    prosody_rules_used: tuple[str, ...]
    changed_insertion_occurrences: int
    acoustically_validated_changed_insertion_occurrences: int
    pending_acoustic_changed_insertion_occurrences: int
    insertion_rules_used: tuple[str, ...]


@dataclass(frozen=True)
class GateReport:
    isolated_pairs_checked: int
    adjacency_pairs_checked: int
    candidate_attempts: int
    candidate_rejection_counts: dict[str, int]
    written_and_espeak_gate_pass: bool
    supplemental_phone_gates_pass: bool
    model_representable: bool
    punctuation_preserved: bool
    repeated_word_invariant_pass: bool


@dataclass(frozen=True)
class BilingualVowelPlan:
    engine_version: int
    profile_id: str
    source_language: str
    listener_language: str
    voice_id: str
    voice_registry_version: str
    voice_registry_sha256: str
    normalized_text: str
    original_source_phonemes: str
    render_reference_phonemes: str
    neutral_phonemes: str
    lens_phonemes: str
    neutral_script: str
    lens_script: str
    target_word_indexes: tuple[int, ...]
    comparison_available: bool
    words: tuple[BilingualCarrierWord, ...]
    coverage: CoverageReport
    gates: GateReport
    plan_sha256: str
    insertion_eligibilities: tuple[dict[str, Any], ...] = ()
    active_prosody_rule_ids: tuple[str, ...] = ()

    def pair_plan(self) -> PairPlan | None:
        if not self.comparison_available:
            return None
        return PairPlan(
            source_phonemes=self.render_reference_phonemes,
            neutral_phonemes=self.neutral_phonemes,
            lens_phonemes=self.lens_phonemes,
            target_word_indexes=self.target_word_indexes,
        )

    def safe_metadata(self) -> dict[str, Any]:
        return {
            "engine_version": self.engine_version,
            "profile_id": self.profile_id,
            "source_language": self.source_language,
            "listener_language": self.listener_language,
            "voice_id": self.voice_id,
            "voice_registry_version": self.voice_registry_version,
            "voice_registry_sha256": self.voice_registry_sha256,
            "comparison_available": self.comparison_available,
            "word_count": len(self.words),
            "insertion_eligibility_count": len(self.insertion_eligibilities),
            "active_prosody_rule_ids": self.active_prosody_rule_ids,
            "coverage": asdict(self.coverage),
            "gates": asdict(self.gates),
            "plan_sha256": self.plan_sha256,
        }


@dataclass(frozen=True)
class BilingualRenderVerification:
    neutral_identity_bit_exact: bool
    equal_nonempty_samples: bool
    finite: bool
    unclipped: bool
    outside_splice_exact_neutral: bool
    full_weight_interior_exact_lens: bool
    boundary_metrics_pass: bool
    localization_pass: bool
    localization_fraction: float
    integrity_pass: bool
    changed_rules_acoustically_validated: bool
    evidence_status: str
    prosody_control_pass: bool = True
    active_prosody_rule_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class BilingualVowelRender:
    plan: BilingualVowelPlan
    neutral_pcm: np.ndarray
    identity_pcm: np.ndarray
    full_lens_pcm: np.ndarray
    lens_pcm: np.ndarray
    alignment: dict[str, Any]
    lens_alignment: dict[str, Any] | None
    splice_windows: tuple[dict[str, Any], ...]
    verification: BilingualRenderVerification
    prosody: dict[str, Any] | None = None


class EnglishMisakiAdapter:
    language_id = "en-US"

    def __init__(self, g2p: Any) -> None:
        self.g2p = g2p
        self._lock = threading.RLock()

    @classmethod
    def load(cls) -> EnglishMisakiAdapter:
        from misaki.en import G2P
        from misaki.espeak import EspeakFallback

        return cls(G2P(british=False, fallback=EspeakFallback(british=False)))

    def analyze(self, normalized_text: str) -> SourceAnalysis:
        with self._lock:
            source_phonemes, tokens = self.g2p(normalized_text)
        if not tokens:
            raise BilingualVowelEngineError(
                "g2p_empty", "English G2P returned no tokens."
            )
        reconstructed = "".join(
            str(token.text) + str(token.whitespace) for token in tokens
        ).strip()
        if reconstructed != normalized_text:
            raise BilingualVowelEngineError(
                "tokenization_drift", "English G2P changed the input tokenization."
            )

        pattern = _WORD_PATTERNS[self.language_id]
        words: list[SourceWord] = []
        separators = [""]
        for token in tokens:
            text = str(token.text)
            phone = "" if token.phonemes is None else str(token.phonemes)
            whitespace = str(token.whitespace)
            if pattern.fullmatch(text):
                if not phone:
                    raise BilingualVowelEngineError(
                        "unpronounceable_source_word",
                        "English G2P returned an empty source word.",
                    )
                words.append(SourceWord(len(words), text, phone))
                separators.append(whitespace)
            else:
                if phone and not set(phone) <= _PHONE_PUNCTUATION:
                    raise BilingualVowelEngineError(
                        "unsupported_nonword_token",
                        "An English non-word token produced unsupported phones.",
                    )
                separators[-1] += phone + whitespace
        source_words = pattern.findall(normalized_text)
        if [word.source for word in words] != source_words:
            raise BilingualVowelEngineError(
                "word_alignment_drift", "English source words lost alignment."
            )
        analysis = SourceAnalysis(
            language_id=self.language_id,
            normalized_text=normalized_text,
            source_phonemes=source_phonemes,
            words=tuple(words),
            phone_separators=tuple(separators),
        )
        if analysis.compose([word.phone for word in words]) != source_phonemes:
            raise BilingualVowelEngineError(
                "source_plan_drift", "English source phone plan is incomplete."
            )
        return analysis


class PortugueseMisakiAdapter:
    language_id = "pt-BR"

    def __init__(self, g2p: Any) -> None:
        self.g2p = g2p
        self._lock = threading.RLock()

    @classmethod
    def load(cls, *, voice_id: str) -> PortugueseMisakiAdapter:
        return cls(NativePortugueseKokoroG2P(VOICE_SPECS_BY_ID[voice_id]))

    def analyze(self, normalized_text: str) -> SourceAnalysis:
        with self._lock:
            raw_phone = self.g2p.phonemize_phrase(normalized_text)
        source_phonemes = unicodedata.normalize("NFD", raw_phone)
        matches = tuple(_PHONE_WORD_RE.finditer(source_phonemes))
        source_words = _WORD_PATTERNS[self.language_id].findall(normalized_text)
        if len(matches) != len(source_words):
            raise BilingualVowelEngineError(
                "word_alignment_drift",
                "Portuguese phrase G2P did not preserve one phone group per word.",
            )
        words = tuple(
            SourceWord(index, source, match.group())
            for index, (source, match) in enumerate(
                zip(source_words, matches, strict=True)
            )
        )
        separators: list[str] = []
        cursor = 0
        for match in matches:
            separators.append(source_phonemes[cursor : match.start()])
            cursor = match.end()
        separators.append(source_phonemes[cursor:])
        return SourceAnalysis(
            language_id=self.language_id,
            normalized_text=normalized_text,
            source_phonemes=source_phonemes,
            words=words,
            phone_separators=tuple(separators),
        )


def _normalize_input(text: str, language_id: str) -> str:
    normalized = unicodedata.normalize("NFC", re.sub(r"\s+", " ", text.strip()))
    if not normalized:
        raise BilingualVowelEngineError(
            "empty_input", "Enter a word or short sentence."
        )
    if len(normalized) > 280:
        raise BilingualVowelEngineError("input_too_long", "Input is too long.")
    if language_id not in _WORD_PATTERNS:
        raise BilingualVowelEngineError(
            "unsupported_source_language", f"Unsupported source language: {language_id}"
        )
    if language_id == "en-US":
        allowed = all(
            character.isascii()
            and (character.isalpha() or character in _INPUT_PUNCTUATION)
            for character in normalized
        )
    else:
        allowed = all(
            character.isalpha() or character in _INPUT_PUNCTUATION
            for character in normalized
        )
    if not allowed:
        raise BilingualVowelEngineError(
            "unsupported_characters", "Use words and basic punctuation only."
        )
    words = _WORD_PATTERNS[language_id].findall(normalized)
    if not 1 <= len(words) <= 40:
        raise BilingualVowelEngineError(
            "unsupported_word_count", "Enter between 1 and 40 words."
        )
    if (len(re.findall(r"[.!?]+", normalized)) or 1) > 2:
        raise BilingualVowelEngineError(
            "unsupported_sentence_count", "Enter at most two sentences."
        )
    if any(len(word) > 1 and word.isupper() for word in words):
        raise BilingualVowelEngineError(
            "unsupported_acronym", "Acronyms and all-caps words are unsupported."
        )
    return normalized


def _pick(values: Sequence[str], digest: bytes, position: int, attempt: int) -> str:
    return values[(digest[position % len(digest)] + position + attempt) % len(values)]


def _surface_for(phone: str) -> str:
    chunks: list[str] = []
    for symbol in unicodedata.normalize("NFD", phone):
        if symbol in _STRUCTURAL_SYMBOLS or unicodedata.combining(symbol):
            continue
        spelling = _SPELLING.get(symbol)
        if spelling is None:
            raise BilingualVowelEngineError(
                "unspellable_phone_plan", f"No carrier spelling exists for {symbol!r}."
            )
        chunks.append(spelling)
    surface = "".join(chunks)
    if not surface or not surface.isascii() or not surface.isalpha():
        raise BilingualVowelEngineError(
            "invalid_carrier_surface", "Carrier spelling is not one ASCII word."
        )
    return surface


def _insert_onset(phone: str, onset: str) -> str:
    first_nonstructural = next(
        (
            index
            for index, symbol in enumerate(phone)
            if symbol not in _STRUCTURAL_SYMBOLS
        ),
        None,
    )
    if first_nonstructural is None:
        raise BilingualVowelEngineError(
            "source_word_without_phone", "Source word has no non-structural phone."
        )
    stress_indexes = [
        index
        for index, symbol in enumerate(phone[: first_nonstructural + 1])
        if symbol in {"ˈ", "ˌ"}
    ]
    insert_at = stress_indexes[0] if stress_indexes else first_nonstructural
    return phone[:insert_at] + onset + phone[insert_at:]


class BilingualVowelPlanner:
    def __init__(
        self,
        *,
        profile: dict[str, Any],
        adapter: SourceAdapter,
        model_vocab: set[str] | frozenset[str],
        nonce_checker: NonceChecker,
        phone_indexes: Sequence[PhoneIndex] = (),
        rules_path: Path = BILINGUAL_RULES_PATH,
    ) -> None:
        if adapter.language_id != profile.get("source_language"):
            raise BilingualVowelEngineError(
                "adapter_language_mismatch", "Source adapter and profile disagree."
            )
        if not nonce_checker.enabled:
            raise BilingualVowelEngineError(
                "nonce_gate_disabled", "The written-word and eSpeak gate is required."
            )
        registry = load_product_voice_registry()
        try:
            selected_voice = registry.resolve(
                profile["source_language"], profile.get("voice_id")
            )
        except ProductVoiceError as exc:
            raise BilingualVowelEngineError(
                "unsupported_product_voice", str(exc)
            ) from exc
        self.profile = {
            **profile,
            "voice_id": selected_voice.voice_id,
            "voice_registry_version": registry.registry_version,
            "voice_registry_sha256": registry.registry_sha256,
        }
        self.adapter = adapter
        self.model_vocab = frozenset(model_vocab)
        self.nonce_checker = nonce_checker
        self.phone_indexes = tuple(phone_indexes)
        self.rules_path = rules_path
        self.engine_version = int(
            profile.get("engine_version", BILINGUAL_ENGINE_VERSION)
        )
        self.rules = self._load_rules(profile)
        self.rule_sources = tuple(sorted(self.rules, key=len, reverse=True))
        self.consonant_rules = self._load_consonant_rules(profile)
        self.consonant_rule_sources = tuple(
            sorted(self.consonant_rules, key=len, reverse=True)
        )

    @staticmethod
    def _load_rules(profile: dict[str, Any]) -> dict[str, VowelRule]:
        rules: dict[str, VowelRule] = {}
        for raw in profile.get("vowel_rules", []):
            source = unicodedata.normalize("NFD", raw["source"])
            target = unicodedata.normalize("NFD", raw["target"])
            if source in rules:
                raise BilingualVowelEngineError(
                    "duplicate_vowel_rule",
                    f"Duplicate normalized vowel rule: {source!r}",
                )
            if len(source) != len(target):
                raise BilingualVowelEngineError(
                    "nonparallel_vowel_rule",
                    f"Controlled vowel rule changes model-token count: {raw['id']}",
                )
            rules[source] = VowelRule(
                rule_id=raw["id"],
                source=source,
                target=target,
                evidence_tier=raw["evidence_tier"],
                acoustic_status=raw["acoustic_status"],
                source_ids=tuple(raw["source_ids"]),
            )
        if not rules:
            raise BilingualVowelEngineError(
                "empty_profile", "Profile has no vowel rules."
            )
        return rules

    @staticmethod
    def _load_consonant_rules(
        profile: dict[str, Any],
    ) -> dict[str, tuple[ConsonantRule, ...]]:
        rules: dict[str, list[ConsonantRule]] = {}
        for raw in profile.get("consonant_rules", []):
            source = unicodedata.normalize("NFD", raw["source"])
            target = unicodedata.normalize("NFD", raw["target"])
            if len(source) != len(target):
                raise BilingualVowelEngineError(
                    "nonparallel_consonant_rule",
                    "Shared-state consonant substitution changes model-token count: "
                    + raw["id"],
                )
            contexts = tuple(raw.get("contexts", ("any",)))
            unsupported_contexts = set(contexts) - {
                "any",
                "word_initial",
                "word_final",
                "intervocalic",
            }
            if unsupported_contexts:
                raise BilingualVowelEngineError(
                    "unsupported_consonant_context",
                    f"Unsupported consonant contexts for {raw['id']}: "
                    + ", ".join(sorted(unsupported_contexts)),
                )
            rule = ConsonantRule(
                rule_id=raw["id"],
                source=source,
                target=target,
                contexts=contexts,
                evidence_tier=raw["evidence_tier"],
                acoustic_status=raw["acoustic_status"],
                source_ids=tuple(raw["source_ids"]),
            )
            if any(existing.contexts == contexts for existing in rules.get(source, ())):
                raise BilingualVowelEngineError(
                    "duplicate_consonant_rule",
                    f"Duplicate normalized consonant rule/context: {source!r} {contexts!r}",
                )
            rules.setdefault(source, []).append(rule)
        return {source: tuple(values) for source, values in rules.items()}

    @classmethod
    def load(
        cls, profile_id: str, *, voice_id: str | None = None
    ) -> BilingualVowelPlanner:
        if importlib.metadata.version("kokoro") != "0.9.4":
            raise BilingualVowelEngineError(
                "renderer_version_mismatch", "Kokoro 0.9.4 is required."
            )
        data = json.loads(BILINGUAL_RULES_PATH.read_text(encoding="utf-8"))
        profiles = {profile["id"]: profile for profile in data["profiles"]}
        try:
            profile = profiles[profile_id]
        except KeyError as exc:
            raise BilingualVowelEngineError(
                "unknown_profile", f"Unknown bilingual vowel profile: {profile_id}"
            ) from exc
        registry = load_product_voice_registry()
        try:
            selected_voice = registry.resolve(profile["source_language"], voice_id)
        except ProductVoiceError as exc:
            raise BilingualVowelEngineError(
                "unsupported_product_voice", str(exc)
            ) from exc
        if registry.profiles.get(profile_id) != profile["source_language"]:
            raise BilingualVowelEngineError(
                "voice_profile_mismatch",
                "Product voice registry and bilingual profile disagree.",
            )
        profile = {
            **profile,
            "voice_id": selected_voice.voice_id,
            "engine_version": BILINGUAL_ENGINE_VERSION,
        }
        files = verify_model_files(download=False)
        config = json.loads(files[CONFIG_FILE].read_text(encoding="utf-8"))
        if profile["source_language"] == "en-US":
            adapter: SourceAdapter = EnglishMisakiAdapter.load()
        elif profile["source_language"] == "pt-BR":
            adapter = PortugueseMisakiAdapter.load(voice_id=selected_voice.voice_id)
        else:  # pragma: no cover - profile schema protects this branch
            raise BilingualVowelEngineError(
                "unsupported_source_language", profile["source_language"]
            )
        return cls(
            profile=profile,
            adapter=adapter,
            model_vocab=set(config["vocab"]),
            nonce_checker=DatabaseNonceChecker(),
            phone_indexes=(KokoroGateIndex(), PortuguesePositiveOnlyIndexV1()),
        )

    def _rule_at(self, phone: str, position: int) -> VowelRule | None:
        for source in self.rule_sources:
            if phone.startswith(source, position):
                return self.rules[source]
        return None

    @staticmethod
    def _phonemic_neighbor(phone: str, position: int, step: int) -> str | None:
        index = position
        while 0 <= index < len(phone):
            symbol = phone[index]
            if symbol not in _STRUCTURAL_SYMBOLS and symbol != _COMBINING_TILDE:
                return symbol
            index += step
        return None

    def _consonant_context_matches(
        self, rule: ConsonantRule, phone: str, position: int
    ) -> bool:
        if "any" in rule.contexts:
            return True
        before = self._phonemic_neighbor(phone, position - 1, -1)
        after = self._phonemic_neighbor(phone, position + len(rule.source), 1)
        vowel_symbols = {
            symbol
            for source in self.rules
            for symbol in source
            if symbol != _COMBINING_TILDE
        }
        return any(
            (
                context == "word_initial"
                and before is None
                or context == "word_final"
                and after is None
                or context == "intervocalic"
                and before in vowel_symbols
                and after in vowel_symbols
            )
            for context in rule.contexts
        )

    def _consonant_rule_at(self, phone: str, position: int) -> ConsonantRule | None:
        for source in self.consonant_rule_sources:
            if not phone.startswith(source, position):
                continue
            for rule in self.consonant_rules[source]:
                if self._consonant_context_matches(rule, phone, position):
                    return rule
        return None

    def _apply_word_prosody(
        self,
        source_phone: str,
        neutral_phone: str,
        lens_phone: str,
        source_to_carrier_offset: dict[int, int],
    ) -> tuple[str, tuple[ProsodyOccurrence, ...]]:
        raw_rule = next(
            (
                rule
                for rule in self.profile.get("prosody_rules", ())
                if rule.get("operation") == "swap_primary_and_initial_secondary_stress"
            ),
            None,
        )
        if raw_rule is None:
            return lens_phone, ()
        secondary = source_phone.find("ˌ")
        primary = source_phone.find("ˈ")
        if secondary < 0 or primary <= secondary:
            return lens_phone, ()
        preceding = source_phone[:secondary]
        if any(
            self._rule_at(preceding, index) is not None
            for index in range(len(preceding))
        ):
            return lens_phone, ()
        if len(neutral_phone) != len(lens_phone):
            raise BilingualVowelEngineError(
                "prosody_alignment_drift",
                "Stress recategorization requires equal neutral/lens token spans.",
            )
        try:
            secondary_offset = source_to_carrier_offset[secondary]
            primary_offset = source_to_carrier_offset[primary]
        except KeyError as exc:
            raise BilingualVowelEngineError(
                "prosody_alignment_drift",
                "A source stress marker lost its carrier-token alignment.",
            ) from exc
        if (
            neutral_phone[secondary_offset] != "ˌ"
            or lens_phone[secondary_offset] != "ˌ"
            or neutral_phone[primary_offset] != "ˈ"
            or lens_phone[primary_offset] != "ˈ"
        ):
            raise BilingualVowelEngineError(
                "prosody_alignment_drift",
                "A source stress marker changed before the controlled prosody step.",
            )

        def measurement_span(
            marker_position: int, marker_offset: int
        ) -> tuple[int, int]:
            position = marker_position + 1
            while position < len(source_phone):
                rule = self._rule_at(source_phone, position)
                if rule is not None:
                    try:
                        vowel_offset = source_to_carrier_offset[position]
                    except KeyError as exc:
                        raise BilingualVowelEngineError(
                            "prosody_alignment_drift",
                            "A stressed vowel lost its carrier-token alignment.",
                        ) from exc
                    end = vowel_offset + len(rule.source)
                    return marker_offset, end - marker_offset
                position += 1
            raise BilingualVowelEngineError(
                "prosody_alignment_drift", "A stress marker has no following vowel."
            )

        secondary_measurement = measurement_span(secondary, secondary_offset)
        primary_measurement = measurement_span(primary, primary_offset)
        values = list(lens_phone)
        values[secondary_offset] = "ˈ"
        values[primary_offset] = "ˌ"
        status = "pending_controlled_stress_acoustic_and_listener_qc"
        occurrences = (
            ProsodyOccurrence(
                source="ˌ",
                target="ˈ",
                rule_id=raw_rule["id"],
                evidence_tier=raw_rule["evidence_tier"],
                acoustic_status=status,
                changed=True,
                phone_offset=secondary_offset,
                phone_length=1,
                measurement_phone_offset=secondary_measurement[0],
                measurement_phone_length=secondary_measurement[1],
            ),
            ProsodyOccurrence(
                source="ˈ",
                target="ˌ",
                rule_id=raw_rule["id"],
                evidence_tier=raw_rule["evidence_tier"],
                acoustic_status=status,
                changed=True,
                phone_offset=primary_offset,
                phone_length=1,
                measurement_phone_offset=primary_measurement[0],
                measurement_phone_length=primary_measurement[1],
            ),
        )
        return "".join(values), occurrences

    def _epenthesis_context(
        self, source_phone: str, position_after_source: int, anchor: str
    ) -> str | None:
        rule = next(
            (
                value
                for value in self.profile.get("insertion_rules", ())
                if value.get("operation") == "insert_after"
            ),
            None,
        )
        if rule is None or anchor not in _EPENTHESIS_OBSTRUENTS:
            return None
        after = self._phonemic_neighbor(source_phone, position_after_source, 1)
        configured = set(rule.get("contexts", ()))
        if (
            after is None
            and anchor not in _BP_LEGAL_CODA_CATEGORIES
            and "word_final_obstruent" in configured
        ):
            return "word_final_obstruent"
        if (
            after in _CONSONANT_CLASS_BY_SYMBOL
            and anchor not in _BP_LEGAL_CODA_CATEGORIES
            and "illegal_consonant_cluster" in configured
        ):
            return "illegal_consonant_cluster"
        return None

    def _register_epenthesis_slot(
        self,
        *,
        source_phone: str,
        position_after_source: int,
        anchor: str,
        insert_after: int,
        pending: list[_PendingInsertion],
    ) -> None:
        context = self._epenthesis_context(source_phone, position_after_source, anchor)
        if context is None:
            return
        pending.append(_PendingInsertion(insert_after=insert_after, context=context))

    def _apply_pending_insertions(
        self,
        *,
        neutral_phone: str,
        lens_phone: str,
        pending: list[_PendingInsertion],
        vowel_occurrences: list[VowelOccurrence],
        consonant_occurrences: list[ConsonantOccurrence],
        prosody_occurrences: list[ProsodyOccurrence],
    ) -> tuple[
        str,
        str,
        list[VowelOccurrence],
        list[ConsonantOccurrence],
        list[ProsodyOccurrence],
        list[InsertionOccurrence],
    ]:
        if not pending:
            return (
                neutral_phone,
                lens_phone,
                vowel_occurrences,
                consonant_occurrences,
                prosody_occurrences,
                [],
            )
        if len(neutral_phone) != len(lens_phone):
            raise BilingualVowelEngineError(
                "insertion_alignment_drift",
                "Latent insertion slots require equal pre-insertion phone plans.",
            )
        slots = sorted(set(pending), key=lambda value: value.insert_after)
        if any(
            slot.insert_after <= 0 or slot.insert_after > len(neutral_phone)
            for slot in slots
        ):
            raise BilingualVowelEngineError(
                "insertion_alignment_drift", "A latent insertion boundary is invalid."
            )
        raw_rule = next(
            value
            for value in self.profile.get("insertion_rules", ())
            if value.get("operation") == "insert_after"
        )
        target = unicodedata.normalize("NFD", raw_rule["target"])
        if len(target) != 1:
            raise BilingualVowelEngineError(
                "nonparallel_insertion_rule",
                "The controlled latent-slot implementation requires one target token.",
            )
        slot_by_boundary = {slot.insert_after: slot for slot in slots}
        neutral_values: list[str] = []
        lens_values: list[str] = []
        insertion_occurrences: list[InsertionOccurrence] = []
        inserted = 0
        for index, (neutral_symbol, lens_symbol) in enumerate(
            zip(neutral_phone, lens_phone, strict=True)
        ):
            neutral_values.append(neutral_symbol)
            lens_values.append(lens_symbol)
            boundary = index + 1
            slot = slot_by_boundary.get(boundary)
            if slot is None:
                continue
            offset = boundary + inserted
            neutral_placeholder = neutral_phone[boundary - 1]
            if neutral_placeholder not in _CONSONANT_CLASS_BY_SYMBOL:
                raise BilingualVowelEngineError(
                    "insertion_alignment_drift",
                    "A latent insertion slot is not anchored to a consonant.",
                )
            neutral_values.append(neutral_placeholder)
            lens_values.append(target)
            insertion_occurrences.append(
                InsertionOccurrence(
                    neutral_placeholder=neutral_placeholder,
                    target=target,
                    rule_id=raw_rule["id"],
                    evidence_tier=raw_rule["evidence_tier"],
                    acoustic_status=(
                        "pending_controlled_epenthesis_acoustic_and_listener_qc"
                    ),
                    context=slot.context,
                    changed=True,
                    phone_offset=offset,
                    phone_length=1,
                )
            )
            inserted += 1

        def shifted(values: list[Any]) -> list[Any]:
            results: list[Any] = []
            for occurrence in values:
                updates = {
                    "phone_offset": occurrence.phone_offset
                    + sum(
                        slot.insert_after <= occurrence.phone_offset for slot in slots
                    )
                }
                if isinstance(occurrence, ProsodyOccurrence):
                    updates["measurement_phone_offset"] = (
                        occurrence.measurement_phone_offset
                        + sum(
                            slot.insert_after <= occurrence.measurement_phone_offset
                            for slot in slots
                        )
                    )
                    measurement_end = (
                        occurrence.measurement_phone_offset
                        + occurrence.measurement_phone_length
                    )
                    updates["measurement_phone_length"] = (
                        occurrence.measurement_phone_length
                        + sum(
                            occurrence.measurement_phone_offset
                            < slot.insert_after
                            <= measurement_end
                            for slot in slots
                        )
                    )
                results.append(replace(occurrence, **updates))
            return results

        return (
            "".join(neutral_values),
            "".join(lens_values),
            shifted(vowel_occurrences),
            shifted(consonant_occurrences),
            shifted(prosody_occurrences),
            insertion_occurrences,
        )

    def _opacity_candidates(self, symbol: str) -> tuple[str, ...]:
        base = _CONSONANT_CLASS_BY_SYMBOL[symbol]
        if self.engine_version < 2:
            return tuple(value for value in base if value != symbol)
        allowed = _COMMON_OPACITY_SYMBOLS[self.profile["source_language"]]
        values = tuple(value for value in base if value in allowed and value != symbol)
        if self.profile.get("insertion_rules"):
            source_is_legal_coda = symbol in _BP_LEGAL_CODA_CATEGORIES
            same_coda_class = tuple(
                value
                for value in values
                if (value in _BP_LEGAL_CODA_CATEGORIES) == source_is_legal_coda
            )
            if same_coda_class:
                values = same_coda_class
        if values:
            return values
        return (symbol,)

    def _candidate(self, key: MappingKey, attempt: int) -> CarrierAssignment:
        digest = hashlib.sha256(
            stable_json(
                {
                    "engine_version": self.engine_version,
                    "key": asdict(key),
                    "attempt": attempt,
                }
            ).encode("utf-8")
        ).digest()
        source_phone = unicodedata.normalize("NFD", key.source_phone)
        neutral: list[str] = []
        lens: list[str] = []
        occurrences: list[VowelOccurrence] = []
        consonant_occurrences: list[ConsonantOccurrence] = []
        prosody_occurrences: list[ProsodyOccurrence] = []
        pending_insertions: list[_PendingInsertion] = []
        source_to_carrier_offset: dict[int, int] = {}
        consonant_count = 0
        position = 0
        while position < len(source_phone):
            symbol = source_phone[position]
            source_to_carrier_offset[position] = sum(len(value) for value in neutral)
            rule = self._rule_at(source_phone, position)
            if rule is not None:
                phone_offset = sum(len(value) for value in neutral)
                neutral.append(rule.source)
                lens.append(rule.target)
                occurrences.append(
                    VowelOccurrence(
                        source=rule.source,
                        target=rule.target,
                        rule_id=rule.rule_id,
                        evidence_tier=rule.evidence_tier,
                        acoustic_status=rule.acoustic_status,
                        changed=rule.changed,
                        phone_offset=phone_offset,
                        phone_length=len(rule.source),
                    )
                )
                position += len(rule.source)
                continue
            consonant_rule = self._consonant_rule_at(source_phone, position)
            if consonant_rule is not None:
                phone_offset = sum(len(value) for value in neutral)
                neutral.append(consonant_rule.source)
                lens.append(consonant_rule.target)
                consonant_occurrences.append(
                    ConsonantOccurrence(
                        source=consonant_rule.source,
                        target=consonant_rule.target,
                        rule_id=consonant_rule.rule_id,
                        evidence_tier=consonant_rule.evidence_tier,
                        acoustic_status=consonant_rule.acoustic_status,
                        changed=consonant_rule.changed,
                        phone_offset=phone_offset,
                        phone_length=len(consonant_rule.source),
                    )
                )
                consonant_count += len(consonant_rule.source)
                self._register_epenthesis_slot(
                    source_phone=source_phone,
                    position_after_source=position + len(consonant_rule.source),
                    anchor=consonant_rule.source[-1],
                    insert_after=sum(len(value) for value in neutral),
                    pending=pending_insertions,
                )
                position += len(consonant_rule.source)
                continue
            if symbol in _STRUCTURAL_SYMBOLS:
                neutral.append(symbol)
                lens.append(symbol)
            elif symbol == _COMBINING_TILDE:
                raise BilingualVowelEngineError(
                    "unmapped_nasal_vowel",
                    "A combining nasal marker escaped the longest-match vowel rules.",
                )
            elif symbol in _CONSONANT_CLASS_BY_SYMBOL:
                if symbol in _PRESERVED_GLIDES:
                    # Kokoro's pt-BR G2P commonly spells a diphthong offglide
                    # as /w/ or /j/ rather than a compact vowel symbol. Keep it
                    # so consonant opacity does not accidentally rewrite the
                    # very vowel trajectory the listener profile is mapping.
                    selected = symbol
                else:
                    values = self._opacity_candidates(symbol)
                    selected = _pick(values, digest, position, attempt)
                neutral.append(selected)
                lens.append(selected)
                consonant_occurrences.append(
                    ConsonantOccurrence(
                        source=symbol,
                        target=symbol,
                        rule_id="identity.consonant_class",
                        evidence_tier="inventory_or_structural_correspondence",
                        acoustic_status="not_required_identity",
                        changed=False,
                        phone_offset=sum(len(value) for value in neutral)
                        - len(selected),
                        phone_length=len(selected),
                    )
                )
                consonant_count += 1
                self._register_epenthesis_slot(
                    source_phone=source_phone,
                    position_after_source=position + 1,
                    anchor=symbol,
                    insert_after=sum(len(value) for value in neutral),
                    pending=pending_insertions,
                )
            else:
                raise BilingualVowelEngineError(
                    "unsupported_source_phone",
                    f"Source phone symbol {symbol!r} has no vowel rule or carrier class.",
                )
            position += 1
        if not occurrences:
            raise BilingualVowelEngineError(
                "source_word_without_vowel", "Source word contains no mapped vowel."
            )

        neutral_phone = "".join(neutral)
        lens_phone = "".join(lens)
        lens_phone, applied_prosody = self._apply_word_prosody(
            source_phone,
            neutral_phone,
            lens_phone,
            source_to_carrier_offset,
        )
        prosody_occurrences.extend(applied_prosody)
        inserted_consonant_count = 0
        if consonant_count == 0:
            onset = _pick(_INSERTED_ONSETS, digest, 0, attempt)
            neutral_phone = _insert_onset(neutral_phone, onset)
            lens_phone = _insert_onset(lens_phone, onset)
            consonant_count += 1
            inserted_consonant_count += 1
            occurrences = [
                replace(occurrence, phone_offset=occurrence.phone_offset + 1)
                for occurrence in occurrences
            ]
            consonant_occurrences = [
                replace(occurrence, phone_offset=occurrence.phone_offset + 1)
                for occurrence in consonant_occurrences
            ]
            prosody_occurrences = [
                replace(
                    occurrence,
                    phone_offset=occurrence.phone_offset + 1,
                    measurement_phone_offset=(occurrence.measurement_phone_offset + 1),
                )
                for occurrence in prosody_occurrences
            ]
            pending_insertions = [
                replace(slot, insert_after=slot.insert_after + 1)
                for slot in pending_insertions
            ]
            consonant_occurrences.insert(
                0,
                ConsonantOccurrence(
                    source=onset,
                    target=onset,
                    rule_id="engineering.inserted_onset",
                    evidence_tier="engineering_semantic_opacity",
                    acoustic_status="not_required_identity",
                    changed=False,
                    phone_offset=0,
                    phone_length=1,
                ),
            )

        # A one-vowel C, VC, or CVC carrier has too little combinatorial room to
        # clear the written-word and predicted-homophone gates consistently.
        # Add identical codas without adding a syllable; all source vowels and
        # stress marks remain untouched. Multisyllabic words only receive the
        # minimum padding needed to keep a gate-clean search possible.
        minimum_consonants = (3 if len(occurrences) == 1 else 2) + min(3, attempt // 32)
        padding_codas = (
            _BP_LEGAL_PADDING_CODAS
            if self.profile.get("insertion_rules")
            else _INSERTED_CODAS
        )
        while consonant_count < minimum_consonants:
            coda = _pick(
                padding_codas,
                digest,
                len(source_phone) + inserted_consonant_count,
                attempt,
            )
            neutral_phone += coda
            lens_phone += coda
            consonant_occurrences.append(
                ConsonantOccurrence(
                    source=coda,
                    target=coda,
                    rule_id="engineering.inserted_coda",
                    evidence_tier="engineering_semantic_opacity",
                    acoustic_status="not_required_identity",
                    changed=False,
                    phone_offset=len(neutral_phone) - len(coda),
                    phone_length=len(coda),
                )
            )
            consonant_count += 1
            inserted_consonant_count += 1
        (
            neutral_phone,
            lens_phone,
            occurrences,
            consonant_occurrences,
            prosody_occurrences,
            insertion_occurrences,
        ) = self._apply_pending_insertions(
            neutral_phone=neutral_phone,
            lens_phone=lens_phone,
            pending=pending_insertions,
            vowel_occurrences=occurrences,
            consonant_occurrences=consonant_occurrences,
            prosody_occurrences=prosody_occurrences,
        )
        return CarrierAssignment(
            neutral_surface=_surface_for(neutral_phone),
            lens_surface=_surface_for(lens_phone),
            neutral_phone=neutral_phone,
            lens_phone=lens_phone,
            vowel_occurrences=tuple(occurrences),
            consonant_occurrences=tuple(consonant_occurrences),
            prosody_occurrences=tuple(prosody_occurrences),
            insertion_occurrences=tuple(insertion_occurrences),
            candidate_attempt=attempt,
            inserted_consonant_count=inserted_consonant_count,
        )

    def _validate_model_phone(self, phone: str) -> None:
        unsupported = sorted(set(phone) - self.model_vocab)
        if unsupported:
            raise BilingualVowelEngineError(
                "unrepresentable_phone_plan",
                "Phone plan contains unsupported Kokoro symbols: "
                + "".join(unsupported),
            )

    def _isolated_reasons(self, assignment: CarrierAssignment) -> set[str]:
        reasons: set[str] = set()
        language = self.profile["source_gate_language"]
        seen: set[tuple[str, str]] = set()
        for side, surface, phone in (
            ("neutral", assignment.neutral_surface, assignment.neutral_phone),
            ("lens", assignment.lens_surface, assignment.lens_phone),
        ):
            signature = (surface, phone)
            if signature in seen:
                continue
            seen.add(signature)
            decision = self.nonce_checker.check(surface, language, None)
            if not decision.accepted:
                reasons.add(
                    f"{side}_written_espeak_{decision.rejection_reason or 'rejected'}"
                )
            for index, phone_index in enumerate(self.phone_indexes):
                if phone_index.phone_match(phone):
                    reasons.add(f"{side}_phone_index_{index}_predicted_homophone")
            try:
                self._validate_model_phone(phone)
            except BilingualVowelEngineError:
                reasons.add(f"{side}_unrepresentable")
        return reasons

    def _resolve_assignments(
        self, keys: Sequence[MappingKey], rejection_counts: Counter[str]
    ) -> tuple[dict[MappingKey, CarrierAssignment], int, int]:
        unique_keys = tuple(dict.fromkeys(keys))
        assignments: dict[MappingKey, CarrierAssignment] = {}
        next_attempt = {key: 0 for key in unique_keys}
        implicated = set(unique_keys)
        total_attempts = 0
        adjacency_checks = 0
        language = self.profile["source_gate_language"]

        for _ in range(MAX_RESOLUTION_ROUNDS):
            for key in unique_keys:
                if key not in implicated:
                    continue
                accepted: CarrierAssignment | None = None
                for attempt in range(
                    next_attempt[key], next_attempt[key] + MAX_CANDIDATE_ATTEMPTS
                ):
                    total_attempts += 1
                    candidate = self._candidate(key, attempt)
                    reasons = self._isolated_reasons(candidate)
                    if not reasons:
                        accepted = candidate
                        break
                    rejection_counts.update(reasons)
                if accepted is None:
                    raise BilingualVowelEngineError(
                        "candidate_search_exhausted",
                        "No gate-clean broad-vowel carrier exists in the bounded search.",
                    )
                assignments[key] = accepted

            conflicts: set[MappingKey] = set()
            for previous_key, current_key in zip(keys, keys[1:]):
                adjacency_checks += 1
                previous = assignments[previous_key]
                current = assignments[current_key]
                for (
                    side,
                    previous_surface,
                    current_surface,
                    previous_phone,
                    current_phone,
                ) in (
                    (
                        "neutral",
                        previous.neutral_surface,
                        current.neutral_surface,
                        previous.neutral_phone,
                        current.neutral_phone,
                    ),
                    (
                        "lens",
                        previous.lens_surface,
                        current.lens_surface,
                        previous.lens_phone,
                        current.lens_phone,
                    ),
                ):
                    decision = self.nonce_checker.check(
                        current_surface, language, previous_surface
                    )
                    if not decision.accepted:
                        rejection_counts[
                            f"{side}_written_espeak_{decision.rejection_reason or 'adjacency_rejected'}"
                        ] += 1
                        conflicts.update((previous_key, current_key))
                    for index, phone_index in enumerate(self.phone_indexes):
                        if phone_index.phone_match(previous_phone + current_phone):
                            rejection_counts[
                                f"{side}_phone_index_{index}_adjacency_predicted_homophone"
                            ] += 1
                            conflicts.update((previous_key, current_key))
            if not conflicts:
                return assignments, total_attempts, adjacency_checks
            implicated = conflicts
            for key in conflicts:
                next_attempt[key] = assignments[key].candidate_attempt + 1
        raise BilingualVowelEngineError(
            "adjacency_search_exhausted",
            "No globally gate-clean broad-vowel carrier exists in the bounded search.",
        )

    def plan(self, text: str) -> BilingualVowelPlan:
        normalized = _normalize_input(text, self.profile["source_language"])
        analysis = self.adapter.analyze(normalized)
        weak_words = (
            _ENGLISH_WEAK_WORDS
            if analysis.language_id == "en-US"
            else PORTUGUESE_WEAK_WORDS
        )
        keys = [
            MappingKey(
                source_casefold=word.source.casefold(),
                source_phone=unicodedata.normalize("NFD", word.phone),
                carrier_role=(
                    "weak" if word.source.casefold() in weak_words else "content"
                ),
                profile_id=self.profile["id"],
            )
            for word in analysis.words
        ]
        rejection_counts: Counter[str] = Counter()
        assignments, candidate_attempts, adjacency_checks = self._resolve_assignments(
            keys, rejection_counts
        )

        words = tuple(
            BilingualCarrierWord(
                word_index=source.word_index,
                source=source.source,
                source_phone=key.source_phone,
                carrier_role=key.carrier_role,
                neutral_surface=assignments[key].neutral_surface,
                lens_surface=assignments[key].lens_surface,
                neutral_phone=assignments[key].neutral_phone,
                lens_phone=assignments[key].lens_phone,
                vowel_occurrences=assignments[key].vowel_occurrences,
                consonant_occurrences=assignments[key].consonant_occurrences,
                prosody_occurrences=assignments[key].prosody_occurrences,
                insertion_occurrences=assignments[key].insertion_occurrences,
                candidate_attempt=assignments[key].candidate_attempt,
                inserted_consonant_count=assignments[key].inserted_consonant_count,
            )
            for source, key in zip(analysis.words, keys, strict=True)
        )
        neutral_phonemes = analysis.compose([word.neutral_phone for word in words])
        lens_phonemes = analysis.compose([word.lens_phone for word in words])
        self._validate_model_phone(neutral_phonemes)
        self._validate_model_phone(lens_phonemes)
        neutral_model_count = sum(
            symbol in self.model_vocab for symbol in neutral_phonemes
        )
        lens_model_count = sum(symbol in self.model_vocab for symbol in lens_phonemes)
        if neutral_model_count != lens_model_count:
            raise BilingualVowelEngineError(
                "plan_token_count_mismatch",
                "Neutral and lens plans have unequal model-token counts.",
            )
        if neutral_model_count > MAX_PHONEME_CHARACTERS:
            raise BilingualVowelEngineError(
                "phone_plan_too_long", "The controlled carrier exceeds Kokoro's limit."
            )

        occurrences = tuple(
            occurrence for word in words for occurrence in word.vowel_occurrences
        )
        consonant_occurrences = tuple(
            occurrence for word in words for occurrence in word.consonant_occurrences
        )
        prosody_occurrences = tuple(
            occurrence for word in words for occurrence in word.prosody_occurrences
        )
        insertion_occurrences = tuple(
            occurrence for word in words for occurrence in word.insertion_occurrences
        )
        target_indexes = tuple(
            word.word_index
            for word in words
            if any(occurrence.changed for occurrence in word.vowel_occurrences)
            or any(occurrence.changed for occurrence in word.consonant_occurrences)
            or any(occurrence.changed for occurrence in word.prosody_occurrences)
            or any(occurrence.changed for occurrence in word.insertion_occurrences)
        )
        changed_occurrences = tuple(
            occurrence for occurrence in occurrences if occurrence.changed
        )
        acoustic_validated = tuple(
            occurrence
            for occurrence in changed_occurrences
            if "exact_category_pass" in occurrence.acoustic_status
        )
        changed_consonant_occurrences = tuple(
            occurrence for occurrence in consonant_occurrences if occurrence.changed
        )
        acoustic_validated_consonants = tuple(
            occurrence
            for occurrence in changed_consonant_occurrences
            if "exact_category_pass" in occurrence.acoustic_status
        )
        acoustic_validated_prosody = tuple(
            occurrence
            for occurrence in prosody_occurrences
            if occurrence.changed
            and "exact_category_pass" in occurrence.acoustic_status
        )
        changed_insertion_occurrences = tuple(
            occurrence for occurrence in insertion_occurrences if occurrence.changed
        )
        acoustic_validated_insertions = tuple(
            occurrence
            for occurrence in changed_insertion_occurrences
            if "exact_category_pass" in occurrence.acoustic_status
        )
        coverage = CoverageReport(
            source_vowel_occurrences=len(occurrences),
            mapped_vowel_occurrences=len(occurrences),
            changed_vowel_occurrences=len(changed_occurrences),
            identity_vowel_occurrences=len(occurrences) - len(changed_occurrences),
            directly_observed_occurrences=sum(
                occurrence.evidence_tier.startswith("direct_")
                or occurrence.evidence_tier == "direct_assimilation"
                for occurrence in occurrences
            ),
            derived_or_structural_occurrences=sum(
                not (
                    occurrence.evidence_tier.startswith("direct_")
                    or occurrence.evidence_tier == "direct_assimilation"
                )
                for occurrence in occurrences
            ),
            acoustically_validated_changed_occurrences=len(acoustic_validated),
            pending_acoustic_changed_occurrences=(
                len(changed_occurrences) - len(acoustic_validated)
            ),
            changed_word_count=len(target_indexes),
            rules_used=tuple(
                sorted({occurrence.rule_id for occurrence in occurrences})
            ),
            source_consonant_occurrences=len(consonant_occurrences),
            mapped_consonant_occurrences=len(consonant_occurrences),
            changed_consonant_occurrences=len(changed_consonant_occurrences),
            identity_consonant_occurrences=(
                len(consonant_occurrences) - len(changed_consonant_occurrences)
            ),
            directly_observed_consonant_occurrences=sum(
                occurrence.evidence_tier.startswith("direct_")
                for occurrence in consonant_occurrences
            ),
            derived_or_structural_consonant_occurrences=sum(
                not occurrence.evidence_tier.startswith("direct_")
                for occurrence in consonant_occurrences
            ),
            acoustically_validated_changed_consonant_occurrences=len(
                acoustic_validated_consonants
            ),
            pending_acoustic_changed_consonant_occurrences=(
                len(changed_consonant_occurrences) - len(acoustic_validated_consonants)
            ),
            consonant_rules_used=tuple(
                sorted({occurrence.rule_id for occurrence in consonant_occurrences})
            ),
            changed_prosody_occurrences=sum(
                occurrence.changed for occurrence in prosody_occurrences
            ),
            acoustically_validated_changed_prosody_occurrences=len(
                acoustic_validated_prosody
            ),
            pending_acoustic_changed_prosody_occurrences=(
                sum(occurrence.changed for occurrence in prosody_occurrences)
                - len(acoustic_validated_prosody)
            ),
            prosody_rules_used=tuple(
                sorted({occurrence.rule_id for occurrence in prosody_occurrences})
            ),
            changed_insertion_occurrences=len(changed_insertion_occurrences),
            acoustically_validated_changed_insertion_occurrences=len(
                acoustic_validated_insertions
            ),
            pending_acoustic_changed_insertion_occurrences=(
                len(changed_insertion_occurrences) - len(acoustic_validated_insertions)
            ),
            insertion_rules_used=tuple(
                sorted({occurrence.rule_id for occurrence in insertion_occurrences})
            ),
        )

        pattern = _WORD_PATTERNS[analysis.language_id]
        neutral_values = iter(word.neutral_surface for word in words)
        lens_values = iter(word.lens_surface for word in words)
        neutral_script = pattern.sub(lambda _: next(neutral_values), normalized)
        lens_script = pattern.sub(lambda _: next(lens_values), normalized)
        source_skeleton = pattern.sub("", normalized)
        punctuation_preserved = (
            pattern.sub("", neutral_script) == source_skeleton
            and pattern.sub("", lens_script) == source_skeleton
        )
        if not punctuation_preserved:
            raise BilingualVowelEngineError(
                "punctuation_drift", "Carrier scripts changed source punctuation."
            )

        repeated_ok = all(
            len(
                {
                    (
                        word.neutral_surface,
                        word.lens_surface,
                        word.neutral_phone,
                        word.lens_phone,
                    )
                    for word in words
                    if word.source.casefold() == casefold
                    and word.source_phone == source_phone
                }
            )
            == 1
            for casefold, source_phone in {
                (word.source.casefold(), word.source_phone) for word in words
            }
        )
        if not repeated_ok:
            raise BilingualVowelEngineError(
                "repeated_word_drift", "Repeated source mappings diverged."
            )

        gates = GateReport(
            isolated_pairs_checked=len(set(keys)),
            adjacency_pairs_checked=adjacency_checks,
            candidate_attempts=candidate_attempts,
            candidate_rejection_counts=dict(sorted(rejection_counts.items())),
            written_and_espeak_gate_pass=True,
            supplemental_phone_gates_pass=True,
            model_representable=True,
            punctuation_preserved=True,
            repeated_word_invariant_pass=True,
        )
        payload = {
            "engine_version": self.engine_version,
            "profile_id": self.profile["id"],
            "voice_id": self.profile["voice_id"],
            "voice_registry_version": self.profile["voice_registry_version"],
            "voice_registry_sha256": self.profile["voice_registry_sha256"],
            "source_phonemes": analysis.source_phonemes,
            "neutral_phonemes": neutral_phonemes,
            "lens_phonemes": lens_phonemes,
            "target_word_indexes": target_indexes,
            "coverage": asdict(coverage),
            "words": [
                {
                    "source_casefold": word.source.casefold(),
                    "source_phone": word.source_phone,
                    "neutral_phone": word.neutral_phone,
                    "lens_phone": word.lens_phone,
                    "candidate_attempt": word.candidate_attempt,
                }
                for word in words
            ],
        }
        plan_sha256 = hashlib.sha256(stable_json(payload).encode("utf-8")).hexdigest()
        return BilingualVowelPlan(
            engine_version=self.engine_version,
            profile_id=self.profile["id"],
            source_language=self.profile["source_language"],
            listener_language=self.profile["listener_language"],
            voice_id=self.profile["voice_id"],
            voice_registry_version=self.profile["voice_registry_version"],
            voice_registry_sha256=self.profile["voice_registry_sha256"],
            normalized_text=normalized,
            original_source_phonemes=analysis.source_phonemes,
            render_reference_phonemes=neutral_phonemes,
            neutral_phonemes=neutral_phonemes,
            lens_phonemes=lens_phonemes,
            neutral_script=neutral_script,
            lens_script=lens_script,
            target_word_indexes=target_indexes,
            comparison_available=bool(target_indexes),
            words=words,
            coverage=coverage,
            gates=gates,
            plan_sha256=plan_sha256,
        )


def bilingual_alignment_record(
    *,
    model: Any,
    plan: BilingualVowelPlan,
    durations: Sequence[int],
    sample_count: int,
) -> dict[str, Any]:
    """Map every changed segment occurrence onto deterministic decoder samples."""

    expected_duration_count = (
        len(_filtered_symbols(model, plan.render_reference_phonemes)) + 2
    )
    if len(durations) != expected_duration_count:
        raise BilingualVowelEngineError(
            "duration_alignment_drift",
            "Duration count differs from the broad carrier reference plan.",
        )
    total_frames = sum(int(value) for value in durations)
    if total_frames <= 0 or sample_count <= 0 or sample_count % total_frames:
        raise BilingualVowelEngineError(
            "sample_alignment_drift",
            "Decoded samples do not map to integral alignment frames.",
        )
    samples_per_frame = sample_count // total_frames
    cumulative = [0]
    for duration in durations:
        cumulative.append(cumulative[-1] + int(duration))
    word_spans = _word_column_spans(model, plan.neutral_phonemes)
    if len(word_spans) != len(plan.words):
        raise BilingualVowelEngineError(
            "word_alignment_drift", "Decoder word spans differ from the broad plan."
        )

    def interval(columns: Sequence[int]) -> dict[str, Any]:
        if not columns:
            raise BilingualVowelEngineError(
                "empty_alignment_interval", "A changed vowel lost all decoder columns."
            )
        start = cumulative[columns[0]] * samples_per_frame
        end = cumulative[columns[-1] + 1] * samples_per_frame
        if end <= start:
            raise BilingualVowelEngineError(
                "empty_alignment_interval", "A decoder interval has no duration."
            )
        return {
            "columns": list(columns),
            "start_sample": start,
            "end_sample_exclusive": end,
            "start_s": start / SAMPLE_RATE_HZ,
            "end_s": end / SAMPLE_RATE_HZ,
        }

    target_words: list[dict[str, Any]] = []
    target_occurrences: list[dict[str, Any]] = []
    for word_index in plan.target_word_indexes:
        word = plan.words[word_index]
        columns = word_spans[word_index]
        if len(columns) != len(word.neutral_phone):
            raise BilingualVowelEngineError(
                "word_column_drift", "Word state columns differ from its phone plan."
            )
        target_words.append({"word_index": word_index, "interval": interval(columns)})
        typed_occurrences = (
            *(("vowel", occurrence) for occurrence in word.vowel_occurrences),
            *(("consonant", occurrence) for occurrence in word.consonant_occurrences),
            *(("prosody", occurrence) for occurrence in word.prosody_occurrences),
            *(("insertion", occurrence) for occurrence in word.insertion_occurrences),
        )
        for within_word_index, (segment_type, occurrence) in enumerate(
            typed_occurrences
        ):
            if not occurrence.changed:
                continue
            start = occurrence.phone_offset
            stop = start + occurrence.phone_length
            selected = columns[start:stop]
            if (
                len(selected) != occurrence.phone_length
                or word.neutral_phone[start:stop] != occurrence.source
                or word.lens_phone[start:stop] != occurrence.target
            ):
                raise BilingualVowelEngineError(
                    "segment_alignment_drift",
                    "A changed segment no longer matches its frozen phone offsets.",
                )
            measurement_start = getattr(occurrence, "measurement_phone_offset", start)
            measurement_length = getattr(
                occurrence, "measurement_phone_length", occurrence.phone_length
            )
            measurement_selected = columns[
                measurement_start : measurement_start + measurement_length
            ]
            if len(measurement_selected) != measurement_length:
                raise BilingualVowelEngineError(
                    "segment_alignment_drift",
                    "A changed segment lost its measurement span.",
                )
            target_occurrences.append(
                {
                    "occurrence_index": len(target_occurrences),
                    "word_index": word_index,
                    "within_word_index": within_word_index,
                    "segment_type": segment_type,
                    "rule_id": occurrence.rule_id,
                    "source": occurrence.source,
                    "target": occurrence.target,
                    "evidence_tier": occurrence.evidence_tier,
                    "acoustic_status": occurrence.acoustic_status,
                    "control_interval": interval(selected),
                    "measurement_interval": interval(measurement_selected),
                }
            )
    expected_changed = (
        plan.coverage.changed_vowel_occurrences
        + plan.coverage.changed_consonant_occurrences
        + plan.coverage.changed_prosody_occurrences
        + plan.coverage.changed_insertion_occurrences
    )
    if len(target_occurrences) != expected_changed:
        raise BilingualVowelEngineError(
            "segment_alignment_drift", "Alignment lost a changed segment occurrence."
        )
    return {
        "duration_count": len(durations),
        "total_alignment_frames": total_frames,
        "samples_per_alignment_frame": samples_per_frame,
        "target_words": target_words,
        "target_occurrences": target_occurrences,
    }


class BilingualVowelRuntime:
    """Local, zero-API renderer for the broad experimental vowel path."""

    def __init__(
        self,
        *,
        planner: BilingualVowelPlanner,
        synthesis: KokoroSynthesisRuntime,
    ) -> None:
        if getattr(synthesis, "voice_id", "af_heart") != planner.profile["voice_id"]:
            raise BilingualVowelEngineError(
                "voice_profile_mismatch", "Renderer voice and vowel profile disagree."
            )
        self.planner = planner
        self.synthesis = synthesis

    @classmethod
    def load(
        cls, profile_id: str, *, voice_id: str | None = None
    ) -> BilingualVowelRuntime:
        planner = BilingualVowelPlanner.load(profile_id, voice_id=voice_id)
        synthesis = _load_pinned_synthesis_voice(planner.profile["voice_id"])
        return cls(planner=planner, synthesis=synthesis)

    @staticmethod
    def _pcm(audio: np.ndarray) -> np.ndarray:
        return np.frombuffer(pcm16_bytes(audio), dtype="<i2").copy()

    def render(self, text: str) -> BilingualVowelRender | BilingualVowelPlan:
        from .kokoro_output_domain_splice import (
            boundary_artifact_report,
            output_domain_splice,
        )
        from .kokoro_typed_diagnostic import localization_report

        plan = self.planner.plan(text)
        pair = plan.pair_plan()
        if pair is None:
            return plan
        rendered: ParityRender = self.synthesis.render_parity_triplet(pair)
        neutral = self._pcm(rendered.neutral)
        identity = self._pcm(rendered.identity)
        full_lens = self._pcm(rendered.lens)
        alignment = bilingual_alignment_record(
            model=self.synthesis.model,
            plan=plan,
            durations=rendered.predicted_durations,
            sample_count=neutral.size,
        )
        target_intervals = [row["interval"] for row in alignment["target_words"]]
        full_localization = localization_report(neutral, full_lens, target_intervals)
        windows = tuple(full_localization.get("inside_windows", ()))
        if not windows:
            raise BilingualVowelEngineError(
                "splice_window_missing",
                "Changed target words produced no splice window.",
            )
        lens, weights = output_domain_splice(neutral, full_lens, windows)
        boundary = boundary_artifact_report(neutral, full_lens, lens, windows)
        localization = localization_report(neutral, lens, target_intervals)
        arrays = (neutral, identity, full_lens, lens)
        clipped = [
            float(np.mean(np.abs(values.astype(np.int64)) >= 32767))
            for values in arrays
        ]
        equal_nonempty = bool(
            neutral.size
            and neutral.size == identity.size == full_lens.size == lens.size
        )
        finite = all(np.isfinite(values.astype(np.float64)).all() for values in arrays)
        outside_exact = bool(
            np.array_equal(lens[weights == 0.0], neutral[weights == 0.0])
        )
        interior_exact = bool(
            np.any(weights == 1.0)
            and np.array_equal(lens[weights == 1.0], full_lens[weights == 1.0])
        )
        acoustic_ready = bool(
            plan.coverage.pending_acoustic_changed_occurrences == 0
            and plan.coverage.pending_acoustic_changed_consonant_occurrences == 0
            and plan.coverage.pending_acoustic_changed_prosody_occurrences == 0
            and plan.coverage.pending_acoustic_changed_insertion_occurrences == 0
        )
        integrity_pass = bool(
            np.array_equal(neutral, identity)
            and equal_nonempty
            and finite
            and all(value < 0.001 for value in clipped)
            and outside_exact
            and interior_exact
            and boundary.get("pass") is True
            and localization.get("pass") is True
        )
        verification = BilingualRenderVerification(
            neutral_identity_bit_exact=bool(np.array_equal(neutral, identity)),
            equal_nonempty_samples=equal_nonempty,
            finite=finite,
            unclipped=all(value < 0.001 for value in clipped),
            outside_splice_exact_neutral=outside_exact,
            full_weight_interior_exact_lens=interior_exact,
            boundary_metrics_pass=bool(boundary.get("pass")),
            localization_pass=bool(localization.get("pass")),
            localization_fraction=float(
                localization.get("inside_difference_energy_fraction", 0.0)
            ),
            integrity_pass=integrity_pass,
            changed_rules_acoustically_validated=acoustic_ready,
            evidence_status=(
                "integrity_pass_all_changed_rules_acoustically_validated"
                if integrity_pass and acoustic_ready
                else (
                    "integrity_pass_acoustic_validation_pending"
                    if integrity_pass
                    else "automatic_integrity_failed"
                )
            ),
        )
        return BilingualVowelRender(
            plan=plan,
            neutral_pcm=neutral,
            identity_pcm=identity,
            full_lens_pcm=full_lens,
            lens_pcm=lens,
            alignment=alignment,
            lens_alignment=alignment,
            splice_windows=windows,
            verification=verification,
        )


def _load_pinned_synthesis_voice(voice_id: str) -> KokoroSynthesisRuntime:
    """Load a voice beside the immutable af_heart synthesis evidence module."""

    if voice_id == "af_heart":
        runtime = KokoroSynthesisRuntime.load(download=False)
        runtime.voice_id = voice_id
        return runtime
    if importlib.metadata.version("kokoro") != KOKORO_VERSION:
        raise BilingualVowelEngineError(
            "renderer_version_mismatch", f"Kokoro {KOKORO_VERSION} is required."
        )
    from kokoro import KModel
    import torch

    from .kokoro_specs import VOICE_SPECS_BY_ID, resolve_pinned_file

    try:
        voice = VOICE_SPECS_BY_ID[voice_id]
    except KeyError as exc:
        raise BilingualVowelEngineError(
            "unknown_voice", f"Unknown pinned Kokoro voice: {voice_id}"
        ) from exc
    voice_path = resolve_pinned_file(voice.filename, download=False)
    if sha256_file(voice_path) != voice.sha256:
        raise BilingualVowelEngineError(
            "voice_hash_mismatch", f"Pinned voice hash changed: {voice_id}"
        )
    files = verify_model_files(download=False)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    with _INFERENCE_LOCK:
        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
        torch.backends.mkldnn.enabled = False
        torch.backends.nnpack.set_flags(False)
        torch.use_deterministic_algorithms(True)
        model = (
            KModel(
                repo_id=MODEL_REPO,
                config=str(files[CONFIG_FILE]),
                model=str(files[MODEL_FILE]),
            )
            .to("cpu")
            .eval()
        )
        voice_pack = torch.load(voice_path, map_location="cpu", weights_only=True)
    runtime = KokoroSynthesisRuntime(model, voice_pack, torch)
    runtime.voice_id = voice_id
    return runtime


def load_profiles(path: Path = BILINGUAL_RULES_PATH) -> dict[str, dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if (
        data.get("schema_version") != 1
        or data.get("engine_version") != BILINGUAL_RULES_VERSION
    ):
        raise BilingualVowelEngineError(
            "profile_schema_mismatch", "Bilingual vowel profile schema drifted."
        )
    profiles = data.get("profiles")
    if not isinstance(profiles, list) or not profiles:
        raise BilingualVowelEngineError(
            "profile_schema_mismatch", "Bilingual vowel profiles are missing."
        )
    by_id = {profile["id"]: profile for profile in profiles}
    if len(by_id) != len(profiles):
        raise BilingualVowelEngineError(
            "profile_schema_mismatch", "Bilingual vowel profile IDs are duplicated."
        )
    return by_id
