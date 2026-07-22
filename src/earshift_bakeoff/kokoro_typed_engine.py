from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.resources
import json
import re
import threading
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

import numpy as np

from .config import ROOT, Paths, load_json_yaml, stable_json
from .kokoro_gate_bridge import KokoroGateIndex
from .kokoro_synthesis import (
    CONFIG_FILE,
    MAX_PHONEME_CHARACTERS,
    PairPlan,
    PairRender,
    KokoroSynthesisRuntime,
    verify_model_files,
)
from .listener_lens import (
    ALLOWED_INPUT_RE,
    LENS_RULES_PATH,
    WORD_RE,
    DatabaseNonceChecker,
    EspeakWordAnalyzer,
    NonceDecision,
    _clean_ipa,
    _nonce_decision,
)
from .util import sha256_file


TYPED_ENGINE_VERSION = 1
RULE_ID = "ptbr.vowel.ae_to_eh"
SOURCE_PHONE = "æ"
TARGET_PHONE = "ɛ"
MAX_CANDIDATE_ATTEMPTS = 256
MAX_RESOLUTION_ROUNDS = 64
MAX_CLIPPED_FRACTION = 0.001

EN_CORE_WEB_SM_VERSION = "3.8.0"
SPACY_VERSION = "3.8.14"

_VOWEL_SYMBOLS = frozenset("aAiIuUeEoOæɑɒɔəɚɜɐɛɪʊʌWYWQᵊᵻɨɯɤøœ")
_CONSONANT_SYMBOLS = frozenset("bd fɡhjk lmnŋpɹrstTvwzʃʒθðʧʤ".replace(" ", ""))
_STRUCTURAL_SYMBOLS = frozenset("ˈˌːʰʲ̃")
_WORD_PUNCTUATION = frozenset(';:,.!?—…"()“”')

_CONTENT_VOWELS = ("ɪ", "ʊ", "ə", "ʌ", "A", "O", "I", "W")
_WEAK_VOWELS = ("ə", "ɪ", "ɐ")
_CONTENT_CONSONANTS = (
    "b",
    "d",
    "f",
    "ɡ",
    "k",
    "l",
    "m",
    "n",
    "p",
    "ɹ",
    "s",
    "t",
    "v",
    "z",
    "ʃ",
    "ʒ",
    "θ",
    "ð",
    "ʧ",
    "ʤ",
    "w",
    "j",
    "h",
    "ŋ",
)
_WEAK_CONSONANTS = ("b", "d", "f", "l", "m", "n", "ɹ", "s", "t", "v", "z")

_SPELLING = {
    "a": "ah",
    "A": "ay",
    "I": "eye",
    "O": "oh",
    "Q": "aw",
    "W": "ow",
    "Y": "oy",
    "i": "ee",
    "u": "oo",
    "æ": "a",
    "ɑ": "ah",
    "ɒ": "ah",
    "ɔ": "aw",
    "ə": "uh",
    "ɚ": "er",
    "ɜ": "er",
    "ɐ": "uh",
    "ɛ": "eh",
    "ɪ": "ih",
    "ʊ": "uu",
    "ʌ": "uh",
    "ᵊ": "uh",
    "ᵻ": "ih",
    "ɨ": "ih",
    "ɯ": "oo",
    "ɤ": "uh",
    "ø": "uh",
    "œ": "uh",
    "b": "b",
    "d": "d",
    "f": "f",
    "ɡ": "g",
    "h": "h",
    "j": "y",
    "k": "k",
    "l": "l",
    "m": "m",
    "n": "n",
    "ŋ": "ng",
    "p": "p",
    "ɹ": "r",
    "s": "s",
    "t": "t",
    "T": "d",
    "v": "v",
    "w": "w",
    "z": "z",
    "ʃ": "sh",
    "ʒ": "zh",
    "θ": "th",
    "ð": "dh",
    "ʧ": "ch",
    "ʤ": "j",
}


class KokoroTypedEngineError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class G2PCallable(Protocol):
    def __call__(self, text: str) -> tuple[str, list[Any]]: ...


class SourceWordAnalyzer(Protocol):
    def phonemize_words(self, words: Sequence[str], voice: str) -> list[str]: ...


class NonceChecker(Protocol):
    @property
    def enabled(self) -> bool: ...

    def check(
        self, surface: str, language: str, previous_surface: str | None
    ) -> NonceDecision: ...


class PhoneIndex(Protocol):
    def phone_match(self, phone: str) -> bool: ...


@dataclass(frozen=True)
class MappingKey:
    source_casefold: str
    source_phone: str
    target_offsets: tuple[int, ...]
    carrier_role: str


@dataclass(frozen=True)
class CarrierAssignment:
    neutral_surface: str
    lens_surface: str
    neutral_phone: str
    lens_phone: str
    candidate_attempt: int


@dataclass(frozen=True)
class TypedWord:
    word_index: int
    source: str
    source_phone: str
    source_espeak_ipa: str
    carrier_role: str
    neutral_surface: str
    lens_surface: str
    neutral_phone: str
    lens_phone: str
    target_offsets: tuple[int, ...]
    candidate_attempt: int


@dataclass(frozen=True)
class GateSummary:
    source_eligibility_agreement: bool
    isolated_pairs_checked: int
    adjacency_pairs_checked: int
    candidate_attempts: int
    candidate_rejection_counts: dict[str, int]
    espeak_gate_pass: bool
    kokoro_phone_gate_pass: bool
    exact_plan_representable: bool


@dataclass(frozen=True)
class TypedPlan:
    engine_version: int
    normalized_text: str
    source_phonemes: str
    neutral_phonemes: str
    lens_phonemes: str
    neutral_script: str
    lens_script: str
    target_word_indexes: tuple[int, ...]
    target_word_count: int
    target_occurrence_count: int
    coverage_count: int
    comparison_available: bool
    words: tuple[TypedWord, ...]
    gate_summary: GateSummary
    plan_sha256: str

    def pair_plan(self) -> PairPlan | None:
        if not self.comparison_available:
            return None
        return PairPlan(
            source_phonemes=self.source_phonemes,
            neutral_phonemes=self.neutral_phonemes,
            lens_phonemes=self.lens_phonemes,
            target_word_indexes=self.target_word_indexes,
        )

    def safe_metadata(self) -> dict[str, Any]:
        return {
            "engine_version": self.engine_version,
            "plan_sha256": self.plan_sha256,
            "target_word_count": self.target_word_count,
            "target_occurrence_count": self.target_occurrence_count,
            "coverage_count": self.coverage_count,
            "comparison_available": self.comparison_available,
            "word_count": len(self.words),
            "gate_summary": asdict(self.gate_summary),
        }


@dataclass(frozen=True)
class RenderIntegrity:
    sample_count: int
    sample_count_equal: bool
    finite: bool
    neutral_clipped_fraction: float
    lens_clipped_fraction: float
    clipping_pass: bool
    pass_all: bool


@dataclass(frozen=True)
class TypedRender:
    plan: TypedPlan
    audio: PairRender
    integrity: RenderIntegrity


def merge_spans(spans: Sequence[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    ordered = sorted(set(spans))
    merged: list[tuple[int, int]] = []
    for start, end in ordered:
        if start < 0 or end <= start:
            raise KokoroTypedEngineError("invalid_target_span", "Invalid target span.")
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return tuple(merged)


def _filtered_count(value: str, vocab: set[str]) -> int:
    unsupported = sorted(set(value) - vocab)
    if unsupported:
        raise KokoroTypedEngineError(
            "unrepresentable_phone_plan",
            f"Phone plan contains unsupported model symbols: {''.join(unsupported)}",
        )
    return len(value)


def _surface_for(phone: str) -> str:
    chunks: list[str] = []
    for symbol in phone:
        if symbol in _STRUCTURAL_SYMBOLS:
            continue
        spelling = _SPELLING.get(symbol)
        if spelling is None:
            raise KokoroTypedEngineError(
                "unspellable_phone_plan",
                f"No frozen carrier spelling exists for {symbol!r}.",
            )
        chunks.append(spelling)
    surface = "".join(chunks)
    if not surface or not surface.isascii() or not surface.isalpha():
        raise KokoroTypedEngineError(
            "invalid_carrier_surface", "Carrier spelling is not one ASCII word."
        )
    return surface


def _pick(inventory: Sequence[str], digest: bytes, position: int, attempt: int) -> str:
    index = (digest[position % len(digest)] + position + attempt) % len(inventory)
    return inventory[index]


def _is_word_token(token: Any) -> bool:
    return WORD_RE.fullmatch(str(token.text)) is not None


def _normalize_input(text: str, policy: dict[str, Any]) -> str:
    normalized = re.sub(r"\s+", " ", text.strip())
    if not normalized:
        raise KokoroTypedEngineError(
            "empty_input", "Enter one or two short English sentences."
        )
    if len(normalized) > int(policy["max_characters"]):
        raise KokoroTypedEngineError("input_too_long", "Input is too long.")
    if not ALLOWED_INPUT_RE.fullmatch(normalized):
        raise KokoroTypedEngineError(
            "unsupported_characters",
            "Use English words, apostrophes, spaces, and basic punctuation only.",
        )
    words = WORD_RE.findall(normalized)
    if not 2 <= len(words) <= int(policy["max_words"]):
        raise KokoroTypedEngineError(
            "unsupported_word_count", "Unsupported word count."
        )
    sentence_count = len(re.findall(r"[.!?]+", normalized)) or 1
    if sentence_count > int(policy["max_sentences"]):
        raise KokoroTypedEngineError(
            "unsupported_sentence_count", "Too many sentences."
        )
    if any(len(word) > 1 and word.isupper() for word in words):
        raise KokoroTypedEngineError("unsupported_acronym", "Acronyms are unsupported.")
    return normalized


class KokoroTypedPlanner:
    def __init__(
        self,
        *,
        g2p: G2PCallable,
        model_vocab: set[str],
        source_analyzer: SourceWordAnalyzer,
        nonce_checker: NonceChecker,
        phone_index: PhoneIndex,
        rules_path: Path = LENS_RULES_PATH,
    ) -> None:
        self.g2p = g2p
        self.model_vocab = model_vocab
        self.source_analyzer = source_analyzer
        self.nonce_checker = nonce_checker
        self.phone_index = phone_index
        self.rules_path = rules_path
        self.rules = load_json_yaml(rules_path)
        self.weak_words = frozenset(
            word.casefold()
            for word in self.rules["weak_carrier_policy"]["source_function_words"]
        )
        self._g2p_lock = threading.RLock()

    @classmethod
    def load(cls) -> KokoroTypedPlanner:
        if importlib.metadata.version("en-core-web-sm") != EN_CORE_WEB_SM_VERSION:
            raise KokoroTypedEngineError(
                "parser_version_mismatch",
                f"en-core-web-sm {EN_CORE_WEB_SM_VERSION} is required.",
            )
        if importlib.metadata.version("spacy") != SPACY_VERSION:
            raise KokoroTypedEngineError(
                "parser_version_mismatch", f"spaCy {SPACY_VERSION} is required."
            )
        from misaki.en import G2P
        from misaki.espeak import EspeakFallback

        files = verify_model_files(download=False)
        config = json.loads(files[CONFIG_FILE].read_text(encoding="utf-8"))
        return cls(
            g2p=G2P(british=False, fallback=EspeakFallback(british=False)),
            model_vocab=set(config["vocab"]),
            source_analyzer=EspeakWordAnalyzer(),
            nonce_checker=DatabaseNonceChecker(),
            phone_index=KokoroGateIndex(),
        )

    def _candidate(self, key: MappingKey, attempt: int) -> CarrierAssignment:
        digest = hashlib.sha256(
            stable_json(
                {
                    "engine_version": TYPED_ENGINE_VERSION,
                    "key": asdict(key),
                    "attempt": attempt,
                }
            ).encode("utf-8")
        ).digest()
        vowels = _WEAK_VOWELS if key.carrier_role == "weak" else _CONTENT_VOWELS
        consonants = (
            _WEAK_CONSONANTS if key.carrier_role == "weak" else _CONTENT_CONSONANTS
        )
        neutral: list[str] = []
        lens: list[str] = []
        target_offsets = set(key.target_offsets)
        for position, symbol in enumerate(key.source_phone):
            if position in target_offsets:
                if symbol != SOURCE_PHONE:
                    raise KokoroTypedEngineError(
                        "target_offset_drift", "Target offset no longer points to /æ/."
                    )
                neutral.append(SOURCE_PHONE)
                lens.append(TARGET_PHONE)
            elif symbol in _STRUCTURAL_SYMBOLS:
                neutral.append(symbol)
                lens.append(symbol)
            elif symbol in _VOWEL_SYMBOLS:
                selected = _pick(vowels, digest, position, attempt)
                neutral.append(selected)
                lens.append(selected)
            elif symbol in _CONSONANT_SYMBOLS:
                selected = _pick(consonants, digest, position, attempt)
                neutral.append(selected)
                lens.append(selected)
            else:
                raise KokoroTypedEngineError(
                    "unsupported_source_phone",
                    f"Source phone symbol {symbol!r} has no frozen carrier class.",
                )
        neutral_phone = "".join(neutral)
        lens_phone = "".join(lens)
        if len(neutral_phone) != len(key.source_phone) or len(lens_phone) != len(
            key.source_phone
        ):
            raise KokoroTypedEngineError(
                "carrier_length_drift", "Carrier phone count changed."
            )
        return CarrierAssignment(
            neutral_surface=_surface_for(neutral_phone),
            lens_surface=_surface_for(lens_phone),
            neutral_phone=neutral_phone,
            lens_phone=lens_phone,
            candidate_attempt=attempt,
        )

    def _isolated_reasons(self, assignment: CarrierAssignment) -> set[str]:
        reasons: set[str] = set()
        seen: set[tuple[str, str]] = set()
        for side, surface, phone in (
            ("neutral", assignment.neutral_surface, assignment.neutral_phone),
            ("lens", assignment.lens_surface, assignment.lens_phone),
        ):
            signature = (surface, phone)
            if signature in seen:
                continue
            seen.add(signature)
            decision = _nonce_decision(self.nonce_checker, surface, "en", None)
            if not decision.accepted:
                reasons.add(f"{side}_espeak_{decision.rejection_reason or 'rejected'}")
            if self.phone_index.phone_match(phone):
                reasons.add(f"{side}_kokoro_predicted_homophone")
            try:
                _filtered_count(phone, self.model_vocab)
            except KokoroTypedEngineError:
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
                    raise KokoroTypedEngineError(
                        "candidate_search_exhausted",
                        "No gate-clean carrier mapping exists within the bounded search.",
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
                    decision = _nonce_decision(
                        self.nonce_checker,
                        current_surface,
                        "en",
                        previous_surface,
                    )
                    if not decision.accepted:
                        rejection_counts[
                            f"{side}_espeak_{decision.rejection_reason or 'adjacency_rejected'}"
                        ] += 1
                        conflicts.update((previous_key, current_key))
                    if self.phone_index.phone_match(previous_phone + current_phone):
                        rejection_counts[
                            f"{side}_kokoro_adjacency_predicted_homophone"
                        ] += 1
                        conflicts.update((previous_key, current_key))
            if not conflicts:
                return assignments, total_attempts, adjacency_checks
            implicated = conflicts
            for key in implicated:
                next_attempt[key] = assignments[key].candidate_attempt + 1

        raise KokoroTypedEngineError(
            "adjacency_search_exhausted",
            "No globally gate-clean carrier assignment exists within the bounded search.",
        )

    def plan(self, text: str) -> TypedPlan:
        candidate_rejections: Counter[str] = Counter()
        normalized = _normalize_input(text, self.rules["runtime_policy"])
        with self._g2p_lock:
            source_phonemes, tokens = self.g2p(normalized)
        if not tokens:
            raise KokoroTypedEngineError("g2p_empty", "Kokoro G2P returned no tokens.")
        reconstructed = "".join(
            str(token.text) + str(token.whitespace) for token in tokens
        ).strip()
        if reconstructed != normalized:
            raise KokoroTypedEngineError(
                "tokenization_drift", "Kokoro G2P changed the input tokenization."
            )
        lexical_tokens = [token for token in tokens if _is_word_token(token)]
        source_words = WORD_RE.findall(normalized)
        if [str(token.text) for token in lexical_tokens] != source_words:
            raise KokoroTypedEngineError(
                "word_alignment_drift",
                "Kokoro G2P word tokens do not match the validated input words.",
            )
        if any(
            token.phonemes is None or not str(token.phonemes)
            for token in lexical_tokens
        ):
            raise KokoroTypedEngineError(
                "unpronounceable_source_word", "A source word has no Kokoro phone plan."
            )
        joined_source = "".join(
            ("❓" if token.phonemes is None else str(token.phonemes))
            + str(token.whitespace)
            for token in tokens
        )
        if joined_source != source_phonemes or "❓" in source_phonemes:
            raise KokoroTypedEngineError(
                "source_plan_drift", "Kokoro source phone plan is incomplete."
            )
        _filtered_count(source_phonemes, self.model_vocab)

        source_espeak = self.source_analyzer.phonemize_words(source_words, "en-us")
        if len(source_espeak) != len(source_words):
            raise KokoroTypedEngineError(
                "espeak_alignment_drift", "eSpeak source analysis lost word alignment."
            )

        keys: list[MappingKey] = []
        target_indexes: list[int] = []
        target_occurrences = 0
        for word_index, (word, token, espeak_ipa) in enumerate(
            zip(source_words, lexical_tokens, source_espeak, strict=True)
        ):
            source_phone = str(token.phonemes)
            target_offsets = tuple(
                index
                for index, symbol in enumerate(source_phone)
                if symbol == SOURCE_PHONE
            )
            espeak_count = _clean_ipa(espeak_ipa).count(SOURCE_PHONE)
            if len(target_offsets) != espeak_count:
                raise KokoroTypedEngineError(
                    "eligible_target_disagreement",
                    "Kokoro and eSpeak disagree about eligible /æ/ coverage.",
                )
            if target_offsets:
                target_indexes.append(word_index)
                target_occurrences += len(target_offsets)
            carrier_role = (
                "weak"
                if word.casefold() in self.weak_words and not target_offsets
                else "content"
            )
            keys.append(
                MappingKey(
                    source_casefold=word.casefold(),
                    source_phone=source_phone,
                    target_offsets=target_offsets,
                    carrier_role=carrier_role,
                )
            )

        assignments, candidate_attempts, adjacency_checks = self._resolve_assignments(
            keys, candidate_rejections
        )
        typed_words: list[TypedWord] = []
        replacement_by_token: dict[int, tuple[str, str]] = {}
        lexical_cursor = 0
        for token_index, token in enumerate(tokens):
            if not _is_word_token(token):
                continue
            key = keys[lexical_cursor]
            assignment = assignments[key]
            typed_words.append(
                TypedWord(
                    word_index=lexical_cursor,
                    source=str(token.text),
                    source_phone=key.source_phone,
                    source_espeak_ipa=_clean_ipa(source_espeak[lexical_cursor]),
                    carrier_role=key.carrier_role,
                    neutral_surface=assignment.neutral_surface,
                    lens_surface=assignment.lens_surface,
                    neutral_phone=assignment.neutral_phone,
                    lens_phone=assignment.lens_phone,
                    target_offsets=key.target_offsets,
                    candidate_attempt=assignment.candidate_attempt,
                )
            )
            replacement_by_token[token_index] = (
                assignment.neutral_phone,
                assignment.lens_phone,
            )
            lexical_cursor += 1

        neutral_chunks: list[str] = []
        lens_chunks: list[str] = []
        for token_index, token in enumerate(tokens):
            if token_index in replacement_by_token:
                neutral_phone, lens_phone = replacement_by_token[token_index]
            else:
                neutral_phone = lens_phone = str(token.phonemes or "")
                if neutral_phone and not set(neutral_phone) <= _WORD_PUNCTUATION:
                    raise KokoroTypedEngineError(
                        "unsupported_nonword_token",
                        "A non-word token produced an unsupported phone plan.",
                    )
            neutral_chunks.append(neutral_phone + str(token.whitespace))
            lens_chunks.append(lens_phone + str(token.whitespace))
        neutral_phonemes = "".join(neutral_chunks)
        lens_phonemes = "".join(lens_chunks)

        source_count = _filtered_count(source_phonemes, self.model_vocab)
        neutral_count = _filtered_count(neutral_phonemes, self.model_vocab)
        lens_count = _filtered_count(lens_phonemes, self.model_vocab)
        if not source_count == neutral_count == lens_count:
            raise KokoroTypedEngineError(
                "plan_token_count_mismatch",
                "Source, neutral, and lens plans have unequal model-token counts.",
            )
        if max(source_count, neutral_count, lens_count) > MAX_PHONEME_CHARACTERS:
            raise KokoroTypedEngineError(
                "phone_plan_too_long", "The Kokoro phone plan is too long."
            )
        differences = [
            (left, right)
            for left, right in zip(neutral_phonemes, lens_phonemes, strict=True)
            if left != right
        ]
        if len(differences) != target_occurrences or any(
            pair != (SOURCE_PHONE, TARGET_PHONE) for pair in differences
        ):
            raise KokoroTypedEngineError(
                "lens_difference_drift",
                "Neutral/lens phone differences no longer match target coverage.",
            )

        neutral_script = WORD_RE.sub(
            lambda match, values=iter(word.neutral_surface for word in typed_words): (
                next(values)
            ),
            normalized,
        )
        lens_script = WORD_RE.sub(
            lambda match, values=iter(word.lens_surface for word in typed_words): next(
                values
            ),
            normalized,
        )
        if WORD_RE.sub("", neutral_script) != WORD_RE.sub("", normalized):
            raise KokoroTypedEngineError(
                "punctuation_drift", "Neutral carrier changed source punctuation."
            )
        if WORD_RE.sub("", lens_script) != WORD_RE.sub("", normalized):
            raise KokoroTypedEngineError(
                "punctuation_drift", "Lens carrier changed source punctuation."
            )

        gate_summary = GateSummary(
            source_eligibility_agreement=True,
            isolated_pairs_checked=len(set(keys)),
            adjacency_pairs_checked=adjacency_checks,
            candidate_attempts=candidate_attempts,
            candidate_rejection_counts=dict(sorted(candidate_rejections.items())),
            espeak_gate_pass=True,
            kokoro_phone_gate_pass=True,
            exact_plan_representable=True,
        )
        payload = {
            "engine_version": TYPED_ENGINE_VERSION,
            "source_phonemes": source_phonemes,
            "neutral_phonemes": neutral_phonemes,
            "lens_phonemes": lens_phonemes,
            "target_word_indexes": sorted(set(target_indexes)),
            "target_occurrence_count": target_occurrences,
            "words": [
                {
                    "source_casefold": word.source.casefold(),
                    "source_phone": word.source_phone,
                    "neutral_phone": word.neutral_phone,
                    "lens_phone": word.lens_phone,
                    "target_offsets": word.target_offsets,
                }
                for word in typed_words
            ],
        }
        plan_hash = hashlib.sha256(stable_json(payload).encode("utf-8")).hexdigest()
        return TypedPlan(
            engine_version=TYPED_ENGINE_VERSION,
            normalized_text=normalized,
            source_phonemes=source_phonemes,
            neutral_phonemes=neutral_phonemes,
            lens_phonemes=lens_phonemes,
            neutral_script=neutral_script,
            lens_script=lens_script,
            target_word_indexes=tuple(sorted(set(target_indexes))),
            target_word_count=len(set(target_indexes)),
            target_occurrence_count=target_occurrences,
            coverage_count=target_occurrences,
            comparison_available=target_occurrences > 0,
            words=tuple(typed_words),
            gate_summary=gate_summary,
            plan_sha256=plan_hash,
        )


def inspect_render(render: PairRender) -> RenderIntegrity:
    neutral = np.asarray(render.neutral).reshape(-1)
    lens = np.asarray(render.lens).reshape(-1)
    sample_count_equal = bool(neutral.size == lens.size and neutral.size > 0)
    finite = bool(np.isfinite(neutral).all() and np.isfinite(lens).all())
    neutral_clipped = (
        float(np.mean(np.abs(neutral) >= 1.0)) if neutral.size and finite else 1.0
    )
    lens_clipped = float(np.mean(np.abs(lens) >= 1.0)) if lens.size and finite else 1.0
    clipping_pass = bool(
        neutral_clipped < MAX_CLIPPED_FRACTION and lens_clipped < MAX_CLIPPED_FRACTION
    )
    return RenderIntegrity(
        sample_count=int(neutral.size if sample_count_equal else 0),
        sample_count_equal=sample_count_equal,
        finite=finite,
        neutral_clipped_fraction=neutral_clipped,
        lens_clipped_fraction=lens_clipped,
        clipping_pass=clipping_pass,
        pass_all=sample_count_equal and finite and clipping_pass,
    )


class KokoroTypedRuntime:
    def __init__(
        self, planner: KokoroTypedPlanner, synthesis: KokoroSynthesisRuntime
    ) -> None:
        self.planner = planner
        self.synthesis = synthesis

    @classmethod
    def load(cls) -> KokoroTypedRuntime:
        return cls(KokoroTypedPlanner.load(), KokoroSynthesisRuntime.load())

    def render(self, text: str) -> TypedRender | TypedPlan:
        plan = self.planner.plan(text)
        pair_plan = plan.pair_plan()
        if pair_plan is None:
            return plan
        audio = self.synthesis.render_pair(pair_plan)
        integrity = inspect_render(audio)
        if not integrity.pass_all:
            raise KokoroTypedEngineError(
                "audio_integrity_failed", "Controlled audio failed runtime integrity."
            )
        return TypedRender(plan=plan, audio=audio, integrity=integrity)


def local_engine_assets() -> dict[str, Any]:
    import inspect

    import en_core_web_sm
    import kokoro.pipeline
    import misaki.en
    from misaki import data as misaki_data

    def tree_hash(root: Path) -> tuple[str, int]:
        digest = hashlib.sha256()
        files = sorted(
            path
            for path in root.rglob("*")
            if path.is_file()
            and "__pycache__" not in path.parts
            and path.suffix != ".pyc"
        )
        for path in files:
            relative = str(path.relative_to(root)).encode("utf-8")
            content = path.read_bytes()
            digest.update(len(relative).to_bytes(4, "big"))
            digest.update(relative)
            digest.update(len(content).to_bytes(8, "big"))
            digest.update(content)
        return digest.hexdigest(), len(files)

    parser_root = Path(inspect.getfile(en_core_web_sm)).parent
    parser_tree_sha256, parser_file_count = tree_hash(parser_root)
    model_files = verify_model_files(download=False)
    return {
        "engine_version": TYPED_ENGINE_VERSION,
        "dependency_lock_sha256": sha256_file(ROOT / "uv.lock"),
        "rules_sha256": hashlib.sha256(LENS_RULES_PATH.read_bytes()).hexdigest(),
        "gate_database_sha256": hashlib.sha256(
            Paths().gate_db.read_bytes()
        ).hexdigest(),
        "kokoro_gate_database_sha256": hashlib.sha256(
            Paths().kokoro_gate_db.read_bytes()
        ).hexdigest(),
        "packages": {
            "kokoro": {
                "version": importlib.metadata.version("kokoro"),
                "pipeline_sha256": sha256_file(Path(inspect.getfile(kokoro.pipeline))),
            },
            "misaki": {
                "version": importlib.metadata.version("misaki"),
                "en_py_sha256": sha256_file(Path(inspect.getfile(misaki.en))),
                "us_gold_sha256": sha256_file(
                    Path(str(importlib.resources.files(misaki_data) / "us_gold.json"))
                ),
                "us_silver_sha256": sha256_file(
                    Path(str(importlib.resources.files(misaki_data) / "us_silver.json"))
                ),
            },
            "spacy": {"version": importlib.metadata.version("spacy")},
            "en_core_web_sm": {
                "version": importlib.metadata.version("en-core-web-sm"),
                "tree_sha256": parser_tree_sha256,
                "file_count": parser_file_count,
            },
            "torch": {"version": importlib.metadata.version("torch")},
        },
        "model_files": {
            name: sha256_file(path) for name, path in sorted(model_files.items())
        },
    }
