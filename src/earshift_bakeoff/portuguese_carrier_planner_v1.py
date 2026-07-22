from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import threading
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Protocol, Sequence

from .config import Paths, stable_json
from .kokoro_specs import (
    LANGUAGE_SPECS,
    VOICE_SPECS_BY_ID,
    VoiceSpec,
    resolve_pinned_file,
)
from .kokoro_synthesis import CONFIG_FILE, MAX_PHONEME_CHARACTERS, MODEL_REPO
from .listener_lens import DatabaseNonceChecker, NonceDecision
from .portuguese_kokoro_gate import PortugueseKokoroGateIndex


PLANNER_VERSION = 1
LANGUAGE_ID = "pt-BR"
WORD_LANGUAGE = "pt"
KOKORO_LANG_CODE = "p"
ESPEAK_VOICE = "pt-br"
PORTUGUESE_RENDERER_CANDIDATE_ENABLED = False
PRODUCTION_ROUTE_AVAILABLE = False
MAX_CANDIDATE_ATTEMPTS = 256
MAX_RESOLUTION_ROUNDS = 64

PORTUGUESE_SMOKE_FIXTURE_ID_V1 = "ptbr-real-and-opaque-smoke-v1"
PORTUGUESE_SMOKE_TEXT_V1 = "A avó comprou pão, e a tia chamou a filha."
PORTUGUESE_SMOKE_TARGET_PHONE_V1 = "ɔ"
TARGET_ANALYSIS_SCOPE = "isolated_native_kokoro_word_predictions"
NATIVE_INDEX_SCOPE = "partial_positive_only_index"

PORTUGUESE_WORD_RE = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)?", flags=re.UNICODE)
_BASIC_PUNCTUATION = frozenset("'’.,!?;: ")
_PHONE_PUNCTUATION = frozenset("'’.,!?;: “”—()")
_STRUCTURAL_PHONE_SYMBOLS = frozenset("ˈˌːʰʲ̃")

# This inventory describes the symbols observed or intentionally supported by the
# pinned pt-BR Kokoro/Misaki path. It is a renderer contract, not a dialect claim.
PORTUGUESE_KOKORO_VOWEL_SYMBOLS = frozenset("AIWYaeiouyæõũɐɔəɛɪʊ")
PORTUGUESE_ORAL_VOWELS = frozenset("aeiouɐɛɔæɪʊə")
PORTUGUESE_NASAL_COMPONENTS = frozenset(("ɐ̃", "ẽ", "ĩ", "õ", "ũ", "ʊ̃"))
PORTUGUESE_KOKORO_CONSONANT_SYMBOLS = frozenset(
    "bdfɡjklmnŋpɾrstvwzʃʒxɲʧʤ"  # Kokoro uses /lj/ for many lh words.
)

_CONTENT_ONSETS = (
    "b",
    "d",
    "f",
    "g",
    "l",
    "m",
    "n",
    "p",
    "r",
    "s",
    "t",
    "v",
    "z",
    "br",
    "dr",
    "fl",
    "fr",
    "gr",
    "pl",
    "pr",
    "tr",
    "vr",
)
_CONTENT_NUCLEI = ("a", "e", "i", "o", "u")
_CONTENT_CODAS = ("", "", "", "l", "m", "n", "r", "s")
_WEAK_ONSETS = ("b", "d", "f", "l", "m", "n", "s", "v", "z")
_WEAK_NUCLEI = ("i", "u", "e")
_WEAK_CODAS = ("m", "n", "l", "s")
_PRODUCTIVE_ENDINGS = ("mente", "ção", "ções")

PORTUGUESE_WEAK_WORDS = frozenset(
    {
        "a",
        "ao",
        "aos",
        "as",
        "com",
        "da",
        "das",
        "de",
        "do",
        "dos",
        "e",
        "em",
        "eu",
        "lhe",
        "lhes",
        "me",
        "na",
        "nas",
        "no",
        "nos",
        "o",
        "os",
        "ou",
        "por",
        "que",
        "se",
        "sem",
        "te",
        "um",
        "uma",
    }
)


class PortugueseCarrierPlannerError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class PortugueseG2P(Protocol):
    language_id: str
    kokoro_lang_code: str
    voice_id: str

    def phonemize_phrase(self, text: str) -> str: ...

    def phonemize_words(self, words: Sequence[str]) -> list[str]: ...


class MandatoryPortugueseGate(Protocol):
    @property
    def enabled(self) -> bool: ...

    def check(
        self, surface: str, language: str, previous_surface: str | None
    ) -> NonceDecision: ...


class SupplementalPositiveIndex(Protocol):
    scope: str

    def phone_match(self, phone: str) -> bool: ...


@dataclass(frozen=True)
class PortugueseMappingKey:
    source_casefold: str
    source_phone: str
    carrier_role: str


@dataclass(frozen=True)
class PortugueseCarrierAssignment:
    surface: str
    phone: str
    candidate_attempt: int


@dataclass(frozen=True)
class PortugueseCarrierWord:
    word_index: int
    source: str
    source_phone: str
    carrier_role: str
    carrier_surface: str
    carrier_phone: str
    target_occurrence_count: int
    candidate_attempt: int


@dataclass(frozen=True)
class PortugueseGateReceipt:
    mandatory_written_espeak_gate_pass: bool
    mandatory_gate_language: str
    mandatory_espeak_voice: str
    native_index_scope: str
    native_positive_only_gate_pass: bool
    native_negative_used_for_clearance: bool
    isolated_candidates_checked: int
    adjacency_pairs_checked: int
    candidate_attempts: int
    candidate_rejection_counts: dict[str, int]
    exact_native_phrase_plan: bool
    model_representable: bool


@dataclass(frozen=True)
class PortugueseCarrierPlan:
    planner_version: int
    language_id: str
    kokoro_lang_code: str
    voice_id: str
    candidate_enabled: bool
    production_route_available: bool
    normalized_text: str
    source_phonemes: str
    carrier_script: str
    carrier_phonemes: str
    target_phone: str
    target_analysis_scope: str
    target_word_indexes: tuple[int, ...]
    target_word_count: int
    target_occurrence_count: int
    target_available: bool
    words: tuple[PortugueseCarrierWord, ...]
    gate_receipt: PortugueseGateReceipt
    plan_sha256: str

    def safe_metadata(self) -> dict[str, Any]:
        """Return routing/cache metadata without typed source or carrier text."""

        return {
            "planner_version": self.planner_version,
            "language_id": self.language_id,
            "kokoro_lang_code": self.kokoro_lang_code,
            "voice_id": self.voice_id,
            "candidate_enabled": self.candidate_enabled,
            "production_route_available": self.production_route_available,
            "plan_sha256": self.plan_sha256,
            "target_phone": self.target_phone,
            "target_analysis_scope": self.target_analysis_scope,
            "target_word_count": self.target_word_count,
            "target_occurrence_count": self.target_occurrence_count,
            "target_available": self.target_available,
            "word_count": len(self.words),
            "gate_receipt": asdict(self.gate_receipt),
        }

    def screening_receipt(self) -> dict[str, Any]:
        """Return the exact local carrier contract used by a frozen screen."""

        return {
            **self.safe_metadata(),
            "source_text_sha256": hashlib.sha256(
                self.normalized_text.encode("utf-8")
            ).hexdigest(),
            "source_phonemes": self.source_phonemes,
            "carrier_script": self.carrier_script,
            "carrier_phonemes": self.carrier_phonemes,
            "target_word_indexes": list(self.target_word_indexes),
            "words": [
                {
                    "word_index": word.word_index,
                    "source_casefold_sha256": hashlib.sha256(
                        word.source.casefold().encode("utf-8")
                    ).hexdigest(),
                    "source_phone": word.source_phone,
                    "carrier_role": word.carrier_role,
                    "carrier_surface": word.carrier_surface,
                    "carrier_phone": word.carrier_phone,
                    "target_occurrence_count": word.target_occurrence_count,
                    "candidate_attempt": word.candidate_attempt,
                }
                for word in self.words
            ],
        }


class NativePortugueseKokoroG2P:
    """Language-scoped adapter around pinned KPipeline(lang_code='p')."""

    language_id = LANGUAGE_ID
    kokoro_lang_code = KOKORO_LANG_CODE

    def __init__(self, voice_spec: VoiceSpec) -> None:
        _validate_voice_spec(voice_spec)
        from kokoro import KPipeline

        self.voice_id = voice_spec.voice_id
        self.pipeline = KPipeline(
            lang_code=voice_spec.kokoro_lang_code,
            repo_id=MODEL_REPO,
            model=False,
        )

    def phonemize_phrase(self, text: str) -> str:
        phone, tokens = self.pipeline.g2p(text)
        if tokens is not None:
            raise PortugueseCarrierPlannerError(
                "native_g2p_contract_drift",
                "Pinned Portuguese Kokoro G2P unexpectedly returned token objects.",
            )
        return unicodedata.normalize("NFC", phone).strip()

    def phonemize_words(self, words: Sequence[str]) -> list[str]:
        phones: list[str] = []
        for word in words:
            phone = self.phonemize_phrase(word)
            if not phone:
                raise PortugueseCarrierPlannerError(
                    "unpronounceable_word",
                    "Pinned Portuguese Kokoro G2P returned an empty word plan.",
                )
            phones.append(phone)
        return phones


class PortuguesePositiveOnlyIndexV1:
    """Fail-closed adapter for the frozen partial native index result."""

    scope = NATIVE_INDEX_SCOPE

    def __init__(self, database=None) -> None:
        database = database or Paths().portuguese_kokoro_gate_db
        self.index = PortugueseKokoroGateIndex(database)
        with sqlite3.connect(f"file:{database}?mode=ro&immutable=1", uri=True) as conn:
            row = conn.execute(
                "SELECT value FROM metadata WHERE key = 'status'"
            ).fetchone()
        try:
            status = json.loads(row[0]) if row is not None else None
        except (TypeError, json.JSONDecodeError) as exc:
            raise PortugueseCarrierPlannerError(
                "native_index_status_invalid",
                "Portuguese native-v1 index status metadata is invalid.",
            ) from exc
        if status != self.scope:
            raise PortugueseCarrierPlannerError(
                "native_index_scope_mismatch",
                "The planner requires the frozen partial-positive-only native index.",
            )

    def phone_match(self, phone: str) -> bool:
        return self.index.phone_match(phone)


def _validate_voice_spec(voice_spec: VoiceSpec) -> None:
    language = LANGUAGE_SPECS[LANGUAGE_ID]
    if (
        voice_spec.language_id != LANGUAGE_ID
        or voice_spec.kokoro_lang_code != KOKORO_LANG_CODE
        or voice_spec.voice_id not in language.voice_ids
    ):
        raise PortugueseCarrierPlannerError(
            "voice_language_mismatch",
            "An explicit pinned pt-BR Kokoro VoiceSpec is required.",
        )
    if language.renderer_candidate_enabled or PORTUGUESE_RENDERER_CANDIDATE_ENABLED:
        raise PortugueseCarrierPlannerError(
            "candidate_flag_drift",
            "The Portuguese renderer candidate must be disabled.",
        )


def _normalize_input(text: str) -> str:
    normalized = unicodedata.normalize("NFC", re.sub(r"\s+", " ", text.strip()))
    if not normalized:
        raise PortugueseCarrierPlannerError("empty_input", "Enter Portuguese text.")
    if len(normalized) > 280:
        raise PortugueseCarrierPlannerError("input_too_long", "Input is too long.")
    if any(
        not (character.isalpha() or character in _BASIC_PUNCTUATION)
        for character in normalized
    ):
        raise PortugueseCarrierPlannerError(
            "unsupported_characters",
            "Use Portuguese words, apostrophes, spaces, and basic punctuation only.",
        )
    words = PORTUGUESE_WORD_RE.findall(normalized)
    if not 2 <= len(words) <= 40:
        raise PortugueseCarrierPlannerError(
            "unsupported_word_count", "Enter between 2 and 40 words."
        )
    if (len(re.findall(r"[.!?]+", normalized)) or 1) > 2:
        raise PortugueseCarrierPlannerError(
            "unsupported_sentence_count", "Enter at most two sentences."
        )
    if any(len(word) > 1 and word.isupper() for word in words):
        raise PortugueseCarrierPlannerError(
            "unsupported_acronym", "Acronyms and all-caps words are unsupported."
        )
    return normalized


def _validate_target_phone(target_phone: str, model_vocab: frozenset[str]) -> str:
    normalized = unicodedata.normalize("NFC", target_phone).strip()
    if (
        not normalized
        or any(
            character.isspace() or character in _PHONE_PUNCTUATION
            for character in normalized
        )
        or all(character in _STRUCTURAL_PHONE_SYMBOLS for character in normalized)
    ):
        raise PortugueseCarrierPlannerError(
            "invalid_target_phone", "Target must be one non-structural phone sequence."
        )
    _validate_model_phone(normalized, model_vocab, "target")
    return normalized


def _validate_model_phone(
    phone: str, model_vocab: frozenset[str], description: str
) -> None:
    unsupported = sorted(set(phone) - model_vocab)
    if unsupported:
        raise PortugueseCarrierPlannerError(
            "unrepresentable_phone_plan",
            f"{description} contains unsupported Kokoro symbols: {''.join(unsupported)}",
        )


def _vowel_group_count(phone: str) -> int:
    count = 0
    in_vowel_group = False
    for character in phone:
        is_vowel = character in PORTUGUESE_KOKORO_VOWEL_SYMBOLS or character == "̃"
        if is_vowel and not in_vowel_group and character != "̃":
            count += 1
        in_vowel_group = is_vowel
    return max(1, count)


def _pick(values: Sequence[str], digest: bytes, cursor: int) -> str:
    return values[digest[cursor % len(digest)] % len(values)]


def _candidate_surface(key: PortugueseMappingKey, attempt: int) -> str:
    digest = hashlib.sha256(
        stable_json(
            {
                "planner_version": PLANNER_VERSION,
                "language_id": LANGUAGE_ID,
                "key": asdict(key),
                "attempt": attempt,
            }
        ).encode("utf-8")
    ).digest()
    if key.carrier_role == "weak":
        # Preserve a bounded monosyllabic search first, then use the frozen
        # disyllabic fallback. The union written-word gate contains virtually
        # every short pt/en/es-shaped monosyllable, so a one-syllable-only
        # inventory cannot honestly satisfy the mandatory written gate.
        syllable_count = 1 if attempt < 32 else 2
        onsets = _WEAK_ONSETS
        nuclei = _WEAK_NUCLEI
        codas = _WEAK_CODAS
    else:
        syllable_count = max(2, min(4, _vowel_group_count(key.source_phone)))
        onsets = _CONTENT_ONSETS
        nuclei = _CONTENT_NUCLEI
        codas = _CONTENT_CODAS

    chunks: list[str] = []
    cursor = 0
    for syllable_index in range(syllable_count):
        chunks.append(_pick(onsets, digest, cursor))
        cursor += 1
        chunks.append(_pick(nuclei, digest, cursor))
        cursor += 1
        if syllable_index == syllable_count - 1:
            chunks.append(_pick(codas, digest, cursor))
        cursor += 1
    return "".join(chunks)


class PortugueseCarrierPlannerV1:
    """Research-only native pt-BR real-text and opaque-carrier planner.

    The target report is computed from isolated native Kokoro word predictions.
    It does not create a listener-lens substitution. Candidate clearance always
    requires the existing written-word and pt eSpeak checks. The frozen native
    v1 index can only add a positive rejection; a negative result is never used
    as clearance evidence.
    """

    def __init__(
        self,
        *,
        voice_spec: VoiceSpec,
        g2p: PortugueseG2P,
        model_vocab: frozenset[str] | set[str],
        mandatory_gate: MandatoryPortugueseGate,
        native_positive_index: SupplementalPositiveIndex,
    ) -> None:
        _validate_voice_spec(voice_spec)
        if (
            g2p.language_id != LANGUAGE_ID
            or g2p.kokoro_lang_code != KOKORO_LANG_CODE
            or g2p.voice_id != voice_spec.voice_id
        ):
            raise PortugueseCarrierPlannerError(
                "g2p_language_mismatch",
                "Portuguese G2P must be scoped to the explicit pt-BR VoiceSpec.",
            )
        if not mandatory_gate.enabled:
            raise PortugueseCarrierPlannerError(
                "mandatory_gate_disabled",
                "The written-word and pt eSpeak gate must be enabled.",
            )
        if native_positive_index.scope != NATIVE_INDEX_SCOPE:
            raise PortugueseCarrierPlannerError(
                "native_index_scope_mismatch",
                "Only the frozen partial-positive-only Portuguese native index is allowed.",
            )
        self.voice_spec = voice_spec
        self.g2p = g2p
        self.model_vocab = frozenset(model_vocab)
        self.mandatory_gate = mandatory_gate
        self.native_positive_index = native_positive_index
        self._g2p_lock = threading.RLock()

    @classmethod
    def load(cls, *, voice_id: str) -> PortugueseCarrierPlannerV1:
        try:
            voice_spec = VOICE_SPECS_BY_ID[voice_id]
        except KeyError as exc:
            raise PortugueseCarrierPlannerError(
                "unknown_voice", f"Unknown pinned Kokoro voice: {voice_id}"
            ) from exc
        _validate_voice_spec(voice_spec)
        config = json.loads(
            resolve_pinned_file(CONFIG_FILE).read_text(encoding="utf-8")
        )
        return cls(
            voice_spec=voice_spec,
            g2p=NativePortugueseKokoroG2P(voice_spec),
            model_vocab=frozenset(config["vocab"]),
            mandatory_gate=DatabaseNonceChecker(),
            native_positive_index=PortuguesePositiveOnlyIndexV1(),
        )

    def _isolated_reasons(self, assignment: PortugueseCarrierAssignment) -> set[str]:
        reasons: set[str] = set()
        decision = self.mandatory_gate.check(assignment.surface, WORD_LANGUAGE, None)
        if not decision.accepted:
            reasons.add(
                "mandatory_written_espeak_" + (decision.rejection_reason or "rejected")
            )
        if self.native_positive_index.phone_match(assignment.phone):
            reasons.add("native_v1_positive_predicted_homophone")
        if assignment.surface.casefold().endswith(_PRODUCTIVE_ENDINGS):
            reasons.add("productive_morphology")
        try:
            _validate_model_phone(assignment.phone, self.model_vocab, "carrier word")
        except PortugueseCarrierPlannerError:
            reasons.add("unrepresentable_carrier_phone")
        return reasons

    def _resolve_assignments(
        self,
        keys: Sequence[PortugueseMappingKey],
        rejection_counts: Counter[str],
    ) -> tuple[dict[PortugueseMappingKey, PortugueseCarrierAssignment], int, int]:
        unique_keys = tuple(dict.fromkeys(keys))
        assignments: dict[PortugueseMappingKey, PortugueseCarrierAssignment] = {}
        next_attempt = {key: 0 for key in unique_keys}
        implicated = set(unique_keys)
        candidate_attempts = 0
        adjacency_checks = 0

        for _ in range(MAX_RESOLUTION_ROUNDS):
            for key in unique_keys:
                if key not in implicated:
                    continue
                accepted: PortugueseCarrierAssignment | None = None
                first_attempt = next_attempt[key]
                for attempt in range(
                    first_attempt, first_attempt + MAX_CANDIDATE_ATTEMPTS
                ):
                    candidate_attempts += 1
                    surface = _candidate_surface(key, attempt)
                    phone = self.g2p.phonemize_words((surface,))[0]
                    candidate = PortugueseCarrierAssignment(surface, phone, attempt)
                    reasons = self._isolated_reasons(candidate)
                    if not reasons:
                        accepted = candidate
                        break
                    rejection_counts.update(reasons)
                if accepted is None:
                    raise PortugueseCarrierPlannerError(
                        "candidate_search_exhausted",
                        "No gate-clean Portuguese carrier exists within the bounded search.",
                    )
                assignments[key] = accepted

            conflicts: set[PortugueseMappingKey] = set()
            for previous_key, current_key in zip(keys, keys[1:]):
                adjacency_checks += 1
                previous = assignments[previous_key]
                current = assignments[current_key]
                decision = self.mandatory_gate.check(
                    current.surface, WORD_LANGUAGE, previous.surface
                )
                if not decision.accepted:
                    rejection_counts[
                        "mandatory_written_espeak_"
                        + (decision.rejection_reason or "adjacency_rejected")
                    ] += 1
                    conflicts.update((previous_key, current_key))
                if self.native_positive_index.phone_match(
                    previous.phone + current.phone
                ):
                    rejection_counts[
                        "native_v1_positive_adjacency_predicted_homophone"
                    ] += 1
                    conflicts.update((previous_key, current_key))
            if not conflicts:
                return assignments, candidate_attempts, adjacency_checks
            implicated = conflicts
            for key in conflicts:
                next_attempt[key] = assignments[key].candidate_attempt + 1

        raise PortugueseCarrierPlannerError(
            "adjacency_search_exhausted",
            "No globally gate-clean Portuguese carrier exists within the bounded search.",
        )

    def plan(self, text: str, *, target_phone: str) -> PortugueseCarrierPlan:
        normalized = _normalize_input(text)
        target = _validate_target_phone(target_phone, self.model_vocab)
        source_words = PORTUGUESE_WORD_RE.findall(normalized)
        rejection_counts: Counter[str] = Counter()

        # Pinned Misaki's Portuguese path has no token objects and its eSpeak
        # backend is not documented as thread-safe. Keep phrase, word, and
        # generated-carrier queries in one serialized planning transaction.
        with self._g2p_lock:
            source_phonemes = self.g2p.phonemize_phrase(normalized)
            source_word_phones = self.g2p.phonemize_words(source_words)
            if len(source_word_phones) != len(source_words):
                raise PortugueseCarrierPlannerError(
                    "source_word_alignment_drift",
                    "Portuguese native G2P lost isolated-word alignment.",
                )
            if not source_phonemes:
                raise PortugueseCarrierPlannerError(
                    "g2p_empty", "Portuguese native G2P returned no phrase plan."
                )
            _validate_model_phone(source_phonemes, self.model_vocab, "source phrase")

            keys = [
                PortugueseMappingKey(
                    source_casefold=word.casefold(),
                    source_phone=phone,
                    carrier_role=(
                        "weak"
                        if word.casefold() in PORTUGUESE_WEAK_WORDS
                        else "content"
                    ),
                )
                for word, phone in zip(source_words, source_word_phones, strict=True)
            ]
            assignments, candidate_attempts, adjacency_checks = (
                self._resolve_assignments(keys, rejection_counts)
            )

            replacement_values = iter(assignments[key].surface for key in keys)
            carrier_script = PORTUGUESE_WORD_RE.sub(
                lambda _: next(replacement_values), normalized
            )
            if PORTUGUESE_WORD_RE.sub("", carrier_script) != PORTUGUESE_WORD_RE.sub(
                "", normalized
            ):
                raise PortugueseCarrierPlannerError(
                    "punctuation_drift", "Opaque carrier changed source punctuation."
                )
            carrier_phonemes = self.g2p.phonemize_phrase(carrier_script)
            if not carrier_phonemes:
                raise PortugueseCarrierPlannerError(
                    "carrier_g2p_empty",
                    "Portuguese native G2P returned no carrier plan.",
                )
            _validate_model_phone(carrier_phonemes, self.model_vocab, "carrier phrase")

        if max(len(source_phonemes), len(carrier_phonemes)) > MAX_PHONEME_CHARACTERS:
            raise PortugueseCarrierPlannerError(
                "phone_plan_too_long", "Portuguese phone plan exceeds Kokoro's limit."
            )

        words: list[PortugueseCarrierWord] = []
        target_indexes: list[int] = []
        target_occurrences = 0
        for index, (source, source_phone, key) in enumerate(
            zip(source_words, source_word_phones, keys, strict=True)
        ):
            occurrence_count = source_phone.count(target)
            if occurrence_count:
                target_indexes.append(index)
                target_occurrences += occurrence_count
            assignment = assignments[key]
            words.append(
                PortugueseCarrierWord(
                    word_index=index,
                    source=source,
                    source_phone=source_phone,
                    carrier_role=key.carrier_role,
                    carrier_surface=assignment.surface,
                    carrier_phone=assignment.phone,
                    target_occurrence_count=occurrence_count,
                    candidate_attempt=assignment.candidate_attempt,
                )
            )

        gate_receipt = PortugueseGateReceipt(
            mandatory_written_espeak_gate_pass=True,
            mandatory_gate_language=WORD_LANGUAGE,
            mandatory_espeak_voice=ESPEAK_VOICE,
            native_index_scope=self.native_positive_index.scope,
            native_positive_only_gate_pass=True,
            native_negative_used_for_clearance=False,
            isolated_candidates_checked=len(set(keys)),
            adjacency_pairs_checked=adjacency_checks,
            candidate_attempts=candidate_attempts,
            candidate_rejection_counts=dict(sorted(rejection_counts.items())),
            exact_native_phrase_plan=True,
            model_representable=True,
        )
        payload = {
            "planner_version": PLANNER_VERSION,
            "language_id": LANGUAGE_ID,
            "kokoro_lang_code": KOKORO_LANG_CODE,
            "voice_id": self.voice_spec.voice_id,
            "candidate_enabled": PORTUGUESE_RENDERER_CANDIDATE_ENABLED,
            "production_route_available": PRODUCTION_ROUTE_AVAILABLE,
            "normalized_text": normalized,
            "source_phonemes": source_phonemes,
            "carrier_script": carrier_script,
            "carrier_phonemes": carrier_phonemes,
            "target_phone": target,
            "target_analysis_scope": TARGET_ANALYSIS_SCOPE,
            "target_word_indexes": target_indexes,
            "target_occurrence_count": target_occurrences,
            "words": [asdict(word) for word in words],
            "gate_receipt": asdict(gate_receipt),
        }
        plan_sha256 = hashlib.sha256(stable_json(payload).encode("utf-8")).hexdigest()
        return PortugueseCarrierPlan(
            planner_version=PLANNER_VERSION,
            language_id=LANGUAGE_ID,
            kokoro_lang_code=KOKORO_LANG_CODE,
            voice_id=self.voice_spec.voice_id,
            candidate_enabled=PORTUGUESE_RENDERER_CANDIDATE_ENABLED,
            production_route_available=PRODUCTION_ROUTE_AVAILABLE,
            normalized_text=normalized,
            source_phonemes=source_phonemes,
            carrier_script=carrier_script,
            carrier_phonemes=carrier_phonemes,
            target_phone=target,
            target_analysis_scope=TARGET_ANALYSIS_SCOPE,
            target_word_indexes=tuple(target_indexes),
            target_word_count=len(target_indexes),
            target_occurrence_count=target_occurrences,
            target_available=bool(target_occurrences),
            words=tuple(words),
            gate_receipt=gate_receipt,
            plan_sha256=plan_sha256,
        )


def plan_portuguese_smoke_fixture_v1(
    planner: PortugueseCarrierPlannerV1,
) -> PortugueseCarrierPlan:
    return planner.plan(
        PORTUGUESE_SMOKE_TEXT_V1,
        target_phone=PORTUGUESE_SMOKE_TARGET_PHONE_V1,
    )


def portuguese_smoke_screening_receipt_v1(
    planner: PortugueseCarrierPlannerV1,
) -> dict[str, Any]:
    plan = plan_portuguese_smoke_fixture_v1(planner)
    return {
        "schema_version": 1,
        "fixture_id": PORTUGUESE_SMOKE_FIXTURE_ID_V1,
        "fixture_text": PORTUGUESE_SMOKE_TEXT_V1,
        "target_phone": PORTUGUESE_SMOKE_TARGET_PHONE_V1,
        **plan.screening_receipt(),
    }
