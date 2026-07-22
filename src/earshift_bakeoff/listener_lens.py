from __future__ import annotations

import copy
import hashlib
import json
import re
import shutil
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

from .config import ROOT, Paths, load_json_yaml, stable_json
from .gates import FOREIGN_SWITCH_RE, CandidateGate, EspeakPhonemizer, canonical_ipa
from .util import atomic_write_json


LENS_RULES_PATH = ROOT / "rules" / "listener_lenses.yaml"
TRANSFORM_ALGORITHM_VERSION = 5
MAX_PAIR_GENERATION_ATTEMPTS = 256
MAX_GLOBAL_RESOLUTION_ROUNDS = 64
WORD_RE = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)?")
ALLOWED_INPUT_RE = re.compile(r"^[A-Za-z'’.,!?;:\s]+$")
IPA_VOWEL_RE = re.compile(r"[aeiouyæɑɒɔəɚɜɝɐɛɞɪɵøœʉʊʌ]+")


class ListenerLensError(ValueError):
    """A user-facing typed-listener-lens validation error."""


class WordAnalyzer(Protocol):
    def phonemize_words(self, words: Sequence[str], voice: str) -> list[str]: ...


class NonceChecker(Protocol):
    @property
    def enabled(self) -> bool: ...

    def accepts(
        self, surface: str, language: str, previous_surface: str | None
    ) -> tuple[bool, str]: ...


@dataclass(frozen=True)
class NonceDecision:
    accepted: bool
    predicted_ipa: str
    rejection_reason: str | None


class EspeakWordAnalyzer:
    def __init__(self, phonemizer: EspeakPhonemizer | None = None) -> None:
        self.phonemizer = phonemizer or EspeakPhonemizer()

    def phonemize_words(self, words: Sequence[str], voice: str) -> list[str]:
        return [
            _clean_ipa(value)
            for value in self.phonemizer.phonemize(
                [word.casefold() for word in words], voice
            )
        ]


class DatabaseNonceChecker:
    """Adapts the existing hash-only dictionary/G2P gate to short nonce tokens."""

    def __init__(self, gate: CandidateGate | None = None) -> None:
        self.gate = gate or CandidateGate()

    @property
    def enabled(self) -> bool:
        return True

    def accepts(
        self, surface: str, language: str, previous_surface: str | None
    ) -> tuple[bool, str]:
        decision = self.check(surface, language, previous_surface)
        return decision.accepted, decision.predicted_ipa

    def check(
        self, surface: str, language: str, previous_surface: str | None
    ) -> NonceDecision:
        if self.gate.text_match(surface):
            return NonceDecision(False, "", "written_word")
        if previous_surface and self.gate.text_match(previous_surface + surface):
            return NonceDecision(False, "", "adjacency_written_word")
        voice = self.gate.voices[language]
        ipa = _clean_ipa(self.gate.phonemizer.phonemize([surface], voice)[0])
        if FOREIGN_SWITCH_RE.search(ipa):
            return NonceDecision(False, ipa, "foreign_language_switch")
        if self.gate.phone_match(language, ipa):
            return NonceDecision(False, ipa, "predicted_homophone")
        if previous_surface:
            previous_ipa = _clean_ipa(
                self.gate.phonemizer.phonemize([previous_surface], voice)[0]
            )
            if self.gate.phone_match(language, previous_ipa + ipa):
                return NonceDecision(False, ipa, "adjacency_predicted_homophone")
        return NonceDecision(True, ipa, None)


class UncheckedNonceChecker:
    """Development-only fallback. Its output must never be shipped as verified."""

    @property
    def enabled(self) -> bool:
        return False

    def accepts(
        self, surface: str, language: str, previous_surface: str | None
    ) -> tuple[bool, str]:
        return True, ""

    def check(
        self, surface: str, language: str, previous_surface: str | None
    ) -> NonceDecision:
        return NonceDecision(True, "", None)


@dataclass(frozen=True)
class AppliedRule:
    rule_id: str
    source: str
    target: str
    occurrences: int
    confidence: str
    description: str
    source_ids: list[str]


@dataclass(frozen=True)
class RuleSlot:
    word_index: int
    neutral_character_span: tuple[int, int]
    lens_character_span: tuple[int, int]
    rule_id: str
    source_ipa: str
    target_ipa: str
    neutral_grapheme: str
    lens_grapheme: str


@dataclass(frozen=True)
class CarrierWord:
    source: str
    source_ipa: str
    listener_ipa: str
    carrier_role: str
    neutral_surface: str
    lens_surface: str
    syllables: int
    applied_rule_ids: list[str]
    slots: list[RuleSlot]
    pair_generation_attempt: int


@dataclass(frozen=True)
class CarrierMappingKey:
    source_casefold: str
    pronunciation_signature: str
    rule_signature: tuple[str, ...]
    carrier_role: str


@dataclass(frozen=True)
class WeakFormAttempt:
    mapping_id: str
    candidate_index: int
    candidate: str
    predicted_ipa: str
    stage: str
    outcome: str
    rejection_reason: str | None


@dataclass(frozen=True)
class WeakFormReport:
    policy_version: int
    eligible_word_count: int
    eligible_mapping_count: int
    selected_mapping_count: int
    candidate_attempt_count: int
    candidate_gate_yield: float | None
    rejected_attempt_count: int
    rejection_reason_counts: dict[str, int]
    attempts: list[WeakFormAttempt]


@dataclass(frozen=True)
class _CarrierAssignment:
    neutral_surface: str
    lens_surface: str
    slots: tuple[RuleSlot, ...]
    pair_generation_attempt: int
    weak_candidate_index: int | None = None
    weak_predicted_ipa: str = ""
    weak_attempts: tuple[WeakFormAttempt, ...] = ()


@dataclass(frozen=True)
class _CarrierWordPlan:
    word_index: int
    source_word: str
    source_ipa: str
    listener_ipa: str
    vowel_units: tuple[str, ...]
    applied_rule_ids: tuple[str, ...]
    carrier_role: str
    mapping_key: CarrierMappingKey


@dataclass(frozen=True)
class LensResult:
    schema_version: int
    cache_key: str
    profile_id: str
    profile_label: str
    claim_label: str
    original_text: str
    neutral_script: str
    lens_script: str
    comparison_available: bool
    words: list[CarrierWord]
    weak_form_report: WeakFormReport
    slots: list[RuleSlot]
    applied_rules: list[AppliedRule]
    warnings: list[str]
    sources: list[dict[str, str]]
    renderer_status: str
    api_calls_made: int
    nonce_gate_enabled: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _clean_ipa(value: str) -> str:
    return canonical_ipa(value).replace("\u200d", "").replace("͡", "")


def _nonce_decision(
    checker: NonceChecker,
    surface: str,
    language: str,
    previous_surface: str | None,
) -> NonceDecision:
    check = getattr(checker, "check", None)
    if callable(check):
        return check(surface, language, previous_surface)
    accepted, predicted_ipa = checker.accepts(surface, language, previous_surface)
    reason = (
        None
        if accepted
        else (
            "adjacency_nonce_gate_rejected"
            if previous_surface is not None
            else "nonce_gate_rejected"
        )
    )
    return NonceDecision(accepted, predicted_ipa, reason)


def _vowel_units(ipa: str) -> list[str]:
    value = ipa.replace("ˈ", "").replace("ˌ", "").replace("ː", "")
    return IPA_VOWEL_RE.findall(value) or ["ə"]


def _listener_ipa(ipa: str, rules_by_source: dict[str, dict[str, Any]]) -> str:
    def replace(match: re.Match[str]) -> str:
        unit = match.group(0)
        rule = rules_by_source.get(unit)
        return rule["target"] if rule else unit

    return IPA_VOWEL_RE.sub(replace, ipa)


def _enabled_rules_by_source(profile: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        rule["source"]: rule
        for rule in profile["transformations"]
        if rule.get("enabled", True)
    }


def _replace_words(text: str, replacements: Sequence[str]) -> str:
    iterator = iter(replacements)
    return WORD_RE.sub(lambda _: next(iterator), text)


def _stable_pick(values: Sequence[str], digest: bytes, cursor: int) -> str:
    return values[digest[cursor % len(digest)] % len(values)]


def _validate_carrier_word(
    neutral_surface: str, lens_surface: str, slots: Sequence[RuleSlot]
) -> None:
    neutral_cursor = 0
    lens_cursor = 0
    neutral_shell: list[str] = []
    lens_shell: list[str] = []
    for slot in slots:
        neutral_start, neutral_end = slot.neutral_character_span
        lens_start, lens_end = slot.lens_character_span
        if (
            not 0
            <= neutral_cursor
            <= neutral_start
            < neutral_end
            <= len(neutral_surface)
        ):
            raise ListenerLensError(
                f"Neutral carrier slot {slot.rule_id} is out of bounds or overlaps."
            )
        if not 0 <= lens_cursor <= lens_start < lens_end <= len(lens_surface):
            raise ListenerLensError(
                f"Listener carrier slot {slot.rule_id} is out of bounds or overlaps."
            )
        if neutral_surface[neutral_start:neutral_end] != slot.neutral_grapheme:
            raise ListenerLensError(f"Neutral carrier slot {slot.rule_id} drifted.")
        if lens_surface[lens_start:lens_end] != slot.lens_grapheme:
            raise ListenerLensError(f"Listener carrier slot {slot.rule_id} drifted.")
        neutral_shell.append(neutral_surface[neutral_cursor:neutral_start])
        lens_shell.append(lens_surface[lens_cursor:lens_start])
        neutral_cursor = neutral_end
        lens_cursor = lens_end
    neutral_shell.append(neutral_surface[neutral_cursor:])
    lens_shell.append(lens_surface[lens_cursor:])
    if neutral_shell != lens_shell:
        raise ListenerLensError("Carrier words differ outside a declared vowel slot.")


class ListenerLensEngine:
    def __init__(
        self,
        *,
        rules_path: Path = LENS_RULES_PATH,
        analyzer: WordAnalyzer | None = None,
        nonce_checker: NonceChecker | None = None,
    ) -> None:
        self.rules_path = rules_path
        self.rules = load_json_yaml(rules_path)
        self.analyzer = analyzer or EspeakWordAnalyzer()
        if nonce_checker is not None:
            self.nonce_checker = nonce_checker
        elif Paths().gate_db.is_file():
            self.nonce_checker = DatabaseNonceChecker()
        else:
            self.nonce_checker = UncheckedNonceChecker()

    @property
    def policy(self) -> dict[str, Any]:
        return copy.deepcopy(self.rules["runtime_policy"])

    def _profile(self, profile_id: str) -> dict[str, Any]:
        for profile in self.rules["profiles"]:
            if profile["id"] == profile_id:
                return profile
        raise ListenerLensError(f"Unsupported listener profile: {profile_id}")

    def normalize_and_validate(self, text: str) -> tuple[str, list[str]]:
        normalized = re.sub(r"\s+", " ", text.strip())
        policy = self.rules["runtime_policy"]
        if not normalized:
            raise ListenerLensError("Enter one or two short English sentences.")
        if len(normalized) > policy["max_characters"]:
            raise ListenerLensError(
                f"Input must be {policy['max_characters']} characters or fewer."
            )
        if not ALLOWED_INPUT_RE.fullmatch(normalized):
            raise ListenerLensError(
                "For this prototype, use plain English words and basic punctuation; "
                "numbers, symbols, URLs, and hyphenated forms are not supported."
            )
        words = WORD_RE.findall(normalized)
        if not 2 <= len(words) <= policy["max_words"]:
            raise ListenerLensError(f"Enter between 2 and {policy['max_words']} words.")
        sentence_count = len(re.findall(r"[.!?]+", normalized)) or 1
        if sentence_count > policy["max_sentences"]:
            raise ListenerLensError(
                f"Enter at most {policy['max_sentences']} sentences."
            )
        if any(len(word) > 1 and word.isupper() for word in words):
            raise ListenerLensError(
                "Acronyms and all-caps tokens are not supported in the prototype."
            )

        warnings: list[str] = []
        for match in WORD_RE.finditer(normalized):
            word = match.group(0)
            prefix = normalized[: match.start()].rstrip()
            if prefix and word[0].isupper() and prefix[-1] not in ".!?":
                warnings.append("proper_names_may_be_unreliable")
                break
        return normalized, warnings

    def cache_key_for(self, text: str, profile_id: str) -> str:
        normalized, _ = self.normalize_and_validate(text)
        self._profile(profile_id)
        rules_hash = hashlib.sha256(self.rules_path.read_bytes()).hexdigest()
        payload = {
            "schema_version": self.rules["schema_version"],
            "algorithm_version": TRANSFORM_ALGORITHM_VERSION,
            "rules_sha256": rules_hash,
            "profile_id": profile_id,
            "text": normalized.casefold(),
        }
        return hashlib.sha256(stable_json(payload).encode("utf-8")).hexdigest()

    def _carrier_pair(
        self,
        *,
        cache_key: str,
        plan: _CarrierWordPlan,
        profile: dict[str, Any],
        first_attempt: int,
    ) -> _CarrierAssignment:
        if plan.carrier_role == "weak":
            return self._weak_carrier_pair(
                cache_key=cache_key,
                plan=plan,
                profile=profile,
                first_attempt=first_attempt,
            )
        inventory = profile["carrier_inventory"]
        rules_by_source = _enabled_rules_by_source(profile)
        source_language = profile["source_language"].split("-", 1)[0]
        mapping_payload = stable_json(asdict(plan.mapping_key))
        for attempt in range(
            first_attempt, first_attempt + MAX_PAIR_GENERATION_ATTEMPTS
        ):
            digest = hashlib.sha256(
                f"{cache_key}\0{mapping_payload}\0{attempt}".encode()
            ).digest()
            neutral_chunks: list[str] = []
            lens_chunks: list[str] = []
            slots: list[RuleSlot] = []
            neutral_cursor = 0
            lens_cursor = 0
            for syllable_index, vowel_unit in enumerate(plan.vowel_units):
                cursor = syllable_index * 3
                rule = rules_by_source.get(vowel_unit)
                if rule and rule.get("carrier_shells"):
                    shells = rule["carrier_shells"]
                    shell = shells[digest[cursor % len(digest)] % len(shells)]
                    onset = shell["onset"]
                    coda = shell["coda"]
                else:
                    onset = _stable_pick(inventory["onsets"], digest, cursor)
                    coda = _stable_pick(inventory["codas"], digest, cursor + 2)
                neutral_nucleus = (
                    rule["neutral_grapheme"]
                    if rule
                    else _stable_pick(inventory["nuclei"], digest, cursor + 1)
                )
                lens_nucleus = rule["lens_grapheme"] if rule else neutral_nucleus
                neutral_start = neutral_cursor + len(onset)
                lens_start = lens_cursor + len(onset)
                neutral_chunk = onset + neutral_nucleus + coda
                lens_chunk = onset + lens_nucleus + coda
                if rule:
                    slots.append(
                        RuleSlot(
                            word_index=-1,
                            neutral_character_span=(
                                neutral_start,
                                neutral_start + len(neutral_nucleus),
                            ),
                            lens_character_span=(
                                lens_start,
                                lens_start + len(lens_nucleus),
                            ),
                            rule_id=rule["id"],
                            source_ipa=rule["source"],
                            target_ipa=rule["target"],
                            neutral_grapheme=neutral_nucleus,
                            lens_grapheme=lens_nucleus,
                        )
                    )
                neutral_chunks.append(neutral_chunk)
                lens_chunks.append(lens_chunk)
                neutral_cursor += len(neutral_chunk)
                lens_cursor += len(lens_chunk)
            neutral_surface = "".join(neutral_chunks)
            lens_surface = "".join(lens_chunks)
            _validate_carrier_word(neutral_surface, lens_surface, slots)
            neutral_accepted, _ = self.nonce_checker.accepts(
                neutral_surface, source_language, None
            )
            lens_accepted, _ = self.nonce_checker.accepts(
                lens_surface, source_language, None
            )
            if neutral_accepted and lens_accepted:
                return _CarrierAssignment(
                    neutral_surface=neutral_surface,
                    lens_surface=lens_surface,
                    slots=tuple(slots),
                    pair_generation_attempt=attempt,
                )
        raise ListenerLensError(
            "Could not create a paired carrier token whose two versions passed the local gates."
        )

    def _weak_carrier_pair(
        self,
        *,
        cache_key: str,
        plan: _CarrierWordPlan,
        profile: dict[str, Any],
        first_attempt: int,
    ) -> _CarrierAssignment:
        policy = self.rules["weak_carrier_policy"]
        syllables = len(plan.vowel_units)
        inventory = [
            item
            for item in policy["candidate_inventory"]
            if item["syllables"] == syllables
        ]
        if not inventory:
            raise ListenerLensError(
                f"No bounded weak carrier inventory exists for {syllables} syllables."
            )
        mapping_payload = stable_json(asdict(plan.mapping_key))
        mapping_id = hashlib.sha256(mapping_payload.encode("utf-8")).hexdigest()[:16]
        start_digest = hashlib.sha256(
            f"{cache_key}\0{mapping_payload}\0weak-policy-{policy['version']}".encode()
        ).digest()
        start = int.from_bytes(start_digest[:4], "big") % len(inventory)
        source_language = profile["source_language"].split("-", 1)[0]
        attempts: list[WeakFormAttempt] = []
        for offset in range(first_attempt, len(inventory)):
            candidate_index = (start + offset) % len(inventory)
            candidate = inventory[candidate_index]["surface"]
            decision = _nonce_decision(
                self.nonce_checker, candidate, source_language, None
            )
            predicted_syllables = (
                len(_vowel_units(decision.predicted_ipa))
                if decision.predicted_ipa
                else syllables
            )
            reason = decision.rejection_reason
            if decision.accepted and predicted_syllables != syllables:
                reason = "predicted_syllable_mismatch"
            accepted = decision.accepted and reason is None
            attempts.append(
                WeakFormAttempt(
                    mapping_id=mapping_id,
                    candidate_index=candidate_index,
                    candidate=candidate,
                    predicted_ipa=decision.predicted_ipa,
                    stage="isolated",
                    outcome="accepted" if accepted else "rejected",
                    rejection_reason=reason,
                )
            )
            if accepted:
                return _CarrierAssignment(
                    neutral_surface=candidate,
                    lens_surface=candidate,
                    slots=(),
                    pair_generation_attempt=offset,
                    weak_candidate_index=candidate_index,
                    weak_predicted_ipa=decision.predicted_ipa,
                    weak_attempts=tuple(attempts),
                )
        raise ListenerLensError(
            "Could not create a gate-clean weak carrier token from the bounded inventory."
        )

    def _resolve_carrier_mappings(
        self,
        *,
        cache_key: str,
        word_plans: Sequence[_CarrierWordPlan],
        profile: dict[str, Any],
    ) -> tuple[
        dict[CarrierMappingKey, _CarrierAssignment],
        list[WeakFormAttempt],
    ]:
        unique_plans: dict[CarrierMappingKey, _CarrierWordPlan] = {}
        for plan in word_plans:
            unique_plans.setdefault(plan.mapping_key, plan)

        source_language = profile["source_language"].split("-", 1)[0]
        assignments: dict[CarrierMappingKey, _CarrierAssignment] = {}
        next_attempt = {key: 0 for key in unique_plans}
        implicated = set(unique_plans)
        weak_attempts: list[WeakFormAttempt] = []

        for _resolution_round in range(MAX_GLOBAL_RESOLUTION_ROUNDS):
            for key, plan in unique_plans.items():
                if key not in implicated:
                    continue
                try:
                    assignments[key] = self._carrier_pair(
                        cache_key=cache_key,
                        plan=plan,
                        profile=profile,
                        first_attempt=next_attempt[key],
                    )
                except ListenerLensError as exc:
                    raise ListenerLensError(
                        "Could not satisfy every gate within the bounded carrier "
                        "inventories while preserving repeated source-word mappings; "
                        "the comparison is unavailable for this input."
                    ) from exc
                weak_attempts.extend(assignments[key].weak_attempts)

            conflicts: set[CarrierMappingKey] = set()
            for previous, current in zip(word_plans, word_plans[1:]):
                previous_assignment = assignments[previous.mapping_key]
                current_assignment = assignments[current.mapping_key]
                surface_pairs = {
                    (
                        previous_assignment.neutral_surface,
                        current_assignment.neutral_surface,
                    ),
                    (
                        previous_assignment.lens_surface,
                        current_assignment.lens_surface,
                    ),
                }
                for previous_surface, current_surface in surface_pairs:
                    decision = _nonce_decision(
                        self.nonce_checker,
                        current_surface,
                        source_language,
                        previous_surface,
                    )
                    if not decision.accepted:
                        conflicts.update((previous.mapping_key, current.mapping_key))
                        for plan, assignment in (
                            (previous, previous_assignment),
                            (current, current_assignment),
                        ):
                            if plan.carrier_role != "weak":
                                continue
                            mapping_payload = stable_json(asdict(plan.mapping_key))
                            weak_attempts.append(
                                WeakFormAttempt(
                                    mapping_id=hashlib.sha256(
                                        mapping_payload.encode("utf-8")
                                    ).hexdigest()[:16],
                                    candidate_index=int(
                                        assignment.weak_candidate_index or 0
                                    ),
                                    candidate=assignment.neutral_surface,
                                    predicted_ipa=assignment.weak_predicted_ipa,
                                    stage="adjacency",
                                    outcome="rejected",
                                    rejection_reason=(
                                        decision.rejection_reason
                                        or "adjacency_nonce_gate_rejected"
                                    ),
                                )
                            )
            if not conflicts:
                return assignments, weak_attempts

            implicated = conflicts
            for key in implicated:
                next_attempt[key] = assignments[key].pair_generation_attempt + 1

        raise ListenerLensError(
            "Could not satisfy every adjacent-token gate while preserving repeated "
            "source-word mappings; the comparison is unavailable for this input."
        )

    def transform(self, text: str, profile_id: str) -> LensResult:
        normalized, warnings = self.normalize_and_validate(text)
        profile = self._profile(profile_id)
        cache_key = self.cache_key_for(normalized, profile_id)
        source_words = WORD_RE.findall(normalized)
        source_ipas = self.analyzer.phonemize_words(
            source_words, profile["source_espeak_voice"]
        )
        if len(source_words) != len(source_ipas):
            raise ListenerLensError("The local G2P analyzer lost word alignment.")

        carrier_words: list[CarrierWord] = []
        aggregate: dict[str, AppliedRule] = {}
        all_slots: list[RuleSlot] = []
        rules_by_source = _enabled_rules_by_source(profile)
        disabled_rule_sources = {
            rule["source"]
            for rule in profile["transformations"]
            if not rule.get("enabled", True)
        }
        disabled_rule_encountered = False
        word_plans: list[_CarrierWordPlan] = []
        weak_source_words = {
            word.casefold()
            for word in self.rules["weak_carrier_policy"]["source_function_words"]
        }

        for index, (source_word, source_ipa) in enumerate(
            zip(source_words, source_ipas)
        ):
            vowel_units = tuple(_vowel_units(source_ipa))
            applied_ids: list[str] = []
            rule_signature: list[str] = []
            for vowel_unit in vowel_units:
                rule = rules_by_source.get(vowel_unit)
                if not rule:
                    disabled_rule_encountered = bool(
                        disabled_rule_encountered or vowel_unit in disabled_rule_sources
                    )
                    rule_signature.append("-")
                    continue
                rule_signature.append(rule["id"])
                if rule["id"] not in applied_ids:
                    applied_ids.append(rule["id"])
                previous = aggregate.get(rule["id"])
                total = 1 + (previous.occurrences if previous else 0)
                aggregate[rule["id"]] = AppliedRule(
                    rule_id=rule["id"],
                    source=rule["source"],
                    target=rule["target"],
                    occurrences=total,
                    confidence=rule["confidence"],
                    description=rule["description"],
                    source_ids=list(rule["source_ids"]),
                )
            carrier_role = (
                "weak"
                if source_word.casefold() in weak_source_words and not applied_ids
                else "content"
            )
            word_plans.append(
                _CarrierWordPlan(
                    word_index=index,
                    source_word=source_word,
                    source_ipa=source_ipa,
                    listener_ipa=_listener_ipa(source_ipa, rules_by_source),
                    vowel_units=vowel_units,
                    applied_rule_ids=tuple(applied_ids),
                    carrier_role=carrier_role,
                    mapping_key=CarrierMappingKey(
                        source_casefold=source_word.casefold(),
                        pronunciation_signature=source_ipa,
                        rule_signature=tuple(rule_signature),
                        carrier_role=carrier_role,
                    ),
                )
            )

        assignments, weak_attempts = self._resolve_carrier_mappings(
            cache_key=cache_key,
            word_plans=word_plans,
            profile=profile,
        )
        weak_plans = [plan for plan in word_plans if plan.carrier_role == "weak"]
        weak_mapping_keys = {plan.mapping_key for plan in weak_plans}
        candidate_attempts = [
            attempt for attempt in weak_attempts if attempt.stage == "isolated"
        ]
        rejected_attempts = [
            attempt for attempt in weak_attempts if attempt.outcome == "rejected"
        ]
        rejection_counts = Counter(
            attempt.rejection_reason
            for attempt in rejected_attempts
            if attempt.rejection_reason
        )
        selected_mapping_count = sum(
            assignments[key].weak_candidate_index is not None
            for key in weak_mapping_keys
        )
        weak_form_report = WeakFormReport(
            policy_version=int(self.rules["weak_carrier_policy"]["version"]),
            eligible_word_count=len(weak_plans),
            eligible_mapping_count=len(weak_mapping_keys),
            selected_mapping_count=selected_mapping_count,
            candidate_attempt_count=len(candidate_attempts),
            candidate_gate_yield=(
                selected_mapping_count / len(candidate_attempts)
                if candidate_attempts
                else None
            ),
            rejected_attempt_count=len(rejected_attempts),
            rejection_reason_counts=dict(sorted(rejection_counts.items())),
            attempts=weak_attempts,
        )
        for plan in word_plans:
            assignment = assignments[plan.mapping_key]
            slots = [
                RuleSlot(
                    word_index=plan.word_index,
                    neutral_character_span=slot.neutral_character_span,
                    lens_character_span=slot.lens_character_span,
                    rule_id=slot.rule_id,
                    source_ipa=slot.source_ipa,
                    target_ipa=slot.target_ipa,
                    neutral_grapheme=slot.neutral_grapheme,
                    lens_grapheme=slot.lens_grapheme,
                )
                for slot in assignment.slots
            ]
            all_slots.extend(slots)
            carrier_words.append(
                CarrierWord(
                    source=plan.source_word,
                    source_ipa=plan.source_ipa,
                    listener_ipa=plan.listener_ipa,
                    carrier_role=plan.carrier_role,
                    neutral_surface=assignment.neutral_surface,
                    lens_surface=assignment.lens_surface,
                    syllables=len(plan.vowel_units),
                    applied_rule_ids=list(plan.applied_rule_ids),
                    slots=slots,
                    pair_generation_attempt=assignment.pair_generation_attempt,
                )
            )

        if not self.nonce_checker.enabled:
            warnings.append("nonce_dictionary_and_homophone_gate_unavailable")
        if disabled_rule_encountered:
            warnings.append("some_listener_rules_excluded_after_calibration")
        if not all_slots:
            warnings.append("no_supported_listener_rule")
        warnings.extend(profile["scope_exclusions"])
        neutral_script = _replace_words(
            normalized, [word.neutral_surface for word in carrier_words]
        )
        lens_script = _replace_words(
            normalized, [word.lens_surface for word in carrier_words]
        )
        if len(WORD_RE.findall(neutral_script)) != len(source_words):
            raise ListenerLensError("The neutral carrier lost word alignment.")
        if len(WORD_RE.findall(lens_script)) != len(source_words):
            raise ListenerLensError("The listener carrier lost word alignment.")
        punctuation_skeleton = WORD_RE.sub("", normalized)
        if WORD_RE.sub("", neutral_script) != punctuation_skeleton:
            raise ListenerLensError("The neutral carrier changed source punctuation.")
        if WORD_RE.sub("", lens_script) != punctuation_skeleton:
            raise ListenerLensError("The listener carrier changed source punctuation.")
        if len(neutral_script) > self.rules["runtime_policy"]["max_carrier_characters"]:
            raise ListenerLensError(
                "The generated carrier exceeds the runtime length limit."
            )
        return LensResult(
            schema_version=self.rules["schema_version"],
            cache_key=cache_key,
            profile_id=profile_id,
            profile_label=profile["display_name"],
            claim_label=profile["claim_label"],
            original_text=normalized,
            neutral_script=neutral_script,
            lens_script=lens_script,
            comparison_available=bool(all_slots),
            words=carrier_words,
            weak_form_report=weak_form_report,
            slots=all_slots,
            applied_rules=list(aggregate.values()),
            warnings=warnings,
            sources=list(profile["sources"]),
            renderer_status=self.rules["runtime_policy"]["renderer_status"],
            api_calls_made=0,
            nonce_gate_enabled=self.nonce_checker.enabled,
        )


class ListenerLensCache:
    def __init__(self, root: Path | None = None) -> None:
        self.root = root or (Paths().cache / "listener-lens")

    def get(self, cache_key: str) -> dict[str, Any] | None:
        path = self.root / f"{cache_key}.json"
        if not path.is_file():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def put(self, cache_key: str, payload: dict[str, Any]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        atomic_write_json(self.root / f"{cache_key}.json", payload)


class ListenerLensService:
    """Cache-first, zero-API service boundary used by the local browser prototype."""

    def __init__(
        self,
        engine: ListenerLensEngine | None = None,
        cache: ListenerLensCache | None = None,
    ) -> None:
        self.engine = engine or ListenerLensEngine()
        self.cache = cache or ListenerLensCache()

    def status(self) -> dict[str, Any]:
        return {
            "profile_ids": [profile["id"] for profile in self.engine.rules["profiles"]],
            "runtime_policy": self.engine.policy,
            "nonce_gate_enabled": self.engine.nonce_checker.enabled,
            "transform_api_calls": 0,
            "audio_renderer": "browser_speech_mock",
            "production_renderer": self.engine.policy["renderer_model"],
            "production_renderer_contract": self.engine.policy["renderer_contract"],
        }

    def transform(self, text: str, profile_id: str) -> dict[str, Any]:
        cache_key = self.engine.cache_key_for(text, profile_id)
        if cached := self.cache.get(cache_key):
            payload = copy.deepcopy(cached)
            payload["runtime"] = {"cache_hit": True, "api_calls_made": 0}
            return payload
        payload = self.engine.transform(text, profile_id).to_dict()
        payload["runtime"] = {"cache_hit": False, "api_calls_made": 0}
        self.cache.put(cache_key, payload)
        return payload


def local_prerequisites() -> dict[str, bool]:
    return {
        "espeak_ng": shutil.which("espeak-ng") is not None,
        "gate_database": Paths().gate_db.is_file(),
    }
