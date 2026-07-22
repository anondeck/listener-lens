from __future__ import annotations

import importlib.metadata
import json
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

from .config import Paths, sha256_json, stable_json
from .gates import CandidateGate
from .kokoro_specs import (
    LANGUAGE_SPECS,
    VOICE_SPECS_BY_ID,
    resolve_pinned_file,
)
from .kokoro_gate_bridge import KokoroGateIndex
from .kokoro_synthesis import (
    CONFIG_FILE,
    MAX_PHONEME_CHARACTERS,
    MODEL_FILE,
    MODEL_HASHES,
    MODEL_REPO,
    MODEL_REVISION,
    RNG_SEED,
    SAMPLE_RATE_HZ,
    SPEED,
)
from .portuguese_carrier_planner_v1 import (
    LANGUAGE_ID,
    NATIVE_INDEX_SCOPE,
    PORTUGUESE_SMOKE_FIXTURE_ID_V1,
    PORTUGUESE_SMOKE_TARGET_PHONE_V1,
    PORTUGUESE_SMOKE_TEXT_V1,
    PortugueseCarrierPlan,
    PortugueseCarrierPlannerV1,
    PortuguesePositiveOnlyIndexV1,
    plan_portuguese_smoke_fixture_v1,
)
from .util import atomic_write_json, sha256_file


RUN_ID = "20260717-ptbr-to-ae-listener-lens-v1"
SCHEMA_VERSION = 1
RULE_ID = "bp-open-mid-back-to-ce-low-back"
NEUTRAL_PHONE = "ɔ"
LENS_PHONE = "ɑ"
SECONDARY_UNRENDERED_RULE_ID = "bp-low-central-to-ce-near-front-low"
RESPONSE_FILENAME = "ptbr-to-ae-listener-lens-v1-response.json"
RESPONSE_SCHEMA_PATH = (
    Paths().root / "docs" / "research" / "track-d-listener-lens-response-schema-v1.json"
)
REQUIRED_REVIEW_FIELDS = ("naturalness", "artifact", "meaning")
PROTOCOL_REVIEW_STATUS = "awaiting_independent_recheck"
EXPECTED_BASE_PLAN_SHA256 = (
    "93623fcef3f6854eb897f8f43e4445c72ba2cfaf2b6f5c0ae3d3ccf52b4a2d0d"
)
TECHNICAL_PROBE_VOICE_ID = min(LANGUAGE_SPECS[LANGUAGE_ID].voice_ids)
MAX_DECODER_CALLS = 5
CEILINGS_HZ = (5500, 5750, 6000)
MIDDLE_FRACTION = 0.50
MIN_VALID_FRAMES = 5
MIN_VALID_FRAME_FRACTION = 0.60
MIN_DIRECTION_COSINE = 0.50
MIN_MAGNITUDE_BARK = 0.25
MIN_ANCHOR_DISTANCE_BARK = 0.25
ANCHOR_DISTANCE_MULTIPLIER = 0.50
LOCALIZATION_PADDING_S = 0.150
LOCALIZATION_MINIMUM = 0.80
MAX_CLIPPED_FRACTION = 0.001

RESEARCH_ROOT = (
    Paths().artifacts / "research" / "20260717-ptbr-to-ame-listener-evidence-v1"
)
EVIDENCE_PATH = RESEARCH_ROOT / "evidence.json"
PROFILE_PATH = RESEARCH_ROOT / "research-profile.json"
PT_GATE_ROOT = (
    Paths().artifacts / "portuguese" / "20260717-pt-kokoro-homophone-index-v1"
)
ENGLISH_KOKORO_GATE_ROOT = (
    Paths().artifacts / "typed-engine" / "20260716-kokoro-gate-bridge-feasibility-v1"
)
PRAAT = Path("/Applications/Praat.app/Contents/MacOS/Praat")
MEASUREMENT_SCRIPT = Paths().root / "scripts" / "praat_sentence_pair_v2_burg.praat"

EXPECTED_EVIDENCE_SHA256 = (
    "3802b8c296f28d23289ad3931684e119ab309a660ff6030d040546a3921d7def"
)
EXPECTED_PROFILE_SHA256 = (
    "c871cc95d9b30cec9818f1796972a4005219398bfd41d95ee59183a8445a680e"
)
EXPECTED_PT_GATE_PROTOCOL_SHA256 = (
    "ae874f9a2b58814963f23c1fd78601e4b35d41db092e59984a1e2efc62d33867"
)
EXPECTED_PT_GATE_RECEIPT_SHA256 = (
    "d24c48a8dff31631e3d204312506177f6298ce333a86a33863832f0fd059f991"
)
EXPECTED_EN_GATE_PROTOCOL_SHA256 = (
    "69607e8edf4daff0e40895e040762aef3f1d26fd6f1400ab715fdb649ea57058"
)
EXPECTED_EN_GATE_RECEIPT_SHA256 = (
    "300dd7224fc665d4be455db3c8e2b8c3776c65c71054af2d368c6da08d8519bc"
)
VOICE_SCREEN_SUMMARY_PATH = (
    Paths().artifacts
    / "voice-screen"
    / "20260717-kokoro-bilingual-voice-screen-v1"
    / "summary.json"
)
EXPECTED_VOICE_SCREEN_SUMMARY_SHA256 = (
    "0b198eb28afc534c059ce051c306312e2605765d4cd7554c4f126548f241def1"
)

REVIEW_SNAPSHOT_PATH = (
    Paths().root
    / "docs"
    / "research"
    / "track-d-reciprocal-feasibility-reviewed-snapshot-v1.json"
)
SELECTION_REPORT_PATH = (
    Paths().root
    / "docs"
    / "research"
    / "track-d-selection-claims-independent-report-v1.json"
)
ACOUSTIC_REPORT_PATH = (
    Paths().root
    / "docs"
    / "research"
    / "track-d-acoustic-instrument-independent-report-v1.json"
)
RESOLUTION_PATH = (
    Paths().root
    / "docs"
    / "research"
    / "track-d-reciprocal-feasibility-resolution-v1.json"
)
ORIGINAL_SELECTION_RECHECK_PATH = (
    Paths().root
    / "docs"
    / "research"
    / "track-d-selection-claims-independent-recheck-v1.json"
)
ORIGINAL_ACOUSTIC_RECHECK_PATH = (
    Paths().root
    / "docs"
    / "research"
    / "track-d-acoustic-instrument-independent-recheck-v1.json"
)
SELECTION_RECHECK_ATTEMPT_PATH = (
    Paths().root
    / "docs"
    / "research"
    / "track-d-selection-claims-recheck-attempt-v1.json"
)
ACOUSTIC_RECHECK_ATTEMPT_PATH = (
    Paths().root
    / "docs"
    / "research"
    / "track-d-acoustic-instrument-recheck-attempt-v1.json"
)
AUTHORIZATION_GATE_RESOLUTION_PATH = (
    Paths().root
    / "docs"
    / "research"
    / "track-d-reciprocal-feasibility-authorization-gate-resolution-v1.json"
)
RECHECK_V2_SCHEMA_PATH = (
    Paths().root / "docs" / "research" / "track-d-independent-recheck-v2.schema.json"
)
SELECTION_RECHECK_V2_PATH = (
    Paths().root
    / "docs"
    / "research"
    / "track-d-selection-claims-independent-recheck-v2.json"
)
ACOUSTIC_RECHECK_V2_PATH = (
    Paths().root
    / "docs"
    / "research"
    / "track-d-acoustic-instrument-independent-recheck-v2.json"
)
FINAL_APPROVAL_RESOLUTION_PATH = (
    Paths().root
    / "docs"
    / "research"
    / "track-d-reciprocal-feasibility-final-approval-resolution-v1.json"
)
EXPECTED_REVIEW_SNAPSHOT_SHA256 = (
    "3a0ea58b5eace3cbe45bc6fc427d59cfe89ef09ad1f8788e546a39e793feab36"
)
EXPECTED_SELECTION_REPORT_SHA256 = (
    "3e00e042896b1cf72801e4649586b08cbda2b4d27cc46deae20717bf30303bef"
)
EXPECTED_ACOUSTIC_REPORT_SHA256 = (
    "10a60ffbc4f5e7c65a63070e9364ae3918f8e2b627cd40ff75a237cc4f4b1861"
)
# Filled only by the resolution pass. Independent recheck files intentionally do
# not exist yet and cannot be manufactured by this implementation author.
EXPECTED_RESOLUTION_SHA256 = (
    "db9192c629f2f5c3dcadf6cda85270b63369eed0eafd9e2141c54f7a12c19e45"
)
EXPECTED_SELECTION_RECHECK_ATTEMPT_SHA256 = (
    "dc1740f8fe9b8cf3f0cc425277832040a65b2f3b972397f925b2a262f4c2cad5"
)
EXPECTED_ACOUSTIC_RECHECK_ATTEMPT_SHA256 = (
    "e8e40903dd78dbbc4ff497393492897d9c38546a4d07af6877418d8b4af2ac60"
)
EXPECTED_AUTHORIZATION_GATE_RESOLUTION_SHA256 = (
    "e710d1700b892a498c3792c3ba00a593223b574335f3befa028cd21a38bc53b4"
)
EXPECTED_RECHECK_V2_SCHEMA_SHA256 = (
    "84a2b0d2628188fcd5816cfbc9d598ed00292f4bf9d5be78b06abfcbb37f88ce"
)
SELECTION_RECHECK_V2_REVIEWER_ID = "/root/track_d_selection_recheck_v2"
ACOUSTIC_RECHECK_V2_REVIEWER_ID = "/root/track_d_acoustic_recheck_v2"
EXPECTED_REVIEWED_PROTOCOL_SHA256 = (
    "493bc8116979f20a67693b3ca53d89725350dace63d77e171c4b51ddd323deb2"
)
EXPECTED_REVIEWED_SEMANTIC_SHA256 = (
    "74b413eee7e3ab08be41ae78b48b2f110d0abe2e72f0525950330ce449048a5f"
)

# This is an audited symbol-level intersection only. Sequences containing any
# other symbol are never queried against the English eSpeak prediction index.
ENGLISH_ESPEAK_EXACT_SHARED_SYMBOLS = frozenset("bdoplsˈˌɔɑæ")
EXPECTED_D1_LIMITS = (
    "The direct monolingual sample was only nine listeners and all were women "
    "from one US region.",
    "Stimuli were isolated vowels extracted from one nonce-word shell, not "
    "running speech or meaning-opaque Portuguese carriers.",
    "The task forced one of ten labels even when a listener was unsure and did "
    "not collect goodness ratings.",
    "The article shows that listener and source dialect affect categorization; "
    "it cannot justify a universal American-English mapping.",
)
EXPECTED_D1_POPULATION = (
    "Nine female Central Valley Californian-English monolingual university "
    "students, aged 20-27; the closest direct evidence for the requested "
    "American-English-listener direction."
)
EXPECTED_D1_METHOD = (
    "Seventy isolated BP vowel tokens, ten speakers by seven stressed vowels, "
    "were extracted from /fVfe/ nonce words. Listeners made a required choice "
    "among ten Californian-English vowel categories represented by keywords."
)

_PHONE_WORD_BOUNDARIES = frozenset(' ;:,.!?—…()[]{}"“”')
_ORAL_VOWELS = frozenset("AIWYaeiouyæɑɔəɛɐɪʊ")
_STRESS_MARKERS = frozenset(("ˈ", "ˌ"))


class ReciprocalFeasibilityProtocolError(RuntimeError):
    pass


@dataclass(frozen=True)
class AuthorizationApprovalPaths:
    selection_recheck: Path
    acoustic_recheck: Path
    final_approval_resolution: Path


def default_authorization_approval_paths() -> AuthorizationApprovalPaths:
    return AuthorizationApprovalPaths(
        selection_recheck=SELECTION_RECHECK_V2_PATH,
        acoustic_recheck=ACOUSTIC_RECHECK_V2_PATH,
        final_approval_resolution=FINAL_APPROVAL_RESOLUTION_PATH,
    )


class PhoneCollisionIndex(Protocol):
    def phone_match(self, phone: str) -> bool: ...


@dataclass(frozen=True)
class ProfileTargetOccurrence:
    occurrence_index: int
    source_word_index: int
    source_phone_offset: int
    profile_character_index: int
    model_column: int
    stress_model_column: int
    neutral_phone: str = NEUTRAL_PHONE
    lens_phone: str = LENS_PHONE


@dataclass(frozen=True)
class EnglishCollisionSequenceReceipt:
    side: str
    sequence_kind: str
    sequence_index: int
    word_indexes: tuple[int, ...]
    espeak_exact_comparison_compatible: bool
    espeak_positive_match: bool | None
    espeak_incompatibility_reason: str | None
    kokoro_exact_comparison_compatible: bool
    kokoro_positive_match: bool | None
    kokoro_incompatibility_reason: str | None


@dataclass(frozen=True)
class SupplementalEnglishCollisionReceipt:
    scope: str
    clearance_role: str
    exact_positive_action: str
    negative_used_for_portuguese_clearance: bool
    sequence_count: int
    espeak_compatible_sequence_count: int
    espeak_incompatible_sequence_count: int
    espeak_positive_match_count: int
    kokoro_positive_match_count: int
    kokoro_incompatible_sequence_count: int
    no_exact_positive_among_compatible_comparisons: bool
    sequences: tuple[EnglishCollisionSequenceReceipt, ...]


@dataclass(frozen=True)
class DerivedPhoneGateReceipt:
    mandatory_portuguese_written_espeak_exact_phone_gate_pass: bool
    portuguese_native_positive_only_screen_no_positive: bool
    portuguese_native_index_scope: str
    portuguese_native_negative_used_for_clearance: bool
    sides_screened: tuple[str, ...]
    portuguese_isolated_phone_plans_checked: int
    portuguese_adjacency_phone_plans_checked: int
    supplemental_english_listener_collision: SupplementalEnglishCollisionReceipt


@dataclass(frozen=True)
class ReciprocalProfilePhonePlan:
    schema_version: int
    fixture_id: str
    rule_id: str
    base_plan_sha256: str
    voice_id: str
    carrier_script: str
    source_alignment_phonemes: str
    neutral_phonemes: str
    lens_phonemes: str
    target_word_indexes: tuple[int, ...]
    target_occurrences: tuple[ProfileTargetOccurrence, ...]
    sanitized_incidental_character_indexes: tuple[int, ...]
    equal_model_token_count: int
    base_gate_receipt: dict[str, Any]
    derived_phone_gate_receipt: DerivedPhoneGateReceipt
    candidate_enabled: bool
    production_route_available: bool
    plan_sha256: str

    def pair_record(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "target_occurrences": [
                asdict(occurrence) for occurrence in self.target_occurrences
            ],
            "derived_phone_gate_receipt": asdict(self.derived_phone_gate_receipt),
        }


def run_dir() -> Path:
    return Paths().artifacts / "research" / RUN_ID


def _word_character_spans(phonemes: str) -> tuple[tuple[int, ...], ...]:
    spans: list[tuple[int, ...]] = []
    current: list[int] = []
    for index, symbol in enumerate(phonemes):
        if symbol in _PHONE_WORD_BOUNDARIES:
            if current:
                spans.append(tuple(current))
                current = []
            continue
        current.append(index)
    if current:
        spans.append(tuple(current))
    return tuple(spans)


def _model_token_count(phonemes: str, model_vocab: frozenset[str]) -> int:
    unsupported = sorted(set(phonemes) - model_vocab)
    if unsupported:
        raise ReciprocalFeasibilityProtocolError(
            "profile plan contains unsupported Kokoro symbols: " + "".join(unsupported)
        )
    return len(phonemes)


def _split_phone_words(phonemes: str) -> tuple[str, ...]:
    return tuple(
        "".join(phonemes[index] for index in span)
        for span in _word_character_spans(phonemes)
    )


def _phone_sequences(
    words: Sequence[str],
) -> tuple[tuple[str, int, tuple[int, ...], str], ...]:
    isolated = tuple(
        ("isolated", index, (index,), phone) for index, phone in enumerate(words)
    )
    adjacent = tuple(
        ("adjacency", index, (index, index + 1), left + right)
        for index, (left, right) in enumerate(zip(words, words[1:], strict=False))
    )
    return isolated + adjacent


def _screen_derived_phone_plans(
    *,
    neutral_phonemes: str,
    lens_phonemes: str,
    mandatory_gate: CandidateGate,
    native_index: PhoneCollisionIndex,
    english_kokoro_index: PhoneCollisionIndex,
    english_kokoro_compatible_symbols: frozenset[str],
    english_espeak_compatible_symbols: frozenset[str],
) -> DerivedPhoneGateReceipt:
    isolated_checks = 0
    adjacency_checks = 0
    english_rows: list[EnglishCollisionSequenceReceipt] = []
    for side, phonemes in (
        ("neutral", neutral_phonemes),
        ("lens", lens_phonemes),
    ):
        words = _split_phone_words(phonemes)
        for sequence_kind, sequence_index, word_indexes, phone in _phone_sequences(
            words
        ):
            if sequence_kind == "isolated":
                isolated_checks += 1
            else:
                adjacency_checks += 1
            if mandatory_gate.phone_match("pt", phone):
                raise ReciprocalFeasibilityProtocolError(
                    f"{side} {sequence_kind} profile phone plan matches the "
                    "mandatory Portuguese written/eSpeak index"
                )
            if native_index.phone_match(phone):
                raise ReciprocalFeasibilityProtocolError(
                    f"{side} {sequence_kind} profile phone plan matches the "
                    "Portuguese native positive-only index"
                )

            espeak_unsupported = sorted(set(phone) - english_espeak_compatible_symbols)
            english_espeak_compatible = not espeak_unsupported
            english_espeak_positive = (
                mandatory_gate.phone_match("en", phone)
                if english_espeak_compatible
                else None
            )
            kokoro_unsupported = sorted(set(phone) - english_kokoro_compatible_symbols)
            english_kokoro_compatible = not kokoro_unsupported
            english_kokoro_positive = (
                english_kokoro_index.phone_match(phone)
                if english_kokoro_compatible
                else None
            )
            row = EnglishCollisionSequenceReceipt(
                side=side,
                sequence_kind=sequence_kind,
                sequence_index=sequence_index,
                word_indexes=word_indexes,
                espeak_exact_comparison_compatible=english_espeak_compatible,
                espeak_positive_match=english_espeak_positive,
                espeak_incompatibility_reason=(
                    None
                    if english_espeak_compatible
                    else "symbols_outside_audited_english_espeak_exact_domain:"
                    + "".join(espeak_unsupported)
                ),
                kokoro_exact_comparison_compatible=english_kokoro_compatible,
                kokoro_positive_match=english_kokoro_positive,
                kokoro_incompatibility_reason=(
                    None
                    if english_kokoro_compatible
                    else "symbols_outside_pinned_english_kokoro_comparison_domain:"
                    + "".join(kokoro_unsupported)
                ),
            )
            english_rows.append(row)
            if english_espeak_positive or english_kokoro_positive is True:
                raise ReciprocalFeasibilityProtocolError(
                    f"{side} {sequence_kind} profile phone plan has an exact "
                    "supplemental English-listener collision"
                )

    english_receipt = SupplementalEnglishCollisionReceipt(
        scope=(
            "positive-only exact English-listener collision evidence beside, not "
            "part of, mandatory Portuguese opacity clearance"
        ),
        clearance_role="supplemental_rejection_only_never_clearance",
        exact_positive_action="reject_profile_plan_before_freeze",
        negative_used_for_portuguese_clearance=False,
        sequence_count=len(english_rows),
        espeak_compatible_sequence_count=sum(
            row.espeak_exact_comparison_compatible for row in english_rows
        ),
        espeak_incompatible_sequence_count=sum(
            not row.espeak_exact_comparison_compatible for row in english_rows
        ),
        espeak_positive_match_count=sum(
            row.espeak_positive_match is True for row in english_rows
        ),
        kokoro_positive_match_count=sum(
            row.kokoro_positive_match is True for row in english_rows
        ),
        kokoro_incompatible_sequence_count=sum(
            not row.kokoro_exact_comparison_compatible for row in english_rows
        ),
        no_exact_positive_among_compatible_comparisons=not any(
            row.espeak_positive_match is True or row.kokoro_positive_match is True
            for row in english_rows
        ),
        sequences=tuple(english_rows),
    )
    return DerivedPhoneGateReceipt(
        mandatory_portuguese_written_espeak_exact_phone_gate_pass=True,
        portuguese_native_positive_only_screen_no_positive=True,
        portuguese_native_index_scope=NATIVE_INDEX_SCOPE,
        portuguese_native_negative_used_for_clearance=False,
        sides_screened=("neutral", "lens"),
        portuguese_isolated_phone_plans_checked=isolated_checks,
        portuguese_adjacency_phone_plans_checked=adjacency_checks,
        supplemental_english_listener_collision=english_receipt,
    )


def build_reciprocal_profile_phone_plan(
    base_plan: PortugueseCarrierPlan,
    *,
    model_vocab: frozenset[str] | set[str],
    mandatory_gate: CandidateGate,
    native_index: PhoneCollisionIndex,
    english_kokoro_index: PhoneCollisionIndex,
    english_kokoro_compatible_symbols: frozenset[str] | set[str],
    english_espeak_compatible_symbols: frozenset[str] | set[str] = (
        ENGLISH_ESPEAK_EXACT_SHARED_SYMBOLS
    ),
) -> ReciprocalProfilePhonePlan:
    """Build the bounded research-only /ɔ/->/ɑ/ layer above planner v1.

    Planner v1 owns the stable Portuguese fixture and the opaque written/eSpeak
    gate. This layer never changes its surface. It removes incidental /ɔ,ɑ/
    symbols from the phrase phone plan, then assigns one stressed carrier nucleus
    to each stressed source /ɔ/. Both derived phone plans are screened again.
    """

    if base_plan.plan_sha256 != EXPECTED_BASE_PLAN_SHA256:
        raise ReciprocalFeasibilityProtocolError("stable Portuguese fixture drifted")
    if (
        base_plan.voice_id != TECHNICAL_PROBE_VOICE_ID
        or base_plan.target_phone != NEUTRAL_PHONE
        or base_plan.candidate_enabled
        or base_plan.production_route_available
    ):
        raise ReciprocalFeasibilityProtocolError(
            "base plan must use the disabled predetermined Portuguese probe"
        )
    gate = base_plan.gate_receipt
    if not (
        gate.mandatory_written_espeak_gate_pass
        and gate.native_positive_only_gate_pass
        and gate.exact_native_phrase_plan
        and gate.model_representable
        and not gate.native_negative_used_for_clearance
    ):
        raise ReciprocalFeasibilityProtocolError("base Portuguese gate did not pass")

    base = list(base_plan.carrier_phonemes)
    spans = _word_character_spans(base_plan.carrier_phonemes)
    if len(spans) != len(base_plan.words):
        raise ReciprocalFeasibilityProtocolError(
            "carrier phone words do not align with the stable planner words"
        )

    incidental = tuple(
        index
        for index, symbol in enumerate(base)
        if symbol in {NEUTRAL_PHONE, LENS_PHONE}
    )
    for index in incidental:
        base[index] = "o" if base[index] == NEUTRAL_PHONE else "a"

    occurrences: list[ProfileTargetOccurrence] = []
    used_profile_indexes: set[int] = set()
    for word_index in base_plan.target_word_indexes:
        source_phone = base_plan.words[word_index].source_phone
        source_offsets = tuple(
            index
            for index, symbol in enumerate(source_phone)
            if symbol == NEUTRAL_PHONE
        )
        if not source_offsets or any(
            offset == 0 or source_phone[offset - 1] not in _STRESS_MARKERS
            for offset in source_offsets
        ):
            raise ReciprocalFeasibilityProtocolError(
                "profile scope permits stressed source /ɔ/ occurrences only"
            )
        if len(source_offsets) != base_plan.words[word_index].target_occurrence_count:
            raise ReciprocalFeasibilityProtocolError(
                "source target coverage drifted from the Portuguese planner"
            )
        span = spans[word_index]
        stressed_nuclei = tuple(
            index
            for index in span
            if base[index] in _ORAL_VOWELS
            and index > span[0]
            and base[index - 1] in _STRESS_MARKERS
            and (index + 1 >= len(base) or base[index + 1] != "̃")
        )
        if len(stressed_nuclei) < len(source_offsets):
            raise ReciprocalFeasibilityProtocolError(
                "target carrier word lacks enough stressed oral nuclei"
            )
        for source_offset, profile_index in zip(
            source_offsets, stressed_nuclei, strict=False
        ):
            if profile_index in used_profile_indexes:
                raise ReciprocalFeasibilityProtocolError(
                    "profile target assignment reused a carrier nucleus"
                )
            used_profile_indexes.add(profile_index)
            occurrences.append(
                ProfileTargetOccurrence(
                    occurrence_index=len(occurrences),
                    source_word_index=word_index,
                    source_phone_offset=source_offset,
                    profile_character_index=profile_index,
                    model_column=profile_index + 1,
                    stress_model_column=profile_index,
                )
            )

    if len(occurrences) != base_plan.target_occurrence_count or not occurrences:
        raise ReciprocalFeasibilityProtocolError(
            "profile target coverage must be exact and nonempty"
        )

    neutral = base.copy()
    lens = base.copy()
    for occurrence in occurrences:
        neutral[occurrence.profile_character_index] = NEUTRAL_PHONE
        lens[occurrence.profile_character_index] = LENS_PHONE
    neutral_phonemes = "".join(neutral)
    lens_phonemes = "".join(lens)
    # The common duration/alignment source is the sanitized, pre-intervention
    # opaque carrier. Neutral F0/noise is computed separately below that common
    # alignment, keeping the two controls explicit rather than calling the
    # neutral phone plan the source.
    source_alignment_phonemes = "".join(base)

    changed = tuple(
        index
        for index, (neutral_symbol, lens_symbol) in enumerate(
            zip(neutral_phonemes, lens_phonemes, strict=True)
        )
        if neutral_symbol != lens_symbol
    )
    target_indexes = tuple(
        occurrence.profile_character_index for occurrence in occurrences
    )
    if changed != target_indexes:
        raise ReciprocalFeasibilityProtocolError(
            "neutral/lens differences must equal the declared targets"
        )
    if (
        neutral_phonemes.count(NEUTRAL_PHONE) != len(occurrences)
        or neutral_phonemes.count(LENS_PHONE) != 0
        or lens_phonemes.count(LENS_PHONE) != len(occurrences)
        or lens_phonemes.count(NEUTRAL_PHONE) != 0
    ):
        raise ReciprocalFeasibilityProtocolError(
            "profile plans contain an undeclared /ɔ/ or /ɑ/"
        )

    vocab = frozenset(model_vocab)
    counts = {
        _model_token_count(plan, vocab)
        for plan in (source_alignment_phonemes, neutral_phonemes, lens_phonemes)
    }
    if len(counts) != 1:
        raise ReciprocalFeasibilityProtocolError(
            "profile plans must have equal model-token counts"
        )
    token_count = next(iter(counts))
    if not 1 <= token_count <= MAX_PHONEME_CHARACTERS:
        raise ReciprocalFeasibilityProtocolError("profile plan is outside model limits")

    derived_receipt = _screen_derived_phone_plans(
        neutral_phonemes=neutral_phonemes,
        lens_phonemes=lens_phonemes,
        mandatory_gate=mandatory_gate,
        native_index=native_index,
        english_kokoro_index=english_kokoro_index,
        english_kokoro_compatible_symbols=frozenset(english_kokoro_compatible_symbols),
        english_espeak_compatible_symbols=frozenset(english_espeak_compatible_symbols),
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "fixture_id": PORTUGUESE_SMOKE_FIXTURE_ID_V1,
        "rule_id": RULE_ID,
        "base_plan_sha256": base_plan.plan_sha256,
        "voice_id": base_plan.voice_id,
        "carrier_script": base_plan.carrier_script,
        "source_alignment_phonemes": source_alignment_phonemes,
        "neutral_phonemes": neutral_phonemes,
        "lens_phonemes": lens_phonemes,
        "target_word_indexes": list(base_plan.target_word_indexes),
        "target_occurrences": [asdict(item) for item in occurrences],
        "sanitized_incidental_character_indexes": list(incidental),
        "equal_model_token_count": token_count,
        "base_gate_receipt": asdict(base_plan.gate_receipt),
        "derived_phone_gate_receipt": asdict(derived_receipt),
        "candidate_enabled": False,
        "production_route_available": False,
    }
    return ReciprocalProfilePhonePlan(
        schema_version=SCHEMA_VERSION,
        fixture_id=PORTUGUESE_SMOKE_FIXTURE_ID_V1,
        rule_id=RULE_ID,
        base_plan_sha256=base_plan.plan_sha256,
        voice_id=base_plan.voice_id,
        carrier_script=base_plan.carrier_script,
        source_alignment_phonemes=source_alignment_phonemes,
        neutral_phonemes=neutral_phonemes,
        lens_phonemes=lens_phonemes,
        target_word_indexes=base_plan.target_word_indexes,
        target_occurrences=tuple(occurrences),
        sanitized_incidental_character_indexes=incidental,
        equal_model_token_count=token_count,
        base_gate_receipt=asdict(base_plan.gate_receipt),
        derived_phone_gate_receipt=derived_receipt,
        candidate_enabled=False,
        production_route_available=False,
        plan_sha256=sha256_json(payload),
    )


def _load_profile_and_evidence() -> tuple[dict[str, Any], dict[str, Any]]:
    if sha256_file(PROFILE_PATH) != EXPECTED_PROFILE_SHA256:
        raise ReciprocalFeasibilityProtocolError("frozen research profile drifted")
    if sha256_file(EVIDENCE_PATH) != EXPECTED_EVIDENCE_SHA256:
        raise ReciprocalFeasibilityProtocolError("frozen literature evidence drifted")
    profile = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    evidence = json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))
    if (
        profile.get("status") != "disabled-research-only"
        or profile.get("enabled") is not False
        or profile.get("production_approved") is not False
        or profile.get("feature_flag", {}).get("default") is not False
    ):
        raise ReciprocalFeasibilityProtocolError(
            "reciprocal research profile is not inert"
        )
    rules = {rule["rule_id"]: rule for rule in profile.get("rules", [])}
    primary = rules.get(RULE_ID)
    secondary = rules.get(SECONDARY_UNRENDERED_RULE_ID)
    expected_primary = {
        "rule_id": RULE_ID,
        "priority": 1,
        "enabled": False,
        "source_phone": NEUTRAL_PHONE,
        "neutral_phone": NEUTRAL_PHONE,
        "lens_phone": LENS_PHONE,
        "target_scope": "exact stressed /ɔ/ occurrences only",
        "evidence_tier": "tier-1-direct-majority-small-sample",
        "stage": "eligible-for-future-disabled-acoustic-feasibility-only",
        "exact_target_coverage_required": True,
        "ordinary_context_local_anchors_required": True,
        "listener_validation_complete": False,
    }
    expected_secondary = {
        "rule_id": SECONDARY_UNRENDERED_RULE_ID,
        "priority": 2,
        "enabled": False,
        "source_phone": "a",
        "neutral_phone": "a",
        "lens_phone": "æ",
        "target_scope": "exact stressed /a/ occurrences only; never unstressed /ɐ/",
        "evidence_tier": "tier-2-direct-weak-majority-small-sample",
        "stage": "eligible-for-future-disabled-acoustic-feasibility-only",
        "exact_target_coverage_required": True,
        "ordinary_context_local_anchors_required": True,
        "listener_validation_complete": False,
    }
    if primary != expected_primary or secondary != expected_secondary:
        raise ReciprocalFeasibilityProtocolError("research rule profile drifted")
    evidence_candidates = {
        row["id"]: row for row in evidence.get("candidate_contrasts", [])
    }
    finding = evidence_candidates.get(RULE_ID)
    secondary_finding = evidence_candidates.get(SECONDARY_UNRENDERED_RULE_ID)
    if evidence.get("decision") != {
        "status": "research-only-disabled",
        "production_approved_mappings": [],
        "strongest_evidenced_candidate": RULE_ID,
        "secondary_candidate": SECONDARY_UNRENDERED_RULE_ID,
        "ambiguous_candidate": "bp-close-mid-front-to-ce-near-high-front",
        "rejected_candidate": "bp-close-mid-back-to-ce-high-back",
    }:
        raise ReciprocalFeasibilityProtocolError("evidence decision boundary drifted")
    direct_source = next(
        (row for row in evidence.get("sources", []) if row.get("id") == "D1"),
        None,
    )
    if direct_source is None or {
        "citation": direct_source.get("citation"),
        "stable_url": direct_source.get("stable_url"),
        "doi": direct_source.get("doi"),
        "population_alignment": direct_source.get("population_alignment"),
        "method": direct_source.get("method"),
        "limits": tuple(direct_source.get("limits", [])),
    } != {
        "citation": (
            "Elvin, J., Tuninetti, A., & Escudero, P. (2018). Non-Native "
            "Dialect Matters: The Perception of European and Brazilian "
            "Portuguese Vowels by Californian English Monolinguals and "
            "Spanish-English Bilinguals. Languages, 3(3), 37."
        ),
        "stable_url": "https://doi.org/10.3390/languages3030037",
        "doi": "10.3390/languages3030037",
        "population_alignment": EXPECTED_D1_POPULATION,
        "method": EXPECTED_D1_METHOD,
        "limits": EXPECTED_D1_LIMITS,
    }:
        raise ReciprocalFeasibilityProtocolError("D1 evidence values drifted")
    if finding is None or {
        "id": finding.get("id"),
        "source_phone": finding.get("source_phone"),
        "listener_phone": finding.get("listener_phone"),
        "decision": finding.get("decision"),
        "confidence_tier": finding.get("confidence_tier"),
        "source_ids": finding.get("source_ids"),
        "response_share": finding.get("observed_response", {}).get("response_share"),
        "listener_group": finding.get("observed_response", {}).get("listener_group"),
    } != {
        "id": RULE_ID,
        "source_phone": NEUTRAL_PHONE,
        "listener_phone": LENS_PHONE,
        "decision": "selected-for-disabled-acoustic-feasibility",
        "confidence_tier": "tier-1-direct-majority-small-sample",
        "source_ids": ["D1"],
        "response_share": 0.72,
        "listener_group": "nine Central Valley CE monolingual women",
    }:
        raise ReciprocalFeasibilityProtocolError("primary perceptual evidence drifted")
    if secondary_finding is None or {
        "id": secondary_finding.get("id"),
        "source_phone": secondary_finding.get("source_phone"),
        "listener_phone": secondary_finding.get("listener_phone"),
        "decision": secondary_finding.get("decision"),
        "confidence_tier": secondary_finding.get("confidence_tier"),
        "source_ids": secondary_finding.get("source_ids"),
        "response_share": secondary_finding.get("observed_response", {}).get(
            "response_share"
        ),
    } != {
        "id": SECONDARY_UNRENDERED_RULE_ID,
        "source_phone": "a",
        "listener_phone": "æ",
        "decision": "selected-for-disabled-acoustic-feasibility",
        "confidence_tier": "tier-2-direct-weak-majority-small-sample",
        "source_ids": ["D1"],
        "response_share": 0.54,
    }:
        raise ReciprocalFeasibilityProtocolError(
            "secondary held perceptual evidence drifted"
        )
    if profile["evidence"] != {
        "artifact": "evidence.json",
        "sha256": EXPECTED_EVIDENCE_SHA256,
        "strongest_primary_source": "https://doi.org/10.3390/languages3030037",
    }:
        raise ReciprocalFeasibilityProtocolError("profile evidence binding drifted")
    return profile, evidence


def _load_voice_screen_summary() -> dict[str, Any]:
    if sha256_file(VOICE_SCREEN_SUMMARY_PATH) != EXPECTED_VOICE_SCREEN_SUMMARY_SHA256:
        raise ReciprocalFeasibilityProtocolError("voice-screen summary drifted")
    summary = json.loads(VOICE_SCREEN_SUMMARY_PATH.read_text(encoding="utf-8"))
    if not (
        summary.get("run_id") == "20260717-kokoro-bilingual-voice-screen-v1"
        and summary.get("status") == "pending-human-review"
        and summary.get("voice_selection_performed") is False
        and summary.get("production_candidate_enabled") is False
        and "selected_voice_id" not in summary
        and "selected_voice_ids" not in summary
        and all(
            item.get("status") == "pending-human-review"
            for item in summary.get("human_reviews", [])
        )
    ):
        raise ReciprocalFeasibilityProtocolError(
            "voice-screen no-selection boundary drifted"
        )
    return summary


def _model_vocab() -> frozenset[str]:
    path = resolve_pinned_file(CONFIG_FILE)
    if sha256_file(path) != MODEL_HASHES[CONFIG_FILE]:
        raise ReciprocalFeasibilityProtocolError("Kokoro config hash mismatch")
    return frozenset(json.loads(path.read_text(encoding="utf-8"))["vocab"])


def _profile_plan() -> tuple[PortugueseCarrierPlan, ReciprocalProfilePhonePlan]:
    planner = PortugueseCarrierPlannerV1.load(voice_id=TECHNICAL_PROBE_VOICE_ID)
    first = plan_portuguese_smoke_fixture_v1(planner)
    second = plan_portuguese_smoke_fixture_v1(planner)
    if stable_json(asdict(first)) != stable_json(asdict(second)):
        raise ReciprocalFeasibilityProtocolError(
            "stable Portuguese fixture is not exactly repeatable"
        )
    model_vocab = _model_vocab()
    profile_plan = build_reciprocal_profile_phone_plan(
        first,
        model_vocab=model_vocab,
        mandatory_gate=CandidateGate(),
        native_index=PortuguesePositiveOnlyIndexV1(),
        english_kokoro_index=KokoroGateIndex(),
        english_kokoro_compatible_symbols=model_vocab,
    )
    return first, profile_plan


def render_manifest() -> list[dict[str, Any]]:
    rows = (
        ("ordinary-anchor-neutral", "ordinary_context_local_anchor", "neutral"),
        ("ordinary-anchor-lens", "ordinary_context_local_anchor", "lens"),
        ("controlled-neutral", "controlled_common_source", "neutral"),
        ("controlled-identity", "controlled_common_source", "identity"),
        ("controlled-lens", "controlled_common_source", "lens"),
    )
    return [
        {"order": order, "slot_id": slot_id, "mode": mode, "role": role}
        for order, (slot_id, mode, role) in enumerate(rows, start=1)
    ]


def _path_hashes(paths: Sequence[Path]) -> dict[str, str]:
    return {
        str(path.relative_to(Paths().root)): sha256_file(path)
        for path in sorted(paths, key=str)
    }


def _load_exact_json(path: Path, expected_sha256: str, label: str) -> dict[str, Any]:
    if not path.is_file() or sha256_file(path) != expected_sha256:
        raise ReciprocalFeasibilityProtocolError(f"{label} immutable hash drifted")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ReciprocalFeasibilityProtocolError(f"{label} must be a JSON object")
    return value


def _validate_review_snapshot() -> dict[str, Any]:
    snapshot = _load_exact_json(
        REVIEW_SNAPSHOT_PATH,
        EXPECTED_REVIEW_SNAPSHOT_SHA256,
        "reviewed snapshot",
    )
    if not (
        snapshot.get("schema_version") == 1
        and snapshot.get("snapshot_id")
        == "track-d-reciprocal-feasibility-reviewed-snapshot-v1"
        and snapshot.get("captured_before_resolution_changes") is True
        and snapshot.get("git_head_at_review")
        == "48ab8bb854f0d07a093445a9e0067f9e8aea30f1"
        and snapshot.get("reviewed_protocol_sha256")
        == EXPECTED_REVIEWED_PROTOCOL_SHA256
        and snapshot.get("reviewed_semantic_sha256")
        == EXPECTED_REVIEWED_SEMANTIC_SHA256
        and sha256_json(snapshot.get("reviewed_semantic"))
        == EXPECTED_REVIEWED_SEMANTIC_SHA256
    ):
        raise ReciprocalFeasibilityProtocolError(
            "reviewed snapshot protocol or semantic binding drifted"
        )
    reviewed_files = snapshot.get("reviewed_file_sha256")
    if (
        not isinstance(reviewed_files, dict)
        or set(reviewed_files)
        != {
            "scripts/prepare_ptbr_listener_lens_feasibility.py",
            "scripts/run_ptbr_listener_lens_feasibility.py",
            "src/earshift_bakeoff/ptbr_listener_lens_feasibility.py",
            "src/earshift_bakeoff/ptbr_listener_lens_feasibility_protocol.py",
            "tests/test_ptbr_listener_lens_feasibility.py",
            "tests/test_ptbr_listener_lens_feasibility_protocol.py",
        }
        or not all(
            isinstance(value, str) and len(value) == 64
            for value in reviewed_files.values()
        )
    ):
        raise ReciprocalFeasibilityProtocolError(
            "reviewed snapshot file inventory drifted"
        )
    return snapshot


def _validate_raw_report(
    *,
    path: Path,
    expected_sha256: str,
    expected_report_id: str,
    expected_role: str,
    snapshot: dict[str, Any],
) -> dict[str, Any]:
    report = _load_exact_json(path, expected_sha256, expected_report_id)
    expected_snapshot = {
        "path": str(REVIEW_SNAPSHOT_PATH.relative_to(Paths().root)),
        "sha256": EXPECTED_REVIEW_SNAPSHOT_SHA256,
        "protocol_sha256": EXPECTED_REVIEWED_PROTOCOL_SHA256,
        "semantic_sha256": EXPECTED_REVIEWED_SEMANTIC_SHA256,
        "protocol_module_sha256": snapshot["reviewed_file_sha256"][
            "src/earshift_bakeoff/ptbr_listener_lens_feasibility_protocol.py"
        ],
        "runner_module_sha256": snapshot["reviewed_file_sha256"][
            "src/earshift_bakeoff/ptbr_listener_lens_feasibility.py"
        ],
    }
    if not (
        report.get("schema_version") == 1
        and report.get("status") == "immutable_raw_report"
        and report.get("report_id") == expected_report_id
        and report.get("reviewer_role") == expected_role
        and isinstance(report.get("reviewer_id"), str)
        and report["reviewer_id"].strip()
        and report.get("verdict") == "approve_with_required_changes"
        and report.get("reviewed_snapshot") == expected_snapshot
        and report.get("approval_is_not_freeze_authorization") is True
        and report.get("independent_recheck_required_after_resolution") is True
    ):
        raise ReciprocalFeasibilityProtocolError(
            f"{expected_report_id} identity, verdict, or snapshot binding drifted"
        )
    findings = report.get("findings")
    if not isinstance(findings, list) or not findings:
        raise ReciprocalFeasibilityProtocolError(
            f"{expected_report_id} findings must be nonempty"
        )
    finding_ids: list[str] = []
    for expected_number, finding in enumerate(findings, start=1):
        if not isinstance(finding, dict) or not (
            finding.get("number") == expected_number
            and isinstance(finding.get("finding_id"), str)
            and finding["finding_id"].strip()
            and finding.get("severity") in {"critical", "high", "medium", "low"}
            and isinstance(finding.get("finding"), str)
            and finding["finding"].strip()
            and isinstance(finding.get("required_resolution"), str)
            and finding["required_resolution"].strip()
        ):
            raise ReciprocalFeasibilityProtocolError(
                f"{expected_report_id} finding numbering or content drifted"
            )
        finding_ids.append(finding["finding_id"])
    if len(finding_ids) != len(set(finding_ids)):
        raise ReciprocalFeasibilityProtocolError(
            f"{expected_report_id} contains duplicate finding IDs"
        )
    required = report.get("required_resolutions")
    if required != [
        {"finding_id": finding_id, "required": True} for finding_id in finding_ids
    ]:
        raise ReciprocalFeasibilityProtocolError(
            f"{expected_report_id} required resolutions are not one-to-one"
        )
    return report


def validate_raw_report_resolution_chain() -> dict[str, Any]:
    """Validate raw reports and the author's resolution without self-approval."""

    snapshot = _validate_review_snapshot()
    reports = (
        _validate_raw_report(
            path=SELECTION_REPORT_PATH,
            expected_sha256=EXPECTED_SELECTION_REPORT_SHA256,
            expected_report_id="track-d-selection-claims-independent-report-v1",
            expected_role="selection-integrity-claims-and-human-review",
            snapshot=snapshot,
        ),
        _validate_raw_report(
            path=ACOUSTIC_REPORT_PATH,
            expected_sha256=EXPECTED_ACOUSTIC_REPORT_SHA256,
            expected_report_id="track-d-acoustic-instrument-independent-report-v1",
            expected_role="acoustic-instrument-alignment-and-thresholds",
            snapshot=snapshot,
        ),
    )
    for key in ("report_id", "reviewer_id", "reviewer_role"):
        if len({report[key] for report in reports}) != 2:
            raise ReciprocalFeasibilityProtocolError(
                f"independent reports require distinct {key} values"
            )
    all_findings = [finding for report in reports for finding in report["findings"]]
    finding_ids = [finding["finding_id"] for finding in all_findings]
    if len(finding_ids) != 15 or len(finding_ids) != len(set(finding_ids)):
        raise ReciprocalFeasibilityProtocolError(
            "independent report finding IDs must be 15 globally unique values"
        )
    resolution = _load_exact_json(
        RESOLUTION_PATH,
        EXPECTED_RESOLUTION_SHA256,
        "Track D resolution",
    )
    expected_reports = [
        {
            "report_id": reports[0]["report_id"],
            "reviewer_role": reports[0]["reviewer_role"],
            "path": str(SELECTION_REPORT_PATH.relative_to(Paths().root)),
            "sha256": EXPECTED_SELECTION_REPORT_SHA256,
        },
        {
            "report_id": reports[1]["report_id"],
            "reviewer_role": reports[1]["reviewer_role"],
            "path": str(ACOUSTIC_REPORT_PATH.relative_to(Paths().root)),
            "sha256": EXPECTED_ACOUSTIC_REPORT_SHA256,
        },
    ]
    if not (
        resolution.get("schema_version") == 1
        and resolution.get("resolution_id")
        == "track-d-reciprocal-feasibility-resolution-v1"
        and resolution.get("status") == PROTOCOL_REVIEW_STATUS
        and isinstance(resolution.get("resolution_author_id"), str)
        and resolution["resolution_author_id"].strip()
        and resolution.get("reviewed_snapshot")
        == {
            "path": str(REVIEW_SNAPSHOT_PATH.relative_to(Paths().root)),
            "sha256": EXPECTED_REVIEW_SNAPSHOT_SHA256,
            "protocol_sha256": EXPECTED_REVIEWED_PROTOCOL_SHA256,
            "semantic_sha256": EXPECTED_REVIEWED_SEMANTIC_SHA256,
        }
        and resolution.get("reports") == expected_reports
        and resolution.get("freeze_authorized") is False
    ):
        raise ReciprocalFeasibilityProtocolError(
            "resolution identity, report bindings, or pending state drifted"
        )
    mappings = resolution.get("finding_resolutions")
    if not isinstance(mappings, list):
        raise ReciprocalFeasibilityProtocolError("finding resolutions are missing")
    by_finding = {finding["finding_id"]: finding for finding in all_findings}
    mapped_ids: list[str] = []
    for mapping in mappings:
        if not isinstance(mapping, dict):
            raise ReciprocalFeasibilityProtocolError("finding resolution is malformed")
        finding_id = mapping.get("finding_id")
        source = by_finding.get(finding_id)
        source_report = (
            next(report for report in reports if source in report["findings"])
            if source is not None
            else None
        )
        if not (
            source is not None
            and source_report is not None
            and mapping.get("source_report_id") == source_report["report_id"]
            and mapping.get("severity") == source["severity"]
            and mapping.get("status") == "implemented_pending_independent_recheck"
            and isinstance(mapping.get("resolution_summary"), str)
            and mapping["resolution_summary"].strip()
            and isinstance(mapping.get("code_paths"), list)
            and mapping["code_paths"]
            and all(
                isinstance(path, str) and path.strip() for path in mapping["code_paths"]
            )
            and isinstance(mapping.get("test_ids"), list)
            and mapping["test_ids"]
            and all(
                isinstance(test_id, str) and test_id.startswith("tests/")
                for test_id in mapping["test_ids"]
            )
        ):
            raise ReciprocalFeasibilityProtocolError(
                f"finding resolution is incomplete or stale: {finding_id}"
            )
        mapped_ids.append(finding_id)
    if mapped_ids != finding_ids or len(mapped_ids) != len(set(mapped_ids)):
        raise ReciprocalFeasibilityProtocolError(
            "resolution must map every report finding exactly once in report order"
        )
    recheck = resolution.get("independent_recheck")
    expected_rechecks = [
        {
            "reviewer_role": reports[0]["reviewer_role"],
            "path": str(ORIGINAL_SELECTION_RECHECK_PATH.relative_to(Paths().root)),
        },
        {
            "reviewer_role": reports[1]["reviewer_role"],
            "path": str(ORIGINAL_ACOUSTIC_RECHECK_PATH.relative_to(Paths().root)),
        },
    ]
    if recheck != {
        "required_count": 2,
        "received_count": 0,
        "expected": expected_rechecks,
        "all_findings_independently_rechecked": False,
        "freeze_authorized": False,
    }:
        raise ReciprocalFeasibilityProtocolError(
            "resolution must remain awaiting two independent rechecks"
        )
    selection_attempt = _load_exact_json(
        SELECTION_RECHECK_ATTEMPT_PATH,
        EXPECTED_SELECTION_RECHECK_ATTEMPT_SHA256,
        "selection recheck attempt",
    )
    acoustic_attempt = _load_exact_json(
        ACOUSTIC_RECHECK_ATTEMPT_PATH,
        EXPECTED_ACOUSTIC_RECHECK_ATTEMPT_SHA256,
        "acoustic recheck attempt",
    )
    if not (
        selection_attempt.get("reviewer_id") == "/root/track_d_selection_recheck"
        and selection_attempt.get("reviewer_role")
        == "selection-integrity-claims-and-human-review"
        and selection_attempt.get("verdict") == "reject"
        and selection_attempt.get("freeze_authorized") is False
        and [
            item.get("finding_id")
            for item in selection_attempt.get("residual_findings", [])
        ]
        == ["selection-02-review-gate-too-weak"]
        and selection_attempt.get("subject_hashes", {}).get("resolution_sha256")
        == EXPECTED_RESOLUTION_SHA256
        and selection_attempt.get("subject_hashes", {}).get("source_report_sha256")
        == EXPECTED_SELECTION_REPORT_SHA256
    ):
        raise ReciprocalFeasibilityProtocolError(
            "selection recheck attempt outcome drifted"
        )
    if not (
        acoustic_attempt.get("status") == "immutable_independent_recheck"
        and acoustic_attempt.get("reviewer_id") == "/root/track_d_acoustic_recheck"
        and acoustic_attempt.get("reviewer_role")
        == "acoustic-instrument-alignment-and-thresholds"
        and acoustic_attempt.get("verdict") == "approve"
        and acoustic_attempt.get("freeze_authorized") is True
        and acoustic_attempt.get("residual_findings") == []
        and acoustic_attempt.get("new_blockers") == []
        and acoustic_attempt.get("resolution", {}).get("sha256")
        == EXPECTED_RESOLUTION_SHA256
        and acoustic_attempt.get("source_report", {}).get("sha256")
        == EXPECTED_ACOUSTIC_REPORT_SHA256
    ):
        raise ReciprocalFeasibilityProtocolError(
            "acoustic recheck attempt outcome drifted"
        )
    authorization_resolution = _load_exact_json(
        AUTHORIZATION_GATE_RESOLUTION_PATH,
        EXPECTED_AUTHORIZATION_GATE_RESOLUTION_SHA256,
        "authorization-gate resolution",
    )
    expected_attempts = [
        {
            "reviewer_role": "selection-integrity-claims-and-human-review",
            "reviewer_id": "/root/track_d_selection_recheck",
            "verdict": "reject",
            "path": str(SELECTION_RECHECK_ATTEMPT_PATH.relative_to(Paths().root)),
            "sha256": EXPECTED_SELECTION_RECHECK_ATTEMPT_SHA256,
        },
        {
            "reviewer_role": "acoustic-instrument-alignment-and-thresholds",
            "reviewer_id": "/root/track_d_acoustic_recheck",
            "verdict": "approve",
            "path": str(ACOUSTIC_RECHECK_ATTEMPT_PATH.relative_to(Paths().root)),
            "sha256": EXPECTED_ACOUSTIC_RECHECK_ATTEMPT_SHA256,
        },
    ]
    contract = authorization_resolution.get("authorization_contract")
    if not (
        authorization_resolution.get("schema_version") == 1
        and authorization_resolution.get("resolution_id")
        == "track-d-reciprocal-feasibility-authorization-gate-resolution-v1"
        and authorization_resolution.get("status")
        == "awaiting_repeated_independent_rechecks"
        and authorization_resolution.get("resolution_author_id")
        == "/root/track_d_feasibility_draft"
        and authorization_resolution.get("original_resolution")
        == {
            "path": str(RESOLUTION_PATH.relative_to(Paths().root)),
            "sha256": EXPECTED_RESOLUTION_SHA256,
        }
        and authorization_resolution.get("first_recheck_attempts") == expected_attempts
        and authorization_resolution.get("residual_finding", {}).get("finding_id")
        == "selection-02-review-gate-too-weak"
        and authorization_resolution.get("residual_finding", {}).get("status")
        == "implemented_pending_repeated_independent_recheck"
        and isinstance(contract, dict)
        and contract.get("approval_schema")
        == {
            "path": str(RECHECK_V2_SCHEMA_PATH.relative_to(Paths().root)),
            "sha256": EXPECTED_RECHECK_V2_SCHEMA_SHA256,
        }
        and authorization_resolution.get("received_recheck_count") == 0
        and authorization_resolution.get("freeze_authorized") is False
        and authorization_resolution.get("self_authorization_permitted") is False
        and sha256_file(RECHECK_V2_SCHEMA_PATH) == EXPECTED_RECHECK_V2_SCHEMA_SHA256
    ):
        raise ReciprocalFeasibilityProtocolError(
            "authorization-gate resolution or schema drifted"
        )
    return {
        "status": "awaiting_repeated_independent_rechecks",
        "required_report_count": 2,
        "received_report_count": 2,
        "reports": expected_reports,
        "reviewed_snapshot_sha256": EXPECTED_REVIEW_SNAPSHOT_SHA256,
        "original_resolution": {
            "path": str(RESOLUTION_PATH.relative_to(Paths().root)),
            "sha256": EXPECTED_RESOLUTION_SHA256,
        },
        "resolved_finding_count": len(mapped_ids),
        "all_original_findings_mapped_to_code_and_tests": True,
        "first_recheck_attempts": expected_attempts,
        "authorization_gate_resolution": {
            "path": str(AUTHORIZATION_GATE_RESOLUTION_PATH.relative_to(Paths().root)),
            "sha256": EXPECTED_AUTHORIZATION_GATE_RESOLUTION_SHA256,
            "resolved_finding_id": "selection-02-review-gate-too-weak",
        },
        "authorization_policy": contract,
        "required_recheck_count": 2,
        "self_authorization_permitted": False,
    }


def validate_independent_review_chain() -> dict[str, Any]:
    """Backward-compatible name for the immutable preapproval chain."""

    return validate_raw_report_resolution_chain()


def _closed_json_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ReciprocalFeasibilityProtocolError(
                f"authorization JSON contains duplicate key: {key}"
            )
        value[key] = item
    return value


def _load_closed_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=_closed_json_pairs
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise ReciprocalFeasibilityProtocolError(
            f"{label} is missing or invalid JSON"
        ) from exc
    if not isinstance(value, dict):
        raise ReciprocalFeasibilityProtocolError(f"{label} must be a JSON object")
    return value


def _require_exact_keys(value: Any, keys: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise ReciprocalFeasibilityProtocolError(f"{label} keys are not exact")
    return value


def validate_authorization_approvals(
    *,
    pending_subject_protocol_sha256: str,
    pending_subject_semantic_sha256: str,
    paths: AuthorizationApprovalPaths | None = None,
) -> dict[str, Any]:
    paths = paths or default_authorization_approval_paths()
    physical = (
        paths.selection_recheck,
        paths.acoustic_recheck,
        paths.final_approval_resolution,
    )
    present = [path.is_file() for path in physical]
    if not any(present):
        return {
            "status": "awaiting_repeated_independent_rechecks",
            "received_recheck_count": 0,
            "rechecks": [],
            "final_approval_resolution": None,
            "all_findings_independently_rechecked": False,
            "freeze_authorized": False,
        }
    if not all(present):
        raise ReciprocalFeasibilityProtocolError(
            "authorization gate hard-fails when only part of the approval set exists"
        )

    final_resolution = _load_closed_json(
        paths.final_approval_resolution, "final approval resolution"
    )
    _require_exact_keys(
        final_resolution,
        {
            "schema_version",
            "status",
            "resolution_id",
            "authorization_gate_resolution",
            "reviewed_pending_subject",
            "rechecks",
            "all_findings_approved",
            "freeze_authorized",
        },
        "final approval resolution",
    )
    gate_binding = _require_exact_keys(
        final_resolution["authorization_gate_resolution"],
        {"path", "sha256"},
        "final approval authorization-gate binding",
    )
    subject_binding = _require_exact_keys(
        final_resolution["reviewed_pending_subject"],
        {"protocol_sha256", "semantic_sha256"},
        "final approval subject binding",
    )
    expected_subject = {
        "protocol_sha256": pending_subject_protocol_sha256,
        "semantic_sha256": pending_subject_semantic_sha256,
    }
    if not (
        final_resolution["schema_version"] == 1
        and final_resolution["status"] == "two_independent_approvals_bound"
        and final_resolution["resolution_id"]
        == "track-d-reciprocal-feasibility-final-approval-resolution-v1"
        and gate_binding
        == {
            "path": str(AUTHORIZATION_GATE_RESOLUTION_PATH.relative_to(Paths().root)),
            "sha256": EXPECTED_AUTHORIZATION_GATE_RESOLUTION_SHA256,
        }
        and subject_binding == expected_subject
        and final_resolution["all_findings_approved"] is True
        and final_resolution["freeze_authorized"] is True
        and isinstance(final_resolution["rechecks"], list)
        and len(final_resolution["rechecks"]) == 2
    ):
        raise ReciprocalFeasibilityProtocolError(
            "final approval resolution is stale, incomplete, or not unanimous"
        )

    selection_report = json.loads(SELECTION_REPORT_PATH.read_text(encoding="utf-8"))
    acoustic_report = json.loads(ACOUSTIC_REPORT_PATH.read_text(encoding="utf-8"))
    slots = (
        {
            "physical_path": paths.selection_recheck,
            "relative_path": str(SELECTION_RECHECK_V2_PATH.relative_to(Paths().root)),
            "recheck_id": "track-d-selection-claims-independent-recheck-v2",
            "reviewer_id": SELECTION_RECHECK_V2_REVIEWER_ID,
            "reviewer_role": "selection-integrity-claims-and-human-review",
            "report_path": str(SELECTION_REPORT_PATH.relative_to(Paths().root)),
            "report_id": "track-d-selection-claims-independent-report-v1",
            "report_sha256": EXPECTED_SELECTION_REPORT_SHA256,
            "finding_ids": [
                item["finding_id"] for item in selection_report["findings"]
            ],
        },
        {
            "physical_path": paths.acoustic_recheck,
            "relative_path": str(ACOUSTIC_RECHECK_V2_PATH.relative_to(Paths().root)),
            "recheck_id": "track-d-acoustic-instrument-independent-recheck-v2",
            "reviewer_id": ACOUSTIC_RECHECK_V2_REVIEWER_ID,
            "reviewer_role": "acoustic-instrument-alignment-and-thresholds",
            "report_path": str(ACOUSTIC_REPORT_PATH.relative_to(Paths().root)),
            "report_id": "track-d-acoustic-instrument-independent-report-v1",
            "report_sha256": EXPECTED_ACOUSTIC_REPORT_SHA256,
            "finding_ids": [item["finding_id"] for item in acoustic_report["findings"]],
        },
    )
    receipts: list[dict[str, Any]] = []
    for index, slot in enumerate(slots):
        bound = _require_exact_keys(
            final_resolution["rechecks"][index],
            {"path", "sha256", "reviewer_id", "reviewer_role"},
            "final approval recheck receipt",
        )
        actual_hash = sha256_file(slot["physical_path"])
        if bound != {
            "path": slot["relative_path"],
            "sha256": actual_hash,
            "reviewer_id": slot["reviewer_id"],
            "reviewer_role": slot["reviewer_role"],
        }:
            raise ReciprocalFeasibilityProtocolError(
                "final approval resolution whole-file recheck hash drifted"
            )
        recheck = _load_closed_json(slot["physical_path"], "independent recheck v2")
        _require_exact_keys(
            recheck,
            {
                "schema_version",
                "status",
                "recheck_id",
                "reviewer_id",
                "reviewer_role",
                "source_report",
                "original_resolution",
                "authorization_gate_resolution",
                "reviewed_pending_subject",
                "finding_reviews",
                "residual_findings",
                "new_blockers",
                "verdict",
                "freeze_authorized",
            },
            "independent recheck v2",
        )
        source_binding = _require_exact_keys(
            recheck["source_report"],
            {"path", "report_id", "sha256"},
            "recheck source-report binding",
        )
        original_binding = _require_exact_keys(
            recheck["original_resolution"],
            {"path", "sha256"},
            "recheck original-resolution binding",
        )
        recheck_gate_binding = _require_exact_keys(
            recheck["authorization_gate_resolution"],
            {"path", "sha256"},
            "recheck authorization-gate binding",
        )
        recheck_subject = _require_exact_keys(
            recheck["reviewed_pending_subject"],
            {"protocol_sha256", "semantic_sha256"},
            "recheck pending-subject binding",
        )
        reviews = recheck["finding_reviews"]
        if not isinstance(reviews, list):
            raise ReciprocalFeasibilityProtocolError(
                "recheck finding reviews must be an ordered list"
            )
        for review in reviews:
            _require_exact_keys(
                review,
                {"finding_id", "verdict", "resolved"},
                "per-finding recheck verdict",
            )
        if not (
            recheck["schema_version"] == 2
            and recheck["status"] == "immutable_independent_authorization_recheck"
            and recheck["recheck_id"] == slot["recheck_id"]
            and recheck["reviewer_id"] == slot["reviewer_id"]
            and recheck["reviewer_role"] == slot["reviewer_role"]
            and source_binding
            == {
                "path": slot["report_path"],
                "report_id": slot["report_id"],
                "sha256": slot["report_sha256"],
            }
            and original_binding
            == {
                "path": str(RESOLUTION_PATH.relative_to(Paths().root)),
                "sha256": EXPECTED_RESOLUTION_SHA256,
            }
            and recheck_gate_binding == gate_binding
            and recheck_subject == expected_subject
            and reviews
            == [
                {"finding_id": finding_id, "verdict": "approve", "resolved": True}
                for finding_id in slot["finding_ids"]
            ]
            and all(review["resolved"] is True for review in reviews)
            and recheck["residual_findings"] == []
            and recheck["new_blockers"] == []
            and recheck["verdict"] == "approve"
            and recheck["freeze_authorized"] is True
        ):
            raise ReciprocalFeasibilityProtocolError(
                "independent recheck is stale, incomplete, rejected, or non-unanimous"
            )
        receipts.append(bound)
    if (
        len({item["reviewer_id"] for item in receipts}) != 2
        or len({item["reviewer_role"] for item in receipts}) != 2
    ):
        raise ReciprocalFeasibilityProtocolError(
            "authorization requires distinct reviewer IDs and roles"
        )
    return {
        "status": "authorized_for_protocol_freeze",
        "received_recheck_count": 2,
        "rechecks": receipts,
        "final_approval_resolution": {
            "path": str(FINAL_APPROVAL_RESOLUTION_PATH.relative_to(Paths().root)),
            "sha256": sha256_file(paths.final_approval_resolution),
        },
        "all_findings_independently_rechecked": True,
        "freeze_authorized": True,
    }


def canonical_pending_protocol_payload(protocol: dict[str, Any]) -> dict[str, Any]:
    canonical = json.loads(stable_json(protocol))
    canonical.pop("protocol_sha256", None)
    canonical["status"] = "authorization_gate_resolved_awaiting_repeated_rechecks"
    review = canonical.get("independent_protocol_review", {})
    if isinstance(review, dict) and "pending_review_basis" in review:
        canonical["independent_protocol_review"] = review["pending_review_basis"]
    repo_inputs = canonical.get("bindings", {}).get("repo_bound_inputs", {})
    if isinstance(repo_inputs, dict):
        for path in (
            str(SELECTION_RECHECK_V2_PATH.relative_to(Paths().root)),
            str(ACOUSTIC_RECHECK_V2_PATH.relative_to(Paths().root)),
            str(FINAL_APPROVAL_RESOLUTION_PATH.relative_to(Paths().root)),
        ):
            repo_inputs.pop(path, None)
    return canonical


def _recheck_semantic_projection(protocol: dict[str, Any]) -> dict[str, Any]:
    canonical = canonical_pending_protocol_payload(protocol)
    return {
        key: value
        for key, value in canonical.items()
        if key not in {"schema_version", "api_calls", "paid_calls"}
    }


def protocol_record(
    approval_paths: AuthorizationApprovalPaths | None = None,
) -> dict[str, Any]:
    if TECHNICAL_PROBE_VOICE_ID != "pf_dora":
        raise ReciprocalFeasibilityProtocolError(
            "lexicographic Portuguese technical-probe voice changed"
        )
    pending_review_basis = validate_raw_report_resolution_chain()
    profile, evidence = _load_profile_and_evidence()
    voice_screen_summary = _load_voice_screen_summary()
    base_plan, phone_plan = _profile_plan()
    direct_source = next(
        source for source in evidence["sources"] if source["id"] == "D1"
    )
    voice = VOICE_SPECS_BY_ID[TECHNICAL_PROBE_VOICE_ID]
    voice_path = resolve_pinned_file(voice.filename)
    model_paths = {
        filename: resolve_pinned_file(filename)
        for filename in (CONFIG_FILE, MODEL_FILE)
    }
    actual_model_hashes = {
        filename: sha256_file(path) for filename, path in model_paths.items()
    }
    if actual_model_hashes != {
        filename: MODEL_HASHES[filename] for filename in (CONFIG_FILE, MODEL_FILE)
    }:
        raise ReciprocalFeasibilityProtocolError("Kokoro model asset drifted")
    if sha256_file(voice_path) != voice.sha256:
        raise ReciprocalFeasibilityProtocolError("technical-probe voice asset drifted")

    gate_protocol = PT_GATE_ROOT / "protocol.json"
    gate_receipt = PT_GATE_ROOT / "full-index-receipt.json"
    english_gate_protocol = ENGLISH_KOKORO_GATE_ROOT / "protocol.json"
    english_gate_receipt = ENGLISH_KOKORO_GATE_ROOT / "full-index-receipt.json"
    if not all(
        path.is_file()
        for path in (
            Paths().gate_db,
            Paths().portuguese_kokoro_gate_db,
            Paths().kokoro_gate_db,
            gate_protocol,
            gate_receipt,
            english_gate_protocol,
            english_gate_receipt,
        )
    ):
        raise ReciprocalFeasibilityProtocolError(
            "required local gate database or receipt is missing"
        )
    expected_gate_hashes = {
        gate_protocol: EXPECTED_PT_GATE_PROTOCOL_SHA256,
        gate_receipt: EXPECTED_PT_GATE_RECEIPT_SHA256,
        english_gate_protocol: EXPECTED_EN_GATE_PROTOCOL_SHA256,
        english_gate_receipt: EXPECTED_EN_GATE_RECEIPT_SHA256,
    }
    if any(
        sha256_file(path) != expected for path, expected in expected_gate_hashes.items()
    ):
        raise ReciprocalFeasibilityProtocolError("frozen gate evidence drifted")
    espeak_path_value = shutil.which("espeak-ng")
    if not espeak_path_value:
        raise ReciprocalFeasibilityProtocolError("espeak-ng is unavailable")
    espeak_path = Path(espeak_path_value).resolve()

    code_paths = (
        Paths().root
        / "src"
        / "earshift_bakeoff"
        / "ptbr_listener_lens_feasibility_protocol.py",
        Paths().root / "src" / "earshift_bakeoff" / "ptbr_listener_lens_feasibility.py",
        Paths().root / "src" / "earshift_bakeoff" / "portuguese_carrier_planner_v1.py",
        Paths().root / "src" / "earshift_bakeoff" / "portuguese_kokoro_gate.py",
        Paths().root / "src" / "earshift_bakeoff" / "kokoro_gate_bridge.py",
        Paths().root / "src" / "earshift_bakeoff" / "kokoro_specs.py",
        Paths().root / "src" / "earshift_bakeoff" / "kokoro_synthesis.py",
        Paths().root / "src" / "earshift_bakeoff" / "gates.py",
        Paths().root / "scripts" / "prepare_ptbr_listener_lens_feasibility.py",
        Paths().root / "scripts" / "run_ptbr_listener_lens_feasibility.py",
        MEASUREMENT_SCRIPT,
        RESPONSE_SCHEMA_PATH,
        RECHECK_V2_SCHEMA_PATH,
        Paths().root / "tests" / "test_ptbr_listener_lens_feasibility.py",
        Paths().root / "tests" / "test_ptbr_listener_lens_feasibility_protocol.py",
    )
    repo_bound_paths = code_paths + (
        EVIDENCE_PATH,
        PROFILE_PATH,
        gate_protocol,
        gate_receipt,
        english_gate_protocol,
        english_gate_receipt,
        VOICE_SCREEN_SUMMARY_PATH,
        REVIEW_SNAPSHOT_PATH,
        SELECTION_REPORT_PATH,
        ACOUSTIC_REPORT_PATH,
        RESOLUTION_PATH,
        SELECTION_RECHECK_ATTEMPT_PATH,
        ACOUSTIC_RECHECK_ATTEMPT_PATH,
        AUTHORIZATION_GATE_RESOLUTION_PATH,
    )
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": RUN_ID,
        "status": "authorization_gate_resolved_awaiting_repeated_rechecks",
        "independent_protocol_review": {
            "status": "awaiting_repeated_independent_rechecks",
            "pending_review_basis": pending_review_basis,
            "pending_subject": None,
            "required_recheck_count": 2,
            "received_recheck_count": 0,
            "rechecks": [],
            "final_approval_resolution": None,
            "all_findings_independently_rechecked": False,
            "freeze_authorized": False,
        },
        "question": (
            "Can the pinned Portuguese Kokoro technical probe realize a controlled "
            "stressed BP /ɔ/ to Californian-English /ɑ/ state change in one fixed "
            "opaque context under local ordinary-anchor acoustic gates?"
        ),
        "claim_boundary": {
            "automatic_pass_proves": (
                "a local median-F1/F2 shift plus a localized waveform difference "
                "on this one fixed technical probe"
            ),
            "automatic_pass_does_not_prove": [
                "perceptual efficacy",
                "population generalization",
                "voice quality or voice selection",
                "production readiness",
                "feature enablement",
            ],
            "feature_flag_remains_false": True,
            "candidate_enabled": False,
            "production_route_available": False,
            "numeric_thresholds": "engineering_nonperceptual_criteria_only",
        },
        "evidence": {
            "profile_path": str(PROFILE_PATH.relative_to(Paths().root)),
            "profile_sha256": sha256_file(PROFILE_PATH),
            "evidence_path": str(EVIDENCE_PATH.relative_to(Paths().root)),
            "evidence_sha256": sha256_file(EVIDENCE_PATH),
            "direct_primary_source": profile["evidence"]["strongest_primary_source"],
            "primary_rule": {
                "rule_id": RULE_ID,
                "source_phone": NEUTRAL_PHONE,
                "lens_phone": LENS_PHONE,
                "response_share": 0.72,
                "confidence_tier": "tier-1-direct-majority-small-sample",
                "source_id": "D1",
                "source_doi": "10.3390/languages3030037",
                "listener_population": direct_source["population_alignment"],
                "stimulus_and_task": direct_source["method"],
                "evidence_limits": direct_source["limits"],
            },
            "secondary_rule": {
                "rule_id": SECONDARY_UNRENDERED_RULE_ID,
                "rendered": False,
                "reason": "one strongest-rule feasibility chain only",
            },
        },
        "fixture": {
            "fixture_id": PORTUGUESE_SMOKE_FIXTURE_ID_V1,
            "source_text": PORTUGUESE_SMOKE_TEXT_V1,
            "target_phone": PORTUGUESE_SMOKE_TARGET_PHONE_V1,
            "base_plan_sha256": base_plan.plan_sha256,
            "base_screening_receipt": base_plan.screening_receipt(),
            "profile_phone_plan": phone_plan.pair_record(),
            "profile_layer_boundary": (
                "bounded research-only phone-plan layer above planner v1; the "
                "frozen Portuguese planner and its surface are not modified"
            ),
        },
        "renderer": {
            "model_repo": MODEL_REPO,
            "model_revision": MODEL_REVISION,
            "model_hashes": actual_model_hashes,
            "kokoro_version": importlib.metadata.version("kokoro"),
            "sample_rate_hz": SAMPLE_RATE_HZ,
            "speed": SPEED,
            "rng_seed": RNG_SEED,
            "voice": asdict(voice),
            "voice_asset_sha256": sha256_file(voice_path),
            "voice_role": (
                "lexicographically first pinned pt-BR VoiceSpec technical probe; "
                "not a blind-screen selection and not a production voice"
            ),
            "voice_result_transfer": (
                "nontransferable to any later selected voice; a selected voice "
                "requires a separately frozen confirmation protocol"
            ),
        },
        "voice_screen_boundary": {
            "summary_path": str(VOICE_SCREEN_SUMMARY_PATH.relative_to(Paths().root)),
            "summary_sha256": EXPECTED_VOICE_SCREEN_SUMMARY_SHA256,
            "status": voice_screen_summary["status"],
            "voice_selection_performed": voice_screen_summary[
                "voice_selection_performed"
            ],
            "technical_probe_voice_id": TECHNICAL_PROBE_VOICE_ID,
            "technical_probe_is_selected_voice": False,
            "selected_voice_requires_separate_freeze": True,
        },
        "gates": {
            "mandatory_portuguese_opacity_clearance": {
                "base_planner_receipt": asdict(base_plan.gate_receipt),
                "derived_written_espeak_exact_phone_pass": (
                    phone_plan.derived_phone_gate_receipt.mandatory_portuguese_written_espeak_exact_phone_gate_pass
                ),
                "written_espeak_database_sha256": sha256_file(Paths().gate_db),
                "clearance_basis": (
                    "Portuguese surface/adjacency written and eSpeak checks plus "
                    "exact derived-phone checks; this is the only clearance chain"
                ),
            },
            "portuguese_native_positive_only_rejection_screen": {
                "receipt": asdict(phone_plan.derived_phone_gate_receipt),
                "database_sha256": sha256_file(Paths().portuguese_kokoro_gate_db),
                "protocol_sha256": sha256_file(gate_protocol),
                "receipt_sha256": sha256_file(gate_receipt),
                "negative_used_for_clearance": False,
            },
            "supplemental_english_listener_collision_evidence": {
                "receipt": asdict(
                    phone_plan.derived_phone_gate_receipt.supplemental_english_listener_collision
                ),
                "espeak_phone_database_sha256": sha256_file(Paths().gate_db),
                "kokoro_database_sha256": sha256_file(Paths().kokoro_gate_db),
                "kokoro_protocol_sha256": sha256_file(english_gate_protocol),
                "kokoro_receipt_sha256": sha256_file(english_gate_receipt),
                "negative_used_for_portuguese_clearance": False,
                "interpretation": (
                    "an exact positive rejects; a negative is only absence from the "
                    "two pinned English prediction indexes, never proof of opacity"
                ),
            },
            "espeak_binary": str(espeak_path),
            "espeak_binary_sha256": sha256_file(espeak_path),
        },
        "render_contract": {
            "precondition": (
                "protocol.json and every repository-backed bound input must be "
                "tracked, staged-clean, unstaged-clean, and byte-identical to one "
                "common HEAD immediately before the attempt marker"
            ),
            "manifest": render_manifest(),
            "maximum_decoder_calls": MAX_DECODER_CALLS,
            "decoder_calls_if_started": MAX_DECODER_CALLS,
            "attempts_per_slot": 1,
            "rerenders": 0,
            "retries": 0,
            "variants": 0,
            "selection": "none",
            "ordinary_anchors": (
                "neutral and lens exact-context plans are decoded independently, "
                "each with its own predicted durations, alignment, F0, and noise"
            ),
            "controlled_triplet": (
                "the sanitized pre-intervention opaque source supplies the common "
                "alignment; the neutral plan supplies shared F0/noise; only declared "
                "target state columns are replaced; neutral and identity reset the "
                "same fixed decoder RNG and must be PCM-identical"
            ),
        },
        "alignment_contract": {
            "model_start_boundary_offset": 1,
            "target_columns": [
                occurrence.model_column for occurrence in phone_plan.target_occurrences
            ],
            "primary_measurement_columns": [
                [occurrence.model_column]
                for occurrence in phone_plan.target_occurrences
            ],
            "descriptive_stress_plus_target_columns": [
                [occurrence.stress_model_column, occurrence.model_column]
                for occurrence in phone_plan.target_occurrences
            ],
            "anchor_intervals": "derived independently from each anchor's own durations",
            "controlled_intervals": "derived once from the common pre-intervention source alignment",
        },
        "automatic_gate": {
            "instrument": "standalone Praat Burg",
            "praat_sha256": sha256_file(PRAAT),
            "measurement_script_sha256": sha256_file(MEASUREMENT_SCRIPT),
            "ceilings_hz": list(CEILINGS_HZ),
            "number_of_formants": 5,
            "window_s": 0.025,
            "time_step_s": 0.005,
            "primary_measurement_region": (
                "middle 50 percent of the target-model-column-only interval"
            ),
            "measurement_window_policy": {
                "primary_fraction": MIDDLE_FRACTION,
                "primary_relative_bounds": [0.25, 0.75],
                "exploratory_fractions": [],
                "compute_middle_40_percent": False,
                "compute_middle_60_percent": False,
                "window_selection": "none",
            },
            "retention": {
                "minimum_valid_f1_f2_frames": MIN_VALID_FRAMES,
                "minimum_valid_fraction": MIN_VALID_FRAME_FRACTION,
                "pre_bark_frame_domain": (
                    "finite F1/F2, both positive, and F2 strictly greater than F1"
                ),
                "reported_counts": [
                    "queried",
                    "finite",
                    "positive_ordered",
                    "retained",
                ],
            },
            "plausibility_hz": {
                "f1": [180, 1200],
                "f2": [600, 3500],
                "minimum_f2_minus_f1": 250,
            },
            "per_ceiling_local_gate": (
                "ordinary anchors plausible; controlled neutral nearer local /ɔ/; "
                "controlled lens nearer local /ɑ/; vector cosine >=0.50; magnitude "
                ">=max(0.25 Bark, half local anchor distance); ordinary-anchor "
                "distance >=0.25 Bark; both controlled points plausible"
            ),
            "minimum_anchor_distance_bark": MIN_ANCHOR_DISTANCE_BARK,
            "threshold_interpretation": "engineering_nonperceptual_criteria_only",
            "ceiling_rule": "all three preregistered ceilings pass; no ceiling selection",
            "localization": {
                "inside": "target interval expanded 150 ms and clipped to file bounds",
                "minimum_squared_difference_energy_fraction": LOCALIZATION_MINIMUM,
                "zero_total_difference": "automatic failure",
            },
            "integrity": [
                "all five WAVs finite mono PCM16 at 24 kHz",
                "clipped fraction below 0.001",
                "controlled sample counts equal and nonzero",
                "controlled neutral and identity bit-identical",
                "state replacement columns exactly equal declared target columns",
                "current protocol, code, asset, gate, plan, and WAV hashes match",
            ],
        },
        "automatic_branches": {
            "truth_table": [
                {
                    "condition": "runtime_integrity_fail",
                    "classification": "automatic_acoustic_feasibility_failed",
                    "localization": "skipped",
                },
                {
                    "condition": "measurement_error_before_acoustic_classification",
                    "classification": "automatic_measurement_inconclusive",
                    "localization": "skipped",
                },
                {
                    "condition": "conclusive_acoustic_fail",
                    "classification": "automatic_acoustic_feasibility_failed",
                    "localization": "skipped",
                },
                {
                    "condition": "acoustic_pass_and_localization_tool_error",
                    "classification": "automatic_measurement_inconclusive",
                    "localization": "attempted_error",
                },
                {
                    "condition": "acoustic_pass_and_localization_pass",
                    "classification": "automatic_acoustic_feasibility_pass__blind_prototype_review_pending",
                    "localization": "complete_pass",
                },
                {
                    "condition": "acoustic_pass_and_localization_fail",
                    "classification": "automatic_acoustic_feasibility_failed",
                    "localization": "complete_fail",
                },
            ],
            "automatic_acoustic_feasibility_pass__blind_prototype_review_pending": (
                "all integrity, anchor, three-ceiling local, and localization gates pass; "
                "generate one separate pending blind prototype review"
            ),
            "automatic_acoustic_feasibility_failed": (
                "any fixed automatic gate fails; preserve evidence and generate no review"
            ),
            "automatic_measurement_inconclusive": (
                "instrument, interval, frame-retention, or localization-tool failure; "
                "preserve evidence, generate no review, and do not change parameters"
            ),
        },
        "review_contract": {
            "generated_only_after_automatic_pass": True,
            "status": "pending_blind_prototype_review",
            "response_filename": RESPONSE_FILENAME,
            "required_fields_per_clip": list(REQUIRED_REVIEW_FIELDS),
            "response_schema_path": str(RESPONSE_SCHEMA_PATH.relative_to(Paths().root)),
            "response_schema_sha256": sha256_file(RESPONSE_SCHEMA_PATH),
            "public_root": "public/review",
            "private_root": "private",
            "completion_rule": (
                "download remains disabled until every required field is nonempty "
                "for all three blinded clips; notes are optional"
            ),
            "review_cannot_enable_or_promote": True,
        },
        "bindings": {
            "code_sha256": _path_hashes(code_paths),
            "repo_bound_inputs": _path_hashes(repo_bound_paths),
            "model_assets": {
                filename: sha256_file(path) for filename, path in model_paths.items()
            },
            "voice_asset": {
                voice.filename: sha256_file(voice_path),
            },
        },
        "api_calls": 0,
        "paid_calls": 0,
    }
    pending_subject_payload = canonical_pending_protocol_payload(payload)
    pending_subject_protocol_sha256 = sha256_json(pending_subject_payload)
    pending_subject_semantic_sha256 = sha256_json(_recheck_semantic_projection(payload))
    authorization = validate_authorization_approvals(
        pending_subject_protocol_sha256=pending_subject_protocol_sha256,
        pending_subject_semantic_sha256=pending_subject_semantic_sha256,
        paths=approval_paths,
    )
    payload["independent_protocol_review"] = {
        "status": authorization["status"],
        "pending_review_basis": pending_review_basis,
        "pending_subject": {
            "protocol_sha256": pending_subject_protocol_sha256,
            "semantic_sha256": pending_subject_semantic_sha256,
        },
        "required_recheck_count": 2,
        **authorization,
    }
    if authorization["freeze_authorized"]:
        payload["status"] = "independent_rechecks_approved_ready_for_freeze"
        approval_bindings = {
            item["path"]: item["sha256"] for item in authorization["rechecks"]
        }
        final_resolution = authorization["final_approval_resolution"]
        approval_bindings[final_resolution["path"]] = final_resolution["sha256"]
        payload["bindings"]["repo_bound_inputs"].update(approval_bindings)
    return {**payload, "protocol_sha256": sha256_json(payload)}


def assert_independent_review_resolved(
    protocol: dict[str, Any],
    *,
    approval_paths: AuthorizationApprovalPaths | None = None,
) -> dict[str, Any]:
    current = protocol_record(approval_paths=approval_paths)
    if stable_json(protocol) != stable_json(current):
        raise ReciprocalFeasibilityProtocolError(
            "freeze requires the exact recomputed current reviewed protocol"
        )
    payload = {
        key: value for key, value in protocol.items() if key != "protocol_sha256"
    }
    if protocol.get("protocol_sha256") != sha256_json(payload):
        raise ReciprocalFeasibilityProtocolError(
            "recomputed reviewed protocol hash does not match"
        )
    review = protocol.get("independent_protocol_review", {})
    if not (
        review.get("status") == "authorized_for_protocol_freeze"
        and review.get("received_recheck_count") == 2
        and len(review.get("rechecks", [])) == 2
        and review.get("all_findings_independently_rechecked") is True
        and review.get("freeze_authorized") is True
        and isinstance(review.get("final_approval_resolution"), dict)
    ):
        raise ReciprocalFeasibilityProtocolError(
            "protocol freeze is blocked pending two exact repeated independent approvals"
        )
    return review


def _verify_authorization_inputs_at_head(
    *,
    approval_paths: AuthorizationApprovalPaths,
    review: dict[str, Any],
    repository: Path,
) -> dict[str, Any]:
    repository = repository.resolve()
    head_result = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
    )
    if head_result.returncode != 0:
        raise ReciprocalFeasibilityProtocolError(
            "authorization inputs require a repository HEAD"
        )
    head = head_result.stdout.strip()
    expected = {
        approval_paths.selection_recheck: review["rechecks"][0]["sha256"],
        approval_paths.acoustic_recheck: review["rechecks"][1]["sha256"],
        approval_paths.final_approval_resolution: review["final_approval_resolution"][
            "sha256"
        ],
    }
    verified: list[dict[str, str]] = []
    for path, expected_hash in expected.items():
        if path.is_symlink():
            raise ReciprocalFeasibilityProtocolError(
                "authorization input cannot be a symlink alias"
            )
        try:
            relative = path.absolute().relative_to(repository).as_posix()
        except ValueError as exc:
            raise ReciprocalFeasibilityProtocolError(
                "authorization input is outside the repository"
            ) from exc
        if sha256_file(path) != expected_hash:
            raise ReciprocalFeasibilityProtocolError(
                f"authorization input hash drifted: {relative}"
            )
        tracked = subprocess.run(
            [
                "git",
                "-C",
                str(repository),
                "ls-files",
                "--error-unmatch",
                "--",
                relative,
            ],
            capture_output=True,
        )
        if tracked.returncode != 0:
            raise ReciprocalFeasibilityProtocolError(
                f"authorization input must be tracked at HEAD: {relative}"
            )
        unstaged = subprocess.run(
            ["git", "-C", str(repository), "diff", "--quiet", "--", relative]
        )
        if unstaged.returncode != 0:
            raise ReciprocalFeasibilityProtocolError(
                f"authorization input has unstaged drift: {relative}"
            )
        staged = subprocess.run(
            [
                "git",
                "-C",
                str(repository),
                "diff",
                "--cached",
                "--quiet",
                "HEAD",
                "--",
                relative,
            ]
        )
        if staged.returncode != 0:
            raise ReciprocalFeasibilityProtocolError(
                f"authorization input has staged drift: {relative}"
            )
        committed = subprocess.run(
            ["git", "-C", str(repository), "show", f"{head}:{relative}"],
            capture_output=True,
        )
        if committed.returncode != 0 or committed.stdout != path.read_bytes():
            raise ReciprocalFeasibilityProtocolError(
                f"authorization input is not byte-identical to HEAD: {relative}"
            )
        verified.append({"path": relative, "sha256": expected_hash})
    final_head = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
    )
    if final_head.returncode != 0 or final_head.stdout.strip() != head:
        raise ReciprocalFeasibilityProtocolError(
            "repository HEAD changed during authorization input verification"
        )
    return {"repository_head": head, "inputs": verified}


def write_frozen_protocol(
    protocol: dict[str, Any],
    destination: Path,
    *,
    approval_paths: AuthorizationApprovalPaths | None = None,
    repository: Path | None = None,
) -> None:
    approval_paths = approval_paths or default_authorization_approval_paths()
    review = assert_independent_review_resolved(protocol, approval_paths=approval_paths)
    _verify_authorization_inputs_at_head(
        approval_paths=approval_paths,
        review=review,
        repository=repository or Paths().root,
    )
    if destination.exists():
        existing = json.loads(destination.read_text(encoding="utf-8"))
        if stable_json(existing) != stable_json(protocol):
            raise ReciprocalFeasibilityProtocolError(
                "existing reciprocal feasibility protocol differs"
            )
        return
    if any(
        path.exists()
        for path in (
            destination.parent / "render-attempt.json",
            destination.parent / "render-records.json",
            destination.parent / "analysis.json",
            destination.parent / "audio",
            destination.parent / "public",
            destination.parent / "private",
            destination.parent / "review-generation-failure.json",
            destination.parent / "review-generation.partial",
            destination.parent / "review.html",
            destination.parent / "review-manifest.json",
            destination.parent / "blind-key.json",
        )
    ):
        raise ReciprocalFeasibilityProtocolError(
            "decoder evidence exists before protocol freeze"
        )
    atomic_write_json(destination, protocol)


def prepare() -> dict[str, Any]:
    protocol = protocol_record()
    write_frozen_protocol(protocol, run_dir() / "protocol.json")
    return protocol


def verify_frozen_protocol() -> dict[str, Any]:
    path = run_dir() / "protocol.json"
    if not path.is_file():
        raise ReciprocalFeasibilityProtocolError(
            "frozen reciprocal feasibility protocol is missing"
        )
    stored = json.loads(path.read_text(encoding="utf-8"))
    current = protocol_record()
    if stable_json(stored) != stable_json(current):
        raise ReciprocalFeasibilityProtocolError(
            "reciprocal feasibility protocol or a bound input drifted"
        )
    assert_independent_review_resolved(stored)
    return stored
