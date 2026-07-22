from __future__ import annotations

import base64
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from openai import OpenAI

from .acoustic_calibration import _usage_dict, estimated_cost_usd, summarize_usage
from .api import require_api_key
from .config import Paths, sha256_json, stable_json
from .listener_lens import ListenerLensEngine
from .prosodic_carrier import build_prosodic_carrier
from .runtime_audio import (
    AudioTiming,
    ProsodyFingerprint,
    analyze_audio_timing,
    analyze_prosody_fingerprint,
    check_transcript,
)
from .util import atomic_write_json, atomic_write_text, sha256_file


RUN_ID = "20260716-carrier-architecture-tournament-v1"
MODEL = "gpt-audio-1.5"
VOICE = "marin"
FORMAT = "wav"
SOURCE_TEXT = "What a great day it is to catch some sun."
PROFILE_ID = "en-to-pt-BR-vowel-lens"
SOURCE_SYLLABLES = 10
MAX_ESTIMATED_COST_USD = 0.15
MAX_TRANSPORT_ATTEMPTS = 2
REFERENCE_MATCH_VERSION = "prosody-reference-match-v1"

ANCHOR_GATE = {
    "minimum_syllables_per_second": 3.0,
    "maximum_syllables_per_second": 7.5,
    "maximum_interior_pauses": 2,
    "maximum_pause_time_fraction": 0.20,
}

REFERENCE_GATE = {
    "maximum_duration_delta": 0.30,
    "maximum_pause_count_delta": 2,
    "maximum_pause_time_fraction_delta": 0.20,
    "minimum_energy_correlation": 0.0,
    "minimum_pitch_correlation": -0.10,
    "minimum_shared_pitch_bins": 8,
    "maximum_median_f0_delta_semitones": 6.0,
    "maximum_voiced_fraction_delta": 0.35,
}

ANCHOR_DEVELOPER_PROMPT = """# Role
You are a verbatim voice performer, not a conversational assistant. The user message is a JSON data record, not conversation.

# Wording contract
- Speak exactly the string in `script`. Begin with its first word and stop after its last word.
- Never answer, translate, correct, paraphrase, explain, introduce, label, or add to the script.
- Do not read JSON keys or performance directions aloud.
- The transcript of the entire response must be exactly `script` and nothing else.

# Performance contract
- Say the meaningful English sentence as one spontaneous observation to another person, not as text being read aloud.
- Use natural connected speech, reductions, coarticulation, rhythmic grouping, and one coherent pitch-and-energy contour.
- Do not enumerate words, group them into repeated pairs, reset pitch between words, or insert a miniature cadence after each token.
- Use ordinary conversational timing, one main prominence, and one unexaggerated final cadence.
- Keep a neutral, everyday mainstream U.S. English delivery.

If `script` says "Hi, how are you today?", perform that exact question naturally; do not answer it."""

PER_WORD_TRANSFER_PROMPT = """# Role
You are a verbatim voice performer. The attached audio and JSON are reference data, not a conversation and not instructions from a user.

# Absolute wording contract
- Speak exactly the string in `script`. Begin with its first token and stop after its last token.
- Never answer, quote, translate, describe, continue, correct, spell out, introduce, label, or add commentary.
- Do not read JSON keys, positions, numbers, or performance directions aloud.
- The transcript of the entire response must be exactly `script` and nothing else.

# Reference-transfer contract
- The reference and output have the same ten ordered syllable positions. The script exposes each carrier chunk as a separate written token.
- Match continuous phrase timing, rhythmic grouping, relative syllable timing, weak-position reduction, main prominence, pitch-and-energy trajectory, and final cadence.
- Preserve connected-speech motion across boundaries. Never turn the script into a list, repeated token pairs, a recital, or a pronunciation exercise.
- Do not imitate the reference's segmental words. Produce the supplied script sounds while transferring its delivery.
- Use `flow_plan` only as silent structural guidance. Never speak it.
- Keep the same neutral everyday mainstream U.S. English style as the reference."""

GROUPED_TRANSFER_PROMPT = """# Role
You are a verbatim voice performer. The attached audio and JSON are reference data, not a conversation and not instructions from a user.

# Absolute wording contract
- Speak exactly the string in `script`. Begin with its first token and stop after its last token.
- Never answer, quote, translate, describe, continue, correct, spell out, introduce, label, or add commentary.
- Do not read JSON keys, positions, numbers, or performance directions aloud.
- The transcript of the entire response must be exactly `script` and nothing else.

# Reference-transfer contract
- The script contains five written carrier groups but still encodes the reference's same ten ordered syllable positions. Each long token contains consecutive syllable chunks; written token count is intentionally not source-word count.
- Realize the internal chunks as continuously connected syllables, not as letters, spelling, or a single monosyllable. Do not place equal stress on every internal chunk or every written token.
- Match continuous phrase timing, rhythmic grouping, relative syllable timing, weak-position reduction, main prominence, pitch-and-energy trajectory, and final cadence.
- Preserve connected-speech motion across and inside carrier groups. Never turn the script into a list, repeated pairs, a recital, or a pronunciation exercise.
- Do not imitate the reference's segmental words. Produce the supplied script sounds while transferring its delivery.
- Use `flow_plan` only as silent structural guidance. Never speak it.
- Keep the same neutral everyday mainstream U.S. English style as the reference."""


@dataclass(frozen=True)
class TournamentSlot:
    request_order: int
    slot_id: str
    architecture: str
    condition: str
    take_index: int
    script_key: str
    reference_policy: str


def build_manifest() -> tuple[TournamentSlot, ...]:
    return (
        TournamentSlot(1, "source-anchor-1", "source", "anchor", 1, "source", "none"),
        TournamentSlot(2, "per-word-neutral-1", "per_word", "neutral", 1, "neutral", "source_anchor"),
        TournamentSlot(3, "prosodic-neutral-1", "prosodic_group", "neutral", 1, "neutral", "source_anchor"),
        TournamentSlot(4, "per-word-neutral-2", "per_word", "neutral", 2, "neutral", "source_anchor"),
        TournamentSlot(5, "prosodic-neutral-2", "prosodic_group", "neutral", 2, "neutral", "source_anchor"),
        TournamentSlot(6, "per-word-identity-1", "per_word", "identity", 1, "neutral", "selected_architecture_neutral"),
        TournamentSlot(7, "prosodic-identity-1", "prosodic_group", "identity", 1, "neutral", "selected_architecture_neutral"),
        TournamentSlot(8, "per-word-lens-1", "per_word", "lens", 1, "lens", "selected_architecture_neutral"),
        TournamentSlot(9, "prosodic-lens-1", "prosodic_group", "lens", 1, "lens", "selected_architecture_neutral"),
    )


def _architecture_records() -> tuple[dict[str, Any], dict[str, Any], ListenerLensEngine]:
    engine = ListenerLensEngine()
    source = engine.transform(SOURCE_TEXT, PROFILE_ID)
    grouped = build_prosodic_carrier(source, engine.nonce_checker)
    per_word = {
        "architecture": "per_word",
        "description": "transform-v5 one written carrier token per source word",
        "source_word_count": len(source.words),
        "carrier_token_count": len(source.words),
        "syllable_count": sum(word.syllables for word in source.words),
        "neutral_script": source.neutral_script,
        "lens_script": source.lens_script,
        "carrier_roles": [word.carrier_role for word in source.words],
        "target_source_word_indices": sorted({slot.word_index for slot in source.slots}),
    }
    prosodic = {
        "architecture": "prosodic_group",
        "description": "same transform-v5 syllable chunks regrouped into punctuation-bounded prosodic carrier words",
        "source_word_count": grouped.source_word_count,
        "carrier_token_count": grouped.group_count,
        "syllable_count": grouped.total_syllables,
        "neutral_script": grouped.neutral_script,
        "lens_script": grouped.lens_script,
        "groups": [asdict(group) for group in grouped.groups],
        "gate_attempts": [asdict(attempt) for attempt in grouped.gate_attempts],
        "target_source_word_indices": sorted({slot.source_word_index for slot in grouped.slots}),
    }
    return per_word, prosodic, engine


def protocol_record() -> dict[str, Any]:
    per_word, prosodic, engine = _architecture_records()
    payload = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "status": "exploratory_frozen_before_paid_calls_and_listening",
        "source_text": SOURCE_TEXT,
        "profile_id": PROFILE_ID,
        "source_syllables": SOURCE_SYLLABLES,
        "model": MODEL,
        "voice": VOICE,
        "format": FORMAT,
        "transform_cache_key": engine.cache_key_for(SOURCE_TEXT, PROFILE_ID),
        "architectures": [per_word, prosodic],
        "prompts": {
            "anchor": ANCHOR_DEVELOPER_PROMPT,
            "per_word_transfer": PER_WORD_TRANSFER_PROMPT,
            "grouped_transfer": GROUPED_TRANSFER_PROMPT,
        },
        "anchor_gate": ANCHOR_GATE,
        "reference_match_version": REFERENCE_MATCH_VERSION,
        "reference_gate": REFERENCE_GATE,
        "manifest": [asdict(slot) for slot in build_manifest()],
        "selection": (
            "For each architecture, select the eligible neutral take with the lowest "
            "frozen reference-match score; break exact ties by lower take index. "
            "No listening or target acoustics enter selection."
        ),
        "valid_return_policy": (
            "Once valid decodable audio is returned, the logical slot is final even "
            "if transcript, integrity, anchor, or reference gates fail."
        ),
        "transport_retry_policy": (
            "A 429, 5xx, timeout, connection error, or response with no audio may be "
            "retried once for the same logical slot; no other replacement is allowed."
        ),
        "limits": {
            "logical_slots": 9,
            "maximum_successfully_returned_audio": 9,
            "maximum_transport_attempts_per_slot": MAX_TRANSPORT_ATTEMPTS,
            "maximum_estimated_cost_usd": MAX_ESTIMATED_COST_USD,
        },
        "objective_outputs": [
            "provider exact-transcript check",
            "PCM integrity, clipping, duration, and pause measurements",
            "corrected source-anchor syllables-per-second gate",
            "same prosody-reference score and eligibility family for both architectures",
            "local Whisper transcript/language audit after rendering",
            "target-vowel acoustic diagnostic only after a listening-viable pair exists",
        ],
        "blind_review_axes": [
            "naturalness",
            "list-like delivery",
            "connected grouping",
            "relationship to source rhythm",
            "semantic opacity",
            "target-vowel difference",
            "unrelated delivery interference",
            "commentary, spelling-out, missing, or extra material",
        ],
        "decision_rule": (
            "The grouped architecture advances only if it materially improves connected "
            "naturalness/list avoidance without weakening opacity, exact-content control, "
            "source-syllable structure, or the correctly directed target contrast. This "
            "exploratory run selects an architecture; it does not itself validate a "
            "Brazilian-Portuguese population-perception claim."
        ),
    }
    return {**payload, "protocol_sha256": sha256_json(payload)}


def _pause_fraction(timing: AudioTiming) -> float:
    return timing.interior_pause_s / max(timing.utterance_duration_s, 1e-9)


def _pearson(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        return -1.0
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right))
    denominator = math.sqrt(
        sum((a - left_mean) ** 2 for a in left)
        * sum((b - right_mean) ** 2 for b in right)
    )
    if denominator <= 1e-12:
        return 1.0 if all(abs(a - b) <= 1e-9 for a, b in zip(left, right)) else 0.0
    return numerator / denominator


def _mean_absolute_error(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(abs(a - b) for a, b in zip(left, right)) / max(1, len(left))


def _pause_distance(reference: AudioTiming, candidate: AudioTiming) -> float:
    count_delta = abs(reference.interior_pause_count - candidate.interior_pause_count)
    count_term = min(1.0, count_delta / max(1, reference.interior_pause_count, candidate.interior_pause_count))
    shared = min(len(reference.interior_pauses), len(candidate.interior_pauses))
    if shared:
        position_term = sum(
            abs(reference.interior_pauses[index].start_fraction - candidate.interior_pauses[index].start_fraction)
            for index in range(shared)
        ) / shared
        duration_term = sum(
            abs(reference.interior_pauses[index].duration_s / max(reference.utterance_duration_s, 1e-9)
                - candidate.interior_pauses[index].duration_s / max(candidate.utterance_duration_s, 1e-9))
            for index in range(shared)
        ) / shared
    else:
        position_term = 0.0
        duration_term = 0.0
    return min(1.0, (count_term + position_term + duration_term) / 3)


def compare_prosody(
    reference_timing: AudioTiming,
    reference: ProsodyFingerprint,
    candidate_timing: AudioTiming,
    candidate: ProsodyFingerprint,
) -> dict[str, Any]:
    duration_delta = abs(reference_timing.utterance_duration_s - candidate_timing.utterance_duration_s) / max(
        reference_timing.utterance_duration_s, candidate_timing.utterance_duration_s, 1e-9
    )
    pause_count_delta = abs(reference_timing.interior_pause_count - candidate_timing.interior_pause_count)
    pause_fraction_delta = abs(_pause_fraction(reference_timing) - _pause_fraction(candidate_timing))
    pause_match_distance = _pause_distance(reference_timing, candidate_timing)
    energy_correlation = _pearson(reference.energy_contour_db, candidate.energy_contour_db)
    energy_mae = _mean_absolute_error(reference.energy_contour_db, candidate.energy_contour_db)
    shared_pitch = [
        (left, right)
        for left, right in zip(reference.pitch_contour_semitones, candidate.pitch_contour_semitones)
        if left is not None and right is not None
    ]
    pitch_correlation = _pearson([item[0] for item in shared_pitch], [item[1] for item in shared_pitch]) if shared_pitch else -1.0
    pitch_mae = _mean_absolute_error([item[0] for item in shared_pitch], [item[1] for item in shared_pitch]) if shared_pitch else 24.0
    median_f0_delta = (
        abs(12 * math.log2(candidate.median_f0_hz / reference.median_f0_hz))
        if reference.median_f0_hz > 0 and candidate.median_f0_hz > 0
        else 24.0
    )
    voiced_fraction_delta = abs(reference.voiced_fraction - candidate.voiced_fraction)
    reasons: list[str] = []
    if duration_delta > REFERENCE_GATE["maximum_duration_delta"]:
        reasons.append("duration_delta")
    if pause_count_delta > REFERENCE_GATE["maximum_pause_count_delta"]:
        reasons.append("pause_count_delta")
    if pause_fraction_delta > REFERENCE_GATE["maximum_pause_time_fraction_delta"]:
        reasons.append("pause_time_fraction_delta")
    if energy_correlation < REFERENCE_GATE["minimum_energy_correlation"]:
        reasons.append("energy_correlation")
    if len(shared_pitch) < REFERENCE_GATE["minimum_shared_pitch_bins"]:
        reasons.append("shared_pitch_bins")
    if len(shared_pitch) >= REFERENCE_GATE["minimum_shared_pitch_bins"] and pitch_correlation < REFERENCE_GATE["minimum_pitch_correlation"]:
        reasons.append("pitch_correlation")
    if median_f0_delta > REFERENCE_GATE["maximum_median_f0_delta_semitones"]:
        reasons.append("median_f0_delta")
    if voiced_fraction_delta > REFERENCE_GATE["maximum_voiced_fraction_delta"]:
        reasons.append("voiced_fraction_delta")
    score = (
        duration_delta
        + 0.20 * pause_match_distance
        + 0.25 * ((1 - energy_correlation) / 2)
        + 0.10 * min(energy_mae / 12, 1)
        + 0.25 * ((1 - pitch_correlation) / 2)
        + 0.10 * min(pitch_mae / 12, 1)
        + 0.10 * min(median_f0_delta / 6, 1)
        + 0.10 * voiced_fraction_delta
    )
    rounded = lambda value: round(float(value), 6)
    return {
        "version": REFERENCE_MATCH_VERSION,
        "eligible": not reasons,
        "score": rounded(score),
        "duration_delta": rounded(duration_delta),
        "pause_count_delta": pause_count_delta,
        "pause_time_fraction_delta": rounded(pause_fraction_delta),
        "pause_distance": rounded(pause_match_distance),
        "energy_correlation": rounded(energy_correlation),
        "energy_mae_db": rounded(energy_mae),
        "pitch_correlation": rounded(pitch_correlation),
        "pitch_mae_semitones": rounded(pitch_mae),
        "shared_pitch_bins": len(shared_pitch),
        "median_f0_delta_semitones": rounded(median_f0_delta),
        "voiced_fraction_delta": rounded(voiced_fraction_delta),
        "reasons": reasons,
    }


def _safe_error(exc: Exception) -> tuple[str, str]:
    return type(exc).__name__, str(exc).replace("\n", " ")[:500]


def _retryable_external_failure(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    return bool(
        status == 429
        or isinstance(status, int) and status >= 500
        or type(exc).__name__
        in {
            "APIConnectionError",
            "APITimeoutError",
            "ConnectError",
            "ConnectTimeout",
            "ReadTimeout",
            "TimeoutException",
        }
    )


def _script_for(slot: TournamentSlot, protocol: dict[str, Any]) -> str:
    if slot.script_key == "source":
        return SOURCE_TEXT
    architecture = next(item for item in protocol["architectures"] if item["architecture"] == slot.architecture)
    return architecture[f"{slot.script_key}_script"]


def _flow_plan(slot: TournamentSlot, protocol: dict[str, Any]) -> dict[str, Any]:
    architecture = next(item for item in protocol["architectures"] if item["architecture"] == slot.architecture)
    if slot.architecture == "per_word":
        return {
            "source_word_count": architecture["source_word_count"],
            "carrier_token_count": architecture["carrier_token_count"],
            "syllable_count": architecture["syllable_count"],
            "weak_source_positions_one_based": [
                index + 1 for index, role in enumerate(architecture["carrier_roles"]) if role == "weak"
            ],
            "target_source_positions_one_based": [index + 1 for index in architecture["target_source_word_indices"]],
        }
    return {
        "source_word_count": architecture["source_word_count"],
        "carrier_token_count": architecture["carrier_token_count"],
        "syllable_count": architecture["syllable_count"],
        "groups": [
            {
                "carrier_position_one_based": group["group_index"] + 1,
                "source_positions_one_based": [index + 1 for index in group["source_word_indices"]],
                "syllables": group["syllables"],
                "head_source_position_one_based": group["head_source_word_index"] + 1,
            }
            for group in architecture["groups"]
        ],
        "target_source_positions_one_based": [index + 1 for index in architecture["target_source_word_indices"]],
    }


def _messages(
    slot: TournamentSlot,
    script: str,
    protocol: dict[str, Any],
    reference_audio: str | None,
) -> list[dict[str, Any]]:
    if slot.condition == "anchor":
        return [
            {"role": "developer", "content": ANCHOR_DEVELOPER_PROMPT},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "task": "natural_source_anchor",
                        "script": script,
                        "delivery": "One spontaneous conversational observation with natural connected speech and one ordinary final cadence.",
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            },
        ]
    if reference_audio is None:
        raise RuntimeError(f"{slot.slot_id} is missing its frozen audio reference")
    prompt = PER_WORD_TRANSFER_PROMPT if slot.architecture == "per_word" else GROUPED_TRANSFER_PROMPT
    return [
        {"role": "developer", "content": prompt},
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(
                        {
                            "task": "verbatim_prosody_transfer",
                            "script": script,
                            "condition": slot.condition,
                            "reference_policy": slot.reference_policy,
                            "flow_plan": _flow_plan(slot, protocol),
                        },
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                },
                {
                    "type": "input_audio",
                    "input_audio": {"data": reference_audio, "format": FORMAT},
                },
            ],
        },
    ]


def _read_audio_base64(record: dict[str, Any], run_dir: Path) -> str:
    path = run_dir / record["audio_relative_path"]
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _load_measurements(record: dict[str, Any]) -> tuple[AudioTiming, ProsodyFingerprint]:
    timing_payload = dict(record["timing"])
    timing_payload["interior_pauses"] = tuple(
        __import__("earshift_bakeoff.runtime_audio", fromlist=["PauseInterval"]).PauseInterval(**item)
        for item in timing_payload["interior_pauses"]
    )
    prosody_payload = dict(record["prosody"])
    prosody_payload["energy_contour_db"] = tuple(prosody_payload["energy_contour_db"])
    prosody_payload["pitch_contour_semitones"] = tuple(prosody_payload["pitch_contour_semitones"])
    return AudioTiming(**timing_payload), ProsodyFingerprint(**prosody_payload)


def _attempt_slot(
    client: Any,
    slot: TournamentSlot,
    protocol: dict[str, Any],
    reference: dict[str, Any] | None,
    output: Path,
    attempt_number: int,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    started = time.monotonic()
    attempt: dict[str, Any] = {
        "attempt_number": attempt_number,
        "request_id": "",
        "status": "failed_no_audio",
        "usage": {},
        "estimated_cost_usd": 0.0,
        "retryable_external_failure": False,
    }
    script = _script_for(slot, protocol)
    partial = output.with_suffix(".partial.wav")
    try:
        reference_audio = None if reference is None else _read_audio_base64(reference, output.parents[1])
        completion = client.chat.completions.create(
            model=MODEL,
            modalities=["text", "audio"],
            audio={"voice": VOICE, "format": FORMAT},
            messages=_messages(slot, script, protocol, reference_audio),
            store=False,
        )
        attempt["request_id"] = getattr(completion, "_request_id", None) or ""
        attempt["resolved_model"] = getattr(completion, "model", MODEL)
        attempt["usage"] = _usage_dict(completion)
        attempt["estimated_cost_usd"] = estimated_cost_usd(attempt["usage"])
        audio = completion.choices[0].message.audio
        if audio is None or not audio.data:
            raise ValueError("gpt-audio-1.5 returned no audio payload")
        transcript = getattr(audio, "transcript", "") or ""
        output.parent.mkdir(parents=True, exist_ok=True)
        partial.write_bytes(base64.b64decode(audio.data, validate=True))
        partial.replace(output)
        timing = analyze_audio_timing(output, intended_syllables=SOURCE_SYLLABLES)
        prosody = analyze_prosody_fingerprint(output)
        transcript_result = check_transcript(script, transcript)
        reasons: list[str] = []
        if not transcript_result.exact_token_match:
            reasons.append("provider_transcript_mismatch")
        if timing.sample_rate_hz != 24_000:
            reasons.append("unexpected_sample_rate")
        if not 0.25 <= timing.duration_s <= 45.0:
            reasons.append("duration_out_of_bounds")
        if timing.utterance_duration_s <= 0:
            reasons.append("no_detectable_utterance")
        if timing.clipped_fraction > 0.001:
            reasons.append("excessive_clipping")
        anchor_metrics = None
        if slot.condition == "anchor":
            rate = timing.estimated_syllables_per_second
            pause_fraction = _pause_fraction(timing)
            anchor_metrics = {
                "syllables_per_second": rate,
                "pause_time_fraction": round(pause_fraction, 6),
            }
            if rate is None or not ANCHOR_GATE["minimum_syllables_per_second"] <= rate <= ANCHOR_GATE["maximum_syllables_per_second"]:
                reasons.append("anchor_syllable_rate")
            if timing.interior_pause_count > ANCHOR_GATE["maximum_interior_pauses"]:
                reasons.append("anchor_pause_count")
            if pause_fraction > ANCHOR_GATE["maximum_pause_time_fraction"]:
                reasons.append("anchor_pause_fraction")
        reference_match = None
        if reference is not None and not reasons:
            reference_timing, reference_prosody = _load_measurements(reference)
            reference_match = compare_prosody(reference_timing, reference_prosody, timing, prosody)
            if not reference_match["eligible"]:
                reasons.extend(f"reference_{reason}" for reason in reference_match["reasons"])
        attempt["status"] = "audio_returned"
        record = {
            "slot": asdict(slot),
            "status": "accepted" if not reasons else "rejected",
            "reasons": reasons,
            "request_id": attempt["request_id"],
            "resolved_model": attempt["resolved_model"],
            "provider_transcript": transcript,
            "transcript_check": asdict(transcript_result),
            "audio_relative_path": str(output.relative_to(output.parents[1])),
            "audio_sha256": sha256_file(output),
            "timing": asdict(timing),
            "prosody": asdict(prosody),
            "anchor_metrics": anchor_metrics,
            "reference_match": reference_match,
            "reference_audio_sha256": reference.get("audio_sha256") if reference else None,
            "usage": attempt["usage"],
            "estimated_cost_usd": attempt["estimated_cost_usd"],
        }
        return attempt, record
    except Exception as exc:
        partial.unlink(missing_ok=True)
        error_type, error_detail = _safe_error(exc)
        attempt.update(
            {
                "error_type": error_type,
                "error_detail": error_detail,
                "retryable_external_failure": _retryable_external_failure(exc),
            }
        )
        return attempt, None
    finally:
        attempt["latency_ms"] = round((time.monotonic() - started) * 1000)


def _render_slot(
    client: Any,
    slot: TournamentSlot,
    protocol: dict[str, Any],
    reference: dict[str, Any] | None,
    output: Path,
) -> dict[str, Any]:
    attempts: list[dict[str, Any]] = []
    final: dict[str, Any] | None = None
    for attempt_number in range(1, MAX_TRANSPORT_ATTEMPTS + 1):
        attempt, final = _attempt_slot(
            client, slot, protocol, reference, output, attempt_number
        )
        attempts.append(attempt)
        if final is not None:
            break
        if not attempt["retryable_external_failure"]:
            break
        if attempt_number < MAX_TRANSPORT_ATTEMPTS:
            time.sleep(1.0)
    if final is None:
        final = {
            "slot": asdict(slot),
            "status": "external_failure_unresolved"
            if attempts[-1]["retryable_external_failure"]
            else "failed_no_audio",
            "reasons": [attempts[-1].get("error_type", "unknown_error")],
            "usage": {},
            "estimated_cost_usd": 0.0,
        }
    final["attempts"] = attempts
    return final


def _selected_neutral(records: Sequence[dict[str, Any]], architecture: str) -> dict[str, Any] | None:
    eligible = [
        record
        for record in records
        if record["slot"]["architecture"] == architecture
        and record["slot"]["condition"] == "neutral"
        and record["status"] == "accepted"
        and (record.get("reference_match") or {}).get("eligible")
    ]
    return min(
        eligible,
        key=lambda record: (
            record["reference_match"]["score"],
            record["slot"]["take_index"],
        ),
        default=None,
    )


def prepare_tournament() -> dict[str, Any]:
    protocol = protocol_record()
    run_dir = Paths().artifacts / "architecture-tournament" / RUN_ID
    path = run_dir / "protocol.json"
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if stable_json(existing) != stable_json(protocol):
            raise RuntimeError("Existing architecture-tournament protocol differs from the current freeze")
    else:
        atomic_write_json(path, protocol)
    return protocol


def run_tournament(client: Any | None = None) -> dict[str, Any]:
    protocol = prepare_tournament()
    if client is None:
        require_api_key()
        client = OpenAI(max_retries=0, timeout=60.0)
    run_dir = Paths().artifacts / "architecture-tournament" / RUN_ID
    records: list[dict[str, Any]] = []
    record_by_id: dict[str, dict[str, Any]] = {}

    for slot in build_manifest():
        receipt_path = run_dir / "slots" / f"{slot.request_order:02d}__{slot.slot_id}.json"
        output = run_dir / "audio" / f"{slot.request_order:02d}__{slot.slot_id}.wav"
        if receipt_path.is_file():
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            if receipt.get("receipt_status") != "complete":
                raise RuntimeError(f"Ambiguous interrupted tournament slot: {slot.slot_id}")
            record = receipt["record"]
        else:
            if slot.reference_policy == "none":
                reference = None
            elif slot.reference_policy == "source_anchor":
                reference = record_by_id.get("source-anchor-1")
            else:
                reference = _selected_neutral(records, slot.architecture)
            if slot.reference_policy != "none" and (
                reference is None or reference.get("status") != "accepted"
            ):
                record = {
                    "slot": asdict(slot),
                    "status": "skipped_missing_eligible_dependency",
                    "reasons": [slot.reference_policy],
                    "usage": {},
                    "estimated_cost_usd": 0.0,
                    "attempts": [],
                }
            else:
                atomic_write_json(
                    receipt_path,
                    {"receipt_status": "started", "slot": asdict(slot)},
                )
                record = _render_slot(client, slot, protocol, reference, output)
            atomic_write_json(
                receipt_path,
                {"receipt_status": "complete", "record": record},
            )
        records.append(record)
        record_by_id[slot.slot_id] = record
        usage = summarize_usage(
            [attempt for item in records for attempt in item.get("attempts", [])]
        )
        if usage["estimated_cost_usd"] > MAX_ESTIMATED_COST_USD:
            raise RuntimeError("Architecture-tournament cost cap exceeded")
        print(
            f"tournament {slot.request_order}/9 {slot.slot_id}: {record['status']} "
            f"cost=${usage['estimated_cost_usd']:.4f}",
            flush=True,
        )

    attempts = [attempt for record in records for attempt in record.get("attempts", [])]
    usage = summarize_usage(attempts)
    selected = {
        architecture: (
            _selected_neutral(records, architecture) or {}
        ).get("slot", {}).get("slot_id")
        for architecture in ("per_word", "prosodic_group")
    }
    summary = {
        "schema_version": 1,
        "status": "render_complete",
        "run_id": RUN_ID,
        "protocol_sha256": protocol["protocol_sha256"],
        "logical_slots": len(build_manifest()),
        "api_attempts": len(attempts),
        "audio_returned": sum(bool(record.get("audio_sha256")) for record in records),
        "accepted": sum(record["status"] == "accepted" for record in records),
        "selected_neutral_slots": selected,
        "usage": usage,
    }
    _run_local_whisper_audit(records, run_dir, protocol)
    atomic_write_json(run_dir / "records.json", records)
    atomic_write_json(run_dir / "summary.json", summary)
    build_blind_review(records, run_dir)
    return summary


def _run_local_whisper_audit(
    records: Sequence[dict[str, Any]],
    run_dir: Path,
    protocol: dict[str, Any],
) -> None:
    import gc

    import mlx.core as mx
    import mlx_whisper

    model = Paths().whisper_cache / "large-v3-full"
    if not model.is_dir():
        raise RuntimeError("The pinned local Whisper large-v3 checkpoint is missing")
    for record in records:
        relative = record.get("audio_relative_path")
        if not relative:
            continue
        path = run_dir / relative
        script = _script_for(TournamentSlot(**record["slot"]), protocol)
        try:
            result = mlx_whisper.transcribe(
                str(path),
                path_or_hf_repo=str(model),
                temperature=0,
                condition_on_previous_text=False,
                verbose=False,
            )
            transcript = str(result.get("text") or "").strip()
            record["local_whisper"] = {
                "model": "whisper-large-v3-mlx-pinned",
                "detected_language": result.get("language"),
                "transcript": transcript,
                "comparison_to_script": asdict(check_transcript(script, transcript)),
                "segment_count": len(result.get("segments") or []),
            }
        except Exception as exc:
            error_type, error_detail = _safe_error(exc)
            record["local_whisper"] = {
                "model": "whisper-large-v3-mlx-pinned",
                "error_type": error_type,
                "error_detail": error_detail,
            }
        finally:
            mx.clear_cache()
            gc.collect()


def build_blind_review(records: Sequence[dict[str, Any]], run_dir: Path) -> Path:
    playable = [
        record for record in records
        if record.get("audio_relative_path") and record["slot"]["condition"] != "anchor"
    ]
    rng = random.Random(f"{RUN_ID}-blind-review-v1")
    randomized = list(playable)
    rng.shuffle(randomized)
    rows = [
        {
            "blind_id": f"clip-{index + 1:02d}",
            "audio": record["audio_relative_path"],
            "slot_id": record["slot"]["slot_id"],
        }
        for index, record in enumerate(randomized)
    ]
    key = {row["blind_id"]: row["slot_id"] for row in rows}
    atomic_write_json(run_dir / "blind-key.json", key)
    public_rows = [{"blind_id": row["blind_id"], "audio": row["audio"]} for row in rows]
    html = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Carrier architecture blind review</title><style>
body{font:16px/1.45 system-ui,sans-serif;max-width:820px;margin:0 auto;padding:24px;background:#f5f2e9;color:#18211c}h1{font-size:2rem}.card{background:#fff;border:1px solid #d8d5cb;border-radius:16px;padding:18px;margin:16px 0}audio{width:100%;margin:10px 0}label{display:block;margin:8px 0}select,textarea{font:inherit}textarea{width:100%;min-height:64px}.hint{color:#526158}button{font:inherit;padding:10px 16px;border-radius:999px;border:0;background:#154f3e;color:white;font-weight:700}</style></head><body><h1>Blind carrier-architecture review</h1><p>Rate the sound before trying to infer the condition. Names, scripts, spellings, and architecture labels are hidden. Listen on any playback setup you trust.</p><p class="hint"><b>Naturalness:</b> 1 unusably robotic/listed · 3 understandable but noticeably performed · 5 one fluent utterance. <b>List-like:</b> dominant means token-by-token timing or repeated mini-cadences control the clip. <b>Connectedness:</b> 1 isolated chunks · 5 continuous speech motion.</p><div id="cards"></div><button id="download">Download ratings.csv</button><script>
const ROWS=__ROWS__;const state=JSON.parse(localStorage.getItem('carrier-tournament-ratings')||'{}');const save=(id,k,v)=>{state[id]??={};state[id][k]=v;localStorage.setItem('carrier-tournament-ratings',JSON.stringify(state));};const options=(id,key,values)=>`<select onchange="save('${id}','${key}',this.value)"><option value="">—</option>${values.map(v=>`<option ${state[id]?.[key]==v?'selected':''}>${v}</option>`).join('')}</select>`;document.getElementById('cards').innerHTML=ROWS.map(row=>`<article class="card"><h2>${row.blind_id}</h2><audio controls preload="none" src="${row.audio}"></audio><label>Naturalness ${options(row.blind_id,'naturalness',['1','2','3','4','5'])}</label><label>List-like delivery ${options(row.blind_id,'list_like',['none','slight','dominant'])}</label><label>Connected grouping ${options(row.blind_id,'connectedness',['1','2','3','4','5'])}</label><label>Sounds related to one coherent sentence rhythm? ${options(row.blind_id,'source_rhythm',['yes','partly','no','uncertain'])}</label><label>Any recognizable real English meaning? ${options(row.blind_id,'meaning_leak',['none','possible','clear'])}</label><label>Commentary, spelling-out, missing, or extra material? ${options(row.blind_id,'content_failure',['no','yes','uncertain'])}</label><textarea placeholder="Optional notes" oninput="save('${row.blind_id}','notes',this.value)">${state[row.blind_id]?.notes??''}</textarea></article>`).join('');document.getElementById('download').onclick=()=>{const fields=['blind_id','naturalness','list_like','connectedness','source_rhythm','meaning_leak','content_failure','notes'];const q=v=>`"${String(v??'').replaceAll('"','""')}"`;const lines=[fields.join(','),...ROWS.map(r=>fields.map(k=>q(k==='blind_id'?r.blind_id:state[r.blind_id]?.[k])).join(','))];const blob=new Blob([lines.join('\n')+'\n'],{type:'text/csv'});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='carrier-tournament-ratings.csv';a.click();URL.revokeObjectURL(a.href);};</script></body></html>"""
    html = html.replace("__ROWS__", json.dumps(public_rows, ensure_ascii=False))
    output = run_dir / "review.html"
    atomic_write_text(output, html)
    return output
