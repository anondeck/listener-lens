from __future__ import annotations

import base64
import gc
import json
import os
import socket
import time
import uuid
import wave
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import mlx.core as mx
import mlx_whisper
import numpy as np
import websocket
from openai import OpenAI

from .acoustic_calibration import _usage_dict, estimated_cost_usd
from .api import require_api_key
from .audio_delexicalization_probe import _content_audit
from .carrier_architecture_tournament import (
    GROUPED_TRANSFER_PROMPT,
    PROFILE_ID,
    REFERENCE_GATE,
    SOURCE_SYLLABLES,
    SOURCE_TEXT,
    compare_prosody,
)
from .config import Paths, sha256_json, stable_json
from .listener_lens import EspeakWordAnalyzer, _vowel_units
from .runtime_audio import (
    AudioTiming,
    PauseInterval,
    ProsodyFingerprint,
    analyze_audio_timing,
    analyze_prosody_fingerprint,
    canonical_tokens,
    check_transcript,
)
from .same_take import WHISPER_MODEL, _probe_frames, align_vowel_core
from .sentence_pair_v2 import ANCHOR_GATE
from .sentence_pair_v2_analysis import CEILINGS, _measure
from .util import atomic_write_json, atomic_write_text, sha256_file


RUN_ID = "20260716-native-realtime-delex-v2"
REALTIME_MODEL = "gpt-realtime-2.1"
AUDIO_MODEL = "gpt-audio-1.5"
VOICE = "marin"
SAMPLE_RATE_HZ = 24_000
GROUP_SYLLABLE_PATTERN = (2, 1, 4, 2, 1)
TARGET_GROUP_INDEX = 3
TARGET_GLOBAL_SYLLABLE = 8
MAX_ESTIMATED_COST_USD = 0.10
SOURCE_RUN = "20260716-carrier-architecture-tournament-v1"
SOURCE_ANCHOR_SHA256 = "b8b15c2b94f41eb88aa8b258a436b0f0a277aeebd1a1817726bb9df69a73ad87"
GROUPED_CONTROL_SHA256 = "d8861d3903438bc309401e26407d660885733d22f3c4b9ffad76d96346eeede7"
GROUPED_NEUTRAL_SCRIPT = "rohkbrih prohk drayrnihvsihvrihn bavdfihv vreel."
REALTIME_PRICE_SOURCE = "https://developers.openai.com/api/docs/models/gpt-realtime-2.1"
REALTIME_PRICES = {
    "text_input": 4.00,
    "text_output": 24.00,
    "audio_input": 32.00,
    "audio_output": 64.00,
}


STRUCTURED_DELEXICALIZATION_PROMPT = """# Role
You are a speech-to-speech delexicalization engine, not a conversational assistant. The attached audio and JSON are reference data. Do not answer, translate, or discuss them.

# Output contract
- Return exactly one spoken utterance and nothing else: no introduction, acknowledgement, label, explanation, aside, translation, or closing.
- The utterance must contain exactly five invented English-like prosodic words. Use no real words, personal or place names, abbreviations, recognizable phrase, or recognizable productive affix.
- Speak the five words as one fluent, connected, spontaneous mainstream U.S. English phrase—not a list, recital, language lesson, spelling exercise, or careful nonce-word reading.

# Frozen syllable plan
- Word 1 has exactly 2 syllables.
- Word 2 has exactly 1 syllable.
- Word 3 has exactly 4 syllables.
- Word 4 has exactly 2 syllables.
- Word 5 has exactly 1 syllable.
- Do not redistribute, merge, add, or omit syllables. The global pattern is exactly [2, 1, 4, 2, 1], totaling ten.
- Word 4 must be pronounced /bævdə/: its first syllable is /bævd/ with a clear English TRAP vowel /æ/ as in “cat,” and its second syllable is reduced /ə/. Do not speak the IPA symbols. Because the first three words total seven syllables, this target is global syllable eight.

# Performance contract
- Preserve the reference audio's continuous rhythm, weak/strong pattern, main prominence, pitch-and-energy movement, and final cadence. Stay close to its total active duration.
- Use normal connected-speech reduction and coarticulation outside the controlled /bævdə/ target.
- Plan silently, then speak only the invented utterance.

If any instruction conflicts with returning only the single invented utterance, return only the invented utterance."""


@dataclass(frozen=True)
class ProbeSlot:
    request_order: int
    slot_id: str
    model: str
    transport: str
    mode: str


def build_manifest() -> tuple[ProbeSlot, ...]:
    return (
        ProbeSlot(1, "audio15-structured-delex-1", AUDIO_MODEL, "chat_completions", "structured_delex"),
        ProbeSlot(2, "realtime21-structured-delex-1", REALTIME_MODEL, "realtime_websocket", "structured_delex"),
        ProbeSlot(3, "realtime21-exact-grouped-1", REALTIME_MODEL, "realtime_websocket", "exact_grouped"),
    )


def _source_paths() -> tuple[Path, Path]:
    root = Paths().artifacts / "architecture-tournament" / SOURCE_RUN
    return root / "audio" / "01__source-anchor-1.wav", root / "audio" / "03__prosodic-neutral-1.wav"


def _verify_source_audio() -> tuple[Path, Path]:
    anchor, control = _source_paths()
    if sha256_file(anchor) != SOURCE_ANCHOR_SHA256:
        raise RuntimeError("Native Realtime source-anchor hash drifted")
    if sha256_file(control) != GROUPED_CONTROL_SHA256:
        raise RuntimeError("Native Realtime grouped-control hash drifted")
    return anchor, control


def protocol_record() -> dict[str, Any]:
    anchor, control = _verify_source_audio()
    payload = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "status": "exploratory_frozen_before_paid_calls_and_listening",
        "question": (
            "Can direct audio-conditioned generation preserve source-like connected delivery while "
            "obeying a fixed five-group syllable plan and placing a measurable /ae/ target at slot eight?"
        ),
        "source_text": SOURCE_TEXT,
        "profile_id": PROFILE_ID,
        "source_syllables": SOURCE_SYLLABLES,
        "voice": VOICE,
        "source_bindings": {
            "anchor_path": str(anchor),
            "anchor_sha256": SOURCE_ANCHOR_SHA256,
            "audio15_exact_grouped_control_path": str(control),
            "audio15_exact_grouped_control_sha256": GROUPED_CONTROL_SHA256,
        },
        "manifest": [asdict(slot) for slot in build_manifest()],
        "prompts": {
            "structured_delex": STRUCTURED_DELEXICALIZATION_PROMPT,
            "exact_grouped": GROUPED_TRANSFER_PROMPT,
        },
        "request_payloads": {
            "structured_delex": _slot_prompt(build_manifest()[0])[1],
            "exact_grouped": _slot_prompt(build_manifest()[2])[1],
        },
        "realtime_transport": {
            "url": f"wss://api.openai.com/v1/realtime?model={REALTIME_MODEL}",
            "session_type": "realtime",
            "output_modalities": ["audio"],
            "input_format": {"type": "audio/pcm", "rate": SAMPLE_RATE_HZ},
            "turn_detection": None,
            "output_format": {"type": "audio/pcm"},
            "voice": VOICE,
            "event_sequence": [
                "session.update",
                "conversation.item.create(input_text,input_audio)",
                "response.create",
            ],
        },
        "structured_content_gate": {
            "provider_word_count": 5,
            "predicted_group_syllables": list(GROUP_SYLLABLE_PATTERN),
            "predicted_total_syllables": SOURCE_SYLLABLES,
            "source_word_overlap": 0,
            "written_word_predicted_homophone_and_adjacency": "all_pass",
            "target_group_index_zero_based": TARGET_GROUP_INDEX,
            "target_global_syllable": TARGET_GLOBAL_SYLLABLE,
            "target_predicted_first_vowel": "ae",
            "local_whisper_audio_leak_screen": (
                "all recognized tokens must independently pass the same written-word, "
                "predicted-homophone, adjacency, and source-overlap checks"
            ),
        },
        "acoustic_gate": {
            "instrument": "frozen standalone-Praat sentence-pair-v2 complete analysis family",
            "ceilings_hz": list(CEILINGS),
            "alignment": "Whisper five-word alignment; word 4, voiced peak in first 58%",
            "target": "neutral /ae/ closer to the frozen Marin /ae/ anchor than /eh/ in every family",
            "note": "This gate qualifies a neutral starting category; it does not construct a lens pair.",
        },
        "audio_gate": {
            "sample_rate_hz": SAMPLE_RATE_HZ,
            "duration_s": [0.25, 45.0],
            "maximum_clipped_fraction": 0.001,
            "reference_gate": REFERENCE_GATE,
        },
        "selection": "none; every valid returned slot is final and analyzed",
        "retry_policy": (
            "one same-slot retry only for 429, 5xx, timeout, connection failure, "
            "or a completed transport returning no audio; quality or gate failure is never retried"
        ),
        "limits": {
            "logical_slots": 3,
            "maximum_successful_audio_returns": 3,
            "maximum_estimated_cost_usd": MAX_ESTIMATED_COST_USD,
        },
        "interpretation": (
            "A passing structured neutral is a candidate architecture for a separate paired /ae/->/eh/ "
            "experiment. It does not establish listener recategorization, sentence-level lens validity, "
            "or arbitrary-input reliability. Realtime and Chat transports are compared descriptively; "
            "one take per cell cannot establish a model-level quality difference."
        ),
        "official_sources": {
            "realtime_websocket": "https://developers.openai.com/api/docs/guides/realtime-websocket",
            "realtime_conversations": "https://developers.openai.com/api/docs/guides/realtime-conversations",
            "realtime_model": REALTIME_PRICE_SOURCE,
        },
    }
    return {**payload, "protocol_sha256": sha256_json(payload)}


def prepare_probe() -> dict[str, Any]:
    protocol = protocol_record()
    path = Paths().artifacts / "native-realtime" / RUN_ID / "protocol.json"
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if stable_json(existing) != stable_json(protocol):
            raise RuntimeError("Existing native-Realtime protocol differs from the freeze")
    else:
        atomic_write_json(path, protocol)
    return protocol


def _pcm16_from_wav(path: Path) -> bytes:
    with wave.open(str(path), "rb") as handle:
        if (handle.getnchannels(), handle.getsampwidth(), handle.getframerate()) != (1, 2, SAMPLE_RATE_HZ):
            raise RuntimeError("Realtime source must be mono 24 kHz PCM16")
        return handle.readframes(handle.getnframes())


def _write_pcm16_wav(path: Path, pcm: bytes) -> None:
    if not pcm or len(pcm) % 2:
        raise RuntimeError("Realtime returned empty or sample-misaligned PCM16")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".partial.wav")
    with wave.open(str(partial), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE_HZ)
        handle.writeframes(pcm)
    partial.replace(path)


def _reference_record(path: Path) -> dict[str, Any]:
    return {
        "timing": asdict(analyze_audio_timing(path, intended_syllables=SOURCE_SYLLABLES)),
        "prosody": asdict(analyze_prosody_fingerprint(path)),
    }


def _measurements(record: dict[str, Any]) -> tuple[AudioTiming, ProsodyFingerprint]:
    timing = dict(record["timing"])
    timing["interior_pauses"] = tuple(PauseInterval(**item) for item in timing["interior_pauses"])
    prosody = dict(record["prosody"])
    prosody["energy_contour_db"] = tuple(prosody["energy_contour_db"])
    prosody["pitch_contour_semitones"] = tuple(prosody["pitch_contour_semitones"])
    return AudioTiming(**timing), ProsodyFingerprint(**prosody)


def _slot_prompt(slot: ProbeSlot) -> tuple[str, dict[str, Any]]:
    if slot.mode == "exact_grouped":
        return GROUPED_TRANSFER_PROMPT, {
            "task": "verbatim_prosody_transfer",
            "script": GROUPED_NEUTRAL_SCRIPT,
            "condition": "neutral",
            "reference_policy": "source_anchor",
            "flow_plan": {"carrier_word_count": 5, "group_syllables": list(GROUP_SYLLABLE_PATTERN)},
        }
    return STRUCTURED_DELEXICALIZATION_PROMPT, {
        "task": "structured_audio_delexicalization",
        "source_syllable_count": SOURCE_SYLLABLES,
        "output_word_count": 5,
        "group_syllables": list(GROUP_SYLLABLE_PATTERN),
        "target": {
            "word_number": 4,
            "global_syllable": 8,
            "phonetic_target": "first syllable /bævd/; second syllable reduced /ə/",
        },
        "output": "one fluent meaning-opaque spoken utterance only",
    }


def _chat_messages(slot: ProbeSlot, wav_base64: str) -> list[dict[str, Any]]:
    prompt, data = _slot_prompt(slot)
    return [
        {"role": "developer", "content": prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": json.dumps(data, separators=(",", ":"))},
                {"type": "input_audio", "input_audio": {"data": wav_base64, "format": "wav"}},
            ],
        },
    ]


def _realtime_usage(usage: dict[str, Any]) -> dict[str, Any]:
    input_details = usage.get("input_token_details") or {}
    output_details = usage.get("output_token_details") or {}
    return {
        "input_tokens": int(usage.get("input_tokens") or 0),
        "output_tokens": int(usage.get("output_tokens") or 0),
        "input_text_tokens": int(input_details.get("text_tokens") or 0),
        "input_audio_tokens": int(input_details.get("audio_tokens") or 0),
        "input_cached_tokens": int(input_details.get("cached_tokens") or 0),
        "output_text_tokens": int(output_details.get("text_tokens") or 0),
        "output_audio_tokens": int(output_details.get("audio_tokens") or 0),
    }


def _realtime_cost(usage: dict[str, Any]) -> float:
    u = _realtime_usage(usage)
    cost = (
        u["input_text_tokens"] * REALTIME_PRICES["text_input"]
        + u["input_audio_tokens"] * REALTIME_PRICES["audio_input"]
        + u["output_text_tokens"] * REALTIME_PRICES["text_output"]
        + u["output_audio_tokens"] * REALTIME_PRICES["audio_output"]
    ) / 1_000_000
    return round(cost, 8)


def _extract_transcript(event: dict[str, Any]) -> str:
    direct = event.get("transcript")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()
    response = event.get("response") or {}
    for item in response.get("output") or []:
        for content in item.get("content") or []:
            value = content.get("transcript") or content.get("text")
            if isinstance(value, str) and value.strip():
                return value.strip()
    return ""


def _safe_realtime_event(event: dict[str, Any]) -> dict[str, Any]:
    safe = {"type": str(event.get("type") or "")}
    for key in ("event_id", "item_id", "response_id"):
        if event.get(key):
            safe[key] = event[key]
    if safe["type"] == "error":
        error = event.get("error") or {}
        safe["error"] = {
            key: error.get(key) for key in ("type", "code", "message", "param", "event_id") if error.get(key) is not None
        }
    return safe


class RealtimeRequestError(RuntimeError):
    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable


def native_realtime_request(
    *,
    slot: ProbeSlot,
    input_pcm16: bytes,
    api_key: str,
    output_rate_hz: int | None = None,
    connect: Callable[..., Any] = websocket.create_connection,
) -> dict[str, Any]:
    prompt, data = _slot_prompt(slot)
    url = f"wss://api.openai.com/v1/realtime?model={REALTIME_MODEL}"
    ws: Any | None = None
    audio_chunks: list[bytes] = []
    transcript = ""
    event_log: list[dict[str, Any]] = []
    session_id = ""
    response_id = ""
    usage: dict[str, Any] = {}
    try:
        ws = connect(
            url,
            header=[f"Authorization: Bearer {api_key}"],
            timeout=75,
            enable_multithread=False,
        )
        created = json.loads(ws.recv())
        event_log.append(_safe_realtime_event(created))
        if created.get("type") == "error":
            raise RealtimeRequestError(str((created.get("error") or {}).get("message") or "Realtime session error"))
        if created.get("type") != "session.created":
            raise RealtimeRequestError(f"Expected session.created, got {created.get('type')!r}")
        session_id = str((created.get("session") or {}).get("id") or "")

        session_event_id = f"evt_{uuid.uuid4().hex}"
        output_format: dict[str, Any] = {"type": "audio/pcm"}
        if output_rate_hz is not None:
            output_format["rate"] = output_rate_hz
        ws.send(json.dumps({
            "event_id": session_event_id,
            "type": "session.update",
            "session": {
                "type": "realtime",
                "model": REALTIME_MODEL,
                "output_modalities": ["audio"],
                "instructions": prompt,
                "audio": {
                    "input": {
                        "format": {"type": "audio/pcm", "rate": SAMPLE_RATE_HZ},
                        "turn_detection": None,
                    },
                    "output": {"format": output_format, "voice": VOICE},
                },
            },
        }))
        while True:
            event = json.loads(ws.recv())
            event_log.append(_safe_realtime_event(event))
            if event.get("type") == "error":
                raise RealtimeRequestError(str((event.get("error") or {}).get("message") or "Realtime session update error"))
            if event.get("type") == "session.updated":
                break

        ws.send(json.dumps({
            "event_id": f"evt_{uuid.uuid4().hex}",
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": json.dumps(data, separators=(",", ":"))},
                    {"type": "input_audio", "audio": base64.b64encode(input_pcm16).decode("ascii")},
                ],
            },
        }))
        ws.send(json.dumps({
            "event_id": f"evt_{uuid.uuid4().hex}",
            "type": "response.create",
            "response": {"output_modalities": ["audio"]},
        }))

        while True:
            event = json.loads(ws.recv())
            event_type = str(event.get("type") or "")
            event_log.append(_safe_realtime_event(event))
            if event_type in {"response.output_audio.delta", "response.audio.delta"}:
                audio_chunks.append(base64.b64decode(event.get("delta") or "", validate=True))
            elif event_type in {"response.output_audio_transcript.done", "response.audio_transcript.done"}:
                transcript = _extract_transcript(event) or transcript
            elif event_type == "error":
                raise RealtimeRequestError(str((event.get("error") or {}).get("message") or "Realtime response error"))
            elif event_type == "response.done":
                response = event.get("response") or {}
                response_id = str(response.get("id") or "")
                usage = response.get("usage") or {}
                transcript = transcript or _extract_transcript(event)
                status = str(response.get("status") or "")
                if status and status not in {"completed", "incomplete"}:
                    raise RealtimeRequestError(f"Realtime response ended with status {status!r}")
                break
        headers = ws.getheaders() if hasattr(ws, "getheaders") else {}
        request_id = ""
        if isinstance(headers, dict):
            request_id = str(headers.get("x-request-id") or headers.get("X-Request-ID") or "")
        return {
            "pcm16": b"".join(audio_chunks),
            "provider_transcript": transcript,
            "usage": usage,
            "estimated_cost_usd": _realtime_cost(usage),
            "request_id": request_id,
            "session_id": session_id,
            "response_id": response_id,
            "event_log": event_log,
        }
    except (socket.timeout, TimeoutError, websocket.WebSocketTimeoutException, websocket.WebSocketConnectionClosedException) as exc:
        raise RealtimeRequestError(f"{type(exc).__name__}: {exc}", retryable=True) from exc
    except (OSError, websocket.WebSocketException) as exc:
        raise RealtimeRequestError(f"{type(exc).__name__}: {exc}", retryable=True) from exc
    finally:
        if ws is not None:
            try:
                ws.close()
            except Exception:
                pass


def _token_pattern_audit(transcript: str) -> dict[str, Any]:
    base = _content_audit(transcript)
    analyzer = EspeakWordAnalyzer()
    tokens = canonical_tokens(transcript)
    ipas = analyzer.phonemize_words(tokens, "en-us") if tokens else []
    vowel_units = [_vowel_units(ipa) for ipa in ipas]
    counts = [len(item) for item in vowel_units]
    target_vowels = vowel_units[TARGET_GROUP_INDEX] if len(vowel_units) > TARGET_GROUP_INDEX else []
    target_first = target_vowels[0] if target_vowels else ""
    target_global_position = sum(counts[:TARGET_GROUP_INDEX]) + 1 if len(counts) > TARGET_GROUP_INDEX else None
    pattern_pass = counts == list(GROUP_SYLLABLE_PATTERN)
    target_pass = target_first == "æ" and target_global_position == TARGET_GLOBAL_SYLLABLE
    return {
        **base,
        "predicted_ipa_by_token": ipas,
        "predicted_vowels_by_token": vowel_units,
        "predicted_group_syllables": counts,
        "group_pattern_pass": pattern_pass,
        "target_predicted_first_vowel": target_first,
        "target_predicted_global_syllable": target_global_position,
        "target_position_pass": target_pass,
        "structured_contract_pass": bool(
            base["dictionary_homophone_adjacency_pass"]
            and base["source_overlap_pass"]
            and base["ascii_nonce_shape"]
            and len(tokens) == 5
            and pattern_pass
            and target_pass
        ),
    }


def _integrity_and_prosody(
    output: Path, anchor_record: dict[str, Any]
) -> dict[str, Any]:
    timing = analyze_audio_timing(output, intended_syllables=SOURCE_SYLLABLES)
    prosody = analyze_prosody_fingerprint(output)
    reasons: list[str] = []
    if timing.sample_rate_hz != SAMPLE_RATE_HZ:
        reasons.append("unexpected_sample_rate")
    if not 0.25 <= timing.duration_s <= 45.0:
        reasons.append("duration_out_of_bounds")
    if timing.utterance_duration_s <= 0:
        reasons.append("no_detectable_utterance")
    if timing.clipped_fraction > 0.001:
        reasons.append("excessive_clipping")
    reference_timing, reference_prosody = _measurements(anchor_record)
    return {
        "integrity_pass": not reasons,
        "integrity_reasons": reasons,
        "timing": asdict(timing),
        "prosody": asdict(prosody),
        "reference_match": compare_prosody(reference_timing, reference_prosody, timing, prosody),
    }


def _render_chat(
    client: Any,
    slot: ProbeSlot,
    anchor_wav_base64: str,
    output: Path,
) -> dict[str, Any]:
    completion = client.chat.completions.create(
        model=slot.model,
        modalities=["text", "audio"],
        audio={"voice": VOICE, "format": "wav"},
        messages=_chat_messages(slot, anchor_wav_base64),
        store=False,
    )
    audio = completion.choices[0].message.audio
    if audio is None or not audio.data:
        raise RuntimeError("Chat Audio returned no audio payload")
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(".partial.wav")
    partial.write_bytes(base64.b64decode(audio.data, validate=True))
    partial.replace(output)
    usage = _usage_dict(completion)
    return {
        "provider_transcript": getattr(audio, "transcript", "") or "",
        "usage": usage,
        "estimated_cost_usd": estimated_cost_usd(usage),
        "request_id": getattr(completion, "_request_id", None) or "",
        "resolved_model": getattr(completion, "model", slot.model),
    }


def _retryable_exception(exc: Exception) -> bool:
    if isinstance(exc, RealtimeRequestError):
        return exc.retryable
    status = getattr(exc, "status_code", None)
    return bool(
        status == 429
        or isinstance(status, int) and status >= 500
        or type(exc).__name__ in {
            "APIConnectionError", "APITimeoutError", "ConnectError", "ConnectTimeout",
            "ReadTimeout", "TimeoutException",
        }
    )


def _safe_error(exc: Exception) -> tuple[str, str]:
    return type(exc).__name__, str(exc).replace("\n", " ")[:500]


def _render_slot(
    *,
    client: Any,
    slot: ProbeSlot,
    anchor_pcm16: bytes,
    anchor_wav_base64: str,
    anchor_record: dict[str, Any],
    api_key: str,
    output: Path,
    realtime_output_rate_hz: int | None = None,
) -> dict[str, Any]:
    attempts: list[dict[str, Any]] = []
    final: dict[str, Any] | None = None
    for attempt_number in (1, 2):
        started = time.monotonic()
        attempt: dict[str, Any] = {
            "attempt_number": attempt_number,
            "status": "failed_no_audio",
            "usage": {},
            "estimated_cost_usd": 0.0,
            "retryable_external_failure": False,
        }
        try:
            if slot.transport == "realtime_websocket":
                response = native_realtime_request(
                    slot=slot,
                    input_pcm16=anchor_pcm16,
                    api_key=api_key,
                    output_rate_hz=realtime_output_rate_hz,
                )
                if not response["pcm16"]:
                    raise RealtimeRequestError("Realtime completed without audio", retryable=True)
                _write_pcm16_wav(output, response.pop("pcm16"))
                result = {**response, "resolved_model": slot.model}
            else:
                result = _render_chat(client, slot, anchor_wav_base64, output)
            attempt.update({
                "status": "audio_returned",
                "request_id": result.get("request_id", ""),
                "response_id": result.get("response_id", ""),
                "session_id": result.get("session_id", ""),
                "usage": result.get("usage") or {},
                "estimated_cost_usd": result.get("estimated_cost_usd") or 0.0,
            })
            final = {
                "slot": asdict(slot),
                "status": "audio_returned_analysis_pending",
                "provider_transcript": str(result.get("provider_transcript") or "").strip(),
                "content_control_pass": False,
                "audio_relative_path": str(output.relative_to(output.parents[1])),
                "audio_sha256": sha256_file(output),
                "request_id": result.get("request_id", ""),
                "response_id": result.get("response_id", ""),
                "session_id": result.get("session_id", ""),
                "resolved_model": result.get("resolved_model", slot.model),
                "event_log": result.get("event_log", []),
                "usage": result.get("usage") or {},
                "estimated_cost_usd": result.get("estimated_cost_usd") or 0.0,
            }
            # Once a valid audio payload has been returned, transcript or evidence
            # failure must never trigger a replacement call. Keep the WAV and report
            # each downstream failure as an exclusion on this final slot.
            try:
                transcript = final["provider_transcript"]
                audit = (
                    asdict(check_transcript(GROUPED_NEUTRAL_SCRIPT, transcript))
                    if slot.mode == "exact_grouped"
                    else _token_pattern_audit(transcript)
                )
                final["transcript_audit"] = audit
                final["content_control_pass"] = bool(
                    audit.get("exact_token_match")
                    if slot.mode == "exact_grouped"
                    else audit.get("structured_contract_pass")
                )
            except Exception as exc:
                final["transcript_audit"] = {
                    "status": "excluded",
                    "reason": f"{type(exc).__name__}: {str(exc)[:300]}",
                }
            try:
                final.update(_integrity_and_prosody(output, anchor_record))
            except Exception as exc:
                final.update({
                    "integrity_pass": False,
                    "integrity_reasons": [f"{type(exc).__name__}: {str(exc)[:300]}"],
                    "reference_match": {"eligible": False, "reasons": ["analysis_error"]},
                })
            final["status"] = "analyzed"
        except Exception as exc:
            output.unlink(missing_ok=True)
            error_type, error_detail = _safe_error(exc)
            attempt.update({
                "error_type": error_type,
                "error_detail": error_detail,
                "retryable_external_failure": _retryable_exception(exc),
            })
        finally:
            attempt["latency_ms"] = round((time.monotonic() - started) * 1000)
            attempts.append(attempt)
        if final is not None or not attempt["retryable_external_failure"]:
            break
        time.sleep(1)
    if final is None:
        final = {
            "slot": asdict(slot),
            "status": "external_failure_unresolved" if attempts[-1]["retryable_external_failure"] else "failed_no_audio",
            "reasons": [attempts[-1].get("error_type", "unknown_error")],
            "usage": {},
            "estimated_cost_usd": 0.0,
        }
    final["attempts"] = attempts
    return final


def _whisper_and_acoustic_audit(record: dict[str, Any], run_dir: Path) -> None:
    relative = record.get("audio_relative_path")
    if not relative:
        return
    path = run_dir / relative
    try:
        prompt = str(record.get("provider_transcript") or "")
        result = mlx_whisper.transcribe(
            str(path),
            path_or_hf_repo=str(WHISPER_MODEL),
            language="en",
            temperature=0,
            condition_on_previous_text=False,
            word_timestamps=True,
            initial_prompt=prompt or None,
            verbose=False,
        )
        transcript = str(result.get("text") or "").strip()
        words = [
            {
                "label": str(word.get("word") or "").strip(),
                "start_s": float(word["start"]),
                "end_s": float(word["end"]),
                "probability": float(word.get("probability") or 0),
            }
            for segment in result.get("segments", [])
            for word in segment.get("words", [])
            if str(word.get("word") or "").strip()
        ]
        leak = _content_audit(transcript)
        record["local_whisper"] = {
            "detected_language": result.get("language"),
            "transcript": transcript,
            "tokens": canonical_tokens(transcript),
            "word_intervals": words,
            "leak_audit": leak,
            "audio_leak_screen_pass": bool(
                leak["dictionary_homophone_adjacency_pass"] and leak["source_overlap_pass"]
            ),
        }
        if record["slot"]["mode"] != "structured_delex":
            return
        acoustic: dict[str, Any] = {"status": "excluded", "exclusion_reasons": []}
        if len(words) != 5:
            acoustic["exclusion_reasons"].append(f"requires_five_word_intervals_got_{len(words)}")
            record["target_acoustic"] = acoustic
            return
        target_word = words[TARGET_GROUP_INDEX]
        with wave.open(str(path), "rb") as handle:
            sample_rate = handle.getframerate()
        core = align_vowel_core(
            _probe_frames(path),
            word_start_s=target_word["start_s"],
            word_end_s=target_word["end_s"],
            search_fraction=(0.05, 0.58),
            sample_rate_hz=sample_rate,
        )
        measurements = {str(ceiling): _measure(path, core, ceiling) for ceiling in CEILINGS}
        families: dict[str, Any] = {}
        for ceiling in CEILINGS:
            key = str(ceiling)
            point = np.asarray([measurements[key]["f1_bark"], measurements[key]["f2_bark"]])
            anchor = ANCHOR_GATE["families"][key]
            source = np.asarray(anchor["source_centroid_bark"])
            target = np.asarray(anchor["target_centroid_bark"])
            source_distance = float(np.linalg.norm(point - source))
            target_distance = float(np.linalg.norm(point - target))
            families[key] = {
                "source_distance_bark": source_distance,
                "target_distance_bark": target_distance,
                "source_category_pass": bool(
                    measurements[key]["plausibility_pass"] and source_distance < target_distance
                ),
            }
        acoustic.update({
            "status": "measurable",
            "target_word_interval": target_word,
            "vowel_core": core,
            "measurements": measurements,
            "families": families,
            "neutral_source_category_pass": all(item["source_category_pass"] for item in families.values()),
        })
        record["target_acoustic"] = acoustic
    except Exception as exc:
        record.setdefault("local_whisper", {})["error"] = f"{type(exc).__name__}: {str(exc)[:500]}"
        record.setdefault("target_acoustic", {"status": "excluded", "exclusion_reasons": []})[
            "exclusion_reasons"
        ].append(f"{type(exc).__name__}: {str(exc)[:300]}")
    finally:
        mx.clear_cache()
        gc.collect()


def _combined_cost(records: list[dict[str, Any]]) -> dict[str, Any]:
    attempts = [attempt for record in records for attempt in record.get("attempts", [])]
    total = round(sum(float(item.get("estimated_cost_usd") or 0) for item in attempts), 6)
    return {
        "estimated_cost_usd": total,
        "attempts_with_usage": sum(bool(item.get("usage")) for item in attempts),
        "per_transport": {
            "chat_completions": round(sum(
                float(record.get("estimated_cost_usd") or 0)
                for record in records if record["slot"]["transport"] == "chat_completions"
            ), 6),
            "realtime_websocket": round(sum(
                float(record.get("estimated_cost_usd") or 0)
                for record in records if record["slot"]["transport"] == "realtime_websocket"
            ), 6),
        },
        "price_sources": {
            "gpt_audio_1_5": "https://developers.openai.com/api/docs/models/gpt-audio-1.5",
            "gpt_realtime_2_1": REALTIME_PRICE_SOURCE,
        },
    }


def run_probe(client: Any | None = None) -> dict[str, Any]:
    protocol = prepare_probe()
    require_api_key()
    api_key = os.environ["OPENAI_API_KEY"].strip()
    client = client or OpenAI(max_retries=0, timeout=60.0)
    anchor, control = _verify_source_audio()
    anchor_pcm16 = _pcm16_from_wav(anchor)
    anchor_wav_base64 = base64.b64encode(anchor.read_bytes()).decode("ascii")
    anchor_record = _reference_record(anchor)
    run_dir = Paths().artifacts / "native-realtime" / RUN_ID
    records: list[dict[str, Any]] = []
    for slot in build_manifest():
        receipt = run_dir / "slots" / f"{slot.request_order:02d}__{slot.slot_id}.json"
        output = run_dir / "audio" / f"{slot.request_order:02d}__{slot.slot_id}.wav"
        if receipt.is_file():
            saved = json.loads(receipt.read_text(encoding="utf-8"))
            if saved.get("receipt_status") != "complete":
                raise RuntimeError(f"Ambiguous interrupted native-Realtime slot: {slot.slot_id}")
            record = saved["record"]
        else:
            atomic_write_json(receipt, {"receipt_status": "started", "slot": asdict(slot)})
            record = _render_slot(
                client=client,
                slot=slot,
                anchor_pcm16=anchor_pcm16,
                anchor_wav_base64=anchor_wav_base64,
                anchor_record=anchor_record,
                api_key=api_key,
                output=output,
            )
            atomic_write_json(receipt, {"receipt_status": "complete", "record": record})
        records.append(record)
        usage = _combined_cost(records)
        if usage["estimated_cost_usd"] > MAX_ESTIMATED_COST_USD:
            raise RuntimeError("Native-Realtime probe cost cap exceeded")
        print(
            f"native probe {slot.request_order}/3 {slot.slot_id}: {record['status']} "
            f"cost=${usage['estimated_cost_usd']:.4f}",
            flush=True,
        )

    for record in records:
        _whisper_and_acoustic_audit(record, run_dir)
    usage = _combined_cost(records)
    summary = {
        "schema_version": 1,
        "status": "probe_complete",
        "run_id": RUN_ID,
        "protocol_sha256": protocol["protocol_sha256"],
        "logical_slots": len(build_manifest()),
        "audio_returned": sum(bool(record.get("audio_sha256")) for record in records),
        "structured_content_passes": sum(
            bool(record.get("content_control_pass"))
            for record in records if record["slot"]["mode"] == "structured_delex"
        ),
        "audio_leak_screen_passes": sum(
            bool((record.get("local_whisper") or {}).get("audio_leak_screen_pass"))
            for record in records if record["slot"]["mode"] == "structured_delex"
        ),
        "neutral_source_category_passes": sum(
            bool((record.get("target_acoustic") or {}).get("neutral_source_category_pass"))
            for record in records if record["slot"]["mode"] == "structured_delex"
        ),
        "usage": usage,
    }
    atomic_write_json(run_dir / "records.json", records)
    atomic_write_json(run_dir / "summary.json", summary)
    _build_review(records, run_dir, control)
    return summary


def _build_review(records: list[dict[str, Any]], run_dir: Path, control: Path) -> None:
    rows = [{
        "blind_id": "clip-01",
        "audio": str(Path("../..") / control.relative_to(Paths().artifacts)),
        "key": "gpt-audio-1.5 exact grouped control",
    }]
    rows.extend({
        "blind_id": f"clip-{index + 2:02d}",
        "audio": record["audio_relative_path"],
        "key": record["slot"]["slot_id"],
    } for index, record in enumerate(records) if record.get("audio_relative_path"))
    import random
    random.Random(f"{RUN_ID}-blind-v1").shuffle(rows)
    atomic_write_json(run_dir / "blind-key.json", {row["blind_id"]: row["key"] for row in rows})
    public = [{"blind_id": row["blind_id"], "audio": row["audio"]} for row in rows]
    html = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Native audio architecture blind review</title><style>body{font:16px/1.5 system-ui;max-width:800px;margin:auto;padding:24px;background:#f5f2e9;color:#17221c}.card{background:white;padding:18px;border:1px solid #d6d3c9;border-radius:16px;margin:16px 0}audio,textarea{width:100%}label{display:block;margin:8px 0}button{padding:10px 16px;border:0;border-radius:99px;background:#154f3e;color:white;font-weight:700}</style></head><body><h1>Blind neutral-carrier architecture review</h1><p>Judge the audible result, not the hidden model. A good candidate is one continuous, natural but meaningless English-like utterance—not a list.</p><div id="cards"></div><button id="download">Download ratings.csv</button><script>const R=__ROWS__;const S=JSON.parse(localStorage.getItem('native-delex-ratings')||'{}');const save=(i,k,v)=>{S[i]??={};S[i][k]=v;localStorage.setItem('native-delex-ratings',JSON.stringify(S))};const sel=(i,k,v)=>`<select onchange="save('${i}','${k}',this.value)"><option value="">—</option>${v.map(x=>`<option ${S[i]?.[k]==x?'selected':''}>${x}</option>`).join('')}</select>`;document.getElementById('cards').innerHTML=R.map(r=>`<section class="card"><h2>${r.blind_id}</h2><audio controls src="${r.audio}"></audio><label>Naturalness ${sel(r.blind_id,'naturalness',['1','2','3','4','5'])}</label><label>List-like ${sel(r.blind_id,'list_like',['none','slight','dominant'])}</label><label>Meaning/name leakage ${sel(r.blind_id,'meaning_leak',['none','possible','clear'])}</label><label>Source-rhythm resemblance ${sel(r.blind_id,'source_rhythm',['yes','partly','no','uncertain'])}</label><label>Commentary or task failure ${sel(r.blind_id,'content_failure',['no','yes','uncertain'])}</label><textarea oninput="save('${r.blind_id}','notes',this.value)" placeholder="Notes">${S[r.blind_id]?.notes??''}</textarea></section>`).join('');document.getElementById('download').onclick=()=>{const F=['blind_id','naturalness','list_like','meaning_leak','source_rhythm','content_failure','notes'];const q=v=>`"${String(v??'').replaceAll('"','""')}"`;const b=new Blob([[F.join(','),...R.map(r=>F.map(k=>q(k==='blind_id'?r.blind_id:S[r.blind_id]?.[k])).join(','))].join('\n')+'\n'],{type:'text/csv'});const a=document.createElement('a');a.href=URL.createObjectURL(b);a.download='native-delex-ratings.csv';a.click()};</script></body></html>""".replace("__ROWS__", json.dumps(public))
    atomic_write_text(run_dir / "review.html", html)
