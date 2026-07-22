from __future__ import annotations

import base64
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from openai import OpenAI

from .acoustic_calibration import _usage_dict, estimated_cost_usd, summarize_usage
from .api import require_api_key
from .audio_conformance import (
    ConformanceSample,
    analyze_audio_timing,
    build_messages,
    check_transcript,
)
from .config import Paths
from .pcm import decode_pcm16_mono
from .sentence_pair_v2 import (
    DELIVERY,
    FORMAT,
    MODEL,
    PROMPT_PROTOCOL,
    RUN_ID,
    VOICE,
    SentenceSlot,
    build_manifest,
    protocol_record,
)
from .util import atomic_write_json, sha256_file, write_csv


EXPECTED_PROTOCOL_SHA256 = (
    "8e5803858a22ef5402bc9dc95a4dc3b6565ea29a7ecd9d89a31c6579471a2691"
)
APPROVAL_CAP_USD = 0.25
RETRY_BACKOFF_S = 1.0


def _safe_error(exc: Exception) -> tuple[str, str]:
    return type(exc).__name__, str(exc).replace("\n", " ")[:500]


def _is_retryable_external_failure(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    if status == 429 or isinstance(status, int) and status >= 500:
        return True
    return type(exc).__name__ in {
        "APIConnectionError",
        "APITimeoutError",
        "ConnectError",
        "ConnectTimeout",
        "ReadTimeout",
        "TimeoutException",
    }


def _attempt(
    *, client: Any, slot: SentenceSlot, attempt_number: int, audio_path: Path
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    started = time.monotonic()
    attempt: dict[str, Any] = {
        "attempt_number": attempt_number,
        "status": "failed",
        "request_id": "",
        "usage": {},
        "estimated_cost_usd": 0.0,
        "audio_returned": False,
        "retryable_external_failure": False,
    }
    final: dict[str, Any] | None = None
    partial = audio_path.with_suffix(".partial.wav")
    try:
        sample = ConformanceSample(
            sample_id=slot.slot_id,
            language="en",
            script=slot.script,
            delivery=DELIVERY,
        )
        completion = client.chat.completions.create(
            model=MODEL,
            modalities=["text", "audio"],
            audio={"voice": VOICE, "format": FORMAT},
            messages=build_messages(sample, PROMPT_PROTOCOL),
            store=False,
        )
        attempt["request_id"] = getattr(completion, "_request_id", None) or ""
        attempt["resolved_model"] = getattr(completion, "model", MODEL)
        attempt["usage"] = _usage_dict(completion)
        attempt["estimated_cost_usd"] = estimated_cost_usd(attempt["usage"])
        message = completion.choices[0].message
        audio = message.audio
        if audio is None or not audio.data:
            raise ValueError("gpt-audio-1.5 returned no audio payload")

        attempt["audio_returned"] = True
        transcript = getattr(audio, "transcript", "") or ""
        partial.parent.mkdir(parents=True, exist_ok=True)
        partial.write_bytes(base64.b64decode(audio.data, validate=True))
        partial.replace(audio_path)
        attempt["status"] = "audio_returned"

        final = {
            "request_order": slot.request_order,
            "slot": asdict(slot),
            "status": "audio_returned",
            "model": MODEL,
            "voice": VOICE,
            "format": FORMAT,
            "request_id": attempt["request_id"],
            "resolved_model": attempt.get("resolved_model", MODEL),
            "provider_transcript": transcript,
            "transcript_check": asdict(check_transcript(slot.script, transcript)),
            "audio_filename": audio_path.name,
            "audio_path": str(audio_path),
            "audio_sha256": sha256_file(audio_path),
            "usage": attempt["usage"],
            "estimated_cost_usd": attempt["estimated_cost_usd"],
            "integrity_ok": False,
        }
        try:
            decoded = decode_pcm16_mono(audio_path)
            timing = analyze_audio_timing(audio_path, intended_syllables=None)
            final["decoded_wav"] = decoded.metadata()
            final["timing"] = asdict(timing)
            final["integrity_ok"] = bool(
                decoded.sample_rate_hz == 24000
                and 0.5 <= decoded.duration_s <= 10.0
                and decoded.clipped_sample_count / max(1, decoded.decoded_sample_count)
                < 0.001
            )
        except Exception as exc:
            error_type, error_detail = _safe_error(exc)
            final["evidentiary_error_type"] = error_type
            final["evidentiary_error_detail"] = error_detail
    except Exception as exc:
        partial.unlink(missing_ok=True)
        error_type, error_detail = _safe_error(exc)
        attempt.update(
            {
                "error_type": error_type,
                "error_detail": error_detail,
                "retryable_external_failure": (
                    not attempt["audio_returned"] and _is_retryable_external_failure(exc)
                ),
            }
        )
    attempt["latency_ms"] = round((time.monotonic() - started) * 1000)
    return attempt, final


def render_slot(
    *, client: Any, slot: SentenceSlot, audio_path: Path, sleep: Any = time.sleep
) -> dict[str, Any]:
    attempts: list[dict[str, Any]] = []
    final: dict[str, Any] | None = None
    for attempt_number in (1, 2):
        attempt, final = _attempt(
            client=client,
            slot=slot,
            attempt_number=attempt_number,
            audio_path=audio_path,
        )
        attempts.append(attempt)
        if final is not None:
            break
        if not attempt["retryable_external_failure"] or attempt_number == 2:
            break
        sleep(RETRY_BACKOFF_S)

    if final is None:
        final = {
            "request_order": slot.request_order,
            "slot": asdict(slot),
            "status": "external_failure_unresolved"
            if attempts[-1]["retryable_external_failure"]
            else "failed_no_audio",
            "model": MODEL,
            "voice": VOICE,
            "format": FORMAT,
            "integrity_ok": False,
            "usage": {},
            "estimated_cost_usd": 0.0,
            "error_type": attempts[-1].get("error_type", ""),
            "error_detail": attempts[-1].get("error_detail", ""),
        }
    final["attempts"] = attempts
    final["attempt_count"] = len(attempts)
    final["external_failure_unresolved"] = bool(
        final["status"] == "external_failure_unresolved"
    )
    return final


def run_sentence_pair_v2(*, client: Any | None = None) -> dict[str, Any]:
    protocol = protocol_record()
    if protocol["protocol_sha256"] != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError("sentence-pair-v2 protocol hash does not match the amendment")
    if client is None:
        require_api_key()
        client = OpenAI(max_retries=0, timeout=60.0)

    run_dir = Paths().artifacts / "sentence-pair-v2" / RUN_ID
    manifest_path = run_dir / "manifest.json"
    if manifest_path.is_file():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != protocol:
            raise RuntimeError("Existing sentence-pair-v2 manifest differs from freeze")
    else:
        if run_dir.exists() and any(run_dir.iterdir()):
            raise RuntimeError("Run directory exists without the frozen manifest")
        atomic_write_json(manifest_path, protocol)

    records: list[dict[str, Any]] = []
    for slot in build_manifest():
        receipt_path = run_dir / "slots" / f"{slot.request_order:03d}__{slot.slot_id}.json"
        audio_path = run_dir / "audio" / f"{slot.request_order:03d}__{slot.slot_id}.wav"
        if receipt_path.is_file():
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            if receipt.get("status") != "complete":
                raise RuntimeError(f"Ambiguous interrupted slot: {slot.slot_id}")
            record = receipt["record"]
        else:
            atomic_write_json(
                receipt_path,
                {"status": "started", "slot": asdict(slot), "attempts": []},
            )
            record = render_slot(client=client, slot=slot, audio_path=audio_path)
            atomic_write_json(receipt_path, {"status": "complete", "record": record})
        records.append(record)
        usage = summarize_usage(
            [attempt for item in records for attempt in item.get("attempts", [])]
        )
        if usage["estimated_cost_usd"] > APPROVAL_CAP_USD:
            raise RuntimeError("sentence-pair-v2 approval cap exceeded")
        print(
            f"sentence-pair-v2 {slot.request_order:02d}/24 {slot.slot_id}: "
            f"{record['status']} attempts={record['attempt_count']} "
            f"exact={bool((record.get('transcript_check') or {}).get('exact_token_match'))} "
            f"cost=${usage['estimated_cost_usd']:.4f}",
            flush=True,
        )

    attempts = [attempt for record in records for attempt in record.get("attempts", [])]
    usage = summarize_usage(attempts)
    summary = {
        "schema_version": 1,
        "status": "render_complete",
        "run_id": RUN_ID,
        "protocol_sha256": protocol["protocol_sha256"],
        "logical_slots": 24,
        "api_attempts": len(attempts),
        "successful_audio_responses": sum(
            record["status"] == "audio_returned" for record in records
        ),
        "exact_transcripts": sum(
            bool((record.get("transcript_check") or {}).get("exact_token_match"))
            for record in records
        ),
        "integrity_passes": sum(bool(record.get("integrity_ok")) for record in records),
        "unresolved_external_failures": sum(
            bool(record.get("external_failure_unresolved")) for record in records
        ),
        "usage": usage,
        "manifest_path": str(manifest_path),
    }
    atomic_write_json(run_dir / "render-records.json", records)
    atomic_write_json(run_dir / "render-summary.json", summary)
    return summary
