from __future__ import annotations

import base64
import csv
import hashlib
import json
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    OpenAI,
    RateLimitError,
)

from .acoustic_calibration import (
    FORMAT,
    MODEL,
    VOICE,
    _usage_dict,
    build_calibration_messages,
    classify_calibration,
    estimated_cost_usd,
    exclusion_reasons,
    summarize_usage,
)
from .acoustic_calibration_amendment import (
    AMENDMENT_PREREGISTRATION_HEADING,
    evaluate_candidate_inventory,
)
from .acoustic_calibration_v3 import (
    ConfirmationStimulus,
    RESULT_FIELDS as V3_RESULT_FIELDS,
    _existing_confirmation_references,
    _flatten_record,
    evaluate_internal_coherence,
)
from .acoustic_reaudit import (
    MAXIMUM_FORMANT_HZ,
    MAX_NUMBER_OF_FORMANTS,
    PRE_EMPHASIS_FROM_HZ,
    TIME_STEP_S,
    VOWEL_CENTER_END_FRACTION,
    VOWEL_CENTER_START_FRACTION,
    WINDOW_LENGTH_S,
    analyze_wav_praat,
)
from .audio_conformance import check_transcript
from .config import DEVLOG_PATH, Paths, stable_json
from .util import atomic_write_json, sha256_file, write_csv


RUN_ID = "20260715-calibration-v3-amended-confirmation"
MANIFEST_SEED = "calibration-v3-amended-confirmation-20260715"
SELECTED_SHELL = "b_V_vd"
SELECTED_TOKENS = {"ih": "bihvd", "ee": "beevd", "a": "bavd", "eh": "behvd"}
MAX_TRANSPORT_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = (1.0, 2.0)
EXPECTED_PROTOCOL_SHA256 = (
    "3c55d9d2c5d30a399829b7b0559f00ec2c9eadb62b59293d44f950d75a3a1814"
)

SOURCE_RUN_ID = "20260715-carrier-v3-calibration"
SOURCE_CONFIRMATION_RUN_ID = "20260715-calibration-v3-confirmatory"
SOURCE_HASHES = {
    "v2_source_manifest": "daca27fd60c43004f69ca35a762d884bdb72f9c78036f3a37f2bf6f4668825d9",
    "v2_praat_protocol": "585d9fa23a74565c7e027dee4edc6e34758feddc33d8dc3c1f17fe4d14df12d2",
    "v2_praat_results": "3f23d9e468a055607be71cc842a477e931edcaf2dc8a8ce6eb55dc695424e246",
    "v2_praat_summary": "0bbe54c8d40fca85e1bec6bdc58511726f138490d411a22fa7eee0bf08eefc28",
    "v3_manifest": "2f09e447cd27342a6d778d11f85c647e88fffd2a05a20d1c6f0837bad45967e0",
    "v3_results": "6e7a6fa449d32b98b92dd3ef952cb95d5c2bec2c8d8365a643f786a61489e400",
    "v3_analysis": "037b5982d4f1447fb078e9f117900fe99a0d53e4a9f3de4f26361049274071a3",
    "v3_summary": "434ffc1d7c07c973144d7eb8227db2c0dced90ed9ddd93f16bcf2918995279d9",
    "gate_database": "cae4b5c9545d1577e9c3ac5892824a9540b234836354fca820e52d0e00567697",
    "gate_receipt": "ad0a953c525bfbf87c05bd4d7968284798ee264f804105c48e31c577a3b528e4",
}

AMENDED_RULE_SPECS = (
    {
        "rule_id": "ptbr.vowel.ih_to_i",
        "source_category": "ih",
        "target_category": "i",
        "source_ipa": "ɪ",
        "target_ipa": "i",
        "tokens": (
            ("z_V_f", "zihf", "zeef"),
            ("v_V_p", "vihp", "veep"),
            (SELECTED_SHELL, SELECTED_TOKENS["ih"], SELECTED_TOKENS["ee"]),
        ),
    },
    {
        "rule_id": "ptbr.vowel.ae_to_eh",
        "source_category": "ae",
        "target_category": "eh",
        "source_ipa": "æ",
        "target_ipa": "ɛ",
        "tokens": (
            ("z_V_f", "zaf", "zehf"),
            ("v_V_p", "vap", "vehp"),
            (SELECTED_SHELL, SELECTED_TOKENS["a"], SELECTED_TOKENS["eh"]),
        ),
    },
)

RESULT_FIELDS = [*V3_RESULT_FIELDS, "attempt_count", "attempts_json"]

ATTEMPT_FIELDS = [
    "request_order",
    "slot_id",
    "attempt_number",
    "status",
    "retryable_transport_failure",
    "request_id",
    "http_status",
    "resolved_model",
    "latency_ms",
    "provider_transcript",
    "audio_sha256",
    "prompt_tokens",
    "prompt_audio_tokens",
    "completion_tokens",
    "completion_audio_tokens",
    "estimated_request_cost_usd",
    "error_type",
    "error_detail",
]


def build_amendment_manifest() -> tuple[ConfirmationStimulus, ...]:
    stimuli: list[ConfirmationStimulus] = []
    specs = (
        (
            "ptbr.vowel.ih_to_i",
            SELECTED_TOKENS["ih"],
            SELECTED_TOKENS["ee"],
            (1, 3),
            (1, 3),
        ),
        (
            "ptbr.vowel.ae_to_eh",
            SELECTED_TOKENS["a"],
            SELECTED_TOKENS["eh"],
            (1, 2),
            (1, 3),
        ),
    )
    for rule_id, neutral, lens, neutral_span, lens_span in specs:
        rule_slug = rule_id.removeprefix("ptbr.vowel.")
        for side, token in (("neutral", neutral), ("lens", lens)):
            for take in (1, 2, 3):
                stimuli.append(
                    ConfirmationStimulus(
                        slot_id=(
                            f"contrast__{rule_slug}__{SELECTED_SHELL}__"
                            f"{side}__take-{take}"
                        ),
                        kind="contrast",
                        token=token,
                        take=take,
                        rule_id=rule_id,
                        shell=SELECTED_SHELL,
                        side=side,  # type: ignore[arg-type]
                        neutral_character_span=neutral_span,
                        lens_character_span=lens_span,
                    )
                )
    if len(stimuli) != 12 or len({item.slot_id for item in stimuli}) != 12:
        raise AssertionError("The amended confirmation requires exactly 12 slots")
    random.Random(MANIFEST_SEED).shuffle(stimuli)
    return tuple(stimuli)


def candidate_gate_audit_record() -> dict[str, Any]:
    paths = Paths()
    gate_receipt = paths.gate_db.with_suffix(".receipt.json")
    if sha256_file(paths.gate_db) != SOURCE_HASHES["gate_database"]:
        raise RuntimeError("Pinned gate database hash changed")
    if sha256_file(gate_receipt) != SOURCE_HASHES["gate_receipt"]:
        raise RuntimeError("Pinned gate receipt hash changed")
    audit = evaluate_candidate_inventory()
    audit.update(
        {
            "gate_database_sha256": SOURCE_HASHES["gate_database"],
            "gate_receipt_sha256": SOURCE_HASHES["gate_receipt"],
        }
    )
    selected = audit.get("selected") or {}
    if selected.get("shell") != SELECTED_SHELL:
        raise RuntimeError("The frozen first-passing replacement shell changed")
    audit["audit_sha256"] = hashlib.sha256(
        stable_json(audit).encode("utf-8")
    ).hexdigest()
    return audit


def _source_paths() -> dict[str, Path]:
    root = Paths().artifacts / "acoustic-calibration"
    v2 = root / SOURCE_RUN_ID
    v3 = root / SOURCE_CONFIRMATION_RUN_ID
    return {
        "v2_source_manifest": v2 / "manifest.json",
        "v2_praat_protocol": v2 / "calibration-v2-praat" / "protocol.json",
        "v2_praat_results": v2 / "calibration-v2-praat" / "results.csv",
        "v2_praat_summary": v2 / "calibration-v2-praat" / "summary.json",
        "v3_manifest": v3 / "manifest.json",
        "v3_results": v3 / "results.csv",
        "v3_analysis": v3 / "analysis.json",
        "v3_summary": v3 / "summary.json",
    }


def _load_v3_reused_records() -> list[dict[str, Any]]:
    run_dir = Paths().artifacts / "acoustic-calibration" / SOURCE_CONFIRMATION_RUN_ID
    records = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((run_dir / "slots").glob("*.json"))
    ]
    selected = [
        record
        for record in records
        if record["stimulus"]["kind"] == "reference"
        or record["stimulus"].get("shell") in {"z_V_f", "v_V_p"}
    ]
    if len(selected) != 26:
        raise RuntimeError("Expected two v3 ae anchors and 24 reused contrast records")
    for record in selected:
        audio_path = run_dir / "audio" / record["audio_filename"]
        if sha256_file(audio_path) != record["audio_sha256"]:
            raise RuntimeError("A reused v3 audio artifact is missing or changed")
    return selected


def reused_evidence_record() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    for name, path in _source_paths().items():
        if sha256_file(path) != SOURCE_HASHES[name]:
            raise RuntimeError(f"Reused source artifact changed: {name}")

    v2_records = _existing_confirmation_references()
    v3_records = _load_v3_reused_records()
    entries = [
        {
            "origin": SOURCE_RUN_ID,
            "slot_id": record["stimulus"]["slot_id"],
            "audio_sha256": record["audio_sha256"],
        }
        for record in v2_records
    ] + [
        {
            "origin": SOURCE_CONFIRMATION_RUN_ID,
            "slot_id": record["stimulus"]["slot_id"],
            "audio_sha256": record["audio_sha256"],
        }
        for record in v3_records
    ]
    entries.sort(key=lambda item: (item["origin"], item["slot_id"]))
    binding = {
        "schema_version": 1,
        "source_artifact_sha256": {
            key: SOURCE_HASHES[key] for key in _source_paths()
        },
        "reused_audio_count": len(entries),
        "reused_audio": entries,
    }
    if len(entries) != 36:
        raise RuntimeError("Amendment must bind exactly 12 anchors and 24 old cells")
    binding["binding_sha256"] = hashlib.sha256(
        stable_json(binding).encode("utf-8")
    ).hexdigest()
    return binding, [*v2_records, *v3_records]


def amendment_protocol_record() -> dict[str, Any]:
    gate_audit = candidate_gate_audit_record()
    evidence, _records = reused_evidence_record()
    protocol: dict[str, Any] = {
        "schema_version": 4,
        "status": "amended_confirmation_not_independent_replication",
        "model": MODEL,
        "voice": VOICE,
        "format": FORMAT,
        "modalities": ["text", "audio"],
        "store": False,
        "sdk_max_retries": 0,
        "manifest_seed": MANIFEST_SEED,
        "request_slots": 12,
        "maximum_successfully_returned_audio": 12,
        "stimuli": [asdict(item) for item in build_amendment_manifest()],
        "selected_candidate": {
            "shell": SELECTED_SHELL,
            "tokens": SELECTED_TOKENS,
            "candidate_gate_audit_sha256": gate_audit["audit_sha256"],
            "gate_database_sha256": SOURCE_HASHES["gate_database"],
            "gate_receipt_sha256": SOURCE_HASHES["gate_receipt"],
        },
        "reused_evidence": {
            "binding_sha256": evidence["binding_sha256"],
            "reused_audio_count": evidence["reused_audio_count"],
            "source_artifact_sha256": evidence["source_artifact_sha256"],
        },
        "measurement": {
            "instrument": "Parselmouth/Praat Burg",
            "time_step_s": TIME_STEP_S,
            "max_number_of_formants": MAX_NUMBER_OF_FORMANTS,
            "maximum_formant_hz": MAXIMUM_FORMANT_HZ,
            "window_length_s": WINDOW_LENGTH_S,
            "pre_emphasis_from_hz": PRE_EMPHASIS_FROM_HZ,
            "vowel_center_fraction": [
                VOWEL_CENTER_START_FRACTION,
                VOWEL_CENTER_END_FRACTION,
            ],
            "exclusions": "calibration-v3 exclusions unchanged",
            "classification": "carrier-v3-preregistered-formant-gates-v1",
            "anchors": "frozen v3 internal-Marin anchors",
        },
        "retry_policy": {
            "maximum_total_attempts_per_slot": MAX_TRANSPORT_ATTEMPTS,
            "retryable_only_if_no_valid_audio": True,
            "retryable_failures": ["HTTP 429", "HTTP 5xx", "connection", "timeout"],
            "backoff_seconds_after_attempts_1_and_2": list(RETRY_BACKOFF_SECONDS),
            "valid_audio_makes_slot_final": True,
            "interrupted_started_attempt_makes_slot_final_failed": True,
            "no_replacement_takes": True,
        },
        "possible_outcomes": {
            "ptbr.vowel.ih_to_i": [
                "exact-category pass",
                "directional-only pass",
                "fail",
            ],
            "ptbr.vowel.ae_to_eh": [
                "exact-category pass",
                "directional-only pass",
                "fail",
            ],
            "ptbr.vowel.uh_to_u": ["excluded"],
        },
        "stopping_rule": (
            "Close calibration after these 12 logical slots; ship each rule only at "
            "the earned claim level, disable failures, and perform no further shell "
            "search, threshold change, or paid calibration loop."
        ),
    }
    protocol["protocol_sha256"] = hashlib.sha256(
        stable_json(protocol).encode("utf-8")
    ).hexdigest()
    return protocol


def _request_id_from_exception(exc: Exception) -> str:
    request_id = getattr(exc, "request_id", None)
    if request_id:
        return str(request_id)
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers:
        return str(headers.get("x-request-id") or "")
    return ""


def _transport_failure(exc: Exception) -> tuple[bool, int | None]:
    status = getattr(exc, "status_code", None)
    if isinstance(exc, (APIConnectionError, APITimeoutError, RateLimitError)):
        return True, status
    if isinstance(exc, APIStatusError) and (status == 429 or (status or 0) >= 500):
        return True, status
    return False, status


def _attempt_request(
    *,
    client: Any,
    stimulus: ConfirmationStimulus,
    attempt_number: int,
    partial_path: Path,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    started = time.monotonic()
    attempt: dict[str, Any] = {
        "attempt_number": attempt_number,
        "status": "failed",
        "retryable_transport_failure": False,
        "request_id": "",
        "usage": {},
        "estimated_cost_usd": 0.0,
    }
    try:
        completion = client.chat.completions.create(
            model=MODEL,
            modalities=["text", "audio"],
            audio={"voice": VOICE, "format": FORMAT},
            messages=build_calibration_messages(stimulus),
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
        attempt["provider_transcript"] = getattr(audio, "transcript", "") or ""
        partial_path.parent.mkdir(parents=True, exist_ok=True)
        partial_path.write_bytes(base64.b64decode(audio.data, validate=True))
        analysis = analyze_wav_praat(partial_path)
        if not analysis.get("decoded_sample_count"):
            raise ValueError("response audio was not a valid decodable PCM WAV")
        attempt.update(
            {
                "status": "valid_audio",
                "audio_sha256": sha256_file(partial_path),
            }
        )
        return attempt, analysis
    except Exception as exc:
        retryable, status = _transport_failure(exc)
        attempt.update(
            {
                "status": "transport_failure" if retryable else "response_failure",
                "retryable_transport_failure": retryable,
                "request_id": attempt.get("request_id") or _request_id_from_exception(exc),
                "http_status": status,
                "error_type": type(exc).__name__,
                "error_detail": str(exc).replace("\n", " ")[:500],
            }
        )
        partial_path.unlink(missing_ok=True)
        return attempt, None
    finally:
        attempt["latency_ms"] = round((time.monotonic() - started) * 1000)


def _final_record(
    *,
    request_order: int,
    stimulus: ConfirmationStimulus,
    audio_path: Path,
    attempts: Sequence[dict[str, Any]],
    analysis: dict[str, Any] | None,
) -> dict[str, Any]:
    valid_attempt = next(
        (attempt for attempt in attempts if attempt["status"] == "valid_audio"), None
    )
    record: dict[str, Any] = {
        "request_order": request_order,
        "stimulus": asdict(stimulus),
        "status": "ok" if valid_attempt else "failed",
        "model": MODEL,
        "voice": VOICE,
        "format": FORMAT,
        "attempts": list(attempts),
        "analysis": analysis or analyze_wav_praat(audio_path),
    }
    if valid_attempt:
        transcript = valid_attempt.get("provider_transcript", "")
        record.update(
            {
                "request_id": valid_attempt.get("request_id", ""),
                "resolved_model": valid_attempt.get("resolved_model", ""),
                "latency_ms": valid_attempt.get("latency_ms", 0),
                "provider_transcript": transcript,
                "transcript_check": asdict(check_transcript(stimulus.token, transcript)),
                "audio_filename": audio_path.name,
                "audio_sha256": valid_attempt["audio_sha256"],
                "audio_integrity_pass": True,
                "usage": valid_attempt.get("usage", {}),
            }
        )
    else:
        final_attempt = attempts[-1]
        record.update(
            {
                "audio_integrity_pass": False,
                "usage": {},
                "error_type": final_attempt.get("error_type", "RequestFailure"),
                "error_detail": final_attempt.get("error_detail", ""),
            }
        )
    transcript_exact = bool(
        (record.get("transcript_check") or {}).get("exact_token_match")
    )
    record["exclusion_reasons"] = exclusion_reasons(
        status=record["status"],
        transcript_exact=transcript_exact,
        analysis=record["analysis"],
    )
    record["estimated_cost_usd"] = round(
        sum(float(attempt.get("estimated_cost_usd") or 0) for attempt in attempts), 9
    )
    return record


def _interrupted_record(
    *,
    request_order: int,
    stimulus: ConfirmationStimulus,
    audio_path: Path,
    attempts: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    finished_attempts = [
        attempt for attempt in attempts if attempt.get("status") != "started"
    ]
    finished_attempts.append(
        {
            "attempt_number": len(attempts),
            "status": "interrupted_unknown",
            "retryable_transport_failure": False,
            "request_id": "",
            "usage": {},
            "estimated_cost_usd": 0.0,
            "error_type": "InterruptedAttemptState",
            "error_detail": (
                "A started attempt had no completed receipt; the slot is final-failed "
                "and was not retried because valid-audio return cannot be excluded."
            ),
        }
    )
    return _final_record(
        request_order=request_order,
        stimulus=stimulus,
        audio_path=audio_path,
        attempts=finished_attempts,
        analysis=None,
    )


def _flatten_amendment_record(record: dict[str, Any]) -> dict[str, Any]:
    row = _flatten_record(record)
    row.update(
        {
            "attempt_count": len(record["attempts"]),
            "attempts_json": json.dumps(record["attempts"], separators=(",", ":")),
        }
    )
    return row


def _flatten_attempt(
    request_order: int, slot_id: str, attempt: dict[str, Any]
) -> dict[str, Any]:
    usage = attempt.get("usage") or {}
    prompt_details = usage.get("prompt_tokens_details") or {}
    completion_details = usage.get("completion_tokens_details") or {}
    return {
        "request_order": request_order,
        "slot_id": slot_id,
        "attempt_number": attempt["attempt_number"],
        "status": attempt.get("status", ""),
        "retryable_transport_failure": attempt.get(
            "retryable_transport_failure", False
        ),
        "request_id": attempt.get("request_id", ""),
        "http_status": attempt.get("http_status", ""),
        "resolved_model": attempt.get("resolved_model", ""),
        "latency_ms": attempt.get("latency_ms", ""),
        "provider_transcript": attempt.get("provider_transcript", ""),
        "audio_sha256": attempt.get("audio_sha256", ""),
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "prompt_audio_tokens": int(prompt_details.get("audio_tokens") or 0),
        "completion_tokens": int(usage.get("completion_tokens") or 0),
        "completion_audio_tokens": int(completion_details.get("audio_tokens") or 0),
        "estimated_request_cost_usd": attempt.get("estimated_cost_usd", 0),
        "error_type": attempt.get("error_type", ""),
        "error_detail": attempt.get("error_detail", ""),
    }


def run_amended_confirmation(
    *,
    client: Any | None = None,
    run_id: str = RUN_ID,
) -> dict[str, Any]:
    from .api import require_api_key

    if AMENDMENT_PREREGISTRATION_HEADING not in DEVLOG_PATH.read_text(
        encoding="utf-8"
    ):
        raise RuntimeError("The complete amended-confirmation preregistration is missing")
    gate_audit = candidate_gate_audit_record()
    evidence, reused_records = reused_evidence_record()
    protocol = amendment_protocol_record()
    if protocol["protocol_sha256"] != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError("Amended-confirmation protocol does not match its freeze")
    if client is None:
        require_api_key()
        client = OpenAI(max_retries=0)

    paths = Paths()
    paths.run_dir(run_id)
    run_dir = paths.artifacts / "acoustic-calibration" / run_id
    manifest_path = run_dir / "manifest.json"
    if manifest_path.is_file():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != protocol:
            raise RuntimeError("Existing amended manifest does not match the freeze")
    else:
        if run_dir.exists() and any(run_dir.iterdir()):
            raise RuntimeError("Amended run directory exists without a manifest")
        atomic_write_json(run_dir / "candidate-gate-audit.json", gate_audit)
        atomic_write_json(run_dir / "reused-evidence.json", evidence)
        atomic_write_json(manifest_path, protocol)

    records: list[dict[str, Any]] = []
    manifest = build_amendment_manifest()
    for request_order, stimulus in enumerate(manifest, start=1):
        receipt_path = run_dir / "slots" / f"{request_order:03d}__{stimulus.slot_id}.json"
        audio_path = run_dir / "audio" / f"{request_order:03d}__{stimulus.slot_id}.wav"
        if receipt_path.is_file():
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            if receipt.get("status") == "complete":
                records.append(receipt["record"])
                continue
            attempts = receipt.get("attempts") or []
            if any(attempt.get("status") == "started" for attempt in attempts):
                record = _interrupted_record(
                    request_order=request_order,
                    stimulus=stimulus,
                    audio_path=audio_path,
                    attempts=attempts,
                )
                atomic_write_json(receipt_path, {"status": "complete", "record": record})
                records.append(record)
                continue
            raise RuntimeError("Unexpected incomplete amendment receipt")

        attempts: list[dict[str, Any]] = []
        final_analysis: dict[str, Any] | None = None
        for attempt_number in range(1, MAX_TRANSPORT_ATTEMPTS + 1):
            attempts.append({"attempt_number": attempt_number, "status": "started"})
            atomic_write_json(receipt_path, {"status": "running", "attempts": attempts})
            partial_path = audio_path.with_suffix(f".attempt-{attempt_number}.partial.wav")
            attempt, analysis = _attempt_request(
                client=client,
                stimulus=stimulus,
                attempt_number=attempt_number,
                partial_path=partial_path,
            )
            attempts[-1] = attempt
            atomic_write_json(receipt_path, {"status": "running", "attempts": attempts})
            if attempt["status"] == "valid_audio":
                partial_path.replace(audio_path)
                final_analysis = analysis
                break
            if not attempt["retryable_transport_failure"]:
                break
            if attempt_number < MAX_TRANSPORT_ATTEMPTS:
                time.sleep(RETRY_BACKOFF_SECONDS[attempt_number - 1])

        record = _final_record(
            request_order=request_order,
            stimulus=stimulus,
            audio_path=audio_path,
            attempts=attempts,
            analysis=final_analysis,
        )
        atomic_write_json(receipt_path, {"status": "complete", "record": record})
        records.append(record)
        print(
            f"amendment {request_order:02d}/12 {stimulus.slot_id}: "
            f"{record['status']} ({len(attempts)} attempt(s), "
            f"{len(record['exclusion_reasons'])} exclusions)",
            flush=True,
        )

    if len(records) != 12:
        raise AssertionError("The amendment must retain all 12 logical slots")
    successful = sum(record["status"] == "ok" for record in records)
    if successful > 12:
        raise AssertionError("The amendment returned more than 12 valid audio clips")

    write_csv(
        run_dir / "results.csv",
        [_flatten_amendment_record(record) for record in records],
        RESULT_FIELDS,
    )
    attempt_rows = [
        _flatten_attempt(record["request_order"], record["stimulus"]["slot_id"], attempt)
        for record in records
        for attempt in record["attempts"]
    ]
    write_csv(run_dir / "attempts.csv", attempt_rows, ATTEMPT_FIELDS)

    references = [
        record for record in reused_records if record["stimulus"]["kind"] == "reference"
    ]
    reused_contrasts = [
        record for record in reused_records if record["stimulus"]["kind"] == "contrast"
    ]
    instrument = evaluate_internal_coherence(references)
    gates = classify_calibration(
        [*references, *reused_contrasts, *records],
        rule_specs=AMENDED_RULE_SPECS,
    )
    outcomes = {
        rule_id: result["outcome"] if instrument["passed"] else "fail"
        for rule_id, result in gates["rules"].items()
    }
    analysis = {
        "schema_version": 4,
        "status": "amended_confirmation_closed",
        "not_independent_replication": True,
        "internal_Marin_sanity_coherence_gate": instrument,
        "unchanged_within_Marin_gates": gates,
        "instrument_qualified_outcomes": outcomes,
        "excluded_rule": "ptbr.vowel.uh_to_u",
        "stopping_rule_reached": True,
    }
    atomic_write_json(run_dir / "analysis.json", analysis)

    all_attempts = [attempt for record in records for attempt in record["attempts"]]
    usage = summarize_usage(all_attempts)
    summary = {
        "schema_version": 4,
        "status": "amended_confirmation_closed",
        "run_id": run_id,
        "protocol_sha256": protocol["protocol_sha256"],
        "logical_request_slots": 12,
        "total_api_attempts": len(all_attempts),
        "successful_audio_returns": successful,
        "transport_failure_attempts": sum(
            attempt.get("status") == "transport_failure" for attempt in all_attempts
        ),
        "exact_transcripts": sum(
            bool((record.get("transcript_check") or {}).get("exact_token_match"))
            for record in records
        ),
        "non_excluded_new_takes": sum(
            not record["exclusion_reasons"] for record in records
        ),
        "excluded_new_takes": sum(bool(record["exclusion_reasons"]) for record in records),
        "instrument_passed": instrument["passed"],
        "outcomes": outcomes,
        "excluded_rule": "ptbr.vowel.uh_to_u",
        "calibration_closed": True,
        "usage": usage,
        "manifest": str(manifest_path),
        "results_csv": str(run_dir / "results.csv"),
        "attempts_csv": str(run_dir / "attempts.csv"),
        "analysis_json": str(run_dir / "analysis.json"),
    }
    atomic_write_json(run_dir / "summary.json", summary)
    return summary
