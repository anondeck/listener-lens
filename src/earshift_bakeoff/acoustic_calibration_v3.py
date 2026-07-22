from __future__ import annotations

import base64
import csv
import hashlib
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np
from openai import OpenAI

from .acoustic_calibration import (
    FORMAT,
    MODEL,
    RULE_SPECS,
    VOICE,
    _usage_dict,
    bark,
    build_calibration_messages,
    classify_calibration,
    estimated_cost_usd,
    exclusion_reasons,
    summarize_usage,
)
from .acoustic_reaudit import analyze_wav_praat
from .audio_conformance import check_transcript
from .config import DEVLOG_PATH, Paths, stable_json
from .util import atomic_write_json, sha256_file, write_csv


PREREGISTRATION_HEADING = (
    "## Calibration-v3 internal-Marin preregistration — July 15, 2026"
)
SOURCE_RUN_ID = "20260715-carrier-v3-calibration"
SOURCE_PROTOCOL_SHA256 = (
    "2860bb2b01f898aaed37fa62a27ec40b0c43a62bc68db36246c6ddddd750f748"
)
SOURCE_V2_PROTOCOL_SHA256 = (
    "cacc1041405c1edf156ecbf4881b3ac6cc3a006e7445e1ae14125f23ec49f551"
)
EXPLORATORY_OUTPUT_DIRECTORY = "calibration-v3-exploratory"
CONFIRMATION_SEED = "calibration-v3-confirmatory-20260715"

F1_ORDER = ("i", "u", "ih", "uh", "eh", "ae")
F2_ORDER = ("u", "uh", "ae", "eh", "ih", "i")
INSTRUMENT_ALLOWED_EXCLUSIONS = {"fewer_than_60_percent_valid_formant_frames"}
INSTRUMENT_MIN_VALID_FRAMES = 5

V3_RULE_SPECS = (
    {
        "rule_id": "ptbr.vowel.ih_to_i",
        "source_category": "ih",
        "target_category": "i",
        "source_ipa": "ɪ",
        "target_ipa": "i",
        "neutral_grapheme": "ih",
        "lens_grapheme": "ee",
        "tokens": (
            ("z_V_f", "zihf", "zeef"),
            ("k_V_sh", "kihsh", "keesh"),
            ("v_V_p", "vihp", "veep"),
        ),
        "spans": {
            "z_V_f": {"neutral": (1, 3), "lens": (1, 3)},
            "k_V_sh": {"neutral": (1, 3), "lens": (1, 3)},
            "v_V_p": {"neutral": (1, 3), "lens": (1, 3)},
        },
    },
    {
        "rule_id": "ptbr.vowel.ae_to_eh",
        "source_category": "ae",
        "target_category": "eh",
        "source_ipa": "æ",
        "target_ipa": "ɛ",
        "neutral_grapheme": "a",
        "lens_grapheme": "eh",
        "tokens": (
            ("z_V_f", "zaf", "zehf"),
            ("k_V_sh", "kash", "kehsh"),
            ("v_V_p", "vap", "vehp"),
        ),
        "spans": {
            "z_V_f": {"neutral": (1, 2), "lens": (1, 3)},
            "k_V_sh": {"neutral": (1, 2), "lens": (1, 3)},
            "v_V_p": {"neutral": (1, 2), "lens": (1, 3)},
        },
    },
)

RESULT_FIELDS = [
    "request_order",
    "slot_id",
    "kind",
    "token",
    "take",
    "reference_category",
    "reference_ipa",
    "rule_id",
    "shell",
    "side",
    "neutral_character_span_json",
    "lens_character_span_json",
    "status",
    "request_id",
    "resolved_model",
    "latency_ms",
    "provider_transcript",
    "exact_token_match",
    "audio_filename",
    "audio_sha256",
    "sample_rate_hz",
    "decoded_sample_count",
    "duration_s",
    "active_start_s",
    "active_end_s",
    "active_duration_s",
    "midpoint_start_s",
    "midpoint_end_s",
    "midpoint_frame_count",
    "valid_formant_frame_count",
    "valid_formant_frame_fraction",
    "clipped_fraction",
    "f1_hz",
    "f2_hz",
    "f1_bark",
    "f2_bark",
    "exclusion_reasons_json",
    "prompt_tokens",
    "prompt_audio_tokens",
    "completion_tokens",
    "completion_audio_tokens",
    "estimated_request_cost_usd",
    "error_type",
    "error_detail",
]


@dataclass(frozen=True)
class ConfirmationStimulus:
    slot_id: str
    kind: Literal["reference", "contrast"]
    token: str
    take: int
    reference_category: str = ""
    reference_ipa: str = ""
    rule_id: str = ""
    shell: str = ""
    side: Literal["neutral", "lens", ""] = ""
    neutral_character_span: tuple[int, int] | None = None
    lens_character_span: tuple[int, int] | None = None


def build_confirmation_manifest() -> tuple[ConfirmationStimulus, ...]:
    stimuli: list[ConfirmationStimulus] = [
        ConfirmationStimulus(
            slot_id=f"reference__ae__take-{take}",
            kind="reference",
            token="bat",
            take=take,
            reference_category="ae",
            reference_ipa="æ",
        )
        for take in (3, 4)
    ]
    for rule in V3_RULE_SPECS:
        rule_slug = rule["rule_id"].removeprefix("ptbr.vowel.")
        for shell, neutral_token, lens_token in rule["tokens"]:
            spans = rule["spans"][shell]
            for side, token in (("neutral", neutral_token), ("lens", lens_token)):
                for take in (1, 2, 3):
                    stimuli.append(
                        ConfirmationStimulus(
                            slot_id=(
                                f"contrast__{rule_slug}__{shell}__{side}__take-{take}"
                            ),
                            kind="contrast",
                            token=token,
                            take=take,
                            rule_id=rule["rule_id"],
                            shell=shell,
                            side=side,  # type: ignore[arg-type]
                            neutral_character_span=spans["neutral"],
                            lens_character_span=spans["lens"],
                        )
                    )
    if len(stimuli) != 38 or len({item.slot_id for item in stimuli}) != 38:
        raise AssertionError("The frozen v3 confirmation manifest requires 38 slots")
    random.Random(CONFIRMATION_SEED).shuffle(stimuli)
    return tuple(stimuli)


def _source_run_dir() -> Path:
    return Paths().artifacts / "acoustic-calibration" / SOURCE_RUN_ID


def _parse_number(value: str, *, integer: bool = False) -> int | float | None:
    if value == "":
        return None
    return int(value) if integer else float(value)


def load_v2_records(run_dir: Path | None = None) -> list[dict[str, Any]]:
    source_dir = run_dir or _source_run_dir()
    manifest = json.loads((source_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("protocol_sha256") != SOURCE_PROTOCOL_SHA256:
        raise RuntimeError("The v3 source render protocol does not match its freeze")
    v2_dir = source_dir / "calibration-v2-praat"
    v2_protocol = json.loads((v2_dir / "protocol.json").read_text(encoding="utf-8"))
    if v2_protocol.get("protocol_sha256") != SOURCE_V2_PROTOCOL_SHA256:
        raise RuntimeError("The v3 source Praat protocol does not match its freeze")
    with (v2_dir / "results.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 66:
        raise RuntimeError("Calibration-v3 exploratory scoring requires 66 records")
    records: list[dict[str, Any]] = []
    for row in rows:
        analysis = {
            "sample_rate_hz": _parse_number(row["sample_rate_hz"], integer=True),
            "decoded_sample_count": _parse_number(
                row["decoded_sample_count"], integer=True
            ),
            "duration_s": _parse_number(row["duration_s"]),
            "clipped_fraction": _parse_number(row["clipped_fraction"]),
            "active_start_s": _parse_number(row["active_start_s"]),
            "active_end_s": _parse_number(row["active_end_s"]),
            "active_duration_s": _parse_number(row["active_duration_s"]),
            "midpoint_start_s": _parse_number(row["midpoint_start_s"]),
            "midpoint_end_s": _parse_number(row["midpoint_end_s"]),
            "midpoint_frame_count": _parse_number(
                row["midpoint_frame_count"], integer=True
            ),
            "valid_formant_frame_count": _parse_number(
                row["valid_formant_frame_count"], integer=True
            ),
            "valid_formant_frame_fraction": _parse_number(
                row["valid_formant_frame_fraction"]
            ),
            "f1_hz": _parse_number(row["f1_hz"]),
            "f2_hz": _parse_number(row["f2_hz"]),
            "f1_bark": _parse_number(row["f1_bark"]),
            "f2_bark": _parse_number(row["f2_bark"]),
            "analysis_errors": json.loads(row["analysis_errors_json"]),
        }
        stimulus = {
            "slot_id": row["slot_id"],
            "kind": row["kind"],
            "token": row["token"],
            "take": int(row["take"]),
            "reference_category": row["reference_category"],
            "reference_ipa": row["reference_ipa"],
            "rule_id": row["rule_id"],
            "shell": row["shell"],
            "side": row["side"],
        }
        records.append(
            {
                "request_order": int(row["request_order"]),
                "stimulus": stimulus,
                "status": row["source_status"],
                "source_exact_token_match": row["source_exact_token_match"]
                == "True",
                "audio_filename": row["audio_filename"],
                "audio_sha256": row["expected_audio_sha256"],
                "audio_integrity_pass": row["audio_integrity_pass"] == "True",
                "analysis": analysis,
                "exclusion_reasons": json.loads(row["exclusion_reasons_json"]),
            }
        )
    return records


def _instrument_take_eligible(record: dict[str, Any]) -> bool:
    analysis = record.get("analysis") or {}
    reasons = set(record.get("exclusion_reasons") or [])
    exact = record.get("source_exact_token_match")
    if exact is None:
        exact = bool((record.get("transcript_check") or {}).get("exact_token_match"))
    return bool(
        record.get("status") == "ok"
        and exact
        and record.get("audio_integrity_pass", True)
        and reasons.issubset(INSTRUMENT_ALLOWED_EXCLUSIONS)
        and int(analysis.get("valid_formant_frame_count") or 0)
        >= INSTRUMENT_MIN_VALID_FRAMES
        and isinstance(analysis.get("f1_hz"), (int, float))
        and isinstance(analysis.get("f2_hz"), (int, float))
    )


def evaluate_internal_coherence(
    reference_records: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    categories: dict[str, Any] = {}
    for category in F1_ORDER:
        takes = [
            record
            for record in reference_records
            if record["stimulus"]["kind"] == "reference"
            and record["stimulus"]["reference_category"] == category
            and _instrument_take_eligible(record)
        ]
        item: dict[str, Any] = {
            "eligible_take_count": len(takes),
            "required_take_count": 2,
            "passed": False,
        }
        if len(takes) == 2:
            f1 = float(np.median([record["analysis"]["f1_hz"] for record in takes]))
            f2 = float(np.median([record["analysis"]["f2_hz"] for record in takes]))
            broad = 180 <= f1 <= 1200 and 600 <= f2 <= 3500 and f2 - f1 >= 250
            item.update(
                {
                    "median_f1_hz": round(f1, 6),
                    "median_f2_hz": round(f2, 6),
                    "f2_minus_f1_hz": round(f2 - f1, 6),
                    "broad_plausibility_pass": broad,
                    "passed": broad,
                }
            )
        else:
            item["failure_reason"] = "requires_exactly_two_measurable_designated_takes"
        categories[category] = item

    measurable = all(item["passed"] for item in categories.values())
    f1_order_pass = bool(
        measurable
        and all(
            categories[left]["median_f1_hz"]
            < categories[right]["median_f1_hz"]
            for left, right in zip(F1_ORDER, F1_ORDER[1:])
        )
    )
    f2_order_pass = bool(
        measurable
        and all(
            categories[left]["median_f2_hz"]
            < categories[right]["median_f2_hz"]
            for left, right in zip(F2_ORDER, F2_ORDER[1:])
        )
    )
    return {
        "protocol": "calibration-v3-internal-Marin-coherence-v1",
        "f1_order_low_to_high": list(F1_ORDER),
        "f2_order_low_to_high": list(F2_ORDER),
        "broad_plausibility_bounds": {
            "f1_hz": [180, 1200],
            "f2_hz": [600, 3500],
            "minimum_f2_minus_f1_hz": 250,
        },
        "categories": categories,
        "all_categories_measurable_and_plausible": measurable,
        "f1_order_pass": f1_order_pass,
        "f2_order_pass": f2_order_pass,
        "passed": measurable and f1_order_pass and f2_order_pass,
        "hillenbrand_role": "directional historical context only; no pass/fail effect",
    }


def _eligible_cell(
    records: Sequence[dict[str, Any]], rule_id: str, shell: str, side: str
) -> dict[str, Any]:
    takes = [
        record
        for record in records
        if record["stimulus"]["rule_id"] == rule_id
        and record["stimulus"]["shell"] == shell
        and record["stimulus"]["side"] == side
        and not record.get("exclusion_reasons")
    ]
    item: dict[str, Any] = {"eligible_take_count": len(takes)}
    if takes:
        item.update(
            {
                "median_f1_hz": round(
                    float(np.median([record["analysis"]["f1_hz"] for record in takes])),
                    6,
                ),
                "median_f2_hz": round(
                    float(np.median([record["analysis"]["f2_hz"] for record in takes])),
                    6,
                ),
            }
        )
    return item


def _exploratory_findings(
    records: Sequence[dict[str, Any]], gate_results: dict[str, Any]
) -> dict[str, Any]:
    ih_rule = gate_results["rules"]["ptbr.vowel.ih_to_i"]
    ih_shell = next(item for item in ih_rule["shells"] if item["shell"] == "z_V_f")
    ae_cells = {
        shell: {
            side: _eligible_cell(records, "ptbr.vowel.ae_to_eh", shell, side)
            for side in ("neutral", "lens")
        }
        for shell in ("z_V_f", "v_V_m")
    }
    uh_cells = {
        shell: {
            side: _eligible_cell(records, "ptbr.vowel.uh_to_u", shell, side)
            for side in ("neutral", "lens")
        }
        for shell in ("z_V_f", "v_V_m")
    }
    return {
        "status": "exploratory_not_confirmatory",
        "ih_to_i_retained_shell": {
            "shell": "z_V_f",
            "neutral": _eligible_cell(
                records, "ptbr.vowel.ih_to_i", "z_V_f", "neutral"
            ),
            "lens": _eligible_cell(
                records, "ptbr.vowel.ih_to_i", "z_V_f", "lens"
            ),
            "directional_pass": ih_shell["directional_pass"],
            "exact_proximity_pass": ih_shell["exact_proximity_pass"],
            "interpretation": "The retained lens cell approaches Marin's /i/ endpoint.",
        },
        "ae_to_eh": {
            "cells": ae_cells,
            "interpretation": (
                "The ae neutral spelling behaves like a name/FACE-vowel carrier in "
                "retained oral cells, reversing the intended within-Marin direction."
            ),
            "redesign": "neutral grapheme a with separate per-side spans",
        },
        "uh_to_u": {
            "cells": uh_cells,
            "interpretation": (
                "Neutral uu and lens oo both occupy a high-vowel region, erasing the "
                "intended contrast."
            ),
            "confirmation_status": "excluded",
        },
    }


def rescore_existing_v3(run_id: str = SOURCE_RUN_ID) -> dict[str, Any]:
    if run_id != SOURCE_RUN_ID:
        raise RuntimeError("The v3 exploratory source run is frozen")
    if PREREGISTRATION_HEADING not in DEVLOG_PATH.read_text(encoding="utf-8"):
        raise RuntimeError("The frozen calibration-v3 preregistration is missing")
    records = load_v2_records()
    references = [record for record in records if record["stimulus"]["kind"] == "reference"]
    instrument = evaluate_internal_coherence(references)
    gate_results = classify_calibration(records, rule_specs=RULE_SPECS)
    outcomes = {
        rule_id: result["outcome"] if instrument["passed"] else "fail"
        for rule_id, result in gate_results["rules"].items()
    }
    analysis = {
        "schema_version": 3,
        "status": "exploratory_not_confirmatory",
        "source_run_id": SOURCE_RUN_ID,
        "instrument_gate": instrument,
        "unchanged_within_Marin_gates": gate_results,
        "instrument_qualified_outcomes": outcomes,
        "findings": _exploratory_findings(records, gate_results),
        "api_calls": 0,
        "estimated_api_cost_usd": 0.0,
    }
    output_dir = _source_run_dir() / EXPLORATORY_OUTPUT_DIRECTORY
    atomic_write_json(output_dir / "analysis.json", analysis)
    summary = {
        "schema_version": 3,
        "status": "exploratory_not_confirmatory",
        "source_run_id": SOURCE_RUN_ID,
        "source_v2_results_sha256": sha256_file(
            _source_run_dir() / "calibration-v2-praat" / "results.csv"
        ),
        "logical_audio_inputs": 66,
        "instrument_passed": instrument["passed"],
        "outcomes": outcomes,
        "api_calls": 0,
        "estimated_api_cost_usd": 0.0,
        "analysis_json": str(output_dir / "analysis.json"),
    }
    atomic_write_json(output_dir / "summary.json", summary)
    return summary


def _existing_confirmation_references() -> list[dict[str, Any]]:
    records = load_v2_records()
    selected = [
        record
        for record in records
        if record["stimulus"]["kind"] == "reference"
        and record["stimulus"]["reference_category"] in {"ih", "i", "eh", "uh", "u"}
    ]
    if len(selected) != 10:
        raise RuntimeError("Confirmation requires ten frozen non-ae reference takes")
    source_audio = _source_run_dir() / "audio"
    for record in selected:
        audio_path = source_audio / record["audio_filename"]
        if not audio_path.is_file() or sha256_file(audio_path) != record["audio_sha256"]:
            raise RuntimeError("A frozen confirmation reference is missing or changed")
    return selected


def confirmation_protocol_record(
    existing_references: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    references = list(existing_references or _existing_confirmation_references())
    protocol: dict[str, Any] = {
        "schema_version": 3,
        "status": "confirmatory",
        "model": MODEL,
        "voice": VOICE,
        "format": FORMAT,
        "store": False,
        "manifest_seed": CONFIRMATION_SEED,
        "request_slots": 38,
        "transport_retries": 0,
        "replacement_takes": 0,
        "stimuli": [asdict(item) for item in build_confirmation_manifest()],
        "rule_specs": V3_RULE_SPECS,
        "existing_reference_inputs": [
            {
                "slot_id": record["stimulus"]["slot_id"],
                "category": record["stimulus"]["reference_category"],
                "take": record["stimulus"]["take"],
                "audio_filename": record["audio_filename"],
                "audio_sha256": record["audio_sha256"],
            }
            for record in references
        ],
        "instrument_protocol": "calibration-v3-internal-Marin-coherence-v1",
        "acoustic_gates": "carrier-v3 gates incorporated verbatim",
    }
    normalized = json.loads(stable_json(protocol))
    normalized["protocol_sha256"] = hashlib.sha256(
        stable_json(normalized).encode("utf-8")
    ).hexdigest()
    return normalized


def _render_confirmation_slot(
    *,
    client: Any,
    stimulus: ConfirmationStimulus,
    request_order: int,
    audio_path: Path,
) -> dict[str, Any]:
    started = time.monotonic()
    record: dict[str, Any] = {
        "request_order": request_order,
        "stimulus": asdict(stimulus),
        "status": "failed",
        "model": MODEL,
        "voice": VOICE,
        "format": FORMAT,
        "usage": {},
        "audio_integrity_pass": False,
    }
    try:
        completion = client.chat.completions.create(
            model=MODEL,
            modalities=["text", "audio"],
            audio={"voice": VOICE, "format": FORMAT},
            messages=build_calibration_messages(stimulus),
            store=False,
        )
        message = completion.choices[0].message
        audio = message.audio
        if audio is None or not audio.data:
            raise RuntimeError("gpt-audio-1.5 returned no audio payload")
        transcript = getattr(audio, "transcript", "") or ""
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        partial = audio_path.with_suffix(audio_path.suffix + ".partial")
        partial.write_bytes(base64.b64decode(audio.data, validate=True))
        partial.replace(audio_path)
        transcript_check = check_transcript(stimulus.token, transcript)
        record.update(
            {
                "status": "ok",
                "request_id": getattr(completion, "_request_id", None) or "",
                "resolved_model": getattr(completion, "model", MODEL),
                "provider_transcript": transcript,
                "transcript_check": asdict(transcript_check),
                "audio_filename": audio_path.name,
                "audio_sha256": sha256_file(audio_path),
                "audio_integrity_pass": True,
                "usage": _usage_dict(completion),
            }
        )
    except Exception as exc:
        record.update(
            {
                "error_type": type(exc).__name__,
                "error_detail": str(exc).replace("\n", " ")[:500],
            }
        )
    record["latency_ms"] = round((time.monotonic() - started) * 1000)
    analysis = analyze_wav_praat(audio_path)
    transcript_exact = bool(
        (record.get("transcript_check") or {}).get("exact_token_match")
    )
    record["analysis"] = analysis
    record["exclusion_reasons"] = exclusion_reasons(
        status=record["status"],
        transcript_exact=transcript_exact,
        analysis=analysis,
    )
    record["estimated_cost_usd"] = estimated_cost_usd(record["usage"])
    return record


def _flatten_record(record: dict[str, Any]) -> dict[str, Any]:
    stimulus = record["stimulus"]
    analysis = record.get("analysis") or {}
    transcript = record.get("transcript_check") or {}
    usage = record.get("usage") or {}
    prompt_details = usage.get("prompt_tokens_details") or {}
    completion_details = usage.get("completion_tokens_details") or {}
    return {
        "request_order": record["request_order"],
        **stimulus,
        "neutral_character_span_json": json.dumps(
            stimulus.get("neutral_character_span"), separators=(",", ":")
        ),
        "lens_character_span_json": json.dumps(
            stimulus.get("lens_character_span"), separators=(",", ":")
        ),
        "status": record.get("status", "failed"),
        "request_id": record.get("request_id", ""),
        "resolved_model": record.get("resolved_model", ""),
        "latency_ms": record.get("latency_ms", ""),
        "provider_transcript": record.get("provider_transcript", ""),
        "exact_token_match": transcript.get("exact_token_match", False),
        "audio_filename": record.get("audio_filename", ""),
        "audio_sha256": record.get("audio_sha256", ""),
        **analysis,
        "exclusion_reasons_json": json.dumps(
            record.get("exclusion_reasons", []), separators=(",", ":")
        ),
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "prompt_audio_tokens": int(prompt_details.get("audio_tokens") or 0),
        "completion_tokens": int(usage.get("completion_tokens") or 0),
        "completion_audio_tokens": int(completion_details.get("audio_tokens") or 0),
        "estimated_request_cost_usd": record.get("estimated_cost_usd", 0),
        "error_type": record.get("error_type", ""),
        "error_detail": record.get("error_detail", ""),
    }


def _interrupted_record(
    request_order: int,
    stimulus: ConfirmationStimulus,
    audio_path: Path,
) -> dict[str, Any]:
    analysis = analyze_wav_praat(audio_path)
    record: dict[str, Any] = {
        "request_order": request_order,
        "stimulus": asdict(stimulus),
        "status": "failed",
        "model": MODEL,
        "voice": VOICE,
        "format": FORMAT,
        "usage": {},
        "audio_integrity_pass": False,
        "error_type": "InterruptedRequestState",
        "error_detail": (
            "A started receipt existed without a completed response; the slot was "
            "not retried under the frozen zero-retry protocol."
        ),
        "analysis": analysis,
        "estimated_cost_usd": 0.0,
    }
    record["exclusion_reasons"] = exclusion_reasons(
        status="failed", transcript_exact=False, analysis=analysis
    )
    return record


def run_confirmation_v3(
    run_id: str,
    *,
    client: Any | None = None,
) -> dict[str, Any]:
    from .api import require_api_key

    if PREREGISTRATION_HEADING not in DEVLOG_PATH.read_text(encoding="utf-8"):
        raise RuntimeError("The frozen calibration-v3 preregistration is missing")
    existing_references = _existing_confirmation_references()
    protocol = confirmation_protocol_record(existing_references)
    if client is None:
        require_api_key()
        client = OpenAI(max_retries=0)

    paths = Paths()
    paths.run_dir(run_id)
    run_dir = paths.artifacts / "acoustic-calibration" / run_id
    manifest_path = run_dir / "manifest.json"
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing != protocol:
            raise RuntimeError("Existing v3 confirmation manifest does not match freeze")
    else:
        if run_dir.exists() and any(run_dir.iterdir()):
            raise RuntimeError("Confirmation directory exists without its manifest")
        atomic_write_json(manifest_path, protocol)

    records: list[dict[str, Any]] = []
    for request_order, stimulus in enumerate(build_confirmation_manifest(), start=1):
        receipt_path = run_dir / "slots" / f"{request_order:03d}__{stimulus.slot_id}.json"
        audio_path = run_dir / "audio" / f"{request_order:03d}__{stimulus.slot_id}.wav"
        if receipt_path.is_file():
            record = json.loads(receipt_path.read_text(encoding="utf-8"))
            if record.get("status") == "started":
                record = _interrupted_record(request_order, stimulus, audio_path)
                atomic_write_json(receipt_path, record)
            records.append(record)
            continue
        atomic_write_json(
            receipt_path,
            {
                "request_order": request_order,
                "stimulus": asdict(stimulus),
                "status": "started",
            },
        )
        record = _render_confirmation_slot(
            client=client,
            stimulus=stimulus,
            request_order=request_order,
            audio_path=audio_path,
        )
        atomic_write_json(receipt_path, record)
        records.append(record)
        print(
            f"confirmation {request_order:02d}/38 {stimulus.slot_id}: "
            f"{record['status']} ({len(record['exclusion_reasons'])} exclusions)",
            flush=True,
        )

    if len(records) != 38:
        raise AssertionError("Confirmation must retain all 38 frozen slots")
    write_csv(run_dir / "results.csv", [_flatten_record(r) for r in records], RESULT_FIELDS)
    new_references = [r for r in records if r["stimulus"]["kind"] == "reference"]
    contrast_records = [r for r in records if r["stimulus"]["kind"] == "contrast"]
    combined_references = [*existing_references, *new_references]
    instrument = evaluate_internal_coherence(combined_references)
    gate_records = [*combined_references, *contrast_records]
    gates = classify_calibration(gate_records, rule_specs=V3_RULE_SPECS)
    outcomes = {
        rule_id: result["outcome"] if instrument["passed"] else "fail"
        for rule_id, result in gates["rules"].items()
    }
    analysis = {
        "schema_version": 3,
        "status": "confirmatory",
        "instrument_gate": instrument,
        "unchanged_within_Marin_gates": gates,
        "instrument_qualified_outcomes": outcomes,
        "excluded_rule": "ptbr.vowel.uh_to_u",
    }
    atomic_write_json(run_dir / "analysis.json", analysis)
    usage = summarize_usage(records)
    summary = {
        "schema_version": 3,
        "status": "confirmatory",
        "run_id": run_id,
        "protocol_sha256": protocol["protocol_sha256"],
        "logical_request_slots": 38,
        "completed_records": len(records),
        "successful_requests": sum(r.get("status") == "ok" for r in records),
        "exact_transcripts": sum(
            bool((r.get("transcript_check") or {}).get("exact_token_match"))
            for r in records
        ),
        "non_excluded_new_takes": sum(not r["exclusion_reasons"] for r in records),
        "excluded_new_takes": sum(bool(r["exclusion_reasons"]) for r in records),
        "instrument_passed": instrument["passed"],
        "outcomes": outcomes,
        "excluded_rule": "ptbr.vowel.uh_to_u",
        "usage": usage,
        "manifest": str(manifest_path),
        "results_csv": str(run_dir / "results.csv"),
        "analysis_json": str(run_dir / "analysis.json"),
    }
    atomic_write_json(run_dir / "summary.json", summary)
    return summary
