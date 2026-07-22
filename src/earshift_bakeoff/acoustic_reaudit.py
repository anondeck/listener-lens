from __future__ import annotations

import csv
import hashlib
import json
import math
import wave
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import parselmouth

from .acoustic_calibration import (
    _decode_pcm_wav,
    _frame_starts,
    bark,
    classify_calibration,
    exclusion_reasons,
)
from .config import DEVLOG_PATH, Paths, stable_json
from .util import atomic_write_json, sha256_file, write_csv


PREREGISTRATION_HEADING = (
    "## Calibration-v2 Praat re-audit preregistration — July 15, 2026"
)
SOURCE_PROTOCOL_SHA256 = (
    "2860bb2b01f898aaed37fa62a27ec40b0c43a62bc68db36246c6ddddd750f748"
)
OUTPUT_DIRECTORY = "calibration-v2-praat"

TIME_STEP_S = 0.005
MAX_NUMBER_OF_FORMANTS = 5.0
MAXIMUM_FORMANT_HZ = 5500.0
WINDOW_LENGTH_S = 0.025
PRE_EMPHASIS_FROM_HZ = 50.0
VOWEL_CENTER_START_FRACTION = 0.25
VOWEL_CENTER_END_FRACTION = 0.75

HILLENBRAND_SOURCE = {
    "citation": (
        "Hillenbrand, Getty, Clark, and Wheeler (1995), Acoustic "
        "characteristics of American English vowels"
    ),
    "doi": "10.1121/1.411872",
    "repository": "santiagobarreda/hillenbrand_et_al_1995",
    "repository_commit": "6dd44decc8ef5b537cbb54732a3b00c8fa652e65",
    "archive_sha256": (
        "2560548591e3a726c88549b6dc9d226995616c26874d601c69fee8b9eee9d730"
    ),
    "vowdata_ds_sha256": (
        "8d9c75980d2b0c1cd64b928c9c13de519eafffc66d8c1ac64a9b0fc01af20d14"
    ),
    "group": "adult women",
    "measure": "steady state",
    "range_definition": "inclusive observed MIN-MAX",
}

HILLENBRAND_FEMALE_RANGES = {
    "ih": {"corpus_code": "ih", "ipa": "ɪ", "f1": [431.0, 556.0], "f2": [2129.0, 2654.0]},
    "i": {"corpus_code": "iy", "ipa": "i", "f1": [331.0, 531.0], "f2": [2359.0, 3049.0]},
    "ae": {"corpus_code": "ae", "ipa": "æ", "f1": [552.0, 893.0], "f2": [1944.0, 2701.0]},
    "eh": {"corpus_code": "eh", "ipa": "ɛ", "f1": [584.0, 981.0], "f2": [1762.0, 2426.0]},
    "uh": {"corpus_code": "oo", "ipa": "ʊ", "f1": [444.0, 617.0], "f2": [987.0, 1619.0]},
    "u": {"corpus_code": "uw", "ipa": "u", "f1": [360.0, 525.0], "f2": [778.0, 1711.0]},
}

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
    "source_status",
    "source_exact_token_match",
    "audio_filename",
    "expected_audio_sha256",
    "observed_audio_sha256",
    "audio_integrity_pass",
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
    "analysis_errors_json",
]


def _active_interval(
    samples: np.ndarray, sample_rate: int
) -> tuple[int, int] | None:
    window = max(1, round(sample_rate * 0.025))
    step = max(1, round(sample_rate * 0.005))
    starts = _frame_starts(samples.size, window, step)
    if not starts.size:
        return None
    rms = np.array(
        [
            math.sqrt(float(np.mean(samples[start : start + window] ** 2)))
            for start in starts
        ]
    )
    peak_rms = float(rms.max(initial=0.0))
    if peak_rms <= 0:
        return None
    active_indices = np.flatnonzero(rms >= peak_rms * 10 ** (-30 / 20))
    if not active_indices.size:
        return None
    active_start = int(starts[active_indices[0]])
    active_end = min(samples.size, int(starts[active_indices[-1]]) + window)
    return active_start, active_end


def analyze_wav_praat(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "sample_rate_hz": None,
        "decoded_sample_count": 0,
        "duration_s": None,
        "clipped_fraction": None,
        "active_start_s": None,
        "active_end_s": None,
        "active_duration_s": None,
        "midpoint_start_s": None,
        "midpoint_end_s": None,
        "midpoint_frame_count": 0,
        "valid_formant_frame_count": 0,
        "valid_formant_frame_fraction": 0.0,
        "f1_hz": None,
        "f2_hz": None,
        "f1_bark": None,
        "f2_bark": None,
        "analysis_errors": [],
    }
    try:
        samples, sample_rate, sample_width = _decode_pcm_wav(path)
    except (EOFError, OSError, ValueError, wave.Error) as exc:
        result["analysis_errors"] = [f"invalid_wav:{type(exc).__name__}"]
        return result
    result["sample_rate_hz"] = sample_rate
    result["decoded_sample_count"] = int(samples.size)
    if not samples.size:
        result["analysis_errors"] = ["absent_audio"]
        return result
    duration = samples.size / sample_rate
    full_scale = 1 - 1 / (2 ** (8 * sample_width - 1))
    result["duration_s"] = round(duration, 6)
    result["clipped_fraction"] = round(
        float(np.mean(np.abs(samples) >= full_scale)), 9
    )

    active = _active_interval(samples, sample_rate)
    if active is None:
        result["analysis_errors"] = ["no_active_interval"]
        return result
    active_start, active_end = active
    active_length = active_end - active_start
    midpoint_start = active_start + round(
        active_length * VOWEL_CENTER_START_FRACTION
    )
    midpoint_end = active_start + round(active_length * VOWEL_CENTER_END_FRACTION)
    result.update(
        {
            "active_start_s": round(active_start / sample_rate, 6),
            "active_end_s": round(active_end / sample_rate, 6),
            "active_duration_s": round(active_length / sample_rate, 6),
            "midpoint_start_s": round(midpoint_start / sample_rate, 6),
            "midpoint_end_s": round(midpoint_end / sample_rate, 6),
        }
    )

    try:
        sound = parselmouth.Sound(samples, sampling_frequency=sample_rate)
        formant = sound.to_formant_burg(
            time_step=TIME_STEP_S,
            max_number_of_formants=MAX_NUMBER_OF_FORMANTS,
            maximum_formant=MAXIMUM_FORMANT_HZ,
            window_length=WINDOW_LENGTH_S,
            pre_emphasis_from=PRE_EMPHASIS_FROM_HZ,
        )
    except Exception as exc:
        result["analysis_errors"] = [f"praat_burg_error:{type(exc).__name__}"]
        return result

    start_s = midpoint_start / sample_rate
    end_s = midpoint_end / sample_rate
    frame_times = [
        float(value)
        for value in formant.xs()
        if start_s <= float(value) <= end_s
    ]
    retained: list[tuple[float, float]] = []
    for time_s in frame_times:
        f1 = float(formant.get_value_at_time(1, time_s))
        f2 = float(formant.get_value_at_time(2, time_s))
        b1 = float(formant.get_bandwidth_at_time(1, time_s))
        b2 = float(formant.get_bandwidth_at_time(2, time_s))
        if not all(math.isfinite(value) for value in (f1, f2, b1, b2)):
            continue
        if not 180 <= f1 <= 1200:
            continue
        if not 600 <= f2 <= 3500 or f2 - f1 < 250:
            continue
        if not 0 < b1 < 700 or not 0 < b2 < 700:
            continue
        retained.append((f1, f2))

    result["midpoint_frame_count"] = len(frame_times)
    result["valid_formant_frame_count"] = len(retained)
    result["valid_formant_frame_fraction"] = round(
        len(retained) / len(frame_times) if frame_times else 0.0, 6
    )
    if retained:
        f1 = float(np.median([value[0] for value in retained]))
        f2 = float(np.median([value[1] for value in retained]))
        result.update(
            {
                "f1_hz": round(f1, 6),
                "f2_hz": round(f2, 6),
                "f1_bark": round(bark(f1), 6),
                "f2_bark": round(bark(f2), 6),
            }
        )
    return result


def validate_hillenbrand_anchors(
    records: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    categories: dict[str, Any] = {}
    for category, reference in HILLENBRAND_FEMALE_RANGES.items():
        takes = [
            record
            for record in records
            if record["stimulus"]["kind"] == "reference"
            and record["stimulus"]["reference_category"] == category
            and not record.get("exclusion_reasons")
        ]
        item: dict[str, Any] = {
            "category": category,
            "ipa": reference["ipa"],
            "hillenbrand_code": reference["corpus_code"],
            "eligible_take_count": len(takes),
            "f1_range_hz": list(reference["f1"]),
            "f2_range_hz": list(reference["f2"]),
            "passed": False,
        }
        if len(takes) == 2:
            f1 = float(np.median([record["analysis"]["f1_hz"] for record in takes]))
            f2 = float(np.median([record["analysis"]["f2_hz"] for record in takes]))
            f1_in_range = reference["f1"][0] <= f1 <= reference["f1"][1]
            f2_in_range = reference["f2"][0] <= f2 <= reference["f2"][1]
            item.update(
                {
                    "anchor_median_f1_hz": round(f1, 6),
                    "anchor_median_f2_hz": round(f2, 6),
                    "f1_in_range": f1_in_range,
                    "f2_in_range": f2_in_range,
                    "passed": f1_in_range and f2_in_range,
                }
            )
        else:
            item["failure_reason"] = "requires_two_non_excluded_anchor_takes"
        categories[category] = item
    return {
        "protocol": "all-six-anchor-medians-inside-hillenbrand-female-min-max-v1",
        "source": HILLENBRAND_SOURCE,
        "categories": categories,
        "passed": all(item["passed"] for item in categories.values()),
        "failed_categories": [
            category for category, item in categories.items() if not item["passed"]
        ],
    }


def _source_rows(run_dir: Path) -> tuple[dict[str, Any], list[dict[str, str]]]:
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("protocol_sha256") != SOURCE_PROTOCOL_SHA256:
        raise RuntimeError("Source calibration protocol hash does not match v2 freeze")
    with (run_dir / "results.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    stimuli = manifest.get("stimuli") or []
    if len(rows) != 66 or len(stimuli) != 66:
        raise RuntimeError("Calibration-v2 requires exactly 66 source rows and slots")
    for request_order, (row, stimulus) in enumerate(zip(rows, stimuli), start=1):
        if int(row["request_order"]) != request_order:
            raise RuntimeError("Source result order is not the frozen manifest order")
        if row["slot_id"] != stimulus["slot_id"]:
            raise RuntimeError("Source result slot does not match frozen manifest")
    return manifest, rows


def _protocol_record(rows: Sequence[dict[str, str]]) -> dict[str, Any]:
    protocol: dict[str, Any] = {
        "schema_version": 2,
        "source_protocol_sha256": SOURCE_PROTOCOL_SHA256,
        "input_audio": [
            {
                "request_order": int(row["request_order"]),
                "slot_id": row["slot_id"],
                "audio_filename": row["audio_filename"],
                "audio_sha256": row["audio_sha256"],
            }
            for row in rows
        ],
        "measurement": {
            "instrument": "Parselmouth/Praat Burg",
            "praat_parselmouth_version": parselmouth.__version__,
            "embedded_praat_version": parselmouth.PRAAT_VERSION,
            "time_step_s": TIME_STEP_S,
            "max_number_of_formants": MAX_NUMBER_OF_FORMANTS,
            "maximum_formant_hz": MAXIMUM_FORMANT_HZ,
            "window_length_s": WINDOW_LENGTH_S,
            "pre_emphasis_from_hz": PRE_EMPHASIS_FROM_HZ,
            "active_interval": "carrier-v3 RMS procedure verbatim",
            "vowel_center_fraction": [
                VOWEL_CENTER_START_FRACTION,
                VOWEL_CENTER_END_FRACTION,
            ],
            "aggregation": "coordinate-wise median of retained frames",
        },
        "hillenbrand_source": HILLENBRAND_SOURCE,
        "hillenbrand_female_ranges": HILLENBRAND_FEMALE_RANGES,
        "v1_gates": "incorporated verbatim from carrier-v3 preregistration",
        "api_calls": 0,
    }
    protocol["protocol_sha256"] = hashlib.sha256(
        stable_json(protocol).encode("utf-8")
    ).hexdigest()
    return protocol


def _bool(value: str) -> bool:
    return value.strip().casefold() == "true"


def _analyze_source_row(
    *, run_dir: Path, stimulus: dict[str, Any], row: dict[str, str]
) -> dict[str, Any]:
    audio_path = run_dir / "audio" / row["audio_filename"]
    observed_sha256 = sha256_file(audio_path) if audio_path.is_file() else ""
    integrity = bool(observed_sha256 and observed_sha256 == row["audio_sha256"])
    analysis = analyze_wav_praat(audio_path) if integrity else {
        "sample_rate_hz": None,
        "decoded_sample_count": 0,
        "duration_s": None,
        "clipped_fraction": None,
        "active_start_s": None,
        "active_end_s": None,
        "active_duration_s": None,
        "midpoint_start_s": None,
        "midpoint_end_s": None,
        "midpoint_frame_count": 0,
        "valid_formant_frame_count": 0,
        "valid_formant_frame_fraction": 0.0,
        "f1_hz": None,
        "f2_hz": None,
        "f1_bark": None,
        "f2_bark": None,
        "analysis_errors": ["missing_or_changed_source_audio"],
    }
    source_status = row["status"]
    transcript_exact = _bool(row["exact_token_match"])
    status = "ok" if integrity and source_status == "ok" else "input_failure"
    reasons = exclusion_reasons(
        status=status, transcript_exact=transcript_exact, analysis=analysis
    )
    if not integrity:
        reasons.insert(0, "source_audio_hash_mismatch_or_missing")
    return {
        "request_order": int(row["request_order"]),
        "stimulus": stimulus,
        "source_status": source_status,
        "source_exact_token_match": transcript_exact,
        "audio_filename": row["audio_filename"],
        "expected_audio_sha256": row["audio_sha256"],
        "observed_audio_sha256": observed_sha256,
        "audio_integrity_pass": integrity,
        "analysis": analysis,
        "exclusion_reasons": list(dict.fromkeys(reasons)),
    }


def _flatten_record(record: dict[str, Any]) -> dict[str, Any]:
    analysis = record["analysis"]
    return {
        "request_order": record["request_order"],
        **record["stimulus"],
        "source_status": record["source_status"],
        "source_exact_token_match": record["source_exact_token_match"],
        "audio_filename": record["audio_filename"],
        "expected_audio_sha256": record["expected_audio_sha256"],
        "observed_audio_sha256": record["observed_audio_sha256"],
        "audio_integrity_pass": record["audio_integrity_pass"],
        **analysis,
        "exclusion_reasons_json": json.dumps(
            record["exclusion_reasons"], separators=(",", ":")
        ),
        "analysis_errors_json": json.dumps(
            analysis.get("analysis_errors", []), separators=(",", ":")
        ),
    }


def run_praat_reaudit(run_id: str) -> dict[str, Any]:
    if PREREGISTRATION_HEADING not in DEVLOG_PATH.read_text(encoding="utf-8"):
        raise RuntimeError("The frozen calibration-v2 preregistration is missing")
    paths = Paths()
    paths.run_dir(run_id)
    run_dir = paths.artifacts / "acoustic-calibration" / run_id
    manifest, rows = _source_rows(run_dir)
    protocol = _protocol_record(rows)
    output_dir = run_dir / OUTPUT_DIRECTORY
    protocol_path = output_dir / "protocol.json"
    summary_path = output_dir / "summary.json"
    if protocol_path.is_file():
        existing = json.loads(protocol_path.read_text(encoding="utf-8"))
        if existing != protocol:
            raise RuntimeError("Existing calibration-v2 protocol does not match freeze")
    else:
        atomic_write_json(protocol_path, protocol)
    if summary_path.is_file():
        return json.loads(summary_path.read_text(encoding="utf-8"))

    records = [
        _analyze_source_row(run_dir=run_dir, stimulus=stimulus, row=row)
        for stimulus, row in zip(manifest["stimuli"], rows)
    ]
    write_csv(
        output_dir / "results.csv",
        [_flatten_record(record) for record in records],
        RESULT_FIELDS,
    )
    instrument = validate_hillenbrand_anchors(records)
    v1_gates = classify_calibration(records)
    outcomes = {
        rule_id: result["outcome"] if instrument["passed"] else "fail"
        for rule_id, result in v1_gates["rules"].items()
    }
    analysis = {
        "schema_version": 2,
        "classification_protocol": (
            "hillenbrand-instrument-check-plus-carrier-v3-gates-verbatim"
        ),
        "instrument_validation": instrument,
        "v1_gate_results": v1_gates,
        "instrument_qualified_outcomes": outcomes,
        "all_rules_pass_directionally": bool(
            instrument["passed"]
            and all(
                outcome in {"exact-category pass", "directional-only pass"}
                for outcome in outcomes.values()
            )
        ),
    }
    atomic_write_json(output_dir / "analysis.json", analysis)
    summary = {
        "schema_version": 2,
        "run_id": run_id,
        "protocol_sha256": protocol["protocol_sha256"],
        "source_protocol_sha256": SOURCE_PROTOCOL_SHA256,
        "logical_audio_inputs": 66,
        "audio_integrity_pass_count": sum(
            record["audio_integrity_pass"] for record in records
        ),
        "non_excluded_takes": sum(
            not record["exclusion_reasons"] for record in records
        ),
        "excluded_takes": sum(bool(record["exclusion_reasons"]) for record in records),
        "instrument_passed": instrument["passed"],
        "failed_anchor_categories": instrument["failed_categories"],
        "outcomes": outcomes,
        "all_rules_pass_directionally": analysis["all_rules_pass_directionally"],
        "api_calls": 0,
        "estimated_api_cost_usd": 0.0,
        "results_csv": str(output_dir / "results.csv"),
        "analysis_json": str(output_dir / "analysis.json"),
    }
    atomic_write_json(summary_path, summary)
    return summary
