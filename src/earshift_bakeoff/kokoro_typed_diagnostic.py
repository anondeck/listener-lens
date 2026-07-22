from __future__ import annotations

import csv
import json
import math
import os
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .config import Paths, stable_json
from .kokoro_synthesis import (
    SAMPLE_RATE_HZ,
    SPEED,
    PairPlan,
    _f0_noise,
    _filtered_symbols,
    _INFERENCE_LOCK,
    _input_ids,
    _predicted_alignment,
    _text_features,
    _validate_plan,
    pcm16_bytes,
    pcm_sha256,
)
from .kokoro_typed_diagnostic_protocol import (
    CEILINGS_HZ,
    DECODER_SLOTS,
    DESCRIPTIVE_WINDOW_PERCENTS,
    EXACT_ANCHOR_VALID_FRACTION,
    FROZEN_V1_RUN_ID,
    LOCALIZATION_MINIMUM,
    MEASUREMENT_SCRIPT,
    MINIMUM_ANCHOR_MAGNITUDE_BARK,
    MINIMUM_CANDIDATE_VALID_FRACTION,
    MINIMUM_DIRECTION_COSINE,
    MINIMUM_VALID_FRAMES,
    PRAAT,
    PRIMARY_WINDOW_PERCENT,
    REPEATED_FIXTURE_ID,
    REPEATED_LENS_PHONEMES,
    REPEATED_MEASUREMENT_COLUMNS,
    REPEATED_NEUTRAL_PHONEMES,
    REPEATED_PLAN_SHA256,
    REPEATED_SOURCE_PHONEMES,
    REPEATED_TARGET_WORD_COLUMNS,
    REPEATED_TARGET_WORD_INDEXES,
    REPEATED_WORD_COLUMNS,
    RNG_SEED,
    RUN_ID,
    STYLE_ROW,
    TARGET_CUE_PADDING_S,
    WINDOW_PERCENTS,
    protocol_record,
    run_dir,
)
from .kokoro_typed_engine import MAX_CLIPPED_FRACTION, KokoroTypedPlanner
from .same_take import bark
from .util import atomic_write_json, sha256_file


RECORDS_FILE = "render-records.json"
ANALYSIS_FILE = "analysis.json"
ATTEMPT_DIR = "attempts"
FROZEN_V1_DIR = Paths().artifacts / "typed-engine" / FROZEN_V1_RUN_ID


def _load_json(path: Path) -> Any:
    if not path.is_file():
        raise RuntimeError(f"required artifact is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _write_once_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        existing = _load_json(path)
        if stable_json(existing) != stable_json(payload):
            raise RuntimeError(f"immutable result differs from recomputation: {path}")
        return
    atomic_write_json(path, payload)


def _write_wav(path: Path, audio: np.ndarray) -> None:
    if path.exists():
        raise RuntimeError(f"one-attempt WAV already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    try:
        with wave.open(str(temporary), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(SAMPLE_RATE_HZ)
            handle.writeframes(pcm16_bytes(audio))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_pcm(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as handle:
        if (
            handle.getnchannels() != 1
            or handle.getsampwidth() != 2
            or handle.getframerate() != SAMPLE_RATE_HZ
        ):
            raise RuntimeError(f"WAV does not match frozen mono PCM16 format: {path}")
        values = np.frombuffer(
            handle.readframes(handle.getnframes()), dtype="<i2"
        ).astype(np.float64)
    if not values.size or not np.isfinite(values).all():
        raise RuntimeError(f"WAV is empty or nonfinite: {path}")
    return values, SAMPLE_RATE_HZ


def _pcm_record(audio: np.ndarray, path: Path) -> dict[str, Any]:
    values = np.asarray(audio, dtype=np.float64).reshape(-1)
    finite = bool(values.size and np.isfinite(values).all())
    clipped_fraction = float(np.mean(np.abs(values) >= 1.0)) if finite else 1.0
    return {
        "relative_path": str(path.relative_to(run_dir())),
        "sample_count": int(values.size),
        "finite": finite,
        "clipped_fraction": clipped_fraction,
        "clipping_pass": bool(clipped_fraction < MAX_CLIPPED_FRACTION),
        "pcm_sha256": pcm_sha256(values) if finite else None,
        "wav_sha256": sha256_file(path),
    }


def merge_sample_intervals(
    intervals: Sequence[tuple[int, int]], sample_count: int
) -> tuple[tuple[int, int], ...]:
    normalized = sorted(
        (max(0, int(start)), min(sample_count, int(end)))
        for start, end in intervals
        if end > 0 and start < sample_count
    )
    merged: list[tuple[int, int]] = []
    for start, end in normalized:
        if end <= start:
            continue
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return tuple(merged)


def localization_report(
    neutral: np.ndarray,
    lens: np.ndarray,
    target_word_intervals: Sequence[dict[str, Any]],
    *,
    sample_rate_hz: int = SAMPLE_RATE_HZ,
) -> dict[str, Any]:
    left = np.asarray(neutral, dtype=np.float64).reshape(-1)
    right = np.asarray(lens, dtype=np.float64).reshape(-1)
    if left.shape != right.shape or not left.size:
        return {
            "sample_count_equal": False,
            "total_difference_energy_positive": False,
            "pass": False,
        }
    padding = round(TARGET_CUE_PADDING_S * sample_rate_hz)
    windows = merge_sample_intervals(
        [
            (
                int(row["start_sample"]) - padding,
                int(row["end_sample_exclusive"]) + padding,
            )
            for row in target_word_intervals
        ],
        len(left),
    )
    mask = np.zeros(len(left), dtype=bool)
    for start, end in windows:
        mask[start:end] = True
    delta = right - left
    energy = delta * delta
    total_energy = float(np.sum(energy))
    inside_energy = float(np.sum(energy[mask]))
    positive = bool(total_energy > 0.0)
    fraction = inside_energy / total_energy if positive else 0.0
    outside = delta[~mask]
    return {
        "sample_count_equal": True,
        "total_difference_energy_positive": positive,
        "inside_windows": [
            {
                "start_sample": start,
                "end_sample_exclusive": end,
                "start_s": start / sample_rate_hz,
                "end_s": end / sample_rate_hz,
            }
            for start, end in windows
        ],
        "inside_difference_energy_fraction": fraction,
        "minimum_inside_difference_energy_fraction": LOCALIZATION_MINIMUM,
        "outside_rms_pcm": (
            float(np.sqrt(np.mean(outside**2))) if outside.size else 0.0
        ),
        "maximum_absolute_pcm_delta": float(np.max(np.abs(delta), initial=0.0)),
        "mean_absolute_pcm_delta": float(np.mean(np.abs(delta))),
        "pass": bool(positive and fraction >= LOCALIZATION_MINIMUM),
    }


def _number(value: str | None) -> float | None:
    try:
        result = float(value) if value is not None else math.nan
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def summarize_frame_table(
    rows: Sequence[dict[str, float | None]],
    interval: dict[str, Any],
    ceiling_hz: int,
    window_percent: int,
) -> dict[str, Any]:
    if window_percent not in WINDOW_PERCENTS:
        raise ValueError(f"unfrozen window percent: {window_percent}")
    fraction = window_percent / 100.0
    duration = float(interval["end_s"]) - float(interval["start_s"])
    start_s = float(interval["start_s"]) + duration * ((1.0 - fraction) / 2.0)
    end_s = float(interval["start_s"]) + duration * ((1.0 + fraction) / 2.0)
    selected = [
        row
        for row in rows
        if row["time_s"] is not None and start_s <= float(row["time_s"]) <= end_s
    ]
    pairs = [
        (float(row["f1_hz"]), float(row["f2_hz"]))
        for row in selected
        if row["f1_hz"] is not None and row["f2_hz"] is not None
    ]
    valid_fraction = len(pairs) / len(selected) if selected else 0.0
    retained = bool(
        len(selected) >= MINIMUM_VALID_FRAMES
        and len(pairs) >= MINIMUM_VALID_FRAMES
        and valid_fraction >= MINIMUM_CANDIDATE_VALID_FRACTION
    )
    result: dict[str, Any] = {
        "ceiling_hz": ceiling_hz,
        "window_percent": window_percent,
        "window_start_s": start_s,
        "window_end_s": end_s,
        "middle_frame_count": len(selected),
        "valid_f1_f2_frame_count": len(pairs),
        "valid_f1_f2_fraction": valid_fraction,
        "retention_pass": retained,
        "measurement_valid": False,
        "plausibility_pass": False,
    }
    if not retained:
        return result
    f1_hz, f2_hz = np.median(np.asarray(pairs, dtype=float), axis=0)
    plausible = bool(
        180 <= f1_hz <= 1200 and 600 <= f2_hz <= 3500 and f2_hz - f1_hz >= 250
    )
    result.update(
        {
            "f1_hz": float(f1_hz),
            "f2_hz": float(f2_hz),
            "f1_bark": bark(float(f1_hz)),
            "f2_bark": bark(float(f2_hz)),
            "plausibility_pass": plausible,
            "measurement_valid": plausible,
        }
    )
    return result


def _frame_table(
    path: Path, interval: dict[str, Any], ceiling_hz: int
) -> list[dict[str, float | None]]:
    with tempfile.TemporaryDirectory(prefix="kokoro-typed-diagnostic-burg-") as temp:
        output = Path(temp) / "frames.tsv"
        subprocess.run(
            [
                str(PRAAT),
                "--run",
                str(MEASUREMENT_SCRIPT),
                str(path),
                str(output),
                f"{float(interval['start_s']):.9f}",
                f"{float(interval['end_s']):.9f}",
                str(ceiling_hz),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        with output.open(encoding="utf-8", newline="") as handle:
            source = list(csv.DictReader(handle, delimiter="\t"))
    return [
        {
            "time_s": _number(row.get("time_s")),
            "f1_hz": _number(row.get("f1_hz")),
            "f2_hz": _number(row.get("f2_hz")),
        }
        for row in source
    ]


def measure_interval_windows(
    path: Path, interval: dict[str, Any], ceiling_hz: int
) -> dict[str, dict[str, Any]]:
    rows = _frame_table(path, interval, ceiling_hz)
    return {
        str(percent): summarize_frame_table(rows, interval, ceiling_hz, percent)
        for percent in WINDOW_PERCENTS
    }


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    a = np.asarray(left, dtype=float)
    b = np.asarray(right, dtype=float)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denominator) if denominator else -1.0


def anchor_precedence(
    *, measurement_valid: bool, medial_valid: bool, phrase_final_valid: bool
) -> str:
    if not measurement_valid:
        return "anchor_measurement_inconclusive"
    if medial_valid and phrase_final_valid:
        return "rescore"
    if medial_valid:
        return "phrase_final_reference_not_realized"
    if phrase_final_valid:
        return "unexpected_medial_reference_failure"
    return "reference_geometry_invalid"


def rescore_attribution(cells: dict[str, dict[str, Any]]) -> dict[str, Any]:
    transported = cells["transported_endpoints__transported_threshold"]["pass"]
    endpoint_switch = cells["local_endpoints__transported_threshold"]["pass"]
    threshold_switch = cells["transported_endpoints__local_threshold"]["pass"]
    local = cells["local_endpoints__local_threshold"]["pass"]
    if transported:
        return {
            "calibration_claim": "transported_calibration_already_passes_under_rescore",
            "mechanical_attribution": "none",
        }
    if not local:
        return {
            "calibration_claim": "transported_calibration_not_mechanically_sufficient_for_this_fixture",
            "mechanical_attribution": "none",
        }
    if endpoint_switch and threshold_switch:
        attribution: str | list[str] = [
            "transported_endpoint_geometry_implicated",
            "transported_magnitude_threshold_implicated",
        ]
    elif endpoint_switch:
        attribution = "transported_endpoint_geometry_implicated"
    elif threshold_switch:
        attribution = "transported_magnitude_threshold_implicated"
    else:
        attribution = "mixed_transported_endpoint_and_threshold_calibration"
    return {
        "calibration_claim": "transported_calibration_mechanically_sufficient_for_this_fixture",
        "mechanical_attribution": attribution,
    }


def candidate_route_decision(candidate: dict[str, Any], *, has_later_span: bool) -> str:
    if candidate["complete_pass"]:
        return "select_for_unseen_confirmation"
    if not candidate["measurement_valid"]:
        return "diagnostic_inconclusive_measurement_or_instrument_failure"
    if not candidate["neutral_source_pass"]:
        return "diagnostic_stopped_neutral_source_reference_failure"
    output_checks = candidate["output_gate"]["checks"]
    if not all(
        output_checks[key]
        for key in (
            "runtime_integrity",
            "exact_state_contract",
            "sample_count_equal",
        )
    ):
        return "diagnostic_inconclusive_runtime_or_integrity_failure"
    if not output_checks["localization_at_least_0_80"]:
        return "candidate_localization_gate_failed"
    if candidate["lens_repairable_failure"] and has_later_span:
        return "advance_to_next_span"
    return "bounded_controlled_span_route_failed"


def _sample_interval(
    columns: Sequence[int], durations: Sequence[int], samples_per_frame: int
) -> dict[str, Any]:
    selected = tuple(int(value) for value in columns)
    if not selected or selected != tuple(range(selected[0], selected[-1] + 1)):
        raise RuntimeError("alignment columns must be nonempty and contiguous")
    start_sample = (
        sum(int(value) for value in durations[: selected[0]]) * samples_per_frame
    )
    end_sample = (
        sum(int(value) for value in durations[: selected[-1] + 1]) * samples_per_frame
    )
    if end_sample <= start_sample:
        raise RuntimeError("alignment interval has nonpositive length")
    return {
        "columns": list(selected),
        "start_sample": start_sample,
        "end_sample_exclusive": end_sample,
        "start_s": start_sample / SAMPLE_RATE_HZ,
        "end_s": end_sample / SAMPLE_RATE_HZ,
    }


def _repeated_alignment(
    *,
    model: Any,
    phonemes: str,
    target_symbol: str,
    durations: Sequence[int],
    sample_count: int,
) -> dict[str, Any]:
    symbols = _filtered_symbols(model, phonemes)
    expected_duration_count = len(symbols) + 2
    if len(durations) != expected_duration_count:
        raise RuntimeError(
            "anchor duration count differs from its own model-token plan"
        )
    if len(phonemes) - 1 != STYLE_ROW:
        raise RuntimeError("anchor no longer selects frozen voice style row 20")
    for stress_column, target_column in REPEATED_MEASUREMENT_COLUMNS:
        if (
            symbols[stress_column - 1] != "ˈ"
            or symbols[target_column - 1] != target_symbol
        ):
            raise RuntimeError(
                "anchor stress/vowel columns drifted from the exact carrier"
            )
    total_frames = sum(int(value) for value in durations)
    if total_frames <= 0 or sample_count % total_frames:
        raise RuntimeError("decoded anchor does not have an integral samples/frame map")
    samples_per_frame = sample_count // total_frames
    occurrences = [
        {
            "occurrence_index": occurrence_index,
            "position": "medial" if occurrence_index == 0 else "phrase-final",
            "target_symbol": target_symbol,
            "measurement_interval": _sample_interval(
                measurement_columns, durations, samples_per_frame
            ),
            "target_word_interval": _sample_interval(
                REPEATED_WORD_COLUMNS[occurrence_index], durations, samples_per_frame
            ),
        }
        for occurrence_index, measurement_columns in enumerate(
            REPEATED_MEASUREMENT_COLUMNS
        )
    ]
    return {
        "duration_count": len(durations),
        "total_alignment_frames": total_frames,
        "samples_per_alignment_frame": samples_per_frame,
        "target_occurrences": occurrences,
        "own_predicted_durations": True,
        "own_alignment": True,
    }


def _ordinary_anchor(runtime: Any, phonemes: str, target_symbol: str) -> dict[str, Any]:
    if phonemes not in {REPEATED_NEUTRAL_PHONEMES, REPEATED_LENS_PHONEMES}:
        raise RuntimeError(
            "ordinary anchor is not one of the two frozen exact carriers"
        )
    with _INFERENCE_LOCK, runtime.torch.no_grad():
        if len(phonemes) - 1 != STYLE_ROW:
            raise RuntimeError("anchor style-row selection drifted")
        ref_s = runtime._reference_style(phonemes)
        features = _text_features(
            runtime.model,
            _input_ids(runtime.model, phonemes, runtime.torch),
            ref_s,
            runtime.torch,
        )
        predicted, alignment = _predicted_alignment(
            runtime.model, features, SPEED, runtime.torch
        )
        f0, noise = _f0_noise(runtime.model, features, alignment, runtime.torch)
        audio = runtime._decode(features["t_en"], alignment, f0, noise, ref_s)
    durations = tuple(int(value) for value in predicted.cpu().tolist())
    alignment_record = _repeated_alignment(
        model=runtime.model,
        phonemes=phonemes,
        target_symbol=target_symbol,
        durations=durations,
        sample_count=len(audio),
    )
    return {
        "audio": audio,
        "predicted_durations": durations,
        "alignment": alignment_record,
        "state_contract": {
            "ordinary_single_text_state": True,
            "own_predicted_durations": True,
            "own_alignment": True,
            "own_f0": True,
            "own_noise": True,
            "style_row": STYLE_ROW,
            "speed": SPEED,
            "device": "cpu",
            "rng_seed": RNG_SEED,
            "pass": True,
        },
    }


def _initial_records(protocol: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "run_id": RUN_ID,
        "protocol_sha256": protocol["protocol_sha256"],
        "status": "in_progress",
        "api_calls_made": 0,
        "openai_calls_made": 0,
        "paid_calls_made": 0,
        "slots": [
            {
                **slot,
                "status": "pending" if slot["order"] <= 2 else "not_reached",
                "reason": (
                    "mandatory_anchor" if slot["order"] <= 2 else "branch_not_evaluated"
                ),
            }
            for slot in protocol["decoder_slots"]
        ],
    }


def _records_path() -> Path:
    return run_dir() / RECORDS_FILE


def _load_or_initialize_records(protocol: dict[str, Any]) -> dict[str, Any]:
    path = _records_path()
    if path.exists():
        records = _load_json(path)
        if records.get("protocol_sha256") != protocol["protocol_sha256"]:
            raise RuntimeError(
                "diagnostic render ledger belongs to a different protocol"
            )
        return records
    audio_dir = run_dir() / "audio"
    attempt_dir = run_dir() / ATTEMPT_DIR
    if (audio_dir.exists() and any(audio_dir.iterdir())) or (
        attempt_dir.exists() and any(attempt_dir.iterdir())
    ):
        raise RuntimeError("orphan diagnostic output exists without a render ledger")
    records = _initial_records(protocol)
    atomic_write_json(path, records)
    return records


def _slot(records: dict[str, Any], slot_id: str) -> dict[str, Any]:
    return next(row for row in records["slots"] if row["slot_id"] == slot_id)


def _persist_records(records: dict[str, Any]) -> None:
    atomic_write_json(_records_path(), records)


def _attempt_marker(slot_id: str) -> Path:
    return run_dir() / ATTEMPT_DIR / f"{slot_id}.json"


def _begin_attempt(records: dict[str, Any], slot_id: str) -> dict[str, Any]:
    row = _slot(records, slot_id)
    marker = _attempt_marker(slot_id)
    if marker.exists():
        if row["status"] == "complete":
            return row
        row["status"] = "interrupted_no_retry"
        row["reason"] = "attempt_marker_exists_without_complete_slot"
        _persist_records(records)
        raise RuntimeError(
            f"decoder slot {slot_id} was already attempted; retry is forbidden"
        )
    marker.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        marker,
        {
            "schema_version": 1,
            "run_id": RUN_ID,
            "protocol_sha256": records["protocol_sha256"],
            "slot_id": slot_id,
            "one_attempt_no_retry": True,
        },
    )
    row["status"] = "attempt_started"
    row["reason"] = "decoder_slot_consumed"
    _persist_records(records)
    return row


def _complete_slot(
    records: dict[str, Any], slot_id: str, payload: dict[str, Any]
) -> None:
    row = _slot(records, slot_id)
    if row["status"] != "attempt_started":
        raise RuntimeError(f"cannot complete decoder slot in state {row['status']}")
    row.update(payload)
    row["status"] = "complete"
    row["reason"] = "one_attempt_complete"
    _persist_records(records)


def _fail_slot(records: dict[str, Any], slot_id: str, exc: Exception) -> None:
    row = _slot(records, slot_id)
    row["status"] = "failed_no_retry"
    row["reason"] = f"{type(exc).__name__}: {str(exc)[:500]}"
    _persist_records(records)


def _render_anchor_slots(records: dict[str, Any]) -> None:
    pending = [
        slot
        for slot in DECODER_SLOTS[:2]
        if _slot(records, slot.slot_id)["status"] != "complete"
    ]
    if not pending:
        return
    from .kokoro_synthesis import KokoroSynthesisRuntime

    runtime = KokoroSynthesisRuntime.load(download=False)
    for slot, target_symbol in zip(DECODER_SLOTS[:2], ("æ", "ɛ"), strict=True):
        row = _slot(records, slot.slot_id)
        if row["status"] == "complete":
            continue
        _begin_attempt(records, slot.slot_id)
        try:
            rendered = _ordinary_anchor(runtime, str(slot.phonemes), target_symbol)
            path = run_dir() / "audio" / f"{slot.order:02d}__{slot.slot_id}.wav"
            _write_wav(path, rendered["audio"])
            pcm = _pcm_record(rendered["audio"], path)
            runtime_pass = bool(
                pcm["finite"] and pcm["clipping_pass"] and pcm["sample_count"] > 0
            )
            if not runtime_pass:
                raise RuntimeError("ordinary anchor failed PCM integrity")
            _complete_slot(
                records,
                slot.slot_id,
                {
                    "phonemes": slot.phonemes,
                    "predicted_durations": list(rendered["predicted_durations"]),
                    "alignment": rendered["alignment"],
                    "state_contract": rendered["state_contract"],
                    "audio": pcm,
                    "runtime_pass": runtime_pass,
                },
            )
        except Exception as exc:
            _fail_slot(records, slot.slot_id, exc)
            raise


def _frozen_repeated_record() -> dict[str, Any]:
    records = _load_json(FROZEN_V1_DIR / "render-records.json")
    return next(
        row for row in records["records"] if row["fixture_id"] == REPEATED_FIXTURE_ID
    )


def _render_conditional_candidate(
    runtime: Any, columns: tuple[int, ...]
) -> dict[str, Any]:
    planner = KokoroTypedPlanner.load()
    plan = planner.plan("The map shows the map.")
    if (
        plan.plan_sha256 != REPEATED_PLAN_SHA256
        or plan.source_phonemes != REPEATED_SOURCE_PHONEMES
        or plan.neutral_phonemes != REPEATED_NEUTRAL_PHONEMES
        or plan.lens_phonemes != REPEATED_LENS_PHONEMES
        or plan.target_word_indexes != REPEATED_TARGET_WORD_INDEXES
    ):
        raise RuntimeError("repeated typed plan drifted before conditional decode")
    pair_plan = PairPlan(
        source_phonemes=plan.source_phonemes,
        neutral_phonemes=plan.neutral_phonemes,
        lens_phonemes=plan.lens_phonemes,
        target_word_indexes=plan.target_word_indexes,
        speed=SPEED,
    )
    frozen = _frozen_repeated_record()
    frozen_durations = tuple(int(value) for value in frozen["predicted_durations"])
    frozen_sample_count = int(frozen["audio"]["neutral"]["sample_count"])
    requested = tuple(int(value) for value in columns)
    with _INFERENCE_LOCK, runtime.torch.no_grad():
        target_columns = _validate_plan(runtime.model, pair_plan)
        if target_columns != REPEATED_TARGET_WORD_COLUMNS:
            raise RuntimeError("base target-word columns drifted")
        ref_s = runtime._reference_style(plan.source_phonemes)
        source_features = _text_features(
            runtime.model,
            _input_ids(runtime.model, plan.source_phonemes, runtime.torch),
            ref_s,
            runtime.torch,
        )
        predicted, alignment = _predicted_alignment(
            runtime.model, source_features, SPEED, runtime.torch
        )
        durations = tuple(int(value) for value in predicted.cpu().tolist())
        if durations != frozen_durations:
            raise RuntimeError(
                "conditional candidate source durations differ from frozen v1"
            )
        neutral_features = _text_features(
            runtime.model,
            _input_ids(runtime.model, plan.neutral_phonemes, runtime.torch),
            ref_s,
            runtime.torch,
        )
        lens_features = _text_features(
            runtime.model,
            _input_ids(runtime.model, plan.lens_phonemes, runtime.torch),
            ref_s,
            runtime.torch,
        )
        state_count = int(neutral_features["t_en"].shape[-1])
        if requested != tuple(sorted(set(requested))):
            raise RuntimeError("conditional columns must be sorted and unique")
        if not requested or requested[0] < 0 or requested[-1] >= state_count:
            raise RuntimeError("conditional columns escape the complete text state")
        if not set(target_columns).issubset(requested):
            raise RuntimeError("conditional span omits a target-word state column")
        f0, noise = _f0_noise(runtime.model, neutral_features, alignment, runtime.torch)
        candidate_state = neutral_features["t_en"].clone()
        candidate_state[:, :, list(requested)] = lens_features["t_en"][
            :, :, list(requested)
        ]
        outside = [index for index in range(state_count) if index not in set(requested)]
        outside_equal = bool(
            runtime.torch.equal(
                candidate_state[:, :, outside], neutral_features["t_en"][:, :, outside]
            )
        )
        inside_equal = bool(
            runtime.torch.equal(
                candidate_state[:, :, list(requested)],
                lens_features["t_en"][:, :, list(requested)],
            )
        )
        audio = runtime._decode(candidate_state, alignment, f0, noise, ref_s)
    sample_count_match = len(audio) == frozen_sample_count
    exact_state_contract_pass = bool(
        durations == frozen_durations
        and tuple(target_columns) == REPEATED_TARGET_WORD_COLUMNS
        and outside_equal
        and inside_equal
        and sample_count_match
        and len(plan.source_phonemes) - 1 == STYLE_ROW
    )
    if not exact_state_contract_pass:
        raise RuntimeError("conditional candidate failed exact shared-state contract")
    return {
        "audio": audio,
        "predicted_durations": durations,
        "state_contract": {
            "source_plan_sha256": plan.plan_sha256,
            "source_derived_durations_match_frozen_v1": True,
            "neutral_derived_f0": True,
            "neutral_derived_noise": True,
            "outside_selected_columns_equal_neutral": outside_equal,
            "inside_selected_columns_equal_lens": inside_equal,
            "selected_columns": list(requested),
            "target_word_columns_contained": True,
            "sample_count_matches_frozen_neutral": sample_count_match,
            "style_row": STYLE_ROW,
            "speed": SPEED,
            "device": "cpu",
            "rng_seed": RNG_SEED,
            "pass": exact_state_contract_pass,
        },
    }


def _render_candidate_slot(records: dict[str, Any], span_id: str) -> None:
    slot = next(slot for slot in DECODER_SLOTS if slot.span_id == span_id)
    row = _slot(records, slot.slot_id)
    if row["status"] == "complete":
        return
    if row["status"] not in {"not_reached", "authorized_by_primary_branch"}:
        raise RuntimeError(f"conditional slot cannot start from {row['status']}")
    row["status"] = "authorized_by_primary_branch"
    row["reason"] = "prior_span_valid_lens_repairable_failure"
    _persist_records(records)
    _begin_attempt(records, slot.slot_id)
    try:
        from .kokoro_synthesis import KokoroSynthesisRuntime

        runtime = KokoroSynthesisRuntime.load(download=False)
        rendered = _render_conditional_candidate(runtime, slot.columns)
        path = run_dir() / "audio" / f"{slot.order:02d}__{slot.slot_id}.wav"
        _write_wav(path, rendered["audio"])
        pcm = _pcm_record(rendered["audio"], path)
        runtime_pass = bool(
            pcm["finite"]
            and pcm["clipping_pass"]
            and pcm["sample_count"] > 0
            and rendered["state_contract"]["pass"]
        )
        if not runtime_pass:
            raise RuntimeError("conditional candidate failed output integrity")
        _complete_slot(
            records,
            slot.slot_id,
            {
                "span_id": span_id,
                "predicted_durations": list(rendered["predicted_durations"]),
                "state_contract": rendered["state_contract"],
                "audio": pcm,
                "runtime_pass": runtime_pass,
            },
        )
    except Exception as exc:
        _fail_slot(records, slot.slot_id, exc)
        raise


def _measure_occurrences(
    path: Path, occurrences: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    measured: list[dict[str, Any]] = []
    for occurrence in occurrences:
        interval = occurrence["measurement_interval"]
        families = {
            str(ceiling): measure_interval_windows(path, interval, ceiling)
            for ceiling in CEILINGS_HZ
        }
        measured.append(
            {
                "occurrence_index": int(occurrence["occurrence_index"]),
                "position": occurrence.get(
                    "position",
                    "medial"
                    if int(occurrence["occurrence_index"]) == 0
                    else "phrase-final",
                ),
                "measurement_interval": interval,
                "families": families,
            }
        )
    return measured


def _measurement_point(measurement: dict[str, Any]) -> np.ndarray:
    return np.asarray([measurement["f1_bark"], measurement["f2_bark"]], dtype=float)


def _transported_family_classification(
    neutral: dict[str, Any], lens: dict[str, Any], anchor: dict[str, Any]
) -> dict[str, Any]:
    source = np.asarray(anchor["full_ae_bark"], dtype=float)
    target = np.asarray(anchor["full_eh_bark"], dtype=float)
    expected = np.asarray(anchor["full_vector_bark"], dtype=float)
    neutral_point = _measurement_point(neutral)
    lens_point = _measurement_point(lens)
    vector = lens_point - neutral_point
    magnitude = float(np.linalg.norm(vector))
    cosine = _cosine(vector, expected)
    source_category = float(np.linalg.norm(neutral_point - source)) < float(
        np.linalg.norm(neutral_point - target)
    )
    target_category = float(np.linalg.norm(lens_point - target)) < float(
        np.linalg.norm(lens_point - source)
    )
    passed = bool(
        anchor["direction_sanity_pass"]
        and neutral["plausibility_pass"]
        and lens["plausibility_pass"]
        and source_category
        and target_category
        and cosine >= MINIMUM_DIRECTION_COSINE
        and magnitude >= float(anchor["product_magnitude_threshold_bark"])
    )
    return {
        "neutral_bark": neutral_point.tolist(),
        "lens_bark": lens_point.tolist(),
        "vector_bark": vector.tolist(),
        "magnitude_bark": magnitude,
        "threshold_bark": float(anchor["product_magnitude_threshold_bark"]),
        "direction_cosine": cosine,
        "neutral_category_pass": source_category,
        "lens_category_pass": target_category,
        "neutral_plausibility_pass": bool(neutral["plausibility_pass"]),
        "lens_plausibility_pass": bool(lens["plausibility_pass"]),
        "pass": passed,
    }


def _exact_float(left: Any, right: Any) -> bool:
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-12)


def _v1_reproduction(
    protocol: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, dict[str, list[dict[str, Any]]]]]:
    records = _load_json(FROZEN_V1_DIR / "render-records.json")
    stored = _load_json(FROZEN_V1_DIR / "analysis.json")
    record_by_fixture = {row["fixture_id"]: row for row in records["records"]}
    stored_by_fixture = {row["fixture_id"]: row for row in stored["fixtures"]}
    geometry = protocol["parents"]["transported_v4_calibration"][
        "context_anchor_geometry"
    ]
    observations: list[dict[str, Any]] = []
    cache: dict[str, dict[str, list[dict[str, Any]]]] = {}
    reproduction_pass = True
    recomputed_fixture_verdicts: dict[str, bool] = {}

    for fixture_id, record in record_by_fixture.items():
        stored_fixture = stored_by_fixture[fixture_id]
        cache[fixture_id] = {}
        paths = {
            role: FROZEN_V1_DIR / record["audio"][role]["relative_path"]
            for role in ("neutral", "lens")
        }
        for role, path in paths.items():
            expected_hash = record["audio"][role]["wav_sha256"]
            if sha256_file(path) != expected_hash:
                raise RuntimeError(f"frozen v1 WAV changed before reproduction: {path}")
            cache[fixture_id][role] = _measure_occurrences(
                path, record["alignment"]["target_occurrences"]
            )

        occurrence_verdicts: list[bool] = []
        for occurrence_index, stored_occurrence in enumerate(
            stored_fixture["target_occurrences"]
        ):
            family_verdicts: list[bool] = []
            for ceiling in CEILINGS_HZ:
                key = str(ceiling)
                neutral = cache[fixture_id]["neutral"][occurrence_index]["families"][
                    key
                ][str(PRIMARY_WINDOW_PERCENT)]
                lens = cache[fixture_id]["lens"][occurrence_index]["families"][key][
                    str(PRIMARY_WINDOW_PERCENT)
                ]
                stored_neutral = stored_occurrence["neutral_measurements"][key]
                stored_lens = stored_occurrence["lens_measurements"][key]
                comparisons: dict[str, bool] = {}
                for role, actual, expected in (
                    ("neutral", neutral, stored_neutral),
                    ("lens", lens, stored_lens),
                ):
                    comparisons[f"{role}_middle_frame_count"] = (
                        actual["middle_frame_count"] == expected["middle_frame_count"]
                    )
                    comparisons[f"{role}_valid_frame_count"] = (
                        actual["valid_f1_f2_frame_count"]
                        == expected["valid_f1_f2_frame_count"]
                    )
                    comparisons[f"{role}_valid_fraction"] = _exact_float(
                        actual["valid_f1_f2_fraction"],
                        expected["valid_f1_f2_fraction"],
                    )
                    comparisons[f"{role}_plausibility"] = (
                        actual["plausibility_pass"] == expected["plausibility_pass"]
                    )
                    for field in ("f1_hz", "f2_hz", "f1_bark", "f2_bark"):
                        comparisons[f"{role}_{field}"] = _exact_float(
                            actual[field], expected[field]
                        )
                recomputed = _transported_family_classification(
                    neutral, lens, geometry["families"][key]
                )
                stored_classification = stored_occurrence["classification"]["families"][
                    key
                ]
                for field in (
                    "magnitude_bark",
                    "threshold_bark",
                    "direction_cosine",
                ):
                    comparisons[f"classification_{field}"] = _exact_float(
                        recomputed[field], stored_classification[field]
                    )
                for field in ("neutral_bark", "lens_bark", "vector_bark"):
                    comparisons[f"classification_{field}"] = all(
                        _exact_float(left, right)
                        for left, right in zip(
                            recomputed[field], stored_classification[field], strict=True
                        )
                    )
                for field in (
                    "neutral_category_pass",
                    "lens_category_pass",
                    "neutral_plausibility_pass",
                    "lens_plausibility_pass",
                    "pass",
                ):
                    comparisons[f"classification_{field}"] = (
                        recomputed[field] == stored_classification[field]
                    )
                family_pass = all(comparisons.values())
                reproduction_pass &= family_pass
                family_verdicts.append(recomputed["pass"])
                observations.append(
                    {
                        "fixture_id": fixture_id,
                        "occurrence_index": occurrence_index,
                        "ceiling_hz": ceiling,
                        "neutral_middle_50": neutral,
                        "lens_middle_50": lens,
                        "recomputed_classification": recomputed,
                        "comparisons": comparisons,
                        "reproduction_pass": family_pass,
                    }
                )
            occurrence_pass = all(family_verdicts)
            occurrence_match = (
                occurrence_pass == stored_occurrence["classification"]["pass"]
                and occurrence_pass == stored_occurrence["pass"]
            )
            reproduction_pass &= occurrence_match
            occurrence_verdicts.append(occurrence_pass)
            observations.append(
                {
                    "fixture_id": fixture_id,
                    "occurrence_index": occurrence_index,
                    "verdict_level": "occurrence",
                    "recomputed_pass": occurrence_pass,
                    "stored_classification_pass": stored_occurrence["classification"][
                        "pass"
                    ],
                    "stored_occurrence_pass": stored_occurrence["pass"],
                    "reproduction_pass": occurrence_match,
                }
            )
        fixture_acoustic = bool(occurrence_verdicts and all(occurrence_verdicts))
        fixture_automatic = bool(
            record["runtime_pass"]
            and fixture_acoustic
            and stored_fixture["localization"]["pass"]
        )
        fixture_match = bool(
            fixture_acoustic == stored_fixture["acoustic_pass"]
            and fixture_automatic == stored_fixture["automatic_replication_pass"]
        )
        reproduction_pass &= fixture_match
        recomputed_fixture_verdicts[fixture_id] = fixture_automatic
        observations.append(
            {
                "fixture_id": fixture_id,
                "verdict_level": "fixture",
                "recomputed_acoustic_pass": fixture_acoustic,
                "stored_acoustic_pass": stored_fixture["acoustic_pass"],
                "recomputed_automatic_pass": fixture_automatic,
                "stored_automatic_pass": stored_fixture["automatic_replication_pass"],
                "reproduction_pass": fixture_match,
            }
        )

    run_automatic = bool(
        recomputed_fixture_verdicts and all(recomputed_fixture_verdicts.values())
    )
    run_match = run_automatic == stored["automatic_replication_pass"]
    reproduction_pass &= run_match
    return (
        {
            "status": "reproduced" if reproduction_pass else "analysis_drift",
            "stored_analysis_sha256": sha256_file(FROZEN_V1_DIR / "analysis.json"),
            "observations": observations,
            "fixture_automatic_verdicts": recomputed_fixture_verdicts,
            "run_automatic_verdict": run_automatic,
            "run_verdict_match": run_match,
            "pass": reproduction_pass,
        },
        cache,
    )


def _anchor_geometry(
    ae: list[dict[str, Any]],
    eh: list[dict[str, Any]],
    transported: dict[str, Any],
) -> dict[str, Any]:
    windows: dict[str, Any] = {}
    for percent in WINDOW_PERCENTS:
        window_key = str(percent)
        occurrence_rows: list[dict[str, Any]] = []
        for occurrence_index in range(2):
            families: dict[str, Any] = {}
            occurrence_valid = True
            for ceiling in CEILINGS_HZ:
                key = str(ceiling)
                ae_measurement = ae[occurrence_index]["families"][key][window_key]
                eh_measurement = eh[occurrence_index]["families"][key][window_key]
                measurement_valid = bool(
                    ae_measurement["measurement_valid"]
                    and eh_measurement["measurement_valid"]
                )
                if measurement_valid:
                    ae_point = _measurement_point(ae_measurement)
                    eh_point = _measurement_point(eh_measurement)
                    vector = eh_point - ae_point
                    magnitude = float(np.linalg.norm(vector))
                    direction_cosine = _cosine(
                        vector, transported["families"][key]["full_vector_bark"]
                    )
                    contrast_checks = {
                        "measurement_valid": True,
                        "magnitude_at_least_0_25_bark": (
                            magnitude >= MINIMUM_ANCHOR_MAGNITUDE_BARK
                        ),
                        "cosine_with_frozen_v4_at_least_0_50": (
                            direction_cosine >= MINIMUM_DIRECTION_COSINE
                        ),
                    }
                    family = {
                        "ae_bark": ae_point.tolist(),
                        "eh_bark": eh_point.tolist(),
                        "vector_bark": vector.tolist(),
                        "magnitude_bark": magnitude,
                        "local_threshold_bark": max(
                            MINIMUM_ANCHOR_MAGNITUDE_BARK, 0.5 * magnitude
                        ),
                        "cosine_with_frozen_v4": direction_cosine,
                        "checks": contrast_checks,
                        "contrast_pass_before_cross_occurrence": all(
                            contrast_checks.values()
                        ),
                    }
                else:
                    family = {
                        "checks": {"measurement_valid": False},
                        "contrast_pass_before_cross_occurrence": False,
                    }
                occurrence_valid &= family["contrast_pass_before_cross_occurrence"]
                families[key] = family
            occurrence_rows.append(
                {
                    "occurrence_index": occurrence_index,
                    "position": "medial" if occurrence_index == 0 else "phrase-final",
                    "families": families,
                    "contrast_pass_before_cross_occurrence": occurrence_valid,
                }
            )

        cross: dict[str, Any] = {}
        for ceiling in CEILINGS_HZ:
            key = str(ceiling)
            left = occurrence_rows[0]["families"][key]
            right = occurrence_rows[1]["families"][key]
            if "vector_bark" in left and "vector_bark" in right:
                cosine = _cosine(left["vector_bark"], right["vector_bark"])
                passed = cosine >= MINIMUM_DIRECTION_COSINE
            else:
                cosine = None
                passed = False
            cross[key] = {"cosine": cosine, "pass": passed}
        cross_pass = all(row["pass"] for row in cross.values())
        for occurrence in occurrence_rows:
            occurrence["contrast_pass"] = bool(
                occurrence["contrast_pass_before_cross_occurrence"] and cross_pass
            )
        windows[window_key] = {
            "occurrences": occurrence_rows,
            "cross_occurrence": cross,
            "cross_occurrence_pass": cross_pass,
            "pass": bool(
                cross_pass and all(row["contrast_pass"] for row in occurrence_rows)
            ),
        }

    primary = windows[str(PRIMARY_WINDOW_PERCENT)]
    measurement_layer: dict[str, Any] = {"sides": {}}
    primary_measurement_valid = True
    for side_id, side in (("ae", ae), ("eh", eh)):
        side_rows: list[dict[str, Any]] = []
        for occurrence in side:
            family_checks: dict[str, Any] = {}
            for ceiling in CEILINGS_HZ:
                measurement = occurrence["families"][str(ceiling)][
                    str(PRIMARY_WINDOW_PERCENT)
                ]
                checks = {
                    "at_least_5_frames": (
                        measurement["middle_frame_count"] >= MINIMUM_VALID_FRAMES
                    ),
                    "at_least_5_valid_pairs": (
                        measurement["valid_f1_f2_frame_count"] >= MINIMUM_VALID_FRAMES
                    ),
                    "valid_fraction_exactly_1": (
                        measurement["valid_f1_f2_fraction"]
                        == EXACT_ANCHOR_VALID_FRACTION
                    ),
                    "plausibility": bool(measurement["plausibility_pass"]),
                }
                family_checks[str(ceiling)] = {
                    "checks": checks,
                    "pass": all(checks.values()),
                }
                primary_measurement_valid &= all(checks.values())
            side_rows.append(
                {
                    "occurrence_index": occurrence["occurrence_index"],
                    "families": family_checks,
                    "pass": all(row["pass"] for row in family_checks.values()),
                }
            )
        measurement_layer["sides"][side_id] = side_rows
    measurement_layer["pass"] = primary_measurement_valid
    medial_valid = bool(primary["occurrences"][0]["contrast_pass"])
    final_valid = bool(primary["occurrences"][1]["contrast_pass"])
    branch = anchor_precedence(
        measurement_valid=primary_measurement_valid,
        medial_valid=medial_valid,
        phrase_final_valid=final_valid,
    )
    primary_signature = [
        family["checks"]
        for occurrence in primary["occurrences"]
        for family in occurrence["families"].values()
    ] + [
        {key: row["pass"] for key, row in primary["cross_occurrence"].items()},
        {"overall_pass": primary["pass"]},
    ]
    sensitivity = {
        str(percent): (
            [
                family["checks"]
                for occurrence in windows[str(percent)]["occurrences"]
                for family in occurrence["families"].values()
            ]
            + [
                {
                    key: row["pass"]
                    for key, row in windows[str(percent)]["cross_occurrence"].items()
                },
                {"overall_pass": windows[str(percent)]["pass"]},
            ]
        )
        != primary_signature
        for percent in DESCRIPTIVE_WINDOW_PERCENTS
    }
    return {
        "ae_measurements": ae,
        "eh_measurements": eh,
        "windows": windows,
        "primary_measurement_layer": measurement_layer,
        "primary_measurement_valid": primary_measurement_valid,
        "primary_contrast_layer": {
            "medial_valid": medial_valid,
            "phrase_final_valid": final_valid,
            "cross_occurrence_valid": primary["cross_occurrence_pass"],
        },
        "primary_medial_contrast_valid": medial_valid,
        "primary_phrase_final_contrast_valid": final_valid,
        "precedence_outcome": branch,
        "descriptive_window_sensitivity": sensitivity,
        "window_sensitive": any(sensitivity.values()),
    }


def _local_family_classification(
    neutral: dict[str, Any],
    lens: dict[str, Any],
    local_family: dict[str, Any],
) -> dict[str, Any]:
    measurement_valid = bool(
        neutral.get("measurement_valid")
        and lens.get("measurement_valid")
        and "ae_bark" in local_family
        and "eh_bark" in local_family
    )
    if not measurement_valid:
        checks = {
            "measurement_valid": False,
            "neutral_plausible": bool(neutral.get("plausibility_pass")),
            "lens_plausible": bool(lens.get("plausibility_pass")),
            "neutral_nearer_local_ae": False,
            "lens_nearer_local_eh": False,
            "direction_cosine_at_least_0_50": False,
            "magnitude_at_least_local_threshold": False,
        }
        return {"checks": checks, "pass": False}
    source = np.asarray(local_family["ae_bark"], dtype=float)
    target = np.asarray(local_family["eh_bark"], dtype=float)
    expected = target - source
    neutral_point = _measurement_point(neutral)
    lens_point = _measurement_point(lens)
    vector = lens_point - neutral_point
    magnitude = float(np.linalg.norm(vector))
    threshold = max(
        MINIMUM_ANCHOR_MAGNITUDE_BARK, 0.5 * float(np.linalg.norm(expected))
    )
    direction_cosine = _cosine(vector, expected)
    neutral_source = float(np.linalg.norm(neutral_point - source)) < float(
        np.linalg.norm(neutral_point - target)
    )
    lens_target = float(np.linalg.norm(lens_point - target)) < float(
        np.linalg.norm(lens_point - source)
    )
    checks = {
        "measurement_valid": True,
        "neutral_plausible": bool(neutral["plausibility_pass"]),
        "lens_plausible": bool(lens["plausibility_pass"]),
        "neutral_nearer_local_ae": neutral_source,
        "lens_nearer_local_eh": lens_target,
        "direction_cosine_at_least_0_50": (
            direction_cosine >= MINIMUM_DIRECTION_COSINE
        ),
        "magnitude_at_least_local_threshold": magnitude >= threshold,
    }
    return {
        "neutral_bark": neutral_point.tolist(),
        "lens_bark": lens_point.tolist(),
        "vector_bark": vector.tolist(),
        "magnitude_bark": magnitude,
        "local_threshold_bark": threshold,
        "direction_cosine": direction_cosine,
        "checks": checks,
        "pass": all(checks.values()),
    }


def _candidate_local_gate(
    *,
    span_id: str,
    neutral: list[dict[str, Any]],
    lens: list[dict[str, Any]],
    anchors: dict[str, Any],
    output_gate: dict[str, Any],
) -> dict[str, Any]:
    windows: dict[str, Any] = {}
    for percent in WINDOW_PERCENTS:
        window_key = str(percent)
        occurrences: list[dict[str, Any]] = []
        for occurrence_index in range(2):
            families: dict[str, Any] = {}
            for ceiling in CEILINGS_HZ:
                key = str(ceiling)
                family = _local_family_classification(
                    neutral[occurrence_index]["families"][key][window_key],
                    lens[occurrence_index]["families"][key][window_key],
                    anchors["windows"][window_key]["occurrences"][occurrence_index][
                        "families"
                    ][key],
                )
                families[key] = family
            occurrences.append(
                {
                    "occurrence_index": occurrence_index,
                    "position": "medial" if occurrence_index == 0 else "phrase-final",
                    "families": families,
                    "pass": all(row["pass"] for row in families.values()),
                }
            )
        windows[window_key] = {
            "occurrences": occurrences,
            "pass": all(row["pass"] for row in occurrences),
        }

    primary = windows[str(PRIMARY_WINDOW_PERCENT)]
    primary_families = [
        family
        for occurrence in primary["occurrences"]
        for family in occurrence["families"].values()
    ]
    measurement_valid = all(
        family["checks"]["measurement_valid"] for family in primary_families
    )
    neutral_source_pass = all(
        family["checks"]["measurement_valid"]
        and family["checks"]["neutral_plausible"]
        and family["checks"]["neutral_nearer_local_ae"]
        for family in primary_families
    )
    output_pass = all(bool(value) for value in output_gate["checks"].values())
    complete_pass = bool(
        measurement_valid and neutral_source_pass and primary["pass"] and output_pass
    )
    lens_repairable_failure = bool(
        measurement_valid
        and neutral_source_pass
        and output_pass
        and not primary["pass"]
    )

    def signature(percent: int) -> list[dict[str, bool]]:
        return [
            family["checks"]
            for occurrence in windows[str(percent)]["occurrences"]
            for family in occurrence["families"].values()
        ] + [{"overall_pass": windows[str(percent)]["pass"]}]

    primary_signature = signature(PRIMARY_WINDOW_PERCENT)
    sensitivity = {
        str(percent): signature(percent) != primary_signature
        for percent in DESCRIPTIVE_WINDOW_PERCENTS
    }
    return {
        "span_id": span_id,
        "neutral_measurements": neutral,
        "lens_measurements": lens,
        "windows": windows,
        "primary_window_percent": PRIMARY_WINDOW_PERCENT,
        "measurement_valid": measurement_valid,
        "neutral_source_pass": neutral_source_pass,
        "output_gate": output_gate,
        "lens_repairable_failure": lens_repairable_failure,
        "complete_pass": complete_pass,
        "descriptive_window_sensitivity": sensitivity,
        "window_sensitive": any(sensitivity.values()),
    }


def _rescore_family(
    *,
    neutral: dict[str, Any],
    lens: dict[str, Any],
    source: Sequence[float],
    target: Sequence[float],
    threshold: float,
) -> dict[str, Any]:
    measurement_valid = bool(
        neutral.get("measurement_valid") and lens.get("measurement_valid")
    )
    if not measurement_valid:
        checks = {
            "measurement_valid": False,
            "neutral_plausible": bool(neutral.get("plausibility_pass")),
            "lens_plausible": bool(lens.get("plausibility_pass")),
            "neutral_nearer_ae": False,
            "lens_nearer_eh": False,
            "direction_cosine_at_least_0_50": False,
            "magnitude_at_least_selected_threshold": False,
        }
        return {"checks": checks, "pass": False}
    ae = np.asarray(source, dtype=float)
    eh = np.asarray(target, dtype=float)
    neutral_point = _measurement_point(neutral)
    lens_point = _measurement_point(lens)
    expected = eh - ae
    vector = lens_point - neutral_point
    magnitude = float(np.linalg.norm(vector))
    direction_cosine = _cosine(vector, expected)
    checks = {
        "measurement_valid": True,
        "neutral_plausible": bool(neutral["plausibility_pass"]),
        "lens_plausible": bool(lens["plausibility_pass"]),
        "neutral_nearer_ae": (
            float(np.linalg.norm(neutral_point - ae))
            < float(np.linalg.norm(neutral_point - eh))
        ),
        "lens_nearer_eh": (
            float(np.linalg.norm(lens_point - eh))
            < float(np.linalg.norm(lens_point - ae))
        ),
        "direction_cosine_at_least_0_50": (
            direction_cosine >= MINIMUM_DIRECTION_COSINE
        ),
        "magnitude_at_least_selected_threshold": magnitude >= threshold,
    }
    return {
        "neutral_bark": neutral_point.tolist(),
        "lens_bark": lens_point.tolist(),
        "endpoint_ae_bark": ae.tolist(),
        "endpoint_eh_bark": eh.tolist(),
        "vector_bark": vector.tolist(),
        "direction_cosine": direction_cosine,
        "magnitude_bark": magnitude,
        "threshold_bark": threshold,
        "checks": checks,
        "pass": all(checks.values()),
    }


def _two_by_two_rescore(
    *,
    neutral: list[dict[str, Any]],
    lens: list[dict[str, Any]],
    anchors: dict[str, Any],
    transported: dict[str, Any],
) -> dict[str, Any]:
    specifications = (
        ("transported_endpoints__transported_threshold", "transported", "transported"),
        ("transported_endpoints__local_threshold", "transported", "local"),
        ("local_endpoints__transported_threshold", "local", "transported"),
        ("local_endpoints__local_threshold", "local", "local"),
    )
    cells: dict[str, Any] = {}
    primary_key = str(PRIMARY_WINDOW_PERCENT)
    for cell_id, endpoint_kind, threshold_kind in specifications:
        occurrence_rows: list[dict[str, Any]] = []
        for occurrence_index in range(2):
            families: dict[str, Any] = {}
            for ceiling in CEILINGS_HZ:
                key = str(ceiling)
                transported_family = transported["families"][key]
                local_family = anchors["windows"][primary_key]["occurrences"][
                    occurrence_index
                ]["families"][key]
                if endpoint_kind == "transported":
                    source = transported_family["full_ae_bark"]
                    target = transported_family["full_eh_bark"]
                else:
                    source = local_family["ae_bark"]
                    target = local_family["eh_bark"]
                threshold = (
                    float(transported_family["product_magnitude_threshold_bark"])
                    if threshold_kind == "transported"
                    else float(local_family["local_threshold_bark"])
                )
                family = _rescore_family(
                    neutral=neutral[occurrence_index]["families"][key][primary_key],
                    lens=lens[occurrence_index]["families"][key][primary_key],
                    source=source,
                    target=target,
                    threshold=threshold,
                )
                families[key] = family
            occurrence_rows.append(
                {
                    "occurrence_index": occurrence_index,
                    "position": "medial" if occurrence_index == 0 else "phrase-final",
                    "families": families,
                    "pass": all(row["pass"] for row in families.values()),
                }
            )
        cells[cell_id] = {
            "endpoint_geometry": endpoint_kind,
            "magnitude_threshold": threshold_kind,
            "occurrences": occurrence_rows,
            "pass": all(row["pass"] for row in occurrence_rows),
        }
    if cells["transported_endpoints__transported_threshold"]["pass"]:
        raise RuntimeError(
            "transported/transported rescore no longer reproduces the frozen repeated-fixture failure"
        )
    return {"cells": cells, **rescore_attribution(cells)}


def _candidate_output_gate(
    span_id: str, records: dict[str, Any]
) -> tuple[dict[str, Any], Path]:
    frozen = _frozen_repeated_record()
    neutral_path = FROZEN_V1_DIR / frozen["audio"]["neutral"]["relative_path"]
    neutral_pcm, rate = _read_pcm(neutral_path)
    if span_id == "target-word":
        lens_path = FROZEN_V1_DIR / frozen["audio"]["lens"]["relative_path"]
        lens_pcm, lens_rate = _read_pcm(lens_path)
        state_contract_pass = bool(
            frozen["plan_sha256"] == REPEATED_PLAN_SHA256
            and tuple(frozen["replaced_columns"]) == REPEATED_TARGET_WORD_COLUMNS
            and frozen["replaced_columns_match_complete_target_words"]
            and frozen["neutral_identity_bit_identical"]
            and frozen["pair_integrity"]["pass_all"]
            and frozen["runtime_pass"]
        )
        runtime_pass = bool(frozen["runtime_pass"])
        output_record = {
            "source": "immutable_frozen_replication_v1",
            "neutral_wav_sha256": sha256_file(neutral_path),
            "lens_wav_sha256": sha256_file(lens_path),
            "selected_columns": list(REPEATED_TARGET_WORD_COLUMNS),
            "state_contract_pass": state_contract_pass,
            "runtime_pass": runtime_pass,
        }
    else:
        slot_spec = next(slot for slot in DECODER_SLOTS if slot.span_id == span_id)
        slot = _slot(records, slot_spec.slot_id)
        if slot["status"] != "complete":
            raise RuntimeError(f"candidate slot is not complete: {span_id}")
        lens_path = run_dir() / slot["audio"]["relative_path"]
        if sha256_file(lens_path) != slot["audio"]["wav_sha256"]:
            raise RuntimeError(f"conditional candidate WAV hash drifted: {span_id}")
        lens_pcm, lens_rate = _read_pcm(lens_path)
        state_contract_pass = bool(slot["state_contract"]["pass"])
        runtime_pass = bool(slot["runtime_pass"])
        output_record = {
            "source": "one_attempt_conditional_decoder_slot",
            "slot_id": slot_spec.slot_id,
            "neutral_wav_sha256": sha256_file(neutral_path),
            "lens_wav_sha256": sha256_file(lens_path),
            "selected_columns": slot["state_contract"]["selected_columns"],
            "state_contract": slot["state_contract"],
            "state_contract_pass": state_contract_pass,
            "runtime_pass": runtime_pass,
        }
    if lens_rate != rate:
        raise RuntimeError("candidate and frozen neutral sample rates differ")
    localization = localization_report(
        neutral_pcm,
        lens_pcm,
        [row["interval"] for row in frozen["alignment"]["target_words"]],
        sample_rate_hz=rate,
    )
    sample_count_equal = len(neutral_pcm) == len(lens_pcm)
    output_record["localization"] = localization
    output_record["sample_count_equal"] = sample_count_equal
    output_record["checks"] = {
        "runtime_integrity": runtime_pass,
        "exact_state_contract": state_contract_pass,
        "sample_count_equal": sample_count_equal,
        "localization_at_least_0_80": bool(localization["pass"]),
    }
    return output_record, lens_path


def _checked_protocol() -> dict[str, Any]:
    path = run_dir() / "protocol.json"
    if not path.is_file():
        raise RuntimeError(
            "diagnostic protocol is not frozen; run the prepare command and commit it before decoding"
        )
    frozen = _load_json(path)
    current = protocol_record()
    if stable_json(frozen) != stable_json(current):
        raise RuntimeError(
            "checked-in diagnostic protocol differs from current bound inputs"
        )
    return frozen


def _verify_measurement_bindings(protocol: dict[str, Any]) -> None:
    measurement = protocol["implementation"]["measurement"]
    if sha256_file(PRAAT) != measurement["praat_sha256"]:
        raise RuntimeError("Praat executable changed after diagnostic freeze")
    if sha256_file(MEASUREMENT_SCRIPT) != measurement["script_sha256"]:
        raise RuntimeError("Praat measurement script changed after diagnostic freeze")


def _anchor_analysis(
    protocol: dict[str, Any], records: dict[str, Any]
) -> dict[str, Any]:
    slots = [_slot(records, slot.slot_id) for slot in DECODER_SLOTS[:2]]
    if not all(row["status"] == "complete" for row in slots):
        return {
            "precedence_outcome": "anchor_decoder_slot_incomplete",
            "primary_measurement_valid": False,
        }
    measurements: list[list[dict[str, Any]]] = []
    for row in slots:
        path = run_dir() / row["audio"]["relative_path"]
        if sha256_file(path) != row["audio"]["wav_sha256"]:
            raise RuntimeError(f"ordinary anchor WAV hash drifted: {row['slot_id']}")
        measurements.append(
            _measure_occurrences(path, row["alignment"]["target_occurrences"])
        )
    transported = protocol["parents"]["transported_v4_calibration"][
        "context_anchor_geometry"
    ]
    return _anchor_geometry(measurements[0], measurements[1], transported)


def _mark_not_reached(
    records: dict[str, Any], *, after_order: int, reason: str
) -> None:
    for row in records["slots"]:
        if row["order"] > after_order and row["status"] in {
            "pending",
            "not_reached",
            "authorized_by_primary_branch",
        }:
            row["status"] = "not_reached"
            row["reason"] = reason
    _persist_records(records)


def _finalize(records: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    records["status"] = "complete"
    records["decoder_slots_attempted"] = sum(
        _attempt_marker(row["slot_id"]).exists() for row in records["slots"]
    )
    records["decoder_slots_complete"] = sum(
        row["status"] == "complete" for row in records["slots"]
    )
    records["one_attempt_slots_respected"] = all(
        row["status"]
        in {
            "complete",
            "not_reached",
            "failed_no_retry",
            "interrupted_no_retry",
        }
        for row in records["slots"]
    )
    _persist_records(records)
    result = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "protocol_sha256": records["protocol_sha256"],
        "status": "analysis_complete",
        "api_calls_made": 0,
        "openai_calls_made": 0,
        "paid_calls_made": 0,
        "render_records_sha256": sha256_file(_records_path()),
        **payload,
    }
    _write_once_json(run_dir() / ANALYSIS_FILE, result)
    return result


def _render_failure_result(
    records: dict[str, Any], *, stage: str, exc: Exception
) -> dict[str, Any]:
    completed = max(
        (row["order"] for row in records["slots"] if row["status"] == "complete"),
        default=0,
    )
    _mark_not_reached(
        records,
        after_order=completed,
        reason="not_reached_after_decoder_runtime_or_integrity_failure",
    )
    return _finalize(
        records,
        {
            "classification": "diagnostic_inconclusive_runtime_or_integrity_failure",
            "claim": "diagnostic_inconclusive_runtime_or_integrity_failure",
            "failure_stage": stage,
            "failure": f"{type(exc).__name__}: {str(exc)[:1000]}",
            "frozen_replication_v1_preserved_failed": True,
            "confirmation_eligible": False,
        },
    )


def run() -> dict[str, Any]:
    existing_analysis = run_dir() / ANALYSIS_FILE
    if existing_analysis.is_file():
        return _load_json(existing_analysis)
    protocol = _checked_protocol()
    _verify_measurement_bindings(protocol)
    records = _load_or_initialize_records(protocol)
    try:
        _render_anchor_slots(records)
    except Exception as exc:
        return _render_failure_result(records, stage="mandatory_anchor_decode", exc=exc)

    try:
        reproduction, cache = _v1_reproduction(protocol)
    except Exception as exc:
        _mark_not_reached(
            records,
            after_order=2,
            reason="not_reached_after_v1_measurement_or_instrument_failure",
        )
        return _finalize(
            records,
            {
                "classification": "diagnostic_inconclusive_measurement_or_instrument_failure",
                "claim": "diagnostic_inconclusive_measurement_or_instrument_failure",
                "failure_stage": "frozen_v1_reproduction",
                "failure": f"{type(exc).__name__}: {str(exc)[:1000]}",
                "frozen_replication_v1_preserved_failed": True,
                "confirmation_eligible": False,
            },
        )
    if not reproduction["pass"]:
        _mark_not_reached(
            records,
            after_order=2,
            reason="not_reached_after_frozen_v1_analysis_drift",
        )
        return _finalize(
            records,
            {
                "classification": "diagnostic_inconclusive_analysis_drift",
                "claim": "diagnostic_inconclusive_measurement_or_instrument_failure",
                "v1_reproduction": reproduction,
                "frozen_replication_v1_preserved_failed": True,
                "confirmation_eligible": False,
            },
        )

    try:
        anchors = _anchor_analysis(protocol, records)
    except Exception as exc:
        _mark_not_reached(
            records,
            after_order=2,
            reason="not_reached_after_anchor_measurement_or_instrument_failure",
        )
        return _finalize(
            records,
            {
                "classification": "diagnostic_inconclusive_measurement_or_instrument_failure",
                "claim": "diagnostic_inconclusive_measurement_or_instrument_failure",
                "failure_stage": "ordinary_anchor_measurement",
                "failure": f"{type(exc).__name__}: {str(exc)[:1000]}",
                "v1_reproduction": reproduction,
                "frozen_replication_v1_preserved_failed": True,
                "confirmation_eligible": False,
            },
        )

    anchor_outcome = anchors["precedence_outcome"]
    if anchor_outcome != "rescore":
        _mark_not_reached(
            records,
            after_order=2,
            reason=f"not_reached_after_{anchor_outcome}",
        )
        claim = (
            "phrase_final_reference_not_realized"
            if anchor_outcome == "phrase_final_reference_not_realized"
            else "diagnostic_inconclusive_measurement_or_instrument_failure"
        )
        return _finalize(
            records,
            {
                "classification": anchor_outcome,
                "claim": claim,
                "v1_reproduction": reproduction,
                "anchors": anchors,
                "frozen_replication_v1_preserved_failed": True,
                "confirmation_eligible": False,
            },
        )

    repeated = cache[REPEATED_FIXTURE_ID]
    transported = protocol["parents"]["transported_v4_calibration"][
        "context_anchor_geometry"
    ]
    rescore = _two_by_two_rescore(
        neutral=repeated["neutral"],
        lens=repeated["lens"],
        anchors=anchors,
        transported=transported,
    )
    candidates: list[dict[str, Any]] = []
    selected_span: str | None = None
    terminal_decision: str | None = None

    for index, span_id in enumerate(
        ("target-word", "target-word-plus-boundaries", "full-contextual-state")
    ):
        if span_id != "target-word":
            try:
                _render_candidate_slot(records, span_id)
            except Exception as exc:
                _mark_not_reached(
                    records,
                    after_order=min(2 + index, 4),
                    reason="not_reached_after_conditional_decoder_runtime_or_integrity_failure",
                )
                return _finalize(
                    records,
                    {
                        "classification": "diagnostic_inconclusive_runtime_or_integrity_failure",
                        "claim": "diagnostic_inconclusive_runtime_or_integrity_failure",
                        "failure_stage": f"conditional_decode_{span_id}",
                        "failure": f"{type(exc).__name__}: {str(exc)[:1000]}",
                        "v1_reproduction": reproduction,
                        "anchors": anchors,
                        "frozen_wav_rescore_2x2": rescore,
                        "candidates": candidates,
                        "frozen_replication_v1_preserved_failed": True,
                        "confirmation_eligible": False,
                    },
                )
        try:
            output_gate, lens_path = _candidate_output_gate(span_id, records)
            if span_id == "target-word":
                lens_measurements = repeated["lens"]
            else:
                lens_measurements = _measure_occurrences(
                    lens_path,
                    _frozen_repeated_record()["alignment"]["target_occurrences"],
                )
            candidate = _candidate_local_gate(
                span_id=span_id,
                neutral=repeated["neutral"],
                lens=lens_measurements,
                anchors=anchors,
                output_gate=output_gate,
            )
        except Exception as exc:
            _mark_not_reached(
                records,
                after_order=min(2 + index, 4),
                reason="not_reached_after_candidate_measurement_or_instrument_failure",
            )
            return _finalize(
                records,
                {
                    "classification": "diagnostic_inconclusive_measurement_or_instrument_failure",
                    "claim": "diagnostic_inconclusive_measurement_or_instrument_failure",
                    "failure_stage": f"candidate_analysis_{span_id}",
                    "failure": f"{type(exc).__name__}: {str(exc)[:1000]}",
                    "v1_reproduction": reproduction,
                    "anchors": anchors,
                    "frozen_wav_rescore_2x2": rescore,
                    "candidates": candidates,
                    "frozen_replication_v1_preserved_failed": True,
                    "confirmation_eligible": False,
                },
            )
        has_later = index < 2
        decision = candidate_route_decision(candidate, has_later_span=has_later)
        candidate["primary_branch_decision"] = decision
        candidates.append(candidate)
        if decision == "select_for_unseen_confirmation":
            selected_span = span_id
            terminal_decision = decision
            _mark_not_reached(
                records,
                after_order=min(2 + index, 4),
                reason="not_reached_after_first_complete_pass",
            )
            break
        if decision == "advance_to_next_span":
            continue
        terminal_decision = decision
        _mark_not_reached(
            records,
            after_order=min(2 + index, 4),
            reason=f"not_reached_after_{decision}",
        )
        break

    confirmation_eligible = selected_span is not None
    if selected_span == "target-word":
        if rescore["calibration_claim"] != (
            "transported_calibration_mechanically_sufficient_for_this_fixture"
        ):
            raise RuntimeError(
                "frozen target-word passed locally without the required 2x2 calibration result"
            )
        classification = rescore["calibration_claim"]
        claim = classification
    elif confirmation_eligible:
        classification = (
            "branch_conditional_span_candidate_selected_for_unseen_confirmation"
        )
        claim = classification
    elif terminal_decision in {
        "diagnostic_inconclusive_measurement_or_instrument_failure",
        "diagnostic_inconclusive_runtime_or_integrity_failure",
        "diagnostic_stopped_neutral_source_reference_failure",
        "candidate_localization_gate_failed",
    }:
        classification = terminal_decision
        claim = terminal_decision
    else:
        classification = "bounded_controlled_span_route_failed"
        claim = classification
    return _finalize(
        records,
        {
            "classification": classification,
            "claim": claim,
            "v1_reproduction": reproduction,
            "anchors": anchors,
            "frozen_wav_rescore_2x2": rescore,
            "candidates": candidates,
            "selected_span": selected_span,
            "mechanical_attribution": (
                rescore["mechanical_attribution"]
                if selected_span == "target-word"
                else None
            ),
            "confirmation_eligible": confirmation_eligible,
            "confirmation_fixture_ids": (
                [
                    "new-repeated-phrase-final",
                    "independent-phrase-final-only",
                ]
                if confirmation_eligible
                else []
            ),
            "confirmation_requires_separate_committed_protocol": confirmation_eligible,
            "frozen_replication_v1_preserved_failed": True,
            "causal_claims_not_supported": [
                "position",
                "duration",
                "state coupling",
                "population perception",
            ],
        },
    )
