#!/usr/bin/env python3
from __future__ import annotations

from collections import Counter, defaultdict
import argparse
import hashlib
import json
import math
import time
from typing import Any, Sequence

import numpy as np

from earshift_bakeoff.config import Paths, sha256_json, stable_json
from earshift_bakeoff.kokoro_synthesis import SAMPLE_RATE_HZ
from earshift_bakeoff.spectral_envelope_warp import (
    FormantWarpSpec,
    SpectralEnvelopeWarpResult,
    spectral_envelope_warp,
)
from earshift_bakeoff.util import atomic_write_json, sha256_file

import run_bilingual_vowel_adaptive_strength_screen_v1 as adaptive
import run_broad_oral_monophthong_contextual_warp_v1 as broad_v1
import run_broad_oral_monophthong_contextual_warp_v2 as broad_v2
import run_english_central_vowel_spectral_correction_v1 as spectral_v1


VERSION = "broad-vowel-acoustic-feedback-v1"
RUN_ID = "20260718-broad-vowel-acoustic-feedback-v1"
PROTOCOL_PATH = Paths().root / "rules" / f"{VERSION}.json"
RUN_DIR = Paths().artifacts / "product-matrix" / RUN_ID
RESULT_PATH = RUN_DIR / "results.json"
V8_RESULT_PATH = broad_v1.V8_RESULT_PATH
V8_DIR = V8_RESULT_PATH.parent
MAXIMUM_ITERATIONS = 3
FEEDBACK_GAIN = 0.75


def _semantic_hash(payload: Any) -> str:
    return hashlib.sha256(stable_json(payload).encode("utf-8")).hexdigest()


def protocol_record() -> dict[str, Any]:
    parent = json.loads(broad_v2.RESULT_PATH.read_text(encoding="utf-8"))
    catalog = broad_v1._eligible_catalog()
    payload = {
        "schema_version": 1,
        "version": VERSION,
        "run_id": RUN_ID,
        "status": "frozen_before_feedback_processing",
        "purpose": (
            "Replace the failed one-shot category assumption with bounded deterministic "
            "measurement feedback across the same broad oral-monophthong denominator."
        ),
        "parents": {
            "broad_v2_result_sha256": sha256_file(broad_v2.RESULT_PATH),
            "broad_v2_record_sha256": parent["record_sha256"],
            "calibration_sha256": sha256_file(broad_v1.CALIBRATION_PATH),
            "v8_result_sha256": sha256_file(V8_RESULT_PATH),
        },
        "scope": {
            "cell_ids": [row["cell_id"] for row in catalog],
            "cell_count": len(catalog),
            "logical_slot_count": 135,
            "target_occurrence_count": 180,
            "maximum_feedback_edits": 135 * MAXIMUM_ITERATIONS,
        },
        "controller": {
            "version": VERSION,
            "maximum_iterations": MAXIMUM_ITERATIONS,
            "feedback_gain": FEEDBACK_GAIN,
            "target": (
                "initial neutral Bark F1/F2 plus the frozen voice/rule natural-anchor "
                "prototype displacement"
            ),
            "update": (
                "At each iteration, measure current F1/F2 and warp from that measured "
                "point toward the fixed target with gain 0.75."
            ),
            "stop": (
                "Stop early only when every occurrence in the slot is exact-category "
                "and the cumulative waveform passes all engineering gates. Otherwise "
                "the third returned state is final; no best-iteration selection."
            ),
        },
        "gates": {
            "acoustic": "same calibrated-ceiling endpoint classifier as broad v2",
            "engineering": (
                "same exact-outside, finite, clipping, cumulative RMS, high-band, "
                "boundary, and localization gates against the original neutral"
            ),
            "invalid_geometry": "fail closed without alternative gain or target",
        },
        "stopping_rule": (
            "Run every slot once through at most three deterministic feedback edits. "
            "No best-iteration selection, rerun, gain search, replacement, or "
            "listening selection."
        ),
        "scope_controls": {
            "kokoro_renders": 0,
            "api_calls": 0,
            "paid_calls": 0,
            "production_enabled": False,
            "deployment": False,
        },
        "source_bindings": {
            path: sha256_file(Paths().root / path)
            for path in (
                "scripts/run_broad_vowel_acoustic_feedback_v1.py",
                "scripts/run_broad_oral_monophthong_contextual_warp_v1.py",
                "scripts/run_broad_oral_monophthong_contextual_warp_v2.py",
                "src/earshift_bakeoff/spectral_envelope_warp.py",
            )
        },
    }
    return {**payload, "protocol_sha256": sha256_json(payload)}


def prepare() -> dict[str, Any]:
    protocol = protocol_record()
    if PROTOCOL_PATH.exists():
        existing = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        if stable_json(existing) != stable_json(protocol):
            raise RuntimeError("acoustic-feedback protocol drifted")
        return existing
    if RUN_DIR.exists():
        raise RuntimeError("acoustic-feedback run exists before protocol")
    atomic_write_json(PROTOCOL_PATH, protocol)
    return protocol


def _rms_change(before: np.ndarray, after: np.ndarray) -> float:
    left = before.astype(np.float64) / 32768.0
    right = after.astype(np.float64) / 32768.0
    left_rms = math.sqrt(float(np.mean(np.square(left))))
    right_rms = math.sqrt(float(np.mean(np.square(right))))
    return 20.0 * math.log10(max(right_rms, 1e-12) / max(left_rms, 1e-12))


def _cumulative_result(
    original: np.ndarray,
    current: np.ndarray,
    latest: SpectralEnvelopeWarpResult,
    clipped_count: int,
) -> SpectralEnvelopeWarpResult:
    rms = []
    for window in latest.edit_windows:
        start = window["start_sample"]
        end = window["end_sample_exclusive"]
        rms.append(_rms_change(original[start:end], current[start:end]))
    outside = latest.weights == 0.0
    return SpectralEnvelopeWarpResult(
        pcm=current,
        weights=latest.weights,
        edit_windows=latest.edit_windows,
        metrics={
            "identity": False,
            "outside_windows_bit_exact": bool(
                np.array_equal(original[outside], current[outside])
            ),
            "finite": bool(np.isfinite(current.astype(np.float64)).all()),
            "clipped_sample_count": clipped_count,
            "rms_db_change_by_window": rms,
        },
    )


def _specs(
    baseline: dict[str, Any],
    current: Sequence[dict[str, Any]],
    targets: Sequence[dict[str, Any]],
) -> tuple[FormantWarpSpec, ...]:
    specs = []
    for occurrence, source, target in zip(
        baseline["occurrence_outcomes"], current, targets, strict=True
    ):
        if (
            not source["measurable"]
            or not target["measurable"]
            or len(source["bins"]) != 1
            or len(target["bins"]) != 1
        ):
            raise RuntimeError("feedback endpoint is not a measurable monophthong")
        source_bin = source["bins"][0]
        target_bin = target["bins"][0]
        interval = occurrence["measurement_interval"]
        specs.append(
            FormantWarpSpec(
                start_sample=int(interval["start_sample"]),
                end_sample_exclusive=int(interval["end_sample_exclusive"]),
                source_f1_hz=float(source_bin["f1_hz"]),
                source_f2_hz=float(source_bin["f2_hz"]),
                target_f1_hz=float(target_bin["f1_hz"]),
                target_f2_hz=float(target_bin["f2_hz"]),
            )
        )
    return tuple(specs)


def _classifications(
    neutral: Sequence[dict[str, Any]],
    targets: Sequence[dict[str, Any]],
    current: Sequence[dict[str, Any]],
) -> tuple[dict[str, Any], ...]:
    return tuple(
        adaptive._analysis_classification(
            source=source,
            target=target,
            neutral=source,
            lens=lens,
            rhotic=False,
        )
        for source, target, lens in zip(neutral, targets, current, strict=True)
    )


def _slot(baseline: dict[str, Any], cell: dict[str, Any]) -> dict[str, Any]:
    contextual = broad_v1._contextual_baseline(baseline, cell)
    analysis = contextual["analysis_by_formant_ceiling"][0]
    ceiling = int(analysis["maximum_formant_hz"])
    neutral_measurements = tuple(analysis["measurements"]["neutral"])
    targets = tuple(analysis["measurements"]["target_anchor"])
    neutral_path = V8_DIR / baseline["audio"]["neutral"]["relative_path"]
    original = adaptive._read_wav(neutral_path)
    current = original.copy()
    current_measurements = neutral_measurements
    intervals = tuple(
        row["measurement_interval"] for row in baseline["occurrence_outcomes"]
    )
    stem = adaptive._safe_name(baseline["logical_slot_id"])
    iterations = []
    cumulative_clipping = 0
    final_engineering = None
    status = "maximum_iterations_reached"
    for iteration in range(1, MAXIMUM_ITERATIONS + 1):
        try:
            specs = _specs(baseline, current_measurements, targets)
            edited = spectral_envelope_warp(
                current,
                specs,
                sample_rate_hz=SAMPLE_RATE_HZ,
                strength=FEEDBACK_GAIN,
            )
            cumulative_clipping += int(edited.metrics["clipped_sample_count"])
            current = edited.pcm
            cumulative = _cumulative_result(
                original, current, edited, cumulative_clipping
            )
            engineering = spectral_v1._engineering(original, cumulative, intervals)
            path = RUN_DIR / "audio" / f"{stem}__feedback-{iteration}.wav"
            audio = adaptive._write_wav(path, current)
            current_measurements = adaptive._measure(
                path=path,
                stem=f"{stem}__feedback-{iteration}",
                intervals=intervals,
                ceiling=ceiling,
                mode="monophthong_core",
            )
            classifications = _classifications(
                neutral_measurements, targets, current_measurements
            )
            classification = adaptive._aggregate(classifications)
            iterations.append(
                {
                    "iteration": iteration,
                    "classification": classification,
                    "engineering": engineering,
                    "audio": audio,
                    "measurements": current_measurements,
                    "occurrence_classifications": classifications,
                }
            )
            final_engineering = engineering
            if classification == "exact_category_pass" and engineering["pass"]:
                status = "exact_category_converged"
                break
            if not engineering["pass"]:
                status = "engineering_gate_failed"
                break
        except (RuntimeError, ValueError) as exc:
            status = "processing_exclusion"
            iterations.append(
                {
                    "iteration": iteration,
                    "classification": "fail",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            break
    final_classifications = (
        iterations[-1].get("occurrence_classifications", ()) if iterations else ()
    )
    classification = (
        adaptive._aggregate(final_classifications)
        if final_classifications and final_engineering and final_engineering["pass"]
        else "fail"
    )
    return {
        "logical_slot_id": baseline["logical_slot_id"],
        "cell_id": baseline["cell_id"],
        "profile_id": baseline["profile_id"],
        "voice_id": baseline["voice_id"],
        "rule_id": baseline["rule_id"],
        "context": baseline["context"],
        "source": baseline["source"],
        "target": baseline["target"],
        "status": status,
        "classification": classification,
        "exact_category_pass": classification == "exact_category_pass",
        "directional_pass": classification
        in {"exact_category_pass", "directional_only_pass"},
        "selected_maximum_formant_hz": ceiling,
        "prototype_displacement_bark": cell["prototype_displacement_bark"],
        "iteration_count": len(iterations),
        "iterations": iterations,
        "api_calls_made": 0,
        "production_enabled": False,
    }


def _summaries(outcomes: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for outcome in outcomes:
        groups[outcome["cell_id"]].append(outcome)
    rows = []
    for cell_id, slots in sorted(groups.items()):
        classification = adaptive._aggregate(slots) if len(slots) == 3 else "fail"
        rows.append(
            {
                "cell_id": cell_id,
                "voice_id": slots[0]["voice_id"],
                "rule_id": slots[0]["rule_id"],
                "classification": classification,
                "production_enabled": False,
            }
        )
    return rows


def run() -> dict[str, Any]:
    if RESULT_PATH.exists():
        return json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if stable_json(protocol) != stable_json(protocol_record()):
        raise RuntimeError("acoustic-feedback protocol or sources drifted")
    catalog = {row["cell_id"]: row for row in broad_v1._eligible_catalog()}
    v8 = json.loads(V8_RESULT_PATH.read_text(encoding="utf-8"))
    baselines = [row for row in v8["outcomes"] if row["cell_id"] in catalog]
    RUN_DIR.mkdir(parents=True, exist_ok=False)
    old_adaptive = adaptive.RUN_DIR
    adaptive.RUN_DIR = RUN_DIR
    started = time.perf_counter()
    try:
        outcomes = [_slot(row, catalog[row["cell_id"]]) for row in baselines]
    finally:
        adaptive.RUN_DIR = old_adaptive
    cells = _summaries(outcomes)
    counts = Counter(row["classification"] for row in cells)
    payload = {
        "schema_version": 1,
        "version": VERSION,
        "run_id": RUN_ID,
        "protocol_sha256": protocol["protocol_sha256"],
        "classification": "broad_vowel_acoustic_feedback_known_fixture_characterization",
        "cell_classification_counts": dict(sorted(counts.items())),
        "cell_count": len(cells),
        "logical_slot_count": len(outcomes),
        "feedback_edit_count": sum(row["iteration_count"] for row in outcomes),
        "kokoro_renders": 0,
        "api_calls_made": 0,
        "production_enabled": False,
        "elapsed_s": time.perf_counter() - started,
        "status_counts": dict(sorted(Counter(row["status"] for row in outcomes).items())),
        "cell_summaries": cells,
        "outcomes": outcomes,
    }
    result = {**payload, "record_sha256": _semantic_hash(payload)}
    atomic_write_json(RESULT_PATH, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "run"))
    args = parser.parse_args()
    result = prepare() if args.command == "prepare" else run()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
