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
    spectral_envelope_warp,
)
from earshift_bakeoff.util import atomic_write_json, sha256_file

import run_bilingual_vowel_adaptive_strength_screen_v1 as adaptive
import run_broad_oral_monophthong_contextual_warp_v1 as broad_v1
import run_broad_vowel_acoustic_feedback_v2 as feedback_v2
import run_english_central_vowel_spectral_correction_v1 as spectral_v1


VERSION = "broad-vowel-jacobian-controller-v1"
RUN_ID = "20260718-broad-vowel-jacobian-controller-v1"
PROTOCOL_PATH = Paths().root / "rules" / f"{VERSION}.json"
RUN_DIR = Paths().artifacts / "product-matrix" / RUN_ID
RESULT_PATH = RUN_DIR / "results.json"
V8_RESULT_PATH = broad_v1.V8_RESULT_PATH
V8_DIR = V8_RESULT_PATH.parent
PROBE_BARK = 0.5
ZERO_COMPONENT_BARK = 0.05
RIDGE_LAMBDA = 0.05
MAXIMUM_CONDITION_NUMBER = 20.0
MAXIMUM_REQUEST_COMPONENT_BARK = 3.5
MAXIMUM_REQUEST_NORM_BARK = 4.0
MAXIMUM_PREDICTED_ERROR_BARK_RMS = 0.18


def _semantic_hash(payload: Any) -> str:
    return hashlib.sha256(stable_json(payload).encode("utf-8")).hexdigest()


def protocol_record() -> dict[str, Any]:
    parent = json.loads(feedback_v2.RESULT_PATH.read_text(encoding="utf-8"))
    catalog = broad_v1._eligible_catalog()
    payload = {
        "schema_version": 1,
        "version": VERSION,
        "run_id": RUN_ID,
        "status": "frozen_before_probe_processing",
        "purpose": (
            "Estimate and invert each occurrence's local two-formant acoustic response "
            "without cumulative edits or scalar-strength selection."
        ),
        "parents": {
            "feedback_v2_result_sha256": sha256_file(feedback_v2.RESULT_PATH),
            "feedback_v2_record_sha256": parent["record_sha256"],
            "calibration_sha256": sha256_file(broad_v1.CALIBRATION_PATH),
            "v8_result_sha256": sha256_file(V8_RESULT_PATH),
        },
        "scope": {
            "cell_ids": [row["cell_id"] for row in catalog],
            "cell_count": len(catalog),
            "logical_slot_count": 135,
            "target_occurrence_count": 180,
            "conditions_per_slot": ["f1_probe", "f2_probe", "solved_final"],
            "maximum_signal_edits": 405,
        },
        "controller": {
            "probe_bark": PROBE_BARK,
            "probe_direction": (
                "The sign of the frozen target component; positive when its absolute "
                "magnitude is below 0.05 Bark."
            ),
            "response_matrix": (
                "Each measured two-formant probe displacement divided by its signed "
                "0.5-Bark requested component."
            ),
            "solver": "ridge least squares (J'J + 0.05 I)^-1 J'd",
            "maximum_condition_number": MAXIMUM_CONDITION_NUMBER,
            "maximum_request_component_bark": MAXIMUM_REQUEST_COMPONENT_BARK,
            "maximum_request_norm_bark": MAXIMUM_REQUEST_NORM_BARK,
            "maximum_predicted_error_bark_rms": MAXIMUM_PREDICTED_ERROR_BARK_RMS,
            "selection": "none; one deterministic solution and one final edit",
        },
        "gates": {
            "probe_integrity": "finite, unclipped, and exact outside edit windows",
            "solution": "finite, bounded, conditioned, plausible, and predicted-error pass",
            "final_acoustic": "same calibrated-ceiling endpoint gate as broad v2",
            "final_engineering": "same complete spectral-v1 engineering gate",
            "aggregation": "all four occurrences across all three contexts",
        },
        "stopping_rule": (
            "Run exactly the two basis probes and at most one solved final per slot. "
            "No alternate probe, regularizer, bound, solution, or rerun."
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
                "scripts/run_broad_vowel_jacobian_controller_v1.py",
                "scripts/run_broad_oral_monophthong_contextual_warp_v1.py",
                "src/earshift_bakeoff/spectral_envelope_warp.py",
                "scripts/run_english_central_vowel_spectral_correction_v1.py",
            )
        },
    }
    return {**payload, "protocol_sha256": sha256_json(payload)}


def prepare() -> dict[str, Any]:
    protocol = protocol_record()
    if PROTOCOL_PATH.exists():
        existing = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        if stable_json(existing) != stable_json(protocol):
            raise RuntimeError("Jacobian-controller protocol drifted")
        return existing
    if RUN_DIR.exists():
        raise RuntimeError("Jacobian-controller run exists before protocol")
    atomic_write_json(PROTOCOL_PATH, protocol)
    return protocol


def _measurement_with_delta(
    neutral: dict[str, Any], delta_bark: Sequence[float]
) -> dict[str, Any]:
    return broad_v1._target_measurement(neutral, list(delta_bark))


def _specs(
    baseline: dict[str, Any],
    neutral: Sequence[dict[str, Any]],
    deltas: Sequence[Sequence[float]],
) -> tuple[FormantWarpSpec, ...]:
    targets = tuple(
        _measurement_with_delta(source, delta)
        for source, delta in zip(neutral, deltas, strict=True)
    )
    specs = []
    for occurrence, source, target in zip(
        baseline["occurrence_outcomes"], neutral, targets, strict=True
    ):
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


def _render_measure(
    *,
    baseline: dict[str, Any],
    neutral_pcm: np.ndarray,
    neutral_measurements: Sequence[dict[str, Any]],
    deltas: Sequence[Sequence[float]],
    ceiling: int,
    label: str,
) -> tuple[Any, dict[str, Any], tuple[dict[str, Any], ...]]:
    specs = _specs(baseline, neutral_measurements, deltas)
    edited = spectral_envelope_warp(
        neutral_pcm,
        specs,
        sample_rate_hz=SAMPLE_RATE_HZ,
        strength=1.0,
    )
    if not (
        edited.metrics["finite"]
        and edited.metrics["outside_windows_bit_exact"]
        and edited.metrics["clipped_sample_count"] == 0
    ):
        raise RuntimeError("probe or final failed universal signal integrity")
    stem = adaptive._safe_name(baseline["logical_slot_id"])
    path = RUN_DIR / "audio" / f"{stem}__{label}.wav"
    audio = adaptive._write_wav(path, edited.pcm)
    intervals = tuple(
        row["measurement_interval"] for row in baseline["occurrence_outcomes"]
    )
    measurements = adaptive._measure(
        path=path,
        stem=f"{stem}__{label}",
        intervals=intervals,
        ceiling=ceiling,
        mode="monophthong_core",
    )
    if not all(row["measurable"] for row in measurements):
        raise RuntimeError("probe or final measurement excluded")
    return edited, audio, measurements


def _probe_deltas(
    target_delta: Sequence[float], dimension: int
) -> tuple[float, float]:
    delta = np.asarray(target_delta, dtype=np.float64)
    sign = 1.0 if abs(float(delta[dimension])) < ZERO_COMPONENT_BARK else math.copysign(
        1.0, float(delta[dimension])
    )
    result = np.zeros(2, dtype=np.float64)
    result[dimension] = sign * PROBE_BARK
    return float(result[0]), float(result[1])


def _solve(
    *,
    neutral: dict[str, Any],
    f1_probe: dict[str, Any],
    f2_probe: dict[str, Any],
    target_delta: Sequence[float],
) -> dict[str, Any]:
    origin = np.asarray(neutral["feature_bark"], dtype=np.float64)
    desired = np.asarray(target_delta, dtype=np.float64)
    probe_requests = np.column_stack(
        (_probe_deltas(desired, 0), _probe_deltas(desired, 1))
    )
    responses = np.column_stack(
        (
            np.asarray(f1_probe["feature_bark"], dtype=np.float64) - origin,
            np.asarray(f2_probe["feature_bark"], dtype=np.float64) - origin,
        )
    )
    jacobian = responses @ np.linalg.inv(probe_requests)
    condition = float(np.linalg.cond(jacobian))
    system = jacobian.T @ jacobian + RIDGE_LAMBDA * np.eye(2)
    request = np.linalg.solve(system, jacobian.T @ desired)
    predicted = jacobian @ request
    error = math.sqrt(float(np.mean(np.square(predicted - desired))))
    finite = bool(
        np.isfinite(jacobian).all()
        and np.isfinite(request).all()
        and math.isfinite(condition)
        and math.isfinite(error)
    )
    passed = bool(
        finite
        and condition <= MAXIMUM_CONDITION_NUMBER
        and float(np.max(np.abs(request))) <= MAXIMUM_REQUEST_COMPONENT_BARK
        and float(np.linalg.norm(request)) <= MAXIMUM_REQUEST_NORM_BARK
        and error <= MAXIMUM_PREDICTED_ERROR_BARK_RMS
    )
    return {
        "jacobian": jacobian.tolist(),
        "condition_number": condition,
        "request_bark": request.tolist(),
        "predicted_displacement_bark": predicted.tolist(),
        "desired_displacement_bark": desired.tolist(),
        "predicted_error_bark_rms": error,
        "pass": passed,
    }


def _slot(baseline: dict[str, Any], cell: dict[str, Any]) -> dict[str, Any]:
    contextual = broad_v1._contextual_baseline(baseline, cell)
    analysis = contextual["analysis_by_formant_ceiling"][0]
    ceiling = int(analysis["maximum_formant_hz"])
    neutral_measurements = tuple(analysis["measurements"]["neutral"])
    target_measurements = tuple(analysis["measurements"]["target_anchor"])
    target_delta = tuple(
        tuple(float(value) for value in cell["prototype_displacement_bark"])
        for _ in neutral_measurements
    )
    neutral_path = V8_DIR / baseline["audio"]["neutral"]["relative_path"]
    neutral_pcm = adaptive._read_wav(neutral_path)
    f1_deltas = tuple(_probe_deltas(delta, 0) for delta in target_delta)
    f2_deltas = tuple(_probe_deltas(delta, 1) for delta in target_delta)
    _, f1_audio, f1_measurements = _render_measure(
        baseline=baseline,
        neutral_pcm=neutral_pcm,
        neutral_measurements=neutral_measurements,
        deltas=f1_deltas,
        ceiling=ceiling,
        label="jacobian-f1-probe",
    )
    _, f2_audio, f2_measurements = _render_measure(
        baseline=baseline,
        neutral_pcm=neutral_pcm,
        neutral_measurements=neutral_measurements,
        deltas=f2_deltas,
        ceiling=ceiling,
        label="jacobian-f2-probe",
    )
    solutions = tuple(
        _solve(neutral=neutral, f1_probe=f1, f2_probe=f2, target_delta=delta)
        for neutral, f1, f2, delta in zip(
            neutral_measurements,
            f1_measurements,
            f2_measurements,
            target_delta,
            strict=True,
        )
    )
    if not all(row["pass"] for row in solutions):
        raise RuntimeError("Jacobian solution failed frozen bounds")
    requests = tuple(tuple(row["request_bark"]) for row in solutions)
    final, final_audio, final_measurements = _render_measure(
        baseline=baseline,
        neutral_pcm=neutral_pcm,
        neutral_measurements=neutral_measurements,
        deltas=requests,
        ceiling=ceiling,
        label="jacobian-solved-final",
    )
    intervals = tuple(
        row["measurement_interval"] for row in baseline["occurrence_outcomes"]
    )
    engineering = spectral_v1._engineering(neutral_pcm, final, intervals)
    classifications = tuple(
        adaptive._analysis_classification(
            source=neutral,
            target=target,
            neutral=neutral,
            lens=lens,
            rhotic=False,
        )
        for neutral, target, lens in zip(
            neutral_measurements,
            target_measurements,
            final_measurements,
            strict=True,
        )
    )
    classification = (
        adaptive._aggregate(classifications) if engineering["pass"] else "fail"
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
        "status": "measured",
        "classification": classification,
        "exact_category_pass": classification == "exact_category_pass",
        "directional_pass": classification
        in {"exact_category_pass", "directional_only_pass"},
        "selected_maximum_formant_hz": ceiling,
        "prototype_displacement_bark": cell["prototype_displacement_bark"],
        "probe_audio": {"f1": f1_audio, "f2": f2_audio},
        "solutions": solutions,
        "final_audio": final_audio,
        "final_engineering": engineering,
        "final_measurements": final_measurements,
        "occurrence_classifications": classifications,
        "signal_edit_count": 3,
        "api_calls_made": 0,
        "production_enabled": False,
    }


def _excluded(baseline: dict[str, Any], exc: Exception) -> dict[str, Any]:
    return {
        "logical_slot_id": baseline["logical_slot_id"],
        "cell_id": baseline["cell_id"],
        "profile_id": baseline["profile_id"],
        "voice_id": baseline["voice_id"],
        "rule_id": baseline["rule_id"],
        "context": baseline["context"],
        "source": baseline["source"],
        "target": baseline["target"],
        "status": "processing_exclusion",
        "classification": "fail",
        "exact_category_pass": False,
        "directional_pass": False,
        "signal_edit_count": 0,
        "error": f"{type(exc).__name__}: {exc}",
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
        raise RuntimeError("Jacobian-controller protocol or sources drifted")
    catalog = {row["cell_id"]: row for row in broad_v1._eligible_catalog()}
    v8 = json.loads(V8_RESULT_PATH.read_text(encoding="utf-8"))
    baselines = [row for row in v8["outcomes"] if row["cell_id"] in catalog]
    RUN_DIR.mkdir(parents=True, exist_ok=False)
    old_adaptive = adaptive.RUN_DIR
    adaptive.RUN_DIR = RUN_DIR
    outcomes = []
    started = time.perf_counter()
    try:
        for baseline in baselines:
            try:
                outcome = _slot(baseline, catalog[baseline["cell_id"]])
            except (RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
                outcome = _excluded(baseline, exc)
            outcomes.append(outcome)
    finally:
        adaptive.RUN_DIR = old_adaptive
    cells = _summaries(outcomes)
    payload = {
        "schema_version": 1,
        "version": VERSION,
        "run_id": RUN_ID,
        "protocol_sha256": protocol["protocol_sha256"],
        "classification": "broad_vowel_jacobian_known_fixture_characterization",
        "cell_classification_counts": dict(
            sorted(Counter(row["classification"] for row in cells).items())
        ),
        "cell_count": len(cells),
        "logical_slot_count": len(outcomes),
        # Count artifacts rather than successful outcome labels: a bounded slot may
        # legitimately stop after writing one or both probes.
        "signal_edit_count": len(tuple((RUN_DIR / "audio").glob("*.wav"))),
        "status_counts": dict(sorted(Counter(row["status"] for row in outcomes).items())),
        "kokoro_renders": 0,
        "api_calls_made": 0,
        "production_enabled": False,
        "elapsed_s": time.perf_counter() - started,
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
