from __future__ import annotations

import json
import math
import os
import subprocess
import time
import wave
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .config import Paths, sha256_json, stable_json
from .kokoro_synthesis import SAMPLE_RATE_HZ
from .kokoro_typed_confirmation import _family_gate, _measure_occurrences
from .kokoro_typed_confirmation_protocol import (
    CEILINGS_HZ,
    PRIMARY_WINDOW_PERCENT,
    WINDOW_PERCENTS,
)
from .kokoro_typed_diagnostic import localization_report
from .kokoro_typed_engine import MAX_CLIPPED_FRACTION
from .util import atomic_write_json, sha256_file


RUN_ID = "20260717-kokoro-output-domain-splice-v1"
PARENT_RUN_ID = "20260717-kokoro-typed-confirmation-v1"
PROTOCOL_FILE = "protocol.json"
ANALYSIS_FILE = "analysis.json"
ADJUDICATION_FILE = "adjudication.json"
TAPER_MS = 10.0
TAPER_SAMPLES = round(TAPER_MS * SAMPLE_RATE_HZ / 1000.0)
BOUNDARY_CONTEXT_MS = 10.0
BOUNDARY_CONTEXT_SAMPLES = round(BOUNDARY_CONTEXT_MS * SAMPLE_RATE_HZ / 1000.0)
MAX_EDGE_DELTA_STEP_PCM = 1.0
MAX_BOUNDARY_DERIVATIVE_RATIO = 1.25
LOCALIZATION_MINIMUM = 0.80
BENCHMARK_WARMUP_ITERATIONS = 100
BENCHMARK_MEASURED_ITERATIONS = 2_000
MAX_LOCALIZATION_MEDIAN_MS = 5.0
MAX_LOCALIZATION_P95_MS = 10.0
EXPECTED_PARENT_CLASSIFICATION = (
    "fresh_unseen_fixture_confirmation_automatic_failed_no_review"
)
EXPECTED_FIXTURES = (
    "new-repeated-phrase-final",
    "independent-phrase-final-only",
)


def run_dir() -> Path:
    return Paths().artifacts / "typed-engine" / RUN_ID


def parent_dir() -> Path:
    return Paths().artifacts / "typed-engine" / PARENT_RUN_ID


def _load_json(path: Path) -> Any:
    if not path.is_file():
        raise RuntimeError(f"required artifact is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _write_once_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        existing = _load_json(path)
        if stable_json(existing) != stable_json(payload):
            raise RuntimeError(f"immutable artifact differs from recomputation: {path}")
        return
    atomic_write_json(path, payload)


def _read_pcm16(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        sample_width = handle.getsampwidth()
        sample_rate = handle.getframerate()
        sample_count = handle.getnframes()
        payload = handle.readframes(sample_count)
    if channels != 1 or sample_width != 2 or sample_rate != SAMPLE_RATE_HZ:
        raise RuntimeError(f"WAV violates frozen mono PCM16/24 kHz contract: {path}")
    values = np.frombuffer(payload, dtype="<i2").copy()
    if values.size != sample_count or not values.size:
        raise RuntimeError(f"WAV is empty or truncated: {path}")
    return values, sample_rate


def _write_pcm16_once(path: Path, values: np.ndarray) -> None:
    if path.exists():
        raise RuntimeError(f"candidate WAV already exists: {path}")
    audio = np.asarray(values, dtype="<i2").reshape(-1)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    try:
        with wave.open(str(temporary), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(SAMPLE_RATE_HZ)
            handle.writeframes(audio.tobytes())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _pcm_sha256(values: np.ndarray) -> str:
    import hashlib

    return hashlib.sha256(np.asarray(values, dtype="<i2").tobytes()).hexdigest()


def _normalized_windows(
    windows: Sequence[dict[str, Any]], sample_count: int
) -> tuple[tuple[int, int], ...]:
    normalized: list[tuple[int, int]] = []
    for row in windows:
        start = int(row["start_sample"])
        end = int(row["end_sample_exclusive"])
        if not 0 <= start < end <= sample_count:
            raise RuntimeError("splice window is outside the bound parent PCM")
        if end - start < 2 * TAPER_SAMPLES:
            raise RuntimeError("splice window is too short for the frozen taper")
        if normalized and start < normalized[-1][1]:
            raise RuntimeError("splice windows overlap or are not ordered")
        normalized.append((start, end))
    if not normalized:
        raise RuntimeError("no frozen splice windows were supplied")
    return tuple(normalized)


def raised_cosine_weights(
    sample_count: int,
    windows: Sequence[dict[str, Any]],
    *,
    taper_samples: int = TAPER_SAMPLES,
) -> np.ndarray:
    if sample_count <= 0 or taper_samples < 2:
        raise ValueError("invalid output-domain splice dimensions")
    if taper_samples != TAPER_SAMPLES:
        raise ValueError("only the frozen 10 ms taper is permitted")
    normalized = _normalized_windows(windows, sample_count)
    weights = np.zeros(sample_count, dtype=np.float64)
    phase = np.linspace(0.0, math.pi, taper_samples, endpoint=True)
    fade = 0.5 - 0.5 * np.cos(phase)
    for start, end in normalized:
        local = np.ones(end - start, dtype=np.float64)
        local[:taper_samples] = fade
        local[-taper_samples:] = fade[::-1]
        weights[start:end] = np.maximum(weights[start:end], local)
    return weights


def output_domain_splice(
    neutral: np.ndarray,
    lens: np.ndarray,
    windows: Sequence[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray]:
    left = np.asarray(neutral, dtype="<i2").reshape(-1)
    right = np.asarray(lens, dtype="<i2").reshape(-1)
    if left.shape != right.shape or not left.size:
        raise ValueError("neutral and lens PCM must have the same nonzero shape")
    weights = raised_cosine_weights(left.size, windows)
    candidate = np.rint(
        left.astype(np.float64)
        + weights * (right.astype(np.float64) - left.astype(np.float64))
    )
    candidate = np.clip(candidate, -32768, 32767).astype("<i2")
    return candidate, weights


def boundary_artifact_report(
    neutral: np.ndarray,
    lens: np.ndarray,
    candidate: np.ndarray,
    windows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    left = np.asarray(neutral, dtype=np.int64).reshape(-1)
    right = np.asarray(lens, dtype=np.int64).reshape(-1)
    output = np.asarray(candidate, dtype=np.int64).reshape(-1)
    if not (left.shape == right.shape == output.shape) or not left.size:
        return {"pass": False, "reason": "shape_mismatch"}
    normalized = _normalized_windows(windows, left.size)
    delta = output - left
    boundaries: list[dict[str, Any]] = []
    for start, end in normalized:
        for edge, sample in (("start", start), ("end", end)):
            if edge == "start":
                before = delta[sample - 1] if sample else 0
                after = delta[sample]
            else:
                before = delta[sample - 1]
                after = delta[sample] if sample < delta.size else 0
            edge_step = float(abs(after - before))
            lo = max(0, sample - BOUNDARY_CONTEXT_SAMPLES)
            hi = min(left.size, sample + BOUNDARY_CONTEXT_SAMPLES + 1)

            def peak_derivative(values: np.ndarray) -> float:
                section = values[lo:hi]
                return (
                    float(np.max(np.abs(np.diff(section)), initial=0))
                    if section.size > 1
                    else 0.0
                )

            candidate_peak = peak_derivative(output)
            neutral_peak = peak_derivative(left)
            lens_peak = peak_derivative(right)
            reference_peak = max(neutral_peak, lens_peak, 1.0)
            ratio = candidate_peak / reference_peak
            boundaries.append(
                {
                    "edge": edge,
                    "sample": sample,
                    "edge_delta_step_pcm": edge_step,
                    "candidate_peak_first_difference_pcm": candidate_peak,
                    "neutral_peak_first_difference_pcm": neutral_peak,
                    "lens_peak_first_difference_pcm": lens_peak,
                    "reference_peak_first_difference_pcm": reference_peak,
                    "candidate_to_reference_derivative_ratio": ratio,
                    "edge_step_pass": edge_step <= MAX_EDGE_DELTA_STEP_PCM,
                    "derivative_ratio_pass": ratio
                    <= MAX_BOUNDARY_DERIVATIVE_RATIO,
                }
            )
    return {
        "metric": "edge-delta-step plus local peak-first-difference ratio",
        "context_samples_each_side": BOUNDARY_CONTEXT_SAMPLES,
        "context_ms_each_side": BOUNDARY_CONTEXT_MS,
        "maximum_edge_delta_step_pcm": max(
            row["edge_delta_step_pcm"] for row in boundaries
        ),
        "maximum_candidate_to_reference_derivative_ratio": max(
            row["candidate_to_reference_derivative_ratio"] for row in boundaries
        ),
        "maximum_allowed_edge_delta_step_pcm": MAX_EDGE_DELTA_STEP_PCM,
        "maximum_allowed_derivative_ratio": MAX_BOUNDARY_DERIVATIVE_RATIO,
        "boundaries": boundaries,
        "pass": all(
            row["edge_step_pass"] and row["derivative_ratio_pass"]
            for row in boundaries
        ),
    }


def _gate_signature(windows: dict[str, Any]) -> dict[str, Any]:
    return {
        window: {
            "pass": row["pass"],
            "occurrences": [
                {
                    "occurrence_index": occurrence["occurrence_index"],
                    "pass": occurrence["pass"],
                    "families": {
                        key: {
                            "pass": family["pass"],
                            "checks": family["checks"],
                        }
                        for key, family in sorted(occurrence["families"].items())
                    },
                }
                for occurrence in row["occurrences"]
            ],
        }
        for window, row in sorted(windows.items())
    }


def _acoustic_report(
    neutral_path: Path,
    candidate_path: Path,
    record: dict[str, Any],
    parent_fixture: dict[str, Any],
    anchors: dict[str, Any],
) -> dict[str, Any]:
    occurrences = record["alignment"]["target_occurrences"]
    neutral_measurements = _measure_occurrences(neutral_path, occurrences)
    candidate_measurements = _measure_occurrences(candidate_path, occurrences)
    windows: dict[str, Any] = {}
    for percent in WINDOW_PERCENTS:
        window_key = str(percent)
        rows: list[dict[str, Any]] = []
        for occurrence_index, occurrence in enumerate(occurrences):
            anchor_index = int(occurrence["anchor_occurrence_index"])
            families: dict[str, Any] = {}
            for ceiling in CEILINGS_HZ:
                key = str(ceiling)
                families[key] = _family_gate(
                    neutral_measurements[occurrence_index]["families"][key][
                        window_key
                    ],
                    candidate_measurements[occurrence_index]["families"][key][
                        window_key
                    ],
                    anchors[window_key]["occurrences"][anchor_index]["families"][
                        key
                    ],
                )
            rows.append(
                {
                    "occurrence_index": occurrence_index,
                    "anchor_occurrence_index": anchor_index,
                    "position": occurrence["position"],
                    "families": families,
                    "pass": all(item["pass"] for item in families.values()),
                }
            )
        windows[window_key] = {
            "occurrences": rows,
            "pass": all(item["pass"] for item in rows),
        }
    baseline_signature = _gate_signature(parent_fixture["windows"])
    candidate_signature = _gate_signature(windows)
    return {
        "neutral_measurements": neutral_measurements,
        "candidate_measurements": candidate_measurements,
        "windows": windows,
        "primary_window_percent": PRIMARY_WINDOW_PERCENT,
        "primary_gate_pass": bool(windows[str(PRIMARY_WINDOW_PERCENT)]["pass"]),
        "all_existing_gate_booleans_preserved": candidate_signature
        == baseline_signature,
        "baseline_gate_signature": baseline_signature,
        "candidate_gate_signature": candidate_signature,
        "pass": bool(
            windows[str(PRIMARY_WINDOW_PERCENT)]["pass"]
            and candidate_signature == baseline_signature
        ),
    }


def _clipped_fraction(values: np.ndarray) -> float:
    audio = np.asarray(values, dtype=np.int64).reshape(-1)
    return float(np.mean(np.abs(audio) >= 32767)) if audio.size else 1.0


def _integrity_report(
    neutral: np.ndarray,
    lens: np.ndarray,
    candidate: np.ndarray,
    weights: np.ndarray,
    record: dict[str, Any],
    candidate_path: Path,
    parent_hashes: set[str],
) -> dict[str, Any]:
    left = np.asarray(neutral, dtype="<i2").reshape(-1)
    right = np.asarray(lens, dtype="<i2").reshape(-1)
    output = np.asarray(candidate, dtype="<i2").reshape(-1)
    outside = weights == 0.0
    full_lens = weights == 1.0
    candidate_wav_sha256 = sha256_file(candidate_path)
    checks = {
        "nonempty_mono_pcm16_24khz": bool(output.size),
        "sample_count_equal": bool(left.size == right.size == output.size),
        "finite": bool(np.isfinite(output.astype(np.float64)).all()),
        "clipping_pass": _clipped_fraction(output) < MAX_CLIPPED_FRACTION,
        "neutral_identity_parent_gate_preserved": bool(
            record["neutral_identity_bit_identical"]
        ),
        "parent_render_runtime_pass_preserved": bool(record["runtime_pass"]),
        "parent_pair_integrity_pass_preserved": bool(
            record["pair_integrity"]["pass_all"]
        ),
        "alignment_and_duration_plan_unchanged": bool(
            record["alignment"]["duration_count"] > 0
            and record["alignment"]["expected_replaced_columns"]
        ),
        "outside_splice_windows_bit_identical_to_neutral": bool(
            np.array_equal(output[outside], left[outside])
        ),
        "full_weight_interior_bit_identical_to_lens": bool(
            full_lens.any() and np.array_equal(output[full_lens], right[full_lens])
        ),
        "candidate_wav_disjoint_from_bound_parent_wavs": candidate_wav_sha256
        not in parent_hashes,
    }
    return {
        "sample_count": int(output.size),
        "candidate_clipped_fraction": _clipped_fraction(output),
        "candidate_pcm_sha256": _pcm_sha256(output),
        "candidate_wav_sha256": candidate_wav_sha256,
        "checks": checks,
        "pass": all(checks.values()),
    }


def _benchmark_localization(
    neutral: np.ndarray,
    candidate: np.ndarray,
    target_intervals: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    for _ in range(BENCHMARK_WARMUP_ITERATIONS):
        localization_report(neutral, candidate, target_intervals)
    elapsed_ms: list[float] = []
    for _ in range(BENCHMARK_MEASURED_ITERATIONS):
        started = time.perf_counter_ns()
        localization_report(neutral, candidate, target_intervals)
        elapsed_ms.append((time.perf_counter_ns() - started) / 1_000_000.0)
    values = np.asarray(elapsed_ms, dtype=np.float64)
    identical = localization_report(neutral, neutral, target_intervals)
    mismatched = localization_report(neutral, candidate[:-1], target_intervals)
    median_ms = float(np.median(values))
    p95_ms = float(np.percentile(values, 95))
    fail_closed = bool(
        identical.get("pass") is False and mismatched.get("pass") is False
    )
    return {
        "clock": "time.perf_counter_ns",
        "warmup_iterations": BENCHMARK_WARMUP_ITERATIONS,
        "measured_iterations": BENCHMARK_MEASURED_ITERATIONS,
        "median_ms": median_ms,
        "p95_ms": p95_ms,
        "minimum_ms": float(np.min(values)),
        "maximum_ms": float(np.max(values)),
        "cheap_thresholds_ms": {
            "median_max": MAX_LOCALIZATION_MEDIAN_MS,
            "p95_max": MAX_LOCALIZATION_P95_MS,
        },
        "fail_closed_checks": {
            "identical_pair_rejected": identical.get("pass") is False,
            "sample_count_mismatch_rejected": mismatched.get("pass") is False,
        },
        "pass": bool(
            median_ms <= MAX_LOCALIZATION_MEDIAN_MS
            and p95_ms <= MAX_LOCALIZATION_P95_MS
            and fail_closed
        ),
    }


def _parent_payloads() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    protocol = _load_json(parent_dir() / "protocol.json")
    records = _load_json(parent_dir() / "render-records.json")
    analysis = _load_json(parent_dir() / "analysis.json")
    if analysis.get("classification") != EXPECTED_PARENT_CLASSIFICATION:
        raise RuntimeError("frozen parent confirmation classification drifted")
    if analysis.get("claim") != "no_positive_generalization_claim":
        raise RuntimeError("frozen parent confirmation claim drifted")
    fixture_ids = tuple(row["fixture_id"] for row in analysis["fixtures"])
    if fixture_ids != EXPECTED_FIXTURES:
        raise RuntimeError("frozen parent fixture inventory drifted")
    return protocol, records, analysis


def protocol_record() -> dict[str, Any]:
    parent_protocol, records, analysis = _parent_payloads()
    records_by_id = {row["fixture_id"]: row for row in records["fixtures"]}
    analysis_by_id = {row["fixture_id"]: row for row in analysis["fixtures"]}
    fixtures: list[dict[str, Any]] = []
    for fixture_id in EXPECTED_FIXTURES:
        record = records_by_id[fixture_id]
        result = analysis_by_id[fixture_id]
        fixtures.append(
            {
                "fixture_id": fixture_id,
                "neutral": record["audio"]["neutral"],
                "lens": record["audio"]["lens"],
                "identity": record["audio"]["identity"],
                "plan_sha256": record["plan_sha256"],
                "target_word_intervals": [
                    row["interval"] for row in record["alignment"]["target_words"]
                ],
                "measurement_intervals": [
                    row["measurement_interval"]
                    for row in record["alignment"]["target_occurrences"]
                ],
                "splice_windows": result["localization"]["inside_windows"],
                "baseline_localization": result["localization"],
                "baseline_primary_acoustic_pass": result["windows"][
                    str(PRIMARY_WINDOW_PERCENT)
                ]["pass"],
            }
        )
    source_path = Path(__file__)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "status": "frozen_before_candidate_artifacts",
        "intervention_name": "output-domain splice",
        "question": (
            "Can one deterministic decoded-PCM splice eliminate out-of-neighborhood "
            "difference energy on both frozen confirmation fixtures without changing "
            "their existing acoustic/integrity verdicts or introducing boundary clicks?"
        ),
        "scope": {
            "candidate_count": 1,
            "fixture_count": 2,
            "candidate_wav_count": 2,
            "api_calls": 0,
            "model_decodes": 0,
            "selection": "none",
            "product_integration": False,
        },
        "parent": {
            "run_id": PARENT_RUN_ID,
            "protocol_sha256": parent_protocol["protocol_sha256"],
            "protocol_file_sha256": sha256_file(parent_dir() / "protocol.json"),
            "render_records_file_sha256": sha256_file(
                parent_dir() / "render-records.json"
            ),
            "analysis_file_sha256": sha256_file(parent_dir() / "analysis.json"),
            "classification": analysis["classification"],
            "claim": analysis["claim"],
        },
        "candidate": {
            "version": "output-domain-splice-v1",
            "boundary_rule": (
                "Use each untouched parent fixture's frozen localization inside_windows "
                "start_sample and end_sample_exclusive verbatim; do not realign, search, "
                "expand, shrink, or select a window."
            ),
            "mix_rule": "candidate = round(neutral + weight * (lens - neutral)) in PCM16",
            "outside_rule": "weight is exactly zero outside the frozen windows",
            "interior_rule": "weight is exactly one after each leading taper and before each trailing taper",
            "taper": {
                "type": "raised-cosine Hann half-window",
                "duration_ms_each_edge": TAPER_MS,
                "samples_each_edge": TAPER_SAMPLES,
                "formula": "0.5 - 0.5*cos(linspace(0, pi, 240, endpoint=true))",
            },
        },
        "artifact_gate": {
            "boundary_context_ms_each_side": BOUNDARY_CONTEXT_MS,
            "boundary_context_samples_each_side": BOUNDARY_CONTEXT_SAMPLES,
            "edge_delta_step_pcm_max": MAX_EDGE_DELTA_STEP_PCM,
            "candidate_to_reference_peak_first_difference_ratio_max": MAX_BOUNDARY_DERIVATIVE_RATIO,
            "reference": "maximum local peak first difference of untouched neutral and untouched lens",
            "fixture_pass": "every start/end boundary passes both metrics",
        },
        "automatic_gates": {
            "localization": {
                "implementation": "unchanged localization_report",
                "minimum_inside_difference_energy_fraction": LOCALIZATION_MINIMUM,
            },
            "acoustic": (
                "Rerun the complete 5500/5750/6000 Hz x 40/50/60% family; require "
                "the primary-50 gate to pass and every boolean gate/check to equal "
                "the untouched baseline signature."
            ),
            "integrity": (
                "Require nonempty mono PCM16/24 kHz, equal sample count, finite PCM, "
                "clipping below 0.001, preserved parent identity/runtime/alignment gates, "
                "neutral identity outside the splice, lens identity at full weight, and "
                "a candidate WAV hash disjoint from all bound parent WAVs."
            ),
        },
        "runtime_localization_benchmark": {
            "clock": "time.perf_counter_ns",
            "warmup_iterations_per_fixture": BENCHMARK_WARMUP_ITERATIONS,
            "measured_iterations_per_fixture": BENCHMARK_MEASURED_ITERATIONS,
            "cheap_thresholds_ms": {
                "median_max": MAX_LOCALIZATION_MEDIAN_MS,
                "p95_max": MAX_LOCALIZATION_P95_MS,
            },
            "fail_closed_cases": ["identical PCM", "sample-count mismatch"],
        },
        "outcomes": {
            "candidate_success": (
                "Both known fixtures pass localization, preserved acoustic/integrity "
                "gates, and the frozen artifact gate; eligible for one unseen "
                "confirmation, with no production integration."
            ),
            "candidate_failure_runtime_gate_success": (
                "Report the cheap fail-closed coverage-gated path as product candidate."
            ),
            "candidate_and_runtime_gate_failure": (
                "Close Kokoro product remediation for Build Week and retain research evidence."
            ),
        },
        "fixtures": fixtures,
        "implementation": {
            "source_relative_path": str(source_path.relative_to(Paths().root)),
            "source_sha256": sha256_file(source_path),
            "measurement_script_relative_path": str(
                parent_protocol["implementation"]["measurement"][
                    "script_relative_path"
                ]
            ),
            "measurement_script_sha256": parent_protocol["implementation"][
                "measurement"
            ]["script_sha256"],
        },
    }
    payload["protocol_sha256"] = sha256_json(payload)
    return payload


def prepare() -> dict[str, Any]:
    destination = run_dir() / PROTOCOL_FILE
    if (run_dir() / ANALYSIS_FILE).exists() or (run_dir() / "audio").exists():
        raise RuntimeError("candidate artifacts already exist; protocol cannot be frozen")
    protocol = protocol_record()
    _write_once_json(destination, protocol)
    return protocol


def _checked_protocol() -> dict[str, Any]:
    path = run_dir() / PROTOCOL_FILE
    protocol = _load_json(path)
    expected = protocol_record()
    if stable_json(protocol) != stable_json(expected):
        raise RuntimeError("frozen output-domain splice protocol drifted")
    if protocol.get("status") != "frozen_before_candidate_artifacts":
        raise RuntimeError("output-domain splice protocol has invalid status")
    return protocol


def _require_frozen_commit(protocol: dict[str, Any]) -> str:
    root = Paths().root
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", str((run_dir() / PROTOCOL_FILE).relative_to(root))],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    if tracked.returncode != 0:
        raise RuntimeError("protocol must be committed before candidate generation")
    relevant = [
        protocol["implementation"]["source_relative_path"],
        str((run_dir() / PROTOCOL_FILE).relative_to(root)),
    ]
    status = subprocess.run(
        ["git", "status", "--porcelain", "--", *relevant],
        cwd=root,
        text=True,
        capture_output=True,
        check=True,
    )
    if status.stdout.strip():
        raise RuntimeError("protocol and implementation must match committed HEAD")
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()


def _fixture_analysis(
    fixture_spec: dict[str, Any],
    record: dict[str, Any],
    parent_fixture: dict[str, Any],
    anchors: dict[str, Any],
    parent_hashes: set[str],
) -> dict[str, Any]:
    neutral_path = parent_dir() / fixture_spec["neutral"]["relative_path"]
    lens_path = parent_dir() / fixture_spec["lens"]["relative_path"]
    if sha256_file(neutral_path) != fixture_spec["neutral"]["wav_sha256"]:
        raise RuntimeError("bound neutral WAV hash drifted")
    if sha256_file(lens_path) != fixture_spec["lens"]["wav_sha256"]:
        raise RuntimeError("bound lens WAV hash drifted")
    neutral, neutral_rate = _read_pcm16(neutral_path)
    lens, lens_rate = _read_pcm16(lens_path)
    if neutral_rate != lens_rate:
        raise RuntimeError("bound parent sample rates differ")
    candidate, weights = output_domain_splice(
        neutral, lens, fixture_spec["splice_windows"]
    )
    candidate_path = run_dir() / "audio" / f"{fixture_spec['fixture_id']}__candidate.wav"
    _write_pcm16_once(candidate_path, candidate)
    localization = localization_report(
        neutral,
        candidate,
        fixture_spec["target_word_intervals"],
        sample_rate_hz=neutral_rate,
    )
    acoustic = _acoustic_report(
        neutral_path, candidate_path, record, parent_fixture, anchors
    )
    artifact = boundary_artifact_report(
        neutral, lens, candidate, fixture_spec["splice_windows"]
    )
    integrity = _integrity_report(
        neutral,
        lens,
        candidate,
        weights,
        record,
        candidate_path,
        parent_hashes,
    )
    benchmark = _benchmark_localization(
        neutral, candidate, fixture_spec["target_word_intervals"]
    )
    candidate_pass = bool(
        localization.get("pass")
        and acoustic["pass"]
        and integrity["pass"]
        and artifact["pass"]
    )
    return {
        "fixture_id": fixture_spec["fixture_id"],
        "untouched_baseline": {
            "inside_difference_energy_fraction": fixture_spec[
                "baseline_localization"
            ]["inside_difference_energy_fraction"],
            "localization_pass": fixture_spec["baseline_localization"]["pass"],
            "neutral_wav_sha256": fixture_spec["neutral"]["wav_sha256"],
            "lens_wav_sha256": fixture_spec["lens"]["wav_sha256"],
        },
        "candidate": {
            "relative_path": str(candidate_path.relative_to(run_dir())),
            "wav_sha256": sha256_file(candidate_path),
            "splice_windows": fixture_spec["splice_windows"],
            "taper_samples_each_edge": TAPER_SAMPLES,
        },
        "localization": localization,
        "acoustic": acoustic,
        "integrity": integrity,
        "boundary_artifact": artifact,
        "localization_runtime_benchmark": benchmark,
        "candidate_pass": candidate_pass,
    }


def outcome_for(
    fixture_passes: Sequence[bool], runtime_gate_pass: bool
) -> tuple[str, str]:
    if len(fixture_passes) == 2 and all(fixture_passes):
        return (
            "candidate_succeeds_both_known_fixtures",
            "eligible_for_one_unseen_confirmation_no_product_integration",
        )
    if runtime_gate_pass:
        return (
            "candidate_fails_runtime_gate_cheap_and_fail_closed",
            "coverage_gated_path_is_product_candidate",
        )
    return (
        "candidate_and_runtime_gate_fail",
        "close_kokoro_product_remediation_for_build_week",
    )


def _acoustic_boolean_deltas(
    baseline: dict[str, Any], candidate: dict[str, Any]
) -> tuple[list[str], list[str]]:
    degradations: list[str] = []
    improvements: list[str] = []
    for window in sorted(baseline):
        baseline_occurrences = baseline[window]["occurrences"]
        candidate_occurrences = candidate[window]["occurrences"]
        for occurrence_index, (left, right) in enumerate(
            zip(baseline_occurrences, candidate_occurrences, strict=True)
        ):
            for ceiling in sorted(left["families"]):
                baseline_family = left["families"][ceiling]
                candidate_family = right["families"][ceiling]
                fields = {
                    "pass": (
                        baseline_family["pass"],
                        candidate_family["pass"],
                    ),
                    **{
                        f"checks.{name}": (
                            baseline_family["checks"][name],
                            candidate_family["checks"][name],
                        )
                        for name in sorted(baseline_family["checks"])
                    },
                }
                for field, (before, after) in fields.items():
                    path = (
                        f"window={window}/occurrence={occurrence_index}/"
                        f"ceiling={ceiling}/{field}"
                    )
                    if before is True and after is False:
                        degradations.append(path)
                    elif before is False and after is True:
                        improvements.append(path)
    return degradations, improvements


def adjudicate() -> dict[str, Any]:
    destination = run_dir() / ADJUDICATION_FILE
    raw = _load_json(run_dir() / ANALYSIS_FILE)
    if raw.get("classification") != (
        "candidate_fails_runtime_gate_cheap_and_fail_closed"
    ):
        raise RuntimeError("raw splice classification is not the known bookkeeping case")
    parent_protocol = _load_json(parent_dir() / PROTOCOL_FILE)
    descriptive_rule = parent_protocol["automatic_gate"]["descriptive_rule"]
    if "never change the primary-50 outcome" not in descriptive_rule:
        raise RuntimeError("parent acoustic gate semantics drifted")
    fixtures: list[dict[str, Any]] = []
    for row in raw["fixtures"]:
        degradations, improvements = _acoustic_boolean_deltas(
            row["acoustic"]["baseline_gate_signature"],
            row["acoustic"]["candidate_gate_signature"],
        )
        candidate_family_passes = all(
            family["pass"]
            for window in row["acoustic"]["candidate_gate_signature"].values()
            for occurrence in window["occurrences"]
            for family in occurrence["families"].values()
        )
        checks = {
            "localization_pass": bool(row["localization"]["pass"]),
            "primary_acoustic_gate_pass": bool(
                row["acoustic"]["primary_gate_pass"]
            ),
            "all_candidate_analysis_family_checks_pass": bool(
                candidate_family_passes
            ),
            "no_acoustic_boolean_degradation": not degradations,
            "integrity_pass": bool(row["integrity"]["pass"]),
            "boundary_artifact_pass": bool(row["boundary_artifact"]["pass"]),
            "runtime_localization_gate_cheap_and_fail_closed": bool(
                row["localization_runtime_benchmark"]["pass"]
            ),
        }
        fixtures.append(
            {
                "fixture_id": row["fixture_id"],
                "candidate_wav_sha256": row["candidate"]["wav_sha256"],
                "checks": checks,
                "acoustic_boolean_degradations": degradations,
                "acoustic_boolean_improvements": improvements,
                "adjudicated_candidate_pass": all(checks.values()),
            }
        )
    success = bool(len(fixtures) == 2 and all(row["adjudicated_candidate_pass"] for row in fixtures))
    if not success:
        raise RuntimeError("gate-semantic correction does not support candidate success")
    payload: dict[str, Any] = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "status": "gate_semantics_adjudicated",
        "raw_analysis_sha256": sha256_file(run_dir() / ANALYSIS_FILE),
        "raw_analysis_internal_sha256": raw["analysis_sha256"],
        "protocol_sha256": raw["protocol_sha256"],
        "raw_classification_preserved": raw["classification"],
        "defect": (
            "The spike classifier promoted exact equality with the parent's descriptive "
            "40/60 signatures into a conjunctive gate, contradicting the bound parent "
            "rule that those settings never change the primary-50 outcome."
        ),
        "correction_boundary": (
            "No candidate audio, window, taper, threshold, acoustic measurement, integrity "
            "measurement, artifact measurement, or runtime benchmark changed; only the "
            "outcome mapping is corrected to the inherited gate semantics."
        ),
        "parent_descriptive_rule": descriptive_rule,
        "fixtures": fixtures,
        "classification": "candidate_succeeds_both_known_fixtures",
        "recommendation": (
            "eligible_for_one_unseen_confirmation_no_product_integration"
        ),
        "eligible_for_one_unseen_confirmation": True,
        "production_integration_authorized": False,
        "api_calls": 0,
        "model_decodes": 0,
    }
    payload["adjudication_sha256"] = sha256_json(payload)
    _write_once_json(destination, payload)
    return payload


def run() -> dict[str, Any]:
    analysis_path = run_dir() / ANALYSIS_FILE
    if analysis_path.exists():
        return _load_json(analysis_path)
    protocol = _checked_protocol()
    commit = _require_frozen_commit(protocol)
    parent_protocol, records, parent_analysis = _parent_payloads()
    records_by_id = {row["fixture_id"]: row for row in records["fixtures"]}
    analysis_by_id = {
        row["fixture_id"]: row for row in parent_analysis["fixtures"]
    }
    all_parent_hashes = {
        row[role]["wav_sha256"]
        for row in (record["audio"] for record in records["fixtures"])
        for role in ("neutral", "identity", "lens")
    }
    fixtures = [
        _fixture_analysis(
            fixture,
            records_by_id[fixture["fixture_id"]],
            analysis_by_id[fixture["fixture_id"]],
            parent_protocol["parents"]["diagnostic"]["local_anchor_geometry"],
            all_parent_hashes,
        )
        for fixture in protocol["fixtures"]
    ]
    runtime_gate_pass = all(
        row["localization_runtime_benchmark"]["pass"] for row in fixtures
    )
    classification, recommendation = outcome_for(
        [row["candidate_pass"] for row in fixtures], runtime_gate_pass
    )
    payload = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "protocol_sha256": protocol["protocol_sha256"],
        "executed_commit": commit,
        "status": "analysis_complete",
        "classification": classification,
        "recommendation": recommendation,
        "candidate_count": 1,
        "candidate_wav_count": len(fixtures),
        "api_calls": 0,
        "model_decodes": 0,
        "fixtures": fixtures,
        "runtime_localization_gate": {
            "cheap_and_fail_closed": runtime_gate_pass,
            "pass": runtime_gate_pass,
        },
        "production_integration_authorized": False,
        "eligible_for_one_unseen_confirmation": classification
        == "candidate_succeeds_both_known_fixtures",
    }
    payload["analysis_sha256"] = sha256_json(payload)
    _write_once_json(analysis_path, payload)
    return payload
