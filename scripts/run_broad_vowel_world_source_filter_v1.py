#!/usr/bin/env python3
from __future__ import annotations

from collections import Counter, defaultdict
import argparse
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Sequence

import numpy as np
import pyworld

from earshift_bakeoff.config import Paths, sha256_json, stable_json
from earshift_bakeoff.kokoro_output_domain_splice import boundary_artifact_report
from earshift_bakeoff.kokoro_synthesis import SAMPLE_RATE_HZ
from earshift_bakeoff.same_take_corrective import low_band_log_spectral_distance_db
from earshift_bakeoff.spectral_envelope_warp import SpectralEnvelopeWarpResult
from earshift_bakeoff.util import atomic_write_json, sha256_file
from earshift_bakeoff.world_formant_editor import (
    WorldFormantSpec,
    world_formant_edit,
)

import run_bilingual_vowel_adaptive_strength_screen_v1 as adaptive
import run_broad_oral_monophthong_contextual_warp_v1 as broad_v1
import run_english_central_vowel_spectral_correction_v1 as spectral_v1


VERSION = "broad-vowel-world-source-filter-v1"
RUN_ID = "20260718-broad-vowel-world-source-filter-v1"
PROTOCOL_PATH = Paths().root / "rules" / f"{VERSION}.json"
RUN_DIR = Paths().artifacts / "product-matrix" / RUN_ID
RESULT_PATH = RUN_DIR / "results.json"
V8_RESULT_PATH = broad_v1.V8_RESULT_PATH
V8_DIR = V8_RESULT_PATH.parent
PRIOR_WORLD_RESULT = (
    Paths().artifacts
    / "same-take"
    / "20260715-same-take-word-v3"
    / "world-pass.json"
)
MAXIMUM_IDENTITY_RMS_DELTA_DB = 1.0
MAXIMUM_IDENTITY_HIGH_BAND_DELTA_DB = 3.0
MAXIMUM_IDENTITY_LOW_BAND_LSD_DB = 6.0


def _semantic_hash(payload: Any) -> str:
    return hashlib.sha256(stable_json(payload).encode("utf-8")).hexdigest()


def _pyworld_extension_path():
    paths = sorted(Path(pyworld.__file__).parent.glob("pyworld*.so"))
    if len(paths) != 1:
        raise RuntimeError("expected exactly one pinned PyWORLD extension")
    return paths[0]


def protocol_record() -> dict[str, Any]:
    catalog = broad_v1._eligible_catalog()
    payload = {
        "schema_version": 1,
        "version": VERSION,
        "run_id": RUN_ID,
        "status": "frozen_before_world_processing",
        "purpose": (
            "Test matched WORLD identity/lens source-filter editing across the full "
            "calibrated 45-cell oral-monophthong denominator."
        ),
        "parents": {
            "v8_result_sha256": sha256_file(V8_RESULT_PATH),
            "calibration_sha256": sha256_file(broad_v1.CALIBRATION_PATH),
            "prior_world_result_sha256": sha256_file(PRIOR_WORLD_RESULT),
        },
        "scope": {
            "cell_ids": [row["cell_id"] for row in catalog],
            "cell_count": len(catalog),
            "logical_slot_count": 135,
            "target_occurrence_count": 180,
            "conditions": ["identity", "lens"],
            "maximum_world_analyses": 135,
            "maximum_world_syntheses": 270,
        },
        "editor": {
            "version": "world-formant-editor-v1",
            "shared_state": "one DIO/StoneMask/CheapTrick/D4C analysis per pair",
            "frame_period_ms": 5.0,
            "fft_size": 1024,
            "formant_warp": (
                "piecewise-linear inverse frequency map of the WORLD spectral "
                "envelope from contextual F1/F2 toward the calibrated prototype; "
                "F3, F0, aperiodicity, duration, and excitation are unchanged"
            ),
            "shift_taper_ms": 10.0,
            "splice_context_ms": 20.0,
            "splice_taper_ms": 10.0,
            "gain": "one identity-derived RMS gain shared by identity and lens",
            "selection": "none; one identity and one full-strength lens",
        },
        "gates": {
            "identity": {
                "outside_windows_bit_exact": True,
                "maximum_rms_delta_db": MAXIMUM_IDENTITY_RMS_DELTA_DB,
                "maximum_high_band_delta_db": MAXIMUM_IDENTITY_HIGH_BAND_DELTA_DB,
                "maximum_low_band_lsd_db": MAXIMUM_IDENTITY_LOW_BAND_LSD_DB,
                "boundary_artifact_gate": "existing output-domain-splice gate",
                "source_category": "existing calibrated neutral endpoint gate",
            },
            "pair_engineering": "complete spectral-v1 engineering gate",
            "acoustic": "existing calibrated endpoint and direction gates",
            "aggregation": "all occurrences across all three frozen contexts",
        },
        "stopping_rule": (
            "One matched WORLD identity/lens pair per slot; no strength selection, "
            "alternate WORLD settings, replacement, or rerun."
        ),
        "scope_controls": {
            "kokoro_renders": 0,
            "api_calls": 0,
            "paid_calls": 0,
            "production_enabled": False,
            "deployment": False,
        },
        "instrument": {
            "pyworld_version": pyworld.__version__,
            "pyworld_extension_sha256": sha256_file(_pyworld_extension_path()),
            "license": "MIT wrapper; bundled WORLD engine modified BSD",
        },
        "source_bindings": {
            path: sha256_file(Paths().root / path)
            for path in (
                "scripts/run_broad_vowel_world_source_filter_v1.py",
                "src/earshift_bakeoff/world_formant_editor.py",
                "scripts/run_broad_oral_monophthong_contextual_warp_v1.py",
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
            raise RuntimeError("broad WORLD protocol drifted")
        return existing
    if RUN_DIR.exists():
        raise RuntimeError("broad WORLD run exists before protocol")
    atomic_write_json(PROTOCOL_PATH, protocol)
    return protocol


def _db_ratio(candidate: np.ndarray, reference: np.ndarray) -> float:
    candidate_rms = math.sqrt(float(np.mean(np.square(candidate.astype(np.float64)))))
    reference_rms = math.sqrt(float(np.mean(np.square(reference.astype(np.float64)))))
    return 20.0 * math.log10(max(candidate_rms, 1e-12) / max(reference_rms, 1e-12))


def _identity_engineering(
    original: np.ndarray,
    identity: np.ndarray,
    windows: Sequence[dict[str, int]],
    intervals: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    rms = []
    high_band = []
    lsd = []
    for window in windows:
        start = int(window["start_sample"])
        end = int(window["end_sample_exclusive"])
        rms.append(_db_ratio(identity[start:end], original[start:end]))
        high_band.append(
            spectral_v1._band_energy_db(identity[start:end], spectral_v1.HIGH_BAND_START_HZ)
            - spectral_v1._band_energy_db(original[start:end], spectral_v1.HIGH_BAND_START_HZ)
        )
    for interval in intervals:
        start = int(interval["start_sample"])
        end = int(interval["end_sample_exclusive"])
        lsd.append(
            low_band_log_spectral_distance_db(
                original[start:end], identity[start:end], SAMPLE_RATE_HZ
            )
        )
    outside = np.ones(original.size, dtype=bool)
    for window in windows:
        outside[int(window["start_sample"]):int(window["end_sample_exclusive"])] = False
    boundary = boundary_artifact_report(original, identity, identity, list(windows))
    passed = bool(
        np.array_equal(original[outside], identity[outside])
        and max(abs(value) for value in rms) <= MAXIMUM_IDENTITY_RMS_DELTA_DB
        and max(abs(value) for value in high_band) <= MAXIMUM_IDENTITY_HIGH_BAND_DELTA_DB
        and max(lsd) <= MAXIMUM_IDENTITY_LOW_BAND_LSD_DB
        and boundary.get("pass")
    )
    return {
        "pass": passed,
        "outside_windows_bit_exact": bool(np.array_equal(original[outside], identity[outside])),
        "rms_delta_db_by_window": rms,
        "high_band_delta_db_by_window": high_band,
        "low_band_lsd_db_by_occurrence": lsd,
        "boundary_metrics_pass": bool(boundary.get("pass")),
    }


def _pair_engineering(
    identity: np.ndarray,
    lens: np.ndarray,
    windows: Sequence[dict[str, int]],
    intervals: Sequence[dict[str, Any]],
    universal_metrics: dict[str, Any],
) -> dict[str, Any]:
    rms = []
    for window in windows:
        start = int(window["start_sample"])
        end = int(window["end_sample_exclusive"])
        rms.append(_db_ratio(lens[start:end], identity[start:end]))
    result = SpectralEnvelopeWarpResult(
        pcm=lens,
        weights=np.zeros(lens.size, dtype=np.float64),
        edit_windows=tuple(windows),
        metrics={
            "finite": universal_metrics["finite"],
            "outside_windows_bit_exact": bool(
                universal_metrics["outside_windows_bit_exact"]
            ),
            "clipped_sample_count": universal_metrics["clipped_sample_count"],
            "rms_db_change_by_window": rms,
        },
    )
    return spectral_v1._engineering(identity, result, intervals)


def _specs(
    baseline: dict[str, Any], neutral: Sequence[dict[str, Any]], cell: dict[str, Any]
) -> tuple[WorldFormantSpec, ...]:
    targets = tuple(
        broad_v1._target_measurement(row, cell["prototype_displacement_bark"])
        for row in neutral
    )
    result = []
    for occurrence, source, target in zip(
        baseline["occurrence_outcomes"], neutral, targets, strict=True
    ):
        source_bin = source["bins"][0]
        target_bin = target["bins"][0]
        interval = occurrence["measurement_interval"]
        result.append(
            WorldFormantSpec(
                start_sample=int(interval["start_sample"]),
                end_sample_exclusive=int(interval["end_sample_exclusive"]),
                source_f1_hz=float(source_bin["f1_hz"]),
                source_f2_hz=float(source_bin["f2_hz"]),
                source_f3_hz=float(source_bin["f3_hz"]),
                target_f1_hz=float(target_bin["f1_hz"]),
                target_f2_hz=float(target_bin["f2_hz"]),
            )
        )
    return tuple(result)


def _slot(baseline: dict[str, Any], cell: dict[str, Any]) -> dict[str, Any]:
    contextual = broad_v1._contextual_baseline(baseline, cell)
    analysis = contextual["analysis_by_formant_ceiling"][0]
    ceiling = int(analysis["maximum_formant_hz"])
    frozen = analysis["measurements"]
    intervals = tuple(row["measurement_interval"] for row in baseline["occurrence_outcomes"])
    neutral_path = V8_DIR / baseline["audio"]["neutral"]["relative_path"]
    original = adaptive._read_wav(neutral_path)
    edited = world_formant_edit(
        original,
        _specs(baseline, frozen["neutral"], cell),
        sample_rate_hz=SAMPLE_RATE_HZ,
    )
    stem = adaptive._safe_name(baseline["logical_slot_id"])
    identity_path = RUN_DIR / "audio" / f"{stem}__world-identity.wav"
    lens_path = RUN_DIR / "audio" / f"{stem}__world-lens.wav"
    identity_audio = adaptive._write_wav(identity_path, edited.identity_pcm)
    lens_audio = adaptive._write_wav(lens_path, edited.lens_pcm)
    identity_measurements = adaptive._measure(
        path=identity_path,
        stem=f"{stem}__world-identity",
        intervals=intervals,
        ceiling=ceiling,
        mode="monophthong_core",
    )
    lens_measurements = adaptive._measure(
        path=lens_path,
        stem=f"{stem}__world-lens",
        intervals=intervals,
        ceiling=ceiling,
        mode="monophthong_core",
    )
    identity_engineering = _identity_engineering(
        original, edited.identity_pcm, edited.edit_windows, intervals
    )
    pair_engineering = _pair_engineering(
        edited.identity_pcm,
        edited.lens_pcm,
        edited.edit_windows,
        intervals,
        edited.metrics,
    )
    classifications = tuple(
        adaptive._analysis_classification(
            source=source,
            target=target,
            neutral=identity,
            lens=lens,
            rhotic=False,
        )
        for source, target, identity, lens in zip(
            frozen["source_anchor"],
            frozen["target_anchor"],
            identity_measurements,
            lens_measurements,
            strict=True,
        )
    )
    classification = (
        adaptive._aggregate(classifications)
        if identity_engineering["pass"] and pair_engineering["pass"]
        else "fail"
    )
    return {
        "logical_slot_id": baseline["logical_slot_id"],
        "cell_id": baseline["cell_id"],
        "profile_id": baseline["profile_id"],
        "voice_id": baseline["voice_id"],
        "rule_id": baseline["rule_id"],
        "context": baseline["context"],
        "status": "measured",
        "classification": classification,
        "identity_audio": identity_audio,
        "lens_audio": lens_audio,
        "editor_metrics": edited.metrics,
        "identity_engineering": identity_engineering,
        "pair_engineering": pair_engineering,
        "identity_measurements": identity_measurements,
        "lens_measurements": lens_measurements,
        "occurrence_classifications": classifications,
        "api_calls_made": 0,
        "production_enabled": False,
    }


def _excluded(baseline: dict[str, Any], exc: Exception) -> dict[str, Any]:
    return {
        "logical_slot_id": baseline["logical_slot_id"],
        "cell_id": baseline["cell_id"],
        "voice_id": baseline["voice_id"],
        "rule_id": baseline["rule_id"],
        "context": baseline["context"],
        "status": "processing_exclusion",
        "classification": "fail",
        "error": f"{type(exc).__name__}: {exc}",
        "api_calls_made": 0,
        "production_enabled": False,
    }


def _summaries(outcomes: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in outcomes:
        groups[row["cell_id"]].append(row)
    summaries = []
    for cell_id, slots in sorted(groups.items()):
        classification = adaptive._aggregate(slots) if len(slots) == 3 else "fail"
        summaries.append(
            {
                "cell_id": cell_id,
                "voice_id": slots[0]["voice_id"],
                "rule_id": slots[0]["rule_id"],
                "classification": classification,
                "production_enabled": False,
            }
        )
    return summaries


def run() -> dict[str, Any]:
    if RESULT_PATH.exists():
        return json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if stable_json(protocol) != stable_json(protocol_record()):
        raise RuntimeError("broad WORLD protocol or sources drifted")
    catalog = {row["cell_id"]: row for row in broad_v1._eligible_catalog()}
    v8 = json.loads(V8_RESULT_PATH.read_text(encoding="utf-8"))
    baselines = [row for row in v8["outcomes"] if row["cell_id"] in catalog]
    if len(baselines) != 135:
        raise RuntimeError("broad WORLD denominator drifted")
    RUN_DIR.mkdir(parents=True, exist_ok=False)
    old_adaptive = adaptive.RUN_DIR
    adaptive.RUN_DIR = RUN_DIR
    outcomes = []
    started = time.perf_counter()
    try:
        for baseline in baselines:
            try:
                outcome = _slot(baseline, catalog[baseline["cell_id"]])
            except (RuntimeError, ValueError) as exc:
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
        "classification": "broad_world_known_fixture_characterization",
        "cell_classification_counts": dict(
            sorted(Counter(row["classification"] for row in cells).items())
        ),
        "cell_count": len(cells),
        "logical_slot_count": len(outcomes),
        "status_counts": dict(sorted(Counter(row["status"] for row in outcomes).items())),
        "world_analysis_count": sum(row["status"] == "measured" for row in outcomes),
        "world_synthesis_count": 2 * sum(row["status"] == "measured" for row in outcomes),
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
