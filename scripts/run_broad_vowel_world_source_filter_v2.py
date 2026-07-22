#!/usr/bin/env python3
from __future__ import annotations

from collections import Counter, defaultdict
import argparse
import hashlib
import json
import time
from typing import Any, Sequence

import numpy as np

from earshift_bakeoff.config import Paths, sha256_json, stable_json
from earshift_bakeoff.util import atomic_write_json, sha256_file

import run_bilingual_vowel_adaptive_strength_screen_v1 as adaptive
import run_broad_oral_monophthong_contextual_warp_v1 as broad_v1
import run_broad_vowel_world_source_filter_v1 as v1


VERSION = "broad-vowel-world-source-filter-v2"
RUN_ID = "20260718-broad-vowel-world-source-filter-v2"
PROTOCOL_PATH = Paths().root / "rules" / f"{VERSION}.json"
RUN_DIR = Paths().artifacts / "product-matrix" / RUN_ID
RESULT_PATH = RUN_DIR / "results.json"
V1_DIR = v1.RUN_DIR
V1_FAILURE_PATH = V1_DIR / "runtime-inconclusive.json"


def _semantic_hash(payload: Any) -> str:
    return hashlib.sha256(stable_json(payload).encode("utf-8")).hexdigest()


def _audio_manifest() -> list[dict[str, str]]:
    return [
        {
            "relative_path": str(path.relative_to(V1_DIR)),
            "sha256": sha256_file(path),
        }
        for path in sorted((V1_DIR / "audio").glob("*.wav"))
    ]


def protocol_record() -> dict[str, Any]:
    manifest = _audio_manifest()
    if len(manifest) != 266:
        raise RuntimeError("frozen v1 WORLD audio inventory drifted")
    payload = {
        "schema_version": 1,
        "version": VERSION,
        "run_id": RUN_ID,
        "status": "frozen_before_measurement_only_recovery",
        "purpose": (
            "Recover the exact frozen v1 WORLD WAVs after its post-processing "
            "aggregation adapter failed."
        ),
        "parents": {
            "v1_protocol_sha256": sha256_file(v1.PROTOCOL_PATH),
            "v1_runtime_inconclusive_sha256": sha256_file(V1_FAILURE_PATH),
            "v1_audio_count": len(manifest),
            "v1_audio_manifest_sha256": sha256_json(manifest),
            "v8_result_sha256": sha256_file(v1.V8_RESULT_PATH),
            "calibration_sha256": sha256_file(broad_v1.CALIBRATION_PATH),
        },
        "correction": {
            "signal_processing": "none",
            "audio_generation": "none",
            "measurement": "reapply frozen v1 gates to hash-bound v1 WAVs",
            "aggregation": (
                "aggregate classification strings directly; exact only when every "
                "slot is exact, directional when every slot is exact/directional"
            ),
        },
        "scope": {
            "cell_count": 45,
            "logical_slot_count": 135,
            "available_matched_pair_count": 133,
            "frozen_wav_count": 266,
            "world_analyses": 0,
            "world_syntheses": 0,
        },
        "stopping_rule": (
            "Evaluate each available hash-bound pair once. Missing v1 pairs remain "
            "processing exclusions. Do not generate or replace audio."
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
                "scripts/run_broad_vowel_world_source_filter_v2.py",
                "scripts/run_broad_vowel_world_source_filter_v1.py",
                "scripts/run_broad_oral_monophthong_contextual_warp_v1.py",
            )
        },
    }
    return {**payload, "protocol_sha256": sha256_json(payload)}


def prepare() -> dict[str, Any]:
    protocol = protocol_record()
    if PROTOCOL_PATH.exists():
        existing = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        if stable_json(existing) != stable_json(protocol):
            raise RuntimeError("WORLD recovery protocol drifted")
        return existing
    if RUN_DIR.exists():
        raise RuntimeError("WORLD recovery directory exists before protocol")
    atomic_write_json(PROTOCOL_PATH, protocol)
    return protocol


def _classification(rows: Sequence[dict[str, Any]]) -> str:
    labels = [row["classification"] for row in rows]
    if labels and all(label == "exact_category_pass" for label in labels):
        return "exact_category_pass"
    if labels and all(
        label in {"exact_category_pass", "directional_only_pass"}
        for label in labels
    ):
        return "directional_only_pass"
    return "fail"


def _slot(baseline: dict[str, Any], cell: dict[str, Any]) -> dict[str, Any]:
    contextual = broad_v1._contextual_baseline(baseline, cell)
    analysis = contextual["analysis_by_formant_ceiling"][0]
    frozen = analysis["measurements"]
    ceiling = int(analysis["maximum_formant_hz"])
    stem = adaptive._safe_name(baseline["logical_slot_id"])
    identity_path = V1_DIR / "audio" / f"{stem}__world-identity.wav"
    lens_path = V1_DIR / "audio" / f"{stem}__world-lens.wav"
    if not identity_path.exists() or not lens_path.exists():
        raise RuntimeError("frozen v1 WORLD pair is absent")
    original_path = v1.V8_DIR / baseline["audio"]["neutral"]["relative_path"]
    original = adaptive._read_wav(original_path)
    identity = adaptive._read_wav(identity_path)
    lens = adaptive._read_wav(lens_path)
    intervals = tuple(
        row["measurement_interval"] for row in baseline["occurrence_outcomes"]
    )
    windows = tuple(
        {
            "start_sample": max(0, int(row["start_sample"]) - 480),
            "end_sample_exclusive": min(
                original.size, int(row["end_sample_exclusive"]) + 480
            ),
        }
        for row in intervals
    )
    identity_measurements = adaptive._measure(
        path=identity_path,
        stem=f"{stem}__recovery-world-identity",
        intervals=intervals,
        ceiling=ceiling,
        mode="monophthong_core",
    )
    lens_measurements = adaptive._measure(
        path=lens_path,
        stem=f"{stem}__recovery-world-lens",
        intervals=intervals,
        ceiling=ceiling,
        mode="monophthong_core",
    )
    outside = np.ones(original.size, dtype=bool)
    for window in windows:
        outside[window["start_sample"]:window["end_sample_exclusive"]] = False
    universal = {
        "finite": True,
        "outside_windows_bit_exact": bool(
            np.array_equal(original[outside], identity[outside])
            and np.array_equal(original[outside], lens[outside])
        ),
        "clipped_sample_count": int(
            np.count_nonzero((identity == -32768) | (identity == 32767))
            + np.count_nonzero((lens == -32768) | (lens == 32767))
        ),
    }
    identity_engineering = v1._identity_engineering(
        original, identity, windows, intervals
    )
    pair_engineering = v1._pair_engineering(
        identity, lens, windows, intervals, universal
    )
    occurrences = tuple(
        adaptive._analysis_classification(
            source=source,
            target=target,
            neutral=neutral,
            lens=shifted,
            rhotic=False,
        )
        for source, target, neutral, shifted in zip(
            frozen["source_anchor"],
            frozen["target_anchor"],
            identity_measurements,
            lens_measurements,
            strict=True,
        )
    )
    classification = (
        _classification(occurrences)
        if identity_engineering["pass"] and pair_engineering["pass"]
        else "fail"
    )
    return {
        "logical_slot_id": baseline["logical_slot_id"],
        "cell_id": baseline["cell_id"],
        "voice_id": baseline["voice_id"],
        "rule_id": baseline["rule_id"],
        "context": baseline["context"],
        "status": "measured_recovery",
        "classification": classification,
        "exact_category_pass": classification == "exact_category_pass",
        "directional_pass": classification
        in {"exact_category_pass", "directional_only_pass"},
        "identity_audio": {
            "relative_path": str(identity_path.relative_to(V1_DIR)),
            "sha256": sha256_file(identity_path),
        },
        "lens_audio": {
            "relative_path": str(lens_path.relative_to(V1_DIR)),
            "sha256": sha256_file(lens_path),
        },
        "identity_engineering": identity_engineering,
        "pair_engineering": pair_engineering,
        "identity_measurements": identity_measurements,
        "lens_measurements": lens_measurements,
        "occurrence_classifications": occurrences,
        "world_analyses": 0,
        "world_syntheses": 0,
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
        "exact_category_pass": False,
        "directional_pass": False,
        "error": f"{type(exc).__name__}: {exc}",
        "world_analyses": 0,
        "world_syntheses": 0,
        "production_enabled": False,
    }


def _summaries(outcomes: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in outcomes:
        groups[row["cell_id"]].append(row)
    return [
        {
            "cell_id": cell_id,
            "voice_id": rows[0]["voice_id"],
            "rule_id": rows[0]["rule_id"],
            "classification": _classification(rows) if len(rows) == 3 else "fail",
            "production_enabled": False,
        }
        for cell_id, rows in sorted(groups.items())
    ]


def run() -> dict[str, Any]:
    if RESULT_PATH.exists():
        return json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if stable_json(protocol) != stable_json(protocol_record()):
        raise RuntimeError("WORLD recovery protocol or inputs drifted")
    catalog = {row["cell_id"]: row for row in broad_v1._eligible_catalog()}
    v8 = json.loads(v1.V8_RESULT_PATH.read_text(encoding="utf-8"))
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
        "slot_classification_counts": dict(
            sorted(Counter(row["classification"] for row in outcomes).items())
        ),
        "status_counts": dict(sorted(Counter(row["status"] for row in outcomes).items())),
        "cell_count": len(cells),
        "logical_slot_count": len(outcomes),
        "world_analyses": 0,
        "world_syntheses": 0,
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
