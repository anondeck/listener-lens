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
import run_broad_vowel_world_source_filter_v1 as world_v1
import run_broad_vowel_world_source_filter_v2 as world_v2


VERSION = "broad-vowel-world-differential-v1"
RUN_ID = "20260718-broad-vowel-world-differential-v1"
PROTOCOL_PATH = Paths().root / "rules" / f"{VERSION}.json"
RUN_DIR = Paths().artifacts / "product-matrix" / RUN_ID
RESULT_PATH = RUN_DIR / "results.json"


def _semantic_hash(payload: Any) -> str:
    return hashlib.sha256(stable_json(payload).encode("utf-8")).hexdigest()


def protocol_record() -> dict[str, Any]:
    manifest = world_v2._audio_manifest()
    payload = {
        "schema_version": 1,
        "version": VERSION,
        "run_id": RUN_ID,
        "status": "frozen_before_differential_processing",
        "purpose": (
            "Cancel shared WORLD identity coloration by applying only the frozen "
            "lens-minus-identity PCM differential to the untouched Kokoro carrier."
        ),
        "parents": {
            "world_v1_protocol_sha256": sha256_file(world_v1.PROTOCOL_PATH),
            "world_v2_result_sha256": sha256_file(world_v2.RESULT_PATH),
            "world_v1_audio_count": len(manifest),
            "world_v1_audio_manifest_sha256": sha256_json(manifest),
            "v8_result_sha256": sha256_file(world_v1.V8_RESULT_PATH),
            "calibration_sha256": sha256_file(broad_v1.CALIBRATION_PATH),
        },
        "intervention": {
            "formula": "derived_lens = original + (world_lens - world_identity)",
            "numeric_domain": "signed integer difference accumulated in int32",
            "encoding": "single round-free saturation to PCM16; any saturation fails",
            "neutral": "untouched original Kokoro PCM",
            "selection": "none; coefficient fixed at exactly 1.0",
            "world_analysis": "none; reuse frozen v1 outputs",
            "world_synthesis": "none; reuse frozen v1 outputs",
        },
        "scope": {
            "cell_count": 45,
            "logical_slot_count": 135,
            "available_input_pair_count": 133,
            "maximum_derived_lens_count": 133,
            "world_analyses": 0,
            "world_syntheses": 0,
        },
        "gates": {
            "neutral_identity": "bit-identical to untouched v8 Kokoro carrier",
            "integrity": "equal, finite, nonempty, unclipped PCM",
            "pair_engineering": "complete existing spectral-v1 engineering gate",
            "acoustic": "unchanged calibrated endpoint and direction gates",
            "aggregation": "all occurrences across all three frozen contexts",
        },
        "stopping_rule": (
            "Create at most one coefficient-1 differential lens for each available "
            "frozen pair. No scaling, strength search, replacement, or rerun."
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
                "scripts/run_broad_vowel_world_differential_v1.py",
                "scripts/run_broad_vowel_world_source_filter_v1.py",
                "scripts/run_broad_vowel_world_source_filter_v2.py",
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
            raise RuntimeError("WORLD differential protocol drifted")
        return existing
    if RUN_DIR.exists():
        raise RuntimeError("WORLD differential run exists before protocol")
    atomic_write_json(PROTOCOL_PATH, protocol)
    return protocol


def _differential_lens(
    original: np.ndarray, identity: np.ndarray, lens: np.ndarray
) -> tuple[np.ndarray, int]:
    source = np.asarray(original, dtype=np.int16).reshape(-1)
    neutral = np.asarray(identity, dtype=np.int16).reshape(-1)
    shifted = np.asarray(lens, dtype=np.int16).reshape(-1)
    if source.size == 0 or neutral.shape != source.shape or shifted.shape != source.shape:
        raise ValueError("WORLD differential inputs do not share nonempty PCM shape")
    combined = source.astype(np.int32) + (
        shifted.astype(np.int32) - neutral.astype(np.int32)
    )
    clipped = int(np.count_nonzero((combined < -32768) | (combined > 32767)))
    return np.clip(combined, -32768, 32767).astype(np.int16), clipped


def _slot(baseline: dict[str, Any], cell: dict[str, Any]) -> dict[str, Any]:
    contextual = broad_v1._contextual_baseline(baseline, cell)
    analysis = contextual["analysis_by_formant_ceiling"][0]
    frozen = analysis["measurements"]
    ceiling = int(analysis["maximum_formant_hz"])
    stem = adaptive._safe_name(baseline["logical_slot_id"])
    identity_path = world_v1.RUN_DIR / "audio" / f"{stem}__world-identity.wav"
    world_lens_path = world_v1.RUN_DIR / "audio" / f"{stem}__world-lens.wav"
    if not identity_path.exists() or not world_lens_path.exists():
        raise RuntimeError("frozen v1 WORLD pair is absent")
    original_path = world_v1.V8_DIR / baseline["audio"]["neutral"]["relative_path"]
    original = adaptive._read_wav(original_path)
    identity = adaptive._read_wav(identity_path)
    world_lens = adaptive._read_wav(world_lens_path)
    differential, clipped = _differential_lens(original, identity, world_lens)
    output_path = RUN_DIR / "audio" / f"{stem}__world-differential-lens.wav"
    output_audio = adaptive._write_wav(output_path, differential)
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
    outside = np.ones(original.size, dtype=bool)
    for window in windows:
        outside[window["start_sample"]:window["end_sample_exclusive"]] = False
    universal = {
        "finite": True,
        "outside_windows_bit_exact": bool(
            np.array_equal(original[outside], differential[outside])
        ),
        "clipped_sample_count": clipped,
    }
    engineering = world_v1._pair_engineering(
        original, differential, windows, intervals, universal
    )
    lens_measurements = adaptive._measure(
        path=output_path,
        stem=f"{stem}__world-differential-lens",
        intervals=intervals,
        ceiling=ceiling,
        mode="monophthong_core",
    )
    occurrences = tuple(
        adaptive._analysis_classification(
            source=source,
            target=target,
            neutral=neutral,
            lens=lens,
            rhotic=False,
        )
        for source, target, neutral, lens in zip(
            frozen["source_anchor"],
            frozen["target_anchor"],
            frozen["neutral"],
            lens_measurements,
            strict=True,
        )
    )
    classification = world_v2._classification(occurrences) if engineering["pass"] else "fail"
    return {
        "logical_slot_id": baseline["logical_slot_id"],
        "cell_id": baseline["cell_id"],
        "voice_id": baseline["voice_id"],
        "rule_id": baseline["rule_id"],
        "context": baseline["context"],
        "status": "measured",
        "classification": classification,
        "exact_category_pass": classification == "exact_category_pass",
        "directional_pass": classification
        in {"exact_category_pass", "directional_only_pass"},
        "differential_audio": output_audio,
        "input_identity_sha256": sha256_file(identity_path),
        "input_world_lens_sha256": sha256_file(world_lens_path),
        "preencoding_clipped_sample_count": clipped,
        "engineering": engineering,
        "lens_measurements": lens_measurements,
        "occurrence_classifications": occurrences,
        "world_analyses": 0,
        "world_syntheses": 0,
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
        "exact_category_pass": False,
        "directional_pass": False,
        "error": f"{type(exc).__name__}: {exc}",
        "world_analyses": 0,
        "world_syntheses": 0,
        "api_calls_made": 0,
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
            "classification": world_v2._classification(rows) if len(rows) == 3 else "fail",
            "production_enabled": False,
        }
        for cell_id, rows in sorted(groups.items())
    ]


def run() -> dict[str, Any]:
    if RESULT_PATH.exists():
        return json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if stable_json(protocol) != stable_json(protocol_record()):
        raise RuntimeError("WORLD differential protocol or inputs drifted")
    catalog = {row["cell_id"]: row for row in broad_v1._eligible_catalog()}
    v8 = json.loads(world_v1.V8_RESULT_PATH.read_text(encoding="utf-8"))
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
        "classification": "broad_world_differential_known_fixture_characterization",
        "cell_classification_counts": dict(
            sorted(Counter(row["classification"] for row in cells).items())
        ),
        "slot_classification_counts": dict(
            sorted(Counter(row["classification"] for row in outcomes).items())
        ),
        "status_counts": dict(sorted(Counter(row["status"] for row in outcomes).items())),
        "cell_count": len(cells),
        "logical_slot_count": len(outcomes),
        "derived_lens_count": sum(row["status"] == "measured" for row in outcomes),
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
