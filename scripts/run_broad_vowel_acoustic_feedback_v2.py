#!/usr/bin/env python3
from __future__ import annotations

from collections import Counter
import argparse
import hashlib
import json
import time
from typing import Any

from earshift_bakeoff.config import Paths, sha256_json, stable_json
from earshift_bakeoff.util import atomic_write_json, sha256_file

import run_bilingual_vowel_adaptive_strength_screen_v1 as adaptive
import run_broad_oral_monophthong_contextual_warp_v1 as broad_v1
import run_broad_vowel_acoustic_feedback_v1 as v1


VERSION = "broad-vowel-acoustic-feedback-v2"
RUN_ID = "20260718-broad-vowel-acoustic-feedback-v2"
PROTOCOL_PATH = Paths().root / "rules" / f"{VERSION}.json"
RUN_DIR = Paths().artifacts / "product-matrix" / RUN_ID
RESULT_PATH = RUN_DIR / "results.json"
V1_FAILURE_PATH = v1.RUN_DIR / "runtime-failure.json"


def _semantic_hash(payload: Any) -> str:
    return hashlib.sha256(stable_json(payload).encode("utf-8")).hexdigest()


def protocol_record() -> dict[str, Any]:
    v1_protocol = json.loads(v1.PROTOCOL_PATH.read_text(encoding="utf-8"))
    failure = json.loads(V1_FAILURE_PATH.read_text(encoding="utf-8"))
    payload = {
        "schema_version": 1,
        "version": VERSION,
        "run_id": RUN_ID,
        "status": "frozen_before_control_flow_correction_run",
        "purpose": "Correct only the v1 per-slot fail-closed exception boundary.",
        "parents": {
            "v1_protocol_sha256": sha256_file(v1.PROTOCOL_PATH),
            "v1_protocol_record_sha256": v1_protocol["protocol_sha256"],
            "v1_runtime_failure_sha256": sha256_file(V1_FAILURE_PATH),
            "v1_partial_artifact_tree_sha256": failure["artifact_tree_sha256"],
        },
        "invariants": {
            "cell_ids": v1_protocol["scope"]["cell_ids"],
            "cell_count": v1_protocol["scope"]["cell_count"],
            "logical_slot_count": v1_protocol["scope"]["logical_slot_count"],
            "maximum_feedback_edits": v1_protocol["scope"][
                "maximum_feedback_edits"
            ],
            "feedback_gain": v1_protocol["controller"]["feedback_gain"],
            "maximum_iterations": v1_protocol["controller"]["maximum_iterations"],
            "mechanism_change": False,
            "target_change": False,
            "threshold_change": False,
            "aggregation_change": False,
        },
        "correction": (
            "Catch RuntimeError or ValueError around complete slot construction, record "
            "that slot as processing_exclusion/fail, and continue the frozen denominator."
        ),
        "stopping_rule": "Run the unchanged denominator once; no replacement or rerun.",
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
                "scripts/run_broad_vowel_acoustic_feedback_v2.py",
                "scripts/run_broad_vowel_acoustic_feedback_v1.py",
                "scripts/run_broad_oral_monophthong_contextual_warp_v1.py",
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
            raise RuntimeError("acoustic-feedback v2 protocol drifted")
        return existing
    if RUN_DIR.exists():
        raise RuntimeError("acoustic-feedback v2 run exists before protocol")
    atomic_write_json(PROTOCOL_PATH, protocol)
    return protocol


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
        "iteration_count": 0,
        "iterations": [],
        "error": f"{type(exc).__name__}: {exc}",
        "api_calls_made": 0,
        "production_enabled": False,
    }


def run() -> dict[str, Any]:
    if RESULT_PATH.exists():
        return json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if stable_json(protocol) != stable_json(protocol_record()):
        raise RuntimeError("acoustic-feedback v2 protocol or sources drifted")
    catalog = {row["cell_id"]: row for row in broad_v1._eligible_catalog()}
    v8 = json.loads(v1.V8_RESULT_PATH.read_text(encoding="utf-8"))
    baselines = [row for row in v8["outcomes"] if row["cell_id"] in catalog]
    RUN_DIR.mkdir(parents=True, exist_ok=False)
    old_adaptive = adaptive.RUN_DIR
    old_v1 = v1.RUN_DIR
    adaptive.RUN_DIR = RUN_DIR
    v1.RUN_DIR = RUN_DIR
    outcomes = []
    started = time.perf_counter()
    try:
        for baseline in baselines:
            try:
                outcome = v1._slot(baseline, catalog[baseline["cell_id"]])
            except (RuntimeError, ValueError) as exc:
                outcome = _excluded(baseline, exc)
            outcomes.append(outcome)
    finally:
        adaptive.RUN_DIR = old_adaptive
        v1.RUN_DIR = old_v1
    cells = v1._summaries(outcomes)
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
