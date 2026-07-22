#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from typing import Any

from earshift_bakeoff.config import Paths
from earshift_bakeoff.kokoro_synthesis import pcm_sha256
from earshift_bakeoff.kokoro_typed_engine import KokoroTypedRuntime, TypedRender
from earshift_bakeoff.util import atomic_write_json, sha256_file


RUN_ID = "20260716-kokoro-typed-engine-v1"
TEST_INPUTS = {
    "fixture-a": "The black cat sat by that mat.",
    "fixture-b": "That black cat sat by that black cat.",
}


def _render(runtime: KokoroTypedRuntime, input_id: str) -> dict[str, Any]:
    result = runtime.render(TEST_INPUTS[input_id])
    if not isinstance(result, TypedRender):
        raise RuntimeError(f"{input_id} unexpectedly has no target")
    return {
        "input_id": input_id,
        "plan_sha256": result.plan.plan_sha256,
        "neutral_pcm_sha256": pcm_sha256(result.audio.neutral),
        "lens_pcm_sha256": pcm_sha256(result.audio.lens),
        "sample_count": result.integrity.sample_count,
        "integrity": asdict(result.integrity),
    }


def main() -> None:
    started = time.perf_counter()
    runtime = KokoroTypedRuntime.load()
    baselines = {
        input_id: _render(runtime, input_id) for input_id in sorted(TEST_INPUTS)
    }

    repeated_manifest = ("fixture-a",) * 4
    mixed_manifest = ("fixture-a", "fixture-b", "fixture-b", "fixture-a")
    with ThreadPoolExecutor(max_workers=4) as executor:
        repeated = list(
            executor.map(lambda value: _render(runtime, value), repeated_manifest)
        )
    with ThreadPoolExecutor(max_workers=4) as executor:
        mixed = list(
            executor.map(lambda value: _render(runtime, value), mixed_manifest)
        )

    def matches_baseline(record: dict[str, Any]) -> bool:
        return record == baselines[record["input_id"]]

    repeated_pass = all(matches_baseline(record) for record in repeated)
    mixed_pass = all(matches_baseline(record) for record in mixed)
    if not repeated_pass or not mixed_pass:
        raise RuntimeError(
            "concurrent controlled synthesis drifted from sequential baseline"
        )

    gate_receipt = (
        Paths().artifacts
        / "typed-engine"
        / "20260716-kokoro-gate-bridge-feasibility-v1"
        / "full-index-receipt.json"
    )
    payload = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "classification": "engineering_regression_only_not_research_evidence",
        "input_contract": (
            "two explicitly test-only inputs; no WAV retained and no replication "
            "fixture or candidate selected"
        ),
        "test_input_hashes": {
            key: hashlib.sha256(value.encode("utf-8")).hexdigest()
            for key, value in sorted(TEST_INPUTS.items())
        },
        "baselines": baselines,
        "repeated_identical": {
            "manifest": list(repeated_manifest),
            "records": repeated,
            "all_match_sequential_baseline": repeated_pass,
        },
        "mixed_different": {
            "manifest": list(mixed_manifest),
            "records": mixed,
            "all_match_correct_input_sequential_baseline": mixed_pass,
        },
        "lock_scope": (
            "process-global reentrant lock covers plan validation, source and carrier "
            "state generation, F0/noise generation, RNG reset, and every decode"
        ),
        "kokoro_gate_receipt_sha256": sha256_file(gate_receipt),
        "wall_seconds": time.perf_counter() - started,
        "api_calls_made": 0,
        "audio_files_retained": 0,
        "pass": repeated_pass and mixed_pass,
    }
    destination = Paths().artifacts / "typed-engine" / RUN_ID / "concurrency.json"
    atomic_write_json(destination, payload)
    print(destination)


if __name__ == "__main__":
    main()
