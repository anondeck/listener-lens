#!/usr/bin/env python3
from __future__ import annotations

from collections import Counter, defaultdict
import argparse
import hashlib
import json
import time
from typing import Any, Sequence

from earshift_bakeoff.config import Paths, sha256_json, stable_json
from earshift_bakeoff.util import atomic_write_json, sha256_file

import run_bilingual_vowel_adaptive_strength_screen_v1 as adaptive
import run_broad_oral_monophthong_contextual_warp_v1 as v1
import run_english_central_vowel_spectral_correction_v1 as spectral_v1


VERSION = "broad-oral-monophthong-contextual-warp-v2"
RUN_ID = "20260718-broad-oral-monophthong-contextual-warp-v2"
PROTOCOL_PATH = Paths().root / "rules" / f"{VERSION}.json"
RUN_DIR = Paths().artifacts / "product-matrix" / RUN_ID
RESULT_PATH = RUN_DIR / "results.json"
V1_RUN_DIR = v1.RUN_DIR
V1_FAILURE_PATH = V1_RUN_DIR / "runtime-failure.json"


def _semantic_hash(payload: Any) -> str:
    return hashlib.sha256(stable_json(payload).encode("utf-8")).hexdigest()


def _tree_hash(root: Any) -> str:
    digest = hashlib.sha256()
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def protocol_record() -> dict[str, Any]:
    v1_protocol = json.loads(v1.PROTOCOL_PATH.read_text(encoding="utf-8"))
    failure = json.loads(V1_FAILURE_PATH.read_text(encoding="utf-8"))
    payload = {
        "schema_version": 1,
        "version": VERSION,
        "run_id": RUN_ID,
        "status": "frozen_before_deterministic_reexecution",
        "purpose": "Correct only v1's postprocessing aggregator defect.",
        "parents": {
            "v1_protocol_sha256": sha256_file(v1.PROTOCOL_PATH),
            "v1_protocol_record_sha256": v1_protocol["protocol_sha256"],
            "v1_runtime_failure_sha256": sha256_file(V1_FAILURE_PATH),
            "v1_artifact_tree_sha256": failure["artifact_tree_sha256"],
        },
        "invariants": {
            "cell_ids": v1_protocol["scope"]["cell_ids"],
            "cell_count": v1_protocol["scope"]["cell_count"],
            "logical_slot_count": v1_protocol["scope"]["logical_slot_count"],
            "candidate_count": v1_protocol["scope"]["candidate_count"],
            "strength_order": v1_protocol["scope"]["strength_order"],
            "mechanism_change": False,
            "threshold_change": False,
            "fixture_change": False,
            "audio_hash_equivalence_required": True,
        },
        "correction": (
            "Aggregate slot classification strings directly instead of passing slot "
            "records to a helper requiring absent exact_category_pass booleans."
        ),
        "stopping_rule": (
            "Reexecute the deterministic denominator once. Any audio filename or hash "
            "difference from v1 makes v2 runtime-inconclusive."
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
                "scripts/run_broad_oral_monophthong_contextual_warp_v2.py",
                "scripts/run_broad_oral_monophthong_contextual_warp_v1.py",
                "scripts/run_english_central_vowel_spectral_correction_v1.py",
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
            raise RuntimeError("broad contextual-warp v2 protocol drifted")
        return existing
    if RUN_DIR.exists():
        raise RuntimeError("broad contextual-warp v2 run exists before protocol")
    atomic_write_json(PROTOCOL_PATH, protocol)
    return protocol


def _summaries(outcomes: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for outcome in outcomes:
        groups[outcome["cell_id"]].append(outcome)
    rows = []
    for cell_id, slots in sorted(groups.items()):
        complete = bool(
            len(slots) == 3
            and all(slot["selection_complete"] for slot in slots)
            and sum(len(slot["occurrence_outcomes"]) for slot in slots) == 4
        )
        classifications = [slot["classification"] for slot in slots]
        classification = "fail"
        if complete and all(row == "exact_category_pass" for row in classifications):
            classification = "exact_category_pass"
        elif complete and all(
            row in {"exact_category_pass", "directional_only_pass"}
            for row in classifications
        ):
            classification = "directional_only_pass"
        rows.append(
            {
                "cell_id": cell_id,
                "voice_id": slots[0]["voice_id"],
                "rule_id": slots[0]["rule_id"],
                "classification": classification,
                "complete": complete,
                "production_enabled": False,
            }
        )
    return rows


def _audio_hashes(root: Any) -> dict[str, str]:
    audio = root / "audio"
    return {
        str(path.relative_to(audio)): sha256_file(path)
        for path in sorted(audio.glob("*.wav"))
    }


def run() -> dict[str, Any]:
    if RESULT_PATH.exists():
        return json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if stable_json(protocol) != stable_json(protocol_record()):
        raise RuntimeError("broad contextual-warp v2 protocol or sources drifted")
    catalog = {row["cell_id"]: row for row in v1._eligible_catalog()}
    v8 = json.loads(v1.V8_RESULT_PATH.read_text(encoding="utf-8"))
    baselines = [row for row in v8["outcomes"] if row["cell_id"] in catalog]
    RUN_DIR.mkdir(parents=True, exist_ok=False)
    old_adaptive = adaptive.RUN_DIR
    old_spectral = spectral_v1.RUN_DIR
    adaptive.RUN_DIR = RUN_DIR
    spectral_v1.RUN_DIR = RUN_DIR
    outcomes = []
    started = time.perf_counter()
    try:
        for baseline in baselines:
            try:
                contextual = v1._contextual_baseline(
                    baseline, catalog[baseline["cell_id"]]
                )
                outcome = spectral_v1._slot_outcome(baseline, contextual)
                outcome["contextual_calibration"] = contextual[
                    "contextual_calibration"
                ]
            except (RuntimeError, ValueError) as exc:
                outcome = {
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
                    "selection_complete": False,
                    "unresolved_occurrence_indexes": list(
                        range(len(baseline["occurrence_outcomes"]))
                    ),
                    "occurrence_selections": [],
                    "occurrence_outcomes": [],
                    "candidates": [],
                    "error": f"{type(exc).__name__}: {exc}",
                    "api_calls_made": 0,
                    "production_enabled": False,
                }
            outcomes.append(outcome)
    finally:
        adaptive.RUN_DIR = old_adaptive
        spectral_v1.RUN_DIR = old_spectral
    v1_hashes = _audio_hashes(V1_RUN_DIR)
    v2_hashes = _audio_hashes(RUN_DIR)
    audio_equivalence = bool(v1_hashes == v2_hashes)
    cells = _summaries(outcomes)
    counts = Counter(row["classification"] for row in cells)
    payload = {
        "schema_version": 1,
        "version": VERSION,
        "run_id": RUN_ID,
        "protocol_sha256": protocol["protocol_sha256"],
        "classification": (
            "broad_oral_monophthong_known_fixture_characterization"
            if audio_equivalence
            else "runtime_inconclusive_audio_equivalence_failure"
        ),
        "audio_hash_equivalence_pass": audio_equivalence,
        "audio_file_count": len(v2_hashes),
        "cell_classification_counts": dict(sorted(counts.items())),
        "cell_count": len(cells),
        "logical_slot_count": len(outcomes),
        "candidate_count": sum(len(row["candidates"]) for row in outcomes),
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
