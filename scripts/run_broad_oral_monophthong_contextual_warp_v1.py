#!/usr/bin/env python3
from __future__ import annotations

from collections import Counter
import argparse
import copy
import hashlib
import json
import time
from typing import Any

import numpy as np

from earshift_bakeoff.config import Paths, sha256_json, stable_json
from earshift_bakeoff.util import atomic_write_json, sha256_file

import run_bilingual_vowel_adaptive_strength_screen_v1 as adaptive
import run_english_central_vowel_spectral_correction_v1 as spectral_v1


VERSION = "broad-oral-monophthong-contextual-warp-v1"
RUN_ID = "20260718-broad-oral-monophthong-contextual-warp-v1"
PROTOCOL_PATH = Paths().root / "rules" / f"{VERSION}.json"
RUN_DIR = Paths().artifacts / "product-matrix" / RUN_ID
RESULT_PATH = RUN_DIR / "results.json"
CALIBRATION_PATH = (
    Paths().artifacts
    / "product-matrix"
    / "20260718-voice-specific-vowel-instrument-calibration-v1"
    / "results.json"
)
V8_RESULT_PATH = adaptive.V8_RESULT_PATH
V8_DIR = V8_RESULT_PATH.parent
STRENGTHS = spectral_v1.STRENGTHS


def _semantic_hash(payload: Any) -> str:
    return hashlib.sha256(stable_json(payload).encode("utf-8")).hexdigest()


def _bark_to_hz(value: float) -> float:
    denominator = 26.81 / (value + 0.53) - 1.0
    if denominator <= 0.0:
        raise ValueError("Bark target cannot be converted to a positive frequency")
    return 1960.0 / denominator


def _eligible_catalog() -> tuple[dict[str, Any], ...]:
    calibration = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))
    v8 = json.loads(V8_RESULT_PATH.read_text(encoding="utf-8"))
    baseline = {row["cell_id"]: row for row in v8["outcomes"]}
    rows = []
    for cell in calibration["catalog"]:
        sample = baseline[cell["cell_id"]]
        if (
            cell["status"] == "calibrated"
            and sample["measurement_mode"] == "monophthong_core"
            and "rhotic" not in cell["rule_id"]
            and "nasal" not in cell["rule_id"]
        ):
            rows.append(cell)
    return tuple(sorted(rows, key=lambda row: row["cell_id"]))


def _source_bindings() -> dict[str, str]:
    paths = (
        "scripts/run_broad_oral_monophthong_contextual_warp_v1.py",
        "scripts/run_english_central_vowel_spectral_correction_v1.py",
        "src/earshift_bakeoff/spectral_envelope_warp.py",
        "scripts/run_bilingual_vowel_adaptive_strength_screen_v1.py",
    )
    return {path: sha256_file(Paths().root / path) for path in paths}


def protocol_record() -> dict[str, Any]:
    catalog = _eligible_catalog()
    v8 = json.loads(V8_RESULT_PATH.read_text(encoding="utf-8"))
    eligible_ids = {row["cell_id"] for row in catalog}
    slots = [row for row in v8["outcomes"] if row["cell_id"] in eligible_ids]
    occurrences = sum(len(row["occurrence_outcomes"]) for row in slots)
    payload = {
        "schema_version": 1,
        "version": VERSION,
        "run_id": RUN_ID,
        "status": "frozen_before_candidate_processing",
        "purpose": (
            "Test one context-local identity-preserving category-control mechanism "
            "across every calibrated oral-monophthong cell in both directions."
        ),
        "parents": {
            "calibration_sha256": sha256_file(CALIBRATION_PATH),
            "calibration_record_sha256": json.loads(
                CALIBRATION_PATH.read_text(encoding="utf-8")
            )["record_sha256"],
            "v8_result_sha256": sha256_file(V8_RESULT_PATH),
            "v8_record_sha256": v8["record_sha256"],
        },
        "scope": {
            "cell_ids": [row["cell_id"] for row in catalog],
            "cell_count": len(catalog),
            "logical_slot_count": len(slots),
            "target_occurrence_count": occurrences,
            "strength_order": list(STRENGTHS),
            "candidate_count": len(slots) * len(STRENGTHS),
            "included": "calibrated nonrhotic, nonnasal oral monophthongs",
            "excluded": "diphthongs, rhotics, nasals, and uncalibrated cells",
        },
        "intervention": {
            "editor": "spectral-envelope-warp-v1",
            "source_endpoint": "actual controlled-neutral F1/F2 at the calibrated ceiling",
            "target_endpoint": (
                "actual neutral Bark point plus the frozen cell-level natural-anchor "
                "target-minus-source displacement"
            ),
            "selection": (
                "first exact-category strength per occurrence in frozen order, "
                "otherwise first directional-only; no listening selection"
            ),
        },
        "gates": {
            "acoustic": "unchanged v8 endpoint thresholds at the calibrated ceiling",
            "engineering": "unchanged spectral-v1 waveform-integrity gates",
            "aggregation": "all four occurrences in all three contexts per cell",
        },
        "stopping_rule": (
            "Process all candidates once without replacement. Known-fixture passes "
            "permit unseen confirmation only."
        ),
        "scope_controls": {
            "kokoro_renders": 0,
            "api_calls": 0,
            "paid_calls": 0,
            "production_enabled": False,
            "deployment": False,
        },
        "source_bindings": _source_bindings(),
    }
    return {**payload, "protocol_sha256": sha256_json(payload)}


def prepare() -> dict[str, Any]:
    protocol = protocol_record()
    if PROTOCOL_PATH.exists():
        existing = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        if stable_json(existing) != stable_json(protocol):
            raise RuntimeError("broad contextual-warp protocol drifted")
        return existing
    if RUN_DIR.exists():
        raise RuntimeError("broad contextual-warp run exists before protocol")
    atomic_write_json(PROTOCOL_PATH, protocol)
    return protocol


def _target_measurement(
    neutral: dict[str, Any], displacement_bark: list[float]
) -> dict[str, Any]:
    if not neutral["measurable"] or len(neutral["bins"]) != 1:
        raise RuntimeError("controlled neutral is not a measurable monophthong")
    result = copy.deepcopy(neutral)
    target_bark = (
        np.asarray(neutral["feature_bark"], dtype=np.float64)
        + np.asarray(displacement_bark, dtype=np.float64)
    )
    if target_bark.shape != (2,) or not np.isfinite(target_bark).all():
        raise RuntimeError("calibrated target displacement is invalid")
    f1_hz = _bark_to_hz(float(target_bark[0]))
    f2_hz = _bark_to_hz(float(target_bark[1]))
    if not 180.0 <= f1_hz < f2_hz <= 3_500.0 or f2_hz - f1_hz < 250.0:
        raise RuntimeError("context-local target endpoint is implausible")
    result["feature_bark"] = [float(value) for value in target_bark]
    result["bins"][0].update(
        {
            "f1_hz": f1_hz,
            "f2_hz": f2_hz,
            "f1_bark": float(target_bark[0]),
            "f2_bark": float(target_bark[1]),
        }
    )
    return result


def _contextual_baseline(
    baseline: dict[str, Any], cell: dict[str, Any]
) -> dict[str, Any]:
    neutral_path = V8_DIR / baseline["audio"]["neutral"]["relative_path"]
    intervals = tuple(
        row["measurement_interval"] for row in baseline["occurrence_outcomes"]
    )
    ceiling = int(cell["selected_maximum_formant_hz"])
    stem = adaptive._safe_name(baseline["logical_slot_id"])
    neutral = adaptive._measure(
        path=neutral_path,
        stem=f"{stem}__contextual-neutral",
        intervals=intervals,
        ceiling=ceiling,
        mode="monophthong_core",
    )
    target = tuple(
        _target_measurement(row, cell["prototype_displacement_bark"])
        for row in neutral
    )
    result = copy.deepcopy(baseline)
    result["analysis_by_formant_ceiling"] = [
        {
            "maximum_formant_hz": ceiling,
            "measurements": {
                "source_anchor": neutral,
                "target_anchor": target,
                "neutral": neutral,
            },
        }
    ]
    result["contextual_calibration"] = {
        "maximum_formant_hz": ceiling,
        "prototype_displacement_bark": cell["prototype_displacement_bark"],
    }
    return result


def run() -> dict[str, Any]:
    if RESULT_PATH.exists():
        return json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if stable_json(protocol) != stable_json(protocol_record()):
        raise RuntimeError("broad contextual-warp protocol or sources drifted")
    catalog = {row["cell_id"]: row for row in _eligible_catalog()}
    v8 = json.loads(V8_RESULT_PATH.read_text(encoding="utf-8"))
    baselines = [row for row in v8["outcomes"] if row["cell_id"] in catalog]
    if len(baselines) != protocol["scope"]["logical_slot_count"]:
        raise RuntimeError("broad contextual-warp denominator drifted")
    RUN_DIR.mkdir(parents=True, exist_ok=False)
    old_adaptive = adaptive.RUN_DIR
    old_spectral = spectral_v1.RUN_DIR
    adaptive.RUN_DIR = RUN_DIR
    spectral_v1.RUN_DIR = RUN_DIR
    started = time.perf_counter()
    outcomes = []
    try:
        for baseline in baselines:
            cell = catalog[baseline["cell_id"]]
            try:
                contextual = _contextual_baseline(baseline, cell)
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
    cells = spectral_v1._summaries(outcomes)
    counts = Counter(row["classification"] for row in cells)
    payload = {
        "schema_version": 1,
        "version": VERSION,
        "run_id": RUN_ID,
        "protocol_sha256": protocol["protocol_sha256"],
        "classification": "broad_oral_monophthong_known_fixture_characterization",
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
