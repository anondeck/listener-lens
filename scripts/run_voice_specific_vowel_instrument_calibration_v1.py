#!/usr/bin/env python3
from __future__ import annotations

from collections import defaultdict
import argparse
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Sequence

import numpy as np

from earshift_bakeoff.bilingual_vowel_acoustics import run_praat_formant_frames
from earshift_bakeoff.bilingual_vowel_acoustics_v2 import (
    MINIMUM_ANCHOR_SEPARATION_BARK_RMS,
    MINIMUM_DIRECTION_COSINE,
    stress_core_measurement,
)
from earshift_bakeoff.config import Paths, sha256_json, stable_json
from earshift_bakeoff.util import atomic_write_json, sha256_file


VERSION = "voice-specific-vowel-instrument-calibration-v1"
RUN_ID = "20260718-voice-specific-vowel-instrument-calibration-v1"
PROTOCOL_PATH = Paths().root / "rules" / f"{VERSION}.json"
RUN_DIR = Paths().artifacts / "product-matrix" / RUN_ID
RESULT_PATH = RUN_DIR / "results.json"
V1_RESULT_PATH = (
    Paths().artifacts
    / "product-matrix"
    / "20260717-bilingual-product-vowel-acoustic-screen-v1"
    / "results.json"
)
V1_DIR = V1_RESULT_PATH.parent
V8_RESULT_PATH = (
    Paths().artifacts
    / "product-matrix"
    / "20260717-bilingual-product-v8-vowel-acoustic-screen"
    / "results.json"
)
PRAAT_PATH = Path("/Applications/Praat.app/Contents/MacOS/Praat")
PRAAT_SCRIPT_PATH = (
    Paths().root / "scripts" / "praat_bilingual_vowel_trajectory_v1.praat"
)
VOICE_ORDER = ("af_heart", "am_michael", "pm_alex", "pf_dora")
FEMALE_CEILING_ORDER = (5_500, 5_250, 5_750, 5_000, 6_000, 4_750, 4_500)
MALE_CEILING_ORDER = (5_000, 4_750, 5_250, 4_500, 5_500, 5_750, 6_000)
CEILING_ORDER_BY_VOICE = {
    "af_heart": FEMALE_CEILING_ORDER,
    "am_michael": MALE_CEILING_ORDER,
    "pm_alex": MALE_CEILING_ORDER,
    "pf_dora": FEMALE_CEILING_ORDER,
}
EXPECTED_OCCURRENCES_PER_CELL = 4


def _semantic_hash(payload: Any) -> str:
    return hashlib.sha256(stable_json(payload).encode("utf-8")).hexdigest()


def _source_bindings() -> dict[str, str]:
    paths = (
        "scripts/run_voice_specific_vowel_instrument_calibration_v1.py",
        "scripts/praat_bilingual_vowel_trajectory_v1.praat",
        "src/earshift_bakeoff/bilingual_vowel_acoustics.py",
        "src/earshift_bakeoff/bilingual_vowel_acoustics_v2.py",
    )
    return {path: sha256_file(Paths().root / path) for path in paths}


def protocol_record() -> dict[str, Any]:
    v1 = json.loads(V1_RESULT_PATH.read_text(encoding="utf-8"))
    v8 = json.loads(V8_RESULT_PATH.read_text(encoding="utf-8"))
    payload = {
        "schema_version": 1,
        "version": VERSION,
        "run_id": RUN_ID,
        "status": "frozen_before_anchor_remeasurement",
        "purpose": (
            "Select a voice- and rule-specific Praat ceiling and displacement "
            "prototype from natural anchors only, before any controlled candidate "
            "audio is inspected."
        ),
        "source_bindings": {
            "v1_result_sha256": sha256_file(V1_RESULT_PATH),
            "v1_record_sha256": v1["record_sha256"],
            "v8_result_sha256": sha256_file(V8_RESULT_PATH),
            "v8_record_sha256": v8["record_sha256"],
            "files": _source_bindings(),
        },
        "scope": {
            "voices": list(VOICE_ORDER),
            "voice_rule_cell_count": 80,
            "logical_slot_count": 240,
            "natural_anchor_wav_count": 480,
            "expected_occurrences_per_cell": EXPECTED_OCCURRENCES_PER_CELL,
        },
        "instrument": {
            "name": "Praat Burg",
            "praat_binary_sha256": sha256_file(PRAAT_PATH),
            "praat_script_sha256": sha256_file(PRAAT_SCRIPT_PATH),
            "ceiling_order_by_voice": {
                voice: list(order) for voice, order in CEILING_ORDER_BY_VOICE.items()
            },
            "time_step_s": 0.005,
            "maximum_formant_count": 5,
            "window_length_s": 0.025,
            "pre_emphasis_from_hz": 50,
        },
        "selection": {
            "candidate_data": "natural source and target anchors only",
            "candidate_audio_excluded": True,
            "per_occurrence_gates": {
                "both_endpoints_measurable": True,
                "minimum_anchor_separation_bark_rms": (
                    MINIMUM_ANCHOR_SEPARATION_BARK_RMS
                ),
            },
            "cell_gates": {
                "required_occurrence_count": EXPECTED_OCCURRENCES_PER_CELL,
                "minimum_direction_cosine_to_cell_median": MINIMUM_DIRECTION_COSINE,
                "all_occurrences_must_pass": True,
            },
            "rule": (
                "Select the first ceiling in the frozen voice-specific order at "
                "which all four occurrences pass. If none passes, leave the cell "
                "uncalibrated; do not choose a best-looking partial setting."
            ),
            "prototype": (
                "Median Bark F1/F2 target-minus-source displacement across all four "
                "passing natural-anchor occurrences at the selected ceiling."
            ),
        },
        "stopping_rule": (
            "Measure every frozen anchor at every listed ceiling once. This run may "
            "publish a calibration catalog but cannot promote a renderer candidate."
        ),
        "scope_controls": {
            "audio_renders": 0,
            "api_calls": 0,
            "paid_calls": 0,
            "candidate_audio_read": False,
            "production_enabled": False,
            "deployment": False,
        },
    }
    return {**payload, "protocol_sha256": sha256_json(payload)}


def prepare() -> dict[str, Any]:
    protocol = protocol_record()
    if PROTOCOL_PATH.exists():
        existing = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        if stable_json(existing) != stable_json(protocol):
            raise RuntimeError("voice-specific calibration protocol drifted")
        return existing
    if RUN_DIR.exists():
        raise RuntimeError("calibration run exists before protocol freeze")
    atomic_write_json(PROTOCOL_PATH, protocol)
    return protocol


def _measure(
    *,
    path: Path,
    stem: str,
    intervals: Sequence[dict[str, Any]],
    ceiling: int,
    mode: str,
) -> tuple[dict[str, Any], ...]:
    frame_path = RUN_DIR / "praat-frames" / f"{stem}__ceiling-{ceiling}.tsv"
    frames = run_praat_formant_frames(
        path,
        frame_path,
        maximum_formant_hz=ceiling,
        praat_path=PRAAT_PATH,
        script_path=PRAAT_SCRIPT_PATH,
    )
    return tuple(
        stress_core_measurement(
            frames,
            start_s=float(interval["start_s"]),
            end_s=float(interval["end_s"]),
            mode=mode,
        )
        for interval in intervals
    )


def _safe_name(value: str) -> str:
    return "".join(character if character.isalnum() else "-" for character in value)


def _distance(left: np.ndarray, right: np.ndarray) -> float:
    return math.sqrt(float(np.mean(np.square(left - right))))


def _slot_measurements(
    v8_outcome: dict[str, Any], v1_outcome: dict[str, Any]
) -> dict[str, Any]:
    source_record = v1_outcome["anchor_audio"]["source"]
    target_record = v1_outcome["anchor_audio"]["target"]
    source_path = V1_DIR / source_record["relative_path"]
    target_path = V1_DIR / target_record["relative_path"]
    if (
        sha256_file(source_path) != source_record["wav_sha256"]
        or sha256_file(target_path) != target_record["wav_sha256"]
    ):
        raise RuntimeError("frozen natural anchor changed")
    order = CEILING_ORDER_BY_VOICE[v8_outcome["voice_id"]]
    stem = _safe_name(v8_outcome["logical_slot_id"])
    rows = []
    for ceiling in order:
        source = _measure(
            path=source_path,
            stem=f"{stem}__source",
            intervals=v8_outcome["source_anchor_intervals"],
            ceiling=ceiling,
            mode=v8_outcome["measurement_mode"],
        )
        target = _measure(
            path=target_path,
            stem=f"{stem}__target",
            intervals=v8_outcome["target_anchor_intervals"],
            ceiling=ceiling,
            mode=v8_outcome["measurement_mode"],
        )
        if len(source) != len(target):
            raise RuntimeError("natural anchor occurrence count differs")
        occurrences = []
        for index, (source_row, target_row) in enumerate(
            zip(source, target, strict=True)
        ):
            measurable = bool(source_row["measurable"] and target_row["measurable"])
            vector = None
            separation = None
            if measurable:
                source_vector = np.asarray(source_row["feature_bark"], dtype=np.float64)
                target_vector = np.asarray(target_row["feature_bark"], dtype=np.float64)
                delta = target_vector - source_vector
                vector = [float(value) for value in delta]
                separation = _distance(source_vector, target_vector)
            occurrences.append(
                {
                    "occurrence_index": index,
                    "measurable": measurable,
                    "anchor_separation_bark_rms": separation,
                    "separation_pass": bool(
                        measurable
                        and separation is not None
                        and separation >= MINIMUM_ANCHOR_SEPARATION_BARK_RMS
                    ),
                    "displacement_bark": vector,
                    "source": source_row,
                    "target": target_row,
                }
            )
        rows.append({"maximum_formant_hz": ceiling, "occurrences": occurrences})
    return {
        "logical_slot_id": v8_outcome["logical_slot_id"],
        "cell_id": v8_outcome["cell_id"],
        "voice_id": v8_outcome["voice_id"],
        "rule_id": v8_outcome["rule_id"],
        "context": v8_outcome["context"],
        "source": v8_outcome["source"],
        "target": v8_outcome["target"],
        "analysis_by_ceiling": rows,
    }


def _cell_ceiling(
    slots: Sequence[dict[str, Any]], ceiling: int
) -> dict[str, Any]:
    occurrences = []
    for slot in slots:
        analysis = next(
            row
            for row in slot["analysis_by_ceiling"]
            if row["maximum_formant_hz"] == ceiling
        )
        for occurrence in analysis["occurrences"]:
            occurrences.append(
                {
                    "logical_slot_id": slot["logical_slot_id"],
                    "context": slot["context"],
                    **occurrence,
                }
            )
    passing = [row for row in occurrences if row["separation_pass"]]
    prototype = None
    if len(passing) == EXPECTED_OCCURRENCES_PER_CELL:
        prototype_array = np.median(
            np.asarray([row["displacement_bark"] for row in passing]), axis=0
        )
        prototype = [float(value) for value in prototype_array]
        prototype_norm = float(np.linalg.norm(prototype_array))
        for row in passing:
            vector = np.asarray(row["displacement_bark"], dtype=np.float64)
            denominator = float(np.linalg.norm(vector) * prototype_norm)
            row["direction_cosine_to_prototype"] = (
                float(np.dot(vector, prototype_array) / denominator)
                if denominator
                else -1.0
            )
    for row in occurrences:
        row.setdefault("direction_cosine_to_prototype", None)
        row["direction_pass"] = bool(
            row["direction_cosine_to_prototype"] is not None
            and row["direction_cosine_to_prototype"] >= MINIMUM_DIRECTION_COSINE
        )
    complete = bool(
        len(occurrences) == EXPECTED_OCCURRENCES_PER_CELL
        and prototype is not None
        and all(row["separation_pass"] and row["direction_pass"] for row in occurrences)
    )
    return {
        "maximum_formant_hz": ceiling,
        "occurrence_count": len(occurrences),
        "measurable_count": sum(row["measurable"] for row in occurrences),
        "separation_pass_count": sum(row["separation_pass"] for row in occurrences),
        "direction_pass_count": sum(row["direction_pass"] for row in occurrences),
        "prototype_displacement_bark": prototype,
        "complete_pass": complete,
        "occurrences": occurrences,
    }


def _catalog(slots: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for slot in slots:
        grouped[slot["cell_id"]].append(slot)
    catalog = []
    for cell_id, rows in sorted(grouped.items()):
        voice_id = rows[0]["voice_id"]
        analyses = [
            _cell_ceiling(rows, ceiling)
            for ceiling in CEILING_ORDER_BY_VOICE[voice_id]
        ]
        selected = next((row for row in analyses if row["complete_pass"]), None)
        catalog.append(
            {
                "cell_id": cell_id,
                "voice_id": voice_id,
                "rule_id": rows[0]["rule_id"],
                "source": rows[0]["source"],
                "target": rows[0]["target"],
                "status": "calibrated" if selected else "uncalibrated",
                "selected_maximum_formant_hz": (
                    selected["maximum_formant_hz"] if selected else None
                ),
                "prototype_displacement_bark": (
                    selected["prototype_displacement_bark"] if selected else None
                ),
                "analysis_by_ceiling": analyses,
                "production_enabled": False,
            }
        )
    return catalog


def run() -> dict[str, Any]:
    if RESULT_PATH.exists():
        return json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if stable_json(protocol) != stable_json(protocol_record()):
        raise RuntimeError("voice-specific calibration protocol or sources drifted")
    v1 = json.loads(V1_RESULT_PATH.read_text(encoding="utf-8"))
    v8 = json.loads(V8_RESULT_PATH.read_text(encoding="utf-8"))
    v1_by_id = {row["logical_slot_id"]: row for row in v1["outcomes"]}
    if len(v8["outcomes"]) != 240 or len(v1_by_id) != 240:
        raise RuntimeError("broad vowel denominator drifted")
    RUN_DIR.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    slots = [
        _slot_measurements(row, v1_by_id[row["logical_slot_id"]])
        for row in v8["outcomes"]
    ]
    catalog = _catalog(slots)
    calibrated = [row for row in catalog if row["status"] == "calibrated"]
    by_voice = {
        voice: {
            "calibrated": sum(
                row["voice_id"] == voice and row["status"] == "calibrated"
                for row in catalog
            ),
            "total": sum(row["voice_id"] == voice for row in catalog),
        }
        for voice in VOICE_ORDER
    }
    payload = {
        "schema_version": 1,
        "version": VERSION,
        "run_id": RUN_ID,
        "protocol_sha256": protocol["protocol_sha256"],
        "classification": "anchor_only_voice_specific_calibration_characterization",
        "voice_rule_cell_count": len(catalog),
        "calibrated_cell_count": len(calibrated),
        "uncalibrated_cell_count": len(catalog) - len(calibrated),
        "logical_slot_count": len(slots),
        "natural_anchor_wav_count": len(slots) * 2,
        "audio_renders": 0,
        "api_calls_made": 0,
        "candidate_audio_read": False,
        "production_enabled": False,
        "elapsed_s": time.perf_counter() - started,
        "summary_by_voice": by_voice,
        "catalog": catalog,
        "slot_measurements": slots,
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
