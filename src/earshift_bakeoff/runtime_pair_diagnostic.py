from __future__ import annotations

import gc
import json
import math
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx_whisper
import numpy as np

from .config import DEVLOG_PATH, Paths
from .pcm import decode_pcm16_mono
from .same_take import WHISPER_MODEL, _probe_frames, align_vowel_core
from .sentence_pair_v2 import (
    ANCHOR_GATE,
    MEASUREMENT_SCRIPT_SHA256,
    PRAAT_SHA256,
)
from .sentence_pair_v2_analysis import CEILINGS, MEASUREMENT_SCRIPT, PRAAT, _measure
from .util import atomic_write_json, sha256_file


RUN_ID = "20260716-post-hardening-selected"
PREREGISTRATION_HEADING = (
    "## Selected runtime-pair acoustic diagnostic — preregistration — July 16, 2026"
)
EXPECTED_HASHES = {
    "neutral": "2ed8f2023db0b61ae7996ce17194e7dd84762c8e49f7daaece965d8dc4873a41",
    "lens": "48e47fb1ce64a3322b7f1b92b7a7a625ec043d5a928df406e0829cf6a6dc7116",
}
TARGET_WORD_INDEX = 7
EXPECTED_WORD_COUNT = 10


def _word_intervals(path: Path, script: str) -> list[dict[str, Any]]:
    result = mlx_whisper.transcribe(
        str(path),
        path_or_hf_repo=str(WHISPER_MODEL),
        language="en",
        temperature=0,
        condition_on_previous_text=False,
        word_timestamps=True,
        initial_prompt=script,
        verbose=False,
    )
    words = [
        word
        for segment in result.get("segments", [])
        for word in segment.get("words", [])
        if str(word.get("word", "")).strip()
    ]
    intervals = [
        {
            "whisper_label": str(word.get("word", "")).strip(),
            "start_s": float(word["start"]),
            "end_s": float(word["end"]),
            "probability": float(word.get("probability") or 0),
        }
        for word in words
    ]
    del result, words
    mx.clear_cache()
    gc.collect()
    if len(intervals) != EXPECTED_WORD_COUNT:
        raise RuntimeError(
            f"requires exactly {EXPECTED_WORD_COUNT} Whisper word intervals; "
            f"got {len(intervals)}"
        )
    if any(
        item["start_s"] < 0
        or item["end_s"] <= item["start_s"]
        or index
        and item["start_s"] < intervals[index - 1]["end_s"] - 1e-6
        for index, item in enumerate(intervals)
    ):
        raise RuntimeError("Whisper word intervals are not monotonic and non-overlapping")
    return intervals


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator else -1.0


def classify_points(
    neutral_by_family: dict[str, tuple[float, float]],
    lens_by_family: dict[str, tuple[float, float]],
) -> dict[str, Any]:
    families: dict[str, Any] = {}
    for ceiling in CEILINGS:
        key = str(ceiling)
        neutral = np.asarray(neutral_by_family[key], dtype=float)
        lens = np.asarray(lens_by_family[key], dtype=float)
        anchor = ANCHOR_GATE["families"][key]
        source = np.asarray(anchor["source_centroid_bark"], dtype=float)
        target = np.asarray(anchor["target_centroid_bark"], dtype=float)
        vector = lens - neutral
        magnitude = float(np.linalg.norm(vector))
        cosine = _cosine(vector, np.asarray(anchor["anchor_vector_bark"], dtype=float))
        neutral_source_distance = float(np.linalg.norm(neutral - source))
        neutral_target_distance = float(np.linalg.norm(neutral - target))
        lens_source_distance = float(np.linalg.norm(lens - source))
        lens_target_distance = float(np.linalg.norm(lens - target))
        neutral_source_pass = neutral_source_distance < neutral_target_distance
        lens_target_pass = lens_target_distance < lens_source_distance
        direction_pass = cosine >= float(ANCHOR_GATE["direction_cosine_minimum"])
        magnitude_pass = magnitude > float(anchor["magnitude_threshold_bark"])
        families[key] = {
            "neutral_bark": neutral.tolist(),
            "lens_bark": lens.tolist(),
            "vector_bark": vector.tolist(),
            "magnitude_bark": magnitude,
            "magnitude_threshold_bark": float(anchor["magnitude_threshold_bark"]),
            "magnitude_pass": magnitude_pass,
            "anchor_direction_cosine": cosine,
            "direction_pass": direction_pass,
            "neutral_source_distance_bark": neutral_source_distance,
            "neutral_target_distance_bark": neutral_target_distance,
            "neutral_source_category_pass": neutral_source_pass,
            "lens_source_distance_bark": lens_source_distance,
            "lens_target_distance_bark": lens_target_distance,
            "lens_target_category_pass": lens_target_pass,
            "category_and_direction_pass": bool(
                neutral_source_pass
                and lens_target_pass
                and direction_pass
                and magnitude_pass
            ),
        }
    complete_category = all(
        item["category_and_direction_pass"] for item in families.values()
    )
    complete_direction = all(
        item["direction_pass"] and item["magnitude_pass"]
        for item in families.values()
    )
    if complete_category:
        classification = "category_and_direction_diagnostic_pass"
    elif complete_direction:
        classification = "directional_only_diagnostic"
    else:
        classification = "diagnostic_fail"
    return {"classification": classification, "families": families}


def analyze_runtime_pair(run_dir: Path | None = None) -> dict[str, Any]:
    run_dir = run_dir or Paths().artifacts / "runtime-pair" / RUN_ID
    if PREREGISTRATION_HEADING not in DEVLOG_PATH.read_text(encoding="utf-8"):
        raise RuntimeError("runtime-pair diagnostic preregistration is missing")
    if sha256_file(PRAAT) != PRAAT_SHA256:
        raise RuntimeError("Praat executable changed")
    if sha256_file(MEASUREMENT_SCRIPT) != MEASUREMENT_SCRIPT_SHA256:
        raise RuntimeError("measurement script changed")
    if not WHISPER_MODEL.is_dir():
        raise RuntimeError("pinned local Whisper model is unavailable")

    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("api_calls_made") != 0 or manifest.get("cache_hit") is not True:
        raise RuntimeError("manifest is not a zero-call cache export")

    records: dict[str, Any] = {}
    try:
        for side in ("neutral", "lens"):
            audio_record = manifest["audio"][side]
            path = run_dir / audio_record["path"]
            digest = sha256_file(path)
            if digest != EXPECTED_HASHES[side] or digest != audio_record["sha256"]:
                raise RuntimeError(f"{side} audio hash mismatch")
            decoded = decode_pcm16_mono(path)
            script = manifest["scripts"][side]
            words = _word_intervals(path, script)
            target_word = words[TARGET_WORD_INDEX]
            core = align_vowel_core(
                _probe_frames(path),
                word_start_s=target_word["start_s"],
                word_end_s=target_word["end_s"],
                search_fraction=(0.10, 0.75),
                sample_rate_hz=decoded.sample_rate_hz,
            )
            measurements = {
                str(ceiling): _measure(path, core, ceiling) for ceiling in CEILINGS
            }
            if not all(item["plausibility_pass"] for item in measurements.values()):
                raise RuntimeError(f"{side} failed Hz plausibility")
            records[side] = {
                "audio_sha256": digest,
                "word_intervals": words,
                "target_word_interval": target_word,
                "vowel_core": core,
                "measurements": measurements,
            }
        classified = classify_points(
            {
                key: (
                    records["neutral"]["measurements"][key]["f1_bark"],
                    records["neutral"]["measurements"][key]["f2_bark"],
                )
                for key in map(str, CEILINGS)
            },
            {
                key: (
                    records["lens"]["measurements"][key]["f1_bark"],
                    records["lens"]["measurements"][key]["f2_bark"],
                )
                for key in map(str, CEILINGS)
            },
        )
        result = {
            "schema_version": 1,
            "status": "analysis_complete",
            "run_id": RUN_ID,
            **classified,
            "records": records,
            "interpretation_limit": (
                "One post-listening selected take per side; this diagnostic cannot "
                "establish repeatability above sentence-level renderer variance."
            ),
        }
    except Exception as exc:
        result = {
            "schema_version": 1,
            "status": "analysis_inconclusive",
            "run_id": RUN_ID,
            "classification": "inconclusive",
            "reason": f"{type(exc).__name__}: {str(exc)[:500]}",
            "records": records,
        }
    atomic_write_json(run_dir / "analysis.json", result)
    return result
