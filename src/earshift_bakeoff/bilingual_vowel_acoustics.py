from __future__ import annotations

import csv
import math
from pathlib import Path
from statistics import median
import subprocess
from typing import Any, Sequence

import numpy as np


VOWEL_TRAJECTORY_ACOUSTIC_VERSION = "bilingual-vowel-trajectory-acoustic-v1"
TRAJECTORY_BINS = (
    (0.10, 0.3666666667),
    (0.3666666667, 0.6333333333),
    (0.6333333333, 0.90),
)
MINIMUM_TOTAL_VALID_FRAMES = 5
MINIMUM_VALID_FRAMES_PER_BIN = 2
MINIMUM_VALID_FRAME_FRACTION = 0.60
MINIMUM_ANCHOR_SEPARATION_BARK_RMS = 0.12
MINIMUM_CONTROLLED_MOVEMENT_BARK_RMS = 0.10
MINIMUM_ANCHOR_MOVEMENT_FRACTION = 0.35
MINIMUM_DIRECTION_COSINE = 0.50
MINIMUM_RHOTIC_ANCHOR_SEPARATION_BARK_RMS = 0.08
MINIMUM_RHOTIC_CONTROLLED_MOVEMENT_BARK_RMS = 0.06


def _number(value: str | None) -> float | None:
    try:
        result = float(value or "")
    except ValueError:
        return None
    return result if math.isfinite(result) else None


def bark(hz: float) -> float:
    return 26.81 * hz / (1960.0 + hz) - 0.53


def run_praat_formant_frames(
    wav_path: Path,
    output_path: Path,
    *,
    maximum_formant_hz: int,
    praat_path: Path,
    script_path: Path,
) -> list[dict[str, float | None]]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        [
            str(praat_path),
            "--run",
            str(script_path),
            str(wav_path),
            str(output_path),
            str(maximum_formant_hz),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr or completed.stdout)
    with output_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    frames = [
        {key: _number(row.get(key)) for key in ("time_s", "f1_hz", "f2_hz", "f3_hz")}
        for row in rows
    ]
    if not frames or any(row["time_s"] is None for row in frames):
        raise RuntimeError("Praat vowel probe returned no valid time frames")
    return frames


def trajectory_measurement(
    frames: Sequence[dict[str, float | None]],
    *,
    start_s: float,
    end_s: float,
) -> dict[str, Any]:
    if not 0 <= start_s < end_s:
        raise ValueError("invalid vowel measurement interval")
    duration = end_s - start_s
    bins: list[dict[str, Any]] = []
    feature: list[float] = []
    total_frames = 0
    valid_frames = 0
    for bin_index, (start_fraction, end_fraction) in enumerate(TRAJECTORY_BINS):
        inclusive_end = bin_index == len(TRAJECTORY_BINS) - 1
        selected = [
            row
            for row in frames
            if row["time_s"] is not None
            and start_s + duration * start_fraction <= float(row["time_s"])
            and (
                float(row["time_s"]) < start_s + duration * end_fraction
                or (
                    inclusive_end
                    and float(row["time_s"]) <= start_s + duration * end_fraction
                )
            )
        ]
        valid = [
            (
                float(row["f1_hz"]),
                float(row["f2_hz"]),
                None if row["f3_hz"] is None else float(row["f3_hz"]),
            )
            for row in selected
            if row["f1_hz"] is not None and row["f2_hz"] is not None
        ]
        total_frames += len(selected)
        valid_frames += len(valid)
        if valid:
            f1_hz = median(row[0] for row in valid)
            f2_hz = median(row[1] for row in valid)
            valid_f3 = [row[2] for row in valid if row[2] is not None]
            f3_hz = median(valid_f3) if valid_f3 else None
            f1_bark = bark(f1_hz)
            f2_bark = bark(f2_hz)
            f3_bark = bark(f3_hz) if f3_hz is not None else None
            plausible = bool(
                180 <= f1_hz <= 1200 and 600 <= f2_hz <= 3500 and f2_hz - f1_hz >= 250
            )
            feature.extend((f1_bark, f2_bark))
        else:
            f1_hz = f2_hz = f3_hz = None
            f1_bark = f2_bark = f3_bark = None
            plausible = False
        bins.append(
            {
                "fraction": [start_fraction, end_fraction],
                "frame_count": len(selected),
                "valid_frame_count": len(valid),
                "f1_hz": f1_hz,
                "f2_hz": f2_hz,
                "f3_hz": f3_hz,
                "f1_bark": f1_bark,
                "f2_bark": f2_bark,
                "f3_bark": f3_bark,
                "plausibility_pass": plausible,
            }
        )
    fraction = valid_frames / total_frames if total_frames else 0.0
    retention_pass = bool(
        valid_frames >= MINIMUM_TOTAL_VALID_FRAMES
        and fraction >= MINIMUM_VALID_FRAME_FRACTION
        and all(
            row["valid_frame_count"] >= MINIMUM_VALID_FRAMES_PER_BIN for row in bins
        )
    )
    plausibility_pass = bool(bins and all(row["plausibility_pass"] for row in bins))
    measurable = bool(retention_pass and plausibility_pass and len(feature) == 6)
    return {
        "version": VOWEL_TRAJECTORY_ACOUSTIC_VERSION,
        "interval_s": [start_s, end_s],
        "duration_ms": duration * 1000.0,
        "total_frame_count": total_frames,
        "valid_frame_count": valid_frames,
        "valid_frame_fraction": fraction,
        "retention_pass": retention_pass,
        "plausibility_pass": plausibility_pass,
        "measurable": measurable,
        "bins": bins,
        "feature_bark": feature if measurable else None,
        "rhoticity_gap_bark": (
            [float(row["f3_bark"]) - float(row["f2_bark"]) for row in bins]
            if measurable and all(row["f3_bark"] is not None for row in bins)
            else None
        ),
    }


def _vector(values: Sequence[float], *, expected_size: int | None = None) -> np.ndarray:
    result = np.asarray(tuple(values), dtype=np.float64)
    if (
        result.ndim != 1
        or result.size == 0
        or (expected_size is not None and result.size != expected_size)
        or not np.isfinite(result).all()
    ):
        raise ValueError("acoustic feature must contain the expected finite values")
    return result


def rms_distance(left: Sequence[float], right: Sequence[float]) -> float:
    a = _vector(left)
    delta = a - _vector(right, expected_size=a.size)
    return math.sqrt(float(np.mean(np.square(delta))))


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    a = _vector(left)
    b = _vector(right, expected_size=a.size)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denominator) if denominator else -1.0


def classify_vowel_endpoint(
    *,
    source_anchor: Sequence[float],
    target_anchor: Sequence[float],
    neutral: Sequence[float],
    lens: Sequence[float],
    minimum_anchor_separation: float = MINIMUM_ANCHOR_SEPARATION_BARK_RMS,
    minimum_controlled_movement: float = MINIMUM_CONTROLLED_MOVEMENT_BARK_RMS,
) -> dict[str, Any]:
    source = _vector(source_anchor)
    target = _vector(target_anchor, expected_size=source.size)
    neutral_point = _vector(neutral, expected_size=source.size)
    lens_point = _vector(lens, expected_size=source.size)
    anchor_vector = target - source
    controlled_vector = lens_point - neutral_point
    anchor_separation = math.sqrt(float(np.mean(np.square(anchor_vector))))
    controlled_movement = math.sqrt(float(np.mean(np.square(controlled_vector))))
    movement_threshold = max(
        minimum_controlled_movement,
        MINIMUM_ANCHOR_MOVEMENT_FRACTION * anchor_separation,
    )
    direction = cosine(controlled_vector, anchor_vector)
    neutral_source_distance = rms_distance(neutral_point, source)
    neutral_target_distance = rms_distance(neutral_point, target)
    lens_source_distance = rms_distance(lens_point, source)
    lens_target_distance = rms_distance(lens_point, target)
    anchor_gate = anchor_separation >= minimum_anchor_separation
    movement_gate = controlled_movement >= movement_threshold
    direction_gate = direction >= MINIMUM_DIRECTION_COSINE
    neutral_endpoint_gate = neutral_source_distance < neutral_target_distance
    lens_endpoint_gate = lens_target_distance < lens_source_distance
    target_gain_gate = lens_target_distance < neutral_target_distance
    source_departure_gate = lens_source_distance > neutral_source_distance
    exact = bool(
        anchor_gate
        and movement_gate
        and direction_gate
        and neutral_endpoint_gate
        and lens_endpoint_gate
        and target_gain_gate
        and source_departure_gate
    )
    directional = bool(
        anchor_gate
        and movement_gate
        and direction_gate
        and target_gain_gate
        and source_departure_gate
    )
    return {
        "anchor_separation_bark_rms": anchor_separation,
        "minimum_anchor_separation_bark_rms": (minimum_anchor_separation),
        "controlled_movement_bark_rms": controlled_movement,
        "minimum_controlled_movement_bark_rms": movement_threshold,
        "direction_cosine": direction,
        "minimum_direction_cosine": MINIMUM_DIRECTION_COSINE,
        "neutral_source_distance_bark_rms": neutral_source_distance,
        "neutral_target_distance_bark_rms": neutral_target_distance,
        "lens_source_distance_bark_rms": lens_source_distance,
        "lens_target_distance_bark_rms": lens_target_distance,
        "anchor_gate_pass": anchor_gate,
        "movement_gate_pass": movement_gate,
        "direction_gate_pass": direction_gate,
        "neutral_endpoint_gate_pass": neutral_endpoint_gate,
        "lens_endpoint_gate_pass": lens_endpoint_gate,
        "target_gain_gate_pass": target_gain_gate,
        "source_departure_gate_pass": source_departure_gate,
        "exact_category_pass": exact,
        "directional_pass": directional,
        "classification": (
            "exact_category_pass"
            if exact
            else "directional_only_pass"
            if directional
            else "fail"
        ),
    }
