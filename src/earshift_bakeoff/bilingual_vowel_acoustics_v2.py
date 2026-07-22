from __future__ import annotations

import math
from statistics import median
from typing import Any, Literal, Sequence

import numpy as np

from .bilingual_vowel_acoustics import bark


VOWEL_ACOUSTIC_VERSION_V2 = "bilingual-vowel-stress-core-acoustic-v2"
MeasurementMode = Literal["monophthong_core", "diphthong_core_trajectory"]
DIPHTHONG_SYMBOLS = frozenset("AIOWY")
MONOPHTHONG_BINS = ((0.25, 0.75),)
DIPHTHONG_BINS = ((0.25, 0.50), (0.50, 0.75))
MINIMUM_VALID_FRAMES_PER_BIN = 3
MINIMUM_VALID_FRAME_FRACTION = 0.60
MINIMUM_ANCHOR_SEPARATION_BARK_RMS = 0.18
MINIMUM_CONTROLLED_MOVEMENT_BARK_RMS = 0.18
MINIMUM_CONTROLLED_MOVEMENT_FRACTION = 0.50
MINIMUM_DIRECTION_COSINE = 0.50
MINIMUM_RHOTIC_ANCHOR_SEPARATION_BARK_RMS = 0.10
MINIMUM_RHOTIC_CONTROLLED_MOVEMENT_BARK_RMS = 0.10


def measurement_mode(source: str, target: str) -> MeasurementMode:
    return (
        "diphthong_core_trajectory"
        if any(symbol in DIPHTHONG_SYMBOLS for symbol in source + target)
        else "monophthong_core"
    )


def stress_core_measurement(
    frames: Sequence[dict[str, float | None]],
    *,
    start_s: float,
    end_s: float,
    mode: MeasurementMode,
) -> dict[str, Any]:
    if not 0 <= start_s < end_s:
        raise ValueError("invalid stress-plus-vowel interval")
    bins = MONOPHTHONG_BINS if mode == "monophthong_core" else DIPHTHONG_BINS
    duration = end_s - start_s
    measurements: list[dict[str, Any]] = []
    feature: list[float] = []
    rhoticity: list[float] = []
    for bin_index, (start_fraction, end_fraction) in enumerate(bins):
        inclusive_end = bin_index == len(bins) - 1
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
            row
            for row in selected
            if row["f1_hz"] is not None and row["f2_hz"] is not None
        ]
        valid_f3 = [row for row in valid if row["f3_hz"] is not None]
        if valid:
            f1_hz = median(float(row["f1_hz"]) for row in valid)
            f2_hz = median(float(row["f2_hz"]) for row in valid)
            f3_hz = (
                median(float(row["f3_hz"]) for row in valid_f3) if valid_f3 else None
            )
            f1_bark = bark(f1_hz)
            f2_bark = bark(f2_hz)
            f3_bark = bark(f3_hz) if f3_hz is not None else None
            plausible = bool(
                180 <= f1_hz <= 1200 and 600 <= f2_hz <= 3500 and f2_hz - f1_hz >= 250
            )
        else:
            f1_hz = f2_hz = f3_hz = None
            f1_bark = f2_bark = f3_bark = None
            plausible = False
        frame_fraction = len(valid) / len(selected) if selected else 0.0
        retention = bool(
            len(valid) >= MINIMUM_VALID_FRAMES_PER_BIN
            and frame_fraction >= MINIMUM_VALID_FRAME_FRACTION
        )
        if retention and plausible:
            feature.extend((float(f1_bark), float(f2_bark)))
            if f3_bark is not None:
                rhoticity.append(float(f3_bark) - float(f2_bark))
        measurements.append(
            {
                "fraction": [start_fraction, end_fraction],
                "frame_count": len(selected),
                "valid_frame_count": len(valid),
                "valid_frame_fraction": frame_fraction,
                "f1_hz": f1_hz,
                "f2_hz": f2_hz,
                "f3_hz": f3_hz,
                "f1_bark": f1_bark,
                "f2_bark": f2_bark,
                "f3_bark": f3_bark,
                "retention_pass": retention,
                "plausibility_pass": plausible,
            }
        )
    measurable = bool(
        measurements
        and all(
            row["retention_pass"] and row["plausibility_pass"] for row in measurements
        )
        and len(feature) == len(bins) * 2
    )
    return {
        "version": VOWEL_ACOUSTIC_VERSION_V2,
        "mode": mode,
        "interval_s": [start_s, end_s],
        "duration_ms": duration * 1000.0,
        "bins": measurements,
        "measurable": measurable,
        "feature_bark": feature if measurable else None,
        "rhoticity_gap_bark": (
            rhoticity if measurable and len(rhoticity) == len(bins) else None
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


def classify_stress_core_endpoint(
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
    exact_movement_threshold = max(
        minimum_controlled_movement,
        MINIMUM_CONTROLLED_MOVEMENT_FRACTION * anchor_separation,
    )
    denominator = float(
        np.linalg.norm(anchor_vector) * np.linalg.norm(controlled_vector)
    )
    direction = (
        float(np.dot(anchor_vector, controlled_vector) / denominator)
        if denominator
        else -1.0
    )

    def distance(left: np.ndarray, right: np.ndarray) -> float:
        return math.sqrt(float(np.mean(np.square(left - right))))

    neutral_source = distance(neutral_point, source)
    neutral_target = distance(neutral_point, target)
    lens_source = distance(lens_point, source)
    lens_target = distance(lens_point, target)
    gates = {
        "anchor_gate_pass": anchor_separation >= minimum_anchor_separation,
        "movement_gate_pass": controlled_movement >= minimum_controlled_movement,
        "exact_movement_gate_pass": controlled_movement >= exact_movement_threshold,
        "direction_gate_pass": direction >= MINIMUM_DIRECTION_COSINE,
        "neutral_endpoint_gate_pass": neutral_source < neutral_target,
        "lens_endpoint_gate_pass": lens_target < lens_source,
        "target_gain_gate_pass": lens_target < neutral_target,
        "source_departure_gate_pass": lens_source > neutral_source,
    }
    directional = bool(
        gates["anchor_gate_pass"]
        and gates["movement_gate_pass"]
        and gates["direction_gate_pass"]
        and gates["target_gain_gate_pass"]
        and gates["source_departure_gate_pass"]
    )
    exact = bool(
        directional
        and gates["exact_movement_gate_pass"]
        and gates["neutral_endpoint_gate_pass"]
        and gates["lens_endpoint_gate_pass"]
    )
    return {
        "anchor_separation_bark_rms": anchor_separation,
        "minimum_anchor_separation_bark_rms": minimum_anchor_separation,
        "controlled_movement_bark_rms": controlled_movement,
        "minimum_controlled_movement_bark_rms": minimum_controlled_movement,
        "minimum_exact_controlled_movement_bark_rms": exact_movement_threshold,
        "minimum_controlled_movement_fraction": (MINIMUM_CONTROLLED_MOVEMENT_FRACTION),
        "direction_cosine": direction,
        "minimum_direction_cosine": MINIMUM_DIRECTION_COSINE,
        "neutral_source_distance_bark_rms": neutral_source,
        "neutral_target_distance_bark_rms": neutral_target,
        "lens_source_distance_bark_rms": lens_source,
        "lens_target_distance_bark_rms": lens_target,
        **gates,
        "directional_pass": directional,
        "exact_category_pass": exact,
        "classification": (
            "exact_category_pass"
            if exact
            else "directional_only_pass"
            if directional
            else "fail"
        ),
    }
