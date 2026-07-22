from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Sequence

import numpy as np


SPECTRAL_CATEGORY_VERSION = "bilingual-vowel-spectral-category-v1"
MINIMUM_ANCHOR_SEPARATION_SCALED_RMS = 0.25
MINIMUM_DIRECTION_COSINE = 0.50
MINIMUM_DIRECTIONAL_MOVEMENT_FRACTION = 0.25
MINIMUM_EXACT_MOVEMENT_FRACTION = 0.50
MINIMUM_HELDOUT_EXACT_ANCHOR_PAIRS = 3
MAXIMUM_REVERSED_HELDOUT_ANCHOR_PAIRS = 0


@dataclass(frozen=True)
class SpectralFeatureConfig:
    sample_rate_hz: int = 24_000
    frame_length_samples: int = 600
    hop_length_samples: int = 240
    fft_size: int = 1_024
    mel_filter_count: int = 32
    minimum_frequency_hz: float = 80.0
    maximum_frequency_hz: float = 7_600.0
    cepstral_coefficient_count: int = 12
    temporal_sample_fractions: tuple[float, ...] = (0.30, 0.50, 0.70)
    log_power_floor: float = 1e-8
    robust_scale_floor: float = 0.05


DEFAULT_FEATURE_CONFIG = SpectralFeatureConfig()


def _finite_vector(
    values: Sequence[float] | np.ndarray,
    *,
    expected_size: int | None = None,
) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64).reshape(-1)
    if (
        vector.size == 0
        or (expected_size is not None and vector.size != expected_size)
        or not np.isfinite(vector).all()
    ):
        raise ValueError("spectral vector must contain the expected finite values")
    return vector


def _hz_to_mel(frequency_hz: np.ndarray | float) -> np.ndarray:
    values = np.asarray(frequency_hz, dtype=np.float64)
    return 2_595.0 * np.log10(1.0 + values / 700.0)


def _mel_to_hz(mel: np.ndarray | float) -> np.ndarray:
    values = np.asarray(mel, dtype=np.float64)
    return 700.0 * (np.power(10.0, values / 2_595.0) - 1.0)


def _mel_filterbank(config: SpectralFeatureConfig) -> np.ndarray:
    if not (
        0
        < config.minimum_frequency_hz
        < config.maximum_frequency_hz
        < config.sample_rate_hz / 2
    ):
        raise ValueError("invalid mel frequency bounds")
    frequencies = np.linspace(
        0.0,
        config.sample_rate_hz / 2,
        config.fft_size // 2 + 1,
        dtype=np.float64,
    )
    mel_points = np.linspace(
        float(_hz_to_mel(config.minimum_frequency_hz)),
        float(_hz_to_mel(config.maximum_frequency_hz)),
        config.mel_filter_count + 2,
        dtype=np.float64,
    )
    hz_points = _mel_to_hz(mel_points)
    filters = np.zeros((config.mel_filter_count, frequencies.size), dtype=np.float64)
    for index in range(config.mel_filter_count):
        left, center, right = hz_points[index : index + 3]
        rising = (frequencies - left) / max(center - left, np.finfo(float).eps)
        falling = (right - frequencies) / max(right - center, np.finfo(float).eps)
        filters[index] = np.maximum(0.0, np.minimum(rising, falling))
    normalizers = filters.sum(axis=1, keepdims=True)
    if np.any(normalizers <= 0):
        raise ValueError("degenerate mel filterbank")
    return filters / normalizers


def _dct_matrix(config: SpectralFeatureConfig) -> np.ndarray:
    # Orthonormal DCT-II. Coefficient zero is intentionally omitted because
    # frame-level log-energy is not part of the vowel-category feature.
    n = np.arange(config.mel_filter_count, dtype=np.float64) + 0.5
    k = np.arange(1, config.cepstral_coefficient_count + 1, dtype=np.float64)[:, None]
    return math.sqrt(2.0 / config.mel_filter_count) * np.cos(
        np.pi * k * n / config.mel_filter_count
    )


def spectral_trajectory_feature(
    pcm: Sequence[int] | np.ndarray,
    *,
    start_sample: int,
    end_sample_exclusive: int,
    config: SpectralFeatureConfig = DEFAULT_FEATURE_CONFIG,
) -> dict[str, Any]:
    values = np.asarray(pcm)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("PCM must be a nonempty mono vector")
    if not np.issubdtype(values.dtype, np.number):
        raise ValueError("PCM must be numeric")
    if not 0 <= start_sample < end_sample_exclusive <= values.size:
        raise ValueError("spectral interval is outside the PCM vector")
    if config.fft_size < config.frame_length_samples:
        raise ValueError("FFT size must contain one complete analysis frame")
    interval = values[start_sample:end_sample_exclusive].astype(np.float64)
    if not np.isfinite(interval).all():
        raise ValueError("PCM contains non-finite samples")
    if interval.size < config.frame_length_samples:
        raise ValueError("spectral interval is shorter than one analysis frame")
    peak = float(np.max(np.abs(interval)))
    if peak <= 0:
        raise ValueError("spectral interval is silent")
    # The input is normally PCM16. Scaling by a fixed constant preserves exact
    # reproducibility and frame-level mean removal makes the feature insensitive
    # to a global amplitude multiplier.
    interval /= 32_768.0
    starts = np.arange(
        0,
        interval.size - config.frame_length_samples + 1,
        config.hop_length_samples,
        dtype=np.int64,
    )
    if starts.size == 0:
        raise ValueError("spectral interval produced no complete frames")
    window = np.hanning(config.frame_length_samples).astype(np.float64)
    frames = np.stack(
        [interval[start : start + config.frame_length_samples] for start in starts]
    )
    frame_rms = np.sqrt(np.mean(np.square(frames), axis=1, keepdims=True))
    frames = frames / np.maximum(frame_rms, 1e-8)
    spectrum = np.fft.rfft(frames * window[None, :], n=config.fft_size, axis=1)
    power = np.square(np.abs(spectrum)) / float(config.fft_size)
    mel_power = power @ _mel_filterbank(config).T
    mel_power /= np.maximum(
        mel_power.max(axis=1, keepdims=True), config.log_power_floor
    )
    log_mel = np.log(np.maximum(mel_power, config.log_power_floor))
    log_mel -= log_mel.mean(axis=1, keepdims=True)
    cepstra = log_mel @ _dct_matrix(config).T
    centers = (
        starts.astype(np.float64) + config.frame_length_samples / 2.0
    ) / interval.size
    temporal = np.stack(
        [
            np.asarray(
                [
                    np.interp(fraction, centers, cepstra[:, coefficient])
                    for coefficient in range(config.cepstral_coefficient_count)
                ],
                dtype=np.float64,
            )
            for fraction in config.temporal_sample_fractions
        ]
    )
    feature = temporal.reshape(-1)
    if not np.isfinite(feature).all():
        raise ValueError("spectral feature contains non-finite values")
    return {
        "version": SPECTRAL_CATEGORY_VERSION,
        "config": asdict(config),
        "interval_samples": [start_sample, end_sample_exclusive],
        "duration_ms": (
            (end_sample_exclusive - start_sample) / config.sample_rate_hz * 1_000.0
        ),
        "frame_count": int(starts.size),
        "feature_size": int(feature.size),
        "feature": feature.tolist(),
    }


def fit_robust_feature_scaler(
    features: Sequence[Sequence[float] | np.ndarray],
    *,
    scale_floor: float = DEFAULT_FEATURE_CONFIG.robust_scale_floor,
) -> dict[str, list[float] | float | int]:
    matrix = np.asarray(features, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] < 2 or matrix.shape[1] == 0:
        raise ValueError("at least two equal-length feature vectors are required")
    if (
        not np.isfinite(matrix).all()
        or not math.isfinite(scale_floor)
        or scale_floor <= 0
    ):
        raise ValueError("robust scaler inputs must be finite and positive")
    center = np.median(matrix, axis=0)
    mad = np.median(np.abs(matrix - center[None, :]), axis=0)
    scale = np.maximum(1.4826 * mad, scale_floor)
    return {
        "feature_size": int(matrix.shape[1]),
        "observation_count": int(matrix.shape[0]),
        "scale_floor": float(scale_floor),
        "center": center.tolist(),
        "scale": scale.tolist(),
    }


def apply_robust_feature_scaler(
    feature: Sequence[float] | np.ndarray,
    scaler: dict[str, Any],
) -> np.ndarray:
    size = int(scaler["feature_size"])
    values = _finite_vector(feature, expected_size=size)
    center = _finite_vector(scaler["center"], expected_size=size)
    scale = _finite_vector(scaler["scale"], expected_size=size)
    if np.any(scale <= 0):
        raise ValueError("robust feature scales must be positive")
    return (values - center) / scale


def classify_spectral_endpoint(
    *,
    source_anchor: Sequence[float] | np.ndarray,
    target_anchor: Sequence[float] | np.ndarray,
    neutral: Sequence[float] | np.ndarray,
    lens: Sequence[float] | np.ndarray,
    minimum_anchor_separation_rms: float = (MINIMUM_ANCHOR_SEPARATION_SCALED_RMS),
    minimum_direction_cosine: float = MINIMUM_DIRECTION_COSINE,
    minimum_directional_movement_fraction: float = (
        MINIMUM_DIRECTIONAL_MOVEMENT_FRACTION
    ),
    minimum_exact_movement_fraction: float = MINIMUM_EXACT_MOVEMENT_FRACTION,
) -> dict[str, Any]:
    source = _finite_vector(source_anchor)
    target = _finite_vector(target_anchor, expected_size=source.size)
    neutral_point = _finite_vector(neutral, expected_size=source.size)
    lens_point = _finite_vector(lens, expected_size=source.size)
    if not (
        0 < minimum_anchor_separation_rms
        and -1 <= minimum_direction_cosine <= 1
        and 0 < minimum_directional_movement_fraction <= minimum_exact_movement_fraction
    ):
        raise ValueError("invalid spectral endpoint thresholds")

    anchor_vector = target - source
    movement_vector = lens_point - neutral_point
    anchor_norm = float(np.linalg.norm(anchor_vector))
    movement_norm = float(np.linalg.norm(movement_vector))
    anchor_rms = math.sqrt(float(np.mean(np.square(anchor_vector))))
    movement_rms = math.sqrt(float(np.mean(np.square(movement_vector))))
    movement_fraction = movement_norm / anchor_norm if anchor_norm else 0.0
    direction_cosine = (
        float(np.dot(anchor_vector, movement_vector) / (anchor_norm * movement_norm))
        if anchor_norm and movement_norm
        else -1.0
    )

    def distance(left: np.ndarray, right: np.ndarray) -> float:
        return math.sqrt(float(np.mean(np.square(left - right))))

    neutral_source = distance(neutral_point, source)
    neutral_target = distance(neutral_point, target)
    lens_source = distance(lens_point, source)
    lens_target = distance(lens_point, target)
    gates = {
        "anchor_gate_pass": anchor_rms >= minimum_anchor_separation_rms,
        "direction_gate_pass": direction_cosine >= minimum_direction_cosine,
        "directional_movement_gate_pass": (
            movement_fraction >= minimum_directional_movement_fraction
        ),
        "exact_movement_gate_pass": (
            movement_fraction >= minimum_exact_movement_fraction
        ),
        "neutral_endpoint_gate_pass": neutral_source < neutral_target,
        "lens_endpoint_gate_pass": lens_target < lens_source,
        "target_gain_gate_pass": lens_target < neutral_target,
        "source_departure_gate_pass": lens_source > neutral_source,
    }
    directional = bool(
        gates["anchor_gate_pass"]
        and gates["direction_gate_pass"]
        and gates["directional_movement_gate_pass"]
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
        "anchor_separation_scaled_rms": anchor_rms,
        "minimum_anchor_separation_scaled_rms": minimum_anchor_separation_rms,
        "controlled_movement_scaled_rms": movement_rms,
        "controlled_movement_fraction_of_anchor": movement_fraction,
        "minimum_directional_movement_fraction": (
            minimum_directional_movement_fraction
        ),
        "minimum_exact_movement_fraction": minimum_exact_movement_fraction,
        "direction_cosine": direction_cosine,
        "minimum_direction_cosine": minimum_direction_cosine,
        "neutral_source_distance_scaled_rms": neutral_source,
        "neutral_target_distance_scaled_rms": neutral_target,
        "lens_source_distance_scaled_rms": lens_source,
        "lens_target_distance_scaled_rms": lens_target,
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


def aggregate_spectral_cell(
    *,
    natural_anchor_records: Sequence[dict[str, Any]],
    candidate_records: Sequence[dict[str, Any]],
    expected_occurrence_count: int = 4,
) -> dict[str, Any]:
    if (
        len(natural_anchor_records) != expected_occurrence_count
        or len(candidate_records) != expected_occurrence_count
    ):
        raise ValueError("spectral cell aggregation requires every frozen occurrence")
    natural_exact = sum(
        bool(record["exact_category_pass"]) for record in natural_anchor_records
    )
    natural_directional = sum(
        bool(record["directional_pass"]) for record in natural_anchor_records
    )
    natural_reversed = sum(
        float(record["direction_cosine"]) < 0.0 for record in natural_anchor_records
    )
    anchor_validation_pass = bool(
        natural_exact >= MINIMUM_HELDOUT_EXACT_ANCHOR_PAIRS
        and natural_reversed <= MAXIMUM_REVERSED_HELDOUT_ANCHOR_PAIRS
    )
    candidate_exact = sum(
        bool(record["exact_category_pass"]) for record in candidate_records
    )
    candidate_directional = sum(
        bool(record["directional_pass"]) for record in candidate_records
    )
    if not anchor_validation_pass:
        classification = "anchor_validation_fail"
    elif candidate_exact == expected_occurrence_count:
        classification = "exact_category_pass"
    elif candidate_directional == expected_occurrence_count:
        classification = "directional_only_pass"
    else:
        classification = "fail"
    return {
        "expected_occurrence_count": expected_occurrence_count,
        "natural_anchor_exact_count": natural_exact,
        "natural_anchor_directional_count": natural_directional,
        "natural_anchor_reversed_count": natural_reversed,
        "minimum_natural_anchor_exact_count": (MINIMUM_HELDOUT_EXACT_ANCHOR_PAIRS),
        "maximum_natural_anchor_reversed_count": (
            MAXIMUM_REVERSED_HELDOUT_ANCHOR_PAIRS
        ),
        "anchor_validation_pass": anchor_validation_pass,
        "candidate_exact_count": candidate_exact,
        "candidate_directional_count": candidate_directional,
        "classification": classification,
        "directional_pass": classification
        in ("exact_category_pass", "directional_only_pass"),
        "exact_category_pass": classification == "exact_category_pass",
    }
