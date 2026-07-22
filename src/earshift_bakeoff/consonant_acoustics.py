from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True)
class SampleInterval:
    start_sample: int
    end_sample_exclusive: int
    sample_rate_hz: int

    @property
    def start_s(self) -> float:
        return self.start_sample / self.sample_rate_hz

    @property
    def end_s(self) -> float:
        return self.end_sample_exclusive / self.sample_rate_hz

    def as_record(self) -> dict[str, int | float]:
        return {
            "start_sample": self.start_sample,
            "end_sample_exclusive": self.end_sample_exclusive,
            "start_s": self.start_s,
            "end_s": self.end_s,
        }


def decoder_column_interval(
    durations: Sequence[int],
    columns: Sequence[int],
    *,
    sample_count: int,
    sample_rate_hz: int,
) -> SampleInterval:
    """Map one or more model columns onto the decoder's exact sample grid."""

    values = tuple(int(value) for value in durations)
    selected = tuple(sorted(set(int(value) for value in columns)))
    if not values or min(values) < 1:
        raise ValueError("decoder durations must be positive")
    if not selected or selected[0] < 0 or selected[-1] >= len(values):
        raise ValueError("target columns are empty or outside the duration plan")
    if selected != tuple(range(selected[0], selected[-1] + 1)):
        raise ValueError("target columns must be contiguous")
    total_frames = sum(values)
    if sample_count <= 0 or sample_count % total_frames:
        raise ValueError("audio no longer has an integral decoder-frame sample grid")
    samples_per_frame = sample_count // total_frames
    start = sum(values[: selected[0]]) * samples_per_frame
    end = sum(values[: selected[-1] + 1]) * samples_per_frame
    if not 0 <= start < end <= sample_count:
        raise ValueError("derived target interval is invalid")
    return SampleInterval(start, end, sample_rate_hz)


def expanded_interval(
    interval: SampleInterval,
    *,
    context_ms: float,
    sample_count: int,
) -> SampleInterval:
    context = round(context_ms * interval.sample_rate_hz / 1000.0)
    return SampleInterval(
        max(0, interval.start_sample - context),
        min(sample_count, interval.end_sample_exclusive + context),
        interval.sample_rate_hz,
    )


def _frame_rms(values: np.ndarray, frame_samples: int) -> np.ndarray:
    if values.size < frame_samples:
        return np.asarray(
            [math.sqrt(float(np.mean(np.square(values))))], dtype=np.float64
        )
    count = values.size // frame_samples
    framed = values[: count * frame_samples].reshape(count, frame_samples)
    return np.sqrt(np.mean(np.square(framed), axis=1))


def _periodicity(values: np.ndarray, sample_rate_hz: int) -> float:
    centered = values - float(np.mean(values))
    energy = float(np.dot(centered, centered))
    if centered.size < 2 or energy <= 0.0:
        return 0.0
    lower = max(1, round(sample_rate_hz / 400.0))
    upper = min(centered.size - 1, round(sample_rate_hz / 70.0))
    if upper < lower:
        return 0.0
    scores = []
    for lag in range(lower, upper + 1):
        left = centered[:-lag]
        right = centered[lag:]
        denominator = math.sqrt(float(np.dot(left, left) * np.dot(right, right)))
        if denominator > 0.0:
            scores.append(float(np.dot(left, right) / denominator))
    return max(0.0, max(scores, default=0.0))


def consonant_acoustic_metrics(
    pcm: np.ndarray,
    interval: SampleInterval,
) -> dict[str, float | int | bool]:
    """Return instrument-like descriptors without assigning a phone category.

    These measures expose stop/frication, voicing, and spectral changes. They are
    deliberately descriptive: a universal phone recognizer and human QC remain
    separate instruments rather than being hidden inside a composite score.
    """

    source = np.asarray(pcm).reshape(-1)
    if source.size < interval.end_sample_exclusive:
        raise ValueError("measurement interval exceeds the PCM")
    values = source[
        interval.start_sample : interval.end_sample_exclusive
    ].astype(np.float64)
    if not values.size or not np.isfinite(values).all():
        raise ValueError("measurement interval is empty or nonfinite")
    if np.issubdtype(source.dtype, np.integer):
        values /= float(np.iinfo(source.dtype).max)
    centered = values - float(np.mean(values))
    windowed = centered * np.hanning(centered.size)
    spectrum = np.abs(np.fft.rfft(windowed))
    power = np.square(spectrum)
    frequencies = np.fft.rfftfreq(centered.size, 1.0 / interval.sample_rate_hz)
    total_power = float(power.sum())
    if total_power <= 0.0 or not np.isfinite(total_power):
        centroid = bandwidth = flatness = 0.0
        band_shares = [0.0, 0.0, 0.0, 0.0]
    else:
        centroid = float(np.sum(frequencies * power) / total_power)
        bandwidth = float(
            np.sqrt(np.sum(np.square(frequencies - centroid) * power) / total_power)
        )
        positive = spectrum[spectrum > 0.0]
        flatness = (
            float(np.exp(np.mean(np.log(positive))) / np.mean(positive))
            if positive.size
            else 0.0
        )
        edges = ((0, 500), (500, 2_000), (2_000, 5_000), (5_000, math.inf))
        band_shares = [
            float(
                power[(frequencies >= low) & (frequencies < high)].sum()
                / total_power
            )
            for low, high in edges
        ]
    rms = math.sqrt(float(np.mean(np.square(values))))
    peak = float(np.max(np.abs(values)))
    crossings = (
        float(np.mean(centered[:-1] * centered[1:] < 0.0))
        if centered.size > 1
        else 0.0
    )
    frame_rms = _frame_rms(values, max(1, round(interval.sample_rate_hz * 0.005)))
    maximum_frame_rms = float(frame_rms.max(initial=0.0))
    closure_threshold = max(1e-7, maximum_frame_rms * 0.20)
    closure_fraction = float(np.mean(frame_rms <= closure_threshold))
    derivative = np.diff(values)
    peak_derivative = float(np.max(np.abs(derivative), initial=0.0))
    return {
        "sample_count": int(values.size),
        "duration_ms": values.size * 1000.0 / interval.sample_rate_hz,
        "finite": True,
        "rms": rms,
        "peak": peak,
        "zero_crossing_rate": crossings,
        "periodicity": _periodicity(values, interval.sample_rate_hz),
        "spectral_centroid_hz": centroid,
        "spectral_bandwidth_hz": bandwidth,
        "spectral_flatness": flatness,
        "band_0_500_share": band_shares[0],
        "band_500_2000_share": band_shares[1],
        "band_2000_5000_share": band_shares[2],
        "band_5000_nyquist_share": band_shares[3],
        "five_ms_low_energy_fraction": closure_fraction,
        "peak_first_difference": peak_derivative,
    }


_DISTANCE_SCALES = {
    "log_rms": 1.0,
    "zero_crossing_rate": 0.10,
    "periodicity": 0.50,
    "spectral_centroid_hz": 3_000.0,
    "spectral_bandwidth_hz": 3_000.0,
    "spectral_flatness": 0.50,
    "band_0_500_share": 0.50,
    "band_500_2000_share": 0.50,
    "band_2000_5000_share": 0.50,
    "band_5000_nyquist_share": 0.50,
    "five_ms_low_energy_fraction": 0.50,
    "log_peak_first_difference": 1.0,
}


def descriptive_feature_vector(metrics: dict[str, Any]) -> dict[str, float]:
    floor = 1e-9
    values = {
        "log_rms": math.log(max(floor, float(metrics["rms"]))),
        "zero_crossing_rate": float(metrics["zero_crossing_rate"]),
        "periodicity": float(metrics["periodicity"]),
        "spectral_centroid_hz": float(metrics["spectral_centroid_hz"]),
        "spectral_bandwidth_hz": float(metrics["spectral_bandwidth_hz"]),
        "spectral_flatness": float(metrics["spectral_flatness"]),
        "band_0_500_share": float(metrics["band_0_500_share"]),
        "band_500_2000_share": float(metrics["band_500_2000_share"]),
        "band_2000_5000_share": float(metrics["band_2000_5000_share"]),
        "band_5000_nyquist_share": float(metrics["band_5000_nyquist_share"]),
        "five_ms_low_energy_fraction": float(metrics["five_ms_low_energy_fraction"]),
        "log_peak_first_difference": math.log(
            max(floor, float(metrics["peak_first_difference"]))
        ),
    }
    return {key: value / _DISTANCE_SCALES[key] for key, value in values.items()}


def descriptive_distance(left: dict[str, Any], right: dict[str, Any]) -> float:
    a = descriptive_feature_vector(left)
    b = descriptive_feature_vector(right)
    return math.sqrt(sum((a[key] - b[key]) ** 2 for key in a) / len(a))

