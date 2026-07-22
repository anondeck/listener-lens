from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence

import numpy as np
import pyworld


WORLD_FORMANT_EDITOR_VERSION = "world-formant-editor-v1"
FRAME_PERIOD_MS = 5.0
FFT_SIZE = 1024
DEFAULT_SHIFT_TAPER_S = 0.01
DEFAULT_SPLICE_CONTEXT_SAMPLES = 480
DEFAULT_SPLICE_TAPER_SAMPLES = 240


@dataclass(frozen=True)
class WorldFormantSpec:
    start_sample: int
    end_sample_exclusive: int
    source_f1_hz: float
    source_f2_hz: float
    source_f3_hz: float
    target_f1_hz: float
    target_f2_hz: float


@dataclass(frozen=True)
class WorldFormantEditResult:
    identity_pcm: np.ndarray
    lens_pcm: np.ndarray
    edit_windows: tuple[dict[str, int], ...]
    metrics: dict[str, Any]
    version: str = WORLD_FORMANT_EDITOR_VERSION


def _validate_specs(
    specs: Sequence[WorldFormantSpec], sample_count: int, sample_rate_hz: int
) -> tuple[WorldFormantSpec, ...]:
    ordered = tuple(sorted(specs, key=lambda item: item.start_sample))
    nyquist = sample_rate_hz / 2.0
    for spec in ordered:
        values = (
            spec.source_f1_hz,
            spec.source_f2_hz,
            spec.source_f3_hz,
            spec.target_f1_hz,
            spec.target_f2_hz,
        )
        if (
            spec.start_sample < 0
            or spec.end_sample_exclusive <= spec.start_sample
            or spec.end_sample_exclusive > sample_count
            or not all(math.isfinite(value) and value > 0.0 for value in values)
            or not spec.source_f1_hz < spec.source_f2_hz < spec.source_f3_hz < nyquist
            or not spec.target_f1_hz < spec.target_f2_hz < spec.source_f3_hz
        ):
            raise ValueError("invalid WORLD formant specification")
    return ordered


def _shift_weight(
    sample: float, start: int, end: int, taper_samples: int
) -> float:
    if sample < start or sample >= end:
        return 0.0
    return max(
        0.0,
        min(
            1.0,
            (sample - start) / taper_samples,
            (end - sample) / taper_samples,
        ),
    )


def warp_world_spectral_envelope(
    spectral_envelope: np.ndarray,
    time_axis: np.ndarray,
    *,
    sample_rate_hz: int,
    specs: Sequence[WorldFormantSpec],
    taper_s: float = DEFAULT_SHIFT_TAPER_S,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Warp WORLD's smooth source-filter envelope; leave excitation untouched."""

    spectral = np.asarray(spectral_envelope, dtype=np.float64)
    times = np.asarray(time_axis, dtype=np.float64)
    if (
        spectral.ndim != 2
        or times.ndim != 1
        or spectral.shape[0] != times.size
        or spectral.shape[1] < 3
        or sample_rate_hz <= 0
        or not math.isfinite(taper_s)
        or taper_s <= 0.0
    ):
        raise ValueError("invalid WORLD spectral-envelope inputs")
    ordered = _validate_specs(specs, math.ceil((times[-1] + 0.1) * sample_rate_hz), sample_rate_hz)
    frequencies = np.linspace(0.0, sample_rate_hz / 2.0, spectral.shape[1])
    output = spectral.copy()
    taper_samples = max(1, round(taper_s * sample_rate_hz))
    affected = 0
    minimum_spacing = math.inf
    for frame_index, time_s in enumerate(times):
        sample = float(time_s) * sample_rate_hz
        active = [
            spec
            for spec in ordered
            if spec.start_sample <= sample < spec.end_sample_exclusive
        ]
        if len(active) > 1:
            raise ValueError("WORLD formant specifications overlap")
        if not active:
            continue
        spec = active[0]
        weight = _shift_weight(
            sample,
            spec.start_sample,
            spec.end_sample_exclusive,
            taper_samples,
        )
        if weight <= 0.0:
            continue
        target_f1 = spec.source_f1_hz + weight * (
            spec.target_f1_hz - spec.source_f1_hz
        )
        target_f2 = spec.source_f2_hz + weight * (
            spec.target_f2_hz - spec.source_f2_hz
        )
        source_knots = np.asarray(
            [0.0, spec.source_f1_hz, spec.source_f2_hz, spec.source_f3_hz, sample_rate_hz / 2.0]
        )
        target_knots = np.asarray(
            [0.0, target_f1, target_f2, spec.source_f3_hz, sample_rate_hz / 2.0]
        )
        spacing = np.diff(target_knots)
        if np.any(spacing <= 0.0):
            raise ValueError("WORLD destination formants are non-monotonic")
        minimum_spacing = min(minimum_spacing, float(np.min(spacing)))
        source_coordinates = np.interp(frequencies, target_knots, source_knots)
        log_power = np.log(np.maximum(spectral[frame_index], 1e-30))
        output[frame_index] = np.exp(
            np.interp(source_coordinates, frequencies, log_power)
        )
        affected += 1
    return output, {
        "affected_frame_count": affected,
        "minimum_destination_knot_spacing_hz": (
            minimum_spacing if affected else None
        ),
    }


def _raised_cosine(length: int, taper_samples: int) -> np.ndarray:
    if length <= 0 or taper_samples <= 0 or taper_samples * 2 > length:
        raise ValueError("invalid WORLD splice taper")
    weights = np.ones(length, dtype=np.float64)
    phase = np.linspace(0.0, np.pi, taper_samples, endpoint=False)
    edge = 0.5 * (1.0 - np.cos(phase))
    weights[:taper_samples] = edge
    weights[-taper_samples:] = edge[::-1]
    return weights


def _rms_gain(reference: np.ndarray, candidate: np.ndarray) -> float:
    ref_rms = math.sqrt(float(np.mean(np.square(reference))))
    candidate_rms = math.sqrt(float(np.mean(np.square(candidate))))
    if not math.isfinite(candidate_rms) or candidate_rms <= 0.0:
        raise ValueError("WORLD synthesis has no finite energy")
    return ref_rms / candidate_rms


def world_formant_edit(
    pcm: np.ndarray,
    specs: Sequence[WorldFormantSpec],
    *,
    sample_rate_hz: int,
    frame_period_ms: float = FRAME_PERIOD_MS,
    fft_size: int = FFT_SIZE,
    context_samples: int = DEFAULT_SPLICE_CONTEXT_SAMPLES,
    splice_taper_samples: int = DEFAULT_SPLICE_TAPER_SAMPLES,
) -> WorldFormantEditResult:
    """Create matched identity/lens PCM using one shared WORLD analysis state."""

    source = np.asarray(pcm, dtype=np.int16).reshape(-1)
    ordered = _validate_specs(specs, source.size, sample_rate_hz)
    windows = tuple(
        {
            "start_sample": max(0, spec.start_sample - context_samples),
            "end_sample_exclusive": min(
                source.size, spec.end_sample_exclusive + context_samples
            ),
        }
        for spec in ordered
    )
    if any(
        left["end_sample_exclusive"] > right["start_sample"]
        for left, right in zip(windows, windows[1:])
    ):
        raise ValueError("WORLD edit windows overlap")
    source_float = source.astype(np.float64) / 32768.0
    f0_raw, time_axis = pyworld.dio(
        source_float, sample_rate_hz, frame_period=frame_period_ms
    )
    f0 = pyworld.stonemask(source_float, f0_raw, time_axis, sample_rate_hz)
    spectral = pyworld.cheaptrick(
        source_float, f0, time_axis, sample_rate_hz, fft_size=fft_size
    )
    aperiodicity = pyworld.d4c(source_float, f0, time_axis, sample_rate_hz)
    warped, warp_metrics = warp_world_spectral_envelope(
        spectral,
        time_axis,
        sample_rate_hz=sample_rate_hz,
        specs=ordered,
    )
    identity_float = pyworld.synthesize(
        f0, spectral, aperiodicity, sample_rate_hz, frame_period=frame_period_ms
    )
    lens_float = pyworld.synthesize(
        f0, warped, aperiodicity, sample_rate_hz, frame_period=frame_period_ms
    )
    if identity_float.size < source.size or lens_float.size < source.size:
        raise RuntimeError("WORLD synthesis is shorter than its source")
    identity_synth = identity_float[: source.size] * 32768.0
    lens_synth = lens_float[: source.size] * 32768.0
    identity_output = source.astype(np.float64)
    lens_output = source.astype(np.float64)
    gains = []
    global_weights = np.zeros(source.size, dtype=np.float64)
    for window in windows:
        start = window["start_sample"]
        end = window["end_sample_exclusive"]
        weights = _raised_cosine(
            end - start, min(splice_taper_samples, (end - start) // 2)
        )
        reference = source[start:end].astype(np.float64)
        identity_component = identity_synth[start:end]
        lens_component = lens_synth[start:end]
        gain = _rms_gain(reference, identity_component)
        identity_output[start:end] = reference + weights * (
            gain * identity_component - reference
        )
        lens_output[start:end] = reference + weights * (
            gain * lens_component - reference
        )
        global_weights[start:end] = weights
        gains.append(gain)
    finite = bool(np.isfinite(identity_output).all() and np.isfinite(lens_output).all())
    identity_rounded = np.rint(identity_output)
    lens_rounded = np.rint(lens_output)
    clipped = int(
        np.count_nonzero((identity_rounded < -32768) | (identity_rounded > 32767))
        + np.count_nonzero((lens_rounded < -32768) | (lens_rounded > 32767))
    )
    identity_pcm = np.clip(identity_rounded, -32768, 32767).astype(np.int16)
    lens_pcm = np.clip(lens_rounded, -32768, 32767).astype(np.int16)
    outside = global_weights == 0.0
    return WorldFormantEditResult(
        identity_pcm=identity_pcm,
        lens_pcm=lens_pcm,
        edit_windows=windows,
        metrics={
            "finite": finite,
            "clipped_sample_count": clipped,
            "outside_windows_bit_exact": bool(
                np.array_equal(identity_pcm[outside], source[outside])
                and np.array_equal(lens_pcm[outside], source[outside])
            ),
            "shared_analysis_state": True,
            "identity_and_lens_gain_by_window": gains,
            "frame_count": int(time_axis.size),
            "voiced_frame_count": int(np.count_nonzero(f0 > 0.0)),
            "fft_size": fft_size,
            **warp_metrics,
        },
    )
