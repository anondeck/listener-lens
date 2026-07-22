from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


SPECTRAL_ENVELOPE_WARP_VERSION = "spectral-envelope-warp-v1"
DEFAULT_CONTEXT_SAMPLES = 480
DEFAULT_TAPER_SAMPLES = 240
DEFAULT_SMOOTHING_HZ = 120.0
DEFAULT_IDENTITY_START_HZ = 3_500.0
DEFAULT_IDENTITY_END_HZ = 4_500.0


@dataclass(frozen=True)
class FormantWarpSpec:
    start_sample: int
    end_sample_exclusive: int
    source_f1_hz: float
    source_f2_hz: float
    target_f1_hz: float
    target_f2_hz: float


@dataclass(frozen=True)
class SpectralEnvelopeWarpResult:
    pcm: np.ndarray
    weights: np.ndarray
    edit_windows: tuple[dict[str, int], ...]
    metrics: dict[str, Any]
    version: str = SPECTRAL_ENVELOPE_WARP_VERSION


def _next_power_of_two(value: int) -> int:
    if value <= 0:
        raise ValueError("FFT length must be positive")
    return 1 << (value - 1).bit_length()


def _validate_spec(spec: FormantWarpSpec, sample_count: int) -> None:
    formants = (
        spec.source_f1_hz,
        spec.source_f2_hz,
        spec.target_f1_hz,
        spec.target_f2_hz,
    )
    if (
        spec.start_sample < 0
        or spec.end_sample_exclusive <= spec.start_sample
        or spec.end_sample_exclusive > sample_count
        or not all(np.isfinite(value) and value > 0.0 for value in formants)
        or spec.source_f1_hz >= spec.source_f2_hz
        or spec.target_f1_hz >= spec.target_f2_hz
    ):
        raise ValueError("invalid formant-warp specification")


def _gaussian_kernel(sigma_bins: float) -> np.ndarray:
    if not np.isfinite(sigma_bins) or sigma_bins <= 0.0:
        raise ValueError("spectral smoothing must be finite and positive")
    radius = max(2, int(np.ceil(4.0 * sigma_bins)))
    positions = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * np.square(positions / sigma_bins))
    return kernel / np.sum(kernel)


def _identity_blend(
    frequencies: np.ndarray,
    *,
    identity_start_hz: float,
    identity_end_hz: float,
) -> np.ndarray:
    if not 0.0 < identity_start_hz < identity_end_hz:
        raise ValueError("identity-band limits are invalid")
    blend = np.ones_like(frequencies, dtype=np.float64)
    blend[frequencies >= identity_end_hz] = 0.0
    transition = (frequencies > identity_start_hz) & (frequencies < identity_end_hz)
    phase = (frequencies[transition] - identity_start_hz) / (
        identity_end_hz - identity_start_hz
    )
    blend[transition] = 0.5 * (1.0 + np.cos(np.pi * phase))
    return blend


def _warp_segment(
    segment: np.ndarray,
    *,
    sample_rate_hz: int,
    source_formants_hz: tuple[float, float],
    target_formants_hz: tuple[float, float],
    strength: float,
    smoothing_hz: float,
    identity_start_hz: float,
    identity_end_hz: float,
) -> np.ndarray:
    values = np.asarray(segment, dtype=np.float64).reshape(-1)
    if values.size < 16:
        raise ValueError("spectral-warp segment is too short")
    if not np.isfinite(strength) or strength < 0.0:
        raise ValueError("spectral-warp strength must be finite and nonnegative")
    if strength == 0.0:
        return values.copy()
    source_f1, source_f2 = source_formants_hz
    target_f1 = source_f1 + strength * (target_formants_hz[0] - source_f1)
    target_f2 = source_f2 + strength * (target_formants_hz[1] - source_f2)
    if not 60.0 < target_f1 < target_f2 < identity_end_hz:
        raise ValueError("strength produced a non-monotonic formant target")

    nfft = _next_power_of_two(max(2_048, values.size * 4))
    frequencies = np.fft.rfftfreq(nfft, d=1.0 / sample_rate_hz)
    spectrum = np.fft.rfft(values, n=nfft)
    magnitude = np.maximum(np.abs(spectrum), 1e-10)
    log_magnitude = np.log(magnitude)
    bin_width_hz = sample_rate_hz / nfft
    envelope = np.convolve(
        log_magnitude,
        _gaussian_kernel(smoothing_hz / bin_width_hz),
        mode="same",
    )

    nyquist = sample_rate_hz / 2.0
    output_knots = np.array(
        [0.0, target_f1, target_f2, identity_end_hz, nyquist],
        dtype=np.float64,
    )
    input_knots = np.array(
        [0.0, source_f1, source_f2, identity_end_hz, nyquist],
        dtype=np.float64,
    )
    inverse_frequencies = np.interp(frequencies, output_knots, input_knots)
    shifted_envelope = np.interp(inverse_frequencies, frequencies, envelope)
    blend = _identity_blend(
        frequencies,
        identity_start_hz=identity_start_hz,
        identity_end_hz=identity_end_hz,
    )
    gain = np.exp((shifted_envelope - envelope) * blend)
    warped = np.fft.irfft(spectrum * gain, n=nfft)[: values.size]

    source_rms = float(np.sqrt(np.mean(np.square(values))))
    warped_rms = float(np.sqrt(np.mean(np.square(warped))))
    if source_rms > 0.0 and warped_rms > 0.0:
        warped *= source_rms / warped_rms
    return warped


def _taper_weights(length: int, taper_samples: int) -> np.ndarray:
    if length <= 0 or taper_samples <= 0 or taper_samples * 2 > length:
        raise ValueError("spectral-warp taper does not fit its edit window")
    weights = np.ones(length, dtype=np.float64)
    phase = np.linspace(0.0, np.pi, taper_samples, endpoint=False)
    edge = 0.5 * (1.0 - np.cos(phase))
    weights[:taper_samples] = edge
    weights[-taper_samples:] = edge[::-1]
    return weights


def spectral_envelope_warp(
    pcm: np.ndarray,
    specs: Sequence[FormantWarpSpec],
    *,
    sample_rate_hz: int,
    strength: float,
    context_samples: int = DEFAULT_CONTEXT_SAMPLES,
    taper_samples: int = DEFAULT_TAPER_SAMPLES,
    smoothing_hz: float = DEFAULT_SMOOTHING_HZ,
    identity_start_hz: float = DEFAULT_IDENTITY_START_HZ,
    identity_end_hz: float = DEFAULT_IDENTITY_END_HZ,
) -> SpectralEnvelopeWarpResult:
    """Warp only the smooth spectral envelope around aligned vowel intervals.

    Harmonic-bin locations and phase remain unchanged, so the operation does not
    resynthesize F0 or timing. Frequencies above the identity band are unchanged
    in the transfer function, and samples outside tapered edit windows remain
    bit-identical after PCM conversion.
    """

    source = np.asarray(pcm, dtype=np.int16).reshape(-1)
    if sample_rate_hz <= 0 or context_samples < 0:
        raise ValueError("invalid spectral-warp sample configuration")
    ordered = tuple(sorted(specs, key=lambda spec: spec.start_sample))
    for spec in ordered:
        _validate_spec(spec, source.size)
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
        raise ValueError("spectral-warp edit windows overlap")
    if strength == 0.0 or not ordered:
        return SpectralEnvelopeWarpResult(
            pcm=source.copy(),
            weights=np.zeros(source.size, dtype=np.float64),
            edit_windows=windows,
            metrics={
                "identity": True,
                "outside_windows_bit_exact": True,
                "finite": True,
                "clipped_sample_count": 0,
                "rms_db_change_by_window": [0.0 for _ in windows],
            },
        )

    output = source.astype(np.float64) / 32768.0
    source_float = output.copy()
    global_weights = np.zeros(source.size, dtype=np.float64)
    rms_changes: list[float] = []
    for spec, window in zip(ordered, windows, strict=True):
        start = window["start_sample"]
        end = window["end_sample_exclusive"]
        before = source_float[start:end]
        warped = _warp_segment(
            before,
            sample_rate_hz=sample_rate_hz,
            source_formants_hz=(spec.source_f1_hz, spec.source_f2_hz),
            target_formants_hz=(spec.target_f1_hz, spec.target_f2_hz),
            strength=strength,
            smoothing_hz=smoothing_hz,
            identity_start_hz=identity_start_hz,
            identity_end_hz=identity_end_hz,
        )
        weights = _taper_weights(end - start, min(taper_samples, (end - start) // 2))
        output[start:end] = before + weights * (warped - before)
        global_weights[start:end] = weights
        before_rms = float(np.sqrt(np.mean(np.square(before))))
        after_rms = float(np.sqrt(np.mean(np.square(output[start:end]))))
        rms_changes.append(
            20.0 * np.log10(max(after_rms, 1e-12) / max(before_rms, 1e-12))
        )

    finite = bool(np.isfinite(output).all())
    rounded = np.rint(output * 32768.0)
    clipped_count = int(np.sum((rounded < -32768.0) | (rounded > 32767.0)))
    result_pcm = np.clip(rounded, -32768.0, 32767.0).astype(np.int16)
    outside = global_weights == 0.0
    return SpectralEnvelopeWarpResult(
        pcm=result_pcm,
        weights=global_weights,
        edit_windows=windows,
        metrics={
            "identity": False,
            "outside_windows_bit_exact": bool(
                np.array_equal(result_pcm[outside], source[outside])
            ),
            "finite": finite,
            "clipped_sample_count": clipped_count,
            "rms_db_change_by_window": rms_changes,
        },
    )
