from __future__ import annotations

import math
import re
import unicodedata
import wave
from array import array
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path
from statistics import median
from typing import Any, Sequence


@dataclass(frozen=True)
class TranscriptCheck:
    exact_token_match: bool
    expected_is_contiguous: bool
    token_similarity: float
    expected_token_count: int
    actual_token_count: int
    extra_token_count: int
    missing_token_count: int


@dataclass(frozen=True)
class PauseInterval:
    start_s: float
    end_s: float
    duration_s: float
    start_fraction: float
    end_fraction: float


@dataclass(frozen=True)
class AudioTiming:
    duration_s: float
    sample_rate_hz: int
    decoded_sample_count: int
    clipped_fraction: float
    utterance_duration_s: float
    estimated_syllables_per_second: float | None
    interior_pause_count: int
    interior_pause_s: float
    interior_pauses: tuple[PauseInterval, ...]

    def to_result_fields(self) -> dict[str, Any]:
        payload = asdict(self)
        pauses = payload.pop("interior_pauses")
        payload["interior_pause_positions_json"] = __import__("json").dumps(
            pauses, ensure_ascii=False, separators=(",", ":")
        )
        return payload


@dataclass(frozen=True)
class ProsodyFingerprint:
    version: str
    bin_count: int
    frame_count: int
    energy_contour_db: tuple[float, ...]
    pitch_contour_semitones: tuple[float | None, ...]
    median_f0_hz: float
    voiced_fraction: float
    energy_span_db: float


def canonical_tokens(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    normalized = normalized.replace("’", "'").replace("‘", "'")
    return re.findall(r"[^\W_]+(?:'[^\W_]+)*", normalized, flags=re.UNICODE)


def _is_contiguous(needle: Sequence[str], haystack: Sequence[str]) -> bool:
    if not needle or len(needle) > len(haystack):
        return False
    width = len(needle)
    return any(
        list(haystack[index : index + width]) == list(needle)
        for index in range(len(haystack) - width + 1)
    )


def check_transcript(expected: str, actual: str) -> TranscriptCheck:
    expected_tokens = canonical_tokens(expected)
    actual_tokens = canonical_tokens(actual)
    matcher = SequenceMatcher(a=expected_tokens, b=actual_tokens, autojunk=False)
    matched = sum(block.size for block in matcher.get_matching_blocks())
    return TranscriptCheck(
        exact_token_match=actual_tokens == expected_tokens,
        expected_is_contiguous=_is_contiguous(expected_tokens, actual_tokens),
        token_similarity=round(matcher.ratio(), 6),
        expected_token_count=len(expected_tokens),
        actual_token_count=len(actual_tokens),
        extra_token_count=max(0, len(actual_tokens) - matched),
        missing_token_count=max(0, len(expected_tokens) - matched),
    )


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = fraction * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _pearson_autocorrelation(frame: Sequence[float], lag: int) -> float:
    left = frame[:-lag]
    right = frame[lag:]
    numerator = sum(a * b for a, b in zip(left, right))
    left_energy = sum(value * value for value in left)
    right_energy = sum(value * value for value in right)
    denominator = math.sqrt(left_energy * right_energy)
    return numerator / denominator if denominator > 1e-12 else 0.0


def analyze_prosody_fingerprint(
    path: Path,
    *,
    bin_count: int = 32,
    silence_dbfs: float = -38.0,
    maximum_analysis_rate_hz: int = 8_000,
    frame_s: float = 0.03,
    hop_s: float = 0.02,
    minimum_f0_hz: float = 60.0,
    maximum_f0_hz: float = 400.0,
    minimum_pitch_correlation: float = 0.30,
) -> ProsodyFingerprint:
    """Return a coarse engineering fingerprint; this is not phonetic analysis."""
    if bin_count < 8 or bin_count > 128:
        raise ValueError("Prosody fingerprint bin count is out of bounds")
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        raw = wav.readframes(wav.getnframes())
    if channels != 1 or sample_width != 2 or sample_rate <= 0:
        raise ValueError("Prosody analysis requires mono 16-bit PCM WAV audio")

    pcm = array("h")
    pcm.frombytes(raw)
    if not pcm:
        raise ValueError("Prosody analysis requires non-empty audio")

    activity_window = max(1, round(sample_rate * 0.02))
    active_windows: list[int] = []
    for window_index, start in enumerate(range(0, len(pcm), activity_window)):
        window = pcm[start : start + activity_window]
        if not window:
            continue
        rms = math.sqrt(sum(value * value for value in window) / len(window))
        dbfs = 20 * math.log10(max(rms / 32768.0, 1e-9))
        if dbfs >= silence_dbfs:
            active_windows.append(window_index)
    if not active_windows:
        raise ValueError("Prosody analysis found no active speech")
    first_sample = active_windows[0] * activity_window
    last_sample = min(len(pcm), (active_windows[-1] + 1) * activity_window)

    downsample_step = max(1, math.ceil(sample_rate / maximum_analysis_rate_hz))
    analysis_rate = sample_rate / downsample_step
    samples = [
        value / 32768.0
        for value in pcm[first_sample:last_sample:downsample_step]
    ]
    frame_length = max(8, round(analysis_rate * frame_s))
    hop_length = max(1, round(analysis_rate * hop_s))
    minimum_lag = max(1, math.floor(analysis_rate / maximum_f0_hz))
    maximum_lag = max(minimum_lag + 1, math.ceil(analysis_rate / minimum_f0_hz))

    energies: list[float] = []
    pitches: list[float | None] = []
    frame_starts = range(0, max(1, len(samples) - frame_length + 1), hop_length)
    for start in frame_starts:
        frame = samples[start : start + frame_length]
        if len(frame) < frame_length:
            frame = [*frame, *([0.0] * (frame_length - len(frame)))]
        rms = math.sqrt(sum(value * value for value in frame) / len(frame))
        dbfs = 20 * math.log10(max(rms, 1e-9))
        energies.append(dbfs)
        if dbfs < silence_dbfs:
            pitches.append(None)
            continue
        frame_mean = sum(frame) / len(frame)
        centered = [value - frame_mean for value in frame]
        upper_lag = min(maximum_lag, len(centered) - 2)
        correlations: list[tuple[int, float]] = []
        for lag in range(minimum_lag, upper_lag + 1):
            correlations.append((lag, _pearson_autocorrelation(centered, lag)))
        best_lag, best_correlation = max(
            correlations, key=lambda item: item[1], default=(0, -1.0)
        )
        peak_threshold = max(
            minimum_pitch_correlation, best_correlation * 0.85
        )
        for item_index in range(1, len(correlations) - 1):
            lag, correlation = correlations[item_index]
            if (
                correlation >= peak_threshold
                and correlation >= correlations[item_index - 1][1]
                and correlation >= correlations[item_index + 1][1]
            ):
                best_lag, best_correlation = lag, correlation
                break
        pitches.append(
            analysis_rate / best_lag
            if best_lag and best_correlation >= minimum_pitch_correlation
            else None
        )

    active_energies = [value for value in energies if value >= silence_dbfs]
    energy_center = median(active_energies or energies)
    valid_pitches = [value for value in pitches if value is not None]
    pitch_center = median(valid_pitches) if valid_pitches else 0.0
    energy_bins: list[list[float]] = [[] for _ in range(bin_count)]
    pitch_bins: list[list[float]] = [[] for _ in range(bin_count)]
    for index, energy in enumerate(energies):
        bin_index = min(bin_count - 1, math.floor(index * bin_count / len(energies)))
        energy_bins[bin_index].append(
            max(-30.0, min(12.0, energy - energy_center))
        )
        pitch = pitches[index]
        if pitch is not None and pitch_center > 0:
            pitch_bins[bin_index].append(12 * math.log2(pitch / pitch_center))

    energy_contour = tuple(
        round(sum(values) / len(values), 4) if values else -30.0
        for values in energy_bins
    )
    pitch_contour = tuple(
        round(median(values), 4) if values else None for values in pitch_bins
    )
    return ProsodyFingerprint(
        version="prosody-fingerprint-v1",
        bin_count=bin_count,
        frame_count=len(energies),
        energy_contour_db=energy_contour,
        pitch_contour_semitones=pitch_contour,
        median_f0_hz=round(pitch_center, 3),
        voiced_fraction=round(len(valid_pitches) / len(pitches), 6),
        energy_span_db=round(
            _percentile(active_energies, 0.90)
            - _percentile(active_energies, 0.10),
            4,
        ),
    )


def analyze_audio_timing(
    path: Path,
    *,
    intended_syllables: int | None,
    silence_dbfs: float = -38.0,
    minimum_pause_s: float = 0.18,
) -> AudioTiming:
    """Measure broad timing/integrity properties without claiming natural prosody."""
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        raw = wav.readframes(wav.getnframes())
    if channels != 1 or sample_width != 2 or sample_rate <= 0:
        raise ValueError("Timing analysis requires mono 16-bit PCM WAV audio")

    samples = array("h")
    samples.frombytes(raw)
    decoded_sample_count = len(samples)
    duration_s = decoded_sample_count / sample_rate
    clipped_fraction = (
        sum(abs(sample) >= 32767 for sample in samples) / decoded_sample_count
        if decoded_sample_count
        else 1.0
    )
    window_samples = max(1, round(sample_rate * 0.02))
    silent_windows: list[bool] = []
    for start in range(0, len(samples), window_samples):
        window = samples[start : start + window_samples]
        if not window:
            continue
        rms = math.sqrt(sum(value * value for value in window) / len(window))
        dbfs = 20 * math.log10(max(rms / 32768.0, 1e-9))
        silent_windows.append(dbfs < silence_dbfs)

    voiced = [index for index, is_silent in enumerate(silent_windows) if not is_silent]
    if not voiced:
        return AudioTiming(
            duration_s=round(duration_s, 3), sample_rate_hz=sample_rate,
            decoded_sample_count=decoded_sample_count,
            clipped_fraction=round(clipped_fraction, 6), utterance_duration_s=0.0,
            estimated_syllables_per_second=None, interior_pause_count=0,
            interior_pause_s=0.0, interior_pauses=(),
        )
    first_voiced, last_voiced = voiced[0], voiced[-1]
    first_voiced_sample = first_voiced * window_samples
    last_voiced_sample = min(decoded_sample_count, (last_voiced + 1) * window_samples)
    utterance_sample_count = last_voiced_sample - first_voiced_sample
    utterance_duration_s = utterance_sample_count / sample_rate
    minimum_windows = max(1, math.ceil(minimum_pause_s * sample_rate / window_samples))
    pauses: list[PauseInterval] = []
    index = first_voiced + 1
    while index < last_voiced:
        if not silent_windows[index]:
            index += 1
            continue
        end = index
        while end <= last_voiced and silent_windows[end]:
            end += 1
        width = end - index
        if width >= minimum_windows:
            pause_start_sample = index * window_samples
            pause_end_sample = min(decoded_sample_count, end * window_samples)
            pause_sample_count = pause_end_sample - pause_start_sample
            pauses.append(PauseInterval(
                start_s=round(pause_start_sample / sample_rate, 3),
                end_s=round(pause_end_sample / sample_rate, 3),
                duration_s=round(pause_sample_count / sample_rate, 3),
                start_fraction=round((pause_start_sample - first_voiced_sample) / utterance_sample_count, 6),
                end_fraction=round((pause_end_sample - first_voiced_sample) / utterance_sample_count, 6),
            ))
        index = end
    rate = intended_syllables / utterance_duration_s if intended_syllables and utterance_duration_s > 0 else None
    return AudioTiming(
        duration_s=round(duration_s, 3), sample_rate_hz=sample_rate,
        decoded_sample_count=decoded_sample_count,
        clipped_fraction=round(clipped_fraction, 6),
        utterance_duration_s=round(utterance_duration_s, 3),
        estimated_syllables_per_second=round(rate, 3) if rate is not None else None,
        interior_pause_count=len(pauses),
        interior_pause_s=round(sum(pause.duration_s for pause in pauses), 3),
        interior_pauses=tuple(pauses),
    )
