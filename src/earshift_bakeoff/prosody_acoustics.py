from __future__ import annotations

import csv
import math
import subprocess
from pathlib import Path
from statistics import median
from typing import Any, Sequence

from .kokoro_synthesis import SAMPLE_RATE_HZ
from .prosody_component import ProsodyComponentRender


def _number(value: str | None) -> float | None:
    try:
        result = float(value or "")
    except ValueError:
        return None
    return result if math.isfinite(result) else None


def run_praat_probe(
    wav_path: Path,
    output_path: Path,
    *,
    praat_path: Path,
    probe_path: Path,
) -> list[dict[str, float | None]]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        [str(praat_path), "--run", str(probe_path), str(wav_path), str(output_path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr or completed.stdout)
    with output_path.open(encoding="utf-8", newline="") as handle:
        source = list(csv.DictReader(handle, delimiter="\t"))
    rows = [
        {key: _number(row.get(key)) for key in ("time_s", "pitch_hz", "rms")}
        for row in source
    ]
    if not rows or any(row["time_s"] is None for row in rows):
        raise RuntimeError("Praat prosody probe returned no valid time frames")
    return rows


def interval_summary(
    frames: Sequence[dict[str, float | None]],
    start_s: float,
    end_s: float,
    *,
    retain_fraction: float = 0.8,
) -> dict[str, Any]:
    if not 0 < retain_fraction <= 1 or not 0 <= start_s < end_s:
        raise ValueError("invalid prosody interval")
    trim = (1.0 - retain_fraction) / 2.0
    retained_start = start_s + (end_s - start_s) * trim
    retained_end = end_s - (end_s - start_s) * trim
    selected = [
        row
        for row in frames
        if row["time_s"] is not None
        and retained_start <= float(row["time_s"]) <= retained_end
    ]

    def values(key: str) -> list[float]:
        return [float(row[key]) for row in selected if row[key] is not None]

    pitch = values("pitch_hz")
    rms = values("rms")
    return {
        "interval_s": [start_s, end_s],
        "retained_interval_s": [retained_start, retained_end],
        "decoder_aligned_duration_ms": (end_s - start_s) * 1000,
        "measurement_frame_count": len(selected),
        "voiced_frame_count": len(pitch),
        "voiced_frame_fraction": len(pitch) / len(selected) if selected else 0.0,
        "median_pitch_hz": median(pitch) if pitch else None,
        "median_rms": median(rms) if rms else None,
    }


def _sample_interval(
    column: int, durations: Sequence[int], sample_count: int
) -> tuple[float, float]:
    return _columns_sample_interval((column,), durations, sample_count)


def _columns_sample_interval(
    columns: Sequence[int], durations: Sequence[int], sample_count: int
) -> tuple[float, float]:
    selected = tuple(int(column) for column in columns)
    if not selected or selected != tuple(range(selected[0], selected[-1] + 1)):
        raise RuntimeError("prosody measurement columns must be contiguous")
    total_frames = sum(int(value) for value in durations)
    if total_frames <= 0 or sample_count % total_frames:
        raise RuntimeError("prosody PCM does not map to integral decoder frames")
    samples_per_frame = sample_count // total_frames
    start = sum(int(value) for value in durations[: selected[0]]) * samples_per_frame
    end = sum(int(value) for value in durations[: selected[-1] + 1]) * samples_per_frame
    if not 0 <= start < end <= sample_count:
        raise RuntimeError("prosody column has no valid sample interval")
    return start / SAMPLE_RATE_HZ, end / SAMPLE_RATE_HZ


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator == 0:
        return None
    return numerator / denominator


def measure_stress_component(
    render: ProsodyComponentRender,
    neutral_frames: Sequence[dict[str, float | None]],
    lens_frames: Sequence[dict[str, float | None]],
    *,
    minimum_frames: int,
    minimum_duration_delta_ms: float,
    minimum_promoted_rms_ratio: float,
    maximum_demoted_rms_ratio: float,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for intervention in render.stress_interventions:
        roles: dict[str, Any] = {}
        for role, column_key in (
            ("promoted", "promoted_vowel_column"),
            ("demoted", "demoted_vowel_column"),
        ):
            column = int(intervention[column_key])
            neutral_start, neutral_end = _sample_interval(
                column, render.neutral_durations, render.neutral_pcm.size
            )
            lens_start, lens_end = _sample_interval(
                column, render.lens_durations, render.lens_pcm.size
            )
            neutral = interval_summary(
                neutral_frames, neutral_start, neutral_end, retain_fraction=0.8
            )
            lens = interval_summary(
                lens_frames, lens_start, lens_end, retain_fraction=0.8
            )
            duration_delta = (
                lens["decoder_aligned_duration_ms"]
                - neutral["decoder_aligned_duration_ms"]
            )
            rms_ratio = _ratio(lens["median_rms"], neutral["median_rms"])
            enough_frames = bool(
                neutral["measurement_frame_count"] >= minimum_frames
                and lens["measurement_frame_count"] >= minimum_frames
            )
            if role == "promoted":
                duration_pass = duration_delta >= minimum_duration_delta_ms
                rms_pass = bool(
                    rms_ratio is not None and rms_ratio >= minimum_promoted_rms_ratio
                )
            else:
                duration_pass = duration_delta <= -minimum_duration_delta_ms
                rms_pass = bool(
                    rms_ratio is not None and rms_ratio <= maximum_demoted_rms_ratio
                )
            roles[role] = {
                "neutral": neutral,
                "lens": lens,
                "duration_delta_ms": duration_delta,
                "rms_ratio": rms_ratio,
                "enough_frames": enough_frames,
                "duration_direction_pass": duration_pass,
                "rms_direction_pass": rms_pass,
                "gate_pass": bool(enough_frames and duration_pass and rms_pass),
            }
        row_pass = all(value["gate_pass"] for value in roles.values())
        rows.append(
            {
                "intervention": intervention,
                "roles": roles,
                "gate_pass": row_pass,
            }
        )
    gate_pass = bool(rows and all(row["gate_pass"] for row in rows))
    return {
        "measurement": (
            "decoder-aligned duration plus standalone-Praat median RMS over the "
            "middle 80% of promoted and demoted vowels"
        ),
        "occurrences": rows,
        "gate_pass": gate_pass,
        "status": "pass" if gate_pass else "fail",
    }


def measure_stress_unit_component(
    render: ProsodyComponentRender,
    neutral_frames: Sequence[dict[str, float | None]],
    lens_frames: Sequence[dict[str, float | None]],
    *,
    minimum_frames: int,
    minimum_duration_delta_ms: float,
    minimum_promoted_rms_ratio: float,
    maximum_demoted_rms_ratio: float,
) -> dict[str, Any]:
    """Measure duration over marker+vowel units and energy over the vowel core."""

    rows: list[dict[str, Any]] = []
    for intervention in render.stress_interventions:
        roles: dict[str, Any] = {}
        for role, marker_key, vowel_key in (
            ("promoted", "promoted_marker_column", "promoted_vowel_column"),
            ("demoted", "demoted_marker_column", "demoted_vowel_column"),
        ):
            marker = int(intervention[marker_key])
            vowel = int(intervention[vowel_key])
            neutral_unit_start, neutral_unit_end = _columns_sample_interval(
                tuple(range(marker, vowel + 1)),
                render.neutral_durations,
                render.neutral_pcm.size,
            )
            lens_unit_start, lens_unit_end = _columns_sample_interval(
                tuple(range(marker, vowel + 1)),
                render.lens_durations,
                render.lens_pcm.size,
            )
            neutral_vowel_start, neutral_vowel_end = _sample_interval(
                vowel, render.neutral_durations, render.neutral_pcm.size
            )
            lens_vowel_start, lens_vowel_end = _sample_interval(
                vowel, render.lens_durations, render.lens_pcm.size
            )
            neutral_unit = interval_summary(
                neutral_frames,
                neutral_unit_start,
                neutral_unit_end,
                retain_fraction=1.0,
            )
            lens_unit = interval_summary(
                lens_frames,
                lens_unit_start,
                lens_unit_end,
                retain_fraction=1.0,
            )
            neutral_vowel = interval_summary(
                neutral_frames,
                neutral_vowel_start,
                neutral_vowel_end,
                retain_fraction=0.8,
            )
            lens_vowel = interval_summary(
                lens_frames,
                lens_vowel_start,
                lens_vowel_end,
                retain_fraction=0.8,
            )
            duration_delta = (
                lens_unit["decoder_aligned_duration_ms"]
                - neutral_unit["decoder_aligned_duration_ms"]
            )
            rms_ratio = _ratio(lens_vowel["median_rms"], neutral_vowel["median_rms"])
            enough_frames = bool(
                neutral_vowel["measurement_frame_count"] >= minimum_frames
                and lens_vowel["measurement_frame_count"] >= minimum_frames
            )
            if role == "promoted":
                duration_pass = duration_delta >= minimum_duration_delta_ms
                rms_pass = bool(
                    rms_ratio is not None and rms_ratio >= minimum_promoted_rms_ratio
                )
            else:
                duration_pass = duration_delta <= -minimum_duration_delta_ms
                rms_pass = bool(
                    rms_ratio is not None and rms_ratio <= maximum_demoted_rms_ratio
                )
            roles[role] = {
                "neutral_stress_unit": neutral_unit,
                "lens_stress_unit": lens_unit,
                "neutral_vowel": neutral_vowel,
                "lens_vowel": lens_vowel,
                "stress_unit_duration_delta_ms": duration_delta,
                "vowel_rms_ratio": rms_ratio,
                "enough_vowel_frames": enough_frames,
                "duration_direction_pass": duration_pass,
                "rms_direction_pass": rms_pass,
                "gate_pass": bool(enough_frames and duration_pass and rms_pass),
            }
        row_pass = all(value["gate_pass"] for value in roles.values())
        rows.append(
            {
                "intervention": intervention,
                "roles": roles,
                "gate_pass": row_pass,
            }
        )
    gate_pass = bool(rows and all(row["gate_pass"] for row in rows))
    return {
        "measurement": (
            "decoder-aligned duration over each stress-marker-plus-vowel unit and "
            "standalone-Praat median RMS over the middle 80% of its vowel"
        ),
        "occurrences": rows,
        "gate_pass": gate_pass,
        "status": "pass" if gate_pass else "fail",
    }


def _thirds(
    frames: Sequence[dict[str, float | None]], interval: dict[str, Any]
) -> tuple[dict[str, Any], ...]:
    start = float(interval["start_s"])
    end = float(interval["end_s"])
    width = (end - start) / 3.0
    return tuple(
        interval_summary(
            frames,
            start + index * width,
            start + (index + 1) * width,
            retain_fraction=0.8,
        )
        for index in range(3)
    )


def measure_question_component(
    render: ProsodyComponentRender,
    neutral_frames: Sequence[dict[str, float | None]],
    lens_frames: Sequence[dict[str, float | None]],
    *,
    minimum_frames: int,
    minimum_voiced_fraction: float,
    minimum_neutral_rise_ratio: float,
    maximum_neutral_end_to_peak_ratio: float,
    maximum_lens_end_to_start_ratio: float,
    maximum_lens_middle_to_start_ratio: float,
) -> dict[str, Any]:
    if len(render.target_intervals) != 1:
        raise RuntimeError("question contour requires exactly one final interval")
    interval = render.target_intervals[0]
    neutral = _thirds(neutral_frames, interval)
    lens = _thirds(lens_frames, interval)
    neutral_pitch = [row["median_pitch_hz"] for row in neutral]
    lens_pitch = [row["median_pitch_hz"] for row in lens]
    neutral_rise = _ratio(neutral_pitch[1], neutral_pitch[0])
    neutral_fall = _ratio(neutral_pitch[2], neutral_pitch[1])
    lens_fall = _ratio(lens_pitch[2], lens_pitch[0])
    lens_middle = _ratio(lens_pitch[1], lens_pitch[0])
    enough_frames = all(
        row["measurement_frame_count"] >= minimum_frames
        and row["voiced_frame_fraction"] >= minimum_voiced_fraction
        for row in (*neutral, *lens)
    )
    checks = {
        "enough_voiced_frames": enough_frames,
        "neutral_rise": bool(
            neutral_rise is not None and neutral_rise >= minimum_neutral_rise_ratio
        ),
        "neutral_fall_after_peak": bool(
            neutral_fall is not None
            and neutral_fall <= maximum_neutral_end_to_peak_ratio
        ),
        "lens_final_fall": bool(
            lens_fall is not None and lens_fall <= maximum_lens_end_to_start_ratio
        ),
        "lens_no_middle_peak": bool(
            lens_middle is not None
            and lens_middle <= maximum_lens_middle_to_start_ratio
        ),
    }
    gate_pass = all(checks.values())
    return {
        "measurement": (
            "standalone-Praat pitch medians over the middle 80% of each third "
            "of the controlled final contour"
        ),
        "interval": interval,
        "neutral_thirds": neutral,
        "lens_thirds": lens,
        "neutral_middle_to_start_ratio": neutral_rise,
        "neutral_end_to_middle_ratio": neutral_fall,
        "lens_end_to_start_ratio": lens_fall,
        "lens_middle_to_start_ratio": lens_middle,
        "checks": checks,
        "gate_pass": gate_pass,
        "status": "pass" if gate_pass else "fail",
    }
