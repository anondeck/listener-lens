from __future__ import annotations

import csv
import hashlib
import html
import json
import math
import subprocess
import wave
from dataclasses import asdict
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

from earshift_bakeoff.bilingual_listener_engine import (
    BILINGUAL_LISTENER_CANDIDATE_VERSION,
    BILINGUAL_LISTENER_RULES_PATH,
    BilingualListenerRuntime,
)
from earshift_bakeoff.bilingual_vowel_engine import BilingualVowelRender
from earshift_bakeoff.config import Paths, stable_json
from earshift_bakeoff.kokoro_synthesis import SAMPLE_RATE_HZ
from earshift_bakeoff.util import atomic_write_json, sha256_file


RUN_ID = "20260717-bidirectional-listener-smoke-v5"
RUN_DIR = Paths().artifacts / "bilingual-listener-engine" / RUN_ID
PRAAT = Path("/opt/homebrew/bin/praat")
PRAAT_PROBE = Paths().root / "scripts" / "praat_same_take_probe.praat"

# Frozen before the first saved render. These are exploratory engineering gates,
# not population-level perception criteria.
MIN_VOICED_FRAME_FRACTION = 0.50
MIN_MEASUREMENT_FRAMES = 5
QUESTION_MIN_RISE_RATIO = 1.05
QUESTION_MAX_END_TO_PEAK_RATIO = 0.90
QUESTION_MIN_FALL_RATIO = 0.90
QUESTION_MAX_LENS_MIDDLE_TO_START_RATIO = 1.05
STRESS_MIN_DURATION_DELTA_MS = 20.0
STRESS_MIN_SECONDARY_CUE_RATIO = 1.02
INSERTION_F1_BOUNDS_HZ = (200.0, 1000.0)
INSERTION_F2_BOUNDS_HZ = (600.0, 3500.0)

FIXTURES = (
    {
        "fixture_id": "english-segments-and-epenthesis",
        "profile_id": "en-US-to-pt-BR-listener-v2",
        "text": "The black cat slept.",
        "purpose": "vowels, /ð/ recategorization, and context-bounded epenthesis",
    },
    {
        "fixture_id": "english-lexical-stress",
        "profile_id": "en-US-to-pt-BR-listener-v2",
        "text": "Information arrives today.",
        "purpose": "initial-secondary versus later-primary lexical-stress candidate",
    },
    {
        "fixture_id": "portuguese-question-contour",
        "profile_id": "pt-BR-to-en-US-listener-v2",
        "text": "Minha filha trabalha no Brasil?",
        "purpose": "segments plus BP rise-fall versus English-listener statement-fall candidate",
    },
)


def _write_wav(path: Path, pcm: np.ndarray) -> dict[str, Any]:
    values = np.asarray(pcm, dtype="<i2").reshape(-1)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with wave.open(str(temporary), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE_HZ)
        handle.writeframes(values.tobytes())
    temporary.replace(path)
    return {
        "relative_path": str(path.relative_to(RUN_DIR)),
        "wav_sha256": sha256_file(path),
        "pcm_sha256": hashlib.sha256(values.tobytes()).hexdigest(),
        "sample_count": int(values.size),
        "duration_s": values.size / SAMPLE_RATE_HZ,
    }


def _number(value: str | None) -> float | None:
    try:
        result = float(value or "")
    except ValueError:
        return None
    return result if math.isfinite(result) else None


def _probe(path: Path, output: Path) -> list[dict[str, float | None]]:
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [str(PRAAT), "--run", str(PRAAT_PROBE), str(path), str(output)],
        check=True,
        capture_output=True,
        text=True,
    )
    with output.open(encoding="utf-8", newline="") as handle:
        source = list(csv.DictReader(handle, delimiter="\t"))
    return [
        {
            key: _number(row.get(key))
            for key in ("time_s", "f1_hz", "f2_hz", "f3_hz", "f4_hz", "pitch_hz", "rms")
        }
        for row in source
    ]


def _summary(
    frames: list[dict[str, float | None]], start_s: float, end_s: float
) -> dict[str, Any]:
    middle_start = start_s + (end_s - start_s) * 0.25
    middle_end = start_s + (end_s - start_s) * 0.75
    selected = [
        row
        for row in frames
        if row["time_s"] is not None and middle_start <= row["time_s"] <= middle_end
    ]

    def values(key: str) -> list[float]:
        return [float(row[key]) for row in selected if row[key] is not None]

    pitch = values("pitch_hz")
    f1 = values("f1_hz")
    f2 = values("f2_hz")
    rms = values("rms")
    return {
        "interval_s": [start_s, end_s],
        "duration_ms": (end_s - start_s) * 1000,
        "middle_frame_count": len(selected),
        "voiced_frame_fraction": len(pitch) / len(selected) if selected else 0.0,
        "median_pitch_hz": median(pitch) if pitch else None,
        "median_f1_hz": median(f1) if f1 else None,
        "median_f2_hz": median(f2) if f2 else None,
        "median_rms": median(rms) if rms else None,
    }


def _thirds(
    frames: list[dict[str, float | None]], interval: dict[str, Any]
) -> list[dict[str, Any]]:
    start = float(interval["start_s"])
    end = float(interval["end_s"])
    width = (end - start) / 3
    return [
        _summary(frames, start + width * index, start + width * (index + 1))
        for index in range(3)
    ]


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator == 0:
        return None
    return numerator / denominator


def _question_measurement(
    result: BilingualVowelRender,
    probes: dict[str, list[dict[str, float | None]]],
) -> dict[str, Any] | None:
    interval = (result.prosody or {}).get("sample_window")
    if interval is None:
        return None
    neutral = _thirds(probes["neutral"], interval)
    lens = _thirds(probes["lens"], interval)
    neutral_pitch = [row["median_pitch_hz"] for row in neutral]
    lens_pitch = [row["median_pitch_hz"] for row in lens]
    neutral_rise = _ratio(neutral_pitch[1], neutral_pitch[0])
    neutral_fall = _ratio(neutral_pitch[2], neutral_pitch[1])
    lens_fall = _ratio(lens_pitch[2], lens_pitch[0])
    lens_middle = _ratio(lens_pitch[1], lens_pitch[0])
    enough_frames = all(
        row["middle_frame_count"] >= MIN_MEASUREMENT_FRAMES
        and row["voiced_frame_fraction"] >= MIN_VOICED_FRAME_FRACTION
        for row in (*neutral, *lens)
    )
    gate_pass = bool(
        enough_frames
        and neutral_rise is not None
        and neutral_rise >= QUESTION_MIN_RISE_RATIO
        and neutral_fall is not None
        and neutral_fall <= QUESTION_MAX_END_TO_PEAK_RATIO
        and lens_fall is not None
        and lens_fall <= QUESTION_MIN_FALL_RATIO
        and lens_middle is not None
        and lens_middle <= QUESTION_MAX_LENS_MIDDLE_TO_START_RATIO
    )
    return {
        "status": "pass" if gate_pass else "fail",
        "interval": interval,
        "neutral_thirds": neutral,
        "lens_thirds": lens,
        "neutral_middle_to_start_ratio": neutral_rise,
        "neutral_end_to_middle_ratio": neutral_fall,
        "lens_end_to_start_ratio": lens_fall,
        "lens_middle_to_start_ratio": lens_middle,
        "gate_pass": gate_pass,
    }


def _stress_measurement(
    result: BilingualVowelRender,
    probes: dict[str, list[dict[str, float | None]]],
) -> dict[str, Any] | None:
    neutral_rows = [
        row
        for row in result.alignment["target_occurrences"]
        if row["segment_type"] == "prosody"
    ]
    lens_alignment = result.lens_alignment or result.alignment
    lens_rows = [
        row
        for row in lens_alignment["target_occurrences"]
        if row["segment_type"] == "prosody"
    ]
    if not neutral_rows:
        return None
    measurements = []
    passes = []
    for neutral_row, lens_row in zip(neutral_rows, lens_rows, strict=True):
        neutral_interval = neutral_row["measurement_interval"]
        lens_interval = lens_row["measurement_interval"]
        neutral_summary = _summary(
            probes["neutral"], neutral_interval["start_s"], neutral_interval["end_s"]
        )
        lens_summary = _summary(
            probes["lens"], lens_interval["start_s"], lens_interval["end_s"]
        )
        duration_delta = lens_summary["duration_ms"] - neutral_summary["duration_ms"]
        pitch_ratio = _ratio(
            lens_summary["median_pitch_hz"], neutral_summary["median_pitch_hz"]
        )
        rms_ratio = _ratio(lens_summary["median_rms"], neutral_summary["median_rms"])
        promoted = neutral_row["source"] == "ˌ" and neutral_row["target"] == "ˈ"
        duration_pass = (
            duration_delta >= STRESS_MIN_DURATION_DELTA_MS
            if promoted
            else duration_delta <= -STRESS_MIN_DURATION_DELTA_MS
        )
        secondary_cue_pass = any(
            ratio is not None
            and (
                ratio >= STRESS_MIN_SECONDARY_CUE_RATIO
                if promoted
                else ratio <= 1 / STRESS_MIN_SECONDARY_CUE_RATIO
            )
            for ratio in (pitch_ratio, rms_ratio)
        )
        row_pass = bool(duration_pass and secondary_cue_pass)
        passes.append(row_pass)
        measurements.append(
            {
                "source": neutral_row["source"],
                "target": neutral_row["target"],
                "neutral": neutral_summary,
                "lens": lens_summary,
                "duration_delta_ms": duration_delta,
                "pitch_ratio": pitch_ratio,
                "rms_ratio": rms_ratio,
                "duration_pass": duration_pass,
                "secondary_cue_pass": secondary_cue_pass,
                "gate_pass": row_pass,
            }
        )
    gate_pass = bool(passes and all(passes))
    return {
        "status": "pass" if gate_pass else "fail",
        "occurrences": measurements,
        "gate_pass": gate_pass,
    }


def _insertion_measurement(
    result: BilingualVowelRender,
    probes: dict[str, list[dict[str, float | None]]],
) -> dict[str, Any] | None:
    rows = [
        row
        for row in result.alignment["target_occurrences"]
        if row["segment_type"] == "insertion"
    ]
    if not rows:
        return None
    measurements = []
    for row in rows:
        interval = row["measurement_interval"]
        neutral = _summary(probes["neutral"], interval["start_s"], interval["end_s"])
        lens = _summary(probes["lens"], interval["start_s"], interval["end_s"])
        gate_pass = bool(
            lens["middle_frame_count"] >= MIN_MEASUREMENT_FRAMES
            and lens["voiced_frame_fraction"] >= MIN_VOICED_FRAME_FRACTION
            and lens["median_f1_hz"] is not None
            and INSERTION_F1_BOUNDS_HZ[0]
            <= lens["median_f1_hz"]
            <= INSERTION_F1_BOUNDS_HZ[1]
            and lens["median_f2_hz"] is not None
            and INSERTION_F2_BOUNDS_HZ[0]
            <= lens["median_f2_hz"]
            <= INSERTION_F2_BOUNDS_HZ[1]
        )
        measurements.append(
            {
                "word_index": row["word_index"],
                "neutral": neutral,
                "lens": lens,
                "gate_pass": gate_pass,
            }
        )
    gate_pass = all(row["gate_pass"] for row in measurements)
    return {
        "status": "pass" if gate_pass else "fail",
        "occurrences": measurements,
        "gate_pass": gate_pass,
    }


def _review_html(records: list[dict[str, Any]]) -> str:
    sections = []
    for index, record in enumerate(records, 1):
        audio = record["audio"]
        sections.append(
            f"""<section><h2>{index}. {html.escape(record["fixture_id"])}</h2>
            <p>{html.escape(record["purpose"])}</p><div class="pair">
            <article><h3>A</h3><audio controls src="{html.escape(audio["neutral"]["relative_path"])}"></audio></article>
            <article><h3>B</h3><audio controls src="{html.escape(audio["lens"]["relative_path"])}"></audio></article>
            </div><p>Automatic integrity: {record["verification"]["integrity_pass"]} · measurement status:
            {html.escape(str(record["measurements"]))}</p></section>"""
        )
    return f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
    <title>Bidirectional listener-engine smoke v2</title><style>body{{font:16px/1.5 system-ui;max-width:960px;margin:auto;padding:24px;background:#f4f1e8;color:#17241e}}section{{background:white;padding:20px;border-radius:14px;margin:18px 0}}.pair{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}audio{{width:100%}}@media(max-width:650px){{.pair{{grid-template-columns:1fr}}}}</style></head>
    <body><h1>Listener-engine smoke v2</h1><p><strong>Exploratory QC only.</strong> Conditions are lettered; scripts and rule identities are not shown during first listening.</p>{"".join(sections)}</body></html>"""


def main() -> None:
    if not PRAAT.exists() or not PRAAT_PROBE.exists():
        raise RuntimeError("standalone Praat or the frozen probe script is missing")
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    protocol = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "classification": "exploratory_all_segment_and_prosody_engineering_smoke",
        "candidate_version": BILINGUAL_LISTENER_CANDIDATE_VERSION,
        "api_calls_authorized": 0,
        "fixtures": FIXTURES,
        "measurement_gates": {
            key: value
            for key, value in globals().items()
            if key.startswith(("MIN_", "QUESTION_", "STRESS_", "INSERTION_"))
            and isinstance(value, (int, float, tuple))
        },
        "rules_sha256": sha256_file(BILINGUAL_LISTENER_RULES_PATH),
        "praat_sha256": sha256_file(PRAAT),
        "praat_probe_sha256": sha256_file(PRAAT_PROBE),
    }
    protocol["protocol_sha256"] = hashlib.sha256(
        stable_json(protocol).encode()
    ).hexdigest()
    atomic_write_json(RUN_DIR / "protocol.json", protocol)

    records: list[dict[str, Any]] = []
    for fixture in FIXTURES:
        runtime = BilingualListenerRuntime.load(fixture["profile_id"])
        result = runtime.render(fixture["text"])
        if not isinstance(result, BilingualVowelRender):
            raise RuntimeError(
                f"fixture produced no comparison: {fixture['fixture_id']}"
            )
        audio = {
            role: _write_wav(
                RUN_DIR / "audio" / f"{fixture['fixture_id']}__{role}.wav", values
            )
            for role, values in (
                ("neutral", result.neutral_pcm),
                ("identity", result.identity_pcm),
                ("full_lens_diagnostic", result.full_lens_pcm),
                ("lens", result.lens_pcm),
            )
        }
        probes = {
            role: _probe(
                RUN_DIR / audio[role]["relative_path"],
                RUN_DIR / "analysis" / f"{fixture['fixture_id']}__{role}.tsv",
            )
            for role in ("neutral", "lens")
        }
        measurements = {
            "question_contour": _question_measurement(result, probes),
            "lexical_stress": _stress_measurement(result, probes),
            "epenthesis": _insertion_measurement(result, probes),
        }
        records.append(
            {
                **fixture,
                "plan_sha256": result.plan.plan_sha256,
                "coverage": asdict(result.plan.coverage),
                "verification": asdict(result.verification),
                "measurements": measurements,
                "audio": audio,
            }
        )

    result_payload = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "protocol_sha256": protocol["protocol_sha256"],
        "api_calls_made": 0,
        "paid_calls_made": 0,
        "fixtures": records,
        "classification": "engineering_integrity_pass_acoustic_and_listener_validation_incomplete",
    }
    result_payload["record_sha256"] = hashlib.sha256(
        stable_json(result_payload).encode()
    ).hexdigest()
    atomic_write_json(RUN_DIR / "records.json", result_payload)
    (RUN_DIR / "review.html").write_text(_review_html(records), encoding="utf-8")
    print(
        json.dumps(
            {
                "review": str(RUN_DIR / "review.html"),
                "records": str(RUN_DIR / "records.json"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
