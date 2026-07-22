from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import wave
from typing import Any

import numpy as np

from earshift_bakeoff.bilingual_vowel_engine import _load_pinned_synthesis_voice
from earshift_bakeoff.config import ROOT, stable_json
from earshift_bakeoff.consonant_acoustics import decoder_column_interval
from earshift_bakeoff.consonant_calibration import (
    ALLOSAURUS_SCIPY_VERSION,
    ALLOSAURUS_VERSION,
    labels_support_expected,
    overlapping_upr_labels,
    parse_allosaurus_timestamps,
)
from earshift_bakeoff.controlled_listener_synthesis import render_natural_condition
from earshift_bakeoff.kokoro_specs import VOICE_SPECS_BY_ID
from earshift_bakeoff.kokoro_synthesis import (
    SAMPLE_RATE_HZ,
    _filtered_symbols,
    pcm16_bytes,
)
from earshift_bakeoff.util import atomic_write_json, sha256_file


RUN_ID = "20260717-consonant-voice-retention-screen-v1"
RUN_DIR = ROOT / "artifacts" / "consonant-calibration" / RUN_ID
ALLOSAURUS_BATCH = ROOT / "scripts" / "run_allosaurus_batch.py"
VOICE_IDS = (
    "af_heart",
    "af_bella",
    "af_nicole",
    "am_fenrir",
    "am_michael",
    "am_puck",
)
RULES = (
    ("enpt.theta_t", "θ", "t", ("θ",), ("t", "tʰ", "t̪")),
    ("enpt.eth_d", "ð", "d", ("ð",), ("d", "d̪")),
)
SHELLS = (
    ("word_initial", "Cˈɪv"),
    ("word_initial", "Cˈɛv"),
    ("word_initial", "Cˈæm"),
    ("word_initial", "Cˈɑn"),
    ("word_initial", "Cˈʌz"),
    ("word_initial", "Cˈuv"),
    ("intervocalic", "vˈɪCəm"),
    ("intervocalic", "mˈɛCəv"),
    ("intervocalic", "lˈæCən"),
    ("intervocalic", "nˈɑCəl"),
    ("intervocalic", "vˈʌCəm"),
    ("intervocalic", "zˈuCəl"),
    ("word_final", "vˈɪC"),
    ("word_final", "mˈɛC"),
    ("word_final", "lˈæC"),
    ("word_final", "nˈɑC"),
    ("word_final", "vˈʌC"),
    ("word_final", "zˈuC"),
)
PREFIX = "mˈɑl vˈɑn "
SUFFIX = " nˈɑl sˈɑv."


def _pcm(audio: np.ndarray) -> np.ndarray:
    return np.frombuffer(pcm16_bytes(audio), dtype="<i2").copy()


def _write_wav(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    try:
        with wave.open(str(temporary), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(SAMPLE_RATE_HZ)
            handle.writeframes(np.asarray(values, dtype="<i2").tobytes())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _columns(model: Any, phonemes: str, phone: str) -> tuple[int, ...]:
    symbols = _filtered_symbols(model, phonemes)
    target = tuple(symbol for symbol in phone if model.vocab.get(symbol) is not None)
    matches = [
        index
        for index in range(len(symbols) - len(target) + 1)
        if symbols[index : index + len(target)] == target
    ]
    if len(matches) != 1:
        raise RuntimeError("voice-screen carrier lost its frozen phone target")
    return tuple(range(matches[0] + 1, matches[0] + len(target) + 1))


def main() -> None:
    if RUN_DIR.exists():
        raise RuntimeError(f"voice retention screen already exists: {RUN_DIR}")
    candidates = []
    for voice_id in VOICE_IDS:
        for rule_id, source, target, source_labels, target_labels in RULES:
            for index, (position, shell) in enumerate(SHELLS, 1):
                source_word = shell.replace("C", source)
                target_word = shell.replace("C", target)
                candidates.append(
                    {
                        "candidate_id": (
                            f"{voice_id}__{rule_id}__{position}__{index:02d}"
                        ),
                        "voice_id": voice_id,
                        "rule_id": rule_id,
                        "position": position,
                        "ordering_index": index,
                        "source": source,
                        "target": target,
                        "source_word": source_word,
                        "target_word": target_word,
                        "source_phonemes": PREFIX + source_word + SUFFIX,
                        "target_phonemes": PREFIX + target_word + SUFFIX,
                        "expected_source_labels": source_labels,
                        "expected_target_labels": target_labels,
                    }
                )
    protocol = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "classification": "multi_voice_consonant_endpoint_retention_screen",
        "api_calls_authorized": 0,
        "voice_ids": VOICE_IDS,
        "voice_hashes": {
            voice_id: VOICE_SPECS_BY_ID[voice_id].sha256 for voice_id in VOICE_IDS
        },
        "candidate_count": len(candidates),
        "candidate_order_sha256": hashlib.sha256(
            stable_json(candidates).encode()
        ).hexdigest(),
        "inventory": {
            "rules": RULES,
            "shells": SHELLS,
            "prefix": PREFIX,
            "suffix": SUFFIX,
        },
        "interpretation": (
            "Phone retention is an eligibility screen, not automatic voice selection. "
            "Naturalness and blind pair QC remain separate human judgments."
        ),
        "retention": (
            "Keep WAVs only for dual-anchor matches; record hashes and instrument output "
            "for every attempted candidate."
        ),
    }
    protocol["protocol_sha256"] = hashlib.sha256(
        stable_json(protocol).encode()
    ).hexdigest()
    RUN_DIR.mkdir(parents=True)
    atomic_write_json(RUN_DIR / "protocol.json", protocol)

    records = []
    with tempfile.TemporaryDirectory(prefix="consonant-voice-screen-") as temp_name:
        temp_dir = Path(temp_name)
        recognizer_inputs = []
        for voice_id in VOICE_IDS:
            runtime = _load_pinned_synthesis_voice(voice_id)
            for candidate in (row for row in candidates if row["voice_id"] == voice_id):
                conditions: dict[str, Any] = {}
                for role, phonemes, phone in (
                    (
                        "source_anchor",
                        candidate["source_phonemes"],
                        candidate["source"],
                    ),
                    (
                        "target_anchor",
                        candidate["target_phonemes"],
                        candidate["target"],
                    ),
                ):
                    rendered = render_natural_condition(
                        runtime,
                        phonemes=phonemes,
                        reference_phonemes=candidate["source_phonemes"],
                    )
                    pcm = _pcm(rendered.audio)
                    interval = decoder_column_interval(
                        rendered.predicted_durations,
                        _columns(runtime.model, phonemes, phone),
                        sample_count=pcm.size,
                        sample_rate_hz=SAMPLE_RATE_HZ,
                    )
                    path = temp_dir / f"{candidate['candidate_id']}__{role}.wav"
                    _write_wav(path, pcm)
                    conditions[role] = {
                        "temporary_path": str(path),
                        "wav_sha256": sha256_file(path),
                        "measurement_interval": interval.as_record(),
                        "predicted_durations_sha256": hashlib.sha256(
                            stable_json(rendered.predicted_durations).encode()
                        ).hexdigest(),
                    }
                    recognizer_inputs.append(
                        {
                            "id": f"{candidate['candidate_id']}::{role}",
                            "path": str(path),
                        }
                    )
                records.append({**candidate, "conditions": conditions})
        input_path = temp_dir / "allosaurus-inputs.json"
        output_path = temp_dir / "allosaurus-output.json"
        atomic_write_json(input_path, {"run_id": RUN_ID, "inputs": recognizer_inputs})
        completed = subprocess.run(
            [
                "uvx",
                "--python",
                "3.10",
                "--from",
                "allosaurus",
                "--with",
                f"scipy=={ALLOSAURUS_SCIPY_VERSION}",
                "python",
                str(ALLOSAURUS_BATCH),
                str(input_path),
                str(output_path),
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=600,
        )
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr or completed.stdout)
        recognized = json.loads(output_path.read_text(encoding="utf-8"))
        if recognized["allosaurus_version"] != ALLOSAURUS_VERSION:
            raise RuntimeError("Allosaurus version drifted")
        outputs = {row["id"]: row["timestamp_output"] for row in recognized["rows"]}
        for record in records:
            instrument: dict[str, Any] = {}
            for role, expected in (
                ("source_anchor", record["expected_source_labels"]),
                ("target_anchor", record["expected_target_labels"]),
            ):
                timestamps = parse_allosaurus_timestamps(
                    outputs[f"{record['candidate_id']}::{role}"]
                )
                interval = record["conditions"][role]["measurement_interval"]
                labels = overlapping_upr_labels(
                    timestamps,
                    start_s=interval["start_s"],
                    end_s=interval["end_s"],
                )
                instrument[role] = {
                    "overlapping_labels": labels,
                    "match": labels_support_expected(labels, expected),
                }
            record["universal_phone_recognizer"] = instrument
            record["dual_anchor_match"] = bool(
                instrument["source_anchor"]["match"]
                and instrument["target_anchor"]["match"]
            )
            for role in ("source_anchor", "target_anchor"):
                temporary_path = Path(record["conditions"][role].pop("temporary_path"))
                if record["dual_anchor_match"]:
                    retained = (
                        RUN_DIR / "audio" / f"{record['candidate_id']}__{role}.wav"
                    )
                    retained.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(temporary_path, retained)
                    record["conditions"][role]["retained_relative_path"] = str(
                        retained.relative_to(RUN_DIR)
                    )
                else:
                    record["conditions"][role]["retained_relative_path"] = None

    summary: dict[str, Any] = {}
    for voice_id in VOICE_IDS:
        summary[voice_id] = {}
        for rule_id, *_ in RULES:
            summary[voice_id][rule_id] = {
                position: sum(
                    row["dual_anchor_match"]
                    for row in records
                    if row["voice_id"] == voice_id
                    and row["rule_id"] == rule_id
                    and row["position"] == position
                )
                for position in ("word_initial", "intervocalic", "word_final")
            }
            summary[voice_id][rule_id]["total"] = sum(
                summary[voice_id][rule_id].values()
            )
    payload = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "protocol_sha256": protocol["protocol_sha256"],
        "api_calls_made": 0,
        "records": records,
        "summary": summary,
        "classification": "voice_retention_screen_complete_no_automatic_selection",
    }
    payload["records_sha256"] = hashlib.sha256(
        stable_json(payload).encode()
    ).hexdigest()
    atomic_write_json(RUN_DIR / "records.json", payload)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
