from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
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
from earshift_bakeoff.kokoro_synthesis import SAMPLE_RATE_HZ, _filtered_symbols, pcm16_bytes
from earshift_bakeoff.util import atomic_write_json, sha256_file


RUN_ID = "20260717-consonant-carrier-search-v1"
RUN_DIR = ROOT / "artifacts" / "consonant-calibration" / RUN_ID
ALLOSAURUS_BATCH = ROOT / "scripts" / "run_allosaurus_batch.py"
PREFIX = "mˈɑl vˈɑn "
SUFFIX = " nˈɑl sˈɑv."
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


def _pcm(audio: np.ndarray) -> np.ndarray:
    return np.frombuffer(pcm16_bytes(audio), dtype="<i2").copy()


def _write_wav(path: Path, values: np.ndarray) -> dict[str, Any]:
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
    return {
        "relative_path": str(path.relative_to(RUN_DIR)),
        "sha256": sha256_file(path),
    }


def _target_columns(model: Any, phonemes: str, phone: str) -> tuple[int, ...]:
    symbols = _filtered_symbols(model, phonemes)
    target = tuple(symbol for symbol in phone if model.vocab.get(symbol) is not None)
    matches = [
        index
        for index in range(len(symbols) - len(target) + 1)
        if symbols[index : index + len(target)] == target
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one target {phone!r} in {phonemes!r}")
    return tuple(range(matches[0] + 1, matches[0] + len(target) + 1))


def main() -> None:
    if RUN_DIR.exists():
        raise RuntimeError(f"carrier search already exists: {RUN_DIR}")
    RUN_DIR.mkdir(parents=True)
    candidates = []
    for rule_id, source, target, source_labels, target_labels in RULES:
        for index, (position, shell) in enumerate(SHELLS, 1):
            source_word = shell.replace("C", source)
            target_word = shell.replace("C", target)
            candidates.append(
                {
                    "candidate_id": f"{rule_id}__{position}__{index:02d}",
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
        "classification": "deterministic_consonant_carrier_shell_search",
        "api_calls_authorized": 0,
        "candidate_count": len(candidates),
        "candidate_order": candidates,
        "selection": (
            "Within each rule and source position, freeze the first two candidates "
            "whose source and natural-target anchors both match the auxiliary phone "
            "instrument. Selection is diagnostic and cannot promote a product rule."
        ),
        "stopping": "Evaluate all 36 frozen candidates once; no replacement candidates.",
    }
    protocol["protocol_sha256"] = hashlib.sha256(
        stable_json(protocol).encode()
    ).hexdigest()
    atomic_write_json(RUN_DIR / "protocol.json", protocol)

    runtime = _load_pinned_synthesis_voice("af_heart")
    records = []
    recognizer_inputs = []
    for candidate in candidates:
        conditions: dict[str, Any] = {}
        for role, phonemes, phone in (
            ("source_anchor", candidate["source_phonemes"], candidate["source"]),
            ("target_anchor", candidate["target_phonemes"], candidate["target"]),
        ):
            rendered = render_natural_condition(
                runtime,
                phonemes=phonemes,
                reference_phonemes=candidate["source_phonemes"],
            )
            pcm = _pcm(rendered.audio)
            columns = _target_columns(runtime.model, phonemes, phone)
            interval = decoder_column_interval(
                rendered.predicted_durations,
                columns,
                sample_count=pcm.size,
                sample_rate_hz=SAMPLE_RATE_HZ,
            )
            path = RUN_DIR / "audio" / f"{candidate['candidate_id']}__{role}.wav"
            conditions[role] = {
                "audio": _write_wav(path, pcm),
                "predicted_durations": rendered.predicted_durations,
                "measurement_interval": interval.as_record(),
            }
            recognizer_inputs.append(
                {"id": f"{candidate['candidate_id']}::{role}", "path": str(path)}
            )
        records.append({**candidate, "conditions": conditions})

    input_path = RUN_DIR / "analysis" / "allosaurus-inputs.json"
    input_path.parent.mkdir(parents=True)
    atomic_write_json(input_path, {"run_id": RUN_ID, "inputs": recognizer_inputs})
    output_path = RUN_DIR / "analysis" / "allosaurus-output.json"
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
        timeout=300,
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
            rows = parse_allosaurus_timestamps(
                outputs[f"{record['candidate_id']}::{role}"]
            )
            interval = record["conditions"][role]["measurement_interval"]
            labels = overlapping_upr_labels(
                rows, start_s=interval["start_s"], end_s=interval["end_s"]
            )
            instrument[role] = {
                "all_timestamps": rows,
                "overlapping_labels": labels,
                "match": labels_support_expected(labels, expected),
            }
        record["universal_phone_recognizer"] = instrument
        record["dual_anchor_match"] = bool(
            instrument["source_anchor"]["match"]
            and instrument["target_anchor"]["match"]
        )

    selected: dict[str, dict[str, list[str]]] = {}
    for rule_id, *_ in RULES:
        selected[rule_id] = {}
        for position in ("word_initial", "intervocalic", "word_final"):
            passing = [
                row["candidate_id"]
                for row in records
                if row["rule_id"] == rule_id
                and row["position"] == position
                and row["dual_anchor_match"]
            ]
            selected[rule_id][position] = passing[:2]
    payload = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "protocol_sha256": protocol["protocol_sha256"],
        "api_calls_made": 0,
        "records": records,
        "first_two_dual_anchor_matches": selected,
        "classification": "carrier_shell_search_complete_no_rule_promotion",
    }
    payload["records_sha256"] = hashlib.sha256(
        stable_json(payload).encode()
    ).hexdigest()
    atomic_write_json(RUN_DIR / "records.json", payload)
    print(json.dumps(selected, indent=2))


if __name__ == "__main__":
    main()
