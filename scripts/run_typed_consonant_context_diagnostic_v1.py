from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import wave
from typing import Any

import numpy as np

from earshift_bakeoff.bilingual_listener_engine import BilingualListenerRuntime
from earshift_bakeoff.bilingual_vowel_engine import bilingual_alignment_record
from earshift_bakeoff.config import ROOT, stable_json
from earshift_bakeoff.consonant_calibration import (
    ALLOSAURUS_SCIPY_VERSION,
    ALLOSAURUS_VERSION,
    labels_support_expected,
    overlapping_upr_labels,
    parse_allosaurus_timestamps,
)
from earshift_bakeoff.controlled_listener_synthesis import (
    render_controlled_listener_triplet,
    render_natural_condition,
)
from earshift_bakeoff.kokoro_synthesis import SAMPLE_RATE_HZ, pcm16_bytes
from earshift_bakeoff.util import atomic_write_json, sha256_file


RUN_ID = "20260717-typed-consonant-context-diagnostic-v3"
RUN_DIR = ROOT / "artifacts" / "consonant-calibration" / RUN_ID
ALLOSAURUS_BATCH = ROOT / "scripts" / "run_allosaurus_batch.py"
FIXTURES = (
    ("theta-initial", "Think about music.", "enpt.theta_t", ("θ",), ("t", "tʰ", "t̪")),
    ("theta-final", "Math is useful.", "enpt.theta_t", ("θ",), ("t", "tʰ", "t̪")),
    ("theta-intervocalic", "A method works.", "enpt.theta_t", ("θ",), ("t", "tʰ", "t̪")),
    ("eth-initial-this", "This black cat slept.", "enpt.eth_d", ("ð",), ("d", "d̪")),
    ("eth-initial-those", "Those cats slept.", "enpt.eth_d", ("ð",), ("d", "d̪")),
    ("eth-final", "Breathe slowly.", "enpt.eth_d", ("ð",), ("d", "d̪")),
)
MODES = ("adjacent", "word")


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
        "wav_sha256": sha256_file(path),
        "pcm_sha256": hashlib.sha256(
            np.asarray(values, dtype="<i2").tobytes()
        ).hexdigest(),
    }


def main() -> None:
    if RUN_DIR.exists():
        raise RuntimeError(f"diagnostic already exists: {RUN_DIR}")
    RUN_DIR.mkdir(parents=True)
    protocol = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "classification": "typed_carrier_consonant_state_context_diagnostic",
        "api_calls_authorized": 0,
        "fixtures": [
            {
                "fixture_id": fixture_id,
                "text": text,
                "rule_id": rule_id,
                "expected_source_labels": source_labels,
                "expected_target_labels": target_labels,
            }
            for fixture_id, text, rule_id, source_labels, target_labels in FIXTURES
        ],
        "candidate_modes": MODES,
        "changed_mechanism": (
            "Add a truly fully conditioned target anchor with its own duration, F0, "
            "noise, and decoder state. Existing controlled candidates remain unchanged "
            "from diagnostic v2."
        ),
        "selection": (
            "No listening selection. Compare the existing adjacent state against the "
            "whole target-carrier-word state on all six frozen typed fixtures."
        ),
        "interpretation": (
            "Auxiliary phone-recognizer agreement diagnoses category retention; it does "
            "not promote a rule or establish listener perception."
        ),
    }
    protocol["protocol_sha256"] = hashlib.sha256(
        stable_json(protocol).encode()
    ).hexdigest()
    atomic_write_json(RUN_DIR / "protocol.json", protocol)

    runtime = BilingualListenerRuntime.load("en-US-to-pt-BR-listener-v2")
    records: list[dict[str, Any]] = []
    recognizer_inputs: list[dict[str, str]] = []
    for fixture_id, text, rule_id, source_labels, target_labels in FIXTURES:
        plan = runtime.planner.plan(text)
        pair = plan.pair_plan()
        if pair is None:
            raise RuntimeError(f"typed fixture produced no comparison: {fixture_id}")
        matching = [
            row
            for word in plan.words
            for row in word.consonant_occurrences
            if row.changed and row.rule_id == rule_id
        ]
        if len(matching) != 1:
            raise RuntimeError(f"{fixture_id} did not produce exactly one target rule")
        natural = render_natural_condition(
            runtime.synthesis,
            phonemes=plan.lens_phonemes,
            reference_phonemes=plan.render_reference_phonemes,
        )
        natural_pcm = _pcm(natural.audio)
        natural_alignment = bilingual_alignment_record(
            model=runtime.synthesis.model,
            plan=plan,
            durations=natural.predicted_durations,
            sample_count=natural_pcm.size,
        )
        natural_occurrences = [
            row
            for row in natural_alignment["target_occurrences"]
            if row["segment_type"] == "consonant" and row["rule_id"] == rule_id
        ]
        if len(natural_occurrences) != 1:
            raise RuntimeError(f"{fixture_id} lost its natural-target alignment")
        natural_path = RUN_DIR / "audio" / f"{fixture_id}__natural-target.wav"
        natural_audio = _write_wav(natural_path, natural_pcm)
        recognizer_inputs.append(
            {"id": f"{fixture_id}::natural_target", "path": str(natural_path)}
        )
        for mode in MODES:
            rendered = render_controlled_listener_triplet(
                runtime.synthesis,
                pair,
                insertion_columns=runtime._insertion_model_columns(plan),
                consonant_columns=runtime._consonant_model_columns(plan),
                consonant_context_mode=mode,
            )
            neutral = _pcm(rendered.neutral)
            identity = _pcm(rendered.identity)
            target = _pcm(rendered.full_lens)
            alignment = bilingual_alignment_record(
                model=runtime.synthesis.model,
                plan=plan,
                durations=rendered.predicted_durations,
                sample_count=neutral.size,
            )
            occurrences = [
                row
                for row in alignment["target_occurrences"]
                if row["segment_type"] == "consonant" and row["rule_id"] == rule_id
            ]
            if len(occurrences) != 1:
                raise RuntimeError(f"{fixture_id} lost its consonant alignment")
            interval = occurrences[0]["measurement_interval"]
            audio: dict[str, Any] = {}
            for role, values in (("source", neutral), ("target", target)):
                path = RUN_DIR / "audio" / f"{fixture_id}__{mode}__{role}.wav"
                audio[role] = _write_wav(path, values)
                recognizer_inputs.append(
                    {"id": f"{fixture_id}::{mode}::{role}", "path": str(path)}
                )
            records.append(
                {
                    "fixture_id": fixture_id,
                    "text": text,
                    "rule_id": rule_id,
                    "mode": mode,
                    "expected_source_labels": source_labels,
                    "expected_target_labels": target_labels,
                    "neutral_phonemes": plan.neutral_phonemes,
                    "lens_phonemes": plan.lens_phonemes,
                    "plan_sha256": plan.plan_sha256,
                    "coverage": asdict(plan.coverage),
                    "measurement_interval": interval,
                    "identity_bit_exact": bool(np.array_equal(neutral, identity)),
                    "audio": audio,
                    "natural_target": {
                        "audio": natural_audio,
                        "predicted_durations": natural.predicted_durations,
                        "measurement_interval": natural_occurrences[0][
                            "measurement_interval"
                        ],
                    },
                }
            )

    manifest_path = RUN_DIR / "analysis" / "allosaurus-inputs.json"
    manifest_path.parent.mkdir(parents=True)
    atomic_write_json(manifest_path, {"run_id": RUN_ID, "inputs": recognizer_inputs})
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
            str(manifest_path),
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
        interval = record["measurement_interval"]
        roles: dict[str, Any] = {}
        for role, expected in (
            ("source", record["expected_source_labels"]),
            ("target", record["expected_target_labels"]),
        ):
            rows = parse_allosaurus_timestamps(
                outputs[f"{record['fixture_id']}::{record['mode']}::{role}"]
            )
            labels = overlapping_upr_labels(
                rows, start_s=interval["start_s"], end_s=interval["end_s"]
            )
            roles[role] = {
                "all_timestamps": rows,
                "overlapping_labels": labels,
                "match": labels_support_expected(labels, expected),
            }
        record["universal_phone_recognizer"] = roles
        natural_rows = parse_allosaurus_timestamps(
            outputs[f"{record['fixture_id']}::natural_target"]
        )
        natural_interval = record["natural_target"]["measurement_interval"]
        natural_labels = overlapping_upr_labels(
            natural_rows,
            start_s=natural_interval["start_s"],
            end_s=natural_interval["end_s"],
        )
        record["universal_phone_recognizer"]["natural_target"] = {
            "all_timestamps": natural_rows,
            "overlapping_labels": natural_labels,
            "match": labels_support_expected(
                natural_labels, record["expected_target_labels"]
            ),
        }
        record["both_anchors_match"] = bool(
            roles["source"]["match"] and roles["target"]["match"]
        )
    summary: dict[str, Any] = {}
    for rule_id in sorted({row["rule_id"] for row in records}):
        summary[rule_id] = {
            mode: {
                "source_matches": sum(
                    row["universal_phone_recognizer"]["source"]["match"]
                    for row in records
                    if row["rule_id"] == rule_id and row["mode"] == mode
                ),
                "target_matches": sum(
                    row["universal_phone_recognizer"]["target"]["match"]
                    for row in records
                    if row["rule_id"] == rule_id and row["mode"] == mode
                ),
                "both_anchor_matches": sum(
                    row["both_anchors_match"]
                    for row in records
                    if row["rule_id"] == rule_id and row["mode"] == mode
                ),
                "natural_target_matches": sum(
                    row["universal_phone_recognizer"]["natural_target"]["match"]
                    for row in records
                    if row["rule_id"] == rule_id and row["mode"] == mode
                ),
            }
            for mode in MODES
        }
    payload = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "protocol_sha256": protocol["protocol_sha256"],
        "api_calls_made": 0,
        "records": records,
        "summary": summary,
        "classification": "diagnostic_complete_no_rule_promotion",
    }
    payload["records_sha256"] = hashlib.sha256(
        stable_json(payload).encode()
    ).hexdigest()
    atomic_write_json(RUN_DIR / "records.json", payload)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
