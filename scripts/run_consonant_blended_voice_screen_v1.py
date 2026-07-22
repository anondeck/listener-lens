from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any

import torch

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
from earshift_bakeoff.kokoro_specs import VOICE_SPECS_BY_ID, resolve_pinned_file
from earshift_bakeoff.kokoro_synthesis import SAMPLE_RATE_HZ
from earshift_bakeoff.util import atomic_write_json, sha256_file
from run_consonant_voice_retention_screen_v1 import (
    ALLOSAURUS_BATCH,
    PREFIX,
    RULES,
    SHELLS,
    SUFFIX,
    _columns,
    _pcm,
    _write_wav,
)


RUN_ID = "20260717-consonant-blended-voice-screen-v1"
RUN_DIR = ROOT / "artifacts" / "consonant-calibration" / RUN_ID
BASE_SCRIPT = ROOT / "scripts" / "run_consonant_voice_retention_screen_v1.py"
BLENDS = (
    ("heart75-michael25", 0.75, 0.25),
    ("heart50-michael50", 0.50, 0.50),
    ("heart25-michael75", 0.25, 0.75),
)


def main() -> None:
    if RUN_DIR.exists():
        raise RuntimeError(f"blended voice screen already exists: {RUN_DIR}")
    candidates = []
    for blend_id, heart_weight, michael_weight in BLENDS:
        for rule_id, source, target, source_labels, target_labels in RULES:
            for index, (position, shell) in enumerate(SHELLS, 1):
                source_word = shell.replace("C", source)
                target_word = shell.replace("C", target)
                candidates.append(
                    {
                        "candidate_id": (
                            f"{blend_id}__{rule_id}__{position}__{index:02d}"
                        ),
                        "blend_id": blend_id,
                        "heart_weight": heart_weight,
                        "michael_weight": michael_weight,
                        "rule_id": rule_id,
                        "position": position,
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
        "classification": "deterministic_kokoro_voice_pack_blend_endpoint_screen",
        "api_calls_authorized": 0,
        "candidate_count": len(candidates),
        "blends": BLENDS,
        "component_voices": {
            voice_id: {
                "sha256": VOICE_SPECS_BY_ID[voice_id].sha256,
                "language_id": VOICE_SPECS_BY_ID[voice_id].language_id,
            }
            for voice_id in ("af_heart", "am_michael")
        },
        "formula": "voice_pack = heart_weight * af_heart + michael_weight * am_michael",
        "official_behavior": (
            "Pinned kokoro.pipeline.KPipeline.load_voice averages comma-separated voice "
            "packs; this screen evaluates explicit deterministic weights of the same "
            "style-pack mechanism."
        ),
        "base_inventory_script_sha256": sha256_file(BASE_SCRIPT),
        "candidate_order_sha256": hashlib.sha256(
            stable_json(candidates).encode()
        ).hexdigest(),
        "interpretation": (
            "A blend is only phoneme-retention eligible. It cannot be selected without "
            "naturalness, accent, sentence-flow, and controlled-pair QC."
        ),
    }
    protocol["protocol_sha256"] = hashlib.sha256(
        stable_json(protocol).encode()
    ).hexdigest()
    RUN_DIR.mkdir(parents=True)
    atomic_write_json(RUN_DIR / "protocol.json", protocol)

    runtime = _load_pinned_synthesis_voice("af_heart")
    heart_pack = runtime.voice_pack.detach().clone()
    michael_path = resolve_pinned_file(
        VOICE_SPECS_BY_ID["am_michael"].filename, download=False
    )
    if sha256_file(michael_path) != VOICE_SPECS_BY_ID["am_michael"].sha256:
        raise RuntimeError("am_michael voice pack hash drifted")
    michael_pack = torch.load(michael_path, map_location="cpu", weights_only=True)
    if heart_pack.shape != michael_pack.shape:
        raise RuntimeError("voice packs cannot be blended due to shape mismatch")

    records = []
    with tempfile.TemporaryDirectory(prefix="consonant-blend-screen-") as temp_name:
        temp_dir = Path(temp_name)
        recognizer_inputs = []
        for blend_id, heart_weight, michael_weight in BLENDS:
            runtime.voice_pack = (
                heart_pack * heart_weight + michael_pack * michael_weight
            )
            runtime.voice_id = blend_id
            blend_hash = hashlib.sha256(
                runtime.voice_pack.detach().cpu().numpy().astype("<f4").tobytes()
            ).hexdigest()
            for candidate in (row for row in candidates if row["blend_id"] == blend_id):
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
                    }
                    recognizer_inputs.append(
                        {
                            "id": f"{candidate['candidate_id']}::{role}",
                            "path": str(path),
                        }
                    )
                records.append(
                    {
                        **candidate,
                        "blend_pack_sha256": blend_hash,
                        "conditions": conditions,
                    }
                )
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
                rows = parse_allosaurus_timestamps(
                    outputs[f"{record['candidate_id']}::{role}"]
                )
                interval = record["conditions"][role]["measurement_interval"]
                labels = overlapping_upr_labels(
                    rows, start_s=interval["start_s"], end_s=interval["end_s"]
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
                temporary = Path(record["conditions"][role].pop("temporary_path"))
                if record["dual_anchor_match"]:
                    retained = (
                        RUN_DIR / "audio" / f"{record['candidate_id']}__{role}.wav"
                    )
                    retained.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(temporary, retained)
                    record["conditions"][role]["retained_relative_path"] = str(
                        retained.relative_to(RUN_DIR)
                    )
                else:
                    record["conditions"][role]["retained_relative_path"] = None

    summary: dict[str, Any] = {}
    for blend_id, *_ in BLENDS:
        summary[blend_id] = {}
        for rule_id, *_ in RULES:
            counts = {
                position: sum(
                    row["dual_anchor_match"]
                    for row in records
                    if row["blend_id"] == blend_id
                    and row["rule_id"] == rule_id
                    and row["position"] == position
                )
                for position in ("word_initial", "intervocalic", "word_final")
            }
            counts["total"] = sum(counts.values())
            summary[blend_id][rule_id] = counts
        summary[blend_id]["covers_both_rule_families"] = all(
            summary[blend_id][rule_id]["total"] > 0 for rule_id, *_ in RULES
        )
    payload = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "protocol_sha256": protocol["protocol_sha256"],
        "api_calls_made": 0,
        "records": records,
        "summary": summary,
        "classification": "blend_retention_screen_complete_no_automatic_selection",
    }
    payload["records_sha256"] = hashlib.sha256(
        stable_json(payload).encode()
    ).hexdigest()
    atomic_write_json(RUN_DIR / "records.json", payload)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
