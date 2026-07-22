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
from earshift_bakeoff.consonant_acoustics import decoder_column_interval, expanded_interval
from earshift_bakeoff.consonant_calibration import (
    ALLOSAURUS_SCIPY_VERSION,
    ALLOSAURUS_VERSION,
    labels_support_expected,
    overlapping_upr_labels,
    parse_allosaurus_timestamps,
)
from earshift_bakeoff.controlled_listener_synthesis import (
    render_controlled_listener_triplet,
)
from earshift_bakeoff.kokoro_output_domain_splice import (
    boundary_artifact_report,
    output_domain_splice,
)
from earshift_bakeoff.kokoro_synthesis import (
    PairPlan,
    SAMPLE_RATE_HZ,
    _filtered_symbols,
    pcm16_bytes,
)
from earshift_bakeoff.kokoro_typed_diagnostic import localization_report
from earshift_bakeoff.util import atomic_write_json, sha256_file


RUN_ID = "20260717-theta-controlled-confirmation-v1"
RUN_DIR = ROOT / "artifacts" / "consonant-calibration" / RUN_ID
PARENT_DIR = (
    ROOT
    / "artifacts"
    / "consonant-calibration"
    / "20260717-consonant-carrier-search-v1"
)
ALLOSAURUS_BATCH = ROOT / "scripts" / "run_allosaurus_batch.py"
EXPECTED_TARGET_LABELS = ("t", "tʰ", "t̪")
CONTEXT_MODES = ("adjacent", "word")
SPLICE_CONTEXT_MS = 25.0


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


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


def _changed_and_target_columns(
    model: Any, neutral: str, lens: str
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    left = _filtered_symbols(model, neutral)
    right = _filtered_symbols(model, lens)
    if len(left) != len(right):
        raise RuntimeError("selected theta carrier changed model-token count")
    changed = tuple(
        index + 1
        for index, (source, target) in enumerate(zip(left, right, strict=True))
        if source != target
    )
    if len(changed) != 1 or left[changed[0] - 1] != "θ" or right[changed[0] - 1] != "t":
        raise RuntimeError("selected carrier is not an isolated theta-to-t change")
    return changed, changed


def main() -> None:
    if RUN_DIR.exists():
        raise RuntimeError(f"theta confirmation already exists: {RUN_DIR}")
    parent_path = PARENT_DIR / "records.json"
    parent = _load_json(parent_path)
    selected_ids = tuple(
        candidate_id
        for values in parent["first_two_dual_anchor_matches"]["enpt.theta_t"].values()
        for candidate_id in values
    )
    if selected_ids != (
        "enpt.theta_t__intervocalic__08",
        "enpt.theta_t__word_final__14",
        "enpt.theta_t__word_final__17",
        "enpt.theta_t__word_initial__05",
    ):
        raise RuntimeError("parent deterministic theta selection drifted")
    by_id = {row["candidate_id"]: row for row in parent["records"]}
    selected = [by_id[candidate_id] for candidate_id in selected_ids]
    protocol = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "classification": "theta_common_rng_controlled_confirmation",
        "api_calls_authorized": 0,
        "parent_records_sha256": sha256_file(parent_path),
        "selected_candidate_ids": selected_ids,
        "context_modes": CONTEXT_MODES,
        "candidate": (
            "target text state plus target-conditioned F0/noise over adjacent or full-word "
            "state; unchanged 25 ms output splice context and 10 ms taper"
        ),
        "automatic_interpretation": (
            "A mode is eligible for blind QC only if at least three of four controlled "
            "targets match the auxiliary /t/ labels and every engineering gate passes."
        ),
        "stopping": "Render each selected fixture once per mode; no replacements.",
    }
    protocol["protocol_sha256"] = hashlib.sha256(
        stable_json(protocol).encode()
    ).hexdigest()
    RUN_DIR.mkdir(parents=True)
    atomic_write_json(RUN_DIR / "protocol.json", protocol)

    runtime = _load_pinned_synthesis_voice("af_heart")
    records = []
    recognizer_inputs = []
    for parent_row in selected:
        neutral_phonemes = parent_row["source_phonemes"]
        lens_phonemes = parent_row["target_phonemes"]
        changed, target_columns = _changed_and_target_columns(
            runtime.model, neutral_phonemes, lens_phonemes
        )
        plan = PairPlan(
            source_phonemes=neutral_phonemes,
            neutral_phonemes=neutral_phonemes,
            lens_phonemes=lens_phonemes,
            target_word_indexes=(2,),
        )
        for mode in CONTEXT_MODES:
            rendered = render_controlled_listener_triplet(
                runtime,
                plan,
                consonant_columns=changed,
                consonant_context_mode=mode,
            )
            neutral = _pcm(rendered.neutral)
            identity = _pcm(rendered.identity)
            full_target = _pcm(rendered.full_lens)
            target_interval = decoder_column_interval(
                rendered.predicted_durations,
                target_columns,
                sample_count=neutral.size,
                sample_rate_hz=SAMPLE_RATE_HZ,
            )
            splice_interval = expanded_interval(
                target_interval,
                context_ms=SPLICE_CONTEXT_MS,
                sample_count=neutral.size,
            )
            window = splice_interval.as_record()
            candidate, weights = output_domain_splice(
                neutral, full_target, (window,)
            )
            boundary = boundary_artifact_report(
                neutral, full_target, candidate, (window,)
            )
            localization = localization_report(neutral, candidate, (window,))
            start = target_interval.start_sample
            end = target_interval.end_sample_exclusive
            checks = {
                "identity_bit_exact": bool(np.array_equal(neutral, identity)),
                "equal_nonempty": bool(
                    neutral.size
                    and neutral.size == identity.size == full_target.size == candidate.size
                ),
                "finite": all(
                    np.isfinite(values.astype(np.float64)).all()
                    for values in (neutral, identity, full_target, candidate)
                ),
                "outside_exact_neutral": bool(
                    np.array_equal(candidate[weights == 0.0], neutral[weights == 0.0])
                ),
                "target_interval_exact_controlled_target": bool(
                    np.array_equal(candidate[start:end], full_target[start:end])
                ),
                "boundary_pass": boundary.get("pass") is True,
                "localization_pass": localization.get("pass") is True,
            }
            audio = {}
            for role, values in (
                ("neutral", neutral),
                ("controlled_target", full_target),
                ("spliced_candidate", candidate),
            ):
                path = (
                    RUN_DIR
                    / "audio"
                    / f"{parent_row['candidate_id']}__{mode}__{role}.wav"
                )
                audio[role] = _write_wav(path, values)
                if role == "controlled_target":
                    recognizer_inputs.append(
                        {
                            "id": f"{parent_row['candidate_id']}::{mode}",
                            "path": str(path),
                        }
                    )
            records.append(
                {
                    "candidate_id": parent_row["candidate_id"],
                    "position": parent_row["position"],
                    "source_word": parent_row["source_word"],
                    "target_word": parent_row["target_word"],
                    "mode": mode,
                    "neutral_phonemes": neutral_phonemes,
                    "lens_phonemes": lens_phonemes,
                    "changed_columns": changed,
                    "target_interval": target_interval.as_record(),
                    "splice_window": window,
                    "consonant_excitation_frame_count": (
                        rendered.consonant_excitation_frame_count
                    ),
                    "engineering_checks": checks,
                    "engineering_pass": all(checks.values()),
                    "boundary": boundary,
                    "localization": localization,
                    "audio": audio,
                }
            )

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
    recognized = _load_json(output_path)
    if recognized["allosaurus_version"] != ALLOSAURUS_VERSION:
        raise RuntimeError("Allosaurus version drifted")
    outputs = {row["id"]: row["timestamp_output"] for row in recognized["rows"]}
    for record in records:
        rows = parse_allosaurus_timestamps(
            outputs[f"{record['candidate_id']}::{record['mode']}"]
        )
        interval = record["target_interval"]
        labels = overlapping_upr_labels(
            rows, start_s=interval["start_s"], end_s=interval["end_s"]
        )
        record["universal_phone_recognizer"] = {
            "all_timestamps": rows,
            "overlapping_labels": labels,
            "target_match": labels_support_expected(labels, EXPECTED_TARGET_LABELS),
        }
    summary = {
        mode: {
            "fixture_count": 4,
            "engineering_passes": sum(
                row["engineering_pass"] for row in records if row["mode"] == mode
            ),
            "controlled_target_matches": sum(
                row["universal_phone_recognizer"]["target_match"]
                for row in records
                if row["mode"] == mode
            ),
        }
        for mode in CONTEXT_MODES
    }
    for values in summary.values():
        values["eligible_for_blind_qc"] = bool(
            values["engineering_passes"] == 4
            and values["controlled_target_matches"] >= 3
        )
    payload = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "protocol_sha256": protocol["protocol_sha256"],
        "api_calls_made": 0,
        "records": records,
        "summary": summary,
        "classification": "theta_controlled_confirmation_complete",
    }
    payload["records_sha256"] = hashlib.sha256(
        stable_json(payload).encode()
    ).hexdigest()
    atomic_write_json(RUN_DIR / "records.json", payload)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
