from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import subprocess
import tempfile
import wave
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .config import Paths, sha256_json, stable_json
from .kokoro_cluster_anchor_calibration import (
    V4_RUN_ID as CALIBRATION_RUN_ID,
    run_dir_v4 as calibration_dir,
)
from .kokoro_cluster_shell import (
    CLUSTER_EXTRA_CONSONANTS,
    CLUSTER_LENS_SHELL,
    CLUSTER_NEUTRAL_SHELL,
    CLUSTER_SHELL_VERSION,
    ClusterShellPlanner,
)
from .kokoro_output_domain_splice import (
    _benchmark_localization,
    boundary_artifact_report,
    output_domain_splice,
)
from .kokoro_output_splice_unseen import (
    _read_pcm,
    _review_html as _shared_review_html,
    _word_intervals,
    phrase_medial_edge_gate,
)
from .kokoro_synthesis import (
    SAMPLE_RATE_HZ,
    PairRender,
    pcm16_bytes,
    target_word_columns,
)
from .kokoro_typed_confirmation import (
    _family_gate,
    _measure_occurrences,
    alignment_record,
)
from .kokoro_typed_confirmation_protocol import (
    CEILINGS_HZ,
    DESCRIPTIVE_WINDOW_PERCENTS,
    LOCALIZATION_MINIMUM,
    MEASUREMENT_SCRIPT,
    PRAAT,
    PRIMARY_WINDOW_PERCENT,
    WINDOW_PERCENTS,
)
from .kokoro_typed_diagnostic import localization_report
from .kokoro_typed_engine import (
    MAX_CLIPPED_FRACTION,
    inspect_render,
    local_engine_assets,
)
from .util import atomic_write_json, atomic_write_text, sha256_file


CONFIRMATION_VERSION = "kokoro-cluster-confirmation-v1"
RUN_ID = "20260718-kokoro-cluster-confirmation-v1"
STRICT_PARENT_RUN_ID = "20260717-kokoro-strict-shell-confirmation-v1"
PROTOCOL_FILE = "protocol.json"
RECORDS_FILE = "render-records.json"
ANALYSIS_FILE = "analysis.json"
REVIEW_FILE = "review.html"
REVIEW_MANIFEST_FILE = "review-manifest.json"
BLIND_KEY_FILE = "blind-key.json"
RESPONSE_FILENAME = "kokoro-cluster-confirmation-v1-response.json"
RAW_RESPONSE_FILE = RESPONSE_FILENAME
MANUAL_RESULT_FILE = "manual-result.json"
ATTEMPT_DIR = "attempts"
BLIND_SEED = 20_260_718_07

# The three unseen inventories are frozen in first-eligible order. Every
# sentence carries exactly the shape the cluster planner supports: a stressed
# /ae/ monosyllable with a single-consonant onset and a consonant-cluster coda.
MEDIAL_INVENTORY = (
    "They kept the task hidden well.",
    "She wrote the fact down twice.",
    "He held the mask near the light.",
    "We watched the camp from above.",
)
FINAL_INVENTORY = (
    "The song was about the past.",
    "She reached out her hand.",
    "The workers built the camp.",
    "He memorized every fact.",
)
REPEATED_INVENTORY = (
    "The camp guards the camp.",
    "The mask hides the mask.",
    "That fact repeats the fact.",
    "The task follows the task.",
)

FIXTURE_SPECS = (
    ("phrase-medial-cluster", "medial", MEDIAL_INVENTORY, 1),
    ("phrase-final-cluster", "final", FINAL_INVENTORY, 1),
    ("repeated-cluster", "repeated", REPEATED_INVENTORY, 2),
)

# Paths that must be committed and clean before any decode; the render
# records bind the exact commit.
TRACKED_CLEAN_PATHS = (
    "src/earshift_bakeoff/kokoro_cluster_anchor_calibration.py",
    "src/earshift_bakeoff/kokoro_cluster_confirmation.py",
    "src/earshift_bakeoff/kokoro_cluster_shell.py",
    "src/earshift_bakeoff/kokoro_output_domain_splice.py",
    "src/earshift_bakeoff/kokoro_output_splice_unseen.py",
    "src/earshift_bakeoff/kokoro_synthesis.py",
    "src/earshift_bakeoff/kokoro_typed_confirmation.py",
    "src/earshift_bakeoff/kokoro_typed_engine.py",
    "scripts/run_kokoro_cluster_confirmation_v1.py",
    "scripts/praat_sentence_pair_v2_burg.praat",
    "uv.lock",
    f"artifacts/typed-engine/{RUN_ID}/{PROTOCOL_FILE}",
    f"artifacts/typed-engine/{CALIBRATION_RUN_ID}/protocol.json",
    f"artifacts/typed-engine/{CALIBRATION_RUN_ID}/analysis.json",
    f"artifacts/typed-engine/{STRICT_PARENT_RUN_ID}/protocol.json",
    f"artifacts/typed-engine/{STRICT_PARENT_RUN_ID}/analysis.json",
)


def run_dir() -> Path:
    return Paths().artifacts / "typed-engine" / RUN_ID


def _load_json(path: Path) -> Any:
    if not path.is_file():
        raise RuntimeError(f"required frozen artifact is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _verify_internal_hash(
    record: dict[str, Any], digest_key: str, *, label: str
) -> None:
    expected = record.get(digest_key)
    if not isinstance(expected, str) or len(expected) != 64:
        raise RuntimeError(f"{label} is missing {digest_key}")
    payload = {key: value for key, value in record.items() if key != digest_key}
    if sha256_json(payload) != expected:
        raise RuntimeError(f"{label} has an invalid {digest_key}")


def _verified_calibration_parent() -> dict[str, Any]:
    protocol = _load_json(calibration_dir() / "protocol.json")
    analysis = _load_json(calibration_dir() / "analysis.json")
    _verify_internal_hash(protocol, "protocol_sha256", label="calibration protocol")
    _verify_internal_hash(analysis, "analysis_sha256", label="calibration analysis")
    if analysis["classification"] != "cluster_anchor_calibration_v4_pass":
        raise RuntimeError("cluster confirmation requires the v4 calibration pass")
    endpoints = analysis["endpoints_by_extra"]
    if set(endpoints) != set(CLUSTER_EXTRA_CONSONANTS) or any(
        set(rows) != {str(ceiling) for ceiling in CEILINGS_HZ}
        for rows in endpoints.values()
    ):
        raise RuntimeError("calibration endpoints do not cover every extra")
    return {
        "run_id": CALIBRATION_RUN_ID,
        "protocol_sha256": protocol["protocol_sha256"],
        "analysis_sha256": analysis["analysis_sha256"],
        "classification": analysis["classification"],
        "endpoints_by_extra": endpoints,
    }


def _verified_strict_parent() -> dict[str, Any]:
    base = Paths().artifacts / "typed-engine" / STRICT_PARENT_RUN_ID
    protocol = _load_json(base / "protocol.json")
    analysis = _load_json(base / "analysis.json")
    if not analysis.get("automatic_pass"):
        raise RuntimeError("strict-shell parent is not an automatic pass")
    return {
        "run_id": STRICT_PARENT_RUN_ID,
        "protocol_sha256": protocol["protocol_sha256"],
        "analysis_sha256": sha256_file(base / "analysis.json"),
        "classification": analysis["classification"],
    }


def _cluster_target_reasons(plan: Any, expected_occurrences: int) -> list[str]:
    reasons: list[str] = []
    if plan.target_occurrence_count != expected_occurrences:
        reasons.append("wrong_target_occurrence_count")
    if len(plan.target_word_indexes) != expected_occurrences:
        reasons.append("wrong_target_word_count")
    for word_index in plan.target_word_indexes:
        word = plan.words[word_index]
        neutral = word.neutral_phone
        lens = word.lens_phone
        if not neutral.startswith(CLUSTER_NEUTRAL_SHELL):
            reasons.append("neutral_not_cluster_shell")
        if not lens.startswith(CLUSTER_LENS_SHELL):
            reasons.append("lens_not_cluster_shell")
        extras = neutral[len(CLUSTER_NEUTRAL_SHELL) :]
        if not extras or any(
            symbol not in CLUSTER_EXTRA_CONSONANTS for symbol in extras
        ):
            reasons.append("extras_outside_preserving_pool")
        if lens[len(CLUSTER_LENS_SHELL) :] != extras:
            reasons.append("lens_extras_differ")
    return reasons


def _role_reasons(plan: Any, role: str) -> list[str]:
    reasons: list[str] = []
    indexes = plan.target_word_indexes
    last_word = len(plan.words) - 1
    if role == "medial":
        if len(indexes) != 1 or not (0 < indexes[0] < last_word):
            reasons.append("target_not_phrase_medial")
    elif role == "final":
        if len(indexes) != 1 or indexes[0] != last_word:
            reasons.append("target_not_phrase_final")
    else:
        if len(indexes) != 2:
            reasons.append("target_not_repeated")
        elif any(
            (plan.words[i].neutral_phone, plan.words[i].lens_phone)
            != (
                plan.words[indexes[0]].neutral_phone,
                plan.words[indexes[0]].lens_phone,
            )
            for i in indexes
        ):
            reasons.append("repeated_carriers_differ")
    return reasons


def _select_fixtures() -> list[dict[str, Any]]:
    planner = ClusterShellPlanner.load()
    prior = [
        path.read_bytes()
        for path in sorted(Paths().artifacts.rglob("*.json"))
        if not path.is_relative_to(run_dir())
    ]
    fixtures: list[dict[str, Any]] = []
    for fixture_id, role, inventory, expected_occurrences in FIXTURE_SPECS:
        attempts: list[dict[str, Any]] = []
        selected: dict[str, Any] | None = None
        for order, text in enumerate(inventory, start=1):
            try:
                first = planner.plan(text)
                second = planner.plan(text)
            except Exception as exc:
                attempts.append(
                    {
                        "order": order,
                        "text": text,
                        "pass": False,
                        "reasons": [f"{type(exc).__name__}: {exc}"],
                    }
                )
                continue
            reasons: list[str] = []
            if stable_json(asdict(first)) != stable_json(asdict(second)):
                reasons.append("plan_not_deterministic")
            if not (
                first.gate_summary.espeak_gate_pass
                and first.gate_summary.kokoro_phone_gate_pass
            ):
                reasons.append("carrier_gate_failed")
            reasons.extend(_cluster_target_reasons(first, expected_occurrences))
            reasons.extend(_role_reasons(first, role))
            if any(text.encode() in payload for payload in prior):
                reasons.append("text_seen_in_prior_artifacts")
            if any(first.plan_sha256.encode() in payload for payload in prior):
                reasons.append("plan_seen_in_prior_artifacts")
            attempts.append(
                {
                    "order": order,
                    "text": text,
                    "plan_sha256": first.plan_sha256,
                    "pass": not reasons,
                    "reasons": reasons,
                }
            )
            if not reasons:
                target_extras = [
                    first.words[index].neutral_phone[len(CLUSTER_NEUTRAL_SHELL) :]
                    for index in first.target_word_indexes
                ]
                selected = {
                    "fixture_id": fixture_id,
                    "role": role,
                    "inventory": list(inventory),
                    "attempts": attempts,
                    "selected_order": order,
                    "text": text,
                    "plan_sha256": first.plan_sha256,
                    "source_phonemes": first.source_phonemes,
                    "neutral_phonemes": first.neutral_phonemes,
                    "lens_phonemes": first.lens_phonemes,
                    "target_word_indexes": list(first.target_word_indexes),
                    "target_occurrence_count": first.target_occurrence_count,
                    "target_extras": target_extras,
                    "words": [asdict(word) for word in first.words],
                    "gate_summary": asdict(first.gate_summary),
                }
                break
        if selected is None:
            raise RuntimeError(
                f"no gate-clean {fixture_id} cluster fixture in inventory"
            )
        fixtures.append(selected)
    return fixtures


def protocol_record() -> dict[str, Any]:
    calibration = _verified_calibration_parent()
    strict = _verified_strict_parent()
    fixtures = _select_fixtures()
    payload: dict[str, Any] = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "version": CONFIRMATION_VERSION,
        "status": "frozen_before_any_render",
        "purpose": (
            "One unseen confirmation of the voiceless cluster shell across "
            "medial, final, and repeated real-word fixtures, judged per extra "
            "against the frozen v4 context-matched endpoints."
        ),
        "parents": {
            "calibration_v4": {
                key: calibration[key]
                for key in (
                    "run_id",
                    "protocol_sha256",
                    "analysis_sha256",
                    "classification",
                )
            },
            "strict_shell_confirmation": strict,
        },
        "cluster_shell_version": CLUSTER_SHELL_VERSION,
        "engine_assets": local_engine_assets(),
        "endpoints_by_extra": calibration["endpoints_by_extra"],
        "fixtures": fixtures,
        "measurement": {
            "praat_script_sha256": sha256_file(MEASUREMENT_SCRIPT),
            "ceilings_hz": list(CEILINGS_HZ),
            "primary_window_percent": PRIMARY_WINDOW_PERCENT,
            "descriptive_window_percents": list(DESCRIPTIVE_WINDOW_PERCENTS),
            "anchor_selection": "each occurrence uses its own extra's endpoints",
        },
        "gates": {
            "runtime": (
                "neutral/identity PCM bit-identity, equal sample counts, "
                "finite unclipped audio, exact neutral outside splice windows, "
                "exact lens in full-weight interiors, boundary metrics"
            ),
            "localization_minimum": LOCALIZATION_MINIMUM,
            "acoustic": (
                "per occurrence and family: neutral nearer its extra's ae "
                "endpoint, lens nearer its extra's eh endpoint, direction "
                "cosine at least 0.50 against the extra's expected vector, "
                "magnitude at least max(0.25, half the extra's endpoint "
                "separation); primary 50% window decides, 40/60 descriptive"
            ),
            "conjunctive": True,
            "one_attempt_per_slot": True,
        },
        "predetermined_outcomes": {
            "automatic_pass": (
                "Open one blind creator QC session (controls plus "
                "neutral/spliced-lens per fixture, frozen manual gates); no "
                "enablement from the automatic result alone."
            ),
            "any_automatic_fail": (
                "Record and freeze the failure, diagnose the mechanism, and "
                "design the smallest separately versioned correction; the "
                "cluster path does not close and nothing is reclassified."
            ),
        },
        "scope": {
            "api_calls": 0,
            "logical_render_pairs": sum(
                fixture["target_occurrence_count"] for fixture in fixtures
            ),
            "production_enabled": False,
        },
    }
    return {**payload, "protocol_sha256": sha256_json(payload)}


def prepare() -> dict[str, Any]:
    protocol = protocol_record()
    destination = run_dir() / PROTOCOL_FILE
    if destination.is_file():
        existing = _load_json(destination)
        if stable_json(existing) != stable_json(protocol):
            raise RuntimeError("existing cluster confirmation protocol differs")
        return existing
    if any(
        (run_dir() / name).exists()
        for name in (RECORDS_FILE, ANALYSIS_FILE, "audio", ATTEMPT_DIR)
    ):
        raise RuntimeError("cluster confirmation output exists before protocol")
    destination.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(destination, protocol)
    return protocol


def _checked_protocol() -> dict[str, Any]:
    frozen = _load_json(run_dir() / PROTOCOL_FILE)
    if stable_json(frozen) != stable_json(protocol_record()):
        raise RuntimeError("cluster confirmation protocol or its inputs drifted")
    return frozen


def _require_commit() -> str:
    subprocess.run(
        ["git", "ls-files", "--error-unmatch", "--", *TRACKED_CLEAN_PATHS],
        cwd=Paths().root,
        check=True,
        capture_output=True,
    )
    if subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--", *TRACKED_CLEAN_PATHS],
        cwd=Paths().root,
    ).returncode:
        raise RuntimeError("cluster confirmation inputs differ from committed HEAD")
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Paths().root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _write_once_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        if stable_json(_load_json(path)) != stable_json(payload):
            raise RuntimeError(f"immutable artifact differs: {path}")
        return
    atomic_write_json(path, payload)


def _write_once_bytes(path: Path, payload: bytes) -> None:
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError(f"immutable bytes differ: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _write_wav(path: Path, values: np.ndarray) -> None:
    if path.exists():
        raise RuntimeError(f"one-attempt WAV already exists: {path}")
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


def _audio(path: Path, values: np.ndarray) -> dict[str, Any]:
    pcm = np.asarray(values, dtype="<i2").reshape(-1)
    clipped = float(np.mean(np.abs(pcm.astype(np.int64)) >= 32767))
    return {
        "relative_path": str(path.relative_to(run_dir())),
        "sample_count": int(pcm.size),
        "finite": bool(pcm.size and np.isfinite(pcm.astype(float)).all()),
        "clipped_fraction": clipped,
        "clipping_pass": clipped < MAX_CLIPPED_FRACTION,
        "pcm_sha256": hashlib.sha256(pcm.tobytes()).hexdigest(),
        "wav_sha256": sha256_file(path),
    }


def _begin(records: dict[str, Any], fixture_id: str) -> list[dict[str, Any]]:
    slots = [row for row in records["slots"] if row["fixture_id"] == fixture_id]
    if any(
        (run_dir() / ATTEMPT_DIR / f"{row['slot_id']}.json").exists()
        for row in slots
    ):
        raise RuntimeError("cluster fixture was already attempted")
    for slot in slots:
        atomic_write_json(
            run_dir() / ATTEMPT_DIR / f"{slot['slot_id']}.json",
            {
                "run_id": RUN_ID,
                "protocol_sha256": records["protocol_sha256"],
                "slot_id": slot["slot_id"],
                "one_attempt_no_retry": True,
            },
        )
        slot["status"] = "attempt_started"
    atomic_write_json(run_dir() / RECORDS_FILE, records)
    return slots


def _render(protocol: dict[str, Any], records: dict[str, Any]) -> None:
    from .kokoro_synthesis import KokoroSynthesisRuntime

    planner = ClusterShellPlanner.load()
    runtime = KokoroSynthesisRuntime.load(download=False)
    endpoints = protocol["endpoints_by_extra"]
    for frozen in protocol["fixtures"]:
        slots = _begin(records, frozen["fixture_id"])
        try:
            plan = planner.plan(frozen["text"])
            if plan.plan_sha256 != frozen["plan_sha256"]:
                raise RuntimeError("cluster plan drifted")
            pair = plan.pair_plan()
            if pair is None:
                raise RuntimeError("cluster fixture lost its pair")
            rendered = runtime.render_parity_triplet(pair)
            columns = target_word_columns(
                runtime.model, plan.neutral_phonemes, plan.target_word_indexes
            )
            if rendered.replaced_columns != columns:
                raise RuntimeError("cluster replaced columns drifted")
            neutral = np.frombuffer(pcm16_bytes(rendered.neutral), dtype="<i2").copy()
            identity = np.frombuffer(
                pcm16_bytes(rendered.identity), dtype="<i2"
            ).copy()
            full_lens = np.frombuffer(pcm16_bytes(rendered.lens), dtype="<i2").copy()
            last_word = len(plan.words) - 1
            anchor_map = [
                1 if word_index == last_word else 0
                for word_index in plan.target_word_indexes
            ]
            alignment = alignment_record(
                model=runtime.model,
                plan=plan,
                durations=rendered.predicted_durations,
                sample_count=neutral.size,
                anchor_occurrence_map=anchor_map,
            )
            extras_by_occurrence = []
            for occurrence in alignment["target_occurrences"]:
                word = plan.words[occurrence["word_index"]]
                extras = word.neutral_phone[len(CLUSTER_NEUTRAL_SHELL) :]
                if not extras or extras[0] not in endpoints:
                    raise RuntimeError(
                        "cluster occurrence extra lacks a calibrated endpoint"
                    )
                extras_by_occurrence.append(extras)
            target_intervals = [row["interval"] for row in alignment["target_words"]]
            baseline = localization_report(neutral, full_lens, target_intervals)
            windows = baseline["inside_windows"]
            lens, weights = output_domain_splice(neutral, full_lens, windows)
            all_words = _word_intervals(
                runtime.model, plan, rendered.predicted_durations, neutral.size
            )
            edge = (
                phrase_medial_edge_gate(
                    plan.target_word_indexes[0], all_words, windows[0]
                )
                if frozen["fixture_id"] == "phrase-medial-cluster"
                else {"pass": True, "reason": "not_medial_fixture"}
            )
            values = {
                "neutral": neutral,
                "identity": identity,
                "full-state-lens-source": full_lens,
                "lens": lens,
            }
            audio: dict[str, Any] = {}
            for role, value in values.items():
                path = run_dir() / "audio" / f"{frozen['fixture_id']}__{role}.wav"
                _write_wav(path, value)
                audio[role] = _audio(path, value)
            raw_integrity = inspect_render(
                PairRender(
                    neutral=rendered.neutral,
                    lens=rendered.lens,
                    predicted_durations=rendered.predicted_durations,
                    replaced_columns=rendered.replaced_columns,
                )
            )
            checks = {
                "exact_plan": plan.plan_sha256 == frozen["plan_sha256"],
                "exact_columns": rendered.replaced_columns == columns,
                "raw_integrity": raw_integrity.pass_all,
                "identity_bit_exact": np.array_equal(neutral, identity),
                "equal_samples": len({value.size for value in values.values()}) == 1,
                "finite_unclipped": all(
                    row["finite"] and row["clipping_pass"] for row in audio.values()
                ),
                "outside_exact_neutral": np.array_equal(
                    lens[weights == 0.0], neutral[weights == 0.0]
                ),
                "interior_exact_full_lens": bool(
                    np.any(weights == 1.0)
                    and np.array_equal(
                        lens[weights == 1.0], full_lens[weights == 1.0]
                    )
                ),
            }
            records["fixtures"].append(
                {
                    "fixture_id": frozen["fixture_id"],
                    "plan_sha256": plan.plan_sha256,
                    "safe_plan_metadata": plan.safe_metadata(),
                    "predicted_durations": list(rendered.predicted_durations),
                    "alignment": alignment,
                    "target_extras_by_occurrence": extras_by_occurrence,
                    "all_word_intervals": all_words,
                    "splice_windows": windows,
                    "phrase_medial_edge_gate": edge,
                    "untouched_full_state_localization": baseline,
                    "audio": audio,
                    "runtime_checks": checks,
                    "runtime_pass": all(checks.values()),
                }
            )
            for slot in slots:
                slot["status"] = "complete"
                slot["audio"] = audio[slot["role"]]
            records["decoder_attempt_count"] = sum(
                row["status"] == "complete" for row in records["slots"]
            )
            atomic_write_json(run_dir() / RECORDS_FILE, records)
        except Exception as exc:
            for slot in slots:
                slot["status"] = "failed_no_retry"
                slot["failure"] = f"{type(exc).__name__}: {exc}"[:1000]
            for slot in records["slots"]:
                if slot["status"] == "pending":
                    slot["status"] = "not_reached"
            records["status"] = "runtime_failure_no_retry"
            atomic_write_json(run_dir() / RECORDS_FILE, records)
            raise
    records["status"] = "render_complete"
    atomic_write_json(run_dir() / RECORDS_FILE, records)


def _cluster_acoustic_report(
    neutral_path: Path,
    lens_path: Path,
    occurrences: Sequence[dict[str, Any]],
    occurrence_extras: Sequence[str],
    endpoints_by_extra: dict[str, Any],
) -> dict[str, Any]:
    if len(occurrence_extras) != len(occurrences):
        raise RuntimeError("occurrence extras do not cover every occurrence")
    neutral_measurements = _measure_occurrences(neutral_path, occurrences)
    lens_measurements = _measure_occurrences(lens_path, occurrences)
    windows: dict[str, Any] = {}
    for percent in WINDOW_PERCENTS:
        key = str(percent)
        rows: list[dict[str, Any]] = []
        for index in range(len(occurrences)):
            extra = occurrence_extras[index][0]
            families: dict[str, Any] = {}
            for ceiling in CEILINGS_HZ:
                ceiling_key = str(ceiling)
                families[ceiling_key] = _family_gate(
                    neutral_measurements[index]["families"][ceiling_key][key],
                    lens_measurements[index]["families"][ceiling_key][key],
                    endpoints_by_extra[extra][ceiling_key],
                )
            rows.append(
                {
                    "occurrence_index": index,
                    "extra_consonant": extra,
                    "coda_extras": occurrence_extras[index],
                    "families": families,
                    "pass": all(row["pass"] for row in families.values()),
                }
            )
        windows[key] = {
            "occurrences": rows,
            "pass": all(row["pass"] for row in rows),
        }

    def signature(percent: int) -> list[Any]:
        row = windows[str(percent)]
        return [
            family["checks"]
            for occurrence in row["occurrences"]
            for family in occurrence["families"].values()
        ] + [{"overall_pass": row["pass"]}]

    sensitivity = {
        str(percent): signature(percent) != signature(PRIMARY_WINDOW_PERCENT)
        for percent in DESCRIPTIVE_WINDOW_PERCENTS
    }
    return {
        "neutral_measurements": neutral_measurements,
        "lens_measurements": lens_measurements,
        "windows": windows,
        "primary_window_percent": PRIMARY_WINDOW_PERCENT,
        "primary_gate_pass": windows[str(PRIMARY_WINDOW_PERCENT)]["pass"],
        "descriptive_window_sensitivity": sensitivity,
        "window_sensitive": any(sensitivity.values()),
        "descriptive_windows_change_outcome": False,
    }


def _analyze_fixture(
    record: dict[str, Any], protocol: dict[str, Any]
) -> dict[str, Any]:
    paths = {
        role: run_dir() / row["relative_path"]
        for role, row in record["audio"].items()
    }
    for role, path in paths.items():
        if sha256_file(path) != record["audio"][role]["wav_sha256"]:
            raise RuntimeError(f"cluster audio hash drifted: {role}")
    neutral = _read_pcm(paths["neutral"])
    identity = _read_pcm(paths["identity"])
    full_lens = _read_pcm(paths["full-state-lens-source"])
    lens = _read_pcm(paths["lens"])
    _, weights = output_domain_splice(neutral, full_lens, record["splice_windows"])
    integrity = {
        **record["runtime_checks"],
        "identity_recheck": np.array_equal(neutral, identity),
        "outside_recheck": np.array_equal(
            lens[weights == 0.0], neutral[weights == 0.0]
        ),
        "interior_recheck": bool(
            np.any(weights == 1.0)
            and np.array_equal(lens[weights == 1.0], full_lens[weights == 1.0])
        ),
    }
    targets = [row["interval"] for row in record["alignment"]["target_words"]]
    boundary = boundary_artifact_report(
        neutral, full_lens, lens, record["splice_windows"]
    )
    localization = localization_report(neutral, lens, targets)
    benchmark = _benchmark_localization(neutral, lens, targets)
    acoustic = _cluster_acoustic_report(
        paths["neutral"],
        paths["lens"],
        record["alignment"]["target_occurrences"],
        record["target_extras_by_occurrence"],
        protocol["endpoints_by_extra"],
    )
    checks = {
        "runtime_and_pcm_integrity": all(integrity.values()),
        "medial_edge_gate": bool(record["phrase_medial_edge_gate"]["pass"]),
        "boundary_gate": bool(boundary["pass"]),
        "primary_50_acoustic_gate": bool(acoustic["primary_gate_pass"]),
        "localization_gate": bool(localization["pass"]),
        "localization_runtime_gate": bool(benchmark["pass"]),
    }
    return {
        "fixture_id": record["fixture_id"],
        "integrity_checks": integrity,
        "phrase_medial_edge_gate": record["phrase_medial_edge_gate"],
        "boundary_artifact": boundary,
        "acoustic": acoustic,
        "untouched_full_state_localization": record[
            "untouched_full_state_localization"
        ],
        "spliced_localization": {
            **localization,
            "expected_by_construction": True,
        },
        "localization_runtime_benchmark": benchmark,
        "automatic_checks": checks,
        "automatic_pass": all(checks.values()),
    }


def _analysis(protocol: dict[str, Any], records: dict[str, Any]) -> dict[str, Any]:
    fixtures: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for record in records["fixtures"]:
        try:
            fixtures.append(_analyze_fixture(record, protocol))
        except Exception as exc:
            failures.append(
                {
                    "fixture_id": record["fixture_id"],
                    "failure": f"{type(exc).__name__}: {exc}"[:1000],
                }
            )
    passed = bool(
        not failures
        and len(fixtures) == len(FIXTURE_SPECS)
        and all(row["automatic_pass"] for row in fixtures)
    )
    payload = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "protocol_sha256": protocol["protocol_sha256"],
        "classification": (
            "cluster_shell_aggregate_automatic_pass_pending_human_qc"
            if passed
            else (
                "cluster_shell_measurement_inconclusive"
                if failures
                else "cluster_shell_aggregate_automatic_failed"
            )
        ),
        "automatic_pass": passed,
        "pending_human_review": passed,
        "fixtures": fixtures,
        "measurement_failures": failures,
        "calibration_parent_preserved": "cluster_anchor_calibration_v4_pass",
        "descriptive_windows_do_not_change_outcome": True,
        "api_calls_made": 0,
        "decoder_attempt_count": records["decoder_attempt_count"],
        "render_records_sha256": sha256_file(run_dir() / RECORDS_FILE),
        "production_enabled": False,
    }
    return {**payload, "analysis_sha256": sha256_json(payload)}


def _layout() -> list[dict[str, Any]]:
    trials = [
        *(
            {
                "fixture_id": fixture_id,
                "condition": "identity-control",
                "roles": ["neutral", "identity"],
            }
            for fixture_id, _, _, _ in FIXTURE_SPECS
        ),
        *(
            {
                "fixture_id": fixture_id,
                "condition": "spliced-lens",
                "roles": ["neutral", "lens"],
            }
            for fixture_id, _, _, _ in FIXTURE_SPECS
        ),
    ]
    rng = random.Random(BLIND_SEED)
    for trial in trials:
        roles = trial.pop("roles")
        rng.shuffle(roles)
        trial["side_roles"] = dict(zip(("A", "B"), roles, strict=True))
    rng.shuffle(trials)
    return [
        {**trial, "trial_id": f"comparison-{index:02d}"}
        for index, trial in enumerate(trials, start=1)
    ]


def _copy_review_audio(
    layout: Sequence[dict[str, Any]], records: dict[str, Any]
) -> list[dict[str, Any]]:
    by_fixture = {row["fixture_id"]: row for row in records["fixtures"]}
    destination = run_dir() / "review-audio"
    destination.mkdir(parents=True, exist_ok=True)
    public: list[dict[str, Any]] = []
    for trial in layout:
        record = by_fixture[trial["fixture_id"]]
        sides: list[dict[str, str]] = []
        for side, role in trial["side_roles"].items():
            source = run_dir() / record["audio"][role]["relative_path"]
            target = destination / f"{trial['trial_id'][-2:]}-{side.lower()}.wav"
            if target.exists():
                if sha256_file(target) != sha256_file(source):
                    raise RuntimeError("cluster blind copy drifted")
            else:
                shutil.copyfile(source, target)
            sides.append({"side": side, "audio": f"review-audio/{target.name}"})
        public.append(
            {
                "trial_id": trial["trial_id"],
                "duration_s": record["audio"]["neutral"]["sample_count"]
                / SAMPLE_RATE_HZ,
                "target_intervals": [
                    {
                        "start_s": row["interval"]["start_s"],
                        "end_s": row["interval"]["end_s"],
                    }
                    for row in record["alignment"]["target_words"]
                ],
                "sides": sides,
            }
        )
    return public


def build_review(protocol: dict[str, Any], analysis: dict[str, Any]) -> dict[str, Any]:
    if analysis.get("automatic_pass") is not True:
        raise RuntimeError("cluster aggregate did not authorize review")
    layout = _layout()
    records = _load_json(run_dir() / RECORDS_FILE)
    public = _copy_review_audio(layout, records)
    key = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "protocol_sha256": protocol["protocol_sha256"],
        "trials": [
            {
                **trial,
                "expected_lens_side": next(
                    (
                        side
                        for side, role in trial["side_roles"].items()
                        if role == "lens"
                    ),
                    None,
                ),
            }
            for trial in layout
        ],
    }
    manifest = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "protocol_sha256": protocol["protocol_sha256"],
        "status": "pending_human_review",
        "trial_count": len(public),
        "response_filename": RESPONSE_FILENAME,
        "public_trials": public,
        "hidden_fields_absent": True,
    }
    _write_once_json(run_dir() / BLIND_KEY_FILE, key)
    _write_once_json(run_dir() / REVIEW_MANIFEST_FILE, manifest)
    html = _shared_review_html(public, protocol["protocol_sha256"])
    html = html.replace("20260717-kokoro-output-splice-unseen-v1", RUN_ID).replace(
        "kokoro-output-splice-unseen-v1-response.json", RESPONSE_FILENAME
    )
    path = run_dir() / REVIEW_FILE
    if path.exists() and path.read_text(encoding="utf-8") != html:
        raise RuntimeError("cluster review HTML drifted")
    if not path.exists():
        atomic_write_text(path, html)
    return manifest


def run() -> dict[str, Any]:
    analysis_path = run_dir() / ANALYSIS_FILE
    if analysis_path.exists():
        analysis = _load_json(analysis_path)
        if analysis.get("automatic_pass") and not (run_dir() / REVIEW_FILE).exists():
            build_review(_checked_protocol(), analysis)
        return analysis
    protocol = _checked_protocol()
    measurement = protocol["measurement"]
    if sha256_file(MEASUREMENT_SCRIPT) != measurement["praat_script_sha256"]:
        raise RuntimeError("measurement script drifted")
    if not Path(PRAAT).exists():
        raise RuntimeError("Praat binary is missing")
    commit = _require_commit()
    records = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "protocol_sha256": protocol["protocol_sha256"],
        "implementation_commit": commit,
        "status": "in_progress",
        "api_calls_made": 0,
        "decoder_attempt_count": 0,
        "slots": [
            {
                "order": order,
                "slot_id": f"{fixture['fixture_id']}__{role}",
                "fixture_id": fixture["fixture_id"],
                "role": role,
                "plan_sha256": fixture["plan_sha256"],
                "one_attempt_no_retry": True,
                "status": "pending",
            }
            for order, (fixture, role) in enumerate(
                (
                    (fixture, role)
                    for fixture in protocol["fixtures"]
                    for role in ("neutral", "identity", "full-state-lens-source")
                ),
                start=1,
            )
        ],
        "fixtures": [],
    }
    atomic_write_json(run_dir() / RECORDS_FILE, records)
    try:
        _render(protocol, records)
    except Exception as exc:
        result = {
            "schema_version": 1,
            "run_id": RUN_ID,
            "protocol_sha256": protocol["protocol_sha256"],
            "classification": "cluster_shell_runtime_inconclusive",
            "automatic_pass": False,
            "pending_human_review": False,
            "failure": f"{type(exc).__name__}: {exc}"[:1000],
            "render_records_sha256": sha256_file(run_dir() / RECORDS_FILE),
        }
        _write_once_json(analysis_path, result)
        return result
    records = _load_json(run_dir() / RECORDS_FILE)
    analysis = _analysis(protocol, records)
    _write_once_json(analysis_path, analysis)
    if analysis["automatic_pass"]:
        build_review(protocol, analysis)
    return analysis


def _side_gate(side: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "naturalness_at_least_4": int(side["naturalness"]) >= 4,
        "sentence_like": side["delivery"] == "sentence-like",
        "no_stable_meaning": side["meaning"] == "none",
        "no_major_artifact": side["artifact"] in {"none", "minor"},
    }
    return {"checks": checks, "pass": all(checks.values())}


def decode_response(path: Path) -> dict[str, Any]:
    protocol = _checked_protocol()
    if path.name != RESPONSE_FILENAME:
        raise RuntimeError(f"response filename must be {RESPONSE_FILENAME}")
    raw = path.read_bytes()
    response = json.loads(raw)
    if response.get("run_id") != RUN_ID or response.get(
        "protocol_sha256"
    ) != protocol["protocol_sha256"]:
        raise RuntimeError("cluster response belongs to another run")
    keys = {
        row["trial_id"]: row
        for row in _load_json(run_dir() / BLIND_KEY_FILE)["trials"]
    }
    rows = response.get("responses")
    if not isinstance(rows, list) or {
        row.get("trial_id") for row in rows
    } != set(keys):
        raise RuntimeError("cluster response trial set is incomplete")
    decoded: list[dict[str, Any]] = []
    fixture_results: dict[str, dict[str, bool]] = {
        fixture_id: {} for fixture_id, _, _, _ in FIXTURE_SPECS
    }
    for row in rows:
        key = keys[row["trial_id"]]
        side_gates = {side: _side_gate(row["sides"][side]) for side in ("A", "B")}
        if key["condition"] == "spliced-lens":
            pair = {
                "strength_at_least_5": int(row["difference_strength"]) >= 5,
                "correct_direction": row["category_judgment"]
                == key["expected_lens_side"],
                "confidence_at_least_3": int(row["confidence"]) >= 3,
                "no_dominant_interference": row["interference"]
                in {"none", "manageable"},
            }
        else:
            pair = {
                "identity_strength_1": int(row["difference_strength"]) == 1,
                "identity_direction_clean": row["category_judgment"]
                in {"same", "neither"},
                "no_dominant_interference": row["interference"]
                in {"none", "manageable"},
            }
        passed = bool(
            all(value["pass"] for value in side_gates.values())
            and all(pair.values())
        )
        fixture_results[key["fixture_id"]][key["condition"]] = passed
        decoded.append(
            {
                "trial_id": row["trial_id"],
                "fixture_id": key["fixture_id"],
                "condition": key["condition"],
                "side_gates": side_gates,
                "pair_checks": pair,
                "pass": passed,
                "replay_count": row.get("replay_count"),
                "notes": row.get("notes", ""),
            }
        )
    fixture_pass = {
        fixture: bool(values.get("identity-control") and values.get("spliced-lens"))
        for fixture, values in fixture_results.items()
    }
    passed = all(fixture_pass.values())
    result = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "protocol_sha256": protocol["protocol_sha256"],
        "classification": (
            "cluster_shell_human_qc_pass"
            if passed
            else "cluster_shell_human_qc_failed"
        ),
        "run_pass": passed,
        "fixture_pass": fixture_pass,
        "decoded_trials": decoded,
        "raw_response_sha256": hashlib.sha256(raw).hexdigest(),
        "production_enabled": False,
    }
    _write_once_bytes(run_dir() / RAW_RESPONSE_FILE, raw)
    _write_once_json(run_dir() / MANUAL_RESULT_FILE, result)
    return result
