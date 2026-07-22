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
from typing import Any

import numpy as np

from .config import Paths, sha256_json, stable_json
from .kokoro_output_domain_splice import (
    _benchmark_localization,
    boundary_artifact_report,
    output_domain_splice,
)
from .kokoro_output_splice_unseen import (
    _acoustic_report,
    _read_pcm,
    _review_html as _shared_review_html,
    _word_intervals,
    phrase_medial_edge_gate,
)
from .kokoro_strict_shell import STRICT_SHELL_VERSION, StrictShellPlanner
from .kokoro_synthesis import (
    KOKORO_VERSION,
    MODEL_HASHES,
    MODEL_REPO,
    MODEL_REVISION,
    RNG_SEED,
    SAMPLE_RATE_HZ,
    SPEED,
    PairRender,
    pcm16_bytes,
    target_word_columns,
)
from .kokoro_typed_confirmation import alignment_record
from .kokoro_typed_confirmation_protocol import (
    CEILINGS_HZ,
    DESCRIPTIVE_WINDOW_PERCENTS,
    MEASUREMENT_SCRIPT,
    PRAAT,
    PRIMARY_WINDOW_PERCENT,
    _verified_diagnostic_parent,
)
from .kokoro_typed_diagnostic import localization_report
from .kokoro_typed_engine import MAX_CLIPPED_FRACTION, inspect_render, local_engine_assets
from .kokoro_validated_shell import VALIDATED_LENS_SHELL, VALIDATED_NEUTRAL_SHELL
from .util import atomic_write_json, atomic_write_text, sha256_file


RUN_ID = "20260717-kokoro-strict-shell-confirmation-v1"
PARENT_RUN_ID = "20260717-kokoro-validated-shell-confirmation-v1"
PROTOCOL_FILE = "protocol.json"
RECORDS_FILE = "render-records.json"
ANALYSIS_FILE = "analysis.json"
REVIEW_FILE = "review.html"
REVIEW_MANIFEST_FILE = "review-manifest.json"
BLIND_KEY_FILE = "blind-key.json"
RESPONSE_FILENAME = "kokoro-strict-shell-confirmation-v1-response.json"
RAW_RESPONSE_FILE = RESPONSE_FILENAME
MANUAL_RESULT_FILE = "manual-result.json"
ATTEMPT_DIR = "attempts"
BLIND_SEED = 20_260_717_06

MEDIAL_INVENTORY = (
    "Quiet voices map distant roads.",
    "Gentle voices map distant roads.",
    "Soft voices map distant roads.",
)
EXPECTED_TEXT = MEDIAL_INVENTORY[0]
EXPECTED_PLAN_SHA256 = "0eda681d2894ce3ee077fdd465e14423fdf8b6eaf98161201410140e077b9d62"
NEW_FIXTURE_ID = "phrase-medial-strict-shell"
REUSED_FIXTURE_IDS = (
    "phrase-final-validated-shell",
    "multiple-repeated-target",
)


def run_dir() -> Path:
    return Paths().artifacts / "typed-engine" / RUN_ID


def parent_dir() -> Path:
    return Paths().artifacts / "typed-engine" / PARENT_RUN_ID


def _load_json(path: Path) -> Any:
    if not path.is_file():
        raise RuntimeError(f"required artifact missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


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


def _select_medial() -> dict[str, Any]:
    planner = StrictShellPlanner.load()
    prior = [
        path.read_bytes()
        for path in sorted(Paths().artifacts.rglob("*.json"))
        if not path.is_relative_to(run_dir())
    ]
    attempts: list[dict[str, Any]] = []
    for order, text in enumerate(MEDIAL_INVENTORY, start=1):
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
        if first.target_word_indexes != (2,) or first.target_occurrence_count != 1:
            reasons.append("wrong_medial_target_shape")
        if not (
            first.gate_summary.espeak_gate_pass
            and first.gate_summary.kokoro_phone_gate_pass
        ):
            reasons.append("carrier_gate_failed")
        target = first.words[first.target_word_indexes[0]]
        if target.neutral_phone != VALIDATED_NEUTRAL_SHELL:
            reasons.append("neutral_not_exact_validated_shell")
        if target.lens_phone != VALIDATED_LENS_SHELL:
            reasons.append("lens_not_exact_validated_shell")
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
            if text != EXPECTED_TEXT or first.plan_sha256 != EXPECTED_PLAN_SHA256:
                raise RuntimeError("strict-shell medial selection drifted")
            return {
                "fixture_id": NEW_FIXTURE_ID,
                "role": "one exact-shell medial target with substantial lexical neighbors",
                "inventory": list(MEDIAL_INVENTORY),
                "attempts": attempts,
                "selected_order": order,
                "text": text,
                "plan_sha256": first.plan_sha256,
                "source_phonemes": first.source_phonemes,
                "neutral_phonemes": first.neutral_phonemes,
                "lens_phonemes": first.lens_phonemes,
                "neutral_script": first.neutral_script,
                "lens_script": first.lens_script,
                "target_word_indexes": list(first.target_word_indexes),
                "target_occurrence_count": first.target_occurrence_count,
                "anchor_occurrence_map": [0],
                "words": [asdict(word) for word in first.words],
                "gate_summary": asdict(first.gate_summary),
            }
    raise RuntimeError("no gate-clean strict-shell medial fixture")


def _verified_parent_successes() -> dict[str, Any]:
    protocol_path = parent_dir() / "protocol.json"
    records_path = parent_dir() / "render-records.json"
    analysis_path = parent_dir() / "analysis.json"
    records = _load_json(records_path)
    analysis = _load_json(analysis_path)
    if analysis.get("classification") != "validated_shell_automatic_failed":
        raise RuntimeError("partial parent classification drifted")
    results = {row["fixture_id"]: row for row in analysis["fixtures"]}
    record_by_id = {row["fixture_id"]: row for row in records["fixtures"]}
    reused: list[dict[str, Any]] = []
    for fixture_id in REUSED_FIXTURE_IDS:
        result = results[fixture_id]
        record = record_by_id[fixture_id]
        if result.get("automatic_pass") is not True:
            raise RuntimeError(f"reused parent fixture is not a pass: {fixture_id}")
        audio: dict[str, Any] = {}
        for role in ("neutral", "identity", "lens", "full-state-lens-source"):
            item = record["audio"][role]
            path = parent_dir() / item["relative_path"]
            if sha256_file(path) != item["wav_sha256"]:
                raise RuntimeError(f"reused audio drifted: {fixture_id}/{role}")
            audio[role] = {**item, "parent_relative_path": item["relative_path"]}
        reused.append(
            {
                "fixture_id": fixture_id,
                "automatic_pass": True,
                "analysis_fixture_sha256": sha256_json(result),
                "record_fixture_sha256": sha256_json(record),
                "audio": audio,
            }
        )
    return {
        "run_id": PARENT_RUN_ID,
        "protocol_file_sha256": sha256_file(protocol_path),
        "render_records_file_sha256": sha256_file(records_path),
        "analysis_file_sha256": sha256_file(analysis_path),
        "classification_preserved": analysis["classification"],
        "reused_without_rerender": reused,
        "failed_extended_shell_fixture_not_reused": "phrase-medial-continuous",
    }


def _layout() -> list[dict[str, Any]]:
    fixture_ids = (NEW_FIXTURE_ID, *REUSED_FIXTURE_IDS)
    trials = [
        *(
            {
                "fixture_id": fixture_id,
                "condition": "identity-control",
                "roles": ["neutral", "identity"],
            }
            for fixture_id in fixture_ids
        ),
        *(
            {
                "fixture_id": fixture_id,
                "condition": "spliced-lens",
                "roles": ["neutral", "lens"],
            }
            for fixture_id in fixture_ids
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


def _tracked(diagnostic: dict[str, Any]) -> list[str]:
    rows = [
        "src/earshift_bakeoff/kokoro_strict_shell.py",
        "src/earshift_bakeoff/kokoro_strict_shell_confirmation.py",
        "src/earshift_bakeoff/kokoro_output_domain_splice.py",
        "src/earshift_bakeoff/kokoro_output_splice_unseen.py",
        "src/earshift_bakeoff/kokoro_synthesis.py",
        "src/earshift_bakeoff/kokoro_typed_engine.py",
        "scripts/run_kokoro_strict_shell_confirmation_v1.py",
        "scripts/praat_sentence_pair_v2_burg.praat",
        "uv.lock",
        f"artifacts/typed-engine/{RUN_ID}/{PROTOCOL_FILE}",
        f"artifacts/typed-engine/{PARENT_RUN_ID}/protocol.json",
        f"artifacts/typed-engine/{PARENT_RUN_ID}/render-records.json",
        f"artifacts/typed-engine/{PARENT_RUN_ID}/analysis.json",
    ]
    rows.extend(row["relative_path"] for row in diagnostic["bound_output_files"])
    return sorted(set(rows))


def protocol_record() -> dict[str, Any]:
    diagnostic = _verified_diagnostic_parent()
    medial = _select_medial()
    parent = _verified_parent_successes()
    root = Paths().root
    sources = {
        "strict_allocator": root / "src/earshift_bakeoff/kokoro_strict_shell.py",
        "confirmation": root
        / "src/earshift_bakeoff/kokoro_strict_shell_confirmation.py",
        "output_splice": root
        / "src/earshift_bakeoff/kokoro_output_domain_splice.py",
        "synthesis": root / "src/earshift_bakeoff/kokoro_synthesis.py",
    }
    payload: dict[str, Any] = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "status": "frozen_before_strict_shell_decode",
        "question": (
            "Does the exact validated shell pass in a new medial context when extended "
            "target words fail closed and the two prior successful fixtures are reused?"
        ),
        "change": {
            "mechanism": "coverage boundary for rule-bearing source words",
            "version": STRICT_SHELL_VERSION,
            "supported_shape": "exact C-stress-/ae/-C source phone plan",
            "unsupported_behavior": "fail closed before synthesis",
            "renderer_or_gate_changes": "none",
            "rationale": (
                "the exact v-stress-vowel-zh shell passed three occurrences; the sole "
                "extended v-stress-vowel-zh-z occurrence failed neutral anchor sanity"
            ),
        },
        "failed_parent": {
            "run_id": PARENT_RUN_ID,
            "classification_preserved": parent["classification_preserved"],
            "failed_fixture": "phrase-medial-continuous",
            "failed_shell": "vˈæʒz / vˈɛʒz",
            "failed_checks": [
                "neutral_nearer_local_ae at 5750 Hz",
                "neutral_nearer_local_ae at 6000 Hz",
            ],
        },
        "reused_parent_successes": parent,
        "new_fixture": medial,
        "manifest": [
            {
                "order": order,
                "slot_id": f"{NEW_FIXTURE_ID}__{role}",
                "fixture_id": NEW_FIXTURE_ID,
                "role": role,
                "plan_sha256": medial["plan_sha256"],
                "one_attempt_no_retry": True,
            }
            for order, role in enumerate(
                ("neutral", "identity", "full-state-lens-source"), start=1
            )
        ],
        "scope": {
            "new_decoder_attempt_ceiling": 3,
            "reused_successful_fixture_count": 2,
            "rerendered_successful_cells": 0,
            "derived_new_spliced_lenses": 1,
            "api_calls": 0,
            "selection_or_replacement": "none",
        },
        "automatic_gate": {
            "unchanged": True,
            "primary_window_percent": PRIMARY_WINDOW_PERCENT,
            "ceilings_hz": list(CEILINGS_HZ),
            "descriptive_windows": list(DESCRIPTIVE_WINDOW_PERCENTS),
            "descriptive_only": True,
            "aggregate_pass": (
                "the new medial fixture passes every frozen automatic gate and both "
                "hash-bound reused fixtures retain their prior automatic passes"
            ),
        },
        "blind_review": {
            "only_after_aggregate_automatic_pass": True,
            "layout": _layout(),
            "trial_count": 6,
            "response_filename": RESPONSE_FILENAME,
            "manual_gates_unchanged": True,
        },
        "parents": {"diagnostic_anchor_geometry": diagnostic},
        "implementation": {
            "source_file_sha256": {
                key: sha256_file(path) for key, path in sorted(sources.items())
            },
            "measurement": {
                "praat_sha256": sha256_file(PRAAT),
                "script_sha256": sha256_file(MEASUREMENT_SCRIPT),
            },
            "engine_assets": local_engine_assets(),
            "renderer": {
                "package": "kokoro",
                "version": KOKORO_VERSION,
                "model_repo": MODEL_REPO,
                "model_revision": MODEL_REVISION,
                "model_hashes": MODEL_HASHES,
                "voice": "af_heart",
                "sample_rate_hz": SAMPLE_RATE_HZ,
                "speed": SPEED,
                "rng_seed": RNG_SEED,
            },
            "tracked_clean_paths": _tracked(diagnostic),
        },
        "stopping_rule": (
            "Attempt the three new decoder slots once. Do not rerender the two reused "
            "successes, replace the medial fixture, change thresholds, or rescue with "
            "descriptive windows."
        ),
    }
    return {**payload, "protocol_sha256": sha256_json(payload)}


def prepare() -> dict[str, Any]:
    protocol = protocol_record()
    path = run_dir() / PROTOCOL_FILE
    if path.exists():
        if stable_json(_load_json(path)) != stable_json(protocol):
            raise RuntimeError("strict-shell protocol differs")
    else:
        if any(
            path.exists()
            for path in (
                run_dir() / RECORDS_FILE,
                run_dir() / ANALYSIS_FILE,
                run_dir() / "audio",
                run_dir() / ATTEMPT_DIR,
            )
        ):
            raise RuntimeError("strict-shell output exists before protocol")
        atomic_write_json(path, protocol)
    return protocol


def _checked_protocol() -> dict[str, Any]:
    frozen = _load_json(run_dir() / PROTOCOL_FILE)
    if stable_json(frozen) != stable_json(protocol_record()):
        raise RuntimeError("strict-shell protocol inputs drifted")
    return frozen


def _require_commit(protocol: dict[str, Any]) -> str:
    paths = protocol["implementation"]["tracked_clean_paths"]
    subprocess.run(
        ["git", "ls-files", "--error-unmatch", "--", *paths],
        cwd=Paths().root,
        check=True,
        capture_output=True,
    )
    if subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--", *paths], cwd=Paths().root
    ).returncode:
        raise RuntimeError("strict-shell inputs differ from committed HEAD")
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Paths().root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _write_wav(path: Path, values: np.ndarray) -> None:
    if path.exists():
        raise RuntimeError("strict-shell one-attempt WAV exists")
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


def _render(protocol: dict[str, Any], records: dict[str, Any]) -> None:
    from .kokoro_synthesis import KokoroSynthesisRuntime

    for slot in records["slots"]:
        atomic_write_json(
            run_dir() / ATTEMPT_DIR / f"{slot['slot_id']}.json",
            {
                "run_id": RUN_ID,
                "protocol_sha256": protocol["protocol_sha256"],
                "slot_id": slot["slot_id"],
                "one_attempt_no_retry": True,
            },
        )
        slot["status"] = "attempt_started"
    atomic_write_json(run_dir() / RECORDS_FILE, records)
    try:
        planner = StrictShellPlanner.load()
        runtime = KokoroSynthesisRuntime.load(download=False)
        frozen = protocol["new_fixture"]
        plan = planner.plan(frozen["text"])
        if plan.plan_sha256 != frozen["plan_sha256"]:
            raise RuntimeError("strict-shell plan drifted")
        pair = plan.pair_plan()
        if pair is None:
            raise RuntimeError("strict-shell fixture lost pair")
        rendered = runtime.render_parity_triplet(pair)
        columns = target_word_columns(
            runtime.model, plan.neutral_phonemes, plan.target_word_indexes
        )
        if rendered.replaced_columns != columns:
            raise RuntimeError("strict-shell target columns drifted")
        neutral = np.frombuffer(pcm16_bytes(rendered.neutral), dtype="<i2").copy()
        identity = np.frombuffer(pcm16_bytes(rendered.identity), dtype="<i2").copy()
        full_lens = np.frombuffer(pcm16_bytes(rendered.lens), dtype="<i2").copy()
        alignment = alignment_record(
            model=runtime.model,
            plan=plan,
            durations=rendered.predicted_durations,
            sample_count=neutral.size,
            anchor_occurrence_map=[0],
        )
        targets = [row["interval"] for row in alignment["target_words"]]
        baseline = localization_report(neutral, full_lens, targets)
        windows = baseline["inside_windows"]
        lens, weights = output_domain_splice(neutral, full_lens, windows)
        all_words = _word_intervals(
            runtime.model, plan, rendered.predicted_durations, neutral.size
        )
        edge = phrase_medial_edge_gate(
            plan.target_word_indexes[0], all_words, windows[0]
        )
        values = {
            "neutral": neutral,
            "identity": identity,
            "full-state-lens-source": full_lens,
            "lens": lens,
        }
        audio: dict[str, Any] = {}
        for role, value in values.items():
            path = run_dir() / "audio" / f"{NEW_FIXTURE_ID}__{role}.wav"
            _write_wav(path, value)
            audio[role] = _audio(path, value)
        integrity = inspect_render(
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
            "raw_integrity": integrity.pass_all,
            "identity_bit_exact": np.array_equal(neutral, identity),
            "equal_samples": len({value.size for value in values.values()}) == 1,
            "finite_unclipped": all(
                row["finite"] and row["clipping_pass"] for row in audio.values()
            ),
            "outside_exact_neutral": np.array_equal(
                lens[weights == 0], neutral[weights == 0]
            ),
            "interior_exact_full_lens": bool(
                np.any(weights == 1)
                and np.array_equal(lens[weights == 1], full_lens[weights == 1])
            ),
        }
        records["fixtures"].append(
            {
                "fixture_id": NEW_FIXTURE_ID,
                "plan_sha256": plan.plan_sha256,
                "safe_plan_metadata": plan.safe_metadata(),
                "predicted_durations": list(rendered.predicted_durations),
                "alignment": alignment,
                "all_word_intervals": all_words,
                "splice_windows": windows,
                "phrase_medial_edge_gate": edge,
                "untouched_full_state_localization": baseline,
                "audio": audio,
                "runtime_checks": checks,
                "runtime_pass": all(checks.values()),
            }
        )
        for slot in records["slots"]:
            slot["status"] = "complete"
            slot["audio"] = audio[slot["role"]]
        records["decoder_attempt_count"] = 3
        records["status"] = "render_complete"
        atomic_write_json(run_dir() / RECORDS_FILE, records)
    except Exception as exc:
        for slot in records["slots"]:
            slot["status"] = "failed_no_retry"
            slot["failure"] = f"{type(exc).__name__}: {exc}"[:1000]
        records["status"] = "runtime_failure_no_retry"
        atomic_write_json(run_dir() / RECORDS_FILE, records)
        raise


def _analyze(record: dict[str, Any], protocol: dict[str, Any]) -> dict[str, Any]:
    paths = {
        role: run_dir() / row["relative_path"] for role, row in record["audio"].items()
    }
    neutral = _read_pcm(paths["neutral"])
    identity = _read_pcm(paths["identity"])
    full_lens = _read_pcm(paths["full-state-lens-source"])
    lens = _read_pcm(paths["lens"])
    _, weights = output_domain_splice(neutral, full_lens, record["splice_windows"])
    integrity = {
        **record["runtime_checks"],
        "hashes_unchanged": all(
            sha256_file(path) == record["audio"][role]["wav_sha256"]
            for role, path in paths.items()
        ),
        "identity_recheck": np.array_equal(neutral, identity),
        "outside_recheck": np.array_equal(
            lens[weights == 0], neutral[weights == 0]
        ),
        "interior_recheck": bool(
            np.any(weights == 1)
            and np.array_equal(lens[weights == 1], full_lens[weights == 1])
        ),
    }
    targets = [row["interval"] for row in record["alignment"]["target_words"]]
    boundary = boundary_artifact_report(
        neutral, full_lens, lens, record["splice_windows"]
    )
    localization = localization_report(neutral, lens, targets)
    benchmark = _benchmark_localization(neutral, lens, targets)
    acoustic = _acoustic_report(
        paths["neutral"],
        paths["lens"],
        record["alignment"]["target_occurrences"],
        protocol["parents"]["diagnostic_anchor_geometry"]["local_anchor_geometry"],
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
        "fixture_id": NEW_FIXTURE_ID,
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
    try:
        new_result = _analyze(records["fixtures"][0], protocol)
        failures: list[str] = []
    except Exception as exc:
        new_result = None
        failures = [f"{type(exc).__name__}: {exc}"[:1000]]
    reused = protocol["reused_parent_successes"]["reused_without_rerender"]
    reused_pass = all(row["automatic_pass"] for row in reused)
    passed = bool(not failures and new_result and new_result["automatic_pass"] and reused_pass)
    payload = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "protocol_sha256": protocol["protocol_sha256"],
        "classification": (
            "strict_shell_aggregate_automatic_pass_pending_human_qc"
            if passed
            else (
                "strict_shell_measurement_inconclusive"
                if failures
                else "strict_shell_aggregate_automatic_failed"
            )
        ),
        "automatic_pass": passed,
        "pending_human_review": passed,
        "new_fixture": new_result,
        "reused_fixture_ids": list(REUSED_FIXTURE_IDS),
        "reused_fixture_automatic_pass": reused_pass,
        "measurement_failures": failures,
        "parent_failure_preserved": "validated_shell_automatic_failed",
        "descriptive_windows_do_not_change_outcome": True,
        "api_calls_made": 0,
        "decoder_attempt_count": records["decoder_attempt_count"],
        "render_records_sha256": sha256_file(run_dir() / RECORDS_FILE),
        "production_enabled": False,
    }
    return {**payload, "analysis_sha256": sha256_json(payload)}


def _review_records(protocol: dict[str, Any]) -> dict[str, dict[str, Any]]:
    current = _load_json(run_dir() / RECORDS_FILE)["fixtures"][0]
    rows = {NEW_FIXTURE_ID: {**current, "base_dir": run_dir()}}
    parent_records = {
        row["fixture_id"]: row
        for row in _load_json(parent_dir() / "render-records.json")["fixtures"]
    }
    for reused in protocol["reused_parent_successes"]["reused_without_rerender"]:
        record = parent_records[reused["fixture_id"]]
        if sha256_json(record) != reused["record_fixture_sha256"]:
            raise RuntimeError("reused parent record drifted")
        rows[reused["fixture_id"]] = {
            **record,
            "base_dir": parent_dir(),
        }
    return rows


def build_review(protocol: dict[str, Any], analysis: dict[str, Any]) -> dict[str, Any]:
    if analysis.get("automatic_pass") is not True:
        raise RuntimeError("strict-shell aggregate did not authorize review")
    layout = _layout()
    records = _review_records(protocol)
    destination = run_dir() / "review-audio"
    destination.mkdir(parents=True, exist_ok=True)
    public: list[dict[str, Any]] = []
    for trial in layout:
        record = records[trial["fixture_id"]]
        sides: list[dict[str, str]] = []
        for side, role in trial["side_roles"].items():
            item = record["audio"][role]
            source = record["base_dir"] / item["relative_path"]
            target = destination / f"{trial['trial_id'][-2:]}-{side.lower()}.wav"
            if target.exists():
                if sha256_file(target) != sha256_file(source):
                    raise RuntimeError("strict-shell blind copy drifted")
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
        "trial_count": 6,
        "response_filename": RESPONSE_FILENAME,
        "public_trials": public,
        "hidden_fields_absent": True,
    }
    _write_once_json(run_dir() / BLIND_KEY_FILE, key)
    _write_once_json(run_dir() / REVIEW_MANIFEST_FILE, manifest)
    html = _shared_review_html(public, protocol["protocol_sha256"])
    html = html.replace(
        "20260717-kokoro-output-splice-unseen-v1", RUN_ID
    ).replace("kokoro-output-splice-unseen-v1-response.json", RESPONSE_FILENAME)
    path = run_dir() / REVIEW_FILE
    if path.exists() and path.read_text(encoding="utf-8") != html:
        raise RuntimeError("strict-shell review HTML drifted")
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
    if sha256_file(PRAAT) != protocol["implementation"]["measurement"]["praat_sha256"]:
        raise RuntimeError("Praat drifted")
    if sha256_file(MEASUREMENT_SCRIPT) != protocol["implementation"]["measurement"][
        "script_sha256"
    ]:
        raise RuntimeError("measurement script drifted")
    commit = _require_commit(protocol)
    records = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "protocol_sha256": protocol["protocol_sha256"],
        "implementation_commit": commit,
        "status": "in_progress",
        "api_calls_made": 0,
        "decoder_attempt_count": 0,
        "slots": [{**row, "status": "pending"} for row in protocol["manifest"]],
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
            "classification": "strict_shell_runtime_inconclusive",
            "automatic_pass": False,
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
    if response.get("run_id") != RUN_ID or response.get("protocol_sha256") != protocol[
        "protocol_sha256"
    ]:
        raise RuntimeError("strict-shell response belongs to another run")
    keys = {
        row["trial_id"]: row
        for row in _load_json(run_dir() / BLIND_KEY_FILE)["trials"]
    }
    rows = response.get("responses")
    if not isinstance(rows, list) or {row.get("trial_id") for row in rows} != set(keys):
        raise RuntimeError("strict-shell response is incomplete")
    decoded: list[dict[str, Any]] = []
    fixture_results: dict[str, dict[str, bool]] = {
        fixture_id: {} for fixture_id in (NEW_FIXTURE_ID, *REUSED_FIXTURE_IDS)
    }
    for row in rows:
        key = keys[row["trial_id"]]
        sides = {side: _side_gate(row["sides"][side]) for side in ("A", "B")}
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
        passed = bool(all(value["pass"] for value in sides.values()) and all(pair.values()))
        fixture_results[key["fixture_id"]][key["condition"]] = passed
        decoded.append(
            {
                "trial_id": row["trial_id"],
                "fixture_id": key["fixture_id"],
                "condition": key["condition"],
                "side_gates": sides,
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
            "strict_shell_human_qc_pass" if passed else "strict_shell_human_qc_failed"
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
