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
from .kokoro_validated_shell import (
    VALIDATED_LENS_SHELL,
    VALIDATED_NEUTRAL_SHELL,
    VALIDATED_SHELL_VERSION,
    ValidatedShellPlanner,
)
from .util import atomic_write_json, atomic_write_text, sha256_file


RUN_ID = "20260717-kokoro-validated-shell-confirmation-v1"
FAILED_PARENT_RUN_ID = "20260717-kokoro-output-splice-unseen-v1"
PROTOCOL_FILE = "protocol.json"
RECORDS_FILE = "render-records.json"
ANALYSIS_FILE = "analysis.json"
REVIEW_FILE = "review.html"
REVIEW_MANIFEST_FILE = "review-manifest.json"
BLIND_KEY_FILE = "blind-key.json"
RESPONSE_FILENAME = "kokoro-validated-shell-confirmation-v1-response.json"
RAW_RESPONSE_FILE = RESPONSE_FILENAME
MANUAL_RESULT_FILE = "manual-result.json"
ATTEMPT_DIR = "attempts"
BLIND_SEED = 20_260_717_05


FIXTURE_INVENTORY = (
    {
        "fixture_id": "phrase-medial-continuous",
        "role": "one medial target with substantial lexical neighbors on both sides",
        "candidates": (
            "We drift slowly past quiet fields.",
            "We move gently past quiet trees.",
            "Softly we pass quiet fields.",
        ),
        "target_count": 1,
        "anchor_map": (0,),
    },
    {
        "fixture_id": "phrase-final-validated-shell",
        "role": "one phrase-final target through the validated shell",
        "candidates": (
            "We drift quietly toward the cap.",
            "The quiet glow reaches the cap.",
            "Soft lights gather near the cap.",
        ),
        "target_count": 1,
        "anchor_map": (1,),
    },
    {
        "fixture_id": "multiple-repeated-target",
        "role": "multiple non-overlapping targets with a repeated source mapping",
        "candidates": (
            "We place one cap beside one cap.",
            "We set one cap beside one cap.",
            "Quiet lights frame the cap beside the cap.",
        ),
        "target_count": 2,
        "anchor_map": (0, 1),
    },
)


EXPECTED = {
    "phrase-medial-continuous": (
        "We drift slowly past quiet fields.",
        "566d21f343bc366f4b7defe54bf14e49d73167aa9d32ff4168c458806ff8f1b3",
    ),
    "phrase-final-validated-shell": (
        "We drift quietly toward the cap.",
        "e3af98ef1af75f05cb8b76c5e02cc7d359a90307af8bfed9dcfd630c89b24da7",
    ),
    "multiple-repeated-target": (
        "We place one cap beside one cap.",
        "20289247f44756d1917464a845824797bf81329e8f4935d337c38c018fba8428",
    ),
}


def run_dir() -> Path:
    return Paths().artifacts / "typed-engine" / RUN_ID


def failed_parent_dir() -> Path:
    return Paths().artifacts / "typed-engine" / FAILED_PARENT_RUN_ID


def _load_json(path: Path) -> Any:
    if not path.is_file():
        raise RuntimeError(f"required artifact is missing: {path}")
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


def _prior_json() -> list[bytes]:
    return [
        path.read_bytes()
        for path in sorted(Paths().artifacts.rglob("*.json"))
        if not path.is_relative_to(run_dir())
    ]


def _fixture_reasons(spec: dict[str, Any], plan: Any) -> list[str]:
    reasons: list[str] = []
    if not plan.comparison_available:
        reasons.append("comparison_unavailable")
    if not plan.gate_summary.espeak_gate_pass:
        reasons.append("espeak_gate_failed")
    if not plan.gate_summary.kokoro_phone_gate_pass:
        reasons.append("kokoro_phone_gate_failed")
    if plan.target_occurrence_count != spec["target_count"]:
        reasons.append("wrong_target_count")
    for index in plan.target_word_indexes:
        word = plan.words[index]
        target = word.target_offsets[0]
        if word.neutral_phone[target - 2 : target + 2] != VALIDATED_NEUTRAL_SHELL:
            reasons.append("neutral_shell_drift")
        if word.lens_phone[target - 2 : target + 2] != VALIDATED_LENS_SHELL:
            reasons.append("lens_shell_drift")
    if spec["fixture_id"] == "phrase-medial-continuous":
        if len(plan.target_word_indexes) != 1 or not (
            1 < plan.target_word_indexes[0] < len(plan.words) - 2
        ):
            reasons.append("target_lacks_substantial_lexical_neighbors")
    elif spec["fixture_id"] == "phrase-final-validated-shell":
        if plan.target_word_indexes != (len(plan.words) - 1,):
            reasons.append("target_not_phrase_final")
    else:
        target_words = [plan.words[index] for index in plan.target_word_indexes]
        if len({word.word_index for word in target_words}) < 2:
            reasons.append("targets_overlap")
        if len({word.source.casefold() for word in target_words}) != 1:
            reasons.append("target_source_not_repeated")
        if len(
            {(word.neutral_phone, word.lens_phone) for word in target_words}
        ) != 1:
            reasons.append("repeated_mapping_drift")
    return reasons


def _select_fixtures() -> list[dict[str, Any]]:
    planner = ValidatedShellPlanner.load()
    prior = _prior_json()
    selected: list[dict[str, Any]] = []
    for spec in FIXTURE_INVENTORY:
        attempts: list[dict[str, Any]] = []
        winner: dict[str, Any] | None = None
        for order, text in enumerate(spec["candidates"], start=1):
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
            reasons = _fixture_reasons(spec, first)
            if stable_json(asdict(first)) != stable_json(asdict(second)):
                reasons.append("plan_not_deterministic")
            if any(text.encode() in payload for payload in prior):
                reasons.append("text_seen_in_prior_artifact")
            if any(first.plan_sha256.encode() in payload for payload in prior):
                reasons.append("plan_seen_in_prior_artifact")
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
                winner = {
                    "fixture_id": spec["fixture_id"],
                    "role": spec["role"],
                    "inventory": list(spec["candidates"]),
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
                    "anchor_occurrence_map": list(spec["anchor_map"]),
                    "words": [asdict(word) for word in first.words],
                    "gate_summary": asdict(first.gate_summary),
                }
                break
        if winner is None:
            raise RuntimeError(f"no frozen validated-shell fixture: {spec['fixture_id']}")
        expected_text, expected_hash = EXPECTED[spec["fixture_id"]]
        if winner["text"] != expected_text or winner["plan_sha256"] != expected_hash:
            raise RuntimeError(f"fixture selection drifted: {spec['fixture_id']}")
        selected.append(winner)
    return selected


def _failed_parent_diagnosis() -> dict[str, Any]:
    protocol_path = failed_parent_dir() / "protocol.json"
    records_path = failed_parent_dir() / "render-records.json"
    analysis_path = failed_parent_dir() / "analysis.json"
    records = _load_json(records_path)
    analysis = _load_json(analysis_path)
    if analysis.get("classification") != "unseen_output_splice_automatic_failed":
        raise RuntimeError("failed unseen parent classification drifted")
    parent_protocol = _load_json(protocol_path)
    anchors = parent_protocol["parents"]["diagnostic_anchor_geometry"][
        "local_anchor_geometry"
    ]
    rows: list[dict[str, Any]] = []
    for record in records["fixtures"]:
        neutral = failed_parent_dir() / record["audio"]["neutral"]["relative_path"]
        full_lens = (
            failed_parent_dir()
            / record["audio"]["full-state-lens-source"]["relative_path"]
        )
        report = _acoustic_report(
            neutral,
            full_lens,
            record["alignment"]["target_occurrences"],
            anchors,
        )
        rows.append(
            {
                "fixture_id": record["fixture_id"],
                "untouched_full_state_lens_primary_pass": report[
                    "primary_gate_pass"
                ],
                "window_pass": {
                    key: value["pass"] for key, value in report["windows"].items()
                },
                "neutral_wav_sha256": sha256_file(neutral),
                "full_state_lens_wav_sha256": sha256_file(full_lens),
            }
        )
    if any(row["untouched_full_state_lens_primary_pass"] for row in rows):
        raise RuntimeError("failed-parent acoustic diagnosis drifted")
    return {
        "run_id": FAILED_PARENT_RUN_ID,
        "protocol_file_sha256": sha256_file(protocol_path),
        "render_records_file_sha256": sha256_file(records_path),
        "analysis_file_sha256": sha256_file(analysis_path),
        "classification_preserved": analysis["classification"],
        "diagnosis": (
            "all three untouched full-state lens sources already fail the primary "
            "acoustic gate; output splicing is not the acoustic failure mechanism"
        ),
        "fixtures": rows,
    }


def _layout() -> list[dict[str, Any]]:
    trials = [
        *(
            {
                "fixture_id": spec["fixture_id"],
                "condition": "identity-control",
                "roles": ["neutral", "identity"],
            }
            for spec in FIXTURE_INVENTORY
        ),
        *(
            {
                "fixture_id": spec["fixture_id"],
                "condition": "spliced-lens",
                "roles": ["neutral", "lens"],
            }
            for spec in FIXTURE_INVENTORY
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
        "src/earshift_bakeoff/kokoro_validated_shell.py",
        "src/earshift_bakeoff/kokoro_validated_shell_confirmation.py",
        "src/earshift_bakeoff/kokoro_output_domain_splice.py",
        "src/earshift_bakeoff/kokoro_output_splice_unseen.py",
        "src/earshift_bakeoff/kokoro_synthesis.py",
        "src/earshift_bakeoff/kokoro_typed_engine.py",
        "scripts/run_kokoro_validated_shell_confirmation_v1.py",
        "scripts/praat_sentence_pair_v2_burg.praat",
        "uv.lock",
        f"artifacts/typed-engine/{RUN_ID}/{PROTOCOL_FILE}",
        f"artifacts/typed-engine/{FAILED_PARENT_RUN_ID}/protocol.json",
        f"artifacts/typed-engine/{FAILED_PARENT_RUN_ID}/render-records.json",
        f"artifacts/typed-engine/{FAILED_PARENT_RUN_ID}/analysis.json",
    ]
    rows.extend(row["relative_path"] for row in diagnostic["bound_output_files"])
    return sorted(set(rows))


def protocol_record() -> dict[str, Any]:
    fixtures = _select_fixtures()
    diagnostic = _verified_diagnostic_parent()
    root = Paths().root
    sources = {
        "candidate_allocator": root / "src/earshift_bakeoff/kokoro_validated_shell.py",
        "confirmation": root
        / "src/earshift_bakeoff/kokoro_validated_shell_confirmation.py",
        "output_splice": root
        / "src/earshift_bakeoff/kokoro_output_domain_splice.py",
        "synthesis": root / "src/earshift_bakeoff/kokoro_synthesis.py",
    }
    payload: dict[str, Any] = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "status": "frozen_before_any_corrective_decode",
        "question": (
            "Does changing only the immediate target carrier allocation to the already "
            "validated v-vowel-zh shell allow unchanged output-domain-splice-v1 to pass?"
        ),
        "failed_parent": _failed_parent_diagnosis(),
        "change": {
            "mechanism": "rule-bearing carrier allocation",
            "version": VALIDATED_SHELL_VERSION,
            "neutral_immediate_shell": VALIDATED_NEUTRAL_SHELL,
            "lens_immediate_shell": VALIDATED_LENS_SHELL,
            "preserved": [
                "source-derived phone length and target offset",
                "all phones outside the immediate C-stress-vowel-C shell",
                "utterance-local repeated source mapping",
                "written-word, predicted-homophone, adjacency, punctuation, and opacity gates",
                "Kokoro model, voice, speed, RNG, shared-state renderer, output-domain splice, taper, thresholds, anchors, measurements, and manual gates",
            ],
            "not_a_reclassification": (
                "the failed unseen-v1 run remains failed; this is a separately versioned "
                "corrective candidate"
            ),
        },
        "fixtures": fixtures,
        "manifest": [
            {
                "order": order,
                "slot_id": f"{fixture['fixture_id']}__{role}",
                "fixture_id": fixture["fixture_id"],
                "role": role,
                "plan_sha256": fixture["plan_sha256"],
                "one_attempt_no_retry": True,
            }
            for order, (fixture, role) in enumerate(
                (
                    (fixture, role)
                    for fixture in fixtures
                    for role in ("neutral", "identity", "full-state-lens-source")
                ),
                start=1,
            )
        ],
        "scope": {
            "fixtures": 3,
            "decoder_attempt_ceiling": 9,
            "derived_spliced_lenses": 3,
            "api_calls": 0,
            "replacement_or_selection": "none",
        },
        "automatic_gate": {
            "unchanged_from_unseen_v1": True,
            "primary_window_percent": PRIMARY_WINDOW_PERCENT,
            "ceilings_hz": list(CEILINGS_HZ),
            "descriptive_windows": list(DESCRIPTIVE_WINDOW_PERCENTS),
            "descriptive_only": True,
            "requirements": [
                "exact plans and runtime integrity",
                "neutral/identity bit identity",
                "equal finite nonempty unclipped PCM",
                "exact neutral outside splice windows and exact full lens in full-weight interiors",
                "unchanged boundary/click metrics",
                "primary 50 percent acoustic pass at every occurrence and ceiling",
                "cheap fail-closed localization runtime",
                "phrase-medial boundaries within aligned spoken neighbors",
            ],
        },
        "blind_review": {
            "only_after_automatic_pass": True,
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
                "praat_path": str(PRAAT),
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
            "Attempt the nine frozen slots once; no replacement, rerender, listening "
            "selection, threshold change, or descriptive-window rescue."
        ),
    }
    return {**payload, "protocol_sha256": sha256_json(payload)}


def prepare() -> dict[str, Any]:
    protocol = protocol_record()
    path = run_dir() / PROTOCOL_FILE
    if path.exists():
        if stable_json(_load_json(path)) != stable_json(protocol):
            raise RuntimeError("corrective protocol differs from frozen artifact")
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
            raise RuntimeError("corrective output exists before protocol")
        atomic_write_json(path, protocol)
    return protocol


def _checked_protocol() -> dict[str, Any]:
    frozen = _load_json(run_dir() / PROTOCOL_FILE)
    if stable_json(frozen) != stable_json(protocol_record()):
        raise RuntimeError("corrective protocol or its inputs drifted")
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
        raise RuntimeError("corrective inputs differ from committed HEAD")
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Paths().root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


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
    if any((run_dir() / ATTEMPT_DIR / f"{row['slot_id']}.json").exists() for row in slots):
        raise RuntimeError("corrective fixture was already attempted")
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

    planner = ValidatedShellPlanner.load()
    runtime = KokoroSynthesisRuntime.load(download=False)
    for frozen in protocol["fixtures"]:
        slots = _begin(records, frozen["fixture_id"])
        try:
            plan = planner.plan(frozen["text"])
            if plan.plan_sha256 != frozen["plan_sha256"]:
                raise RuntimeError("corrective plan drifted")
            pair = plan.pair_plan()
            if pair is None:
                raise RuntimeError("corrective fixture lost its pair")
            rendered = runtime.render_parity_triplet(pair)
            columns = target_word_columns(
                runtime.model, plan.neutral_phonemes, plan.target_word_indexes
            )
            if rendered.replaced_columns != columns:
                raise RuntimeError("corrective replaced columns drifted")
            neutral = np.frombuffer(pcm16_bytes(rendered.neutral), dtype="<i2").copy()
            identity = np.frombuffer(pcm16_bytes(rendered.identity), dtype="<i2").copy()
            full_lens = np.frombuffer(pcm16_bytes(rendered.lens), dtype="<i2").copy()
            alignment = alignment_record(
                model=runtime.model,
                plan=plan,
                durations=rendered.predicted_durations,
                sample_count=neutral.size,
                anchor_occurrence_map=frozen["anchor_occurrence_map"],
            )
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
                if frozen["fixture_id"] == "phrase-medial-continuous"
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


def _analyze_fixture(record: dict[str, Any], protocol: dict[str, Any]) -> dict[str, Any]:
    paths = {
        role: run_dir() / row["relative_path"] for role, row in record["audio"].items()
    }
    for role, path in paths.items():
        if sha256_file(path) != record["audio"][role]["wav_sha256"]:
            raise RuntimeError(f"corrective audio hash drifted: {role}")
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
        and len(fixtures) == len(FIXTURE_INVENTORY)
        and all(row["automatic_pass"] for row in fixtures)
    )
    payload = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "protocol_sha256": protocol["protocol_sha256"],
        "classification": (
            "validated_shell_automatic_pass_pending_human_qc"
            if passed
            else (
                "validated_shell_measurement_inconclusive"
                if failures
                else "validated_shell_automatic_failed"
            )
        ),
        "automatic_pass": passed,
        "pending_human_review": passed,
        "fixtures": fixtures,
        "measurement_failures": failures,
        "failed_parent_preserved": "unseen_output_splice_automatic_failed",
        "descriptive_windows_do_not_change_outcome": True,
        "api_calls_made": 0,
        "decoder_attempt_count": records["decoder_attempt_count"],
        "render_records_sha256": sha256_file(run_dir() / RECORDS_FILE),
        "production_enabled": False,
    }
    return {**payload, "analysis_sha256": sha256_json(payload)}


def _copy_review_audio(layout: Sequence[dict[str, Any]], records: dict[str, Any]) -> list[dict[str, Any]]:
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
                    raise RuntimeError("corrective blind copy drifted")
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
        raise RuntimeError("automatic gates did not authorize corrective review")
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
    html = html.replace(FAILED_PARENT_RUN_ID, RUN_ID).replace(
        "kokoro-output-splice-unseen-v1-response.json", RESPONSE_FILENAME
    )
    path = run_dir() / REVIEW_FILE
    if path.exists() and path.read_text(encoding="utf-8") != html:
        raise RuntimeError("corrective review HTML drifted")
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
        records = _load_json(run_dir() / RECORDS_FILE)
        result = {
            "schema_version": 1,
            "run_id": RUN_ID,
            "protocol_sha256": protocol["protocol_sha256"],
            "classification": "validated_shell_runtime_inconclusive",
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
    if response.get("run_id") != RUN_ID or response.get("protocol_sha256") != protocol[
        "protocol_sha256"
    ]:
        raise RuntimeError("response belongs to another run")
    keys = {
        row["trial_id"]: row
        for row in _load_json(run_dir() / BLIND_KEY_FILE)["trials"]
    }
    rows = response.get("responses")
    if not isinstance(rows, list) or {row.get("trial_id") for row in rows} != set(keys):
        raise RuntimeError("response trial set is incomplete")
    decoded: list[dict[str, Any]] = []
    fixture_results: dict[str, dict[str, bool]] = {
        spec["fixture_id"]: {} for spec in FIXTURE_INVENTORY
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
        fixture: bool(rows.get("identity-control") and rows.get("spliced-lens"))
        for fixture, rows in fixture_results.items()
    }
    passed = all(fixture_pass.values())
    result = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "protocol_sha256": protocol["protocol_sha256"],
        "classification": (
            "validated_shell_human_qc_pass"
            if passed
            else "validated_shell_human_qc_failed"
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
