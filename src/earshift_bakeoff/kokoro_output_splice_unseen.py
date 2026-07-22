from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import subprocess
import tempfile
import wave
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .config import Paths, sha256_json, stable_json
from .kokoro_output_domain_splice import (
    BENCHMARK_MEASURED_ITERATIONS,
    BENCHMARK_WARMUP_ITERATIONS,
    BOUNDARY_CONTEXT_MS,
    LOCALIZATION_MINIMUM,
    MAX_BOUNDARY_DERIVATIVE_RATIO,
    MAX_EDGE_DELTA_STEP_PCM,
    MAX_LOCALIZATION_MEDIAN_MS,
    MAX_LOCALIZATION_P95_MS,
    TAPER_MS,
    TAPER_SAMPLES,
    _benchmark_localization,
    boundary_artifact_report,
    output_domain_splice,
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
    _word_column_spans,
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
    MEASUREMENT_SCRIPT,
    PRAAT,
    PRIMARY_WINDOW_PERCENT,
    WINDOW_PERCENTS,
    _verified_diagnostic_parent,
)
from .kokoro_typed_diagnostic import localization_report
from .kokoro_typed_engine import (
    MAX_CLIPPED_FRACTION,
    KokoroTypedPlanner,
    TypedPlan,
    inspect_render,
    local_engine_assets,
)
from .util import atomic_write_json, atomic_write_text, sha256_file


RUN_ID = "20260717-kokoro-output-splice-unseen-v1"
PARENT_SPLICE_RUN_ID = "20260717-kokoro-output-domain-splice-v1"
PROTOCOL_FILE = "protocol.json"
RECORDS_FILE = "render-records.json"
ANALYSIS_FILE = "analysis.json"
REVIEW_FILE = "review.html"
REVIEW_MANIFEST_FILE = "review-manifest.json"
BLIND_KEY_FILE = "blind-key.json"
RESPONSE_FILENAME = "kokoro-output-splice-unseen-v1-response.json"
RAW_RESPONSE_FILE = RESPONSE_FILENAME
MANUAL_RESULT_FILE = "manual-result.json"
ATTEMPT_DIR = "attempts"
TARGET_PADDING_S = 0.150
BLIND_SEED = 20_260_717_04


@dataclass(frozen=True)
class FixtureInventory:
    fixture_id: str
    role: str
    candidates: tuple[str, ...]
    expected_target_count: int
    anchor_occurrence_map: tuple[int, ...]


FIXTURE_INVENTORIES = (
    FixtureInventory(
        fixture_id="phrase-medial-continuous",
        role="one phrase-medial target with both padded splice edges in speech",
        candidates=(
            "The rabbit moves quietly.",
            "A lantern glows beside us.",
            "The captain speaks softly.",
            "Our cabin feels peaceful.",
        ),
        expected_target_count=1,
        anchor_occurrence_map=(0,),
    ),
    FixtureInventory(
        fixture_id="phrase-final-new-context",
        role="one phrase-final target in a target carrier context absent from the parent confirmation",
        candidates=(
            "We drift quietly toward the map.",
            "The soft glow reaches the cap.",
            "We move slowly toward the lamp.",
            "They speak softly near the badge.",
        ),
        expected_target_count=1,
        anchor_occurrence_map=(1,),
    ),
    FixtureInventory(
        fixture_id="multiple-repeated-target",
        role="multiple non-overlapping targets including one repeated source word",
        candidates=(
            "The rabbit follows the rabbit.",
            "A rabbit follows a rabbit.",
            "The captain follows the captain.",
            "Our cabin mirrors our cabin.",
        ),
        expected_target_count=2,
        anchor_occurrence_map=(0, 1),
    ),
)


EXPECTED_SELECTIONS = {
    "phrase-medial-continuous": {
        "text": "The rabbit moves quietly.",
        "plan_sha256": "3d06a73449caf7128127d153ef26481b2bc953631f3645aca4d9dd8df8a0f735",
    },
    "phrase-final-new-context": {
        "text": "We drift quietly toward the map.",
        "plan_sha256": "a83641911d70e8fab61bae60eda5df5d702e5b86bf91198406b8fb085f5695f7",
    },
    "multiple-repeated-target": {
        "text": "The rabbit follows the rabbit.",
        "plan_sha256": "5f0d618bbc79d844665bfd25bbf571a4b0beee62e15a77228ebb24599997e59e",
    },
}


def run_dir() -> Path:
    return Paths().artifacts / "typed-engine" / RUN_ID


def parent_splice_dir() -> Path:
    return Paths().artifacts / "typed-engine" / PARENT_SPLICE_RUN_ID


def _load_json(path: Path) -> Any:
    if not path.is_file():
        raise RuntimeError(f"required artifact is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _write_once_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        if stable_json(_load_json(path)) != stable_json(payload):
            raise RuntimeError(f"immutable artifact differs from recomputation: {path}")
        return
    atomic_write_json(path, payload)


def _write_once_bytes(path: Path, payload: bytes) -> None:
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError(f"immutable bytes differ from existing artifact: {path}")
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


def _prior_artifact_snapshot() -> tuple[list[tuple[Path, bytes]], dict[str, Any]]:
    files: list[tuple[Path, bytes]] = []
    rows: list[dict[str, str]] = []
    for path in sorted(Paths().artifacts.rglob("*.json")):
        if path.is_relative_to(run_dir()):
            continue
        payload = path.read_bytes()
        files.append((path, payload))
        rows.append(
            {
                "relative_path": str(path.relative_to(Paths().root)),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    return files, {
        "json_file_count": len(rows),
        "inventory_sha256": sha256_json(rows),
    }


def _fixture_shape_reasons(
    inventory: FixtureInventory, plan: TypedPlan
) -> list[str]:
    reasons: list[str] = []
    if not plan.comparison_available:
        reasons.append("comparison_unavailable")
    if not plan.gate_summary.espeak_gate_pass:
        reasons.append("espeak_gate_failed")
    if not plan.gate_summary.kokoro_phone_gate_pass:
        reasons.append("kokoro_phone_gate_failed")
    if plan.target_occurrence_count != inventory.expected_target_count:
        reasons.append("wrong_target_occurrence_count")
    indexes = plan.target_word_indexes
    if inventory.fixture_id == "phrase-medial-continuous":
        if len(indexes) != 1 or not (0 < indexes[0] < len(plan.words) - 1):
            reasons.append("target_not_phrase_medial")
    elif inventory.fixture_id == "phrase-final-new-context":
        if len(indexes) != 1 or indexes[0] != len(plan.words) - 1:
            reasons.append("target_not_phrase_final")
        if indexes:
            word = plan.words[indexes[0]]
            if (word.neutral_phone, word.lens_phone) == ("vˈæʒ", "vˈɛʒ"):
                reasons.append("parent_target_carrier_context_reused")
    elif inventory.fixture_id == "multiple-repeated-target":
        if len(indexes) < 2 or len(set(indexes)) != len(indexes):
            reasons.append("targets_not_multiple_nonoverlapping_words")
        else:
            target_words = [plan.words[index] for index in indexes]
            source_groups: dict[str, list[Any]] = {}
            for word in target_words:
                source_groups.setdefault(word.source.casefold(), []).append(word)
            repeated = next(
                (rows for rows in source_groups.values() if len(rows) >= 2), None
            )
            if repeated is None:
                reasons.append("no_repeated_source_target")
            elif any(
                (word.neutral_phone, word.lens_phone)
                != (repeated[0].neutral_phone, repeated[0].lens_phone)
                for word in repeated[1:]
            ):
                reasons.append("repeated_source_mapping_drift")
    return reasons


def _select_fixtures() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    planner = KokoroTypedPlanner.load()
    prior_files, snapshot = _prior_artifact_snapshot()
    selected: list[dict[str, Any]] = []
    for inventory in FIXTURE_INVENTORIES:
        attempts: list[dict[str, Any]] = []
        winner: dict[str, Any] | None = None
        for order, text in enumerate(inventory.candidates, start=1):
            try:
                first = planner.plan(text)
                second = planner.plan(text)
            except Exception as exc:
                attempts.append(
                    {
                        "order": order,
                        "text": text,
                        "gate_clean": False,
                        "rejection_reasons": [f"{type(exc).__name__}: {exc}"],
                    }
                )
                continue
            reasons = _fixture_shape_reasons(inventory, first)
            if stable_json(asdict(first)) != stable_json(asdict(second)):
                reasons.append("plan_not_deterministic")
            text_bytes = text.encode("utf-8")
            plan_bytes = first.plan_sha256.encode("ascii")
            if any(text_bytes in payload for _, payload in prior_files):
                reasons.append("source_text_present_in_prior_artifacts")
            if any(plan_bytes in payload for _, payload in prior_files):
                reasons.append("plan_hash_present_in_prior_artifacts")
            attempt = {
                "order": order,
                "text": text,
                "plan_sha256": first.plan_sha256,
                "gate_clean": not reasons,
                "rejection_reasons": reasons,
                "gate_summary": asdict(first.gate_summary),
            }
            attempts.append(attempt)
            if not reasons:
                winner = {
                    "fixture_id": inventory.fixture_id,
                    "role": inventory.role,
                    "inventory": list(inventory.candidates),
                    "candidate_attempts": attempts,
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
                    "anchor_occurrence_map": list(inventory.anchor_occurrence_map),
                    "words": [asdict(word) for word in first.words],
                    "gate_summary": asdict(first.gate_summary),
                    "previously_unheard_artifact_check": {
                        "source_text_absent": True,
                        "plan_hash_absent": True,
                    },
                }
                break
        if winner is None:
            raise RuntimeError(
                f"no gate-clean candidate in frozen inventory: {inventory.fixture_id}"
            )
        expected = EXPECTED_SELECTIONS[inventory.fixture_id]
        if winner["text"] != expected["text"] or winner["plan_sha256"] != expected[
            "plan_sha256"
        ]:
            raise RuntimeError(f"deterministic selection drifted: {inventory.fixture_id}")
        selected.append(winner)
    return selected, snapshot


def _verified_parent_splice() -> dict[str, Any]:
    protocol_path = parent_splice_dir() / "protocol.json"
    analysis_path = parent_splice_dir() / "analysis.json"
    adjudication_path = parent_splice_dir() / "adjudication.json"
    analysis = _load_json(analysis_path)
    adjudication = _load_json(adjudication_path)
    if adjudication.get("classification") != "candidate_succeeds_both_known_fixtures":
        raise RuntimeError("parent splice adjudication no longer authorizes confirmation")
    if adjudication.get("eligible_for_one_unseen_confirmation") is not True:
        raise RuntimeError("parent splice is not eligible for unseen confirmation")
    if adjudication.get("production_integration_authorized") is not False:
        raise RuntimeError("parent splice unexpectedly claims production integration")
    return {
        "run_id": PARENT_SPLICE_RUN_ID,
        "protocol_file_sha256": sha256_file(protocol_path),
        "analysis_file_sha256": sha256_file(analysis_path),
        "adjudication_file_sha256": sha256_file(adjudication_path),
        "raw_classification": analysis["classification"],
        "adjudicated_classification": adjudication["classification"],
        "eligibility": "exactly_one_unseen_confirmation",
    }


def _tracked_paths(diagnostic: dict[str, Any]) -> list[str]:
    paths = [
        "src/earshift_bakeoff/kokoro_output_splice_unseen.py",
        "src/earshift_bakeoff/kokoro_output_domain_splice.py",
        "src/earshift_bakeoff/kokoro_synthesis.py",
        "src/earshift_bakeoff/kokoro_typed_engine.py",
        "src/earshift_bakeoff/kokoro_typed_confirmation.py",
        "src/earshift_bakeoff/kokoro_typed_confirmation_protocol.py",
        "src/earshift_bakeoff/kokoro_typed_diagnostic.py",
        "scripts/run_kokoro_output_splice_unseen_v1.py",
        "scripts/praat_sentence_pair_v2_burg.praat",
        "uv.lock",
        f"artifacts/typed-engine/{RUN_ID}/{PROTOCOL_FILE}",
        f"artifacts/typed-engine/{PARENT_SPLICE_RUN_ID}/protocol.json",
        f"artifacts/typed-engine/{PARENT_SPLICE_RUN_ID}/analysis.json",
        f"artifacts/typed-engine/{PARENT_SPLICE_RUN_ID}/adjudication.json",
    ]
    paths.extend(row["relative_path"] for row in diagnostic["bound_output_files"])
    return sorted(set(paths))


def _logical_trials() -> list[dict[str, Any]]:
    return [
        *(
            {
                "fixture_id": inventory.fixture_id,
                "condition": "identity-control",
                "roles": ["neutral", "identity"],
            }
            for inventory in FIXTURE_INVENTORIES
        ),
        *(
            {
                "fixture_id": inventory.fixture_id,
                "condition": "spliced-lens",
                "roles": ["neutral", "lens"],
            }
            for inventory in FIXTURE_INVENTORIES
        ),
    ]


def blinded_trial_plan() -> list[dict[str, Any]]:
    rng = random.Random(BLIND_SEED)
    trials = _logical_trials()
    for trial in trials:
        roles = list(trial.pop("roles"))
        rng.shuffle(roles)
        trial["side_roles"] = dict(zip(("A", "B"), roles, strict=True))
    rng.shuffle(trials)
    return [
        {**trial, "trial_id": f"comparison-{index:02d}"}
        for index, trial in enumerate(trials, start=1)
    ]


def protocol_record() -> dict[str, Any]:
    fixtures, prior_snapshot = _select_fixtures()
    diagnostic = _verified_diagnostic_parent()
    parent_splice = _verified_parent_splice()
    root = Paths().root
    code_paths = {
        "confirmation_measurement": root
        / "src/earshift_bakeoff/kokoro_typed_confirmation.py",
        "output_domain_splice": root
        / "src/earshift_bakeoff/kokoro_output_domain_splice.py",
        "protocol_and_runner": root
        / "src/earshift_bakeoff/kokoro_output_splice_unseen.py",
        "synthesis": root / "src/earshift_bakeoff/kokoro_synthesis.py",
        "typed_planner": root / "src/earshift_bakeoff/kokoro_typed_engine.py",
    }
    payload: dict[str, Any] = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "status": "frozen_before_any_unseen_decode",
        "question": (
            "Does unchanged output-domain-splice-v1 pass the complete automatic and "
            "creator blind-QC gates on three deterministically selected unseen fixtures?"
        ),
        "scope": {
            "fixture_count": 3,
            "logical_output_roles": ["neutral", "identity", "lens"],
            "decoder_source_roles": ["neutral", "identity", "full-state-lens-source"],
            "decoder_attempt_ceiling": 9,
            "derived_spliced_lens_count": 3,
            "api_calls": 0,
            "replacement_fixtures": 0,
            "selective_rerenders": 0,
            "listening_selection": 0,
        },
        "parents": {
            "output_domain_splice_v1": parent_splice,
            "diagnostic_anchor_geometry": diagnostic,
        },
        "previously_unheard_inventory": prior_snapshot,
        "fixtures": fixtures,
        "render_manifest": [
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
        "intervention": {
            "name": "output-domain-splice-v1",
            "boundary_rule": (
                "merge each target word's source-derived alignment interval padded by "
                "exactly 150 ms per edge and clipped to the decoded sample range"
            ),
            "taper": {
                "kind": "raised cosine",
                "milliseconds_each_edge": TAPER_MS,
                "samples_each_edge": TAPER_SAMPLES,
            },
            "mixing_formula": "round(neutral + weight * (full_state_lens - neutral))",
            "unchanged_parent_implementation_sha256": sha256_file(
                code_paths["output_domain_splice"]
            ),
            "phrase_medial_edge_rule": (
                "the padded start must lie strictly inside the aligned preceding word and "
                "the padded end strictly inside the aligned following word"
            ),
        },
        "automatic_gate": {
            "required": [
                "exact deterministic planner and renderer runtime gates",
                "neutral and identity PCM bit identity",
                "equal finite nonempty unclipped mono PCM16 24 kHz audio",
                "candidate exact-neutral identity wherever splice weight is zero",
                "candidate exact-full-state-lens identity wherever splice weight is one",
                "all frozen click metrics at every splice edge",
                "primary 50 percent acoustic gate at every occurrence and 5500/5750/6000 Hz ceiling",
                "localization runtime median <=5 ms and p95 <=10 ms with identical and mismatched inputs rejected",
                "phrase-medial splice edges satisfy the frozen aligned-neighbor rule",
            ],
            "primary_window_percent": PRIMARY_WINDOW_PERCENT,
            "descriptive_window_percents": list(DESCRIPTIVE_WINDOW_PERCENTS),
            "descriptive_only": (
                "40 and 60 percent results are reported for sensitivity and may neither "
                "fail nor rescue the primary outcome"
            ),
            "localization": {
                "minimum": LOCALIZATION_MINIMUM,
                "expected_fraction": 1.0,
                "expectation_reason": (
                    "100 percent is expected by construction because the candidate is "
                    "exactly neutral outside the scoring windows"
                ),
            },
            "boundary_metrics": {
                "context_ms_each_side": BOUNDARY_CONTEXT_MS,
                "maximum_edge_delta_step_pcm": MAX_EDGE_DELTA_STEP_PCM,
                "maximum_derivative_ratio": MAX_BOUNDARY_DERIVATIVE_RATIO,
            },
            "runtime_benchmark": {
                "warmup_iterations": BENCHMARK_WARMUP_ITERATIONS,
                "measured_iterations": BENCHMARK_MEASURED_ITERATIONS,
                "maximum_median_ms": MAX_LOCALIZATION_MEDIAN_MS,
                "maximum_p95_ms": MAX_LOCALIZATION_P95_MS,
            },
        },
        "blind_review": {
            "only_after_every_automatic_gate_passes": True,
            "seed": BLIND_SEED,
            "trial_count": 6,
            "layout": blinded_trial_plan(),
            "hidden": [
                "source sentence",
                "carrier script and phones",
                "fixture and condition labels",
                "audio roles and filenames",
            ],
            "same_synchronized_target_cue_every_condition": True,
            "manual_gate": {
                "identity": (
                    "both side gates pass; strength 1; category same or neither; no "
                    "dominant unrelated interference"
                ),
                "lens": (
                    "both side gates pass; correct /ae/ to /eh/ direction; strength >=5; "
                    "confidence >=3; no dominant unrelated interference"
                ),
                "per_side": (
                    "naturalness >=4; sentence-like delivery; no stable recoverable "
                    "meaning; no major artifact"
                ),
                "run": "both trials pass for all three fixtures",
            },
            "response_filename": RESPONSE_FILENAME,
        },
        "implementation": {
            "source_file_sha256": {
                name: sha256_file(path) for name, path in sorted(code_paths.items())
            },
            "measurement": {
                "praat_path": str(PRAAT),
                "praat_sha256": sha256_file(PRAAT),
                "script_relative_path": str(MEASUREMENT_SCRIPT.relative_to(root)),
                "script_sha256": sha256_file(MEASUREMENT_SCRIPT),
                "ceiling_hz_family": list(CEILINGS_HZ),
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
                "device": "cpu",
            },
            "tracked_clean_paths": _tracked_paths(diagnostic),
        },
        "outcomes": {
            "automatic_pass": "build one frozen blind review and continue disabled service integration",
            "automatic_failure": "preserve result, diagnose exact failed mechanism, no review",
            "human_pass": "eligible for local typed-path integration behind the disabled flag",
            "human_failure": "preserve result and permit only a separately versioned bounded correction",
        },
        "stopping_rule": (
            "Attempt exactly the nine decoder slots once. Never replace a fixture, rerender, "
            "select, tune, or change this protocol after decode begins. Descriptive 40/60 "
            "windows cannot change the outcome."
        ),
    }
    return {**payload, "protocol_sha256": sha256_json(payload)}


def prepare() -> dict[str, Any]:
    protocol = protocol_record()
    destination = run_dir() / PROTOCOL_FILE
    if destination.exists():
        if stable_json(_load_json(destination)) != stable_json(protocol):
            raise RuntimeError("existing protocol differs from deterministic freeze")
    else:
        forbidden = [
            run_dir() / RECORDS_FILE,
            run_dir() / ANALYSIS_FILE,
            run_dir() / "audio",
            run_dir() / ATTEMPT_DIR,
            run_dir() / REVIEW_FILE,
        ]
        if any(path.exists() for path in forbidden):
            raise RuntimeError("unseen output exists before protocol freeze")
        atomic_write_json(destination, protocol)
    return protocol


def _checked_protocol() -> dict[str, Any]:
    frozen = _load_json(run_dir() / PROTOCOL_FILE)
    if stable_json(frozen) != stable_json(protocol_record()):
        raise RuntimeError("frozen unseen protocol differs from its bound implementation")
    return frozen


def _require_committed_inputs(protocol: dict[str, Any]) -> str:
    paths = protocol["implementation"]["tracked_clean_paths"]
    subprocess.run(
        ["git", "ls-files", "--error-unmatch", "--", *paths],
        cwd=Paths().root,
        check=True,
        capture_output=True,
        text=True,
    )
    clean = subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--", *paths],
        cwd=Paths().root,
        check=False,
    )
    if clean.returncode:
        raise RuntimeError("unseen protocol inputs differ from committed HEAD")
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Paths().root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _read_pcm(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as handle:
        if (
            handle.getnchannels() != 1
            or handle.getsampwidth() != 2
            or handle.getframerate() != SAMPLE_RATE_HZ
        ):
            raise RuntimeError(f"WAV violates mono PCM16/24k contract: {path}")
        values = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2").copy()
    if not values.size:
        raise RuntimeError(f"WAV is empty: {path}")
    return values


def _write_pcm_once(path: Path, values: np.ndarray) -> None:
    if path.exists():
        raise RuntimeError(f"one-attempt WAV exists: {path}")
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


def _audio_record(path: Path, values: np.ndarray) -> dict[str, Any]:
    pcm = np.asarray(values, dtype="<i2").reshape(-1)
    clipped_fraction = float(np.mean(np.abs(pcm.astype(np.int64)) >= 32767))
    return {
        "relative_path": str(path.relative_to(run_dir())),
        "sample_count": int(pcm.size),
        "finite": bool(pcm.size and np.isfinite(pcm.astype(float)).all()),
        "clipped_fraction": clipped_fraction,
        "clipping_pass": clipped_fraction < MAX_CLIPPED_FRACTION,
        "pcm_sha256": hashlib.sha256(pcm.tobytes()).hexdigest(),
        "wav_sha256": sha256_file(path),
    }


def _word_intervals(
    model: Any, plan: TypedPlan, durations: Sequence[int], sample_count: int
) -> list[dict[str, Any]]:
    total_frames = sum(int(value) for value in durations)
    if total_frames <= 0 or sample_count % total_frames:
        raise RuntimeError("decoded samples do not map to integral alignment frames")
    samples_per_frame = sample_count // total_frames
    spans = _word_column_spans(model, plan.neutral_phonemes)
    if len(spans) != len(plan.words):
        raise RuntimeError("word alignment count drifted")
    cumulative = np.concatenate(([0], np.cumsum(np.asarray(durations, dtype=int))))
    result: list[dict[str, Any]] = []
    for word, columns in zip(plan.words, spans, strict=True):
        start = int(cumulative[columns[0]] * samples_per_frame)
        end = int(cumulative[columns[-1] + 1] * samples_per_frame)
        result.append(
            {
                "word_index": word.word_index,
                "start_sample": start,
                "end_sample_exclusive": end,
                "start_s": start / SAMPLE_RATE_HZ,
                "end_s": end / SAMPLE_RATE_HZ,
            }
        )
    return result


def phrase_medial_edge_gate(
    target_word_index: int,
    all_word_intervals: Sequence[dict[str, Any]],
    splice_window: dict[str, Any],
) -> dict[str, Any]:
    if not 0 < target_word_index < len(all_word_intervals) - 1:
        return {"pass": False, "reason": "target_lacks_two_aligned_neighbors"}
    previous = all_word_intervals[target_word_index - 1]
    following = all_word_intervals[target_word_index + 1]
    start = int(splice_window["start_sample"])
    end = int(splice_window["end_sample_exclusive"])
    start_inside = previous["start_sample"] < start < previous["end_sample_exclusive"]
    end_inside = following["start_sample"] < end < following["end_sample_exclusive"]
    return {
        "rule": "each padded splice edge lies strictly inside its aligned neighboring word",
        "target_word_index": target_word_index,
        "start_edge_sample": start,
        "end_edge_sample": end,
        "preceding_word_interval": previous,
        "following_word_interval": following,
        "start_inside_preceding_word": start_inside,
        "end_inside_following_word": end_inside,
        "pass": bool(start_inside and end_inside),
    }


def _begin_attempts(records: dict[str, Any], fixture_id: str) -> list[dict[str, Any]]:
    slots = [row for row in records["slots"] if row["fixture_id"] == fixture_id]
    if any((run_dir() / ATTEMPT_DIR / f"{row['slot_id']}.json").exists() for row in slots):
        raise RuntimeError(f"fixture already consumed its one attempt: {fixture_id}")
    for row in slots:
        marker = run_dir() / ATTEMPT_DIR / f"{row['slot_id']}.json"
        atomic_write_json(
            marker,
            {
                "schema_version": 1,
                "run_id": RUN_ID,
                "protocol_sha256": records["protocol_sha256"],
                "slot_id": row["slot_id"],
                "one_attempt_no_retry": True,
            },
        )
        row["status"] = "attempt_started"
    atomic_write_json(run_dir() / RECORDS_FILE, records)
    return slots


def _initial_records(protocol: dict[str, Any], commit: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "run_id": RUN_ID,
        "protocol_sha256": protocol["protocol_sha256"],
        "implementation_commit": commit,
        "status": "in_progress",
        "api_calls_made": 0,
        "decoder_attempt_count": 0,
        "slots": [
            {**row, "status": "pending"} for row in protocol["render_manifest"]
        ],
        "fixtures": [],
    }


def _render(protocol: dict[str, Any], records: dict[str, Any]) -> None:
    from .kokoro_synthesis import KokoroSynthesisRuntime

    planner = KokoroTypedPlanner.load()
    runtime = KokoroSynthesisRuntime.load(download=False)
    for frozen in protocol["fixtures"]:
        fixture_id = frozen["fixture_id"]
        slots = _begin_attempts(records, fixture_id)
        try:
            plan = planner.plan(frozen["text"])
            if plan.plan_sha256 != frozen["plan_sha256"]:
                raise RuntimeError("typed plan drifted after protocol freeze")
            pair_plan = plan.pair_plan()
            if pair_plan is None:
                raise RuntimeError("frozen fixture lost its pair")
            rendered = runtime.render_parity_triplet(pair_plan)
            expected_columns = target_word_columns(
                runtime.model, plan.neutral_phonemes, plan.target_word_indexes
            )
            if rendered.replaced_columns != expected_columns:
                raise RuntimeError("renderer replaced unexpected state columns")
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
            target_intervals = [
                row["interval"] for row in alignment["target_words"]
            ]
            untouched_localization = localization_report(
                neutral, full_lens, target_intervals
            )
            splice_windows = untouched_localization["inside_windows"]
            lens, weights = output_domain_splice(neutral, full_lens, splice_windows)
            word_intervals = _word_intervals(
                runtime.model,
                plan,
                rendered.predicted_durations,
                neutral.size,
            )
            edge_gate = (
                phrase_medial_edge_gate(
                    plan.target_word_indexes[0], word_intervals, splice_windows[0]
                )
                if fixture_id == "phrase-medial-continuous"
                else {"pass": True, "reason": "not_the_phrase_medial_fixture"}
            )
            audio_values = {
                "neutral": neutral,
                "identity": identity,
                "full-state-lens-source": full_lens,
                "lens": lens,
            }
            audio: dict[str, Any] = {}
            for role, values in audio_values.items():
                path = run_dir() / "audio" / f"{fixture_id}__{role}.wav"
                _write_pcm_once(path, values)
                audio[role] = _audio_record(path, values)
            integrity = inspect_render(
                PairRender(
                    neutral=rendered.neutral,
                    lens=rendered.lens,
                    predicted_durations=rendered.predicted_durations,
                    replaced_columns=rendered.replaced_columns,
                )
            )
            runtime_checks = {
                "plan_exact": plan.plan_sha256 == frozen["plan_sha256"],
                "target_columns_exact": rendered.replaced_columns == expected_columns,
                "raw_renderer_integrity": integrity.pass_all,
                "neutral_identity_bit_identical": np.array_equal(neutral, identity),
                "equal_sample_count": len({row.size for row in audio_values.values()}) == 1,
                "finite_nonempty_unclipped": all(
                    row["finite"] and row["clipping_pass"]
                    for row in audio.values()
                ),
                "outside_windows_exact_neutral": np.array_equal(
                    lens[weights == 0.0], neutral[weights == 0.0]
                ),
                "full_weight_interior_exact_full_lens": bool(
                    np.any(weights == 1.0)
                    and np.array_equal(
                        lens[weights == 1.0], full_lens[weights == 1.0]
                    )
                ),
            }
            record = {
                "fixture_id": fixture_id,
                "plan_sha256": plan.plan_sha256,
                "safe_plan_metadata": plan.safe_metadata(),
                "predicted_durations": list(rendered.predicted_durations),
                "replaced_columns": list(rendered.replaced_columns),
                "alignment": alignment,
                "all_word_intervals": word_intervals,
                "splice_windows": splice_windows,
                "phrase_medial_edge_gate": edge_gate,
                "untouched_full_state_localization": untouched_localization,
                "audio": audio,
                "runtime_checks": runtime_checks,
                "runtime_pass": all(runtime_checks.values()),
            }
            records["fixtures"].append(record)
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
    records["all_runtime_gates_pass"] = bool(
        len(records["fixtures"]) == len(FIXTURE_INVENTORIES)
        and all(row["runtime_pass"] for row in records["fixtures"])
    )
    atomic_write_json(run_dir() / RECORDS_FILE, records)


def _acoustic_report(
    neutral_path: Path,
    lens_path: Path,
    occurrences: Sequence[dict[str, Any]],
    anchors: dict[str, Any],
) -> dict[str, Any]:
    neutral_measurements = _measure_occurrences(neutral_path, occurrences)
    lens_measurements = _measure_occurrences(lens_path, occurrences)
    windows: dict[str, Any] = {}
    for percent in WINDOW_PERCENTS:
        key = str(percent)
        rows: list[dict[str, Any]] = []
        for index, occurrence in enumerate(occurrences):
            anchor_index = int(occurrence["anchor_occurrence_index"])
            families: dict[str, Any] = {}
            for ceiling in CEILINGS_HZ:
                ceiling_key = str(ceiling)
                families[ceiling_key] = _family_gate(
                    neutral_measurements[index]["families"][ceiling_key][key],
                    lens_measurements[index]["families"][ceiling_key][key],
                    anchors[key]["occurrences"][anchor_index]["families"][ceiling_key],
                )
            rows.append(
                {
                    "occurrence_index": index,
                    "anchor_occurrence_index": anchor_index,
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
    audio_paths = {
        role: run_dir() / row["relative_path"] for role, row in record["audio"].items()
    }
    for role, path in audio_paths.items():
        if sha256_file(path) != record["audio"][role]["wav_sha256"]:
            raise RuntimeError(f"WAV hash drifted: {role}")
    neutral = _read_pcm(audio_paths["neutral"])
    identity = _read_pcm(audio_paths["identity"])
    full_lens = _read_pcm(audio_paths["full-state-lens-source"])
    lens = _read_pcm(audio_paths["lens"])
    _, weights = output_domain_splice(neutral, full_lens, record["splice_windows"])
    integrity_checks = {
        **record["runtime_checks"],
        "neutral_identity_bit_identical_recheck": np.array_equal(neutral, identity),
        "outside_windows_exact_neutral_recheck": np.array_equal(
            lens[weights == 0.0], neutral[weights == 0.0]
        ),
        "full_weight_interior_exact_full_lens_recheck": bool(
            np.any(weights == 1.0)
            and np.array_equal(lens[weights == 1.0], full_lens[weights == 1.0])
        ),
    }
    boundary = boundary_artifact_report(
        neutral, full_lens, lens, record["splice_windows"]
    )
    target_intervals = [
        row["interval"] for row in record["alignment"]["target_words"]
    ]
    localization = localization_report(neutral, lens, target_intervals)
    benchmark = _benchmark_localization(neutral, lens, target_intervals)
    acoustic = _acoustic_report(
        audio_paths["neutral"],
        audio_paths["lens"],
        record["alignment"]["target_occurrences"],
        protocol["parents"]["diagnostic_anchor_geometry"]["local_anchor_geometry"],
    )
    automatic_checks = {
        "runtime_and_exact_pcm_integrity": all(integrity_checks.values()),
        "phrase_medial_edge_rule": bool(record["phrase_medial_edge_gate"]["pass"]),
        "boundary_click_metrics": bool(boundary["pass"]),
        "primary_50_acoustic_gate": bool(acoustic["primary_gate_pass"]),
        "localization_at_least_0_80": bool(localization["pass"]),
        "localization_runtime_cheap_fail_closed": bool(benchmark["pass"]),
    }
    return {
        "fixture_id": record["fixture_id"],
        "integrity_checks": integrity_checks,
        "phrase_medial_edge_gate": record["phrase_medial_edge_gate"],
        "boundary_artifact": boundary,
        "acoustic": acoustic,
        "untouched_full_state_localization": record[
            "untouched_full_state_localization"
        ],
        "spliced_localization": {
            **localization,
            "expected_by_construction": True,
            "expectation_reason": (
                "the candidate is exact-neutral outside the scoring windows"
            ),
        },
        "localization_runtime_benchmark": benchmark,
        "automatic_checks": automatic_checks,
        "automatic_pass": all(automatic_checks.values()),
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
    automatic_pass = bool(
        not failures
        and len(fixtures) == len(FIXTURE_INVENTORIES)
        and all(row["automatic_pass"] for row in fixtures)
    )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "protocol_sha256": protocol["protocol_sha256"],
        "status": "analysis_complete",
        "classification": (
            "unseen_output_splice_automatic_pass_pending_human_qc"
            if automatic_pass
            else (
                "unseen_output_splice_measurement_inconclusive"
                if failures
                else "unseen_output_splice_automatic_failed"
            )
        ),
        "automatic_pass": automatic_pass,
        "pending_human_review": automatic_pass,
        "fixture_count": len(fixtures),
        "measurement_failures": failures,
        "fixtures": fixtures,
        "descriptive_40_60_windows_cannot_change_outcome": True,
        "localization_100_percent_expected_by_construction": True,
        "api_calls_made": 0,
        "decoder_attempt_count": records["decoder_attempt_count"],
        "render_records_sha256": sha256_file(run_dir() / RECORDS_FILE),
        "production_integration_authorized": False,
    }
    return {**payload, "analysis_sha256": sha256_json(payload)}


def automatic_outcome(
    primary_pass: bool, descriptive_40_pass: bool, descriptive_60_pass: bool
) -> bool:
    del descriptive_40_pass, descriptive_60_pass
    return bool(primary_pass)


def _review_audio(
    layout: Sequence[dict[str, Any]], records: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    destination = run_dir() / "review-audio"
    destination.mkdir(parents=True, exist_ok=True)
    public: list[dict[str, Any]] = []
    for trial in layout:
        record = records[trial["fixture_id"]]
        sides: list[dict[str, str]] = []
        for side, role in trial["side_roles"].items():
            source = run_dir() / record["audio"][role]["relative_path"]
            target = destination / f"{trial['trial_id'][-2:]}-{side.lower()}.wav"
            if target.exists():
                if sha256_file(target) != sha256_file(source):
                    raise RuntimeError("blind audio copy drifted")
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


def _review_html(public: list[dict[str, Any]], protocol_sha256: str) -> str:
    data = json.dumps(public, ensure_ascii=False).replace("</", "<\\/")
    return f"""<!doctype html><html lang=en><head><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'><title>Blind controlled-pair QC</title><style>body{{font:17px/1.45 system-ui;max-width:960px;margin:auto;padding:22px;background:#f3f0e7;color:#17221c}}section{{background:#fff;border:1px solid #d3d0c7;border-radius:15px;padding:18px;margin:15px 0}}.pair{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}.side{{margin:0}}audio,select,textarea{{width:100%;box-sizing:border-box}}label{{display:block;margin:9px 0}}.timeline{{height:15px;background:#d7ddd8;border-radius:20px;position:relative;margin:8px 0}}.cue{{position:absolute;height:100%;background:#db7c37}}.head{{position:absolute;width:2px;height:100%;background:#164e3d}}.active{{outline:3px solid #db7c37}}button{{padding:12px 20px;border:0;border-radius:24px;background:#164e3d;color:white;font-weight:700}}button:disabled{{opacity:.4}}@media(max-width:700px){{.pair{{grid-template-columns:1fr}}}}</style></head><body><h1>Blind controlled-pair QC</h1><p>Six randomized comparisons cover three previously unheard structures. Conditions, spellings, sentences, roles, and filenames are hidden. The orange cue is shown identically in every condition.</p><div id=trials></div><button id=download disabled>Download review JSON</button><script>const R={data};const RUN='{RUN_ID}',P='{protocol_sha256}',F='{RESPONSE_FILENAME}',K='kokoro-output-splice-unseen-v1';const S=JSON.parse(localStorage.getItem(K)||'{{}}');S.id??=crypto.randomUUID();S.r??={{}};const O=(v,l=v)=>`<option value="${{v}}">${{l}}</option>`;const Sel=(id,scope,side,field,label,opts)=>`<label>${{label}}<select data-id="${{id}}" data-scope="${{scope}}" data-side="${{side}}" data-field="${{field}}"><option value="">—</option>${{opts}}</select></label>`;const state=id=>(S.r[id]??={{sides:{{A:{{}},B:{{}}}},pair:{{}},plays:{{A:0,B:0}}}});const side=(t,s)=>`<section class=side><h3>Clip ${{s.side}}</h3><audio controls src="${{s.audio}}" data-id="${{t.trial_id}}" data-side="${{s.side}}"></audio><div class=timeline>${{t.target_intervals.map(x=>`<i class=cue style="left:${{100*x.start_s/t.duration_s}}%;width:${{100*(x.end_s-x.start_s)/t.duration_s}}%"></i>`).join('')}}<i class=head></i></div>${{Sel(t.trial_id,'side',s.side,'naturalness','Naturalness (1–5)',[1,2,3,4,5].map(x=>O(x)).join(''))}}${{Sel(t.trial_id,'side',s.side,'delivery','Delivery',O('sentence-like','Sentence-like')+O('slightly-list-like','Slightly list-like')+O('dominantly-list-like','Dominantly list-like')+O('other','Other'))}}${{Sel(t.trial_id,'side',s.side,'meaning','Stable recoverable meaning',O('none','None')+O('isolated-word','Isolated possible word')+O('coherent','Coherent phrase/sentence'))}}${{Sel(t.trial_id,'side',s.side,'artifact','Artifact',O('none','None')+O('minor','Minor')+O('major','Major')+O('uncertain','Uncertain'))}}</section>`;document.getElementById('trials').innerHTML=R.map((t,i)=>`<section><h2>Comparison ${{i+1}} of ${{R.length}}</h2><div class=pair>${{t.sides.map(s=>side(t,s)).join('')}}</div>${{Sel(t.trial_id,'pair','','strength','Difference strength (1 none · 7 strong)',[1,2,3,4,5,6,7].map(x=>O(x)).join(''))}}${{Sel(t.trial_id,'pair','','direction','Which side is closer to the vowel in “bet”?',O('A')+O('B')+O('same','Same')+O('neither','Neither')+O('uncertain','Uncertain'))}}${{Sel(t.trial_id,'pair','','confidence','Confidence (1–5)',[1,2,3,4,5].map(x=>O(x)).join(''))}}${{Sel(t.trial_id,'pair','','interference','Unrelated delivery interference',O('none','None')+O('manageable','Manageable')+O('dominant','Dominant')+O('uncertain','Uncertain'))}}<label>Notes<textarea data-id="${{t.trial_id}}" data-scope=pair data-side="" data-field=notes></textarea></label><p>Play starts: <b data-count="${{t.trial_id}}">0</b></p></section>`).join('');const fields=['naturalness','delivery','meaning','artifact'],pairs=['strength','direction','confidence','interference'];function update(){{for(const t of R){{const x=state(t.trial_id);document.querySelector(`[data-count="${{t.trial_id}}"]`).textContent=x.plays.A+x.plays.B}}localStorage.setItem(K,JSON.stringify(S));document.getElementById('download').disabled=!R.every(t=>{{const x=state(t.trial_id);return ['A','B'].every(s=>fields.every(f=>String(x.sides[s][f]??'')!==''))&&pairs.every(f=>String(x.pair[f]??'')!=='')}})}}document.querySelectorAll('[data-field]').forEach(e=>{{const x=state(e.dataset.id),o=e.dataset.scope==='side'?x.sides[e.dataset.side]:x.pair;e.value=o[e.dataset.field]??'';e.oninput=()=>{{o[e.dataset.field]=e.value;update()}}}});document.querySelectorAll('audio').forEach(a=>{{const t=R.find(x=>x.trial_id===a.dataset.id),line=a.nextElementSibling,head=line.querySelector('.head');a.onplay=()=>{{document.querySelectorAll('audio').forEach(b=>{{if(b!==a)b.pause()}});state(a.dataset.id).plays[a.dataset.side]++;update()}};a.ontimeupdate=()=>{{head.style.left=`${{100*a.currentTime/a.duration}}%`;line.classList.toggle('active',t.target_intervals.some(x=>a.currentTime>=x.start_s&&a.currentTime<=x.end_s)&&!a.paused)}}}});document.getElementById('download').onclick=()=>{{const responses=R.map(t=>{{const x=state(t.trial_id);return{{trial_id:t.trial_id,sides:x.sides,difference_strength:+x.pair.strength,category_judgment:x.pair.direction,confidence:+x.pair.confidence,interference:x.pair.interference,notes:x.pair.notes||'',play_starts:x.plays,replay_count:x.plays.A+x.plays.B}}}});const p={{schema_version:1,run_id:RUN,protocol_sha256:P,session_id:S.id,saved_at:new Date().toISOString(),responses}},b=new Blob([JSON.stringify(p,null,2)+'\\n'],{{type:'application/json'}}),l=document.createElement('a');l.href=URL.createObjectURL(b);l.download=F;l.click()}};update();</script></body></html>"""


def build_review(protocol: dict[str, Any], analysis: dict[str, Any]) -> dict[str, Any]:
    if analysis.get("automatic_pass") is not True:
        raise RuntimeError("automatic gates did not authorize review")
    layout = blinded_trial_plan()
    records_payload = _load_json(run_dir() / RECORDS_FILE)
    records = {row["fixture_id"]: row for row in records_payload["fixtures"]}
    public = _review_audio(layout, records)
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
    html = _review_html(public, protocol["protocol_sha256"])
    path = run_dir() / REVIEW_FILE
    if path.exists() and path.read_text(encoding="utf-8") != html:
        raise RuntimeError("review HTML drifted")
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
        raise RuntimeError("Praat changed after freeze")
    if sha256_file(MEASUREMENT_SCRIPT) != protocol["implementation"]["measurement"][
        "script_sha256"
    ]:
        raise RuntimeError("measurement script changed after freeze")
    commit = _require_committed_inputs(protocol)
    records = _initial_records(protocol, commit)
    atomic_write_json(run_dir() / RECORDS_FILE, records)
    try:
        _render(protocol, records)
    except Exception as exc:
        records = _load_json(run_dir() / RECORDS_FILE)
        payload = {
            "schema_version": 1,
            "run_id": RUN_ID,
            "protocol_sha256": protocol["protocol_sha256"],
            "classification": "unseen_output_splice_runtime_inconclusive",
            "automatic_pass": False,
            "pending_human_review": False,
            "failure": f"{type(exc).__name__}: {exc}"[:1000],
            "render_records_sha256": sha256_file(run_dir() / RECORDS_FILE),
        }
        _write_once_json(analysis_path, payload)
        return payload
    records = _load_json(run_dir() / RECORDS_FILE)
    analysis = _analysis(protocol, records)
    _write_once_json(analysis_path, analysis)
    if analysis["automatic_pass"]:
        build_review(protocol, analysis)
    return analysis


def _side_gate(side: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "naturalness_at_least_4": int(side["naturalness"]) >= 4,
        "sentence_like_delivery": side["delivery"] == "sentence-like",
        "no_stable_recoverable_meaning": side["meaning"] == "none",
        "no_major_artifact": side["artifact"] in {"none", "minor"},
    }
    return {"checks": checks, "pass": all(checks.values())}


def decode_response(path: Path) -> dict[str, Any]:
    protocol = _checked_protocol()
    analysis = _load_json(run_dir() / ANALYSIS_FILE)
    if analysis.get("automatic_pass") is not True:
        raise RuntimeError("this run is not eligible for human review")
    if path.name != RESPONSE_FILENAME:
        raise RuntimeError(f"response filename must be {RESPONSE_FILENAME}")
    raw = path.read_bytes()
    response = json.loads(raw)
    if response.get("run_id") != RUN_ID or response.get("protocol_sha256") != protocol[
        "protocol_sha256"
    ]:
        raise RuntimeError("response belongs to another run or protocol")
    key = _load_json(run_dir() / BLIND_KEY_FILE)
    keys = {row["trial_id"]: row for row in key["trials"]}
    rows = response.get("responses")
    if not isinstance(rows, list) or {row.get("trial_id") for row in rows} != set(keys):
        raise RuntimeError("response trial inventory is incomplete")
    decoded: list[dict[str, Any]] = []
    fixture_results: dict[str, dict[str, bool]] = {
        row.fixture_id: {} for row in FIXTURE_INVENTORIES
    }
    for row in rows:
        trial = keys[row["trial_id"]]
        sides = {side: _side_gate(row["sides"][side]) for side in ("A", "B")}
        side_pass = all(value["pass"] for value in sides.values())
        if trial["condition"] == "spliced-lens":
            pair_checks = {
                "strength_at_least_5": int(row["difference_strength"]) >= 5,
                "correct_ae_to_eh_direction": row["category_judgment"]
                == trial["expected_lens_side"],
                "confidence_at_least_3": int(row["confidence"]) >= 3,
                "no_dominant_interference": row["interference"]
                in {"none", "manageable"},
            }
        else:
            pair_checks = {
                "clean_identity_strength": int(row["difference_strength"]) == 1,
                "clean_identity_direction": row["category_judgment"]
                in {"same", "neither"},
                "no_dominant_interference": row["interference"]
                in {"none", "manageable"},
            }
        passed = bool(side_pass and all(pair_checks.values()))
        fixture_results[trial["fixture_id"]][trial["condition"]] = passed
        decoded.append(
            {
                "trial_id": row["trial_id"],
                "fixture_id": trial["fixture_id"],
                "condition": trial["condition"],
                "side_gates": sides,
                "pair_checks": pair_checks,
                "pass": passed,
                "replay_count": row.get("replay_count"),
                "notes": row.get("notes", ""),
            }
        )
    fixture_pass = {
        fixture_id: bool(
            conditions.get("identity-control") and conditions.get("spliced-lens")
        )
        for fixture_id, conditions in fixture_results.items()
    }
    run_pass = all(fixture_pass.values())
    result = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "protocol_sha256": protocol["protocol_sha256"],
        "status": "human_review_complete",
        "classification": (
            "unseen_output_splice_human_qc_pass"
            if run_pass
            else "unseen_output_splice_human_qc_failed"
        ),
        "run_pass": run_pass,
        "fixture_pass": fixture_pass,
        "decoded_trials": decoded,
        "raw_response_sha256": hashlib.sha256(raw).hexdigest(),
        "production_enabled": False,
    }
    _write_once_bytes(run_dir() / RAW_RESPONSE_FILE, raw)
    _write_once_json(run_dir() / MANUAL_RESULT_FILE, result)
    return result
