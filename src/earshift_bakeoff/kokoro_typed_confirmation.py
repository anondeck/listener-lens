from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import wave
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .config import Paths, stable_json
from .kokoro_synthesis import (
    SAMPLE_RATE_HZ,
    PairRender,
    _filtered_symbols,
    _word_column_spans,
    pcm16_bytes,
    pcm_sha256,
    target_word_columns,
)
from .kokoro_typed_confirmation_protocol import (
    CEILINGS_HZ,
    CONFIRMATION_FIXTURES,
    DESCRIPTIVE_WINDOW_PERCENTS,
    MEASUREMENT_SCRIPT,
    MINIMUM_DIRECTION_COSINE,
    PRAAT,
    PRIMARY_WINDOW_PERCENT,
    REVIEW_RESPONSE_FILENAME,
    RUN_ID,
    WINDOW_PERCENTS,
    blinded_trial_plan,
    protocol_record,
    run_dir,
)
from .kokoro_typed_diagnostic import localization_report, measure_interval_windows
from .kokoro_typed_engine import (
    MAX_CLIPPED_FRACTION,
    KokoroTypedPlanner,
    TypedPlan,
    inspect_render,
)
from .util import atomic_write_json, atomic_write_text, sha256_file


RECORDS_FILE = "render-records.json"
ANALYSIS_FILE = "analysis.json"
BLIND_KEY_FILE = "blind-key.json"
REVIEW_MANIFEST_FILE = "review-manifest.json"
REVIEW_FILE = "review.html"
RAW_RESPONSE_FILE = REVIEW_RESPONSE_FILENAME
MANUAL_RESULT_FILE = "manual-result.json"
ATTEMPT_DIR = "attempts"


def _load_json(path: Path) -> Any:
    if not path.is_file():
        raise RuntimeError(f"required artifact is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _write_once_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        existing = _load_json(path)
        if stable_json(existing) != stable_json(payload):
            raise RuntimeError(f"immutable artifact differs from recomputation: {path}")
        return
    atomic_write_json(path, payload)


def _write_once_bytes(path: Path, payload: bytes) -> None:
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError(f"immutable artifact differs byte-for-byte: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".partial", dir=path.parent
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _write_wav(path: Path, audio: np.ndarray) -> None:
    if path.exists():
        raise RuntimeError(f"one-attempt confirmation WAV already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    try:
        with wave.open(str(temporary), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(SAMPLE_RATE_HZ)
            handle.writeframes(pcm16_bytes(audio))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_pcm(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as handle:
        if (
            handle.getnchannels() != 1
            or handle.getsampwidth() != 2
            or handle.getframerate() != SAMPLE_RATE_HZ
        ):
            raise RuntimeError(f"confirmation WAV violates mono PCM16 contract: {path}")
        values = np.frombuffer(
            handle.readframes(handle.getnframes()), dtype="<i2"
        ).astype(np.float64)
    if not values.size or not np.isfinite(values).all():
        raise RuntimeError(f"confirmation WAV is empty or nonfinite: {path}")
    return values, SAMPLE_RATE_HZ


def _pcm_record(audio: np.ndarray, path: Path) -> dict[str, Any]:
    values = np.asarray(audio, dtype=np.float64).reshape(-1)
    finite = bool(values.size and np.isfinite(values).all())
    clipped_fraction = float(np.mean(np.abs(values) >= 1.0)) if finite else 1.0
    return {
        "relative_path": str(path.relative_to(run_dir())),
        "sample_count": int(values.size),
        "finite": finite,
        "clipped_fraction": clipped_fraction,
        "clipping_pass": bool(clipped_fraction < MAX_CLIPPED_FRACTION),
        "pcm_sha256": pcm_sha256(values) if finite else None,
        "wav_sha256": sha256_file(path),
    }


def _sample_interval(
    columns: Sequence[int], durations: Sequence[int], samples_per_frame: int
) -> dict[str, Any]:
    selected = tuple(int(value) for value in columns)
    if not selected or selected != tuple(range(selected[0], selected[-1] + 1)):
        raise RuntimeError("alignment interval must be nonempty and contiguous")
    start_sample = (
        sum(int(value) for value in durations[: selected[0]]) * samples_per_frame
    )
    end_sample = (
        sum(int(value) for value in durations[: selected[-1] + 1]) * samples_per_frame
    )
    if end_sample <= start_sample:
        raise RuntimeError("alignment interval has nonpositive duration")
    return {
        "columns": list(selected),
        "start_sample": start_sample,
        "end_sample_exclusive": end_sample,
        "start_s": start_sample / SAMPLE_RATE_HZ,
        "end_s": end_sample / SAMPLE_RATE_HZ,
    }


def alignment_record(
    *,
    model: Any,
    plan: TypedPlan,
    durations: Sequence[int],
    sample_count: int,
    anchor_occurrence_map: Sequence[int],
) -> dict[str, Any]:
    expected_count = len(_filtered_symbols(model, plan.source_phonemes)) + 2
    if len(durations) != expected_count:
        raise RuntimeError(
            "duration count differs from the fixture's source token plan"
        )
    total_frames = sum(int(value) for value in durations)
    if total_frames <= 0 or sample_count % total_frames:
        raise RuntimeError("decoded samples do not map to integral alignment frames")
    samples_per_frame = sample_count // total_frames
    word_spans = _word_column_spans(model, plan.neutral_phonemes)
    if len(word_spans) != len(plan.words):
        raise RuntimeError("word spans differ from the typed plan")
    expected_replaced = target_word_columns(
        model, plan.neutral_phonemes, plan.target_word_indexes
    )
    occurrences: list[dict[str, Any]] = []
    target_words: list[dict[str, Any]] = []
    for word_index in plan.target_word_indexes:
        word = plan.words[word_index]
        span = word_spans[word_index]
        if len(span) != len(word.neutral_phone):
            raise RuntimeError("word state columns drifted from its phone plan")
        word_interval = _sample_interval(span, durations, samples_per_frame)
        target_words.append({"word_index": word_index, "interval": word_interval})
        for within_word_index, offset in enumerate(word.target_offsets):
            if word.neutral_phone[offset] != "æ" or word.lens_phone[offset] != "ɛ":
                raise RuntimeError("target offset no longer maps /ae/ to /eh/")
            if offset < 1 or word.neutral_phone[offset - 1] not in {"ˈ", "ˌ"}:
                raise RuntimeError("confirmation target lacks its stress column")
            stress_column = span[offset - 1]
            target_column = span[offset]
            occurrence_index = len(occurrences)
            occurrences.append(
                {
                    "occurrence_index": occurrence_index,
                    "anchor_occurrence_index": int(
                        anchor_occurrence_map[occurrence_index]
                    ),
                    "position": (
                        "medial"
                        if int(anchor_occurrence_map[occurrence_index]) == 0
                        else "phrase-final"
                    ),
                    "word_index": word_index,
                    "within_word_index": within_word_index,
                    "stress_column": stress_column,
                    "target_column": target_column,
                    "measurement_interval": _sample_interval(
                        (stress_column, target_column), durations, samples_per_frame
                    ),
                    "target_word_interval": word_interval,
                }
            )
    if len(occurrences) != plan.target_occurrence_count:
        raise RuntimeError("alignment lost a precommitted target occurrence")
    if len(anchor_occurrence_map) != len(occurrences):
        raise RuntimeError("anchor mapping does not cover every occurrence")
    return {
        "duration_count": len(durations),
        "total_alignment_frames": total_frames,
        "samples_per_alignment_frame": samples_per_frame,
        "expected_replaced_columns": list(expected_replaced),
        "target_occurrences": occurrences,
        "target_words": target_words,
        "own_source_derived_durations": True,
        "own_source_derived_alignment": True,
        "own_fixture_neutral_f0_noise": True,
    }


def _checked_protocol() -> dict[str, Any]:
    path = run_dir() / "protocol.json"
    if not path.is_file():
        raise RuntimeError(
            "confirmation protocol is not frozen; prepare and commit it before rendering"
        )
    frozen = _load_json(path)
    current = protocol_record()
    if stable_json(frozen) != stable_json(current):
        raise RuntimeError(
            "confirmation protocol differs from its bound implementation"
        )
    return frozen


def _require_committed_inputs(protocol: dict[str, Any]) -> str:
    paths = protocol["implementation"]["committed_before_render"]["tracked_clean_paths"]
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
    if clean.returncode != 0:
        raise RuntimeError("confirmation or bound diagnostic inputs differ from HEAD")
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Paths().root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _verify_measurement_bindings(protocol: dict[str, Any]) -> None:
    measurement = protocol["implementation"]["measurement"]
    if sha256_file(PRAAT) != measurement["praat_sha256"]:
        raise RuntimeError("Praat changed after confirmation freeze")
    if sha256_file(MEASUREMENT_SCRIPT) != measurement["script_sha256"]:
        raise RuntimeError("measurement script changed after confirmation freeze")


def _initial_records(protocol: dict[str, Any], commit: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "run_id": RUN_ID,
        "protocol_sha256": protocol["protocol_sha256"],
        "status": "in_progress",
        "implementation_commit": commit,
        "api_calls_made": 0,
        "openai_calls_made": 0,
        "paid_calls_made": 0,
        "slots": [
            {**row, "status": "pending", "reason": "fixed_required_decode"}
            for row in protocol["render_manifest"]
        ],
        "fixtures": [],
    }


def _records_path() -> Path:
    return run_dir() / RECORDS_FILE


def _slot(records: dict[str, Any], slot_id: str) -> dict[str, Any]:
    return next(row for row in records["slots"] if row["slot_id"] == slot_id)


def _persist_records(records: dict[str, Any]) -> None:
    atomic_write_json(_records_path(), records)


def _attempt_marker(slot_id: str) -> Path:
    return run_dir() / ATTEMPT_DIR / f"{slot_id}.json"


def _begin_fixture_attempts(
    records: dict[str, Any], fixture_id: str
) -> list[dict[str, Any]]:
    rows = [row for row in records["slots"] if row["fixture_id"] == fixture_id]
    if any(_attempt_marker(row["slot_id"]).exists() for row in rows):
        for row in rows:
            if row["status"] != "complete":
                row["status"] = "interrupted_no_retry"
                row["reason"] = "attempt_marker_exists_without_complete_triplet"
        _persist_records(records)
        raise RuntimeError(f"confirmation fixture was already attempted: {fixture_id}")
    for row in rows:
        marker = _attempt_marker(row["slot_id"])
        marker.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(
            marker,
            {
                "schema_version": 1,
                "run_id": RUN_ID,
                "protocol_sha256": records["protocol_sha256"],
                "slot_id": row["slot_id"],
                "fixture_id": fixture_id,
                "role": row["role"],
                "one_attempt_no_retry": True,
            },
        )
        row["status"] = "attempt_started"
        row["reason"] = "decode_attempt_consumed"
    _persist_records(records)
    return rows


def _parent_wav_hashes(protocol: dict[str, Any]) -> set[str]:
    hashes: set[str] = set()
    for row in protocol["parents"]["diagnostic"]["bound_output_files"]:
        if row["relative_path"].endswith(".wav"):
            hashes.add(row["sha256"])
    hashes.update(
        row["wav_sha256"]
        for row in protocol["parents"]["frozen_failed_replication_v1"].get(
            "bound_wavs", []
        )
    )
    return hashes


def _render_confirmation(protocol: dict[str, Any], records: dict[str, Any]) -> None:
    from .kokoro_synthesis import KokoroSynthesisRuntime

    planner = KokoroTypedPlanner.load()
    runtime = KokoroSynthesisRuntime.load(download=False)
    fixture_protocol = {row["fixture_id"]: row for row in protocol["fixtures"]}
    parent_hashes = _parent_wav_hashes(protocol)
    for fixture in CONFIRMATION_FIXTURES:
        existing = next(
            (
                row
                for row in records["fixtures"]
                if row["fixture_id"] == fixture.fixture_id
            ),
            None,
        )
        if existing is not None and existing.get("runtime_pass"):
            continue
        slots = _begin_fixture_attempts(records, fixture.fixture_id)
        try:
            plan = planner.plan(fixture.text)
            frozen = fixture_protocol[fixture.fixture_id]
            if plan.plan_sha256 != frozen["expected_plan_sha256"]:
                raise RuntimeError("confirmation plan drifted after protocol commit")
            pair_plan = plan.pair_plan()
            if pair_plan is None:
                raise RuntimeError("confirmation fixture unexpectedly lacks a pair")
            rendered = runtime.render_parity_triplet(pair_plan)
            expected_columns = tuple(
                int(value) for value in frozen["target_word_columns"]
            )
            if rendered.replaced_columns != expected_columns:
                raise RuntimeError("target-word replacement columns drifted")
            paths: dict[str, Path] = {}
            arrays = {
                "neutral": rendered.neutral,
                "identity": rendered.identity,
                "lens": rendered.lens,
            }
            for row in slots:
                path = (
                    run_dir()
                    / "audio"
                    / f"{int(row['order']):02d}__{row['slot_id']}.wav"
                )
                _write_wav(path, arrays[row["role"]])
                paths[row["role"]] = path
            pcm = {role: _pcm_record(arrays[role], paths[role]) for role in arrays}
            alignment = alignment_record(
                model=runtime.model,
                plan=plan,
                durations=rendered.predicted_durations,
                sample_count=len(rendered.neutral),
                anchor_occurrence_map=frozen["anchor_occurrence_map"],
            )
            identity_equal = pcm16_bytes(rendered.neutral) == pcm16_bytes(
                rendered.identity
            )
            integrity = inspect_render(
                PairRender(
                    neutral=rendered.neutral,
                    lens=rendered.lens,
                    predicted_durations=rendered.predicted_durations,
                    replaced_columns=rendered.replaced_columns,
                )
            )
            disjoint = all(
                row["wav_sha256"] not in parent_hashes for row in pcm.values()
            )
            sample_counts = {row["sample_count"] for row in pcm.values()}
            runtime_pass = bool(
                len(sample_counts) == 1
                and next(iter(sample_counts), 0) > 0
                and identity_equal
                and integrity.pass_all
                and alignment["expected_replaced_columns"] == list(expected_columns)
                and all(row["finite"] and row["clipping_pass"] for row in pcm.values())
                and disjoint
            )
            record = {
                "fixture_id": fixture.fixture_id,
                "source_text_sha256": hashlib.sha256(
                    fixture.text.encode("utf-8")
                ).hexdigest(),
                "plan_sha256": plan.plan_sha256,
                "plan_safe_metadata": plan.safe_metadata(),
                "predicted_durations": list(rendered.predicted_durations),
                "replaced_columns": list(rendered.replaced_columns),
                "alignment": alignment,
                "audio": pcm,
                "neutral_identity_bit_identical": identity_equal,
                "pair_integrity": asdict(integrity),
                "new_wavs_disjoint_from_bound_parents": disjoint,
                "runtime_pass": runtime_pass,
            }
            if not runtime_pass:
                raise RuntimeError("confirmation fixture failed runtime integrity")
            records["fixtures"].append(record)
            for row in slots:
                row["status"] = "complete"
                row["reason"] = "one_attempt_complete"
                row["audio"] = pcm[row["role"]]
            _persist_records(records)
        except Exception as exc:
            for row in slots:
                if row["status"] != "complete":
                    row["status"] = "failed_no_retry"
                    row["reason"] = f"{type(exc).__name__}: {str(exc)[:500]}"
            for row in records["slots"]:
                if row["status"] == "pending":
                    row["status"] = "not_reached"
                    row["reason"] = "earlier_fixed_decode_failed"
            records["status"] = "runtime_failure_no_retry"
            _persist_records(records)
            raise
    records["status"] = "render_complete"
    records["decoder_attempt_count"] = sum(
        _attempt_marker(row["slot_id"]).exists() for row in records["slots"]
    )
    records["all_runtime_gates_pass"] = bool(
        len(records["fixtures"]) == len(CONFIRMATION_FIXTURES)
        and all(row["runtime_pass"] for row in records["fixtures"])
    )
    records["one_attempt_slots_respected"] = all(
        row["status"] in {"complete", "failed_no_retry", "not_reached"}
        for row in records["slots"]
    )
    _persist_records(records)


def _measure_occurrences(
    path: Path, occurrences: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    return [
        {
            "occurrence_index": occurrence["occurrence_index"],
            "anchor_occurrence_index": occurrence["anchor_occurrence_index"],
            "position": occurrence["position"],
            "measurement_interval": occurrence["measurement_interval"],
            "families": {
                str(ceiling): measure_interval_windows(
                    path, occurrence["measurement_interval"], ceiling
                )
                for ceiling in CEILINGS_HZ
            },
        }
        for occurrence in occurrences
    ]


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    a = np.asarray(left, dtype=float)
    b = np.asarray(right, dtype=float)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denominator) if denominator else -1.0


def _family_gate(
    neutral: dict[str, Any], lens: dict[str, Any], anchor: dict[str, Any]
) -> dict[str, Any]:
    measurement_valid = bool(
        neutral.get("measurement_valid")
        and lens.get("measurement_valid")
        and "ae_bark" in anchor
        and "eh_bark" in anchor
    )
    if not measurement_valid:
        checks = {
            "measurement_valid": False,
            "neutral_plausible": bool(neutral.get("plausibility_pass")),
            "lens_plausible": bool(lens.get("plausibility_pass")),
            "neutral_nearer_local_ae": False,
            "lens_nearer_local_eh": False,
            "direction_cosine_at_least_0_50": False,
            "magnitude_at_least_local_threshold": False,
        }
        return {"checks": checks, "pass": False}
    ae = np.asarray(anchor["ae_bark"], dtype=float)
    eh = np.asarray(anchor["eh_bark"], dtype=float)
    neutral_point = np.asarray([neutral["f1_bark"], neutral["f2_bark"]], dtype=float)
    lens_point = np.asarray([lens["f1_bark"], lens["f2_bark"]], dtype=float)
    expected = eh - ae
    vector = lens_point - neutral_point
    threshold = max(0.25, 0.5 * float(np.linalg.norm(expected)))
    magnitude = float(np.linalg.norm(vector))
    direction = _cosine(vector, expected)
    checks = {
        "measurement_valid": True,
        "neutral_plausible": bool(neutral["plausibility_pass"]),
        "lens_plausible": bool(lens["plausibility_pass"]),
        "neutral_nearer_local_ae": (
            float(np.linalg.norm(neutral_point - ae))
            < float(np.linalg.norm(neutral_point - eh))
        ),
        "lens_nearer_local_eh": (
            float(np.linalg.norm(lens_point - eh))
            < float(np.linalg.norm(lens_point - ae))
        ),
        "direction_cosine_at_least_0_50": direction >= MINIMUM_DIRECTION_COSINE,
        "magnitude_at_least_local_threshold": magnitude >= threshold,
    }
    return {
        "neutral_bark": neutral_point.tolist(),
        "lens_bark": lens_point.tolist(),
        "anchor_ae_bark": ae.tolist(),
        "anchor_eh_bark": eh.tolist(),
        "vector_bark": vector.tolist(),
        "direction_cosine": direction,
        "magnitude_bark": magnitude,
        "local_threshold_bark": threshold,
        "checks": checks,
        "pass": all(checks.values()),
    }


def _analyze_fixture(
    record: dict[str, Any], protocol: dict[str, Any]
) -> dict[str, Any]:
    neutral_path = run_dir() / record["audio"]["neutral"]["relative_path"]
    lens_path = run_dir() / record["audio"]["lens"]["relative_path"]
    for role, path in (("neutral", neutral_path), ("lens", lens_path)):
        if sha256_file(path) != record["audio"][role]["wav_sha256"]:
            raise RuntimeError(f"confirmation WAV hash drifted: {path}")
    neutral_measurements = _measure_occurrences(
        neutral_path, record["alignment"]["target_occurrences"]
    )
    lens_measurements = _measure_occurrences(
        lens_path, record["alignment"]["target_occurrences"]
    )
    anchors = protocol["parents"]["diagnostic"]["local_anchor_geometry"]
    windows: dict[str, Any] = {}
    for percent in WINDOW_PERCENTS:
        window_key = str(percent)
        occurrences: list[dict[str, Any]] = []
        for occurrence_index, occurrence in enumerate(
            record["alignment"]["target_occurrences"]
        ):
            anchor_index = int(occurrence["anchor_occurrence_index"])
            families: dict[str, Any] = {}
            for ceiling in CEILINGS_HZ:
                key = str(ceiling)
                families[key] = _family_gate(
                    neutral_measurements[occurrence_index]["families"][key][window_key],
                    lens_measurements[occurrence_index]["families"][key][window_key],
                    anchors[window_key]["occurrences"][anchor_index]["families"][key],
                )
            occurrences.append(
                {
                    "occurrence_index": occurrence_index,
                    "anchor_occurrence_index": anchor_index,
                    "position": occurrence["position"],
                    "families": families,
                    "pass": all(row["pass"] for row in families.values()),
                }
            )
        windows[window_key] = {
            "occurrences": occurrences,
            "pass": all(row["pass"] for row in occurrences),
        }

    def signature(percent: int) -> list[dict[str, bool]]:
        return [
            family["checks"]
            for occurrence in windows[str(percent)]["occurrences"]
            for family in occurrence["families"].values()
        ] + [{"overall_pass": windows[str(percent)]["pass"]}]

    sensitivity = {
        str(percent): signature(percent) != signature(PRIMARY_WINDOW_PERCENT)
        for percent in DESCRIPTIVE_WINDOW_PERCENTS
    }
    neutral_pcm, rate = _read_pcm(neutral_path)
    lens_pcm, lens_rate = _read_pcm(lens_path)
    if rate != lens_rate:
        raise RuntimeError("confirmation pair sample rates differ")
    localization = localization_report(
        neutral_pcm,
        lens_pcm,
        [row["interval"] for row in record["alignment"]["target_words"]],
        sample_rate_hz=rate,
    )
    runtime_checks = {
        "render_runtime_pass": bool(record["runtime_pass"]),
        "neutral_identity_bit_identical": bool(
            record["neutral_identity_bit_identical"]
        ),
        "new_wavs_disjoint_from_bound_parents": bool(
            record["new_wavs_disjoint_from_bound_parents"]
        ),
        "localization_at_least_0_80": bool(localization["pass"]),
    }
    primary_pass = windows[str(PRIMARY_WINDOW_PERCENT)]["pass"]
    automatic_pass = bool(primary_pass and all(runtime_checks.values()))
    return {
        "fixture_id": record["fixture_id"],
        "status": "measurable",
        "neutral_measurements": neutral_measurements,
        "lens_measurements": lens_measurements,
        "windows": windows,
        "primary_window_percent": PRIMARY_WINDOW_PERCENT,
        "descriptive_window_sensitivity": sensitivity,
        "window_sensitive": any(sensitivity.values()),
        "localization": localization,
        "runtime_checks": runtime_checks,
        "automatic_pass": automatic_pass,
    }


def _analysis_payload(
    protocol: dict[str, Any], records: dict[str, Any]
) -> dict[str, Any]:
    fixtures: list[dict[str, Any]] = []
    measurement_failures: list[dict[str, str]] = []
    for record in records["fixtures"]:
        try:
            fixtures.append(_analyze_fixture(record, protocol))
        except Exception as exc:
            measurement_failures.append(
                {
                    "fixture_id": str(record.get("fixture_id", "unknown")),
                    "failure": f"{type(exc).__name__}: {str(exc)[:1000]}",
                }
            )
    if measurement_failures:
        classification = (
            "fresh_unseen_fixture_confirmation_inconclusive_measurement_failure"
        )
        automatic_pass = False
    else:
        automatic_pass = bool(
            len(fixtures) == len(CONFIRMATION_FIXTURES)
            and all(row["automatic_pass"] for row in fixtures)
        )
        classification = (
            "fresh_unseen_fixture_confirmation_automatic_pass_pending_human_review"
            if automatic_pass
            else "fresh_unseen_fixture_confirmation_automatic_failed_no_review"
        )
    return {
        "schema_version": 1,
        "run_id": RUN_ID,
        "protocol_sha256": protocol["protocol_sha256"],
        "status": "analysis_complete",
        "classification": classification,
        "claim": (
            "bounded_controlled_target_word_generalization_only"
            if automatic_pass
            else "no_positive_generalization_claim"
        ),
        "selected_span": "target-word",
        "automatic_confirmation_pass": automatic_pass,
        "pending_human_review": automatic_pass,
        "measurement_failures": measurement_failures,
        "fixtures": fixtures,
        "render_records_sha256": sha256_file(_records_path()),
        "api_calls_made": 0,
        "openai_calls_made": 0,
        "paid_calls_made": 0,
        "frozen_replication_v1_preserved_failed": True,
        "diagnostic_parent_classification": protocol["parents"]["diagnostic"][
            "classification"
        ],
        "causal_claims_not_supported": [
            "root cause",
            "position",
            "duration",
            "state coupling",
            "population perception",
        ],
    }


def _runtime_failure_analysis(
    protocol: dict[str, Any], records: dict[str, Any], exc: Exception
) -> dict[str, Any]:
    payload = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "protocol_sha256": protocol["protocol_sha256"],
        "status": "analysis_complete",
        "classification": "fresh_unseen_fixture_confirmation_inconclusive_runtime_failure",
        "claim": "no_positive_generalization_claim",
        "automatic_confirmation_pass": False,
        "pending_human_review": False,
        "failure": f"{type(exc).__name__}: {str(exc)[:1000]}",
        "completed_fixture_ids": [
            str(row.get("fixture_id", "unknown")) for row in records["fixtures"]
        ],
        "render_records_sha256": sha256_file(_records_path()),
        "api_calls_made": 0,
        "openai_calls_made": 0,
        "paid_calls_made": 0,
        "frozen_replication_v1_preserved_failed": True,
    }
    _write_once_json(run_dir() / ANALYSIS_FILE, payload)
    return payload


def _records_by_fixture() -> dict[str, dict[str, Any]]:
    records = _load_json(_records_path())
    return {row["fixture_id"]: row for row in records["fixtures"]}


def _ensure_review_audio(
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
            target = (
                destination
                / f"{trial['trial_id'].replace('comparison-', '')}-{side.lower()}.wav"
            )
            if target.exists():
                if sha256_file(target) != sha256_file(source):
                    raise RuntimeError(
                        "existing blind review copy differs from its source"
                    )
            else:
                shutil.copyfile(source, target)
            sides.append(
                {
                    "side": side,
                    "audio": f"review-audio/{target.name}",
                    "sha256": sha256_file(target),
                }
            )
        intervals = [
            {
                "start_s": row["interval"]["start_s"],
                "end_s": row["interval"]["end_s"],
            }
            for row in record["alignment"]["target_words"]
        ]
        public.append(
            {
                "trial_id": trial["trial_id"],
                "duration_s": record["audio"]["neutral"]["sample_count"]
                / SAMPLE_RATE_HZ,
                "target_intervals": intervals,
                "sides": sides,
            }
        )
    return public


def _review_html(public: list[dict[str, Any]], protocol_sha256: str) -> str:
    template = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Blind English controlled-engine confirmation</title><style>:root{color-scheme:light}body{font:17px/1.5 system-ui;max-width:920px;margin:auto;padding:24px;background:#f4f1e8;color:#17221c}.intro,.trial,.side{background:white;border:1px solid #d3d0c7;border-radius:16px;padding:20px;margin:16px 0}.pair{display:grid;grid-template-columns:1fr 1fr;gap:14px}.side{margin:0}.timeline{height:16px;background:#d8ddd8;border-radius:99px;position:relative;margin:10px 0 16px;overflow:hidden}.cue{position:absolute;height:100%;background:#d87b35}.playhead{position:absolute;top:0;bottom:0;width:2px;background:#154f3e}.target-now .timeline{outline:3px solid #d87b35}.target-status{font-weight:700;color:#805025;min-height:1.5em}audio,select,textarea{width:100%;box-sizing:border-box}label{display:block;margin:10px 0}textarea{min-height:78px}button{padding:12px 20px;border:0;border-radius:999px;background:#154f3e;color:white;font-weight:700}button:disabled{opacity:.45}.muted{color:#57645e}@media(max-width:680px){.pair{grid-template-columns:1fr}}</style></head><body><section class="intro"><h1>Blind English controlled-engine confirmation</h1><p>Four randomized comparisons cover two previously unheard carrier structures. Two are exact identity checks and two contain the fixed vowel candidate. Their order and sides are hidden.</p><p>The orange bands mark target positions. Complete every field before downloading the exact response file.</p></section><div id="trials"></div><button id="download" disabled>Download response</button><script>const R=__PUBLIC__;const PROTOCOL='__PROTOCOL__';const RUN='__RUN__';const F='__FILENAME__';const K='kokoro-en-typed-confirmation-v1-review';const S=JSON.parse(localStorage.getItem(K)||'{}');S.session_id??=crypto.randomUUID();S.trials??={};const save=()=>localStorage.setItem(K,JSON.stringify(S));const trialState=id=>(S.trials[id]??={sides:{A:{},B:{}},pair:{},play_starts:{A:0,B:0}});const option=(value,label)=>`<option value="${value}">${label}</option>`;const selector=(id,scope,side,field,options,label)=>`<label>${label}<select data-id="${id}" data-scope="${scope}" data-side="${side||''}" data-field="${field}"><option value="">—</option>${options}</select></label>`;const sideCard=(trial,side)=>`<section class="side"><h3>Clip ${side.side}</h3><div class="player"><audio controls preload="metadata" src="${side.audio}" data-id="${trial.trial_id}" data-side="${side.side}"></audio><div class="timeline">${trial.target_intervals.map(x=>`<i class="cue" style="left:${100*x.start_s/trial.duration_s}%;width:${100*(x.end_s-x.start_s)/trial.duration_s}%"></i>`).join('')}<i class="playhead"></i></div><div class="target-status">Target position</div></div>${selector(trial.trial_id,'side',side.side,'naturalness',[1,2,3,4,5].map(n=>option(n,n)).join(''),'Naturalness (1 unusable · 5 fully natural)')}${selector(trial.trial_id,'side',side.side,'delivery',option('sentence-like','Sentence-like')+option('slightly-list-like','Slightly list-like')+option('dominantly-list-like','Dominantly list-like')+option('other','Other'),'Delivery')}${selector(trial.trial_id,'side',side.side,'meaning',option('none','None')+option('isolated-possible-word','Isolated possible word')+option('coherent-phrase','Coherent phrase')+option('clear-source-sentence','Clear source sentence'),'Stable English meaning in the gibberish')}${selector(trial.trial_id,'side',side.side,'artifact',option('none','None')+option('minor','Minor')+option('major','Major')+option('uncertain','Uncertain'),'Artifact or defect')}</section>`;document.getElementById('trials').innerHTML=R.map((trial,index)=>`<section class="trial"><h2>Comparison ${index+1} of ${R.length}</h2><div class="pair">${trial.sides.map(side=>sideCard(trial,side)).join('')}</div><p class="muted">Recorded play starts: <strong data-replays="${trial.trial_id}">0</strong></p>${selector(trial.trial_id,'pair','','difference_strength',[1,2,3,4,5,6,7].map(n=>option(n,n)).join(''),'Difference strength (1 none · 7 very strong)')}${selector(trial.trial_id,'pair','','category_judgment',option('A','A')+option('B','B')+option('same','Same')+option('uncertain','Uncertain')+option('neither','Neither'),'Which side, if either, is closer to the vowel in “bet”?')}${selector(trial.trial_id,'pair','','confidence',[1,2,3,4,5].map(n=>option(n,n)).join(''),'Confidence (1 guessing · 5 highly confident)')}${selector(trial.trial_id,'pair','','interference',option('none','None')+option('manageable','Manageable')+option('dominant','Dominant')+option('uncertain','Uncertain'),'Unrelated delivery interference')}<label>Notes (optional)<textarea data-id="${trial.trial_id}" data-scope="pair" data-side="" data-field="notes"></textarea></label></section>`).join('');const requiredSide=['naturalness','delivery','meaning','artifact'],requiredPair=['difference_strength','category_judgment','confidence','interference'];const complete=()=>R.every(t=>{const x=trialState(t.trial_id);return ['A','B'].every(s=>requiredSide.every(f=>String(x.sides[s][f]??'')!==''))&&requiredPair.every(f=>String(x.pair[f]??'')!=='')});const update=()=>{for(const t of R){const x=trialState(t.trial_id);document.querySelector(`[data-replays="${t.trial_id}"]`).textContent=String((x.play_starts.A||0)+(x.play_starts.B||0))}document.getElementById('download').disabled=!complete();save()};document.querySelectorAll('[data-field]').forEach(el=>{const x=trialState(el.dataset.id),target=el.dataset.scope==='side'?x.sides[el.dataset.side]:x.pair;el.value=target[el.dataset.field]??'';el.addEventListener('input',()=>{target[el.dataset.field]=el.value;update()})});document.querySelectorAll('audio').forEach(audio=>{const box=audio.closest('.player'),head=box.querySelector('.playhead'),status=box.querySelector('.target-status'),trial=R.find(x=>x.trial_id===audio.dataset.id);const draw=()=>{if(!Number.isFinite(audio.duration)||audio.duration<=0)return;head.style.left=`${Math.min(100,100*audio.currentTime/audio.duration)}%`;const active=trial.target_intervals.some(x=>audio.currentTime>=x.start_s&&audio.currentTime<=x.end_s)&&!audio.paused;box.classList.toggle('target-now',active);status.textContent=active?'TARGET NOW':'Target position'};for(const event of ['loadedmetadata','timeupdate','pause','ended'])audio.addEventListener(event,draw);audio.addEventListener('play',()=>{document.querySelectorAll('audio').forEach(other=>{if(other!==audio&&!other.paused)other.pause()});trialState(audio.dataset.id).play_starts[audio.dataset.side]++;draw();update()})});update();document.getElementById('download').addEventListener('click',()=>{if(!complete())return;const responses=R.map(t=>{const x=trialState(t.trial_id);return{trial_id:t.trial_id,sides:x.sides,difference_strength:Number(x.pair.difference_strength),category_judgment:x.pair.category_judgment,confidence:Number(x.pair.confidence),interference:x.pair.interference,notes:x.pair.notes??'',play_starts:x.play_starts,replay_count:(x.play_starts.A||0)+(x.play_starts.B||0)}});const payload={schema_version:1,run_id:RUN,protocol_sha256:PROTOCOL,session_id:S.session_id,saved_at:new Date().toISOString(),responses};const blob=new Blob([JSON.stringify(payload,null,2),String.fromCharCode(10)],{type:'application/json'}),link=document.createElement('a');link.href=URL.createObjectURL(blob);link.download=F;link.click()});</script></body></html>"""
    return (
        template.replace(
            "__PUBLIC__", json.dumps(public, ensure_ascii=False).replace("</", "<\\/")
        )
        .replace("__PROTOCOL__", protocol_sha256)
        .replace("__RUN__", RUN_ID)
        .replace("__FILENAME__", REVIEW_RESPONSE_FILENAME)
    )


def build_review(protocol: dict[str, Any], analysis: dict[str, Any]) -> dict[str, Any]:
    if analysis.get("automatic_confirmation_pass") is not True:
        raise RuntimeError("automatic confirmation did not authorize blind review")
    layout = blinded_trial_plan()
    records = _records_by_fixture()
    public = _ensure_review_audio(layout, records)
    key = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "protocol_sha256": protocol["protocol_sha256"],
        "trials": [
            {
                "trial_id": trial["trial_id"],
                "fixture_id": trial["fixture_id"],
                "condition": trial["condition"],
                "side_roles": trial["side_roles"],
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
        "status": "pending-human-review",
        "purpose": protocol["blind_review"]["purpose"],
        "estimated_minutes": protocol["blind_review"]["estimated_minutes"],
        "response_filename": REVIEW_RESPONSE_FILENAME,
        "trial_count": len(public),
        "public_trials": public,
        "hidden_fields_absent": True,
    }
    _write_once_json(run_dir() / BLIND_KEY_FILE, key)
    _write_once_json(run_dir() / REVIEW_MANIFEST_FILE, manifest)
    html_path = run_dir() / REVIEW_FILE
    html = _review_html(public, protocol["protocol_sha256"])
    if html_path.exists():
        if html_path.read_text(encoding="utf-8") != html:
            raise RuntimeError("existing confirmation review page differs")
    else:
        atomic_write_text(html_path, html)
    return manifest


def run() -> dict[str, Any]:
    analysis_path = run_dir() / ANALYSIS_FILE
    if analysis_path.is_file():
        analysis = _load_json(analysis_path)
        if (
            analysis.get("automatic_confirmation_pass")
            and not (run_dir() / REVIEW_MANIFEST_FILE).is_file()
        ):
            build_review(_checked_protocol(), analysis)
        return analysis
    protocol = _checked_protocol()
    _verify_measurement_bindings(protocol)
    commit = _require_committed_inputs(protocol)
    if _records_path().is_file():
        records = _load_json(_records_path())
        if records.get("protocol_sha256") != protocol["protocol_sha256"]:
            raise RuntimeError("confirmation render ledger belongs to another protocol")
    else:
        audio_dir = run_dir() / "audio"
        attempts = run_dir() / ATTEMPT_DIR
        if (audio_dir.exists() and any(audio_dir.iterdir())) or (
            attempts.exists() and any(attempts.iterdir())
        ):
            raise RuntimeError("orphan confirmation output exists without a ledger")
        records = _initial_records(protocol, commit)
        _persist_records(records)
    try:
        _render_confirmation(protocol, records)
    except Exception as exc:
        return _runtime_failure_analysis(protocol, records, exc)
    analysis = _analysis_payload(protocol, records)
    _write_once_json(analysis_path, analysis)
    if analysis["automatic_confirmation_pass"]:
        build_review(protocol, analysis)
    return analysis


def _side_gate(side: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "naturalness": int(side["naturalness"]) in {4, 5},
        "delivery": side["delivery"] == "sentence-like",
        "stable_recoverable_meaning": side["meaning"] == "none",
        "artifact": side["artifact"] in {"none", "minor"},
    }
    return {"checks": checks, "pass": all(checks.values())}


def decode_response(path: Path) -> dict[str, Any]:
    protocol = _checked_protocol()
    analysis = _load_json(run_dir() / ANALYSIS_FILE)
    if analysis.get("automatic_confirmation_pass") is not True:
        raise RuntimeError("confirmation review is not eligible")
    if path.name != REVIEW_RESPONSE_FILENAME:
        raise RuntimeError(
            f"review response must use the frozen filename {REVIEW_RESPONSE_FILENAME}"
        )
    raw = path.read_bytes()
    response = json.loads(raw)
    if response.get("run_id") != RUN_ID:
        raise RuntimeError("review response belongs to a different run")
    if response.get("protocol_sha256") != protocol["protocol_sha256"]:
        raise RuntimeError("review response belongs to a different protocol")
    key = _load_json(run_dir() / BLIND_KEY_FILE)
    key_by_trial = {row["trial_id"]: row for row in key["trials"]}
    rows = response.get("responses")
    if not isinstance(rows, list) or len(rows) != len(key_by_trial):
        raise RuntimeError("review response is incomplete")
    if {row.get("trial_id") for row in rows} != set(key_by_trial):
        raise RuntimeError("review trial set differs from the blind key")
    decoded: list[dict[str, Any]] = []
    fixture_results = {fixture.fixture_id: {} for fixture in CONFIRMATION_FIXTURES}
    for row in rows:
        key_row = key_by_trial[row["trial_id"]]
        side_results = {side: _side_gate(row["sides"][side]) for side in ("A", "B")}
        side_pass = all(result["pass"] for result in side_results.values())
        interference_pass = row["interference"] in {"none", "manageable"}
        if key_row["condition"] == "lens-candidate":
            pair_checks = {
                "difference_strength": int(row["difference_strength"]) >= 5,
                "category_direction": row["category_judgment"]
                == key_row["expected_lens_side"],
                "confidence": int(row["confidence"]) >= 3,
                "interference": interference_pass,
            }
        else:
            pair_checks = {
                "difference_strength": int(row["difference_strength"]) == 1,
                "category_direction": row["category_judgment"] in {"same", "neither"},
                "confidence": True,
                "interference": interference_pass,
            }
        passed = bool(side_pass and all(pair_checks.values()))
        fixture_results[key_row["fixture_id"]][key_row["condition"]] = passed
        decoded.append(
            {
                "trial_id": row["trial_id"],
                "fixture_id": key_row["fixture_id"],
                "condition": key_row["condition"],
                "side_results": side_results,
                "pair_checks": pair_checks,
                "pass": passed,
                "replay_count": row.get("replay_count"),
                "notes": row.get("notes", ""),
            }
        )
    fixture_pass = {
        fixture_id: bool(
            branches.get("identity-catch") and branches.get("lens-candidate")
        )
        for fixture_id, branches in fixture_results.items()
    }
    run_pass = all(fixture_pass.values())
    result = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "protocol_sha256": protocol["protocol_sha256"],
        "status": "human_review_complete",
        "classification": (
            "bounded_creator_review_pass"
            if run_pass
            else "bounded_creator_review_failed_no_promotion"
        ),
        "run_pass": run_pass,
        "fixture_pass": fixture_pass,
        "decoded_trials": decoded,
        "raw_response_filename": REVIEW_RESPONSE_FILENAME,
        "raw_response_sha256": hashlib.sha256(raw).hexdigest(),
        "frozen_replication_v1_preserved_failed": True,
    }
    _write_once_bytes(run_dir() / RAW_RESPONSE_FILE, raw)
    _write_once_json(run_dir() / MANUAL_RESULT_FILE, result)
    return result
