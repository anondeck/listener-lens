from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .config import Paths, sha256_json, stable_json
from .kokoro_cluster_confirmation_v2 import (
    ANALYSIS_FILE as V2_ANALYSIS_FILE,
    PROTOCOL_FILE as V2_PROTOCOL_FILE,
    RECORDS_FILE as V2_RECORDS_FILE,
    RUN_ID as V2_RUN_ID,
    _analyze_fixture,
    _render_candidates,
    run_dir as v2_run_dir,
)
from .kokoro_typed_confirmation_protocol import CEILINGS_HZ
from .util import atomic_write_json, sha256_file


CONFIRMATION_VERSION = "kokoro-cluster-confirmation-v3"
RUN_ID = "20260718-kokoro-cluster-confirmation-v3"
PROTOCOL_FILE = "protocol.json"
RECORDS_FILE = "render-records.json"
ANALYSIS_FILE = "analysis.json"
MINIMUM_STABLE_FAMILIES = 2
MINIMUM_CROSS_FAMILY_VECTOR_COSINE = 0.75

TRACKED_CLEAN_PATHS = (
    "src/earshift_bakeoff/kokoro_cluster_confirmation_v2.py",
    "src/earshift_bakeoff/kokoro_cluster_confirmation_v3.py",
    "src/earshift_bakeoff/kokoro_cluster_shell.py",
    "src/earshift_bakeoff/kokoro_output_domain_splice.py",
    "src/earshift_bakeoff/kokoro_output_splice_unseen.py",
    "src/earshift_bakeoff/kokoro_synthesis.py",
    "src/earshift_bakeoff/kokoro_typed_confirmation.py",
    "src/earshift_bakeoff/kokoro_typed_engine.py",
    "scripts/run_kokoro_cluster_confirmation_v3.py",
    f"artifacts/typed-engine/{RUN_ID}/{PROTOCOL_FILE}",
    f"artifacts/typed-engine/{V2_RUN_ID}/{V2_PROTOCOL_FILE}",
    f"artifacts/typed-engine/{V2_RUN_ID}/{V2_RECORDS_FILE}",
    f"artifacts/typed-engine/{V2_RUN_ID}/{V2_ANALYSIS_FILE}",
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


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    a = np.asarray(left, dtype=float)
    b = np.asarray(right, dtype=float)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denominator) if denominator else 0.0


def _verified_v2_parent() -> dict[str, Any]:
    protocol_path = v2_run_dir() / V2_PROTOCOL_FILE
    records_path = v2_run_dir() / V2_RECORDS_FILE
    analysis_path = v2_run_dir() / V2_ANALYSIS_FILE
    protocol = _load_json(protocol_path)
    records = _load_json(records_path)
    analysis = _load_json(analysis_path)
    _verify_internal_hash(protocol, "protocol_sha256", label="v2 protocol")
    _verify_internal_hash(analysis, "analysis_sha256", label="v2 analysis")
    if (
        analysis.get("classification") != "cluster_shell_v2_anchor_calibration_failed"
        or analysis.get("candidate_decodes_attempted") is not False
        or records.get("status") != "anchor_calibration_failed"
        or records.get("decoder_attempt_count") != 0
    ):
        raise RuntimeError("v2 parent is not the frozen anchor-only failure")
    if sha256_file(records_path) != analysis.get("render_records_sha256"):
        raise RuntimeError("v2 render records differ from the frozen analysis")
    return {
        "protocol": protocol,
        "records": records,
        "binding": {
            "run_id": V2_RUN_ID,
            "protocol_sha256": protocol["protocol_sha256"],
            "protocol_file_sha256": sha256_file(protocol_path),
            "records_file_sha256": sha256_file(records_path),
            "analysis_sha256": analysis["analysis_sha256"],
            "analysis_file_sha256": sha256_file(analysis_path),
            "classification": analysis["classification"],
        },
    }


def _family_selection(parent: dict[str, Any]) -> dict[str, Any]:
    gates = parent["records"]["same_context_anchors"]["gates"]
    selected: dict[str, Any] = {}
    for fixture in parent["protocol"]["fixtures"]:
        fixture_id = fixture["fixture_id"]
        selected[fixture_id] = {}
        for position in gates[fixture_id]:
            passing = [
                ceiling
                for ceiling in CEILINGS_HZ
                if position[str(ceiling)]["pass"] is True
            ]
            if len(passing) < MINIMUM_STABLE_FAMILIES:
                raise RuntimeError(
                    f"{fixture_id} position {position['position']} lacks two "
                    "stable analysis families"
                )
            vectors: dict[str, list[float]] = {}
            for ceiling in passing:
                cell = position[str(ceiling)]
                ae = np.asarray(cell["ae"]["endpoint_bark"], dtype=float)
                eh = np.asarray(cell["eh"]["endpoint_bark"], dtype=float)
                vectors[str(ceiling)] = [float(value) for value in eh - ae]

            # Prefer the largest contiguous family, then the lower-ceiling
            # family under a tie.  This ordering is deterministic and uses
            # anchor data only; candidate audio does not yet exist.
            candidates = [
                passing[start : start + width]
                for width in range(len(passing), MINIMUM_STABLE_FAMILIES - 1, -1)
                for start in range(len(passing) - width + 1)
                if all(
                    right - left == 1
                    for left, right in zip(
                        (
                            CEILINGS_HZ.index(value)
                            for value in passing[start : start + width]
                        ),
                        (
                            CEILINGS_HZ.index(value)
                            for value in passing[start + 1 : start + width]
                        ),
                    )
                )
            ]
            chosen: list[int] | None = None
            chosen_cosines: dict[str, float] = {}
            for candidate in candidates:
                cosines = {
                    f"{left}:{right}": _cosine(vectors[str(left)], vectors[str(right)])
                    for index, left in enumerate(candidate)
                    for right in candidate[index + 1 :]
                }
                if cosines and min(cosines.values()) >= (
                    MINIMUM_CROSS_FAMILY_VECTOR_COSINE
                ):
                    chosen = candidate
                    chosen_cosines = cosines
                    break
            if chosen is None:
                raise RuntimeError(
                    f"{fixture_id} position {position['position']} has incoherent "
                    "stable-family vectors"
                )
            anchors = {
                str(ceiling): {
                    "ae_bark": position[str(ceiling)]["ae"]["endpoint_bark"],
                    "eh_bark": position[str(ceiling)]["eh"]["endpoint_bark"],
                }
                for ceiling in chosen
            }
            selected[fixture_id][str(position["position"])] = {
                "individually_passing_ceilings_hz": passing,
                "selected_ceilings_hz": chosen,
                "excluded_ceilings_hz": [
                    ceiling for ceiling in CEILINGS_HZ if ceiling not in chosen
                ],
                "cross_family_vector_cosines": chosen_cosines,
                "anchors": anchors,
            }
    return selected


def protocol_record() -> dict[str, Any]:
    parent = _verified_v2_parent()
    selection = _family_selection(parent)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "version": CONFIRMATION_VERSION,
        "status": "frozen_before_candidate_render",
        "purpose": (
            "Evaluate the still-unheard v2 cluster candidates using only "
            "same-context Praat settings whose anchor endpoints independently "
            "passed stability, separation, and direction gates."
        ),
        "parent": parent["binding"],
        "fixtures": parent["protocol"]["fixtures"],
        "analysis_family_selection": {
            "ordering_hz": list(CEILINGS_HZ),
            "minimum_adjacent_stable_families": MINIMUM_STABLE_FAMILIES,
            "minimum_cross_family_vector_cosine": (MINIMUM_CROSS_FAMILY_VECTOR_COSINE),
            "selection_uses_anchor_data_only": True,
            "candidate_audio_unavailable_at_selection": True,
            "selected_by_fixture_position": selection,
            "candidate_gate": (
                "Every occurrence must pass every selected stable family at "
                "the unchanged primary 50% acoustic gate."
            ),
        },
        "unchanged": {
            "fixtures": True,
            "cluster_shell": True,
            "renderer_and_seed": True,
            "splice_and_boundary_rules": True,
            "per_family_acoustic_formulas": True,
            "integrity_and_localization_gates": True,
            "one_attempt_per_candidate_slot": True,
        },
        "predetermined_outcomes": {
            "automatic_pass": "eligible for one blinded creator QC package",
            "any_failure": (
                "freeze the exact failure; do not replace fixtures, change "
                "thresholds, or selectively rerender"
            ),
        },
        "scope": {
            "api_calls": 0,
            "reused_anchor_decodes": 42,
            "new_anchor_decodes": 0,
            "candidate_decoder_slots": len(parent["protocol"]["fixtures"]) * 3,
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
            raise RuntimeError("existing v3 cluster protocol differs")
        return existing
    if any(
        (run_dir() / name).exists() for name in (RECORDS_FILE, ANALYSIS_FILE, "audio")
    ):
        raise RuntimeError("v3 output exists before its protocol")
    destination.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(destination, protocol)
    return protocol


def _checked_protocol() -> dict[str, Any]:
    frozen = _load_json(run_dir() / PROTOCOL_FILE)
    if stable_json(frozen) != stable_json(protocol_record()):
        raise RuntimeError("v3 protocol or a bound parent artifact drifted")
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
        raise RuntimeError("v3 inputs differ from committed HEAD")
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Paths().root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _anchors(protocol: dict[str, Any]) -> dict[str, Any]:
    return {
        fixture_id: {position: row["anchors"] for position, row in positions.items()}
        for fixture_id, positions in protocol["analysis_family_selection"][
            "selected_by_fixture_position"
        ].items()
    }


def run() -> dict[str, Any]:
    analysis_path = run_dir() / ANALYSIS_FILE
    if analysis_path.is_file():
        return _load_json(analysis_path)
    protocol = _checked_protocol()
    commit = _require_commit()
    records: dict[str, Any] = {
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
    run_dir().mkdir(parents=True, exist_ok=True)
    atomic_write_json(run_dir() / RECORDS_FILE, records)
    from .kokoro_synthesis import KokoroSynthesisRuntime

    try:
        runtime = KokoroSynthesisRuntime.load(download=False)
        _render_candidates(protocol, records, runtime, base_dir=run_dir())
    except Exception as exc:
        payload = {
            "schema_version": 1,
            "run_id": RUN_ID,
            "protocol_sha256": protocol["protocol_sha256"],
            "classification": "cluster_shell_v3_runtime_inconclusive",
            "automatic_pass": False,
            "pending_human_review": False,
            "failure": f"{type(exc).__name__}: {exc}"[:1000],
            "render_records_sha256": sha256_file(run_dir() / RECORDS_FILE),
            "api_calls_made": 0,
            "production_enabled": False,
        }
        result = {**payload, "analysis_sha256": sha256_json(payload)}
        atomic_write_json(analysis_path, result)
        return result

    records = _load_json(run_dir() / RECORDS_FILE)
    anchors = _anchors(protocol)
    fixtures: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for record in records["fixtures"]:
        try:
            fixtures.append(
                _analyze_fixture(
                    record,
                    anchors,
                    base_dir=run_dir(),
                )
            )
        except Exception as exc:
            failures.append(
                {
                    "fixture_id": record["fixture_id"],
                    "failure": f"{type(exc).__name__}: {exc}"[:1000],
                }
            )
    passed = bool(
        not failures
        and len(fixtures) == len(protocol["fixtures"])
        and all(row["automatic_pass"] for row in fixtures)
    )
    payload = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "protocol_sha256": protocol["protocol_sha256"],
        "classification": (
            "cluster_shell_v3_aggregate_automatic_pass_pending_human_qc"
            if passed
            else (
                "cluster_shell_v3_measurement_inconclusive"
                if failures
                else "cluster_shell_v3_aggregate_automatic_failed"
            )
        ),
        "automatic_pass": passed,
        "pending_human_review": passed,
        "fixtures": fixtures,
        "measurement_failures": failures,
        "selected_analysis_families": protocol["analysis_family_selection"][
            "selected_by_fixture_position"
        ],
        "v2_failure_preserved": "cluster_shell_v2_anchor_calibration_failed",
        "api_calls_made": 0,
        "decoder_attempt_count": records["decoder_attempt_count"],
        "render_records_sha256": sha256_file(run_dir() / RECORDS_FILE),
        "production_enabled": False,
    }
    result = {**payload, "analysis_sha256": sha256_json(payload)}
    atomic_write_json(analysis_path, result)
    return result
