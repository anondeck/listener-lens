from __future__ import annotations

import json
import random
import shutil
import subprocess
from pathlib import Path
from typing import Any

from .config import Paths, sha256_json, stable_json
from .kokoro_cluster_confirmation_v2 import _analyze_fixture, _render_candidates
from .kokoro_cluster_confirmation_v3 import (
    ANALYSIS_FILE as V3_ANALYSIS_FILE,
    PROTOCOL_FILE as V3_PROTOCOL_FILE,
    RECORDS_FILE as V3_RECORDS_FILE,
    RUN_ID as V3_RUN_ID,
    _verify_internal_hash,
    protocol_record as v3_protocol_record,
    run_dir as v3_run_dir,
)
from .kokoro_output_splice_unseen import _review_html as _shared_review_html
from .util import atomic_write_json, sha256_file


CONFIRMATION_VERSION = "kokoro-cluster-confirmation-v4"
RUN_ID = "20260718-kokoro-cluster-confirmation-v4"
PROTOCOL_FILE = "protocol.json"
RECORDS_FILE = "render-records.json"
ANALYSIS_FILE = "analysis.json"
REVIEW_FILE = "review.html"
REVIEW_MANIFEST_FILE = "review-manifest.json"
BLIND_KEY_FILE = "blind-key.json"
RESPONSE_FILENAME = "kokoro-cluster-confirmation-v4-response.json"
BLIND_SEED = 20_260_718_09

TRACKED_CLEAN_PATHS = (
    "src/earshift_bakeoff/kokoro_cluster_confirmation_v2.py",
    "src/earshift_bakeoff/kokoro_cluster_confirmation_v3.py",
    "src/earshift_bakeoff/kokoro_cluster_confirmation_v4.py",
    "src/earshift_bakeoff/kokoro_cluster_shell.py",
    "src/earshift_bakeoff/kokoro_output_domain_splice.py",
    "src/earshift_bakeoff/kokoro_output_splice_unseen.py",
    "src/earshift_bakeoff/kokoro_synthesis.py",
    "src/earshift_bakeoff/kokoro_typed_confirmation.py",
    "src/earshift_bakeoff/kokoro_typed_engine.py",
    "scripts/run_kokoro_cluster_confirmation_v4.py",
    f"artifacts/typed-engine/{RUN_ID}/{PROTOCOL_FILE}",
    f"artifacts/typed-engine/{V3_RUN_ID}/{V3_PROTOCOL_FILE}",
    f"artifacts/typed-engine/{V3_RUN_ID}/{V3_RECORDS_FILE}",
    f"artifacts/typed-engine/{V3_RUN_ID}/{V3_ANALYSIS_FILE}",
)


def run_dir() -> Path:
    return Paths().artifacts / "typed-engine" / RUN_ID


def _load_json(path: Path) -> Any:
    if not path.is_file():
        raise RuntimeError(f"required frozen artifact is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _verified_v3_parent() -> dict[str, Any]:
    protocol_path = v3_run_dir() / V3_PROTOCOL_FILE
    records_path = v3_run_dir() / V3_RECORDS_FILE
    analysis_path = v3_run_dir() / V3_ANALYSIS_FILE
    protocol = _load_json(protocol_path)
    records = _load_json(records_path)
    analysis = _load_json(analysis_path)
    if stable_json(protocol) != stable_json(v3_protocol_record()):
        raise RuntimeError("v3 protocol or its bound inputs drifted")
    _verify_internal_hash(analysis, "analysis_sha256", label="v3 analysis")
    if (
        analysis.get("classification") != "cluster_shell_v3_runtime_inconclusive"
        or "is not in the subpath" not in analysis.get("failure", "")
        or records.get("decoder_attempt_count") != 0
        or records.get("fixtures") != []
    ):
        raise RuntimeError("v3 parent is not the frozen path-plumbing failure")
    if sha256_file(records_path) != analysis.get("render_records_sha256"):
        raise RuntimeError("v3 records differ from the frozen analysis")
    return {
        "protocol": protocol,
        "binding": {
            "run_id": V3_RUN_ID,
            "protocol_sha256": protocol["protocol_sha256"],
            "protocol_file_sha256": sha256_file(protocol_path),
            "records_file_sha256": sha256_file(records_path),
            "analysis_sha256": analysis["analysis_sha256"],
            "analysis_file_sha256": sha256_file(analysis_path),
            "classification": analysis["classification"],
            "completed_decoder_slots": records["decoder_attempt_count"],
        },
    }


def protocol_record() -> dict[str, Any]:
    parent = _verified_v3_parent()
    v3 = parent["protocol"]
    payload: dict[str, Any] = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "version": CONFIRMATION_VERSION,
        "status": "frozen_before_candidate_render",
        "purpose": (
            "Execute the unchanged v3 family-gated candidate design after "
            "passing the versioned output-directory through audio metadata."
        ),
        "parent": parent["binding"],
        "fixtures": v3["fixtures"],
        "analysis_family_selection": v3["analysis_family_selection"],
        "path_correction": {
            "mechanism": "explicit_versioned_base_directory_for_audio_metadata",
            "temporary_directory_regression_required": True,
            "v3_candidate_evidence_reused": False,
        },
        "unchanged": v3["unchanged"],
        "predetermined_outcomes": v3["predetermined_outcomes"],
        "scope": {
            **v3["scope"],
            "candidate_decoder_slots": len(v3["fixtures"]) * 3,
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
            raise RuntimeError("existing v4 cluster protocol differs")
        return existing
    if any(
        (run_dir() / name).exists() for name in (RECORDS_FILE, ANALYSIS_FILE, "audio")
    ):
        raise RuntimeError("v4 output exists before its protocol")
    destination.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(destination, protocol)
    return protocol


def _checked_protocol() -> dict[str, Any]:
    frozen = _load_json(run_dir() / PROTOCOL_FILE)
    if stable_json(frozen) != stable_json(protocol_record()):
        raise RuntimeError("v4 protocol or a bound parent artifact drifted")
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
        raise RuntimeError("v4 inputs differ from committed HEAD")
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
            "classification": "cluster_shell_v4_runtime_inconclusive",
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
            fixtures.append(_analyze_fixture(record, anchors, base_dir=run_dir()))
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
            "cluster_shell_v4_aggregate_automatic_pass_pending_human_qc"
            if passed
            else (
                "cluster_shell_v4_measurement_inconclusive"
                if failures
                else "cluster_shell_v4_aggregate_automatic_failed"
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
        "v3_failure_preserved": "cluster_shell_v3_runtime_inconclusive",
        "api_calls_made": 0,
        "decoder_attempt_count": records["decoder_attempt_count"],
        "render_records_sha256": sha256_file(run_dir() / RECORDS_FILE),
        "production_enabled": False,
    }
    result = {**payload, "analysis_sha256": sha256_json(payload)}
    atomic_write_json(analysis_path, result)
    return result


def _layout(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    trials = [
        *(
            {
                "fixture_id": fixture["fixture_id"],
                "condition": "identity-control",
                "roles": ["neutral", "identity"],
            }
            for fixture in protocol["fixtures"]
        ),
        *(
            {
                "fixture_id": fixture["fixture_id"],
                "condition": "spliced-lens",
                "roles": ["neutral", "lens"],
            }
            for fixture in protocol["fixtures"]
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


def build_review() -> dict[str, Any]:
    protocol = _checked_protocol()
    analysis = _load_json(run_dir() / ANALYSIS_FILE)
    records = _load_json(run_dir() / RECORDS_FILE)
    if analysis.get("automatic_pass") is not True:
        raise RuntimeError("v4 automatic result does not authorize review")
    by_fixture = {row["fixture_id"]: row for row in records["fixtures"]}
    layout = _layout(protocol)
    review_audio = run_dir() / "review-audio"
    review_audio.mkdir(parents=True, exist_ok=True)
    public: list[dict[str, Any]] = []
    for trial in layout:
        record = by_fixture[trial["fixture_id"]]
        sides: list[dict[str, str]] = []
        for side, role in trial["side_roles"].items():
            source = run_dir() / record["audio"][role]["relative_path"]
            target = review_audio / f"{trial['trial_id'][-2:]}-{side.lower()}.wav"
            if target.exists():
                if sha256_file(target) != sha256_file(source):
                    raise RuntimeError("v4 blind audio copy drifted")
            else:
                shutil.copyfile(source, target)
            sides.append({"side": side, "audio": f"review-audio/{target.name}"})
        public.append(
            {
                "trial_id": trial["trial_id"],
                "duration_s": record["audio"]["neutral"]["sample_count"] / 24000,
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
        "trial_count": len(public),
        "response_filename": RESPONSE_FILENAME,
        "public_trials": public,
        "hidden_fields_absent": True,
    }
    for path, payload in (
        (run_dir() / BLIND_KEY_FILE, key),
        (run_dir() / REVIEW_MANIFEST_FILE, manifest),
    ):
        if path.exists() and stable_json(_load_json(path)) != stable_json(payload):
            raise RuntimeError(f"v4 review artifact drifted: {path.name}")
        if not path.exists():
            atomic_write_json(path, payload)
    html = _shared_review_html(public, protocol["protocol_sha256"])
    html = html.replace("20260717-kokoro-output-splice-unseen-v1", RUN_ID).replace(
        "kokoro-output-splice-unseen-v1-response.json", RESPONSE_FILENAME
    )
    review_path = run_dir() / REVIEW_FILE
    if review_path.exists() and review_path.read_text(encoding="utf-8") != html:
        raise RuntimeError("v4 review HTML drifted")
    if not review_path.exists():
        review_path.write_text(html, encoding="utf-8")
    return manifest
