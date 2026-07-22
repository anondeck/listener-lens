from __future__ import annotations

import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .config import Paths, sha256_json, stable_json
from .kokoro_synthesis import (
    KOKORO_VERSION,
    MODEL_HASHES,
    MODEL_REPO,
    MODEL_REVISION,
    RNG_SEED,
    SAMPLE_RATE_HZ,
    SPEED,
)
from .kokoro_typed_diagnostic_protocol import CONFIRMATION_FIXTURES
from .kokoro_typed_engine import KokoroTypedPlanner, local_engine_assets
from .util import atomic_write_json, sha256_file


RUN_ID = "20260717-kokoro-typed-confirmation-v1"
PARENT_DIAGNOSTIC_RUN_ID = "20260717-kokoro-typed-diagnostic-v1"
FROZEN_REPLICATION_RUN_ID = "20260716-kokoro-typed-replication-v1"
SELECTED_SPAN = "target-word"
EXPECTED_DIAGNOSTIC_CLASSIFICATION = (
    "transported_calibration_mechanically_sufficient_for_this_fixture"
)
EXPECTED_MECHANICAL_ATTRIBUTION = "mixed_transported_endpoint_and_threshold_calibration"
PRIMARY_WINDOW_PERCENT = 50
DESCRIPTIVE_WINDOW_PERCENTS = (40, 60)
WINDOW_PERCENTS = (40, 50, 60)
CEILINGS_HZ = (5500, 5750, 6000)
LOCALIZATION_MINIMUM = 0.80
MINIMUM_DIRECTION_COSINE = 0.50
MINIMUM_MAGNITUDE_BARK = 0.25
MINIMUM_VALID_FRAMES = 5
MINIMUM_VALID_FRACTION = 0.60
BLIND_SEED = 20_260_717_02
REVIEW_RESPONSE_FILENAME = "kokoro-en-typed-confirmation-v1-response.json"
REVIEW_ESTIMATED_MINUTES = 8
PRAAT = Path("/Applications/Praat.app/Contents/MacOS/Praat")
MEASUREMENT_SCRIPT = Paths().root / "scripts" / "praat_sentence_pair_v2_burg.praat"


ANCHOR_OCCURRENCE_MAP = {
    "new-repeated-phrase-final": (0, 1),
    "independent-phrase-final-only": (1,),
}


def run_dir() -> Path:
    return Paths().artifacts / "typed-engine" / RUN_ID


def diagnostic_dir() -> Path:
    return Paths().artifacts / "typed-engine" / PARENT_DIAGNOSTIC_RUN_ID


def frozen_replication_dir() -> Path:
    return Paths().artifacts / "typed-engine" / FROZEN_REPLICATION_RUN_ID


def _load_json(path: Path) -> Any:
    if not path.is_file():
        raise RuntimeError(f"required frozen artifact is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def logical_trials() -> list[dict[str, Any]]:
    return [
        *(
            {
                "logical_id": f"{fixture.fixture_id}__identity-catch",
                "fixture_id": fixture.fixture_id,
                "condition": "identity-catch",
                "source_roles": ["neutral", "identity"],
            }
            for fixture in CONFIRMATION_FIXTURES
        ),
        *(
            {
                "logical_id": f"{fixture.fixture_id}__lens-candidate",
                "fixture_id": fixture.fixture_id,
                "condition": "lens-candidate",
                "source_roles": ["neutral", "lens"],
            }
            for fixture in CONFIRMATION_FIXTURES
        ),
    ]


def blinded_trial_plan() -> list[dict[str, Any]]:
    rng = random.Random(BLIND_SEED)
    trials = logical_trials()
    for trial in trials:
        trial["source_roles"] = list(trial["source_roles"])
        rng.shuffle(trial["source_roles"])
    rng.shuffle(trials)
    return [
        {
            **trial,
            "trial_id": f"comparison-{index:02d}",
            "side_roles": {
                side: role
                for side, role in zip(
                    ("A", "B"), trial.pop("source_roles"), strict=True
                )
            },
        }
        for index, trial in enumerate(trials, start=1)
    ]


def _verified_diagnostic_parent() -> dict[str, Any]:
    protocol_path = diagnostic_dir() / "protocol.json"
    records_path = diagnostic_dir() / "render-records.json"
    analysis_path = diagnostic_dir() / "analysis.json"
    protocol = _load_json(protocol_path)
    records = _load_json(records_path)
    analysis = _load_json(analysis_path)
    if analysis.get("classification") != EXPECTED_DIAGNOSTIC_CLASSIFICATION:
        raise RuntimeError(
            "diagnostic did not authorize the frozen confirmation branch"
        )
    if analysis.get("selected_span") != SELECTED_SPAN:
        raise RuntimeError("diagnostic selected a different confirmation span")
    if analysis.get("mechanical_attribution") != EXPECTED_MECHANICAL_ATTRIBUTION:
        raise RuntimeError("diagnostic mechanical attribution drifted")
    if analysis.get("confirmation_eligible") is not True:
        raise RuntimeError("diagnostic does not mark confirmation eligible")
    if analysis.get("frozen_replication_v1_preserved_failed") is not True:
        raise RuntimeError("diagnostic no longer preserves replication-v1 as failed")
    if records.get("status") != "complete" or not records.get(
        "one_attempt_slots_respected"
    ):
        raise RuntimeError("diagnostic render ledger is incomplete")
    if analysis.get("protocol_sha256") != protocol.get("protocol_sha256"):
        raise RuntimeError("diagnostic analysis/protocol binding drifted")
    if analysis.get("render_records_sha256") != sha256_file(records_path):
        raise RuntimeError("diagnostic analysis/render-record binding drifted")

    precommit = protocol["confirmation_precommit"]
    if precommit.get("same_set_for_every_selected_span") is not True:
        raise RuntimeError("diagnostic did not freeze one common confirmation set")
    expected = [asdict(fixture) for fixture in CONFIRMATION_FIXTURES]
    actual = [
        {key: row[key] for key in expected[index]}
        for index, row in enumerate(precommit["fixtures"])
    ]
    if stable_json(actual) != stable_json(expected):
        raise RuntimeError("diagnostic confirmation fixture precommit drifted")

    files = [protocol_path, records_path, analysis_path]
    for slot in records["slots"]:
        if slot["status"] == "complete":
            audio = diagnostic_dir() / slot["audio"]["relative_path"]
            if sha256_file(audio) != slot["audio"]["wav_sha256"]:
                raise RuntimeError(f"diagnostic parent WAV drifted: {audio}")
            files.append(audio)
        marker = diagnostic_dir() / "attempts" / f"{slot['slot_id']}.json"
        if marker.is_file():
            files.append(marker)
    return {
        "run_id": PARENT_DIAGNOSTIC_RUN_ID,
        "protocol_sha256": protocol["protocol_sha256"],
        "protocol_file_sha256": sha256_file(protocol_path),
        "render_records_sha256": sha256_file(records_path),
        "analysis_sha256": sha256_file(analysis_path),
        "classification": analysis["classification"],
        "selected_span": analysis["selected_span"],
        "mechanical_attribution": analysis["mechanical_attribution"],
        "confirmation_eligible": True,
        "local_anchor_geometry": analysis["anchors"]["windows"],
        "bound_output_files": [
            {
                "relative_path": str(path.relative_to(Paths().root)),
                "sha256": sha256_file(path),
            }
            for path in sorted(set(files))
        ],
    }


def _verified_frozen_failure() -> dict[str, Any]:
    protocol_path = frozen_replication_dir() / "protocol.json"
    records_path = frozen_replication_dir() / "render-records.json"
    analysis_path = frozen_replication_dir() / "analysis.json"
    protocol = _load_json(protocol_path)
    records = _load_json(records_path)
    analysis = _load_json(analysis_path)
    if analysis.get("classification") != "automatic_replication_failed_no_promotion":
        raise RuntimeError(
            "replication-v1 no longer has its immutable failed classification"
        )
    if records.get("status") != "render_complete":
        raise RuntimeError("replication-v1 render ledger is no longer complete")
    if records.get("one_pass_stopping_rule_satisfied") is not True:
        raise RuntimeError("replication-v1 stopping-rule evidence drifted")
    if protocol.get("protocol_sha256") != records.get("protocol_sha256"):
        raise RuntimeError("replication-v1 protocol/render-record binding drifted")
    if protocol.get("protocol_sha256") != analysis.get("protocol_sha256"):
        raise RuntimeError("replication-v1 protocol/analysis binding drifted")
    if analysis.get("render_records_sha256") != sha256_file(records_path):
        raise RuntimeError("replication-v1 analysis/render-record binding drifted")
    bound_wavs: list[dict[str, str]] = []
    for fixture in records.get("records", []):
        for role, audio in sorted(fixture["audio"].items()):
            path = frozen_replication_dir() / audio["relative_path"]
            actual = sha256_file(path)
            if actual != audio["wav_sha256"]:
                raise RuntimeError(f"replication-v1 parent WAV drifted: {path}")
            bound_wavs.append(
                {
                    "fixture_id": fixture["fixture_id"],
                    "role": role,
                    "relative_path": str(path.relative_to(Paths().root)),
                    "wav_sha256": actual,
                }
            )
    if len(bound_wavs) != records.get("logical_wav_outputs"):
        raise RuntimeError("replication-v1 bound WAV count drifted")
    return {
        "run_id": FROZEN_REPLICATION_RUN_ID,
        "protocol_sha256": protocol["protocol_sha256"],
        "protocol_file_sha256": sha256_file(protocol_path),
        "render_records_sha256": sha256_file(records_path),
        "analysis_sha256": sha256_file(analysis_path),
        "classification": analysis["classification"],
        "preservation": "immutable_failed_result_not_reclassified",
        "bound_wavs": bound_wavs,
    }


def _fixture_records() -> list[dict[str, Any]]:
    planner = KokoroTypedPlanner.load()
    records: list[dict[str, Any]] = []
    for fixture in CONFIRMATION_FIXTURES:
        first = planner.plan(fixture.text)
        second = planner.plan(fixture.text)
        if stable_json(asdict(first)) != stable_json(asdict(second)):
            raise RuntimeError(
                f"confirmation plan is not repeatable: {fixture.fixture_id}"
            )
        checks = {
            "plan_sha256": first.plan_sha256 == fixture.expected_plan_sha256,
            "source_phonemes": first.source_phonemes
            == fixture.expected_source_phonemes,
            "neutral_phonemes": first.neutral_phonemes
            == fixture.expected_neutral_phonemes,
            "lens_phonemes": first.lens_phonemes == fixture.expected_lens_phonemes,
            "target_word_indexes": first.target_word_indexes
            == fixture.expected_target_word_indexes,
            "target_occurrences": first.target_occurrence_count
            == fixture.expected_target_occurrences,
            "comparison_available": first.comparison_available,
            "espeak_gate": first.gate_summary.espeak_gate_pass,
            "kokoro_gate": first.gate_summary.kokoro_phone_gate_pass,
        }
        if not all(checks.values()):
            raise RuntimeError(f"confirmation fixture drifted: {fixture.fixture_id}")
        anchor_map = ANCHOR_OCCURRENCE_MAP[fixture.fixture_id]
        if len(anchor_map) != first.target_occurrence_count:
            raise RuntimeError("anchor occurrence map does not cover every target")
        records.append(
            {
                **asdict(fixture),
                "anchor_occurrence_map": list(anchor_map),
                "words": [asdict(word) for word in first.words],
                "gate_summary": asdict(first.gate_summary),
                "plan_checks": checks,
            }
        )
    return records


def _tracked_before_render(
    diagnostic: dict[str, Any], frozen_failure: dict[str, Any]
) -> list[str]:
    paths = [
        "src/earshift_bakeoff/kokoro_typed_confirmation.py",
        "src/earshift_bakeoff/kokoro_typed_confirmation_protocol.py",
        "src/earshift_bakeoff/kokoro_typed_diagnostic.py",
        "src/earshift_bakeoff/kokoro_typed_diagnostic_protocol.py",
        "src/earshift_bakeoff/kokoro_synthesis.py",
        "src/earshift_bakeoff/kokoro_typed_engine.py",
        "src/earshift_bakeoff/config.py",
        "src/earshift_bakeoff/util.py",
        "scripts/run_kokoro_typed_confirmation_v1.py",
        "scripts/praat_sentence_pair_v2_burg.praat",
        "uv.lock",
        f"artifacts/typed-engine/{RUN_ID}/protocol.json",
        f"artifacts/typed-engine/{FROZEN_REPLICATION_RUN_ID}/protocol.json",
        f"artifacts/typed-engine/{FROZEN_REPLICATION_RUN_ID}/render-records.json",
        f"artifacts/typed-engine/{FROZEN_REPLICATION_RUN_ID}/analysis.json",
    ]
    paths.extend(row["relative_path"] for row in diagnostic["bound_output_files"])
    paths.extend(row["relative_path"] for row in frozen_failure["bound_wavs"])
    return sorted(set(paths))


def protocol_record() -> dict[str, Any]:
    root = Paths().root
    diagnostic = _verified_diagnostic_parent()
    frozen_failure = _verified_frozen_failure()
    fixtures = _fixture_records()
    code_paths = {
        "confirmation": root
        / "src"
        / "earshift_bakeoff"
        / "kokoro_typed_confirmation.py",
        "confirmation_protocol": root
        / "src"
        / "earshift_bakeoff"
        / "kokoro_typed_confirmation_protocol.py",
        "config": root / "src" / "earshift_bakeoff" / "config.py",
        "diagnostic_measurement": root
        / "src"
        / "earshift_bakeoff"
        / "kokoro_typed_diagnostic.py",
        "runner": root / "scripts" / "run_kokoro_typed_confirmation_v1.py",
        "synthesis": root / "src" / "earshift_bakeoff" / "kokoro_synthesis.py",
        "typed_engine": root / "src" / "earshift_bakeoff" / "kokoro_typed_engine.py",
        "util": root / "src" / "earshift_bakeoff" / "util.py",
    }
    payload: dict[str, Any] = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "status": "frozen_before_any_confirmation_decode",
        "question": (
            "Does the fixed target-word controlled synthesis route satisfy the complete "
            "automatic local-anchor gate on the two precommitted unseen confirmation fixtures?"
        ),
        "scope": {
            "selected_span": SELECTED_SPAN,
            "fixture_count": 2,
            "required_decodes_per_fixture": ["neutral", "identity", "lens"],
            "maximum_decoder_attempts": 6,
            "api_calls": 0,
            "openai_calls": 0,
            "paid_calls": 0,
            "selection_or_replacement": "none",
            "listener": "one informed creator-listener only after automatic pass",
        },
        "claim_boundary": {
            "allowed": [
                "fresh_unseen_fixture_confirmation_automatic_pass_pending_human_review",
                "fresh_unseen_fixture_confirmation_automatic_failed_no_review",
                "fresh_unseen_fixture_confirmation_inconclusive_measurement_failure",
                "fresh_unseen_fixture_confirmation_inconclusive_runtime_failure",
                "bounded_controlled_target_word_generalization_only",
                "no_positive_generalization_claim",
            ],
            "forbidden": [
                "replication-v1 reclassified or rescued",
                "root cause confirmed",
                "position, duration, or coupling isolated",
                "Brazilian-Portuguese population or profile-fit evidence",
                "production promotion before blind human review",
            ],
        },
        "parents": {
            "diagnostic": diagnostic,
            "frozen_failed_replication_v1": frozen_failure,
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
                "device": "cpu",
                "rng_seed": RNG_SEED,
            },
            "committed_before_render": {
                "required": True,
                "tracked_clean_paths": _tracked_before_render(
                    diagnostic, frozen_failure
                ),
                "runtime_check": (
                    "every required path must be tracked by git and identical to HEAD before "
                    "the first attempt marker is written"
                ),
            },
        },
        "fixtures": fixtures,
        "render_manifest": [
            {
                "order": order,
                "slot_id": f"{fixture.fixture_id}__{role}",
                "fixture_id": fixture.fixture_id,
                "role": role,
                "plan_sha256": fixture.expected_plan_sha256,
                "selected_span": SELECTED_SPAN,
                "selected_columns": list(fixture.target_word_columns),
                "one_attempt_no_retry": True,
            }
            for order, (fixture, role) in enumerate(
                (
                    (fixture, role)
                    for fixture in CONFIRMATION_FIXTURES
                    for role in ("neutral", "identity", "lens")
                ),
                start=1,
            )
        ],
        "render_contract": {
            "per_fixture": (
                "derive that fixture's own source durations/alignment and neutral F0/noise; "
                "replace only its complete target-word state columns; decode neutral, "
                "identity, and lens with the same voice/style/speed/seed/excitation contract"
            ),
            "identity": "neutral and identity PCM16 must be bit-identical",
            "one_attempt": (
                "write all three role-specific attempt markers before the atomic fixture "
                "triplet; any interruption consumes the marked roles and forbids retry"
            ),
            "no_selection": "no rerender, replacement, parameter change, or listening selection",
        },
        "automatic_gate": {
            "primary_window_percent": PRIMARY_WINDOW_PERCENT,
            "descriptive_window_percents": list(DESCRIPTIVE_WINDOW_PERCENTS),
            "same_frame_table": True,
            "measurement_valid": (
                "at least 5 frames and 5 valid F1/F2 pairs, valid fraction >=0.60, "
                "and plausible F1/F2 for both neutral and lens"
            ),
            "occurrence_anchor_mapping": {
                key: list(value) for key, value in ANCHOR_OCCURRENCE_MAP.items()
            },
            "anchor_calibration_boundary": (
                "reuse the diagnostic's frozen ordinary exact-carrier endpoints by "
                "predeclared medial/phrase-final position; do not recalibrate on the "
                "unseen confirmation fixtures"
            ),
            "per_occurrence_per_ceiling": (
                "neutral nearer the mapped same-window local AE endpoint; lens nearer its "
                "local EH endpoint; vector cosine >=0.50; magnitude >=max(0.25 Bark, "
                "half local endpoint distance); both sides plausible"
            ),
            "descriptive_rule": (
                "40/60 use their same-window diagnostic endpoints and thresholds and set "
                "window_sensitive when any conjunct/verdict differs; they never change "
                "the primary-50 outcome"
            ),
            "runtime_integrity": [
                "all six outputs are nonempty finite mono PCM16 at 24 kHz",
                "clipped fraction below 0.001",
                "equal sample count within each triplet",
                "neutral/identity PCM16 bit identity",
                "predicted-duration count and complete target-word columns match each plan",
                "at least 0.80 positive squared PCM difference energy inside padded target-word windows",
                "new confirmation WAV hashes do not match any bound parent WAV hash",
            ],
            "fixture_pass": "all target occurrences pass every primary ceiling plus all runtime gates",
            "run_pass": "both frozen fixtures pass",
        },
        "predetermined_outcomes": {
            "automatic_pass": (
                "write the blind review session and classify "
                "fresh_unseen_fixture_confirmation_automatic_pass_pending_human_review; "
                "do not promote before its human result"
            ),
            "automatic_substantive_failure": (
                "retain every output, do not build a review, classify "
                "fresh_unseen_fixture_confirmation_automatic_failed_no_review"
            ),
            "measurement_or_instrument_failure": (
                "retain every output, do not build a review, classify "
                "fresh_unseen_fixture_confirmation_inconclusive_measurement_failure"
            ),
            "runtime_or_interruption_failure": (
                "consume the affected attempt markers, forbid retry, retain the ledger, "
                "and do not build a review"
            ),
        },
        "blind_review": {
            "only_after_automatic_pass": True,
            "purpose": "English controlled-engine unseen-fixture confirmation QC",
            "estimated_minutes": REVIEW_ESTIMATED_MINUTES,
            "response_filename": REVIEW_RESPONSE_FILENAME,
            "logical_trials": logical_trials(),
            "frozen_layout": blinded_trial_plan(),
            "blinding_seed": BLIND_SEED,
            "trial_count": 4,
            "hidden": [
                "source text",
                "carrier phones",
                "fixture id",
                "condition",
                "role",
                "filename",
                "blind key",
            ],
            "response_schema": {
                "per_side": {
                    "naturalness": "integer 1-5; pass 4 or 5",
                    "delivery": "sentence-like passes",
                    "stable_recoverable_meaning": "none passes",
                    "artifact": "none or minor passes",
                },
                "pair": {
                    "difference_strength": "integer 1-7",
                    "category_judgment": "A, B, same, uncertain, or neither",
                    "confidence": "integer 1-5",
                    "unrelated_interference": "none, manageable, dominant, or uncertain",
                    "replay_count": "descriptive only",
                },
            },
            "candidate_pass": (
                "both side gates pass; strength >=5; actual lens is closer to the vowel "
                "in bet; confidence >=3; interference none or manageable"
            ),
            "identity_pass": (
                "both side gates pass; strength ==1; category same or neither; "
                "interference none or manageable"
            ),
            "run_pass": "both branches pass for both fixtures",
            "raw_response_preserved_byte_for_byte": True,
        },
        "stopping_rule": (
            "After this protocol is committed, attempt each of the six fixed decodes at "
            "most once. Analyze every returned fixture and occurrence. Never rerender or "
            "change spans, fixtures, thresholds, anchors, parameters, or review layout."
        ),
    }
    return {**payload, "protocol_sha256": sha256_json(payload)}


def prepare() -> dict[str, Any]:
    protocol = protocol_record()
    destination = run_dir() / "protocol.json"
    if destination.is_file():
        existing = _load_json(destination)
        if stable_json(existing) != stable_json(protocol):
            raise RuntimeError(
                "existing confirmation protocol differs from the code-bound freeze"
            )
    else:
        forbidden = (
            run_dir() / "render-records.json",
            run_dir() / "analysis.json",
            run_dir() / "audio",
            run_dir() / "attempts",
            run_dir() / "review.html",
            run_dir() / "review-manifest.json",
        )
        if any(path.exists() for path in forbidden):
            raise RuntimeError("confirmation output exists before the protocol freeze")
        atomic_write_json(destination, protocol)
    return protocol
