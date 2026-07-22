from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
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
from .kokoro_typed_engine import KokoroTypedPlanner, TypedPlan, local_engine_assets
from .sentence_pair_v2_analysis import CEILINGS
from .util import atomic_write_json, sha256_file


RUN_ID = "20260716-kokoro-typed-replication-v1"
ENGINE_RUN_ID = "20260716-kokoro-typed-engine-v1"
PARENT_V4_RUN_ID = "20260716-kokoro-common-rng-confirmation-v4"
PARENT_QC_RUN_ID = "20260716-kokoro-stronger-span-product-qc-v1"
ENGINE_COMMIT = "ce7ea2ac780bfba760ad4a4dbbbac7a7749fc263"
SELECTED_SPAN = "target-word"
BLIND_SEED = 20_260_716_03
LOCALIZATION_MINIMUM = 0.80
TARGET_CUE_PADDING_S = 0.150
PRAAT = Path("/Applications/Praat.app/Contents/MacOS/Praat")
MEASUREMENT_SCRIPT = Paths().root / "scripts" / "praat_sentence_pair_v2_burg.praat"


@dataclass(frozen=True)
class Fixture:
    fixture_id: str
    role: str
    text: str
    expected_plan_sha256: str
    expected_target_word_indexes: tuple[int, ...]
    expected_target_occurrences: int


FIXTURES = (
    Fixture(
        fixture_id="single-target",
        role="one eligible target in a short declarative carrier",
        text="The map rests.",
        expected_plan_sha256=(
            "afff11cab7e8e91fec0403e8c701f1059efb46c4369fed67b31dc141ce1d6fcf"
        ),
        expected_target_word_indexes=(1,),
        expected_target_occurrences=1,
    ),
    Fixture(
        fixture_id="multi-target-repeated",
        role="two occurrences of one repeated eligible source word",
        text="The map shows the map.",
        expected_plan_sha256=(
            "216acd77e421233d8731ded360af5798b7071a1e90b64480b9cebc165e5725d5"
        ),
        expected_target_word_indexes=(1, 4),
        expected_target_occurrences=2,
    ),
    Fixture(
        fixture_id="rhythm-punctuation-weak",
        role="commas plus repeated weak-function-word behavior",
        text="The map, in the sun, will rest.",
        expected_plan_sha256=(
            "debeb7dff16cbdba3495ccbc34241b161a742b724152b220e5015a317440c44a"
        ),
        expected_target_word_indexes=(1,),
        expected_target_occurrences=1,
    ),
)


def run_dir() -> Path:
    return Paths().artifacts / "typed-engine" / RUN_ID


def _artifact(run_id: str, filename: str) -> Path:
    family = "typed-engine" if run_id == ENGINE_RUN_ID else "phoneme-renderer"
    return Paths().artifacts / family / run_id / filename


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"required frozen artifact is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _verified_parents() -> dict[str, Any]:
    parity_path = _artifact(ENGINE_RUN_ID, "parity.json")
    parity = _load_json(parity_path)
    if parity.get("pass") is not True or parity.get("status") != "passed":
        raise RuntimeError("frozen-v4 implementation parity is not passing")

    v4_protocol_path = _artifact(PARENT_V4_RUN_ID, "protocol.json")
    v4_records_path = _artifact(PARENT_V4_RUN_ID, "records.json")
    v4_summary_path = _artifact(PARENT_V4_RUN_ID, "summary.json")
    v4_protocol = _load_json(v4_protocol_path)
    v4_summary = _load_json(v4_summary_path)
    target_word = v4_summary["pair_results"][SELECTED_SPAN]
    if target_word.get("pass") is not True:
        raise RuntimeError("frozen v4 target-word candidate lacks its acoustic pass")

    qc_protocol_path = _artifact(PARENT_QC_RUN_ID, "protocol.json")
    qc_result_path = _artifact(PARENT_QC_RUN_ID, "manual-result.json")
    qc_protocol = _load_json(qc_protocol_path)
    qc_result = _load_json(qc_result_path)
    selection = qc_result.get("selection", {})
    if selection.get("selected_candidate") != SELECTED_SPAN:
        raise RuntimeError("the frozen product-QC selection is not target-word")
    if qc_result.get("session_outcome", {}).get("classification") != (
        "clean_catch_candidate_passes"
    ):
        raise RuntimeError(
            "the frozen stronger-span session did not authorize replication"
        )

    return {
        "engine_parity": {
            "run_id": ENGINE_RUN_ID,
            "file_sha256": sha256_file(parity_path),
            "neutral_pcm_sha256": parity["actual"]["neutral"]["pcm_sha256"],
            "identity_pcm_sha256": parity["actual"]["identity"]["pcm_sha256"],
            "lens_pcm_sha256": parity["actual"]["lens"]["pcm_sha256"],
            "pass": True,
        },
        "v4": {
            "run_id": PARENT_V4_RUN_ID,
            "protocol_sha256": v4_protocol["protocol_sha256"],
            "protocol_file_sha256": sha256_file(v4_protocol_path),
            "records_sha256": sha256_file(v4_records_path),
            "summary_sha256": sha256_file(v4_summary_path),
            "selected_span": SELECTED_SPAN,
            "selected_span_acoustic_pass": True,
            "selected_span_localization": target_word["difference_localization"],
            "context_anchor_geometry": v4_summary["context_anchor_geometry"],
        },
        "creator_product_qc": {
            "run_id": PARENT_QC_RUN_ID,
            "protocol_sha256": qc_protocol["protocol_sha256"],
            "protocol_file_sha256": sha256_file(qc_protocol_path),
            "manual_result_sha256": sha256_file(qc_result_path),
            "outcome": "clean_catch_candidate_passes",
            "selected_candidate": SELECTED_SPAN,
        },
    }


def _plan_record(fixture: Fixture, plan: TypedPlan) -> dict[str, Any]:
    if plan.plan_sha256 != fixture.expected_plan_sha256:
        raise RuntimeError(f"{fixture.fixture_id} plan hash changed")
    if plan.target_word_indexes != fixture.expected_target_word_indexes:
        raise RuntimeError(f"{fixture.fixture_id} target word indexes changed")
    if plan.target_occurrence_count != fixture.expected_target_occurrences:
        raise RuntimeError(f"{fixture.fixture_id} target occurrence count changed")
    if not plan.comparison_available or not plan.gate_summary.espeak_gate_pass:
        raise RuntimeError(f"{fixture.fixture_id} no longer passes the spelling gate")
    if not plan.gate_summary.kokoro_phone_gate_pass:
        raise RuntimeError(f"{fixture.fixture_id} no longer passes the Kokoro gate")

    words = [asdict(word) for word in plan.words]
    if fixture.fixture_id == "multi-target-repeated":
        target_words = [words[index] for index in fixture.expected_target_word_indexes]
        signatures = {
            (
                row["neutral_surface"],
                row["lens_surface"],
                row["neutral_phone"],
                row["lens_phone"],
            )
            for row in target_words
        }
        if len(signatures) != 1:
            raise RuntimeError("repeated target mapping is not invariant")

    return {
        "fixture_id": fixture.fixture_id,
        "role": fixture.role,
        "source_text": fixture.text,
        "plan_sha256": plan.plan_sha256,
        "source_phonemes": plan.source_phonemes,
        "neutral_phonemes": plan.neutral_phonemes,
        "lens_phonemes": plan.lens_phonemes,
        "neutral_script": plan.neutral_script,
        "lens_script": plan.lens_script,
        "target_word_indexes": list(plan.target_word_indexes),
        "target_word_count": plan.target_word_count,
        "target_occurrence_count": plan.target_occurrence_count,
        "coverage_count": plan.coverage_count,
        "words": words,
        "gate_summary": asdict(plan.gate_summary),
    }


def _fixture_records() -> list[dict[str, Any]]:
    planner = KokoroTypedPlanner.load()
    records: list[dict[str, Any]] = []
    for fixture in FIXTURES:
        first = planner.plan(fixture.text)
        second = planner.plan(fixture.text)
        if stable_json(asdict(first)) != stable_json(asdict(second)):
            raise RuntimeError(f"{fixture.fixture_id} plan is not exactly repeatable")
        records.append(_plan_record(fixture, first))
    return records


def logical_trials() -> list[dict[str, Any]]:
    return [
        *(
            {
                "logical_id": f"{fixture.fixture_id}__identity-catch",
                "fixture_id": fixture.fixture_id,
                "condition": "identity-catch",
                "source_roles": ["neutral", "identity"],
            }
            for fixture in FIXTURES
        ),
        *(
            {
                "logical_id": f"{fixture.fixture_id}__lens-candidate",
                "fixture_id": fixture.fixture_id,
                "condition": "lens-candidate",
                "source_roles": ["neutral", "lens"],
            }
            for fixture in FIXTURES
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


def protocol_record() -> dict[str, Any]:
    parents = _verified_parents()
    fixtures = _fixture_records()
    assets = local_engine_assets()
    payload = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "status": "frozen_before_any_novel_replication_fixture_audio",
        "question": (
            "Does the arbitrary-text engine reproduce the selected target-word "
            "/ae/->/eh/ shared-state/common-RNG candidate across three distinct, "
            "preregistered carrier structures?"
        ),
        "scope": {
            "rule": "ptbr.vowel.ae_to_eh",
            "source_category": "/ae/",
            "lens_category": "/eh/",
            "candidate_span": SELECTED_SPAN,
            "fixtures": 3,
            "triplets_per_fixture": 1,
            "logical_wav_outputs": 9,
            "api_calls": 0,
            "openai_calls": 0,
            "paid_calls": 0,
            "listener": "one informed creator-listener",
            "interpretation": (
                "typed-engine generalization and product/artifact QC only; not "
                "Brazilian-Portuguese population or profile-fit evidence"
            ),
        },
        "parents": parents,
        "engine": {
            "commit": ENGINE_COMMIT,
            "typed_engine_module_sha256": sha256_file(
                Paths().root / "src" / "earshift_bakeoff" / "kokoro_typed_engine.py"
            ),
            "synthesis_module_sha256": sha256_file(
                Paths().root / "src" / "earshift_bakeoff" / "kokoro_synthesis.py"
            ),
            "assets": assets,
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
            "pair_contract": (
                "source-derived durations/alignment; neutral-carrier F0/noise; "
                "separate neutral/lens text states; replace the union of complete "
                "target-word columns; identical voice, speed, length, and decoder "
                "excitation; render each triplet atomically"
            ),
        },
        "fixtures": fixtures,
        "render_manifest": [
            {
                "order": order,
                "slot_id": f"{fixture.fixture_id}__{role}",
                "fixture_id": fixture.fixture_id,
                "role": role,
                "plan_sha256": fixture.expected_plan_sha256,
            }
            for order, (fixture, role) in enumerate(
                (
                    (fixture, role)
                    for fixture in FIXTURES
                    for role in ("neutral", "identity", "lens")
                ),
                start=1,
            )
        ],
        "runtime_gate": {
            "must_pass": [
                "exact frozen fixture plan and gate receipt",
                "one and only one atomic triplet per fixture",
                "neutral and identity PCM16 are bit-identical",
                "neutral, identity, and lens have equal nonzero sample counts",
                "all samples are finite",
                "clipped fraction is below 0.001 on every side",
                "predicted-duration count matches the complete model-token plan",
                "replaced columns equal the complete target-word-column union",
            ],
            "production_path": True,
        },
        "target_alignment": {
            "source": "the engine's source-derived predicted durations",
            "column_mapping": (
                "model columns include the leading boundary at index zero; each "
                "lexical target offset maps to its word column at the same offset"
            ),
            "sample_mapping": (
                "derive samples per alignment frame as decoded sample count divided "
                "by sum(predicted durations); require a positive integer ratio"
            ),
            "measurement_interval": (
                "from the stress-marker column immediately preceding each target "
                "vowel through the end of that vowel column"
            ),
            "complete_word_interval": (
                "from the first through the last replaced column of each target word"
            ),
            "selection": "none; every preregistered target occurrence is measured",
        },
        "replication_only_acoustic_gate": {
            "never_a_runtime_gate": True,
            "instrument": "standalone Praat Burg",
            "praat_sha256": sha256_file(PRAAT),
            "measurement_script_sha256": sha256_file(MEASUREMENT_SCRIPT),
            "maximum_formant_hz_family": list(CEILINGS),
            "number_of_formants": 5,
            "window_s": 0.025,
            "time_step_s": 0.005,
            "measurement_region": "middle 50 percent of each stress-plus-target interval",
            "retention": "at least 5 valid F1/F2 frames and at least 60 percent",
            "plausibility_hz": {
                "f1": [180, 1200],
                "f2": [600, 3500],
                "minimum_f2_minus_f1": 250,
            },
            "anchor_geometry": parents["v4"]["context_anchor_geometry"],
            "per_occurrence_per_ceiling": (
                "neutral is nearer the frozen full-carrier /ae/ endpoint; lens is "
                "nearer its /eh/ endpoint; vector cosine with the endpoint vector "
                "is at least 0.5; magnitude is at least max(0.25 Bark, half the "
                "frozen full-carrier endpoint distance); both measurements are plausible"
            ),
            "fixture_pass": "every target occurrence passes at all three ceilings",
            "run_pass": "all three fixtures pass",
            "no_acoustic_selection": True,
        },
        "replication_only_localization_gate": {
            "never_a_runtime_gate": True,
            "difference": "lens PCM16 minus neutral PCM16",
            "inside": (
                "union of complete target-word intervals expanded by 150 ms on both "
                "sides, clipped to file bounds"
            ),
            "minimum_inside_squared_difference_energy_fraction": LOCALIZATION_MINIMUM,
            "outside_rms_pcm": "reported descriptively",
            "fixture_pass": "inside fraction >= 0.80",
            "run_pass": "all three fixtures pass",
        },
        "blind_listener_protocol": {
            "logical_trials": logical_trials(),
            "frozen_layout": blinded_trial_plan(),
            "blinding_seed": BLIND_SEED,
            "trial_order": "deterministically randomized",
            "side_order": "independently and deterministically randomized",
            "hidden": [
                "source text",
                "carrier script",
                "phonemes",
                "token identity",
                "condition",
                "role",
                "filename",
                "blind key",
            ],
            "cue": (
                "the same synchronized target-word-position cue appears on both "
                "branches and both sides of each fixture; multiple targets receive "
                "multiple cues"
            ),
            "branches_per_fixture": [
                "neutral versus bit-identical identity catch",
                "neutral versus lens candidate",
            ],
            "all_ratings_before_decode": True,
            "raw_response_preserved_byte_for_byte": True,
            "response_schema": {
                "per_side": {
                    "naturalness": "integer 1-5; pass 4 or 5",
                    "delivery": "sentence-like passes; list-like does not",
                    "stable_recoverable_meaning": "none passes",
                    "artifact": "none or minor passes",
                },
                "pair": {
                    "difference_strength": "integer 1-7",
                    "category_judgment": "A, B, same, uncertain, or neither",
                    "confidence": "integer 1-5",
                    "unrelated_interference": (
                        "none, manageable, dominant, or uncertain"
                    ),
                    "replay_count": "descriptive only",
                    "notes": "optional string",
                },
            },
            "candidate_branch_pass": (
                "both sides satisfy all side gates; difference strength >=5; "
                "the actual lens side is judged closer to the vowel in bet; "
                "confidence >=3; interference is none or manageable"
            ),
            "identity_branch_pass": (
                "both sides satisfy all side gates; difference strength ==1; "
                "category judgment is same or neither; interference is none or manageable"
            ),
            "fixture_pass": "both its candidate and identity branches pass",
            "run_pass": "all three fixtures pass",
        },
        "predetermined_outcomes": {
            "replication_pass": (
                "only if every runtime, acoustic, localization, candidate-listening, "
                "and identity-catch gate passes for all three fixtures: promote the "
                "typed target-word engine to the production candidate path with a "
                "bounded one-rule claim"
            ),
            "automatic_failure": (
                "retain all artifacts; do not open listener review for a failed "
                "fixture; do not promote the architecture"
            ),
            "manual_failure_or_catch_flag": (
                "retain all automatic evidence and raw ratings; do not promote; do "
                "not characterize the listener as unreliable"
            ),
            "instrument_or_analysis_failure": (
                "classify replication as inconclusive_measurement_failure unless the "
                "frozen WAV itself violates a preregistered runtime gate"
            ),
        },
        "stopping_rule": (
            "Render exactly one neutral/identity/lens triplet for each of the three "
            "fixtures, with no replacement, selection, rerender, fixture change, "
            "threshold change, or parameter tuning inside this run. Analyze every "
            "returned file and target occurrence. Promotion occurs only after the "
            "single frozen blind creator session also passes."
        ),
    }
    return {**payload, "protocol_sha256": sha256_json(payload)}


def prepare() -> dict[str, Any]:
    protocol = protocol_record()
    destination = run_dir() / "protocol.json"
    if destination.is_file():
        existing = json.loads(destination.read_text(encoding="utf-8"))
        if stable_json(existing) != stable_json(protocol):
            raise RuntimeError(
                "existing typed replication protocol differs from freeze"
            )
    else:
        atomic_write_json(destination, protocol)
    return protocol
