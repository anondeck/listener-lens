from __future__ import annotations

import json
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
from .kokoro_typed_engine import KokoroTypedPlanner, local_engine_assets
from .util import atomic_write_json, sha256_file


RUN_ID = "20260717-kokoro-typed-diagnostic-v1"
FROZEN_V1_RUN_ID = "20260716-kokoro-typed-replication-v1"
FROZEN_V4_RUN_ID = "20260716-kokoro-common-rng-confirmation-v4"
REPEATED_FIXTURE_ID = "multi-target-repeated"
PRIMARY_WINDOW_PERCENT = 50
DESCRIPTIVE_WINDOW_PERCENTS = (40, 60)
WINDOW_PERCENTS = (40, 50, 60)
CEILINGS_HZ = (5500, 5750, 6000)
LOCALIZATION_MINIMUM = 0.80
TARGET_CUE_PADDING_S = 0.150
MINIMUM_ANCHOR_MAGNITUDE_BARK = 0.25
MINIMUM_DIRECTION_COSINE = 0.50
MINIMUM_VALID_FRAMES = 5
MINIMUM_CANDIDATE_VALID_FRACTION = 0.60
EXACT_ANCHOR_VALID_FRACTION = 1.0

PRAAT = Path("/Applications/Praat.app/Contents/MacOS/Praat")
MEASUREMENT_SCRIPT = Paths().root / "scripts" / "praat_sentence_pair_v2_burg.praat"

REPEATED_SOURCE_PHONEMES = "ðə mˈæp ʃˈOz ðə mˈæp."
REPEATED_NEUTRAL_PHONEMES = "zɪ ʒˈæʒ ɡˈOh zɪ ʒˈæʒ."
REPEATED_LENS_PHONEMES = "zɪ ʒˈɛʒ ɡˈOh zɪ ʒˈɛʒ."
REPEATED_PLAN_SHA256 = (
    "216acd77e421233d8731ded360af5798b7071a1e90b64480b9cebc165e5725d5"
)
REPEATED_TARGET_WORD_INDEXES = (1, 4)
REPEATED_TARGET_WORD_COLUMNS = (4, 5, 6, 7, 17, 18, 19, 20)
REPEATED_TARGET_WORD_PLUS_BOUNDARIES_COLUMNS = (
    3,
    4,
    5,
    6,
    7,
    8,
    16,
    17,
    18,
    19,
    20,
    21,
)
REPEATED_FULL_CONTEXTUAL_STATE_COLUMNS = tuple(range(23))
REPEATED_MEASUREMENT_COLUMNS = ((5, 6), (18, 19))
REPEATED_WORD_COLUMNS = ((4, 5, 6, 7), (17, 18, 19, 20))
STYLE_ROW = 20


@dataclass(frozen=True)
class DecoderSlot:
    order: int
    slot_id: str
    role: str
    phonemes: str | None
    span_id: str | None
    columns: tuple[int, ...]


DECODER_SLOTS = (
    DecoderSlot(
        order=1,
        slot_id="ordinary-anchor-ae",
        role="independent_ordinary_exact_carrier_anchor",
        phonemes=REPEATED_NEUTRAL_PHONEMES,
        span_id=None,
        columns=(),
    ),
    DecoderSlot(
        order=2,
        slot_id="ordinary-anchor-eh",
        role="independent_ordinary_exact_carrier_anchor",
        phonemes=REPEATED_LENS_PHONEMES,
        span_id=None,
        columns=(),
    ),
    DecoderSlot(
        order=3,
        slot_id="candidate-target-word-plus-boundaries",
        role="conditional_shared_state_candidate",
        phonemes=None,
        span_id="target-word-plus-boundaries",
        columns=REPEATED_TARGET_WORD_PLUS_BOUNDARIES_COLUMNS,
    ),
    DecoderSlot(
        order=4,
        slot_id="candidate-full-contextual-state",
        role="conditional_shared_state_candidate",
        phonemes=None,
        span_id="full-contextual-state",
        columns=REPEATED_FULL_CONTEXTUAL_STATE_COLUMNS,
    ),
)


@dataclass(frozen=True)
class ConfirmationFixture:
    fixture_id: str
    role: str
    text: str
    expected_plan_sha256: str
    expected_target_word_indexes: tuple[int, ...]
    expected_target_occurrences: int
    expected_source_phonemes: str
    expected_neutral_phonemes: str
    expected_lens_phonemes: str
    target_word_columns: tuple[int, ...]
    target_word_plus_boundaries_columns: tuple[int, ...]
    full_contextual_state_columns: tuple[int, ...]


CONFIRMATION_FIXTURES = (
    ConfirmationFixture(
        fixture_id="new-repeated-phrase-final",
        role="new repeated target with one phrase-final occurrence",
        text="The cap turns near the cap.",
        expected_plan_sha256=(
            "c83bab90075c75619ed7c164cb4f325fc94a1325cb71c0c7e0fb7e87ba36320b"
        ),
        expected_target_word_indexes=(1, 5),
        expected_target_occurrences=2,
        expected_source_phonemes="ðə kˈæp tˈɜɹnz nˌɪɹ ðə kˈæp.",
        expected_neutral_phonemes="zɪ vˈæʒ ʒˈʌbzŋ vˌIŋ zɪ vˈæʒ.",
        expected_lens_phonemes="zɪ vˈɛʒ ʒˈʌbzŋ vˌIŋ zɪ vˈɛʒ.",
        target_word_columns=(4, 5, 6, 7, 24, 25, 26, 27),
        target_word_plus_boundaries_columns=(
            3,
            4,
            5,
            6,
            7,
            8,
            23,
            24,
            25,
            26,
            27,
            28,
        ),
        full_contextual_state_columns=tuple(range(30)),
    ),
    ConfirmationFixture(
        fixture_id="independent-phrase-final-only",
        role="independent carrier with one phrase-final target",
        text="We rest near the cap.",
        expected_plan_sha256=(
            "1f03a5383c38d504bd5bbd565f105675081660e0c214e33c48189d3001c748d7"
        ),
        expected_target_word_indexes=(4,),
        expected_target_occurrences=1,
        expected_source_phonemes="wˌi ɹˈɛst nˌɪɹ ðə kˈæp.",
        expected_neutral_phonemes="zˌɪ ɹˈɪhm vˌIŋ zɪ vˈæʒ.",
        expected_lens_phonemes="zˌɪ ɹˈɪhm vˌIŋ zɪ vˈɛʒ.",
        target_word_columns=(19, 20, 21, 22),
        target_word_plus_boundaries_columns=(18, 19, 20, 21, 22, 23),
        full_contextual_state_columns=tuple(range(25)),
    ),
)


CONFIRMATION_PLANNING_REGISTER = (
    {
        "order": 1,
        "text": "The cap turns near the cap.",
        "plan_sha256": "c83bab90075c75619ed7c164cb4f325fc94a1325cb71c0c7e0fb7e87ba36320b",
        "result": "eligible_chosen_repeated_plus_phrase_final",
    },
    {
        "order": 2,
        "text": "The flag waits by the flag.",
        "plan_sha256": "c691f13216fa48aa58e9069c2001e4e07bec8c767311a35156e0f906eadfa2d7",
        "result": "eligible_not_chosen",
    },
    {
        "order": 3,
        "text": "A black flag marks the path.",
        "plan_sha256": None,
        "result": "bounded_gate_failure",
    },
    {
        "order": 4,
        "text": "We set the cap beside the map.",
        "plan_sha256": "4b93f8d2b5a4d1a74cc96f9cba384532d8deadcd8ec360a1fa705524033d2799",
        "result": "eligible_not_exact_repeated_same_target",
    },
    {
        "order": 5,
        "text": "The path bends past the cap.",
        "plan_sha256": "d7cf2d82d47068d96c68aa226ba0d988f00671f8a83d5391b5b8a0b4aa83d119",
        "result": "eligible_multi_target_not_exact_repeated",
    },
    {
        "order": 6,
        "text": "A lamp waits by the path.",
        "plan_sha256": None,
        "result": "bounded_gate_failure",
    },
    {
        "order": 7,
        "text": "We rest near the cap.",
        "plan_sha256": "1f03a5383c38d504bd5bbd565f105675081660e0c214e33c48189d3001c748d7",
        "result": "eligible_chosen_independent_phrase_final_only",
    },
    {
        "order": 8,
        "text": "The sun rests by the cap.",
        "plan_sha256": "5f273bf81bcec096f232319c55cc240e2a452a6f32e120f7ba2fe7cb636f0971",
        "result": "eligible_not_chosen",
    },
    {
        "order": 9,
        "text": "The train stops by the cap.",
        "plan_sha256": "fe99e84990820d47d910be0f79f5446e529f3196fbeedc5a916b3b14f58b1669",
        "result": "eligible_not_chosen",
    },
    {
        "order": 10,
        "text": "Please wait for the cap.",
        "plan_sha256": "b7a1a42ac2a4b0e243823cd01a6dae2d4f07645f2bf5ee4130f56756329dae65",
        "result": "eligible_not_chosen",
    },
    {
        "order": 11,
        "text": "A bird rests on the cap.",
        "plan_sha256": None,
        "result": "bounded_gate_failure",
    },
    {
        "order": 12,
        "text": "The map points to the cap.",
        "plan_sha256": "f4a38e4d56c1c9fa57bdef4e94af3a209ea13e74173426cb4bbe77521a23dfde",
        "result": "eligible_not_chosen",
    },
    {
        "order": 13,
        "text": "They looked at the cap.",
        "plan_sha256": "4d483e29770d8bdd161cf49251735dde06fd9235efcab3440c41129d43c8bfa1",
        "result": "eligible_not_chosen",
    },
    {
        "order": 14,
        "text": "The cup is near the cap.",
        "plan_sha256": "11ad6e70e90b3f4f3e40e33cb678ce82e59197765aec073938275ae35a674e51",
        "result": "eligible_not_chosen",
    },
    {
        "order": 15,
        "text": "We can find the cap.",
        "plan_sha256": "53b97a4cb6171785f4607338f718b56bd79535b1bb68a082e72ef64e19bc89ca",
        "result": "eligible_not_chosen",
    },
    {
        "order": 16,
        "text": "Please move that black cap.",
        "plan_sha256": "4af591071b1026e61607665d61ee63c1089f03209632f2583aadf1543d03d5a1",
        "result": "eligible_not_chosen",
    },
    {
        "order": 17,
        "text": "The boat passed the black flag.",
        "plan_sha256": "73356648a25887fe2cafbca08d87606461a8b46f088be6c110815a174750d65a",
        "result": "eligible_not_chosen",
    },
    {
        "order": 18,
        "text": "Please follow that path.",
        "plan_sha256": "28758b37675f07d82d20f39f6d071dac00fe0c79788ade8a88c5bc6b8b6c30ff",
        "result": "eligible_not_chosen",
    },
    {
        "order": 19,
        "text": "The note marks the path.",
        "plan_sha256": "218910e4a4e0d7bc455a89609642ff585ecfb0a393445ef4d608085968dd3720",
        "result": "eligible_not_chosen",
    },
    {
        "order": 20,
        "text": "We walked down the path.",
        "plan_sha256": "b11d438a7c858a4a2844f269982868120db2e44d9dc74d49892a3a451e201e47",
        "result": "eligible_not_chosen",
    },
)


def run_dir() -> Path:
    return Paths().artifacts / "typed-engine" / RUN_ID


def _track_a_artifact(run_id: str, filename: str) -> Path:
    family = "typed-engine" if run_id == FROZEN_V1_RUN_ID else "phoneme-renderer"
    return Paths().artifacts / family / run_id / filename


def _load_json(path: Path) -> Any:
    if not path.is_file():
        raise RuntimeError(f"required frozen artifact is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _v1_audio_manifest(records: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    base = _track_a_artifact(FROZEN_V1_RUN_ID, "render-records.json").parent
    for fixture in records["records"]:
        for role in ("neutral", "identity", "lens"):
            audio = fixture["audio"][role]
            path = base / audio["relative_path"]
            actual = sha256_file(path)
            if actual != audio["wav_sha256"]:
                raise RuntimeError(f"frozen v1 WAV hash drifted: {path}")
            rows.append(
                {
                    "run_id": FROZEN_V1_RUN_ID,
                    "fixture_id": fixture["fixture_id"],
                    "role": role,
                    "relative_path": str(path.relative_to(Paths().root)),
                    "wav_sha256": actual,
                    "pcm_sha256": audio["pcm_sha256"],
                }
            )
    return rows


def _v4_audio_manifest(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    base = _track_a_artifact(FROZEN_V4_RUN_ID, "records.json").parent
    for record in records:
        path = base / record["audio_relative_path"]
        actual = sha256_file(path)
        if actual != record["audio_sha256"]:
            raise RuntimeError(f"frozen v4 WAV hash drifted: {path}")
        rows.append(
            {
                "run_id": FROZEN_V4_RUN_ID,
                "slot_id": record["slot_id"],
                "relative_path": str(path.relative_to(Paths().root)),
                "wav_sha256": actual,
            }
        )
    return rows


def _walk_scalar_values(value: Any) -> list[str]:
    if isinstance(value, dict):
        return [item for child in value.values() for item in _walk_scalar_values(child)]
    if isinstance(value, list):
        return [item for child in value for item in _walk_scalar_values(child)]
    return [value] if isinstance(value, str) else []


def _verify_confirmation_novelty() -> None:
    forbidden = {fixture.text for fixture in CONFIRMATION_FIXTURES} | {
        fixture.expected_plan_sha256 for fixture in CONFIRMATION_FIXTURES
    }
    current = run_dir().resolve()
    collisions: list[str] = []
    for path in sorted(Paths().artifacts.rglob("*.json")):
        if current in path.resolve().parents:
            continue
        try:
            values = set(_walk_scalar_values(_load_json(path)))
        except (OSError, json.JSONDecodeError):
            continue
        matches = sorted(values & forbidden)
        if matches:
            collisions.append(f"{path.relative_to(Paths().root)}: {matches}")
    if collisions:
        raise RuntimeError(
            "confirmation fixture is not novel against prior Track A JSON artifacts: "
            + "; ".join(collisions)
        )


def _confirmation_records() -> list[dict[str, Any]]:
    _verify_confirmation_novelty()
    planner = KokoroTypedPlanner.load()
    rows: list[dict[str, Any]] = []
    for fixture in CONFIRMATION_FIXTURES:
        plan = planner.plan(fixture.text)
        expected = {
            "plan_sha256": fixture.expected_plan_sha256,
            "target_word_indexes": fixture.expected_target_word_indexes,
            "target_occurrence_count": fixture.expected_target_occurrences,
            "source_phonemes": fixture.expected_source_phonemes,
            "neutral_phonemes": fixture.expected_neutral_phonemes,
            "lens_phonemes": fixture.expected_lens_phonemes,
        }
        actual = {
            "plan_sha256": plan.plan_sha256,
            "target_word_indexes": plan.target_word_indexes,
            "target_occurrence_count": plan.target_occurrence_count,
            "source_phonemes": plan.source_phonemes,
            "neutral_phonemes": plan.neutral_phonemes,
            "lens_phonemes": plan.lens_phonemes,
        }
        if actual != expected:
            raise RuntimeError(f"confirmation plan drifted: {fixture.fixture_id}")
        if not plan.comparison_available:
            raise RuntimeError(
                f"confirmation fixture lost comparison: {fixture.fixture_id}"
            )
        rows.append(
            {
                **asdict(fixture),
                "comparison_available": plan.comparison_available,
                "gate_summary": asdict(plan.gate_summary),
            }
        )
    return rows


def _verified_parents() -> dict[str, Any]:
    v1_protocol_path = _track_a_artifact(FROZEN_V1_RUN_ID, "protocol.json")
    v1_records_path = _track_a_artifact(FROZEN_V1_RUN_ID, "render-records.json")
    v1_analysis_path = _track_a_artifact(FROZEN_V1_RUN_ID, "analysis.json")
    v1_protocol = _load_json(v1_protocol_path)
    v1_records = _load_json(v1_records_path)
    v1_analysis = _load_json(v1_analysis_path)
    if v1_analysis.get("classification") != "automatic_replication_failed_no_promotion":
        raise RuntimeError("frozen replication-v1 is no longer classified as failed")
    repeated_protocol = next(
        row
        for row in v1_protocol["fixtures"]
        if row["fixture_id"] == REPEATED_FIXTURE_ID
    )
    repeated_record = next(
        row for row in v1_records["records"] if row["fixture_id"] == REPEATED_FIXTURE_ID
    )
    if (
        repeated_protocol["plan_sha256"] != REPEATED_PLAN_SHA256
        or repeated_protocol["source_phonemes"] != REPEATED_SOURCE_PHONEMES
        or repeated_protocol["neutral_phonemes"] != REPEATED_NEUTRAL_PHONEMES
        or repeated_protocol["lens_phonemes"] != REPEATED_LENS_PHONEMES
        or tuple(repeated_record["replaced_columns"]) != REPEATED_TARGET_WORD_COLUMNS
    ):
        raise RuntimeError(
            "frozen repeated fixture no longer matches the diagnostic precommit"
        )

    v4_protocol_path = _track_a_artifact(FROZEN_V4_RUN_ID, "protocol.json")
    v4_records_path = _track_a_artifact(FROZEN_V4_RUN_ID, "records.json")
    v4_summary_path = _track_a_artifact(FROZEN_V4_RUN_ID, "summary.json")
    v4_protocol = _load_json(v4_protocol_path)
    v4_records = _load_json(v4_records_path)
    v4_summary = _load_json(v4_summary_path)
    geometry = v4_summary["context_anchor_geometry"]
    if geometry.get("pass") is not True:
        raise RuntimeError("frozen v4 context anchor geometry is not passing")

    audio_manifest = [
        *_v1_audio_manifest(v1_records),
        *_v4_audio_manifest(v4_records),
    ]
    if len(audio_manifest) != 22:
        raise RuntimeError("the diagnostic must bind exactly 9 v1 and 13 v4 WAVs")
    return {
        "frozen_failed_replication_v1": {
            "run_id": FROZEN_V1_RUN_ID,
            "protocol_sha256": v1_protocol["protocol_sha256"],
            "protocol_file_sha256": sha256_file(v1_protocol_path),
            "render_records_sha256": sha256_file(v1_records_path),
            "analysis_sha256": sha256_file(v1_analysis_path),
            "classification": v1_analysis["classification"],
            "preservation": "immutable_failed_result_not_reclassified",
            "repeated_fixture_record": repeated_record,
        },
        "transported_v4_calibration": {
            "run_id": FROZEN_V4_RUN_ID,
            "protocol_sha256": v4_protocol["protocol_sha256"],
            "protocol_file_sha256": sha256_file(v4_protocol_path),
            "records_sha256": sha256_file(v4_records_path),
            "summary_sha256": sha256_file(v4_summary_path),
            "context_anchor_geometry": geometry,
        },
        "bound_parent_wavs": audio_manifest,
    }


def protocol_record() -> dict[str, Any]:
    parents = _verified_parents()
    confirmation = _confirmation_records()
    root = Paths().root
    code_paths = {
        "diagnostic": root / "src" / "earshift_bakeoff" / "kokoro_typed_diagnostic.py",
        "diagnostic_protocol": root
        / "src"
        / "earshift_bakeoff"
        / "kokoro_typed_diagnostic_protocol.py",
        "synthesis": root / "src" / "earshift_bakeoff" / "kokoro_synthesis.py",
        "typed_engine": root / "src" / "earshift_bakeoff" / "kokoro_typed_engine.py",
        "bark_implementation": root / "src" / "earshift_bakeoff" / "same_take.py",
        "config": root / "src" / "earshift_bakeoff" / "config.py",
        "runner": root / "scripts" / "run_kokoro_typed_diagnostic.py",
        "util": root / "src" / "earshift_bakeoff" / "util.py",
    }
    payload: dict[str, Any] = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "status": "frozen_before_any_diagnostic_decoder_slot",
        "question": (
            "Can occurrence-specific exact-carrier calibration mechanically account "
            "for the frozen repeated-fixture gate failure, and if not, does the first "
            "predeclared broader controlled state span satisfy the complete local gate?"
        ),
        "claim_boundary": {
            "allowed": [
                "transported_calibration_mechanically_sufficient_for_this_fixture",
                "transported_endpoint_geometry_implicated",
                "transported_magnitude_threshold_implicated",
                "mixed_transported_endpoint_and_threshold_calibration",
                "phrase_final_reference_not_realized",
                "branch_conditional_span_candidate_selected_for_unseen_confirmation",
                "bounded_controlled_span_route_failed",
                "diagnostic_inconclusive_measurement_or_instrument_failure",
                "diagnostic_inconclusive_runtime_or_integrity_failure",
                "diagnostic_stopped_neutral_source_reference_failure",
                "candidate_localization_gate_failed",
            ],
            "forbidden": [
                "root cause confirmed",
                "position or duration causally isolated",
                "coupling causally isolated",
                "frozen replication-v1 rescued or reclassified",
                "Brazilian-Portuguese population or profile-fit evidence",
            ],
            "minimum_0_25_bark": "preregistered engineering design threshold, not an empirical perceptual boundary",
        },
        "phase_1_forensic_audit": {
            "frozen_outcome": (
                "replication-v1 remains immutable and failed under its frozen protocol; "
                "nothing in this diagnostic can reclassify it"
            ),
            "evidence_table": [
                {
                    "observation": "integrity_and_planning_gates",
                    "evidence": (
                        "all bound artifact hashes, neutral/identity PCM equality, nonempty "
                        "finite PCM, clipping, exact replacement columns, plan identity, "
                        "and stored formant plausibility checks pass"
                    ),
                    "implication": "no frozen integrity or column-coverage failure was found",
                },
                {
                    "observation": "repeated_occurrence_duration_asymmetry",
                    "evidence": {
                        "medial_target_word_alignment_frames": 9,
                        "phrase_final_target_word_alignment_frames": 21,
                        "medial_target_word_ms": 225,
                        "phrase_final_target_word_ms": 525,
                        "medial_measurement_interval_ms": 125,
                        "phrase_final_measurement_interval_ms": 200,
                    },
                    "implication": (
                        "position, duration, and phrase-finality covary in this fixture and "
                        "cannot be causally separated by the frozen WAV"
                    ),
                },
                {
                    "observation": "latent_to_decoded_attenuation",
                    "evidence": {
                        "medial_latent_delta_energy": 45.7457,
                        "phrase_final_latent_delta_energy": 46.0768,
                        "medial_decoded_word_delta_rms_pcm": 1873.85,
                        "phrase_final_decoded_word_delta_rms_pcm": 369.32,
                        "medial_decoded_word_delta_energy": 18.96e9,
                        "phrase_final_decoded_word_delta_energy": 1.72e9,
                    },
                    "implication": (
                        "similar latent change energy can yield a much smaller decoded "
                        "phrase-final word difference; this does not identify the mechanism"
                    ),
                },
                {
                    "observation": "measurement_window_sensitivity",
                    "evidence": (
                        "the frozen repeated phrase-final occurrence passes the descriptive "
                        "middle-40-percent family but fails the primary 50 and descriptive 60"
                    ),
                    "implication": (
                        "window choice is a material sensitivity; 40/60 cannot rescue, veto, "
                        "or select any branch"
                    ),
                },
                {
                    "observation": "transported_context_mismatch",
                    "evidence": (
                        "replication-v1 applies endpoints transported from the v4 "
                        "/tæʧ/→/tɛʧ/ shell and its duration/context to the repeated "
                        "/ʒæʒ/→/ʒɛʒ/ shell"
                    ),
                    "implication": (
                        "the transported calibration is a high-priority confound, not a "
                        "confirmed cause"
                    ),
                },
            ],
            "ranked_hypotheses": [
                {
                    "rank": 1,
                    "hypothesis": "transported_calibration_and_context_mismatch",
                    "strength": "highest_confidence_confound",
                    "claim_limit": "not causal until the frozen 2x2 rescore is complete",
                },
                {
                    "rank": 2,
                    "hypothesis": "inseparable_position_duration_and_phrase_final_sensitivity",
                    "strength": "high",
                    "claim_limit": "the frozen fixture cannot separate these factors",
                },
                {
                    "rank": 3,
                    "hypothesis": "synthesis_control_or_state_span_weakness",
                    "strength": "medium_high",
                    "claim_limit": "conditional span results cannot isolate coupling mechanism",
                },
                {
                    "rank": 4,
                    "hypothesis": "measurement_window_sensitivity",
                    "strength": "medium",
                    "claim_limit": "40/60 are descriptive only; 50 remains primary",
                },
                {
                    "rank": 5,
                    "hypothesis": "causal_implementation_or_alignment_defect",
                    "strength": "unsupported_low",
                    "claim_limit": (
                        "the exact columns and frozen measurements are internally consistent; "
                        "noncausal evidence-chain bugs below do not explain the positive-energy failure"
                    ),
                },
            ],
            "noncausal_old_code_findings": [
                {
                    "finding": "zero-delta localization in v1 reports fraction 1.0/pass",
                    "frozen_result_effect": (
                        "none: every frozen candidate has positive difference energy"
                    ),
                    "remediation": (
                        "only this new version returns total_difference_energy_positive=false "
                        "and fails zero-delta localization"
                    ),
                },
                {
                    "finding": (
                        "the v1 analyzer does not independently rehash every WAV/exact fixture "
                        "set at analysis time and its analysis output can be overwritten"
                    ),
                    "frozen_result_effect": (
                        "none found: the audit independently rehashed all frozen parents"
                    ),
                    "remediation": (
                        "only this new version binds all parent WAV/code hashes, uses one-attempt "
                        "markers, and writes the terminal analysis immutably"
                    ),
                },
            ],
        },
        "internal_review_resolution": {
            "status": "two_independent_read_only_critiques_completed_and_resolved_before_freeze",
            "novel_audio_available_to_critics": False,
            "all_findings_resolved": True,
            "critics": [
                {
                    "role": "acoustic_and_instrument_validity",
                    "independent_read_only": True,
                    "findings_and_resolutions": [
                        {
                            "finding": (
                                "the 40/50/60 measurement family lacked a single "
                                "predeclared decision role"
                            ),
                            "resolution": (
                                "50 is the sole primary branch; 40/60 come from the same "
                                "frame table, use same-window endpoints/thresholds, and only "
                                "set window_sensitive when a conjunct or verdict differs"
                            ),
                        },
                        {
                            "finding": (
                                "reusing the frozen candidate alignment for ordinary anchors "
                                "would make the local endpoints circular"
                            ),
                            "resolution": (
                                "AE/AE and EH/EH are mandatory independent ordinary decodes "
                                "with their own durations, alignment, F0, noise, and side-specific intervals"
                            ),
                        },
                        {
                            "finding": "local anchors needed substantive contrast validity",
                            "resolution": (
                                "separate measurement and contrast layers require exact valid "
                                "fraction 1.0, plausibility, >=0.25 Bark, cosine with v4 >=0.50, "
                                "and cross-occurrence cosine >=0.50 at every ceiling"
                            ),
                        },
                        {
                            "finding": "the stronger state spans and period boundary were ambiguous",
                            "resolution": (
                                "freeze target word {4..7,17..20}, boundary {3..8,16..21} "
                                "including period column 21, and full state {0..22}"
                            ),
                        },
                        {
                            "finding": (
                                "analysis drift could masquerade as a local-endpoint result"
                            ),
                            "resolution": (
                                "reproduce every stored v1 primary point, frame/retention value, "
                                "classification conjunct, and occurrence/fixture/run verdict before deriving endpoints"
                            ),
                        },
                        {
                            "finding": "candidate completeness needed non-acoustic gates",
                            "resolution": (
                                "retain exact state/output integrity and >=0.80 positive-energy "
                                "localization; the new version fails zero-delta localization"
                            ),
                        },
                    ],
                },
                {
                    "role": "selection_integrity_outcome_branches_and_claims",
                    "independent_read_only": True,
                    "findings_and_resolutions": [
                        {
                            "finding": (
                                "a favorable local calibration is in-sample and cannot confirm root cause"
                            ),
                            "resolution": (
                                "freeze all four transported/local endpoint × transported/local "
                                "threshold cells and allow only mechanical sufficiency/implication labels"
                            ),
                        },
                        {
                            "finding": "anchor validity and branch precedence were underspecified",
                            "resolution": (
                                "measurement invalid stops first; otherwise medial/final validity "
                                "maps exactly to rescore, phrase-final-not-realized, unexpected-medial, "
                                "or reference-geometry-invalid"
                            ),
                        },
                        {
                            "finding": "conditional rendering could create retry or selection degrees of freedom",
                            "resolution": (
                                "freeze four one-attempt decoder slots with pre-decode markers; "
                                "target word is evaluated first, then boundary, then full, and first complete pass wins"
                            ),
                        },
                        {
                            "finding": (
                                "neutral, runtime/state, localization, and route-exhaustion stops "
                                "must remain distinguishable"
                            ),
                            "resolution": (
                                "only valid lens-repairable acoustic failure advances; neutral-source, "
                                "runtime/integrity, localization, and final-route exhaustion have separate labels"
                            ),
                        },
                        {
                            "finding": "unseen confirmation fixtures could be outcome-selected",
                            "resolution": (
                                "freeze the full planning-only consideration register, deterministic "
                                "selection rule, two exact fixtures/plans/columns, all-artifact novelty "
                                "scan, and the same fixture set for every selected span"
                            ),
                        },
                        {
                            "finding": (
                                "the frozen target-word candidate must not receive a broader-span label"
                            ),
                            "resolution": (
                                "a target-word pass reports transported calibration mechanical "
                                "sufficiency plus its 2x2 attribution; only boundary/full may report "
                                "branch_conditional_span_candidate_selected_for_unseen_confirmation"
                            ),
                        },
                    ],
                },
            ],
        },
        "scope": {
            "fixture": REPEATED_FIXTURE_ID,
            "novel_decoder_slots": 4,
            "mandatory_ordinary_anchor_slots": 2,
            "maximum_conditional_candidate_slots": 2,
            "api_calls": 0,
            "openai_calls": 0,
            "paid_calls": 0,
            "selection": "fixed order; first complete primary-50-percent pass wins",
        },
        "parents": parents,
        "implementation": {
            "source_file_sha256": {
                name: sha256_file(path) for name, path in sorted(code_paths.items())
            },
            "measurement": {
                "praat_path": str(PRAAT),
                "praat_sha256": sha256_file(PRAAT),
                "script_relative_path": str(MEASUREMENT_SCRIPT.relative_to(root)),
                "script_sha256": sha256_file(MEASUREMENT_SCRIPT),
                "bark_source_file": "src/earshift_bakeoff/same_take.py",
            },
            "engine_assets": local_engine_assets(),
            "renderer": {
                "package": "kokoro",
                "version": KOKORO_VERSION,
                "model_repo": MODEL_REPO,
                "model_revision": MODEL_REVISION,
                "model_hashes": MODEL_HASHES,
                "voice": "af_heart",
                "voice_style_row": STYLE_ROW,
                "sample_rate_hz": SAMPLE_RATE_HZ,
                "speed": SPEED,
                "device": "cpu",
                "rng_seed": RNG_SEED,
            },
            "runtime_requirement": (
                "the checked-in protocol must equal protocol_record(), so every bound "
                "code, instrument, model, voice, gate, parser, and parent WAV hash must match"
            ),
        },
        "decoder_slots": [asdict(slot) for slot in DECODER_SLOTS],
        "anchor_contract": {
            "carrier_skeleton": "zɪ ʒˈ{V}ʒ ɡˈOh zɪ ʒˈ{V}ʒ.",
            "ae_phonemes": REPEATED_NEUTRAL_PHONEMES,
            "eh_phonemes": REPEATED_LENS_PHONEMES,
            "same": [
                "voice",
                "style row 20",
                "speed",
                "CPU device",
                "RNG seed",
                "token skeleton",
            ],
            "independent": [
                "text feature pass",
                "predicted durations",
                "alignment",
                "F0 state",
                "noise state",
                "ordinary decoder call",
            ],
            "not_required": "anchor sample counts or duration sequences need not be equal",
            "occurrence_measurement_columns": [
                list(row) for row in REPEATED_MEASUREMENT_COLUMNS
            ],
            "measurement_intervals": "derived separately from each anchor's own predicted durations and alignment",
        },
        "measurement_protocol": {
            "ceiling_hz_family": list(CEILINGS_HZ),
            "single_frame_table_per_audio_interval_and_ceiling": True,
            "primary_window_percent": PRIMARY_WINDOW_PERCENT,
            "descriptive_sensitivity_window_percents": list(
                DESCRIPTIVE_WINDOW_PERCENTS
            ),
            "window_definition": "centered 40, 50, or 60 percent of the same complete stress-plus-vowel frame table",
            "primary_branch_only": True,
            "descriptive_rule": (
                "40/60 use their same-window anchor endpoints and thresholds; label "
                "window_sensitive if any conjunct or verdict differs from 50; never "
                "advance, select, rescue, or veto a candidate"
            ),
            "v1_reproduction_stop": (
                "before deriving any new endpoint, reproduce every stored replication-v1 "
                "middle-50 F1/F2 point, retained/frame count, plausibility, family verdict, "
                "occurrence verdict, and fixture verdict; any drift stops as inconclusive"
            ),
        },
        "anchor_validity": {
            "measurement_valid": (
                "at primary 50, every side/occurrence/ceiling has at least 5 frames, "
                "at least 5 F1/F2 pairs, valid fraction exactly 1.0, and plausible F1/F2"
            ),
            "contrast_valid_per_occurrence": (
                "at every ceiling, EH-minus-AE magnitude >=0.25 Bark and cosine with "
                "the frozen v4 expected vector >=0.50"
            ),
            "cross_occurrence_valid": (
                "at every ceiling, medial and phrase-final local anchor vectors have cosine >=0.50"
            ),
            "precedence": [
                "any measurement or instrument invalid => anchor_measurement_inconclusive; no rescore or candidate render",
                "medial valid and phrase-final valid => rescore",
                "medial valid and phrase-final substantively invalid => phrase_final_reference_not_realized; no candidate render",
                "medial invalid and phrase-final valid => unexpected_medial_reference_failure; no candidate render",
                "both substantively invalid => reference_geometry_invalid; no candidate render",
            ],
        },
        "frozen_wav_rescore": {
            "fixture": REPEATED_FIXTURE_ID,
            "wav_source": "immutable replication-v1 neutral and target-word lens",
            "primary_window_percent": 50,
            "cells": [
                {"endpoint_geometry": endpoint, "magnitude_threshold": threshold}
                for endpoint in ("transported_v4", "local_occurrence_specific")
                for threshold in ("transported_v4", "local_occurrence_specific")
            ],
            "all_conjuncts_recorded": [
                "measurement validity",
                "neutral plausibility and nearer AE endpoint",
                "lens plausibility and nearer EH endpoint",
                "vector cosine >=0.50",
                "magnitude >= selected max(0.25, half endpoint distance) threshold",
            ],
            "interpretation": (
                "a local/local pass can show only that transported calibration is "
                "mechanically sufficient to account for this frozen fixture gate; the "
                "one-axis cells label endpoint, threshold, both, or mixed implication"
            ),
        },
        "candidate_route": {
            "fixed_order": [
                {
                    "span_id": "target-word",
                    "source": "frozen replication-v1 WAV",
                    "columns": list(REPEATED_TARGET_WORD_COLUMNS),
                },
                {
                    "span_id": "target-word-plus-boundaries",
                    "source": "conditional decoder slot 3",
                    "columns": list(REPEATED_TARGET_WORD_PLUS_BOUNDARIES_COLUMNS),
                },
                {
                    "span_id": "full-contextual-state",
                    "source": "conditional decoder slot 4",
                    "columns": list(REPEATED_FULL_CONTEXTUAL_STATE_COLUMNS),
                },
            ],
            "period_column": 21,
            "primary_local_gate": (
                "for every occurrence and ceiling, neutral is nearer its local AE endpoint, "
                "lens is nearer its local EH endpoint, vector cosine >=0.50, magnitude >= "
                "max(0.25 Bark, half the local endpoint distance), and measurements are plausible"
            ),
            "complete_pass": [
                "all occurrences and ceilings pass the primary 50-percent local gate",
                "output is finite, nonempty, below 0.001 clipped fraction, and sample-count matched",
                "exact source-derived duration/common-F0/common-noise/state-replacement contract passes",
                "at least 0.80 of squared PCM difference energy is in padded target-word windows",
            ],
            "advance": (
                "only a measurement-valid substantive lens-category/direction/magnitude "
                "failure with every neutral source check and non-acoustic gate passing advances"
            ),
            "stop": (
                "any instrument/measurement invalidity, neutral measurement/source failure, "
                "runtime/integrity/state/localization failure, or exhausted final span stops"
            ),
            "first_complete_pass_wins": True,
            "no_retry": "each of the four decoder slots has one attempt; later slots remain not_reached after stopping",
        },
        "confirmation_precommit": {
            "status": "fixtures_frozen_now_but_no_confirmation_audio_authorized_until_diagnostic_eligibility",
            "run_bound": "at most one fresh confirmatory replication using this entire two-fixture set",
            "eligibility": "complete local target-word pass or first broader-span complete pass",
            "fixtures": confirmation,
            "planning_only_consideration_register": list(
                CONFIRMATION_PLANNING_REGISTER
            ),
            "selection_rule": (
                "choose the first exact repeated same lexical target with a phrase-final "
                "occurrence, then the first eligible single-target phrase-final fixture; "
                "no acoustic outcome existed for any considered text"
            ),
            "same_set_for_every_selected_span": True,
            "novelty_check": (
                "exact chosen source texts and plan hashes absent from every prior "
                "artifacts/**/*.json value at protocol creation, excluding only this run; "
                "this covers every prior audio-bearing JSON manifest"
            ),
            "pre_write_repo_scan": (
                "a direct whole-repository text/hash scan before these constants were "
                "written found no chosen text or plan-hash match"
            ),
            "disjoint_audio": "no confirmation audio exists; future audio hashes must not equal any bound parent or diagnostic WAV hash",
            "separate_freeze": (
                "derive and commit a separate confirmation protocol naming the mechanically "
                "selected span before decoding either fixture"
            ),
        },
        "stopping_rule": (
            "Preserve the frozen replication-v1 failure. Decode the two independent anchors "
            "once. Only after reproduction and anchor precedence permit rescoring, evaluate "
            "the frozen target-word WAV, then at most the two broader one-attempt slots in "
            "fixed order. First complete primary-50 pass wins; 40/60 are descriptive only. "
            "Never rerender, replace, tune, or select by listening."
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
                "existing diagnostic protocol differs from the code-bound freeze"
            )
    else:
        atomic_write_json(destination, protocol)
    return protocol
