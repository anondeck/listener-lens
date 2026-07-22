#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
from typing import Any
import wave

import numpy as np

from earshift_bakeoff.bilingual_listener_engine import load_listener_profiles
from earshift_bakeoff.bilingual_vowel_engine import _load_pinned_synthesis_voice
from earshift_bakeoff.bilingual_vowel_replicated_anchors import (
    MAXIMUM_REVERSED_SEED_PAIRS_PER_OCCURRENCE,
    MINIMUM_EXACT_SEED_PAIRS_PER_OCCURRENCE,
    REPLICATED_ANCHOR_VERSION,
    TRAINING_SEEDS,
    aggregate_replicated_anchor_cell,
    aggregate_replicated_anchor_occurrence,
    render_seeded_natural_conditions,
)
from earshift_bakeoff.bilingual_vowel_spectral_category import (
    DEFAULT_FEATURE_CONFIG,
    SPECTRAL_CATEGORY_VERSION,
    apply_robust_feature_scaler,
    classify_spectral_endpoint,
    fit_robust_feature_scaler,
    spectral_trajectory_feature,
)
from earshift_bakeoff.config import Paths, stable_json
from earshift_bakeoff.kokoro_gate_bridge import KokoroGateIndex
from earshift_bakeoff.kokoro_synthesis import (
    CONFIG_FILE,
    RNG_SEED,
    pcm16_bytes,
    verify_model_files,
)
from earshift_bakeoff.listener_lens import DatabaseNonceChecker
from earshift_bakeoff.portuguese_carrier_planner_v1 import (
    PortuguesePositiveOnlyIndexV1,
)
from earshift_bakeoff.util import atomic_write_json, sha256_file

from run_bilingual_product_v8_vowel_acoustic_screen import _planner_v8


PROTOCOL_VERSION = "bilingual-vowel-replicated-anchor-calibration-v1"
PROTOCOL_PATH = Paths().root / "rules" / f"{PROTOCOL_VERSION}.json"
RUN_ID = "20260717-bilingual-vowel-replicated-anchor-calibration-v1"
RUN_DIR = Paths().artifacts / "product-matrix" / RUN_ID
V8_MANIFEST_PATH = (
    Paths().artifacts
    / "product-matrix"
    / "20260717-bilingual-product-v8-vowel-manifest"
    / "manifest.json"
)
V8_RESULT_PATH = (
    Paths().artifacts
    / "product-matrix"
    / "20260717-bilingual-product-v8-vowel-acoustic-screen"
    / "results.json"
)
V8_DIR = V8_RESULT_PATH.parent
V1_RESULT_PATH = (
    Paths().artifacts
    / "product-matrix"
    / "20260717-bilingual-product-vowel-acoustic-screen-v1"
    / "results.json"
)
V1_DIR = V1_RESULT_PATH.parent
WORD_RESULT_PATH = (
    Paths().artifacts
    / "product-matrix"
    / "20260717-bilingual-vowel-word-context-screen-v1"
    / "results.json"
)
WORD_DIR = WORD_RESULT_PATH.parent
FULL_RESULT_PATH = (
    Paths().artifacts
    / "product-matrix"
    / "20260717-bilingual-vowel-full-context-screen-v1"
    / "results.json"
)
FULL_DIR = FULL_RESULT_PATH.parent
ADAPTIVE_RESULT_PATH = (
    Paths().artifacts
    / "product-matrix"
    / "20260717-bilingual-vowel-adaptive-strength-screen-v1"
    / "results.json"
)
ADAPTIVE_DIR = ADAPTIVE_RESULT_PATH.parent
REACHABILITY_RESULT_PATH = (
    Paths().artifacts
    / "product-matrix"
    / "20260717-bilingual-g2p-reachability-v1"
    / "results.json"
)
VOICE_ORDER = ("af_heart", "am_michael", "pm_alex", "pf_dora")
EXPECTED_CORE_CELL_COUNT = 36
EXPECTED_SLOT_COUNT = 108
EXPECTED_OCCURRENCE_COUNT = 144
EXPECTED_REFERENCE_CELL_COUNT = 12
MINIMUM_REFERENCE_CONCORDANT_CELL_COUNT = 10


def _semantic_hash(payload: dict[str, Any]) -> str:
    semantic = dict(payload)
    semantic.pop("record_sha256", None)
    return hashlib.sha256(stable_json(semantic).encode("utf-8")).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _feature_hash(feature: list[float] | np.ndarray) -> str:
    return hashlib.sha256(np.asarray(feature, dtype="<f8").tobytes()).hexdigest()


def _pcm_hash(pcm: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(pcm, dtype="<i2").tobytes()).hexdigest()


def _natural_pcm(audio: np.ndarray) -> np.ndarray:
    return np.frombuffer(pcm16_bytes(audio), dtype="<i2").copy()


def _read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as handle:
        if (
            handle.getnchannels() != 1
            or handle.getsampwidth() != 2
            or handle.getframerate() != DEFAULT_FEATURE_CONFIG.sample_rate_hz
        ):
            raise RuntimeError(f"unexpected WAV format: {path}")
        return np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2").copy()


def _load_protocol(parents: dict[str, dict[str, Any]]) -> dict[str, Any]:
    protocol = _load_json(PROTOCOL_PATH)
    expected_keys = {
        "schema_version",
        "protocol_version",
        "status",
        "production_enabled",
        "purpose",
        "parent_bindings",
        "scope",
        "replicated_anchor_rendering",
        "context_matched_validation",
        "reference_candidate_ladder",
        "global_instrument_sanity",
        "claim_limits",
        "stopping_rule",
        "source_bindings",
    }
    if set(protocol) != expected_keys:
        raise RuntimeError("replicated-anchor protocol schema drifted")
    bindings = protocol["parent_bindings"]
    if bindings["v8_manifest_sha256"] != sha256_file(V8_MANIFEST_PATH):
        raise RuntimeError("replicated-anchor manifest drifted")
    for label, (path, result) in parents.items():
        if (
            bindings[f"{label}_result_sha256"] != sha256_file(path)
            or bindings[f"{label}_record_sha256"] != result["record_sha256"]
        ):
            raise RuntimeError(f"replicated-anchor parent drifted: {label}")
    scope = protocol["scope"]
    rendering = protocol["replicated_anchor_rendering"]
    validation = protocol["context_matched_validation"]
    sanity = protocol["global_instrument_sanity"]
    if (
        protocol["schema_version"] != 1
        or protocol["protocol_version"] != PROTOCOL_VERSION
        or protocol["status"] != "frozen_before_first_context_matched_anchor_render"
        or protocol["production_enabled"] is not False
        or scope["typed_core_cell_count"] != EXPECTED_CORE_CELL_COUNT
        or scope["logical_slot_count"] != EXPECTED_SLOT_COUNT
        or scope["occurrence_count"] != EXPECTED_OCCURRENCE_COUNT
        or tuple(scope["voices_in_order"]) != VOICE_ORDER
        or tuple(rendering["training_seeds_in_order"]) != TRAINING_SEEDS
        or rendering["baseline_parity_seed"] != RNG_SEED
        or rendering["spectral_feature_version"] != SPECTRAL_CATEGORY_VERSION
        or rendering["spectral_feature_config"]
        != json.loads(stable_json(asdict(DEFAULT_FEATURE_CONFIG)))
        or validation["minimum_exact_seed_pairs_per_occurrence"]
        != MINIMUM_EXACT_SEED_PAIRS_PER_OCCURRENCE
        or validation["maximum_reversed_seed_pairs_per_occurrence"]
        != MAXIMUM_REVERSED_SEED_PAIRS_PER_OCCURRENCE
        or sanity["reference_candidate_cell_count"] != EXPECTED_REFERENCE_CELL_COUNT
        or sanity["minimum_reference_concordant_cell_count"]
        != MINIMUM_REFERENCE_CONCORDANT_CELL_COUNT
        or protocol["stopping_rule"]["api_calls_allowed"] != 0
        or protocol["stopping_rule"]["failure_cell_evaluation_allowed"] is not False
    ):
        raise RuntimeError("replicated-anchor protocol binding drifted")
    for binding in protocol["source_bindings"]:
        if sha256_file(Paths().root / binding["path"]) != binding["sha256"]:
            raise RuntimeError(f"replicated-anchor source drifted: {binding['path']}")
    return protocol


def _current_reference_ladder(
    *,
    core_ids: set[str],
    v8: dict[str, Any],
    word: dict[str, Any],
    full: dict[str, Any],
    adaptive: dict[str, Any],
) -> dict[str, str]:
    ladder: dict[str, str] = {
        row["cell_id"]: "v8"
        for row in v8["cell_summaries"]
        if row["cell_id"] in core_ids and row["classification"] != "fail"
    }
    for label, result in (
        ("word_context", word),
        ("full_context", full),
        ("adaptive_strength", adaptive),
    ):
        for row in result["cell_summaries"]:
            if row["cell_id"] in core_ids and row["candidate_classification"] != "fail":
                ladder[row["cell_id"]] = label
    return ladder


def _candidate_audio(
    *,
    rung: str,
    logical_slot_id: str,
    v8_outcome: dict[str, Any],
    outcomes_by_rung: dict[str, dict[str, dict[str, Any]]],
) -> tuple[Path, dict[str, Any], Path, dict[str, Any]]:
    neutral_record = v8_outcome["audio"]["neutral"]
    neutral_path = V8_DIR / neutral_record["relative_path"]
    if rung == "v8":
        lens_record = v8_outcome["audio"]["lens"]
        lens_path = V8_DIR / lens_record["relative_path"]
    elif rung == "word_context":
        row = outcomes_by_rung[rung][logical_slot_id]
        lens_record = row["audio"]["word_context_lens"]
        lens_path = WORD_DIR / lens_record["relative_path"]
    elif rung == "full_context":
        row = outcomes_by_rung[rung][logical_slot_id]
        lens_record = row["audio"]["full_context_lens"]
        lens_path = FULL_DIR / lens_record["relative_path"]
    elif rung == "adaptive_strength":
        row = outcomes_by_rung[rung][logical_slot_id]
        lens_record = row["adaptive_audio"]
        if lens_record is None:
            raise RuntimeError(f"adaptive reference has no audio: {logical_slot_id}")
        lens_path = ADAPTIVE_DIR / lens_record["relative_path"]
    else:  # pragma: no cover - protocol and ladder constrain this
        raise RuntimeError(f"unsupported evidence rung: {rung}")
    for path, record in ((neutral_path, neutral_record), (lens_path, lens_record)):
        if sha256_file(path) != record["wav_sha256"]:
            raise RuntimeError(f"reference candidate WAV drifted: {path}")
    return neutral_path, neutral_record, lens_path, lens_record


def _feature_receipt(
    pcm: np.ndarray, interval: dict[str, Any]
) -> tuple[list[float], dict[str, Any]]:
    extracted = spectral_trajectory_feature(
        pcm,
        start_sample=int(interval["start_sample"]),
        end_sample_exclusive=int(interval["end_sample_exclusive"]),
    )
    feature = extracted.pop("feature")
    return feature, {
        "feature_sha256": _feature_hash(feature),
        "duration_ms": extracted["duration_ms"],
        "frame_count": extracted["frame_count"],
        "feature_size": extracted["feature_size"],
    }


def _pcm_receipt(pcm: np.ndarray) -> dict[str, Any]:
    values = np.asarray(pcm, dtype="<i2")
    return {
        "pcm_sha256": _pcm_hash(values),
        "sample_count": int(values.size),
        "peak_abs_pcm16": int(np.max(np.abs(values.astype(np.int32)))),
        "finite_nonempty_unclipped": bool(
            values.size > 0
            and np.isfinite(values.astype(np.float64)).all()
            and np.max(np.abs(values.astype(np.int32))) < 32767
        ),
    }


def _evaluate_observation(
    observation: dict[str, Any], observations: list[dict[str, Any]]
) -> dict[str, Any]:
    slot_id = observation["logical_slot_id"]
    voice_pool = [
        row
        for row in observations
        if row["voice_id"] == observation["voice_id"]
        and row["logical_slot_id"] != slot_id
    ]
    scaler_features = [
        feature
        for row in voice_pool
        for side in ("source_features", "target_features")
        for feature in row[side].values()
    ]
    scaler = fit_robust_feature_scaler(scaler_features)

    def scaled(feature: list[float]) -> np.ndarray:
        return apply_robust_feature_scaler(feature, scaler)

    natural_records = []
    for heldout_seed in TRAINING_SEEDS:
        other_seeds = tuple(seed for seed in TRAINING_SEEDS if seed != heldout_seed)
        source_centroid = np.mean(
            np.stack(
                [scaled(observation["source_features"][seed]) for seed in other_seeds]
            ),
            axis=0,
        )
        target_centroid = np.mean(
            np.stack(
                [scaled(observation["target_features"][seed]) for seed in other_seeds]
            ),
            axis=0,
        )
        record = classify_spectral_endpoint(
            source_anchor=source_centroid,
            target_anchor=target_centroid,
            neutral=scaled(observation["source_features"][heldout_seed]),
            lens=scaled(observation["target_features"][heldout_seed]),
        )
        natural_records.append({"heldout_seed": heldout_seed, **record})

    candidate_record = None
    identity_record = None
    if observation["candidate_neutral_feature"] is not None:
        source_centroid = np.mean(
            np.stack(
                [
                    scaled(observation["source_features"][seed])
                    for seed in TRAINING_SEEDS
                ]
            ),
            axis=0,
        )
        target_centroid = np.mean(
            np.stack(
                [
                    scaled(observation["target_features"][seed])
                    for seed in TRAINING_SEEDS
                ]
            ),
            axis=0,
        )
        neutral = scaled(observation["candidate_neutral_feature"])
        lens = scaled(observation["candidate_lens_feature"])
        candidate_record = classify_spectral_endpoint(
            source_anchor=source_centroid,
            target_anchor=target_centroid,
            neutral=neutral,
            lens=lens,
        )
        identity_record = classify_spectral_endpoint(
            source_anchor=source_centroid,
            target_anchor=target_centroid,
            neutral=neutral,
            lens=neutral,
        )
    aggregate = aggregate_replicated_anchor_occurrence(
        natural_seed_records=natural_records,
        candidate_record=candidate_record,
    )
    return {
        "cell_id": observation["cell_id"],
        "logical_slot_id": slot_id,
        "context": observation["context"],
        "voice_id": observation["voice_id"],
        "profile_id": observation["profile_id"],
        "rule_id": observation["rule_id"],
        "source": observation["source"],
        "target": observation["target"],
        "occurrence_index": observation["occurrence_index"],
        "reference_rung": observation["reference_rung"],
        "scaler_pool_natural_anchor_count": len(scaler_features),
        "natural_seed_classifications": natural_records,
        "candidate_classification": candidate_record,
        "identity_negative_control": identity_record,
        "aggregate": aggregate,
        "feature_receipts": observation["feature_receipts"],
    }


def main() -> int:
    if RUN_DIR.exists():
        raise RuntimeError(f"refusing to overwrite replicated-anchor run: {RUN_DIR}")
    manifest = _load_json(V8_MANIFEST_PATH)
    v8 = _load_json(V8_RESULT_PATH)
    v1 = _load_json(V1_RESULT_PATH)
    word = _load_json(WORD_RESULT_PATH)
    full = _load_json(FULL_RESULT_PATH)
    adaptive = _load_json(ADAPTIVE_RESULT_PATH)
    reachability = _load_json(REACHABILITY_RESULT_PATH)
    parents = {
        "v8": (V8_RESULT_PATH, v8),
        "v1": (V1_RESULT_PATH, v1),
        "word": (WORD_RESULT_PATH, word),
        "full": (FULL_RESULT_PATH, full),
        "adaptive": (ADAPTIVE_RESULT_PATH, adaptive),
        "reachability": (REACHABILITY_RESULT_PATH, reachability),
    }
    protocol = _load_protocol(parents)
    core_ids = set(protocol["scope"]["typed_core_cell_ids_in_order"])
    reference_ladder = _current_reference_ladder(
        core_ids=core_ids, v8=v8, word=word, full=full, adaptive=adaptive
    )
    expected_ladder = {
        row["cell_id"]: row["evidence_rung"]
        for row in protocol["reference_candidate_ladder"]["cells_in_order"]
    }
    if reference_ladder != expected_ladder:
        raise RuntimeError("current reference evidence ladder drifted")

    slots = [row for row in manifest["slots"] if row["cell_id"] in core_ids]
    if len(slots) != EXPECTED_SLOT_COUNT:
        raise RuntimeError("typed-core replicated-anchor slot count drifted")
    v8_by_id = {row["logical_slot_id"]: row for row in v8["outcomes"]}
    v1_by_id = {row["logical_slot_id"]: row for row in v1["outcomes"]}
    outcomes_by_rung = {
        "word_context": {row["logical_slot_id"]: row for row in word["outcomes"]},
        "full_context": {row["logical_slot_id"]: row for row in full["outcomes"]},
        "adaptive_strength": {
            row["logical_slot_id"]: row for row in adaptive["outcomes"]
        },
    }
    profiles = load_listener_profiles()
    files = verify_model_files(download=False)
    model_vocab = set(
        json.loads(files[CONFIG_FILE].read_text(encoding="utf-8"))["vocab"]
    )
    nonce_checker = DatabaseNonceChecker()
    phone_indexes = (KokoroGateIndex(), PortuguesePositiveOnlyIndexV1())
    observations: list[dict[str, Any]] = []
    slot_receipts: list[dict[str, Any]] = []
    started = time.perf_counter()
    for voice_id in VOICE_ORDER:
        synthesis = _load_pinned_synthesis_voice(voice_id)
        for slot in sorted(
            (row for row in slots if row["voice_id"] == voice_id),
            key=lambda row: row["logical_slot_id"],
        ):
            planner = _planner_v8(
                slot=slot,
                profiles=profiles,
                model_vocab=model_vocab,
                nonce_checker=nonce_checker,
                phone_indexes=phone_indexes,
            )
            plan = planner.plan(slot["fixture_spec"]["text"])
            if plan.plan_sha256 != slot["v8_plan_sha256"]:
                raise RuntimeError(f"v8 plan drifted: {slot['logical_slot_id']}")
            seeds = (RNG_SEED, *TRAINING_SEEDS)
            source_render = render_seeded_natural_conditions(
                synthesis,
                phonemes=plan.neutral_phonemes,
                reference_phonemes=plan.render_reference_phonemes,
                seeds=seeds,
            )
            target_render = render_seeded_natural_conditions(
                synthesis,
                phonemes=plan.lens_phonemes,
                reference_phonemes=plan.render_reference_phonemes,
                seeds=seeds,
            )
            source_pcm = {
                seed: _natural_pcm(source_render.audio_by_seed[seed]) for seed in seeds
            }
            target_pcm = {
                seed: _natural_pcm(target_render.audio_by_seed[seed]) for seed in seeds
            }
            v8_outcome = v8_by_id[slot["logical_slot_id"]]
            v1_outcome = v1_by_id[slot["logical_slot_id"]]
            source_record = v1_outcome["anchor_audio"]["source"]
            target_record = v1_outcome["anchor_audio"]["target"]
            source_path = V1_DIR / source_record["relative_path"]
            target_path = V1_DIR / target_record["relative_path"]
            baseline_source_pass = bool(
                _pcm_hash(source_pcm[RNG_SEED]) == source_record["pcm_sha256"]
                and sha256_file(source_path) == source_record["wav_sha256"]
            )
            baseline_target_pass = bool(
                _pcm_hash(target_pcm[RNG_SEED]) == target_record["pcm_sha256"]
                and sha256_file(target_path) == target_record["wav_sha256"]
            )
            if not baseline_source_pass or not baseline_target_pass:
                raise RuntimeError(
                    f"seeded natural baseline parity failed: {slot['logical_slot_id']}"
                )
            source_intervals = v8_outcome["source_anchor_intervals"]
            target_intervals = v8_outcome["target_anchor_intervals"]
            controlled_rows = v8_outcome["occurrence_outcomes"]
            if (
                not len(source_intervals)
                == len(target_intervals)
                == len(controlled_rows)
            ):
                raise RuntimeError(f"interval count drifted: {slot['logical_slot_id']}")
            rung = reference_ladder.get(slot["cell_id"])
            candidate_neutral_pcm = candidate_lens_pcm = None
            candidate_receipt = None
            if rung is not None:
                neutral_path, neutral_record, lens_path, lens_record = _candidate_audio(
                    rung=rung,
                    logical_slot_id=slot["logical_slot_id"],
                    v8_outcome=v8_outcome,
                    outcomes_by_rung=outcomes_by_rung,
                )
                candidate_neutral_pcm = _read_wav(neutral_path)
                candidate_lens_pcm = _read_wav(lens_path)
                candidate_receipt = {
                    "rung": rung,
                    "neutral_wav_sha256": neutral_record["wav_sha256"],
                    "lens_wav_sha256": lens_record["wav_sha256"],
                }
            slot_receipts.append(
                {
                    "logical_slot_id": slot["logical_slot_id"],
                    "cell_id": slot["cell_id"],
                    "voice_id": voice_id,
                    "reference_rung": rung,
                    "baseline_source_parity_pass": baseline_source_pass,
                    "baseline_target_parity_pass": baseline_target_pass,
                    "candidate_audio": candidate_receipt,
                    "source_seed_audio": {
                        str(seed): _pcm_receipt(source_pcm[seed]) for seed in seeds
                    },
                    "target_seed_audio": {
                        str(seed): _pcm_receipt(target_pcm[seed]) for seed in seeds
                    },
                }
            )
            for index, controlled in enumerate(controlled_rows):
                source_features: dict[int, list[float]] = {}
                target_features: dict[int, list[float]] = {}
                source_receipts: dict[str, Any] = {}
                target_receipts: dict[str, Any] = {}
                for seed in TRAINING_SEEDS:
                    feature, receipt = _feature_receipt(
                        source_pcm[seed], source_intervals[index]
                    )
                    source_features[seed] = feature
                    source_receipts[str(seed)] = receipt
                    feature, receipt = _feature_receipt(
                        target_pcm[seed], target_intervals[index]
                    )
                    target_features[seed] = feature
                    target_receipts[str(seed)] = receipt
                neutral_feature = lens_feature = None
                neutral_receipt = lens_receipt = None
                if candidate_neutral_pcm is not None:
                    neutral_feature, neutral_receipt = _feature_receipt(
                        candidate_neutral_pcm, controlled["measurement_interval"]
                    )
                    lens_feature, lens_receipt = _feature_receipt(
                        candidate_lens_pcm, controlled["measurement_interval"]
                    )
                observations.append(
                    {
                        "cell_id": slot["cell_id"],
                        "logical_slot_id": slot["logical_slot_id"],
                        "context": slot["context"],
                        "voice_id": voice_id,
                        "profile_id": slot["profile_id"],
                        "rule_id": slot["rule_id"],
                        "source": slot["source"],
                        "target": slot["target"],
                        "occurrence_index": controlled["occurrence_index"],
                        "reference_rung": rung,
                        "source_features": source_features,
                        "target_features": target_features,
                        "candidate_neutral_feature": neutral_feature,
                        "candidate_lens_feature": lens_feature,
                        "feature_receipts": {
                            "natural_source_by_seed": source_receipts,
                            "natural_target_by_seed": target_receipts,
                            "candidate_neutral": neutral_receipt,
                            "candidate_lens": lens_receipt,
                        },
                    }
                )
    if len(observations) != EXPECTED_OCCURRENCE_COUNT:
        raise RuntimeError("replicated-anchor occurrence count drifted")
    evaluated = [_evaluate_observation(row, observations) for row in observations]
    cell_summaries = []
    for cell_id in sorted(core_ids):
        rows = [row for row in evaluated if row["cell_id"] == cell_id]
        aggregate = aggregate_replicated_anchor_cell([row["aggregate"] for row in rows])
        cell_summaries.append(
            {
                "cell_id": cell_id,
                "profile_id": rows[0]["profile_id"],
                "voice_id": rows[0]["voice_id"],
                "rule_id": rows[0]["rule_id"],
                "source": rows[0]["source"],
                "target": rows[0]["target"],
                "reference_rung": rows[0]["reference_rung"],
                "replicated_anchor": aggregate,
                "product_enabled": False,
            }
        )
    reference = [row for row in cell_summaries if row["reference_rung"] is not None]
    concordant = [
        row for row in reference if row["replicated_anchor"]["directional_pass"]
    ]
    identity_controls = [
        row["identity_negative_control"]
        for row in evaluated
        if row["identity_negative_control"] is not None
    ]
    identity_false_positives = [
        row for row in identity_controls if row["directional_pass"]
    ]
    all_seed_audio = [
        receipt
        for slot in slot_receipts
        for side in ("source_seed_audio", "target_seed_audio")
        for receipt in slot[side].values()
    ]
    baseline_parity_pass = bool(
        all(
            row["baseline_source_parity_pass"] and row["baseline_target_parity_pass"]
            for row in slot_receipts
        )
    )
    instrument_pass = bool(
        baseline_parity_pass
        and all(row["finite_nonempty_unclipped"] for row in all_seed_audio)
        and len(reference) == EXPECTED_REFERENCE_CELL_COUNT
        and len(concordant) >= MINIMUM_REFERENCE_CONCORDANT_CELL_COUNT
        and not identity_false_positives
    )
    result: dict[str, Any] = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "protocol_version": PROTOCOL_VERSION,
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "replicated_anchor_version": REPLICATED_ANCHOR_VERSION,
        "spectral_category_version": SPECTRAL_CATEGORY_VERSION,
        "classification": (
            "context_matched_anchor_instrument_pass_no_failure_cells_evaluated"
            if instrument_pass
            else "context_matched_anchor_instrument_fail_no_failure_cells_evaluated"
        ),
        "production_enabled": False,
        "api_calls_made": 0,
        "new_local_decoder_render_count": len(slot_receipts)
        * 2
        * (len(TRAINING_SEEDS) + 1),
        "retained_new_wav_count": 0,
        "typed_core_cell_count": len(cell_summaries),
        "logical_slot_count": len(slot_receipts),
        "occurrence_count": len(evaluated),
        "baseline_parity_condition_count": len(slot_receipts) * 2,
        "baseline_parity_pass": baseline_parity_pass,
        "training_seed_condition_count": len(slot_receipts) * 2 * len(TRAINING_SEEDS),
        "training_seed_integrity_pass_count": sum(
            row["finite_nonempty_unclipped"] for row in all_seed_audio
        ),
        "reference_candidate_cell_count": len(reference),
        "reference_concordant_cell_count": len(concordant),
        "minimum_reference_concordant_cell_count": (
            MINIMUM_REFERENCE_CONCORDANT_CELL_COUNT
        ),
        "identity_negative_control_count": len(identity_controls),
        "identity_negative_control_false_positive_count": len(identity_false_positives),
        "instrument_pass": instrument_pass,
        "anchor_valid_core_cell_count": sum(
            row["replicated_anchor"]["all_anchor_occurrences_valid"]
            for row in cell_summaries
        ),
        "reference_concordant_cell_ids": [row["cell_id"] for row in concordant],
        "nonconcordant_reference_cell_ids": [
            row["cell_id"] for row in reference if row not in concordant
        ],
        "parent_bindings": protocol["parent_bindings"],
        "slot_receipts": slot_receipts,
        "cell_summaries": cell_summaries,
        "outcomes": evaluated,
        "claim_limits": protocol["claim_limits"],
        "elapsed_s": time.perf_counter() - started,
    }
    result["record_sha256"] = _semantic_hash(result)
    RUN_DIR.mkdir(parents=True, exist_ok=False)
    atomic_write_json(RUN_DIR / "results.json", result)
    print(
        stable_json(
            {
                key: result[key]
                for key in (
                    "classification",
                    "new_local_decoder_render_count",
                    "typed_core_cell_count",
                    "logical_slot_count",
                    "occurrence_count",
                    "baseline_parity_pass",
                    "training_seed_integrity_pass_count",
                    "reference_candidate_cell_count",
                    "reference_concordant_cell_count",
                    "minimum_reference_concordant_cell_count",
                    "identity_negative_control_false_positive_count",
                    "anchor_valid_core_cell_count",
                    "instrument_pass",
                    "elapsed_s",
                    "record_sha256",
                )
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
