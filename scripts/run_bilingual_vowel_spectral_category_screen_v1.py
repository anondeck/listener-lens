#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Iterable
import wave

import numpy as np

from earshift_bakeoff.bilingual_vowel_spectral_category import (
    DEFAULT_FEATURE_CONFIG,
    MAXIMUM_REVERSED_HELDOUT_ANCHOR_PAIRS,
    MINIMUM_HELDOUT_EXACT_ANCHOR_PAIRS,
    SPECTRAL_CATEGORY_VERSION,
    aggregate_spectral_cell,
    apply_robust_feature_scaler,
    classify_spectral_endpoint,
    fit_robust_feature_scaler,
    spectral_trajectory_feature,
)
from earshift_bakeoff.config import Paths, stable_json
from earshift_bakeoff.util import atomic_write_json, sha256_file


PROTOCOL_VERSION = "bilingual-vowel-spectral-category-screen-v1"
PROTOCOL_PATH = Paths().root / "rules" / f"{PROTOCOL_VERSION}.json"
RUN_ID = "20260717-bilingual-vowel-spectral-category-screen-v1"
RUN_DIR = Paths().artifacts / "product-matrix" / RUN_ID
V8_RESULT_PATH = (
    Paths().artifacts
    / "product-matrix"
    / "20260717-bilingual-product-v8-vowel-acoustic-screen"
    / "results.json"
)
V8_AUDIO_DIR = V8_RESULT_PATH.parent
V1_RESULT_PATH = (
    Paths().artifacts
    / "product-matrix"
    / "20260717-bilingual-product-vowel-acoustic-screen-v1"
    / "results.json"
)
V1_AUDIO_DIR = V1_RESULT_PATH.parent
REACHABILITY_RESULT_PATH = (
    Paths().artifacts
    / "product-matrix"
    / "20260717-bilingual-g2p-reachability-v1"
    / "results.json"
)
REFERENCE_PASS_MINIMUM = 16
EXPECTED_REFERENCE_PASS_COUNT = 21
EXPECTED_CELL_COUNT = 80
EXPECTED_OCCURRENCE_COUNT = 320


def _semantic_hash(payload: Any) -> str:
    semantic = dict(payload)
    semantic.pop("record_sha256", None)
    return hashlib.sha256(stable_json(semantic).encode("utf-8")).hexdigest()


def _feature_hash(feature: Iterable[float]) -> str:
    values = np.asarray(tuple(feature), dtype="<f8")
    return hashlib.sha256(values.tobytes()).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as handle:
        if (
            handle.getnchannels() != 1
            or handle.getsampwidth() != 2
            or handle.getframerate() != DEFAULT_FEATURE_CONFIG.sample_rate_hz
        ):
            raise RuntimeError(f"unexpected WAV format: {path}")
        return np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2").copy()


def _load_protocol(
    v8_result: dict[str, Any],
    v1_result: dict[str, Any],
    reachability_result: dict[str, Any],
) -> dict[str, Any]:
    protocol = _load_json(PROTOCOL_PATH)
    expected_keys = {
        "schema_version",
        "protocol_version",
        "status",
        "production_enabled",
        "purpose",
        "parent_bindings",
        "scope",
        "feature_extraction",
        "leave_context_out_validation",
        "candidate_classification",
        "global_instrument_sanity",
        "claim_limits",
        "stopping_rule",
        "source_bindings",
    }
    if set(protocol) != expected_keys:
        raise RuntimeError("spectral category protocol schema drifted")
    bindings = protocol["parent_bindings"]
    validation = protocol["leave_context_out_validation"]
    sanity = protocol["global_instrument_sanity"]
    if (
        protocol["schema_version"] != 1
        or protocol["protocol_version"] != PROTOCOL_VERSION
        or protocol["status"] != "frozen_before_first_spectral_category_evaluation"
        or protocol["production_enabled"] is not False
        or bindings["v8_result_sha256"] != sha256_file(V8_RESULT_PATH)
        or bindings["v8_record_sha256"] != v8_result["record_sha256"]
        or bindings["v1_result_sha256"] != sha256_file(V1_RESULT_PATH)
        or bindings["v1_record_sha256"] != v1_result["record_sha256"]
        or bindings["reachability_result_sha256"]
        != sha256_file(REACHABILITY_RESULT_PATH)
        or bindings["reachability_record_sha256"]
        != reachability_result["record_sha256"]
        or protocol["scope"]["voice_rule_cell_count"] != EXPECTED_CELL_COUNT
        or protocol["scope"]["occurrence_count"] != EXPECTED_OCCURRENCE_COUNT
        or protocol["feature_extraction"]["version"] != SPECTRAL_CATEGORY_VERSION
        or protocol["feature_extraction"]["config"]
        != json.loads(stable_json(asdict(DEFAULT_FEATURE_CONFIG)))
        or validation["minimum_exact_heldout_pairs"]
        != MINIMUM_HELDOUT_EXACT_ANCHOR_PAIRS
        or validation["maximum_reversed_heldout_pairs"]
        != MAXIMUM_REVERSED_HELDOUT_ANCHOR_PAIRS
        or sanity["frozen_reference_pass_cell_count"] != EXPECTED_REFERENCE_PASS_COUNT
        or sanity["minimum_reference_concordant_cell_count"] != REFERENCE_PASS_MINIMUM
        or protocol["stopping_rule"]["api_calls_allowed"] != 0
        or protocol["stopping_rule"]["new_audio_renders_allowed"] != 0
    ):
        raise RuntimeError("spectral category protocol binding drifted")
    for binding in protocol["source_bindings"]:
        if sha256_file(Paths().root / binding["path"]) != binding["sha256"]:
            raise RuntimeError(f"spectral source drifted: {binding['path']}")
    return protocol


def _without_feature(record: dict[str, Any]) -> dict[str, Any]:
    output = dict(record)
    feature = output.pop("feature")
    output["feature_sha256"] = _feature_hash(feature)
    return output


def _extract_observations(v8_result: dict[str, Any]) -> list[dict[str, Any]]:
    audio_cache: dict[Path, np.ndarray] = {}

    def samples(path: Path) -> np.ndarray:
        if path not in audio_cache:
            audio_cache[path] = _read_wav(path)
        return audio_cache[path]

    observations: list[dict[str, Any]] = []
    for outcome in sorted(
        v8_result["outcomes"], key=lambda row: row["logical_slot_id"]
    ):
        source_path = (
            V1_AUDIO_DIR / outcome["audio"]["reused_source_anchor"]["relative_path"]
        )
        target_path = (
            V1_AUDIO_DIR / outcome["audio"]["reused_target_anchor"]["relative_path"]
        )
        neutral_path = V8_AUDIO_DIR / outcome["audio"]["neutral"]["relative_path"]
        lens_path = V8_AUDIO_DIR / outcome["audio"]["lens"]["relative_path"]
        intervals = outcome["occurrence_outcomes"]
        if not (
            len(intervals)
            == len(outcome["source_anchor_intervals"])
            == len(outcome["target_anchor_intervals"])
        ):
            raise RuntimeError(f"interval count drifted: {outcome['logical_slot_id']}")
        for index, occurrence in enumerate(intervals):
            source_interval = outcome["source_anchor_intervals"][index]
            target_interval = outcome["target_anchor_intervals"][index]
            controlled_interval = occurrence["measurement_interval"]
            if (
                source_interval["start_sample"] != controlled_interval["start_sample"]
                or source_interval["end_sample_exclusive"]
                != controlled_interval["end_sample_exclusive"]
            ):
                raise RuntimeError(
                    f"neutral/source interval drifted: {outcome['logical_slot_id']}"
                )

            def feature(path: Path, interval: dict[str, Any]) -> dict[str, Any]:
                return spectral_trajectory_feature(
                    samples(path),
                    start_sample=int(interval["start_sample"]),
                    end_sample_exclusive=int(interval["end_sample_exclusive"]),
                )

            source = feature(source_path, source_interval)
            target = feature(target_path, target_interval)
            neutral = feature(neutral_path, controlled_interval)
            lens = feature(lens_path, controlled_interval)
            if source["feature"] != neutral["feature"]:
                raise RuntimeError(
                    f"neutral/source feature identity failed: {outcome['logical_slot_id']}"
                )
            observations.append(
                {
                    "cell_id": outcome["cell_id"],
                    "logical_slot_id": outcome["logical_slot_id"],
                    "context": outcome["context"],
                    "voice_id": outcome["voice_id"],
                    "profile_id": outcome["profile_id"],
                    "rule_id": outcome["rule_id"],
                    "source": outcome["source"],
                    "target": outcome["target"],
                    "claim_limit": next(
                        row["claim_limit"]
                        for row in v8_result["cell_summaries"]
                        if row["cell_id"] == outcome["cell_id"]
                    ),
                    "occurrence_index": occurrence["occurrence_index"],
                    "source_feature": source["feature"],
                    "target_feature": target["feature"],
                    "neutral_feature": neutral["feature"],
                    "lens_feature": lens["feature"],
                    "feature_receipts": {
                        "natural_source": _without_feature(source),
                        "natural_target": _without_feature(target),
                        "controlled_neutral": _without_feature(neutral),
                        "controlled_lens": _without_feature(lens),
                    },
                }
            )
    return observations


def _evaluate_observation(
    observation: dict[str, Any],
    observations: list[dict[str, Any]],
) -> dict[str, Any]:
    heldout_slot = observation["logical_slot_id"]
    voice_pool = [
        row
        for row in observations
        if row["voice_id"] == observation["voice_id"]
        and row["logical_slot_id"] != heldout_slot
    ]
    scaler_features = [
        feature
        for row in voice_pool
        for feature in (row["source_feature"], row["target_feature"])
    ]
    scaler = fit_robust_feature_scaler(scaler_features)
    training = [
        row
        for row in observations
        if row["cell_id"] == observation["cell_id"]
        and row["logical_slot_id"] != heldout_slot
    ]
    if len(training) < 2:
        raise RuntimeError(
            f"insufficient leave-slot-out anchors: {observation['logical_slot_id']}"
        )

    def scaled(feature: list[float]) -> np.ndarray:
        return apply_robust_feature_scaler(feature, scaler)

    source_centroid = np.mean(
        np.stack([scaled(row["source_feature"]) for row in training]), axis=0
    )
    target_centroid = np.mean(
        np.stack([scaled(row["target_feature"]) for row in training]), axis=0
    )
    source = scaled(observation["source_feature"])
    target = scaled(observation["target_feature"])
    neutral = scaled(observation["neutral_feature"])
    lens = scaled(observation["lens_feature"])
    natural = classify_spectral_endpoint(
        source_anchor=source_centroid,
        target_anchor=target_centroid,
        neutral=source,
        lens=target,
    )
    candidate = classify_spectral_endpoint(
        source_anchor=source_centroid,
        target_anchor=target_centroid,
        neutral=neutral,
        lens=lens,
    )
    identity = classify_spectral_endpoint(
        source_anchor=source_centroid,
        target_anchor=target_centroid,
        neutral=neutral,
        lens=neutral,
    )
    return {
        "cell_id": observation["cell_id"],
        "logical_slot_id": observation["logical_slot_id"],
        "context": observation["context"],
        "voice_id": observation["voice_id"],
        "profile_id": observation["profile_id"],
        "rule_id": observation["rule_id"],
        "source": observation["source"],
        "target": observation["target"],
        "claim_limit": observation["claim_limit"],
        "occurrence_index": observation["occurrence_index"],
        "heldout_logical_slot_id": heldout_slot,
        "scaler_pool_natural_anchor_count": len(scaler_features),
        "training_source_anchor_count": len(training),
        "training_target_anchor_count": len(training),
        "training_feature_hashes": {
            "source_centroid_scaled_f64": _feature_hash(source_centroid),
            "target_centroid_scaled_f64": _feature_hash(target_centroid),
        },
        "feature_receipts": observation["feature_receipts"],
        "natural_anchor_validation": natural,
        "candidate_classification": candidate,
        "identity_negative_control": identity,
    }


def _cell_summaries(
    evaluated: list[dict[str, Any]],
    v8_result: dict[str, Any],
    typed_core_ids: set[str],
) -> list[dict[str, Any]]:
    v8_by_cell = {row["cell_id"]: row for row in v8_result["cell_summaries"]}
    summaries: list[dict[str, Any]] = []
    for cell_id in sorted({row["cell_id"] for row in evaluated}):
        records = [row for row in evaluated if row["cell_id"] == cell_id]
        aggregate = aggregate_spectral_cell(
            natural_anchor_records=[
                row["natural_anchor_validation"] for row in records
            ],
            candidate_records=[row["candidate_classification"] for row in records],
        )
        v8 = v8_by_cell[cell_id]
        summaries.append(
            {
                "cell_id": cell_id,
                "profile_id": records[0]["profile_id"],
                "voice_id": records[0]["voice_id"],
                "rule_id": records[0]["rule_id"],
                "source": records[0]["source"],
                "target": records[0]["target"],
                "claim_limit": records[0]["claim_limit"],
                "typed_core": cell_id in typed_core_ids,
                "frozen_v8_classification": v8["classification"],
                "frozen_v8_human_qc_eligible": v8["automatic_human_qc_eligible"],
                "spectral": aggregate,
                "spectral_only_product_promotion_allowed": False,
                "product_enabled": False,
            }
        )
    return summaries


def main() -> int:
    started = time.monotonic()
    v8_result = _load_json(V8_RESULT_PATH)
    v1_result = _load_json(V1_RESULT_PATH)
    reachability = _load_json(REACHABILITY_RESULT_PATH)
    protocol = _load_protocol(v8_result, v1_result, reachability)
    typed_core_ids = set(protocol["scope"]["typed_core_cell_ids_in_order"])
    observations = _extract_observations(v8_result)
    if len(observations) != EXPECTED_OCCURRENCE_COUNT:
        raise RuntimeError("spectral observation count drifted")
    evaluated = [_evaluate_observation(row, observations) for row in observations]
    summaries = _cell_summaries(evaluated, v8_result, typed_core_ids)
    if len(summaries) != EXPECTED_CELL_COUNT:
        raise RuntimeError("spectral cell count drifted")

    reference = [row for row in summaries if row["frozen_v8_classification"] != "fail"]
    reference_concordant = [
        row for row in reference if row["spectral"]["directional_pass"]
    ]
    negative_false_positives = [
        row for row in evaluated if row["identity_negative_control"]["directional_pass"]
    ]
    instrument_sanity_pass = bool(
        len(reference) == EXPECTED_REFERENCE_PASS_COUNT
        and len(reference_concordant) >= REFERENCE_PASS_MINIMUM
        and not negative_false_positives
    )
    spectral_counts: dict[str, int] = {}
    for row in summaries:
        classification = row["spectral"]["classification"]
        spectral_counts[classification] = spectral_counts.get(classification, 0) + 1
    core = [row for row in summaries if row["typed_core"]]
    core_candidate_passes = [row for row in core if row["spectral"]["directional_pass"]]
    core_new_candidates = [
        row
        for row in core_candidate_passes
        if row["frozen_v8_classification"] == "fail"
    ]
    result: dict[str, Any] = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "protocol_version": PROTOCOL_VERSION,
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "spectral_category_version": SPECTRAL_CATEGORY_VERSION,
        "classification": (
            "spectral_instrument_sanity_pass_no_product_promotion"
            if instrument_sanity_pass
            else "spectral_instrument_sanity_fail_no_product_promotion"
        ),
        "production_enabled": False,
        "api_calls_made": 0,
        "new_audio_renders_made": 0,
        "parent_bindings": protocol["parent_bindings"],
        "voice_rule_cell_count": len(summaries),
        "occurrence_count": len(evaluated),
        "feature_extraction_error_count": 0,
        "identity_negative_control_false_positive_count": len(negative_false_positives),
        "instrument_sanity": {
            "frozen_v8_reference_pass_cell_count": len(reference),
            "minimum_reference_concordant_cell_count": REFERENCE_PASS_MINIMUM,
            "reference_concordant_cell_count": len(reference_concordant),
            "reference_concordance_fraction": len(reference_concordant)
            / len(reference),
            "identity_negative_control_count": len(evaluated),
            "identity_negative_control_false_positive_count": len(
                negative_false_positives
            ),
            "pass": instrument_sanity_pass,
        },
        "cell_classification_counts": spectral_counts,
        "typed_core": {
            "cell_count": len(core),
            "spectral_anchor_validated_count": sum(
                row["spectral"]["anchor_validation_pass"] for row in core
            ),
            "spectral_candidate_pass_count": len(core_candidate_passes),
            "spectral_candidate_pass_cell_ids": [
                row["cell_id"] for row in core_candidate_passes
            ],
            "new_spectral_candidate_count_against_frozen_v8": len(core_new_candidates),
            "new_spectral_candidate_cell_ids_against_frozen_v8": [
                row["cell_id"] for row in core_new_candidates
            ],
            "interpretation": (
                "Descriptive complementary evidence only. A spectral candidate "
                "does not rewrite frozen Praat outcomes or authorize product use."
            ),
        },
        "cell_summaries": summaries,
        "outcomes": evaluated,
        "claim_limits": protocol["claim_limits"],
        "elapsed_s": time.monotonic() - started,
    }
    result["record_sha256"] = _semantic_hash(result)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    atomic_write_json(RUN_DIR / "results.json", result)
    print(
        stable_json(
            {
                key: result[key]
                for key in (
                    "classification",
                    "voice_rule_cell_count",
                    "occurrence_count",
                    "instrument_sanity",
                    "cell_classification_counts",
                    "typed_core",
                    "elapsed_s",
                    "record_sha256",
                )
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
