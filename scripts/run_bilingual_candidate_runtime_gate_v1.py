#!/usr/bin/env python3
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from earshift_bakeoff.bilingual_listener_engine import (
    BilingualListenerPlanner,
)
from earshift_bakeoff.bilingual_listener_engine_v8 import (
    bilingual_alignment_record_v8,
)
from earshift_bakeoff.bilingual_vowel_engine import _load_pinned_synthesis_voice
from earshift_bakeoff.bilingual_vowel_replicated_anchors import (
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
)
from earshift_bakeoff.config import Paths, stable_json
from earshift_bakeoff.util import atomic_write_json, sha256_file

import run_bilingual_vowel_replicated_anchor_calibration_v1 as calibration
import run_bilingual_vowel_unseen_typed_confirmation_v1 as unseen


PROTOCOL_VERSION = "bilingual-candidate-runtime-gate-v1"
PROTOCOL_PATH = Paths().root / "rules" / f"{PROTOCOL_VERSION}.json"
RUN_ID = "20260718-bilingual-candidate-runtime-gate-v1"
RUN_DIR = Paths().artifacts / "product-matrix" / RUN_ID
SCALER_PATH = RUN_DIR / "voice-scalers.json"
RESULT_PATH = RUN_DIR / "results.json"
UNSEEN_RESULT_PATH = unseen.RUN_DIR / "results.json"
EXPECTED_SLOT_COUNT = 84
EXPECTED_OCCURRENCE_COUNT = 112
EXPECTED_NATURAL_DECODER_RENDER_COUNT = 504
EXPECTED_PRIOR_PASS_COUNT = 18


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected an object in {path}")
    return value


def _semantic_hash(payload: dict[str, Any]) -> str:
    semantic = dict(payload)
    semantic.pop("record_sha256", None)
    return hashlib.sha256(stable_json(semantic).encode("utf-8")).hexdigest()


def _feature_hash(feature: list[float]) -> str:
    return hashlib.sha256(
        np.asarray(feature, dtype="<f8").tobytes()
    ).hexdigest()


def _feature_record(pcm: np.ndarray, interval: dict[str, Any]) -> dict[str, Any]:
    feature, receipt = calibration._feature_receipt(pcm, interval)
    if receipt["feature_sha256"] != _feature_hash(feature):
        raise RuntimeError("spectral feature receipt drifted")
    return {"feature": feature, "receipt": receipt}


def _load_protocol(
    *, manifest: dict[str, Any], unseen_result: dict[str, Any]
) -> dict[str, Any]:
    protocol = _load_json(PROTOCOL_PATH)
    if (
        protocol.get("schema_version") != 1
        or protocol.get("protocol_version") != PROTOCOL_VERSION
        or protocol.get("status") != "frozen_before_runtime_gate_execution"
        or protocol.get("production_enabled") is not False
        or protocol.get("scope", {}).get("logical_slot_count")
        != EXPECTED_SLOT_COUNT
        or protocol.get("scope", {}).get("target_occurrence_count")
        != EXPECTED_OCCURRENCE_COUNT
        or protocol.get("scope", {}).get("prior_unseen_pass_cell_count")
        != EXPECTED_PRIOR_PASS_COUNT
        or protocol.get("rendering", {}).get("natural_decoder_render_count")
        != EXPECTED_NATURAL_DECODER_RENDER_COUNT
        or tuple(protocol.get("rendering", {}).get("training_seeds_in_order", ()))
        != TRAINING_SEEDS
        or protocol.get("stopping_rule", {}).get("api_calls_allowed") != 0
        or protocol.get("stopping_rule", {}).get("candidate_audio_rerenders_allowed")
        != 0
        or protocol.get("stopping_rule", {}).get("product_promotion_allowed")
        is not False
    ):
        raise RuntimeError("runtime-gate protocol drifted")
    bindings = protocol["parent_bindings"]
    if (
        bindings.get("unseen_manifest_sha256") != sha256_file(unseen.MANIFEST_PATH)
        or bindings.get("unseen_manifest_record_sha256")
        != manifest.get("record_sha256")
        or bindings.get("unseen_result_sha256")
        != sha256_file(UNSEEN_RESULT_PATH)
        or bindings.get("unseen_result_record_sha256")
        != unseen_result.get("record_sha256")
    ):
        raise RuntimeError("runtime-gate parent binding drifted")
    for binding in protocol.get("source_bindings", ()):
        if sha256_file(Paths().root / binding["path"]) != binding["sha256"]:
            raise RuntimeError(f"runtime-gate source drifted: {binding['path']}")
    return protocol


def _scaled(feature: list[float], scaler: dict[str, Any]) -> np.ndarray:
    return apply_robust_feature_scaler(feature, scaler)


def _classify_anchor_pair(
    *,
    source: dict[int, list[float]],
    target: dict[int, list[float]],
    heldout_seed: int,
    scaler: dict[str, Any],
) -> dict[str, Any]:
    other = tuple(seed for seed in TRAINING_SEEDS if seed != heldout_seed)
    source_centroid = np.mean(
        np.stack([_scaled(source[seed], scaler) for seed in other]), axis=0
    )
    target_centroid = np.mean(
        np.stack([_scaled(target[seed], scaler) for seed in other]), axis=0
    )
    return {
        "heldout_seed": heldout_seed,
        **classify_spectral_endpoint(
            source_anchor=source_centroid,
            target_anchor=target_centroid,
            neutral=_scaled(source[heldout_seed], scaler),
            lens=_scaled(target[heldout_seed], scaler),
        ),
    }


def _classify_candidate(
    *,
    source: dict[int, list[float]],
    target: dict[int, list[float]],
    neutral: list[float],
    lens: list[float],
    scaler: dict[str, Any],
) -> dict[str, Any]:
    source_centroid = np.mean(
        np.stack([_scaled(source[seed], scaler) for seed in TRAINING_SEEDS]), axis=0
    )
    target_centroid = np.mean(
        np.stack([_scaled(target[seed], scaler) for seed in TRAINING_SEEDS]), axis=0
    )
    return classify_spectral_endpoint(
        source_anchor=source_centroid,
        target_anchor=target_centroid,
        neutral=_scaled(neutral, scaler),
        lens=_scaled(lens, scaler),
    )


def main() -> int:
    if RUN_DIR.exists():
        raise RuntimeError(f"refusing to overwrite runtime gate: {RUN_DIR}")
    manifest = _load_json(unseen.MANIFEST_PATH)
    unseen_result = _load_json(UNSEEN_RESULT_PATH)
    protocol = _load_protocol(manifest=manifest, unseen_result=unseen_result)
    if (
        manifest.get("logical_slot_count") != EXPECTED_SLOT_COUNT
        or unseen_result.get("logical_slot_count") != EXPECTED_SLOT_COUNT
        or unseen_result.get("target_occurrence_count") != EXPECTED_OCCURRENCE_COUNT
        or unseen_result.get("unseen_automatic_pass_count")
        != EXPECTED_PRIOR_PASS_COUNT
    ):
        raise RuntimeError("runtime-gate denominator drifted")
    outcome_by_slot = {
        row["logical_slot_id"]: row for row in unseen_result["outcomes"]
    }
    base_planners: dict[tuple[str, str], BilingualListenerPlanner] = {}
    isolated_planners: dict[tuple[str, str, str], Any] = {}
    observations: list[dict[str, Any]] = []
    natural_decoder_render_count = 0
    started = time.perf_counter()
    for voice_id in unseen.VOICE_ORDER:
        synthesis = _load_pinned_synthesis_voice(voice_id)
        for slot in sorted(
            (row for row in manifest["slots"] if row["voice_id"] == voice_id),
            key=lambda row: row["logical_slot_id"],
        ):
            outcome = outcome_by_slot[slot["logical_slot_id"]]
            base_key = (slot["profile_id"], voice_id)
            if base_key not in base_planners:
                base_planners[base_key] = BilingualListenerPlanner.load(
                    slot["profile_id"], voice_id=voice_id
                )
            planner_key = (slot["profile_id"], voice_id, slot["rule_id"])
            if planner_key not in isolated_planners:
                isolated_planners[planner_key] = unseen._isolated_planner(
                    base=base_planners[base_key],
                    profile_id=slot["profile_id"],
                    voice_id=voice_id,
                    rule_id=slot["rule_id"],
                )
            planner = isolated_planners[planner_key]
            plan = planner.plan(slot["fixture_spec"]["text"])
            if plan.plan_sha256 != slot["plan_sha256"]:
                raise RuntimeError(f"runtime-gate plan drifted: {slot['logical_slot_id']}")
            neutral_record = outcome["audio"]["neutral"]
            lens_record = outcome["audio"]["lens"]
            if lens_record is None:
                raise RuntimeError("runtime-gate candidate WAV is missing")
            neutral_path = unseen.RUN_DIR / neutral_record["relative_path"]
            lens_path = unseen.RUN_DIR / lens_record["relative_path"]
            if (
                sha256_file(neutral_path) != neutral_record["wav_sha256"]
                or sha256_file(lens_path) != lens_record["wav_sha256"]
            ):
                raise RuntimeError("runtime-gate candidate WAV hash drifted")
            neutral_pcm = calibration._read_wav(neutral_path)
            lens_pcm = calibration._read_wav(lens_path)
            source_render = render_seeded_natural_conditions(
                synthesis,
                phonemes=plan.neutral_phonemes,
                reference_phonemes=plan.render_reference_phonemes,
                seeds=TRAINING_SEEDS,
            )
            target_render = render_seeded_natural_conditions(
                synthesis,
                phonemes=plan.lens_phonemes,
                reference_phonemes=plan.render_reference_phonemes,
                seeds=TRAINING_SEEDS,
            )
            natural_decoder_render_count += 2 * len(TRAINING_SEEDS)
            source_pcm = {
                seed: calibration._natural_pcm(source_render.audio_by_seed[seed])
                for seed in TRAINING_SEEDS
            }
            target_pcm = {
                seed: calibration._natural_pcm(target_render.audio_by_seed[seed])
                for seed in TRAINING_SEEDS
            }
            source_alignment = bilingual_alignment_record_v8(
                model=synthesis.model,
                plan=plan,
                durations=source_render.predicted_durations,
                sample_count=next(iter(source_pcm.values())).size,
            )
            target_alignment = bilingual_alignment_record_v8(
                model=synthesis.model,
                plan=plan,
                durations=target_render.predicted_durations,
                sample_count=next(iter(target_pcm.values())).size,
            )
            candidate_alignment = bilingual_alignment_record_v8(
                model=synthesis.model,
                plan=plan,
                durations=source_render.predicted_durations,
                sample_count=neutral_pcm.size,
            )
            source_rows = unseen._target_rows(source_alignment, slot["rule_id"])
            target_rows = unseen._target_rows(target_alignment, slot["rule_id"])
            candidate_rows = unseen._target_rows(candidate_alignment, slot["rule_id"])
            expected = slot["fixture_spec"]["expected_target_occurrence_count"]
            if not len(source_rows) == len(target_rows) == len(candidate_rows) == expected:
                raise RuntimeError("runtime-gate occurrence alignment drifted")
            for index, candidate_row in enumerate(candidate_rows):
                source_features = {
                    seed: _feature_record(
                        source_pcm[seed], source_rows[index]["measurement_interval"]
                    )
                    for seed in TRAINING_SEEDS
                }
                target_features = {
                    seed: _feature_record(
                        target_pcm[seed], target_rows[index]["measurement_interval"]
                    )
                    for seed in TRAINING_SEEDS
                }
                neutral_feature = _feature_record(
                    neutral_pcm, candidate_row["measurement_interval"]
                )
                lens_feature = _feature_record(
                    lens_pcm, candidate_row["measurement_interval"]
                )
                observations.append(
                    {
                        "cell_id": slot["cell_id"],
                        "logical_slot_id": slot["logical_slot_id"],
                        "context": slot["context"],
                        "profile_id": slot["profile_id"],
                        "voice_id": voice_id,
                        "rule_id": slot["rule_id"],
                        "occurrence_index": candidate_row["occurrence_index"],
                        "source_features": {
                            seed: record["feature"]
                            for seed, record in source_features.items()
                        },
                        "target_features": {
                            seed: record["feature"]
                            for seed, record in target_features.items()
                        },
                        "neutral_feature": neutral_feature["feature"],
                        "lens_feature": lens_feature["feature"],
                        "feature_receipts": {
                            "source_by_seed": {
                                str(seed): record["receipt"]
                                for seed, record in source_features.items()
                            },
                            "target_by_seed": {
                                str(seed): record["receipt"]
                                for seed, record in target_features.items()
                            },
                            "neutral": neutral_feature["receipt"],
                            "lens": lens_feature["receipt"],
                        },
                    }
                )
    if (
        natural_decoder_render_count != EXPECTED_NATURAL_DECODER_RENDER_COUNT
        or len(observations) != EXPECTED_OCCURRENCE_COUNT
    ):
        raise RuntimeError("runtime-gate execution denominator drifted")

    scalers: dict[str, Any] = {}
    for voice_id in unseen.VOICE_ORDER:
        voice_features = [
            feature
            for row in observations
            if row["voice_id"] == voice_id
            for side in ("source_features", "target_features")
            for feature in row[side].values()
        ]
        scaler = fit_robust_feature_scaler(voice_features)
        scalers[voice_id] = scaler
    scaler_result: dict[str, Any] = {
        "schema_version": 1,
        "version": PROTOCOL_VERSION,
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "spectral_category_version": SPECTRAL_CATEGORY_VERSION,
        "feature_config": calibration.asdict(DEFAULT_FEATURE_CONFIG),
        "training_seeds": list(TRAINING_SEEDS),
        "voice_scalers": scalers,
        "production_enabled": False,
    }
    scaler_result["record_sha256"] = _semantic_hash(scaler_result)
    RUN_DIR.mkdir(parents=True, exist_ok=False)
    atomic_write_json(SCALER_PATH, scaler_result)

    occurrence_results: list[dict[str, Any]] = []
    identity_false_positive_count = 0
    for row in observations:
        scaler = scalers[row["voice_id"]]
        natural = [
            _classify_anchor_pair(
                source=row["source_features"],
                target=row["target_features"],
                heldout_seed=seed,
                scaler=scaler,
            )
            for seed in TRAINING_SEEDS
        ]
        candidate = _classify_candidate(
            source=row["source_features"],
            target=row["target_features"],
            neutral=row["neutral_feature"],
            lens=row["lens_feature"],
            scaler=scaler,
        )
        identity = _classify_candidate(
            source=row["source_features"],
            target=row["target_features"],
            neutral=row["neutral_feature"],
            lens=row["neutral_feature"],
            scaler=scaler,
        )
        identity_false_positive_count += int(identity["directional_pass"])
        aggregate = aggregate_replicated_anchor_occurrence(
            natural_seed_records=natural,
            candidate_record=candidate,
        )
        occurrence_results.append(
            {
                "cell_id": row["cell_id"],
                "logical_slot_id": row["logical_slot_id"],
                "context": row["context"],
                "profile_id": row["profile_id"],
                "voice_id": row["voice_id"],
                "rule_id": row["rule_id"],
                "occurrence_index": row["occurrence_index"],
                "natural_seed_classifications": natural,
                "identity_negative_control": identity,
                "candidate": candidate,
                "aggregate": aggregate,
                "feature_receipts": row["feature_receipts"],
            }
        )

    prior_by_cell = {row["cell_id"]: row for row in unseen_result["cell_summaries"]}
    cell_results: list[dict[str, Any]] = []
    for cell_id, prior in sorted(prior_by_cell.items()):
        rows = [row for row in occurrence_results if row["cell_id"] == cell_id]
        aggregate = aggregate_replicated_anchor_cell(
            [row["aggregate"] for row in rows],
            expected_occurrence_count=len(rows),
        )
        prior_pass = bool(prior["unseen_automatic_pass"])
        runtime_gate_pass = bool(
            prior_pass
            and aggregate["directional_pass"]
            and identity_false_positive_count == 0
        )
        cell_results.append(
            {
                "cell_id": cell_id,
                "profile_id": prior["profile_id"],
                "voice_id": prior["voice_id"],
                "rule_id": prior["rule_id"],
                "candidate_rung": prior["candidate_rung"],
                "prior_unseen_classification": prior["replicated_anchor"][
                    "classification"
                ],
                "prior_unseen_pass": prior_pass,
                "runtime_gate": aggregate,
                "runtime_gate_pass": runtime_gate_pass,
                "product_enabled": False,
            }
        )
    runtime_passes = [row for row in cell_results if row["runtime_gate_pass"]]
    lost_prior = [
        row["cell_id"]
        for row in cell_results
        if row["prior_unseen_pass"] and not row["runtime_gate_pass"]
    ]
    result: dict[str, Any] = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "protocol_version": PROTOCOL_VERSION,
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "classification": (
            "runtime_gate_instrument_sanity_fail"
            if identity_false_positive_count
            else "runtime_gate_complete_no_product_promotion"
        ),
        "production_enabled": False,
        "api_calls_made": 0,
        "candidate_audio_rerenders_made": 0,
        "natural_decoder_render_count": natural_decoder_render_count,
        "logical_slot_count": EXPECTED_SLOT_COUNT,
        "target_occurrence_count": len(occurrence_results),
        "identity_negative_control_count": len(occurrence_results),
        "identity_negative_control_false_positive_count": (
            identity_false_positive_count
        ),
        "prior_unseen_pass_count": EXPECTED_PRIOR_PASS_COUNT,
        "runtime_gate_pass_count": len(runtime_passes),
        "lost_prior_pass_cell_ids": lost_prior,
        "runtime_gate_classification_counts": dict(
            Counter(row["runtime_gate"]["classification"] for row in cell_results)
        ),
        "voice_scaler_path": str(SCALER_PATH.relative_to(Paths().root)),
        "voice_scaler_sha256": sha256_file(SCALER_PATH),
        "voice_scaler_record_sha256": scaler_result["record_sha256"],
        "cell_results": cell_results,
        "occurrence_results": occurrence_results,
        "claim_limits": protocol["claim_limits"],
        "elapsed_s": time.perf_counter() - started,
    }
    result["record_sha256"] = _semantic_hash(result)
    atomic_write_json(RESULT_PATH, result)
    print(
        stable_json(
            {
                key: result[key]
                for key in (
                    "classification",
                    "natural_decoder_render_count",
                    "target_occurrence_count",
                    "identity_negative_control_false_positive_count",
                    "prior_unseen_pass_count",
                    "runtime_gate_pass_count",
                    "lost_prior_pass_cell_ids",
                    "runtime_gate_classification_counts",
                    "api_calls_made",
                    "elapsed_s",
                    "record_sha256",
                )
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
