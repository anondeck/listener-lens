#!/usr/bin/env python3
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Sequence
import wave

import numpy as np

from earshift_bakeoff.bilingual_listener_engine import load_listener_profiles
from earshift_bakeoff.bilingual_vowel_engine import _load_pinned_synthesis_voice
from earshift_bakeoff.bilingual_vowel_replicated_anchors import (
    TRAINING_SEEDS,
    aggregate_replicated_anchor_cell,
    aggregate_replicated_anchor_occurrence,
    render_seeded_natural_conditions,
)
from earshift_bakeoff.bilingual_vowel_spectral_category import (
    apply_robust_feature_scaler,
    classify_spectral_endpoint,
    fit_robust_feature_scaler,
)
from earshift_bakeoff.config import Paths, stable_json
from earshift_bakeoff.kokoro_gate_bridge import KokoroGateIndex
from earshift_bakeoff.kokoro_output_domain_splice import boundary_artifact_report
from earshift_bakeoff.kokoro_synthesis import CONFIG_FILE, RNG_SEED, verify_model_files
from earshift_bakeoff.kokoro_typed_diagnostic import localization_report
from earshift_bakeoff.listener_lens import DatabaseNonceChecker
from earshift_bakeoff.portuguese_carrier_planner_v1 import (
    PortuguesePositiveOnlyIndexV1,
)
from earshift_bakeoff.util import atomic_write_json, sha256_file

import run_bilingual_vowel_replicated_anchor_calibration_v1 as calibration
from run_bilingual_vowel_adaptive_strength_screen_v1 import (
    STRENGTHS,
    _occurrence_windows,
)


PROTOCOL_VERSION = "bilingual-vowel-replicated-anchor-failure-screen-v1"
PROTOCOL_PATH = Paths().root / "rules" / f"{PROTOCOL_VERSION}.json"
RUN_ID = "20260718-bilingual-vowel-replicated-anchor-failure-screen-v1"
RUN_DIR = Paths().artifacts / "product-matrix" / RUN_ID
CALIBRATION_RESULT_PATH = (
    Paths().artifacts
    / "product-matrix"
    / "20260717-bilingual-vowel-replicated-anchor-calibration-v1"
    / "results.json"
)
CALIBRATION_ERRATUM_PATH = CALIBRATION_RESULT_PATH.parent / "accounting-erratum.json"
V8_MANIFEST_PATH = calibration.V8_MANIFEST_PATH
V8_RESULT_PATH = calibration.V8_RESULT_PATH
V8_DIR = V8_RESULT_PATH.parent
ADAPTIVE_RESULT_PATH = calibration.ADAPTIVE_RESULT_PATH
ADAPTIVE_DIR = ADAPTIVE_RESULT_PATH.parent
VOICE_ORDER = calibration.VOICE_ORDER
EXPECTED_CORE_CELL_COUNT = 36
EXPECTED_ANCHOR_SLOT_COUNT = 108
EXPECTED_ANCHOR_OCCURRENCE_COUNT = 144
EXPECTED_FAILURE_CELL_COUNT = 23
EXPECTED_FAILURE_SLOT_COUNT = 69
EXPECTED_FAILURE_OCCURRENCE_COUNT = 92
ADAPTIVE_CELL_COUNT = 2
V8_CELL_COUNT = 21


def _semantic_hash(payload: dict[str, Any]) -> str:
    semantic = dict(payload)
    semantic.pop("record_sha256", None)
    return hashlib.sha256(stable_json(semantic).encode("utf-8")).hexdigest()


def _safe_name(value: str) -> str:
    return "".join(character if character.isalnum() else "-" for character in value)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_protocol(
    *,
    calibration_result: dict[str, Any],
    v8_result: dict[str, Any],
    adaptive_result: dict[str, Any],
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
        "anchor_reproduction",
        "candidate_hierarchy",
        "classification_and_aggregation",
        "claim_limits",
        "stopping_rule",
        "source_bindings",
    }
    if set(protocol) != expected_keys:
        raise RuntimeError("replicated-anchor failure protocol schema drifted")
    parents = protocol["parent_bindings"]
    expected_parents = {
        "calibration": (CALIBRATION_RESULT_PATH, calibration_result),
        "v8": (V8_RESULT_PATH, v8_result),
        "adaptive": (ADAPTIVE_RESULT_PATH, adaptive_result),
    }
    for label, (path, result) in expected_parents.items():
        if (
            parents[f"{label}_result_sha256"] != sha256_file(path)
            or parents[f"{label}_record_sha256"] != result["record_sha256"]
        ):
            raise RuntimeError(f"replicated-anchor failure parent drifted: {label}")
    if parents["calibration_erratum_sha256"] != sha256_file(
        CALIBRATION_ERRATUM_PATH
    ) or parents["v8_manifest_sha256"] != sha256_file(V8_MANIFEST_PATH):
        raise RuntimeError("replicated-anchor failure binding drifted")
    scope = protocol["scope"]
    hierarchy = protocol["candidate_hierarchy"]
    stopping = protocol["stopping_rule"]
    if (
        protocol["schema_version"] != 1
        or protocol["protocol_version"] != PROTOCOL_VERSION
        or protocol["status"]
        != "frozen_before_first_eligible_failure_candidate_evaluation"
        or protocol["production_enabled"] is not False
        or scope["calibrated_core_cell_count"] != EXPECTED_CORE_CELL_COUNT
        or scope["eligible_failure_cell_count"] != EXPECTED_FAILURE_CELL_COUNT
        or scope["eligible_failure_slot_count"] != EXPECTED_FAILURE_SLOT_COUNT
        or scope["eligible_failure_occurrence_count"]
        != EXPECTED_FAILURE_OCCURRENCE_COUNT
        or hierarchy["v8_cell_count"] != V8_CELL_COUNT
        or hierarchy["adaptive_strength_cell_count"] != ADAPTIVE_CELL_COUNT
        or tuple(hierarchy["adaptive_strength_order"]) != STRENGTHS
        or stopping["api_calls_allowed"] != 0
        or stopping["new_candidate_decoder_renders_allowed"] != 0
        or stopping["product_promotion_allowed"] is not False
    ):
        raise RuntimeError("replicated-anchor failure protocol contract drifted")
    for binding in protocol["source_bindings"]:
        if sha256_file(Paths().root / binding["path"]) != binding["sha256"]:
            raise RuntimeError(
                f"replicated-anchor failure source drifted: {binding['path']}"
            )
    return protocol


def _write_wav(path: Path, pcm: np.ndarray) -> dict[str, Any]:
    values = np.asarray(pcm, dtype="<i2").reshape(-1)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with wave.open(str(temporary), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(calibration.DEFAULT_FEATURE_CONFIG.sample_rate_hz)
        handle.writeframes(values.tobytes())
    temporary.replace(path)
    return {
        "relative_path": str(path.relative_to(RUN_DIR)),
        "wav_sha256": sha256_file(path),
        "pcm_sha256": calibration._pcm_hash(values),
        "sample_count": int(values.size),
        "duration_s": values.size / calibration.DEFAULT_FEATURE_CONFIG.sample_rate_hz,
    }


def _context_endpoints(
    observation: dict[str, Any], observations: list[dict[str, Any]]
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    voice_pool = [
        row
        for row in observations
        if row["voice_id"] == observation["voice_id"]
        and row["logical_slot_id"] != observation["logical_slot_id"]
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

    source_centroid = np.mean(
        np.stack(
            [scaled(observation["source_features"][seed]) for seed in TRAINING_SEEDS]
        ),
        axis=0,
    )
    target_centroid = np.mean(
        np.stack(
            [scaled(observation["target_features"][seed]) for seed in TRAINING_SEEDS]
        ),
        axis=0,
    )
    return scaler, source_centroid, target_centroid


def _classify_candidate_feature(
    *,
    feature: list[float],
    neutral_feature: list[float],
    scaler: dict[str, Any],
    source_centroid: np.ndarray,
    target_centroid: np.ndarray,
) -> dict[str, Any]:
    return classify_spectral_endpoint(
        source_anchor=source_centroid,
        target_anchor=target_centroid,
        neutral=apply_robust_feature_scaler(neutral_feature, scaler),
        lens=apply_robust_feature_scaler(feature, scaler),
    )


def _select_strength(records: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    for desired in ("exact_category_pass", "directional_only_pass"):
        for record in records:
            if record["classification"] == desired:
                return record
    return None


def _composite_integrity(
    *,
    neutral: np.ndarray,
    candidate: np.ndarray,
    windows: Sequence[dict[str, Any]],
    intervals: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    mask = np.zeros(neutral.size, dtype=bool)
    for window in windows:
        mask[int(window["start_sample"]) : int(window["end_sample_exclusive"])] = True
    boundary = boundary_artifact_report(neutral, candidate, candidate, windows)
    localization = localization_report(neutral, candidate, intervals)
    return {
        "equal_nonempty_samples": bool(neutral.size and neutral.size == candidate.size),
        "finite": bool(np.isfinite(candidate.astype(np.float64)).all()),
        "unclipped": bool(np.mean(np.abs(candidate.astype(np.int64)) >= 32767) < 0.001),
        "outside_windows_exact_neutral": bool(
            np.array_equal(candidate[~mask], neutral[~mask])
        ),
        "boundary_metrics_pass": bool(boundary.get("pass")),
        "localization_pass": bool(localization.get("pass")),
        "localization_fraction": float(
            localization.get("inside_difference_energy_fraction", 0.0)
        ),
        "integrity_pass": bool(
            neutral.size
            and neutral.size == candidate.size
            and np.isfinite(candidate.astype(np.float64)).all()
            and np.mean(np.abs(candidate.astype(np.int64)) >= 32767) < 0.001
            and np.array_equal(candidate[~mask], neutral[~mask])
            and boundary.get("pass") is True
            and localization.get("pass") is True
        ),
    }


def main() -> int:
    if RUN_DIR.exists():
        raise RuntimeError(f"refusing to overwrite failure screen: {RUN_DIR}")
    calibration_result = _load_json(CALIBRATION_RESULT_PATH)
    manifest = _load_json(V8_MANIFEST_PATH)
    v8 = _load_json(V8_RESULT_PATH)
    adaptive = _load_json(ADAPTIVE_RESULT_PATH)
    protocol = _load_protocol(
        calibration_result=calibration_result,
        v8_result=v8,
        adaptive_result=adaptive,
    )
    eligible_cells = {
        row["cell_id"]: row["candidate_rung"]
        for row in protocol["candidate_hierarchy"]["cells_in_order"]
    }
    if len(eligible_cells) != EXPECTED_FAILURE_CELL_COUNT:
        raise RuntimeError("eligible failure cell list drifted")
    core_ids = set(protocol["scope"]["calibrated_core_cell_ids_in_order"])
    slots = [row for row in manifest["slots"] if row["cell_id"] in core_ids]
    if len(slots) != EXPECTED_ANCHOR_SLOT_COUNT:
        raise RuntimeError("anchor reproduction slot count drifted")
    v8_by_id = {row["logical_slot_id"]: row for row in v8["outcomes"]}
    adaptive_by_id = {row["logical_slot_id"]: row for row in adaptive["outcomes"]}
    calibration_slots = {
        row["logical_slot_id"]: row for row in calibration_result["slot_receipts"]
    }
    calibration_occurrences = {
        (row["logical_slot_id"], row["occurrence_index"]): row
        for row in calibration_result["outcomes"]
    }
    profiles = load_listener_profiles()
    files = verify_model_files(download=False)
    model_vocab = set(
        json.loads(files[CONFIG_FILE].read_text(encoding="utf-8"))["vocab"]
    )
    nonce_checker = DatabaseNonceChecker()
    phone_indexes = (KokoroGateIndex(), PortuguesePositiveOnlyIndexV1())
    observations: list[dict[str, Any]] = []
    reproduced_audio_count = reproduced_feature_count = 0
    started = time.perf_counter()
    seeds = (RNG_SEED, *TRAINING_SEEDS)
    for voice_id in VOICE_ORDER:
        synthesis = _load_pinned_synthesis_voice(voice_id)
        for slot in sorted(
            (row for row in slots if row["voice_id"] == voice_id),
            key=lambda row: row["logical_slot_id"],
        ):
            planner = calibration._planner_v8(
                slot=slot,
                profiles=profiles,
                model_vocab=model_vocab,
                nonce_checker=nonce_checker,
                phone_indexes=phone_indexes,
            )
            plan = planner.plan(slot["fixture_spec"]["text"])
            if plan.plan_sha256 != slot["v8_plan_sha256"]:
                raise RuntimeError(
                    f"failure-screen plan drifted: {slot['logical_slot_id']}"
                )
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
                seed: calibration._natural_pcm(source_render.audio_by_seed[seed])
                for seed in seeds
            }
            target_pcm = {
                seed: calibration._natural_pcm(target_render.audio_by_seed[seed])
                for seed in seeds
            }
            frozen_slot = calibration_slots[slot["logical_slot_id"]]
            for side, values in (("source", source_pcm), ("target", target_pcm)):
                for seed, pcm in values.items():
                    receipt = calibration._pcm_receipt(pcm)
                    if receipt != frozen_slot[f"{side}_seed_audio"][str(seed)]:
                        raise RuntimeError(
                            f"replicated anchor PCM changed: {slot['logical_slot_id']}"
                        )
                    reproduced_audio_count += 1
            v8_outcome = v8_by_id[slot["logical_slot_id"]]
            source_intervals = v8_outcome["source_anchor_intervals"]
            target_intervals = v8_outcome["target_anchor_intervals"]
            controlled_rows = v8_outcome["occurrence_outcomes"]
            for index, controlled in enumerate(controlled_rows):
                source_features: dict[int, list[float]] = {}
                target_features: dict[int, list[float]] = {}
                for seed in TRAINING_SEEDS:
                    feature, receipt = calibration._feature_receipt(
                        source_pcm[seed], source_intervals[index]
                    )
                    source_features[seed] = feature
                    frozen = calibration_occurrences[
                        (slot["logical_slot_id"], controlled["occurrence_index"])
                    ]["feature_receipts"]["natural_source_by_seed"][str(seed)]
                    if receipt != frozen:
                        raise RuntimeError("source anchor feature reproduction failed")
                    reproduced_feature_count += 1
                    feature, receipt = calibration._feature_receipt(
                        target_pcm[seed], target_intervals[index]
                    )
                    target_features[seed] = feature
                    frozen = calibration_occurrences[
                        (slot["logical_slot_id"], controlled["occurrence_index"])
                    ]["feature_receipts"]["natural_target_by_seed"][str(seed)]
                    if receipt != frozen:
                        raise RuntimeError("target anchor feature reproduction failed")
                    reproduced_feature_count += 1
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
                        "measurement_interval": controlled["measurement_interval"],
                        "source_features": source_features,
                        "target_features": target_features,
                    }
                )
    if len(observations) != EXPECTED_ANCHOR_OCCURRENCE_COUNT:
        raise RuntimeError("anchor reproduction occurrence count drifted")

    anchor_evaluations: dict[tuple[str, int], dict[str, Any]] = {}
    for observation in observations:
        frozen = calibration_occurrences[
            (observation["logical_slot_id"], observation["occurrence_index"])
        ]
        anchor_evaluations[
            (observation["logical_slot_id"], observation["occurrence_index"])
        ] = {
            "natural_seed_classifications": frozen["natural_seed_classifications"],
            "aggregate": frozen["aggregate"],
        }

    candidate_outcomes: list[dict[str, Any]] = []
    retained_composites = 0
    for cell_id, rung in sorted(eligible_cells.items()):
        cell_observations = [row for row in observations if row["cell_id"] == cell_id]
        for slot_id in sorted({row["logical_slot_id"] for row in cell_observations}):
            slot_observations = [
                row for row in cell_observations if row["logical_slot_id"] == slot_id
            ]
            v8_outcome = v8_by_id[slot_id]
            neutral_record = v8_outcome["audio"]["neutral"]
            neutral_path = V8_DIR / neutral_record["relative_path"]
            if sha256_file(neutral_path) != neutral_record["wav_sha256"]:
                raise RuntimeError(f"failure neutral WAV drifted: {slot_id}")
            neutral_pcm = calibration._read_wav(neutral_path)
            intervals = [row["measurement_interval"] for row in slot_observations]
            occurrence_records = []
            candidate_audio_record = None
            candidate_integrity = None
            selection_records: list[dict[str, Any] | None] = []
            if rung == "v8":
                lens_record = v8_outcome["audio"]["lens"]
                lens_path = V8_DIR / lens_record["relative_path"]
                if sha256_file(lens_path) != lens_record["wav_sha256"]:
                    raise RuntimeError(f"failure lens WAV drifted: {slot_id}")
                candidate_pcm = calibration._read_wav(lens_path)
                for observation in slot_observations:
                    scaler, source_centroid, target_centroid = _context_endpoints(
                        observation, observations
                    )
                    neutral_feature, _ = calibration._feature_receipt(
                        neutral_pcm, observation["measurement_interval"]
                    )
                    lens_feature, _ = calibration._feature_receipt(
                        candidate_pcm, observation["measurement_interval"]
                    )
                    classified = _classify_candidate_feature(
                        feature=lens_feature,
                        neutral_feature=neutral_feature,
                        scaler=scaler,
                        source_centroid=source_centroid,
                        target_centroid=target_centroid,
                    )
                    anchor = anchor_evaluations[
                        (slot_id, observation["occurrence_index"])
                    ]
                    occurrence_records.append(
                        {
                            "occurrence_index": observation["occurrence_index"],
                            "selection": {"rung": "v8"},
                            "candidate": classified,
                            "aggregate": aggregate_replicated_anchor_occurrence(
                                natural_seed_records=anchor[
                                    "natural_seed_classifications"
                                ],
                                candidate_record=classified,
                            ),
                        }
                    )
                    selection_records.append({"rung": "v8"})
                candidate_audio_record = {
                    "source": "frozen_v8",
                    "wav_sha256": lens_record["wav_sha256"],
                    "relative_path": str(lens_path.relative_to(Paths().root)),
                }
                candidate_integrity = v8_outcome["verification"]
            elif rung == "adaptive_strength":
                adaptive_outcome = adaptive_by_id[slot_id]
                candidates = adaptive_outcome["strength_candidates"]
                if not all(
                    candidate["verification"]["integrity_pass"]
                    for candidate in candidates
                ):
                    raise RuntimeError(f"adaptive grid integrity drifted: {slot_id}")
                candidate_pcm_by_label: dict[str, np.ndarray] = {}
                for candidate in candidates:
                    record = candidate["audio"]
                    path = ADAPTIVE_DIR / record["relative_path"]
                    if sha256_file(path) != record["wav_sha256"]:
                        raise RuntimeError(f"adaptive grid WAV drifted: {path}")
                    candidate_pcm_by_label[candidate["label"]] = calibration._read_wav(
                        path
                    )
                for observation in slot_observations:
                    scaler, source_centroid, target_centroid = _context_endpoints(
                        observation, observations
                    )
                    neutral_feature, _ = calibration._feature_receipt(
                        neutral_pcm, observation["measurement_interval"]
                    )
                    strength_records = []
                    for candidate in candidates:
                        lens_feature, _ = calibration._feature_receipt(
                            candidate_pcm_by_label[candidate["label"]],
                            observation["measurement_interval"],
                        )
                        classified = _classify_candidate_feature(
                            feature=lens_feature,
                            neutral_feature=neutral_feature,
                            scaler=scaler,
                            source_centroid=source_centroid,
                            target_centroid=target_centroid,
                        )
                        strength_records.append(
                            {
                                "label": candidate["label"],
                                "state_strength": candidate["state_strength"],
                                **classified,
                            }
                        )
                    selected = _select_strength(strength_records)
                    selection_records.append(selected)
                    anchor = anchor_evaluations[
                        (slot_id, observation["occurrence_index"])
                    ]
                    occurrence_records.append(
                        {
                            "occurrence_index": observation["occurrence_index"],
                            "strength_candidates": strength_records,
                            "selection": selected,
                            "candidate": selected,
                            "aggregate": aggregate_replicated_anchor_occurrence(
                                natural_seed_records=anchor[
                                    "natural_seed_classifications"
                                ],
                                candidate_record=selected,
                            ),
                        }
                    )
                if all(selection_records):
                    windows = _occurrence_windows(intervals, neutral_pcm.size)
                    composite = neutral_pcm.copy()
                    for window, selection in zip(
                        windows, selection_records, strict=True
                    ):
                        assert selection is not None
                        source = candidate_pcm_by_label[selection["label"]]
                        start = int(window["start_sample"])
                        end = int(window["end_sample_exclusive"])
                        composite[start:end] = source[start:end]
                    candidate_integrity = _composite_integrity(
                        neutral=neutral_pcm,
                        candidate=composite,
                        windows=windows,
                        intervals=intervals,
                    )
                    stem = _safe_name(slot_id)
                    path = RUN_DIR / "audio" / f"{stem}__spectral-adaptive.wav"
                    candidate_audio_record = _write_wav(path, composite)
                    retained_composites += 1
                    for observation, occurrence in zip(
                        slot_observations, occurrence_records, strict=True
                    ):
                        scaler, source_centroid, target_centroid = _context_endpoints(
                            observation, observations
                        )
                        neutral_feature, _ = calibration._feature_receipt(
                            neutral_pcm, observation["measurement_interval"]
                        )
                        composite_feature, _ = calibration._feature_receipt(
                            composite, observation["measurement_interval"]
                        )
                        remeasured = _classify_candidate_feature(
                            feature=composite_feature,
                            neutral_feature=neutral_feature,
                            scaler=scaler,
                            source_centroid=source_centroid,
                            target_centroid=target_centroid,
                        )
                        if (
                            remeasured["classification"]
                            != occurrence["aggregate"]["classification"]
                        ):
                            raise RuntimeError(
                                "adaptive composite changed selected classification"
                            )
                        occurrence["composite_remeasurement"] = remeasured
            else:  # pragma: no cover - protocol binds the two supported rungs
                raise RuntimeError(f"unsupported failure candidate rung: {rung}")
            slot_aggregate = aggregate_replicated_anchor_cell(
                [row["aggregate"] for row in occurrence_records],
                expected_occurrence_count=len(occurrence_records),
            )
            if candidate_integrity is not None and not candidate_integrity.get(
                "integrity_pass"
            ):
                slot_aggregate = {
                    **slot_aggregate,
                    "classification": "fail",
                    "directional_pass": False,
                    "exact_category_pass": False,
                    "integrity_override": "candidate_integrity_fail",
                }
            candidate_outcomes.append(
                {
                    "cell_id": cell_id,
                    "logical_slot_id": slot_id,
                    "context": slot_observations[0]["context"],
                    "voice_id": slot_observations[0]["voice_id"],
                    "profile_id": slot_observations[0]["profile_id"],
                    "rule_id": slot_observations[0]["rule_id"],
                    "source": slot_observations[0]["source"],
                    "target": slot_observations[0]["target"],
                    "candidate_rung": rung,
                    "candidate_audio": candidate_audio_record,
                    "candidate_integrity": candidate_integrity,
                    "occurrences": occurrence_records,
                    "aggregate": slot_aggregate,
                    "product_enabled": False,
                }
            )
    if (
        len(candidate_outcomes) != EXPECTED_FAILURE_SLOT_COUNT
        or sum(len(row["occurrences"]) for row in candidate_outcomes)
        != EXPECTED_FAILURE_OCCURRENCE_COUNT
    ):
        raise RuntimeError("failure candidate outcome count drifted")
    cell_summaries = []
    for cell_id, rung in sorted(eligible_cells.items()):
        rows = [row for row in candidate_outcomes if row["cell_id"] == cell_id]
        occurrence_records = [
            occurrence["aggregate"] for row in rows for occurrence in row["occurrences"]
        ]
        aggregate = aggregate_replicated_anchor_cell(occurrence_records)
        if not all(row["candidate_integrity"].get("integrity_pass") for row in rows):
            aggregate = {
                **aggregate,
                "classification": "fail",
                "directional_pass": False,
                "exact_category_pass": False,
                "integrity_override": "one_or_more_slot_integrity_fail",
            }
        nasal = "̃" in rows[0]["source"] or "̃" in rows[0]["target"]
        cell_summaries.append(
            {
                "cell_id": cell_id,
                "profile_id": rows[0]["profile_id"],
                "voice_id": rows[0]["voice_id"],
                "rule_id": rows[0]["rule_id"],
                "source": rows[0]["source"],
                "target": rows[0]["target"],
                "candidate_rung": rung,
                "replicated_anchor": aggregate,
                "nasality_gate_required": nasal,
                "automatic_blind_qc_eligible": bool(
                    aggregate["directional_pass"] and not nasal
                ),
                "product_enabled": False,
            }
        )
    classification_counts = dict(
        Counter(row["replicated_anchor"]["classification"] for row in cell_summaries)
    )
    result: dict[str, Any] = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "protocol_version": PROTOCOL_VERSION,
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "classification": "eligible_failure_cells_evaluated_no_product_promotion",
        "production_enabled": False,
        "api_calls_made": 0,
        "new_candidate_decoder_renders_made": 0,
        "anchor_reproduction_decoder_render_count": reproduced_audio_count,
        "anchor_reproduction_pcm_pass_count": reproduced_audio_count,
        "anchor_reproduction_feature_count": reproduced_feature_count,
        "anchor_reproduction_feature_pass_count": reproduced_feature_count,
        "eligible_failure_cell_count": len(cell_summaries),
        "eligible_failure_slot_count": len(candidate_outcomes),
        "eligible_failure_occurrence_count": sum(
            len(row["occurrences"]) for row in candidate_outcomes
        ),
        "candidate_rung_counts": dict(
            Counter(row["candidate_rung"] for row in cell_summaries)
        ),
        "cell_classification_counts": classification_counts,
        "new_context_matched_candidate_pass_count": sum(
            row["replicated_anchor"]["directional_pass"] for row in cell_summaries
        ),
        "new_oral_blind_qc_queue_count": sum(
            row["automatic_blind_qc_eligible"] for row in cell_summaries
        ),
        "nasal_candidate_pass_pending_nasality_count": sum(
            row["replicated_anchor"]["directional_pass"]
            and row["nasality_gate_required"]
            for row in cell_summaries
        ),
        "retained_adaptive_composite_wav_count": retained_composites,
        "adaptive_selection_strength_counts": dict(
            Counter(
                occurrence["selection"]["label"]
                for row in candidate_outcomes
                if row["candidate_rung"] == "adaptive_strength"
                for occurrence in row["occurrences"]
                if occurrence["selection"] is not None
            )
        ),
        "ineligible_failure_cell": protocol["scope"]["ineligible_failure_cell"],
        "parent_bindings": protocol["parent_bindings"],
        "cell_summaries": cell_summaries,
        "outcomes": candidate_outcomes,
        "claim_limits": protocol["claim_limits"],
        "elapsed_s": time.perf_counter() - started,
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
                    "anchor_reproduction_pcm_pass_count",
                    "anchor_reproduction_feature_pass_count",
                    "eligible_failure_cell_count",
                    "eligible_failure_slot_count",
                    "eligible_failure_occurrence_count",
                    "candidate_rung_counts",
                    "cell_classification_counts",
                    "new_context_matched_candidate_pass_count",
                    "new_oral_blind_qc_queue_count",
                    "nasal_candidate_pass_pending_nasality_count",
                    "retained_adaptive_composite_wav_count",
                    "api_calls_made",
                    "new_candidate_decoder_renders_made",
                    "elapsed_s",
                    "record_sha256",
                )
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
