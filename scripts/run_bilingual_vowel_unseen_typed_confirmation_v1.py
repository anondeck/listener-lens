#!/usr/bin/env python3
from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Sequence
import wave

import numpy as np

from earshift_bakeoff.bilingual_listener_engine import (
    BilingualListenerPlanner,
    load_listener_profiles,
)
from earshift_bakeoff.bilingual_listener_engine_v8 import (
    BilingualListenerPlannerV8,
    BilingualListenerRuntimeV8,
    bilingual_alignment_record_v8,
)
from earshift_bakeoff.bilingual_product_isolation import isolate_listener_profile
from earshift_bakeoff.bilingual_vowel_engine import (
    BilingualVowelPlan,
    BilingualVowelRender,
    _load_pinned_synthesis_voice,
)
from earshift_bakeoff.bilingual_vowel_full_context import (
    BilingualVowelFullContextRuntime,
)
from earshift_bakeoff.bilingual_vowel_replicated_anchors import (
    TRAINING_SEEDS,
    aggregate_replicated_anchor_cell,
    aggregate_replicated_anchor_occurrence,
    render_seeded_natural_conditions,
)
from earshift_bakeoff.bilingual_vowel_state_strength import (
    BilingualVowelStateStrengthRuntime,
)
from earshift_bakeoff.bilingual_vowel_word_context import (
    BilingualVowelWordContextRuntime,
)
from earshift_bakeoff.config import Paths, stable_json
from earshift_bakeoff.kokoro_synthesis import RNG_SEED, SAMPLE_RATE_HZ
from earshift_bakeoff.util import atomic_write_json, sha256_file

import run_bilingual_vowel_replicated_anchor_calibration_v1 as calibration
import run_bilingual_vowel_replicated_anchor_failure_screen_v1 as failure_screen
from run_bilingual_vowel_adaptive_strength_screen_v1 import (
    STRENGTHS,
    _label,
    _occurrence_windows,
)


PROTOCOL_VERSION = "bilingual-vowel-unseen-typed-confirmation-v1"
PROTOCOL_PATH = Paths().root / "rules" / f"{PROTOCOL_VERSION}.json"
RUN_ID = "20260718-bilingual-vowel-unseen-typed-confirmation-v1"
RUN_DIR = Paths().artifacts / "product-matrix" / RUN_ID
MANIFEST_PATH = (
    Paths().artifacts
    / "product-matrix"
    / "20260718-bilingual-vowel-unseen-typed-manifest-v1"
    / "manifest.json"
)
CALIBRATION_RESULT_PATH = calibration.RUN_DIR / "results.json"
FAILURE_RESULT_PATH = failure_screen.RUN_DIR / "results.json"
EXPECTED_CELL_COUNT = 28
EXPECTED_RULE_GROUP_COUNT = 15
EXPECTED_SLOT_COUNT = 84
EXPECTED_OCCURRENCE_COUNT = 112
EXPECTED_NATURAL_DECODER_RENDER_COUNT = 672
EXPECTED_CANDIDATE_RENDER_SET_COUNT = 174
VOICE_ORDER = ("af_heart", "am_michael", "pm_alex", "pf_dora")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _semantic_hash(payload: dict[str, Any]) -> str:
    semantic = dict(payload)
    semantic.pop("record_sha256", None)
    return hashlib.sha256(stable_json(semantic).encode("utf-8")).hexdigest()


def _safe_name(value: str) -> str:
    return "".join(character if character.isalnum() else "-" for character in value)


def _write_wav(path: Path, pcm: np.ndarray) -> dict[str, Any]:
    values = np.asarray(pcm, dtype="<i2").reshape(-1)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with wave.open(str(temporary), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE_HZ)
        handle.writeframes(values.tobytes())
    temporary.replace(path)
    return {
        "relative_path": str(path.relative_to(RUN_DIR)),
        "wav_sha256": sha256_file(path),
        "pcm_sha256": calibration._pcm_hash(values),
        "sample_count": int(values.size),
        "duration_s": values.size / SAMPLE_RATE_HZ,
    }


def _load_protocol(
    manifest: dict[str, Any],
    calibration_result: dict[str, Any],
    failure: dict[str, Any],
) -> dict[str, Any]:
    protocol = _load_json(PROTOCOL_PATH)
    bindings = protocol["parent_bindings"]
    for label, path, result in (
        ("manifest", MANIFEST_PATH, manifest),
        ("calibration", CALIBRATION_RESULT_PATH, calibration_result),
        ("failure", FAILURE_RESULT_PATH, failure),
    ):
        if (
            bindings[f"{label}_sha256"] != sha256_file(path)
            or bindings[f"{label}_record_sha256"] != result["record_sha256"]
        ):
            raise RuntimeError(f"unseen typed confirmation parent drifted: {label}")
    scope = protocol["scope"]
    rendering = protocol["rendering"]
    if (
        protocol["schema_version"] != 1
        or protocol["protocol_version"] != PROTOCOL_VERSION
        or protocol["status"] != "frozen_before_first_unseen_audio_render"
        or protocol["production_enabled"] is not False
        or scope["oral_candidate_cell_count"] != EXPECTED_CELL_COUNT
        or scope["rule_group_count"] != EXPECTED_RULE_GROUP_COUNT
        or scope["logical_slot_count"] != EXPECTED_SLOT_COUNT
        or scope["target_occurrence_count"] != EXPECTED_OCCURRENCE_COUNT
        or rendering["natural_anchor_decoder_render_count"]
        != EXPECTED_NATURAL_DECODER_RENDER_COUNT
        or rendering["candidate_render_set_count"]
        != EXPECTED_CANDIDATE_RENDER_SET_COUNT
        or tuple(rendering["training_seeds_in_order"]) != TRAINING_SEEDS
        or rendering["baseline_seed"] != RNG_SEED
        or tuple(rendering["adaptive_strength_order"]) != STRENGTHS
        or protocol["stopping_rule"]["api_calls_allowed"] != 0
        or protocol["stopping_rule"]["product_promotion_allowed"] is not False
    ):
        raise RuntimeError("unseen typed confirmation protocol drifted")
    for binding in protocol["source_bindings"]:
        if sha256_file(Paths().root / binding["path"]) != binding["sha256"]:
            raise RuntimeError(
                f"unseen typed confirmation source drifted: {binding['path']}"
            )
    return protocol


def _isolated_planner(
    *,
    base: BilingualListenerPlanner,
    profile_id: str,
    voice_id: str,
    rule_id: str,
) -> Any:
    profile = isolate_listener_profile(load_listener_profiles()[profile_id], rule_id)
    return BilingualListenerPlannerV8(
        profile={
            **profile,
            "voice_id": voice_id,
            "voice_registry_version": base.profile["voice_registry_version"],
            "voice_registry_sha256": base.profile["voice_registry_sha256"],
        },
        adapter=base.adapter,
        model_vocab=set(base.model_vocab),
        nonce_checker=base.nonce_checker,
        phone_indexes=base.phone_indexes,
    )


def _runtime(
    *, rung: str, planner: Any, synthesis: Any, strength: float | None = None
) -> Any:
    if rung == "v8":
        return BilingualListenerRuntimeV8(planner=planner, synthesis=synthesis)
    if rung == "word_context":
        return BilingualVowelWordContextRuntime(planner=planner, synthesis=synthesis)
    if rung == "full_context":
        return BilingualVowelFullContextRuntime(planner=planner, synthesis=synthesis)
    if rung == "adaptive_strength" and strength is not None:
        return BilingualVowelStateStrengthRuntime(
            planner=planner, synthesis=synthesis, state_strength=strength
        )
    raise RuntimeError(f"unsupported unseen candidate rung: {rung}")


def _target_rows(alignment: dict[str, Any], rule_id: str) -> list[dict[str, Any]]:
    return [
        row
        for row in alignment["target_occurrences"]
        if row["segment_type"] == "vowel" and row["rule_id"] == rule_id
    ]


def _feature_or_error(
    pcm: np.ndarray, interval: dict[str, Any]
) -> tuple[list[float] | None, dict[str, Any]]:
    try:
        return calibration._feature_receipt(pcm, interval)
    except (ValueError, RuntimeError) as exc:
        return None, {
            "measurement_error": type(exc).__name__,
            "message": str(exc),
            "interval": interval,
        }


def _measurement_failure_record(reason: Any) -> dict[str, Any]:
    return {
        "classification": "measurement_fail",
        "directional_pass": False,
        "exact_category_pass": False,
        "direction_cosine": 0.0,
        "measurement_error": reason,
    }


def _render_slot(
    *, slot: dict[str, Any], planner: Any, synthesis: Any
) -> tuple[dict[str, Any], list[dict[str, Any]], int, int]:
    plan: BilingualVowelPlan = planner.plan(slot["fixture_spec"]["text"])
    if plan.plan_sha256 != slot["plan_sha256"]:
        raise RuntimeError(f"unseen typed plan drifted: {slot['logical_slot_id']}")
    expected_occurrences = slot["fixture_spec"]["expected_target_occurrence_count"]
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
        seed: calibration._natural_pcm(source_render.audio_by_seed[seed])
        for seed in seeds
    }
    target_pcm = {
        seed: calibration._natural_pcm(target_render.audio_by_seed[seed])
        for seed in seeds
    }
    source_alignment = bilingual_alignment_record_v8(
        model=synthesis.model,
        plan=plan,
        durations=source_render.predicted_durations,
        sample_count=source_pcm[RNG_SEED].size,
    )
    target_alignment = bilingual_alignment_record_v8(
        model=synthesis.model,
        plan=plan,
        durations=target_render.predicted_durations,
        sample_count=target_pcm[RNG_SEED].size,
    )
    source_rows = _target_rows(source_alignment, slot["rule_id"])
    target_rows = _target_rows(target_alignment, slot["rule_id"])
    if (
        len(source_rows) != expected_occurrences
        or len(target_rows) != expected_occurrences
    ):
        raise RuntimeError("unseen natural anchor occurrence count drifted")

    rendered_by_label: dict[str, BilingualVowelRender] = {}
    candidate_render_sets = 0
    if slot["candidate_rung"] == "adaptive_strength":
        for strength in STRENGTHS:
            rendered = _runtime(
                rung="adaptive_strength",
                planner=planner,
                synthesis=synthesis,
                strength=strength,
            ).render(slot["fixture_spec"]["text"])
            if not isinstance(rendered, BilingualVowelRender):
                raise RuntimeError("adaptive unseen candidate produced no comparison")
            rendered_by_label[_label(strength)] = rendered
            candidate_render_sets += 1
    else:
        rendered = _runtime(
            rung=slot["candidate_rung"], planner=planner, synthesis=synthesis
        ).render(slot["fixture_spec"]["text"])
        if not isinstance(rendered, BilingualVowelRender):
            raise RuntimeError("unseen candidate produced no comparison")
        rendered_by_label[slot["candidate_rung"]] = rendered
        candidate_render_sets += 1
    first = next(iter(rendered_by_label.values()))
    controlled_rows = _target_rows(first.alignment, slot["rule_id"])
    if len(controlled_rows) != expected_occurrences:
        raise RuntimeError("unseen candidate occurrence count drifted")
    neutral = first.neutral_pcm
    if not np.array_equal(neutral, source_pcm[RNG_SEED]):
        raise RuntimeError("unseen candidate neutral differs from natural baseline")
    for label, rendered in rendered_by_label.items():
        if (
            rendered.plan.plan_sha256 != plan.plan_sha256
            or not np.array_equal(rendered.neutral_pcm, neutral)
            or not rendered.verification.integrity_pass
            or len(_target_rows(rendered.alignment, slot["rule_id"]))
            != expected_occurrences
        ):
            raise RuntimeError(f"unseen candidate integrity drifted: {label}")

    observations: list[dict[str, Any]] = []
    for index, controlled in enumerate(controlled_rows):
        source_features: dict[int, list[float]] = {}
        target_features: dict[int, list[float]] = {}
        source_receipts: dict[str, Any] = {}
        target_receipts: dict[str, Any] = {}
        measurement_errors: list[dict[str, Any]] = []
        for seed in TRAINING_SEEDS:
            feature, receipt = _feature_or_error(
                source_pcm[seed], source_rows[index]["measurement_interval"]
            )
            source_receipts[str(seed)] = receipt
            if feature is None:
                measurement_errors.append({"condition": "natural_source", **receipt})
            else:
                source_features[seed] = feature
            feature, receipt = _feature_or_error(
                target_pcm[seed], target_rows[index]["measurement_interval"]
            )
            target_receipts[str(seed)] = receipt
            if feature is None:
                measurement_errors.append({"condition": "natural_target", **receipt})
            else:
                target_features[seed] = feature
        neutral_feature, neutral_receipt = _feature_or_error(
            neutral, controlled["measurement_interval"]
        )
        if neutral_feature is None:
            measurement_errors.append(
                {"condition": "candidate_neutral", **neutral_receipt}
            )
        candidate_features: dict[str, list[float]] = {}
        candidate_receipts: dict[str, Any] = {}
        for label, rendered in rendered_by_label.items():
            feature, receipt = _feature_or_error(
                rendered.lens_pcm, controlled["measurement_interval"]
            )
            candidate_receipts[label] = receipt
            if feature is None:
                measurement_errors.append(
                    {"condition": f"candidate:{label}", **receipt}
                )
            else:
                candidate_features[label] = feature
        observations.append(
            {
                "cell_id": slot["cell_id"],
                "logical_slot_id": slot["logical_slot_id"],
                "context": slot["context"],
                "voice_id": slot["voice_id"],
                "profile_id": slot["profile_id"],
                "rule_id": slot["rule_id"],
                "source": slot["source"],
                "target": slot["target"],
                "occurrence_index": controlled["occurrence_index"],
                "measurement_interval": controlled["measurement_interval"],
                "source_features": source_features,
                "target_features": target_features,
                "candidate_neutral_feature": None,
                "candidate_lens_feature": None,
                "reference_rung": None,
                "feature_receipts": {
                    "natural_source_by_seed": source_receipts,
                    "natural_target_by_seed": target_receipts,
                    "candidate_neutral": neutral_receipt,
                    "candidate_by_label": candidate_receipts,
                },
                "neutral_feature": neutral_feature,
                "candidate_features": candidate_features,
                "measurement_errors": measurement_errors,
            }
        )
    work = {
        "slot": slot,
        "plan": plan,
        "neutral": neutral,
        "rendered_by_label": rendered_by_label,
        "controlled_rows": controlled_rows,
        "natural_audio_receipts": {
            "source_by_seed": {
                str(seed): calibration._pcm_receipt(source_pcm[seed]) for seed in seeds
            },
            "target_by_seed": {
                str(seed): calibration._pcm_receipt(target_pcm[seed]) for seed in seeds
            },
        },
    }
    natural_decodes = len(seeds) * 2
    return work, observations, natural_decodes, candidate_render_sets


def _select_adaptive(records: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    for desired in ("exact_category_pass", "directional_only_pass"):
        for record in records:
            if record["aggregate"]["classification"] == desired:
                return record
    return None


def _candidate_classification(
    *,
    observation: dict[str, Any],
    all_observations: list[dict[str, Any]],
    feature: list[float],
) -> dict[str, Any]:
    scaler, source_centroid, target_centroid = failure_screen._context_endpoints(
        observation, all_observations
    )
    return failure_screen._classify_candidate_feature(
        feature=feature,
        neutral_feature=observation["neutral_feature"],
        scaler=scaler,
        source_centroid=source_centroid,
        target_centroid=target_centroid,
    )


def _evaluate_slot(
    *,
    work: dict[str, Any],
    observations: list[dict[str, Any]],
    scaler_observations: list[dict[str, Any]],
    anchor_evaluations: dict[tuple[str, int], dict[str, Any]],
) -> dict[str, Any]:
    slot = work["slot"]
    slot_observations = sorted(
        (
            row
            for row in observations
            if row["logical_slot_id"] == slot["logical_slot_id"]
        ),
        key=lambda row: row["occurrence_index"],
    )
    occurrence_records: list[dict[str, Any]] = []
    selected: list[dict[str, Any] | None] = []
    identity_false_positives = 0
    for observation in slot_observations:
        anchor = anchor_evaluations[
            (slot["logical_slot_id"], observation["occurrence_index"])
        ]
        measurement_failed = bool(observation["measurement_errors"])
        identity = (
            _measurement_failure_record(observation["measurement_errors"])
            if measurement_failed
            else _candidate_classification(
                observation=observation,
                all_observations=scaler_observations,
                feature=observation["neutral_feature"],
            )
        )
        identity_false_positives += int(identity["directional_pass"])
        if slot["candidate_rung"] == "adaptive_strength":
            strength_records = []
            for strength in STRENGTHS:
                label = _label(strength)
                candidate = (
                    _measurement_failure_record(observation["measurement_errors"])
                    if measurement_failed
                    else _candidate_classification(
                        observation=observation,
                        all_observations=scaler_observations,
                        feature=observation["candidate_features"][label],
                    )
                )
                aggregate = aggregate_replicated_anchor_occurrence(
                    natural_seed_records=anchor["natural_seed_classifications"],
                    candidate_record=candidate,
                )
                strength_records.append(
                    {
                        "label": label,
                        "state_strength": strength,
                        "candidate": candidate,
                        "aggregate": aggregate,
                    }
                )
            choice = _select_adaptive(strength_records)
            selected.append(choice)
            aggregation_candidate = (
                strength_records[0]["candidate"]
                if choice is None
                else choice["candidate"]
            )
            occurrence_records.append(
                {
                    "occurrence_index": observation["occurrence_index"],
                    "anchor": anchor["aggregate"],
                    "identity_negative_control": identity,
                    "strength_candidates": strength_records,
                    "selection": choice,
                    "candidate": aggregation_candidate,
                    "aggregate": aggregate_replicated_anchor_occurrence(
                        natural_seed_records=anchor["natural_seed_classifications"],
                        candidate_record=aggregation_candidate,
                    ),
                    "unresolved_after_frozen_strength_grid": choice is None,
                }
            )
        else:
            label = slot["candidate_rung"]
            candidate = (
                _measurement_failure_record(observation["measurement_errors"])
                if measurement_failed
                else _candidate_classification(
                    observation=observation,
                    all_observations=scaler_observations,
                    feature=observation["candidate_features"][label],
                )
            )
            aggregate = aggregate_replicated_anchor_occurrence(
                natural_seed_records=anchor["natural_seed_classifications"],
                candidate_record=candidate,
            )
            selected.append({"label": label, "candidate": candidate})
            occurrence_records.append(
                {
                    "occurrence_index": observation["occurrence_index"],
                    "anchor": anchor["aggregate"],
                    "identity_negative_control": identity,
                    "selection": {"rung": label},
                    "candidate": candidate,
                    "aggregate": aggregate,
                }
            )

    neutral = work["neutral"]
    candidate_pcm: np.ndarray | None = None
    candidate_integrity: dict[str, Any]
    if slot["candidate_rung"] == "adaptive_strength":
        intervals = [row["measurement_interval"] for row in work["controlled_rows"]]
        windows = _occurrence_windows(intervals, neutral.size)
        if all(selected):
            candidate_pcm = neutral.copy()
            for window, choice in zip(windows, selected, strict=True):
                assert choice is not None
                source = work["rendered_by_label"][choice["label"]].lens_pcm
                start = int(window["start_sample"])
                end = int(window["end_sample_exclusive"])
                candidate_pcm[start:end] = source[start:end]
            candidate_integrity = failure_screen._composite_integrity(
                neutral=neutral,
                candidate=candidate_pcm,
                windows=windows,
                intervals=intervals,
            )
            for observation, occurrence in zip(
                slot_observations, occurrence_records, strict=True
            ):
                feature, receipt = _feature_or_error(
                    candidate_pcm, observation["measurement_interval"]
                )
                if feature is None:
                    raise RuntimeError(
                        "completed unseen adaptive composite lost a selected feature"
                    )
                remeasured = _candidate_classification(
                    observation=observation,
                    all_observations=scaler_observations,
                    feature=feature,
                )
                reaggregated = aggregate_replicated_anchor_occurrence(
                    natural_seed_records=anchor_evaluations[
                        (slot["logical_slot_id"], observation["occurrence_index"])
                    ]["natural_seed_classifications"],
                    candidate_record=remeasured,
                )
                if (
                    reaggregated["classification"]
                    != occurrence["aggregate"]["classification"]
                ):
                    raise RuntimeError(
                        "unseen adaptive composite changed classification"
                    )
                occurrence["composite_remeasurement"] = {
                    "candidate": remeasured,
                    "aggregate": reaggregated,
                    "feature_receipt": receipt,
                }
        else:
            candidate_integrity = {
                "integrity_pass": False,
                "reason": "one_or_more_occurrences_unresolved",
            }
    else:
        rendered = work["rendered_by_label"][slot["candidate_rung"]]
        candidate_pcm = rendered.lens_pcm
        candidate_integrity = asdict(rendered.verification)

    stem = _safe_name(slot["logical_slot_id"])
    neutral_record = _write_wav(RUN_DIR / "audio" / f"{stem}__neutral.wav", neutral)
    candidate_record = (
        None
        if candidate_pcm is None
        else _write_wav(RUN_DIR / "audio" / f"{stem}__lens.wav", candidate_pcm)
    )
    slot_aggregate = aggregate_replicated_anchor_cell(
        [row["aggregate"] for row in occurrence_records],
        expected_occurrence_count=len(occurrence_records),
    )
    if not candidate_integrity.get("integrity_pass"):
        slot_aggregate = {
            **slot_aggregate,
            "classification": "fail",
            "directional_pass": False,
            "exact_category_pass": False,
            "integrity_override": "candidate_integrity_fail",
        }
    return {
        "logical_slot_id": slot["logical_slot_id"],
        "cell_id": slot["cell_id"],
        "profile_id": slot["profile_id"],
        "voice_id": slot["voice_id"],
        "rule_id": slot["rule_id"],
        "source": slot["source"],
        "target": slot["target"],
        "context": slot["context"],
        "candidate_rung": slot["candidate_rung"],
        "fixture_spec": slot["fixture_spec"],
        "plan_sha256": work["plan"].plan_sha256,
        "natural_audio_receipts": work["natural_audio_receipts"],
        "candidate_integrity": candidate_integrity,
        "identity_negative_control_false_positive_count": identity_false_positives,
        "measurement_excluded_occurrence_count": sum(
            bool(row["measurement_errors"]) for row in slot_observations
        ),
        "audio": {"neutral": neutral_record, "lens": candidate_record},
        "occurrences": occurrence_records,
        "aggregate": slot_aggregate,
        "product_enabled": False,
    }


def main() -> int:
    if RUN_DIR.exists():
        raise RuntimeError(f"refusing to overwrite unseen confirmation: {RUN_DIR}")
    manifest = _load_json(MANIFEST_PATH)
    calibration_result = _load_json(CALIBRATION_RESULT_PATH)
    failure = _load_json(FAILURE_RESULT_PATH)
    protocol = _load_protocol(manifest, calibration_result, failure)
    if (
        manifest["cell_count"] != EXPECTED_CELL_COUNT
        or manifest["rule_group_count"] != EXPECTED_RULE_GROUP_COUNT
        or manifest["logical_slot_count"] != EXPECTED_SLOT_COUNT
        or manifest["expected_occurrence_count"] != EXPECTED_OCCURRENCE_COUNT
    ):
        raise RuntimeError("unseen manifest denominator drifted")
    started = time.perf_counter()
    all_observations: list[dict[str, Any]] = []
    works: list[dict[str, Any]] = []
    natural_decoder_renders = 0
    candidate_render_sets = 0
    base_planners: dict[tuple[str, str], BilingualListenerPlanner] = {}
    isolated_planners: dict[tuple[str, str, str], Any] = {}
    for voice_id in VOICE_ORDER:
        synthesis = _load_pinned_synthesis_voice(voice_id)
        for slot in sorted(
            (row for row in manifest["slots"] if row["voice_id"] == voice_id),
            key=lambda row: row["logical_slot_id"],
        ):
            base_key = (slot["profile_id"], voice_id)
            if base_key not in base_planners:
                base_planners[base_key] = BilingualListenerPlanner.load(
                    slot["profile_id"], voice_id=voice_id
                )
            planner_key = (slot["profile_id"], voice_id, slot["rule_id"])
            if planner_key not in isolated_planners:
                isolated_planners[planner_key] = _isolated_planner(
                    base=base_planners[base_key],
                    profile_id=slot["profile_id"],
                    voice_id=voice_id,
                    rule_id=slot["rule_id"],
                )
            work, observations, natural_count, candidate_count = _render_slot(
                slot=slot,
                planner=isolated_planners[planner_key],
                synthesis=synthesis,
            )
            works.append(work)
            all_observations.extend(observations)
            natural_decoder_renders += natural_count
            candidate_render_sets += candidate_count
    if (
        len(works) != EXPECTED_SLOT_COUNT
        or len(all_observations) != EXPECTED_OCCURRENCE_COUNT
        or natural_decoder_renders != EXPECTED_NATURAL_DECODER_RENDER_COUNT
        or candidate_render_sets != EXPECTED_CANDIDATE_RENDER_SET_COUNT
    ):
        raise RuntimeError("unseen confirmation render denominator drifted")

    incomplete_slot_ids = {
        row["logical_slot_id"] for row in all_observations if row["measurement_errors"]
    }
    scaler_observations = [
        row
        for row in all_observations
        if row["logical_slot_id"] not in incomplete_slot_ids
    ]
    anchor_evaluations: dict[tuple[str, int], dict[str, Any]] = {}
    for observation in all_observations:
        if observation["measurement_errors"]:
            natural = [
                {
                    "heldout_seed": seed,
                    **_measurement_failure_record(observation["measurement_errors"]),
                }
                for seed in TRAINING_SEEDS
            ]
            evaluated = {
                "natural_seed_classifications": natural,
                "aggregate": aggregate_replicated_anchor_occurrence(
                    natural_seed_records=natural,
                    candidate_record=None,
                ),
            }
        else:
            evaluated = calibration._evaluate_observation(
                observation, scaler_observations
            )
        anchor_evaluations[
            (observation["logical_slot_id"], observation["occurrence_index"])
        ] = evaluated
    outcomes = [
        _evaluate_slot(
            work=work,
            observations=all_observations,
            scaler_observations=scaler_observations,
            anchor_evaluations=anchor_evaluations,
        )
        for work in works
    ]
    identity_false_positives = sum(
        row["identity_negative_control_false_positive_count"] for row in outcomes
    )
    cell_summaries = []
    for cell in manifest["candidate_cells"]:
        rows = [row for row in outcomes if row["cell_id"] == cell["cell_id"]]
        occurrences = [
            occurrence["aggregate"] for row in rows for occurrence in row["occurrences"]
        ]
        aggregate = aggregate_replicated_anchor_cell(occurrences)
        if not all(row["candidate_integrity"].get("integrity_pass") for row in rows):
            aggregate = {
                **aggregate,
                "classification": "fail",
                "directional_pass": False,
                "exact_category_pass": False,
                "integrity_override": "one_or_more_slot_integrity_fail",
            }
        if identity_false_positives:
            aggregate = {
                **aggregate,
                "classification": "instrument_sanity_fail",
                "directional_pass": False,
                "exact_category_pass": False,
                "instrument_override": "identity_false_positive",
            }
        cell_summaries.append(
            {
                **cell,
                "replicated_anchor": aggregate,
                "unseen_automatic_pass": bool(aggregate["directional_pass"]),
                "blind_human_qc_eligible": bool(aggregate["directional_pass"]),
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
        "classification": (
            "unseen_typed_confirmation_instrument_sanity_fail"
            if identity_false_positives
            else "unseen_typed_confirmation_complete_no_product_promotion"
        ),
        "production_enabled": False,
        "api_calls_made": 0,
        "natural_anchor_decoder_render_count": natural_decoder_renders,
        "candidate_render_set_count": candidate_render_sets,
        "oral_candidate_cell_count": len(cell_summaries),
        "logical_slot_count": len(outcomes),
        "target_occurrence_count": sum(len(row["occurrences"]) for row in outcomes),
        "identity_negative_control_count": len(all_observations),
        "identity_negative_control_false_positive_count": identity_false_positives,
        "cell_classification_counts": classification_counts,
        "unseen_automatic_pass_count": sum(
            row["unseen_automatic_pass"] for row in cell_summaries
        ),
        "blind_human_qc_queue_count": sum(
            row["blind_human_qc_eligible"] for row in cell_summaries
        ),
        "anchor_valid_cell_count": sum(
            row["replicated_anchor"]["all_anchor_occurrences_valid"]
            for row in cell_summaries
        ),
        "candidate_integrity_slot_pass_count": sum(
            row["candidate_integrity"].get("integrity_pass", False) for row in outcomes
        ),
        "measurement_excluded_slot_count": len(incomplete_slot_ids),
        "measurement_excluded_occurrence_count": sum(
            row["measurement_excluded_occurrence_count"] for row in outcomes
        ),
        "retained_wav_count": sum(
            1
            for row in outcomes
            for record in row["audio"].values()
            if record is not None
        ),
        "adaptive_selection_strength_counts": dict(
            Counter(
                occurrence["selection"]["label"]
                for row in outcomes
                if row["candidate_rung"] == "adaptive_strength"
                for occurrence in row["occurrences"]
                if occurrence["selection"] is not None
            )
        ),
        "parent_bindings": protocol["parent_bindings"],
        "cell_summaries": cell_summaries,
        "outcomes": outcomes,
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
                    "natural_anchor_decoder_render_count",
                    "candidate_render_set_count",
                    "oral_candidate_cell_count",
                    "logical_slot_count",
                    "target_occurrence_count",
                    "identity_negative_control_false_positive_count",
                    "candidate_integrity_slot_pass_count",
                    "measurement_excluded_slot_count",
                    "measurement_excluded_occurrence_count",
                    "anchor_valid_cell_count",
                    "cell_classification_counts",
                    "unseen_automatic_pass_count",
                    "blind_human_qc_queue_count",
                    "retained_wav_count",
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
