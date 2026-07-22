from __future__ import annotations

import base64
from dataclasses import asdict, dataclass
import hashlib
import io
import json
from pathlib import Path
import time
from typing import Any
import wave

import numpy as np

from .bilingual_candidate_registry import (
    BilingualCandidateCell,
    BilingualCandidateRegistry,
    CandidatePlanDecision,
    load_bilingual_candidate_registry,
)
from .bilingual_candidate_isolation import isolate_listener_profile_set
from .bilingual_listener_engine import (
    BilingualListenerPlanner,
    load_listener_profiles,
)
from .bilingual_listener_engine_v8 import (
    BilingualListenerPlannerV8,
    BilingualListenerRuntimeV8,
    bilingual_alignment_record_v8,
)
from .bilingual_product_isolation import (
    active_changed_rule_ids,
    isolate_listener_profile,
)
from .bilingual_vowel_engine import (
    BilingualVowelPlan,
    BilingualVowelRender,
    _load_pinned_synthesis_voice,
)
from .bilingual_vowel_full_context import BilingualVowelFullContextRuntime
from .bilingual_vowel_replicated_anchors import (
    TRAINING_SEEDS,
    aggregate_replicated_anchor_cell,
    aggregate_replicated_anchor_occurrence,
    render_seeded_natural_conditions,
)
from .bilingual_vowel_spectral_category import (
    apply_robust_feature_scaler,
    classify_spectral_endpoint,
    spectral_trajectory_feature,
)
from .bilingual_vowel_state_strength import BilingualVowelStateStrengthRuntime
from .bilingual_vowel_word_context import BilingualVowelWordContextRuntime
from .config import ROOT
from .kokoro_synthesis import SAMPLE_RATE_HZ, pcm16_bytes
from .util import sha256_file


RUNTIME_SCALER_PATH = (
    ROOT
    / "artifacts"
    / "product-matrix"
    / "20260718-bilingual-candidate-runtime-gate-v1"
    / "voice-scalers.json"
)
RUNTIME_GATE_VERSION = "bilingual-candidate-runtime-gate-v1"
ADAPTIVE_RUNTIME_STRENGTH = 1.0


class BilingualCandidateRuntimeError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class BilingualCandidateScopeError(BilingualCandidateRuntimeError):
    pass


class BilingualCandidateAcousticGateError(BilingualCandidateRuntimeError):
    pass


@dataclass(frozen=True)
class BilingualCompositionCandidate:
    source_plan: BilingualVowelPlan
    isolated_plan: BilingualVowelPlan
    cells: tuple[BilingualCandidateCell, ...]
    omitted_rule_ids: tuple[str, ...]
    render: BilingualVowelRender
    acoustic: dict[str, Any]


def _load_scalers(
    registry: BilingualCandidateRegistry,
    path: Path = RUNTIME_SCALER_PATH,
) -> dict[str, dict[str, Any]]:
    if sha256_file(path) != registry.runtime_gate_scaler_sha256:
        raise BilingualCandidateRuntimeError(
            "runtime_scaler_hash_mismatch", "The frozen voice scaler changed."
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BilingualCandidateRuntimeError(
            "runtime_scaler_unavailable", "The frozen voice scaler is unavailable."
        ) from exc
    if (
        not isinstance(payload, dict)
        or payload.get("version") != RUNTIME_GATE_VERSION
        or payload.get("production_enabled") is not False
        or not isinstance(payload.get("voice_scalers"), dict)
    ):
        raise BilingualCandidateRuntimeError(
            "runtime_scaler_contract_mismatch", "The voice scaler contract drifted."
        )
    scalers = payload["voice_scalers"]
    for voice_id, scaler in scalers.items():
        if (
            not isinstance(voice_id, str)
            or scaler.get("feature_size") != 36
            or scaler.get("observation_count", 0) < 2
            or len(scaler.get("center", ())) != 36
            or len(scaler.get("scale", ())) != 36
            or not all(float(value) > 0 for value in scaler["scale"])
        ):
            raise BilingualCandidateRuntimeError(
                "runtime_scaler_contract_mismatch",
                "A voice scaler has invalid dimensions.",
            )
    return scalers


def _pcm(audio: np.ndarray) -> np.ndarray:
    return np.frombuffer(pcm16_bytes(audio), dtype="<i2").copy()


def _pcm_hash(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype="<i2").tobytes()).hexdigest()


def _wav_bytes(values: np.ndarray) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE_HZ)
        handle.writeframes(np.asarray(values, dtype="<i2").tobytes())
    return output.getvalue()


def _audio_record(values: np.ndarray) -> dict[str, Any]:
    pcm = np.asarray(values, dtype="<i2").reshape(-1)
    wav = _wav_bytes(pcm)
    return {
        "mime_type": "audio/wav",
        "base64": base64.b64encode(wav).decode("ascii"),
        "sha256": hashlib.sha256(wav).hexdigest(),
        "pcm_sha256": _pcm_hash(pcm),
        "sample_count": int(pcm.size),
        "duration_s": pcm.size / SAMPLE_RATE_HZ,
    }


def _feature(pcm: np.ndarray, interval: dict[str, Any]) -> list[float]:
    record = spectral_trajectory_feature(
        pcm,
        start_sample=int(interval["start_sample"]),
        end_sample_exclusive=int(interval["end_sample_exclusive"]),
    )
    return list(record["feature"])


def _scaled(feature: list[float], scaler: dict[str, Any]) -> np.ndarray:
    return apply_robust_feature_scaler(feature, scaler)


def _natural_anchor_classifications(
    *,
    source_features: dict[int, list[float]],
    target_features: dict[int, list[float]],
    scaler: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for heldout_seed in TRAINING_SEEDS:
        other = tuple(seed for seed in TRAINING_SEEDS if seed != heldout_seed)
        source_centroid = np.mean(
            np.stack([_scaled(source_features[seed], scaler) for seed in other]),
            axis=0,
        )
        target_centroid = np.mean(
            np.stack([_scaled(target_features[seed], scaler) for seed in other]),
            axis=0,
        )
        rows.append(
            {
                "heldout_seed": heldout_seed,
                **classify_spectral_endpoint(
                    source_anchor=source_centroid,
                    target_anchor=target_centroid,
                    neutral=_scaled(source_features[heldout_seed], scaler),
                    lens=_scaled(target_features[heldout_seed], scaler),
                ),
            }
        )
    return rows


def _candidate_classification(
    *,
    source_features: dict[int, list[float]],
    target_features: dict[int, list[float]],
    neutral_feature: list[float],
    lens_feature: list[float],
    scaler: dict[str, Any],
) -> dict[str, Any]:
    source_centroid = np.mean(
        np.stack(
            [_scaled(source_features[seed], scaler) for seed in TRAINING_SEEDS]
        ),
        axis=0,
    )
    target_centroid = np.mean(
        np.stack(
            [_scaled(target_features[seed], scaler) for seed in TRAINING_SEEDS]
        ),
        axis=0,
    )
    return classify_spectral_endpoint(
        source_anchor=source_centroid,
        target_anchor=target_centroid,
        neutral=_scaled(neutral_feature, scaler),
        lens=_scaled(lens_feature, scaler),
    )


def _target_rows(
    alignment: dict[str, Any], rule_id: str
) -> list[dict[str, Any]]:
    return [
        row
        for row in alignment["target_occurrences"]
        if row["segment_type"] == "vowel" and row["rule_id"] == rule_id
    ]


@dataclass(frozen=True)
class _CurrentContextAnchors:
    source_pcm: dict[int, np.ndarray]
    target_pcm: dict[int, np.ndarray]
    source_alignment: dict[str, Any]
    target_alignment: dict[str, Any]


def _render_current_context_anchors(
    *,
    render: BilingualVowelRender,
    synthesis: Any,
) -> _CurrentContextAnchors:
    plan = render.plan
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
    source_pcm = {
        seed: _pcm(source_render.audio_by_seed[seed]) for seed in TRAINING_SEEDS
    }
    target_pcm = {
        seed: _pcm(target_render.audio_by_seed[seed]) for seed in TRAINING_SEEDS
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
    return _CurrentContextAnchors(
        source_pcm=source_pcm,
        target_pcm=target_pcm,
        source_alignment=source_alignment,
        target_alignment=target_alignment,
    )


def _evaluate_current_context_rule(
    *,
    cell: BilingualCandidateCell,
    render: BilingualVowelRender,
    anchors: _CurrentContextAnchors,
    scaler: dict[str, Any],
) -> dict[str, Any]:
    source_rows = _target_rows(anchors.source_alignment, cell.rule_id)
    target_rows = _target_rows(anchors.target_alignment, cell.rule_id)
    candidate_rows = _target_rows(render.alignment, cell.rule_id)
    if not source_rows or not len(source_rows) == len(target_rows) == len(candidate_rows):
        raise BilingualCandidateAcousticGateError(
            "runtime_acoustic_alignment_drift",
            "Current-context anchor occurrences do not match the candidate.",
        )
    occurrence_results: list[dict[str, Any]] = []
    identity_false_positives = 0
    for index, candidate_row in enumerate(candidate_rows):
        source_features = {
            seed: _feature(
                anchors.source_pcm[seed],
                source_rows[index]["measurement_interval"],
            )
            for seed in TRAINING_SEEDS
        }
        target_features = {
            seed: _feature(
                anchors.target_pcm[seed],
                target_rows[index]["measurement_interval"],
            )
            for seed in TRAINING_SEEDS
        }
        neutral_feature = _feature(
            render.neutral_pcm, candidate_row["measurement_interval"]
        )
        lens_feature = _feature(
            render.lens_pcm, candidate_row["measurement_interval"]
        )
        natural = _natural_anchor_classifications(
            source_features=source_features,
            target_features=target_features,
            scaler=scaler,
        )
        candidate = _candidate_classification(
            source_features=source_features,
            target_features=target_features,
            neutral_feature=neutral_feature,
            lens_feature=lens_feature,
            scaler=scaler,
        )
        identity = _candidate_classification(
            source_features=source_features,
            target_features=target_features,
            neutral_feature=neutral_feature,
            lens_feature=neutral_feature,
            scaler=scaler,
        )
        identity_false_positives += int(identity["directional_pass"])
        aggregate = aggregate_replicated_anchor_occurrence(
            natural_seed_records=natural,
            candidate_record=candidate,
        )
        occurrence_results.append(
            {
                "occurrence_index": candidate_row["occurrence_index"],
                "aggregate": aggregate,
                "candidate": candidate,
                "identity_negative_control_directional": identity[
                    "directional_pass"
                ],
            }
        )
    aggregate = aggregate_replicated_anchor_cell(
        [
            {
                **row["aggregate"],
                "candidate_evaluated": True,
                "classification": row["aggregate"]["classification"],
                "directional_pass": row["aggregate"]["directional_pass"],
                "exact_category_pass": row["aggregate"][
                    "exact_category_pass"
                ],
            }
            for row in occurrence_results
        ],
        expected_occurrence_count=len(occurrence_results),
    )
    passed = bool(
        render.verification.integrity_pass
        and aggregate["directional_pass"]
        and identity_false_positives == 0
    )
    return {
        "version": RUNTIME_GATE_VERSION,
        "rule_id": cell.rule_id,
        "voice_id": cell.voice_id,
        "occurrence_count": len(occurrence_results),
        "natural_decoder_render_count": 2 * len(TRAINING_SEEDS),
        "identity_false_positive_count": identity_false_positives,
        "classification": aggregate["classification"],
        "directional_pass": aggregate["directional_pass"],
        "exact_category_pass": aggregate["exact_category_pass"],
        "integrity_pass": render.verification.integrity_pass,
        "pass": passed,
        "occurrences": occurrence_results,
    }


def evaluate_current_context_acoustics(
    *,
    cell: BilingualCandidateCell,
    render: BilingualVowelRender,
    synthesis: Any,
    scaler: dict[str, Any],
) -> dict[str, Any]:
    anchors = _render_current_context_anchors(render=render, synthesis=synthesis)
    return _evaluate_current_context_rule(
        cell=cell,
        render=render,
        anchors=anchors,
        scaler=scaler,
    )


def evaluate_current_context_composition_acoustics(
    *,
    cells: tuple[BilingualCandidateCell, ...],
    render: BilingualVowelRender,
    synthesis: Any,
    scaler: dict[str, Any],
) -> dict[str, Any]:
    if (
        len(cells) < 2
        or len({cell.rule_id for cell in cells}) != len(cells)
        or len({cell.voice_id for cell in cells}) != 1
        or any(not cell.automatic_pass or cell.candidate_rung != "v8" for cell in cells)
    ):
        raise BilingualCandidateScopeError(
            "unsupported_rule_composition",
            "Composition requires at least two unique passing v8 cells for one voice.",
        )
    anchors = _render_current_context_anchors(render=render, synthesis=synthesis)
    cell_results = [
        _evaluate_current_context_rule(
            cell=cell,
            render=render,
            anchors=anchors,
            scaler=scaler,
        )
        for cell in cells
    ]
    return {
        "version": "bilingual-candidate-v8-composition-gate-v1",
        "voice_id": cells[0].voice_id,
        "rule_count": len(cells),
        "rule_ids": [cell.rule_id for cell in cells],
        "occurrence_count": sum(row["occurrence_count"] for row in cell_results),
        "shared_natural_decoder_render_count": 2 * len(TRAINING_SEEDS),
        "identity_false_positive_count": sum(
            row["identity_false_positive_count"] for row in cell_results
        ),
        "integrity_pass": render.verification.integrity_pass,
        "pass": bool(
            render.verification.integrity_pass
            and all(row["pass"] for row in cell_results)
        ),
        "cells": cell_results,
    }


def _count_rule_occurrences(plan: BilingualVowelPlan, rule_id: str) -> int:
    return sum(
        occurrence.changed and occurrence.rule_id == rule_id
        for word in plan.words
        for occurrence in word.vowel_occurrences
    )


class BilingualCandidateRuntime:
    def __init__(
        self,
        *,
        registry: BilingualCandidateRegistry,
        base_planner: BilingualListenerPlanner,
        synthesis: Any,
        scaler: dict[str, Any],
    ) -> None:
        self.registry = registry
        self.base_planner = base_planner
        self.synthesis = synthesis
        self.scaler = scaler
        self._planners: dict[str, BilingualListenerPlannerV8] = {}
        self._composition_planners: dict[
            tuple[str, ...], BilingualListenerPlannerV8
        ] = {}

    @classmethod
    def load(cls, profile_id: str, voice_id: str) -> BilingualCandidateRuntime:
        registry = load_bilingual_candidate_registry()
        base_planner = BilingualListenerPlanner.load(profile_id, voice_id=voice_id)
        scalers = _load_scalers(registry)
        if voice_id not in scalers:
            raise BilingualCandidateRuntimeError(
                "runtime_voice_scaler_missing", "No runtime scaler exists for the voice."
            )
        return cls(
            registry=registry,
            base_planner=base_planner,
            synthesis=_load_pinned_synthesis_voice(voice_id),
            scaler=scalers[voice_id],
        )

    def _isolated_planner(self, rule_id: str) -> BilingualListenerPlannerV8:
        if rule_id not in self._planners:
            profile = isolate_listener_profile(
                load_listener_profiles()[self.base_planner.profile["id"]], rule_id
            )
            self._planners[rule_id] = BilingualListenerPlannerV8(
                profile={
                    **profile,
                    "voice_id": self.base_planner.profile["voice_id"],
                    "voice_registry_version": self.base_planner.profile[
                        "voice_registry_version"
                    ],
                    "voice_registry_sha256": self.base_planner.profile[
                        "voice_registry_sha256"
                    ],
                },
                adapter=self.base_planner.adapter,
                model_vocab=set(self.base_planner.model_vocab),
                nonce_checker=self.base_planner.nonce_checker,
                phone_indexes=self.base_planner.phone_indexes,
            )
        return self._planners[rule_id]

    def _candidate_runtime(
        self, cell: BilingualCandidateCell, planner: BilingualListenerPlannerV8
    ) -> Any:
        if cell.candidate_rung == "v8":
            return BilingualListenerRuntimeV8(
                planner=planner, synthesis=self.synthesis
            )
        if cell.candidate_rung == "word_context":
            return BilingualVowelWordContextRuntime(
                planner=planner, synthesis=self.synthesis
            )
        if cell.candidate_rung == "full_context":
            return BilingualVowelFullContextRuntime(
                planner=planner, synthesis=self.synthesis
            )
        if cell.candidate_rung == "adaptive_strength":
            return BilingualVowelStateStrengthRuntime(
                planner=planner,
                synthesis=self.synthesis,
                state_strength=ADAPTIVE_RUNTIME_STRENGTH,
            )
        raise BilingualCandidateRuntimeError(
            "unsupported_candidate_rung", "The candidate rung is not implemented."
        )

    def _composition_planner(
        self, rule_ids: tuple[str, ...]
    ) -> BilingualListenerPlannerV8:
        key = tuple(sorted(rule_ids))
        if key not in self._composition_planners:
            profile = isolate_listener_profile_set(
                load_listener_profiles()[self.base_planner.profile["id"]], key
            )
            self._composition_planners[key] = BilingualListenerPlannerV8(
                profile={
                    **profile,
                    "voice_id": self.base_planner.profile["voice_id"],
                    "voice_registry_version": self.base_planner.profile[
                        "voice_registry_version"
                    ],
                    "voice_registry_sha256": self.base_planner.profile[
                        "voice_registry_sha256"
                    ],
                },
                adapter=self.base_planner.adapter,
                model_vocab=set(self.base_planner.model_vocab),
                nonce_checker=self.base_planner.nonce_checker,
                phone_indexes=self.base_planner.phone_indexes,
            )
        return self._composition_planners[key]

    def render_v8_composition_candidate(
        self, text: str
    ) -> BilingualCompositionCandidate:
        source_plan = self.base_planner.plan(text)
        changed_rule_ids = active_changed_rule_ids(source_plan)
        passing_cells = tuple(
            cell
            for rule_id in changed_rule_ids
            if (
                (cell := self.registry.cell(
                    source_plan.profile_id, source_plan.voice_id, rule_id
                ))
                is not None
                and cell.automatic_pass
            )
        )
        if not 2 <= len(passing_cells) <= 3:
            raise BilingualCandidateScopeError(
                "unsupported_rule_composition",
                "The v8 composition spike requires two or three passing rules.",
            )
        if any(cell.candidate_rung != "v8" for cell in passing_cells):
            raise BilingualCandidateScopeError(
                "unsupported_mixed_rung_composition",
                "The first composition spike cannot mix candidate rungs.",
            )
        selected_rule_ids = tuple(sorted(cell.rule_id for cell in passing_cells))
        cells_by_id = {cell.rule_id: cell for cell in passing_cells}
        cells = tuple(cells_by_id[rule_id] for rule_id in selected_rule_ids)
        planner = self._composition_planner(selected_rule_ids)
        isolated_plan = planner.plan(text)
        if active_changed_rule_ids(isolated_plan) != selected_rule_ids:
            raise BilingualCandidateRuntimeError(
                "composition_plan_rule_drift",
                "The composition plan did not preserve its exact selected rules.",
            )
        for cell in cells:
            source_count = _count_rule_occurrences(source_plan, cell.rule_id)
            isolated_count = _count_rule_occurrences(isolated_plan, cell.rule_id)
            if source_count <= 0 or source_count != isolated_count:
                raise BilingualCandidateRuntimeError(
                    "composition_plan_occurrence_drift",
                    "Composition changed a selected rule occurrence count.",
                )
        rendered = BilingualListenerRuntimeV8(
            planner=planner, synthesis=self.synthesis
        ).render(text)
        if not isinstance(rendered, BilingualVowelRender):
            raise BilingualCandidateRuntimeError(
                "composition_render_missing",
                "The composition plan produced no controlled audio pair.",
            )
        acoustic = evaluate_current_context_composition_acoustics(
            cells=cells,
            render=rendered,
            synthesis=self.synthesis,
            scaler=self.scaler,
        )
        return BilingualCompositionCandidate(
            source_plan=source_plan,
            isolated_plan=isolated_plan,
            cells=cells,
            omitted_rule_ids=tuple(
                rule_id
                for rule_id in changed_rule_ids
                if rule_id not in selected_rule_ids
            ),
            render=rendered,
            acoustic=acoustic,
        )

    def contract(self) -> dict[str, Any]:
        return {
            "service_contract_version": self.registry.service_contract_version,
            "candidate_id": self.registry.candidate_id,
            "candidate_state_sha256": self.registry.state_sha256,
            "runtime_gate_result_sha256": self.registry.runtime_gate_result_sha256,
            "runtime_gate_scaler_sha256": self.registry.runtime_gate_scaler_sha256,
            "profile_id": self.base_planner.profile["id"],
            "voice_id": self.base_planner.profile["voice_id"],
            "voice_registry_version": self.base_planner.profile[
                "voice_registry_version"
            ],
            "voice_registry_sha256": self.base_planner.profile[
                "voice_registry_sha256"
            ],
            "production_enabled": False,
            "human_qc_status": "pending",
        }

    def render(self, text: str) -> dict[str, Any]:
        started = time.perf_counter()
        source_plan = self.base_planner.plan(text)
        decision: CandidatePlanDecision = self.registry.evaluate_plan(source_plan)
        if decision.status == "no_supported_sounds":
            return {
                "schema_version": 1,
                "status": "no_supported_sounds",
                "message": (
                    "No independently confirmed oral-vowel rule is available for "
                    "this voice and sentence yet."
                ),
                "candidate_contract": self.contract(),
                "coverage": decision.safe_metadata(),
                "api_calls_made": 0,
            }
        if not decision.render_eligible or decision.cell is None:
            raise BilingualCandidateScopeError(
                decision.status,
                "The typed input has no independently eligible vowel candidate.",
            )
        cell = decision.cell
        planner = self._isolated_planner(cell.rule_id)
        isolated_plan = planner.plan(text)
        if active_changed_rule_ids(isolated_plan) != (cell.rule_id,):
            raise BilingualCandidateRuntimeError(
                "isolated_plan_rule_drift",
                "The candidate render plan did not isolate exactly one rule.",
            )
        source_count = _count_rule_occurrences(source_plan, cell.rule_id)
        isolated_count = _count_rule_occurrences(isolated_plan, cell.rule_id)
        if source_count <= 0 or source_count != isolated_count:
            raise BilingualCandidateRuntimeError(
                "isolated_plan_occurrence_drift",
                "The candidate render plan changed target occurrence count.",
            )
        rendered = self._candidate_runtime(cell, planner).render(text)
        if not isinstance(rendered, BilingualVowelRender):
            raise BilingualCandidateRuntimeError(
                "candidate_render_missing", "The eligible plan produced no audio pair."
            )
        acoustic = evaluate_current_context_acoustics(
            cell=cell,
            render=rendered,
            synthesis=self.synthesis,
            scaler=self.scaler,
        )
        if not acoustic["pass"]:
            raise BilingualCandidateAcousticGateError(
                "runtime_acoustic_gate_rejected",
                "The current typed context failed its acoustic or integrity gate.",
            )
        return {
            "schema_version": 1,
            "status": "ready_pending_human_qc",
            "claim_tier": "runtime_acoustic_pass_human_qc_pending",
            "candidate_contract": self.contract(),
            "transform": {
                "schema_version": 1,
                "profile_id": isolated_plan.profile_id,
                "voice_id": isolated_plan.voice_id,
                "original_text": isolated_plan.normalized_text,
                "neutral_script": isolated_plan.neutral_script,
                "lens_script": isolated_plan.lens_script,
                "comparison_available": True,
                "plan_sha256": isolated_plan.plan_sha256,
                "applied_rules": [
                    {
                        "rule_id": cell.rule_id,
                        "source_ipa": cell.source,
                        "target_ipa": cell.target,
                        "occurrences": isolated_count,
                    }
                ],
                "omitted_rule_ids": list(decision.omitted_rule_ids),
                "partial_profile_coverage": bool(decision.omitted_rule_ids),
            },
            "audio": {
                "neutral": _audio_record(rendered.neutral_pcm),
                "lens": _audio_record(rendered.lens_pcm),
            },
            "verification": {
                "status": "runtime_acoustic_gates_passed",
                "plan_sha256": isolated_plan.plan_sha256,
                "target_occurrence_count": isolated_count,
                "neutral_pcm_sha256": _pcm_hash(rendered.neutral_pcm),
                "identity_pcm_sha256": _pcm_hash(rendered.identity_pcm),
                "lens_pcm_sha256": _pcm_hash(rendered.lens_pcm),
                "render_integrity": asdict(rendered.verification),
                "acoustic": acoustic,
                "elapsed_ms": (time.perf_counter() - started) * 1_000.0,
                "api_calls_made": 0,
            },
            "cache_hit": False,
            "api_calls_made": 0,
        }
