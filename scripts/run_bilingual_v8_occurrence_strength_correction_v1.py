from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any
import wave

import numpy as np

from earshift_bakeoff.bilingual_candidate_runtime import (
    BilingualCandidateRuntime,
    _evaluate_current_context_rule,
    _pcm,
    _pcm_hash,
    _render_current_context_anchors,
    _wav_bytes,
)
from earshift_bakeoff.bilingual_listener_engine import (
    SEGMENT_SPLICE_CONTEXT_SAMPLES,
)
from earshift_bakeoff.bilingual_listener_engine_v8 import BilingualListenerRuntimeV8
from earshift_bakeoff.bilingual_vowel_engine import BilingualVowelRender
from earshift_bakeoff.bilingual_vowel_occurrence_strength import (
    OCCURRENCE_STRENGTH_CANDIDATE_VERSION,
    OccurrenceStrengthSpec,
    render_occurrence_strength_full_lens,
)
from earshift_bakeoff.bilingual_vowel_replicated_anchors import TRAINING_SEEDS
from earshift_bakeoff.config import ROOT, stable_json
from earshift_bakeoff.kokoro_output_domain_splice import (
    boundary_artifact_report,
    output_domain_splice,
)
from earshift_bakeoff.kokoro_typed_diagnostic import localization_report
from earshift_bakeoff.util import atomic_write_json, sha256_file


PROTOCOL_VERSION = "bilingual-v8-occurrence-strength-correction-v1"
PROTOCOL_PATH = ROOT / "rules" / f"{PROTOCOL_VERSION}.json"
RUN_DIR = (
    ROOT
    / "artifacts"
    / "product-matrix"
    / "20260718-bilingual-v8-occurrence-strength-correction-v1"
)
PARENT_RESULT_PATH = (
    ROOT
    / "artifacts"
    / "product-matrix"
    / "20260718-bilingual-v8-composition-unseen-confirmation-v2"
    / "results.json"
)


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object in {path}")
    return value


def _semantic_hash(value: dict[str, Any]) -> str:
    payload = dict(value)
    payload.pop("record_sha256", None)
    return hashlib.sha256(stable_json(payload).encode("utf-8")).hexdigest()


def _validate_protocol(protocol: dict[str, Any]) -> None:
    intervention = protocol.get("intervention", {})
    binding = protocol.get("failed_occurrence_binding", {})
    if (
        protocol.get("schema_version") != 1
        or protocol.get("protocol_version") != PROTOCOL_VERSION
        or protocol.get("status") != "frozen_before_first_candidate_render"
        or protocol.get("api_calls_allowed") != 0
        or protocol.get("production_enabled") is not False
        or intervention.get("candidate_version")
        != OCCURRENCE_STRENGTH_CANDIDATE_VERSION
        or intervention.get("baseline_equivalence_strength") != 1.0
        or tuple(intervention.get("alternative_strength_order", ()))
        != (0.75, 1.25, 0.5, 1.5, 2.0)
        or intervention.get("selection_rule")
        != "first_complete_composition_gate_pass_under_frozen_order"
        or binding
        != {
            "fixture_id": "heart_unseen_continuous",
            "text": "One good cook took books.",
            "rule_id": "enpt.uh_u",
            "rule_occurrence_ordinal_zero_based": 2,
            "global_occurrence_index": 3,
            "word_index": 3,
            "source": "ʊ",
            "target": "u",
            "failure_mechanism": "target_gain_gate_only",
        }
    ):
        raise ValueError("occurrence-strength protocol contract drifted")
    for source in protocol["bindings"]:
        path = ROOT / source["path"]
        if sha256_file(path) != source["sha256"]:
            raise ValueError(f"occurrence-strength binding drifted: {source['path']}")


def _parent_heart(parent: dict[str, Any]) -> dict[str, Any]:
    if (
        sha256_file(PARENT_RESULT_PATH)
        != "6afcd959d5fc95c2668d2232ce3b461db185c83b0f4d238b3969367ba84cf2a3"
        or parent.get("record_sha256")
        != "4ee292c546702f4cb016eb8248f57c97b08fbd997561ebb47c89a45543c72a3b"
    ):
        raise ValueError("frozen unseen-composition parent drifted")
    return next(
        row
        for row in parent["fixtures"]
        if row["fixture_id"] == "heart_unseen_continuous"
    )


def _read_pcm(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as handle:
        if (
            handle.getnchannels() != 1
            or handle.getsampwidth() != 2
            or handle.getframerate() != 24_000
        ):
            raise ValueError("parent WAV contract drifted")
        payload = handle.readframes(handle.getnframes())
    values = np.frombuffer(payload, dtype="<i2").copy()
    if not values.size:
        raise ValueError("parent WAV is empty")
    return values


def _write_bytes_once(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to replace frozen artifact: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".partial", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _audio_receipt(path: Path, pcm: np.ndarray) -> dict[str, Any]:
    wav = _wav_bytes(pcm)
    _write_bytes_once(path, wav)
    return {
        "relative_path": str(path.relative_to(RUN_DIR)),
        "wav_sha256": hashlib.sha256(wav).hexdigest(),
        "pcm_sha256": _pcm_hash(pcm),
        "sample_count": int(pcm.size),
        "duration_s": int(pcm.size) / 24_000,
    }


def _composition_acoustics(
    *,
    cells: tuple[Any, ...],
    render: BilingualVowelRender,
    anchors: Any,
    scaler: dict[str, Any],
) -> dict[str, Any]:
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


def _raw_occurrence_windows(render: BilingualVowelRender) -> dict[int, dict[str, int]]:
    windows: dict[int, dict[str, int]] = {}
    for row in render.alignment["target_occurrences"]:
        if row["segment_type"] != "vowel":
            raise ValueError("occurrence-strength parent is not vowel-only")
        interval = row["measurement_interval"]
        windows[int(row["occurrence_index"])] = {
            "start_sample": max(
                0,
                int(interval["start_sample"]) - SEGMENT_SPLICE_CONTEXT_SAMPLES,
            ),
            "end_sample_exclusive": min(
                render.neutral_pcm.size,
                int(interval["end_sample_exclusive"])
                + SEGMENT_SPLICE_CONTEXT_SAMPLES,
            ),
        }
    ordered = [windows[index] for index in sorted(windows)]
    if any(
        left["end_sample_exclusive"] > right["start_sample"]
        for left, right in zip(ordered, ordered[1:])
    ):
        raise ValueError("failed occurrence does not have an independent splice window")
    baseline = [
        {
            "start_sample": int(row["start_sample"]),
            "end_sample_exclusive": int(row["end_sample_exclusive"]),
        }
        for row in render.splice_windows
    ]
    if ordered != baseline:
        raise ValueError("baseline v8 windows no longer match occurrence windows")
    return windows


def _corrected_render(
    *,
    baseline: BilingualVowelRender,
    alternative_full_lens: np.ndarray,
    strength: float,
    failed_window: dict[str, int],
) -> tuple[BilingualVowelRender, dict[str, Any]]:
    start = failed_window["start_sample"]
    end = failed_window["end_sample_exclusive"]
    composite_full_lens = baseline.full_lens_pcm.copy()
    composite_full_lens[start:end] = alternative_full_lens[start:end]
    lens, weights = output_domain_splice(
        baseline.neutral_pcm, composite_full_lens, baseline.splice_windows
    )
    target_intervals = [
        row["measurement_interval"]
        for row in baseline.alignment["target_occurrences"]
    ]
    started = time.perf_counter()
    localization = localization_report(
        baseline.neutral_pcm, lens, target_intervals
    )
    localization_elapsed_ms = (time.perf_counter() - started) * 1000
    boundary = boundary_artifact_report(
        baseline.neutral_pcm, composite_full_lens, lens, baseline.splice_windows
    )
    arrays = (
        baseline.neutral_pcm,
        baseline.identity_pcm,
        composite_full_lens,
        lens,
    )
    clipped = [
        float(np.mean(np.abs(values.astype(np.int64)) >= 32767))
        for values in arrays
    ]
    equal_nonempty = bool(
        baseline.neutral_pcm.size
        and len({values.size for values in arrays}) == 1
    )
    finite = all(np.isfinite(values.astype(np.float64)).all() for values in arrays)
    outside_exact = bool(
        np.array_equal(
            lens[weights == 0.0], baseline.neutral_pcm[weights == 0.0]
        )
    )
    interior_exact = bool(
        np.any(weights == 1.0)
        and np.array_equal(
            lens[weights == 1.0], composite_full_lens[weights == 1.0]
        )
    )
    integrity_pass = bool(
        np.array_equal(baseline.neutral_pcm, baseline.identity_pcm)
        and equal_nonempty
        and finite
        and all(value < 0.001 for value in clipped)
        and outside_exact
        and interior_exact
        and boundary.get("pass") is True
        and localization.get("pass") is True
        and baseline.verification.prosody_control_pass
    )
    verification = replace(
        baseline.verification,
        neutral_identity_bit_exact=bool(
            np.array_equal(baseline.neutral_pcm, baseline.identity_pcm)
        ),
        equal_nonempty_samples=equal_nonempty,
        finite=finite,
        unclipped=all(value < 0.001 for value in clipped),
        outside_splice_exact_neutral=outside_exact,
        full_weight_interior_exact_lens=interior_exact,
        boundary_metrics_pass=bool(boundary.get("pass")),
        localization_pass=bool(localization.get("pass")),
        localization_fraction=float(
            localization.get("inside_difference_energy_fraction", 0.0)
        ),
        integrity_pass=integrity_pass,
        changed_rules_acoustically_validated=False,
        evidence_status=(
            "integrity_pass_acoustic_validation_pending"
            if integrity_pass
            else "automatic_integrity_failed"
        ),
    )
    outside_failed = np.ones(lens.size, dtype=bool)
    outside_failed[start:end] = False
    diagnostics = {
        "strength": strength,
        "failed_occurrence_window": failed_window,
        "baseline_lens_exact_outside_failed_window": bool(
            np.array_equal(lens[outside_failed], baseline.lens_pcm[outside_failed])
        ),
        "alternative_full_lens_changed_samples_total": int(
            np.count_nonzero(alternative_full_lens != baseline.full_lens_pcm)
        ),
        "alternative_full_lens_changed_samples_retained_window": int(
            np.count_nonzero(
                alternative_full_lens[start:end]
                != baseline.full_lens_pcm[start:end]
            )
        ),
        "localization_runtime_ms": localization_elapsed_ms,
        "boundary": boundary,
        "localization": localization,
        "verification": asdict(verification),
    }
    return (
        replace(
            baseline,
            full_lens_pcm=composite_full_lens,
            lens_pcm=lens,
            verification=verification,
            prosody={
                **baseline.prosody,
                "occurrence_strength_candidate": {
                    "version": OCCURRENCE_STRENGTH_CANDIDATE_VERSION,
                    "strength": strength,
                    "rule_id": "enpt.uh_u",
                    "rule_occurrence_ordinal_zero_based": 2,
                    "global_occurrence_index": 3,
                    "word_index": 3,
                },
            },
        ),
        diagnostics,
    )


def main() -> int:
    if RUN_DIR.exists():
        raise FileExistsError(f"refusing to overwrite frozen run: {RUN_DIR}")
    protocol = _load_object(PROTOCOL_PATH)
    _validate_protocol(protocol)
    parent = _load_object(PARENT_RESULT_PATH)
    heart = _parent_heart(parent)
    binding = protocol["failed_occurrence_binding"]
    runtime = BilingualCandidateRuntime.load(
        heart["profile_id"], heart["voice_id"]
    )
    rule_ids = tuple(sorted(heart["selected_rule_occurrences"]))
    planner = runtime._composition_planner(rule_ids)
    plan = planner.plan(binding["text"])
    cells = tuple(
        runtime.registry.cell(heart["profile_id"], heart["voice_id"], rule_id)
        for rule_id in rule_ids
    )
    if any(cell is None for cell in cells):
        raise ValueError("a frozen composition cell disappeared")
    typed_cells = tuple(cell for cell in cells if cell is not None)
    baseline = BilingualListenerRuntimeV8(
        planner=planner, synthesis=runtime.synthesis
    ).render(binding["text"])
    if not isinstance(baseline, BilingualVowelRender):
        raise ValueError("frozen Heart fixture produced no baseline pair")
    parent_audio_dir = PARENT_RESULT_PATH.parent
    parent_neutral = _read_pcm(parent_audio_dir / heart["audio"]["neutral"]["relative_path"])
    parent_lens = _read_pcm(parent_audio_dir / heart["audio"]["lens"]["relative_path"])
    baseline_binding_pass = bool(
        plan.plan_sha256 == heart["isolated_plan_sha256"]
        and baseline.plan.plan_sha256 == heart["isolated_plan_sha256"]
        and np.array_equal(baseline.neutral_pcm, parent_neutral)
        and np.array_equal(baseline.lens_pcm, parent_lens)
        and _pcm_hash(baseline.neutral_pcm) == heart["audio"]["neutral"]["pcm_sha256"]
        and _pcm_hash(baseline.lens_pcm) == heart["audio"]["lens"]["pcm_sha256"]
    )
    if not baseline_binding_pass:
        raise ValueError("baseline v8 render no longer reproduces the frozen parent")
    windows = _raw_occurrence_windows(baseline)
    failed_window = windows[binding["global_occurrence_index"]]
    anchors = _render_current_context_anchors(
        render=baseline, synthesis=runtime.synthesis
    )
    baseline_acoustic = _composition_acoustics(
        cells=typed_cells,
        render=baseline,
        anchors=anchors,
        scaler=runtime.scaler,
    )
    if stable_json(baseline_acoustic) != stable_json(heart["acoustic"]):
        raise ValueError("shared-anchor baseline acoustic result drifted")
    equivalence_spec = OccurrenceStrengthSpec(
        rule_id=binding["rule_id"],
        rule_occurrence_ordinal=binding["rule_occurrence_ordinal_zero_based"],
        expected_occurrence_index=binding["global_occurrence_index"],
        expected_word_index=binding["word_index"],
        expected_source=binding["source"],
        expected_target=binding["target"],
        strength=1.0,
    )
    equivalence = render_occurrence_strength_full_lens(
        runtime=runtime.synthesis,
        plan=plan,
        specs=(equivalence_spec,),
    )
    equivalence_neutral = _pcm(equivalence.neutral)
    equivalence_identity = _pcm(equivalence.identity)
    equivalence_full_lens = _pcm(equivalence.full_lens)
    baseline_equivalence = {
        "strength": 1.0,
        "neutral_pcm_exact": bool(
            np.array_equal(equivalence_neutral, baseline.neutral_pcm)
        ),
        "identity_pcm_exact": bool(
            np.array_equal(equivalence_identity, baseline.identity_pcm)
        ),
        "full_lens_pcm_exact": bool(
            np.array_equal(equivalence_full_lens, baseline.full_lens_pcm)
        ),
        "predicted_durations_match": bool(
            sum(equivalence.predicted_durations)
            == baseline.alignment["total_alignment_frames"]
        ),
        "decoder_column_strengths": list(equivalence.decoder_column_strengths),
    }
    baseline_equivalence["pass"] = all(
        baseline_equivalence[key]
        for key in (
            "neutral_pcm_exact",
            "identity_pcm_exact",
            "full_lens_pcm_exact",
            "predicted_durations_match",
        )
    )
    if not baseline_equivalence["pass"]:
        raise ValueError("strength-1 candidate is not equivalent to frozen v8")
    attempts: list[dict[str, Any]] = []
    selected_render: BilingualVowelRender | None = None
    selected_strength: float | None = None
    started = time.perf_counter()
    for strength in protocol["intervention"]["alternative_strength_order"]:
        attempt_started = time.perf_counter()
        spec = replace(equivalence_spec, strength=float(strength))
        alternative = render_occurrence_strength_full_lens(
            runtime=runtime.synthesis,
            plan=plan,
            specs=(spec,),
        )
        alternative_neutral = _pcm(alternative.neutral)
        alternative_identity = _pcm(alternative.identity)
        alternative_full_lens = _pcm(alternative.full_lens)
        neutral_control_pass = bool(
            np.array_equal(alternative_neutral, baseline.neutral_pcm)
            and np.array_equal(alternative_identity, baseline.identity_pcm)
            and sum(alternative.predicted_durations)
            == baseline.alignment["total_alignment_frames"]
        )
        if not neutral_control_pass:
            raise ValueError("alternative strength changed the neutral control")
        corrected, diagnostics = _corrected_render(
            baseline=baseline,
            alternative_full_lens=alternative_full_lens,
            strength=float(strength),
            failed_window=failed_window,
        )
        acoustic = _composition_acoustics(
            cells=typed_cells,
            render=corrected,
            anchors=anchors,
            scaler=runtime.scaler,
        )
        passed = bool(
            diagnostics["baseline_lens_exact_outside_failed_window"]
            and corrected.verification.integrity_pass
            and acoustic["pass"]
        )
        attempts.append(
            {
                "strength": float(strength),
                "neutral_control_pass": neutral_control_pass,
                "decoder_column_strengths": list(
                    alternative.decoder_column_strengths
                ),
                "diagnostics": diagnostics,
                "acoustic": acoustic,
                "automatic_pass": passed,
                "elapsed_s": time.perf_counter() - attempt_started,
            }
        )
        if passed:
            selected_render = corrected
            selected_strength = float(strength)
            break
    correction_pass = selected_render is not None
    RUN_DIR.mkdir(parents=True, exist_ok=False)
    selected_audio = None
    if selected_render is not None:
        selected_audio = _audio_receipt(
            RUN_DIR / "audio" / "heart_unseen_continuous__corrected-lens.wav",
            selected_render.lens_pcm,
        )
    result = {
        "schema_version": 1,
        "run_id": RUN_DIR.name,
        "protocol_version": PROTOCOL_VERSION,
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "candidate_version": OCCURRENCE_STRENGTH_CANDIDATE_VERSION,
        "classification": (
            "known_failure_occurrence_correction_pass_eligible_fresh_unseen_confirmation"
            if correction_pass
            else "known_failure_occurrence_correction_failed_preserve_parent_failure"
        ),
        "production_enabled": False,
        "api_calls_made": 0,
        "parent_result_sha256": sha256_file(PARENT_RESULT_PATH),
        "parent_result_record_sha256": parent["record_sha256"],
        "parent_classification_preserved": parent["classification"],
        "parent_automatic_pass_count_preserved": parent["automatic_pass_count"],
        "baseline_binding_pass": baseline_binding_pass,
        "baseline_acoustic": baseline_acoustic,
        "baseline_equivalence": baseline_equivalence,
        "failed_occurrence_binding": binding,
        "failed_occurrence_window": failed_window,
        "attempt_count": len(attempts),
        "attempted_strengths": [row["strength"] for row in attempts],
        "selected_strength": selected_strength,
        "selected_audio": selected_audio,
        "attempts": attempts,
        "shared_anchor_render_count": 2 * len(TRAINING_SEEDS),
        "candidate_render_count": 1 + len(attempts),
        "elapsed_s": time.perf_counter() - started,
        "human_review_generated": False,
        "fresh_unseen_confirmation_required": correction_pass,
        "interpretation_limit": protocol["outcomes"]["interpretation_limit"],
    }
    result["record_sha256"] = _semantic_hash(result)
    atomic_write_json(RUN_DIR / "results.json", result)
    print(
        json.dumps(
            {
                "classification": result["classification"],
                "attempted_strengths": result["attempted_strengths"],
                "selected_strength": result["selected_strength"],
                "api_calls_made": 0,
                "result_sha256": sha256_file(RUN_DIR / "results.json"),
                "record_sha256": result["record_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0 if correction_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
