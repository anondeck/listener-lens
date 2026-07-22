from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any

import numpy as np

from .bilingual_candidate_runtime import (
    BilingualCandidateRuntime,
    BilingualCandidateRuntimeError,
    BilingualCandidateScopeError,
    BilingualCompositionCandidate,
    _count_rule_occurrences,
    evaluate_current_context_composition_acoustics,
)
from .bilingual_listener_engine import SEGMENT_SPLICE_CONTEXT_SAMPLES
from .bilingual_listener_engine_v8 import (
    BilingualListenerRuntimeV8,
    bilingual_alignment_record_v8,
)
from .bilingual_product_isolation import active_changed_rule_ids
from .bilingual_v8_factorized_composition import _validate_factorized_plans
from .bilingual_vowel_engine import (
    BilingualRenderVerification,
    BilingualVowelEngineError,
    BilingualVowelPlan,
    BilingualVowelRender,
)
from .kokoro_synthesis import (
    RNG_SEED,
    KokoroSynthesisError,
    _f0_noise,
    _INFERENCE_LOCK,
    _input_ids,
    _predicted_alignment,
    _text_features,
    pcm16_bytes,
)


SEPARATED_DECODE_COMPOSITION_VERSION = "v8-rule-separated-decode-composition-v3"


@dataclass(frozen=True)
class RuleSeparatedControlledRender:
    neutral: np.ndarray
    identity: np.ndarray
    full_lens_by_rule: dict[str, np.ndarray]
    predicted_durations: tuple[int, ...]
    rule_decoder_columns: dict[str, tuple[int, ...]]


def _pcm(audio: np.ndarray) -> np.ndarray:
    return np.frombuffer(pcm16_bytes(audio), dtype="<i2").copy()


def _pcm_hash(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype="<i2").tobytes()).hexdigest()


def render_rule_separated_triplet(
    *,
    synthesis: Any,
    combined_plan: BilingualVowelPlan,
    rule_ids: tuple[str, ...],
) -> RuleSeparatedControlledRender:
    state_plans = _validate_factorized_plans(
        synthesis=synthesis,
        combined_plan=combined_plan,
        rule_ids=rule_ids,
    )
    pair = combined_plan.pair_plan()
    if pair is None:
        raise BilingualVowelEngineError(
            "separated_decode_pair_missing", "The combined plan has no changed pair."
        )
    torch = synthesis.torch
    with _INFERENCE_LOCK, torch.no_grad():
        ref_s = synthesis._reference_style(pair.source_phonemes)
        source_features = _text_features(
            synthesis.model,
            _input_ids(synthesis.model, pair.source_phonemes, torch),
            ref_s,
            torch,
        )
        predicted_durations, alignment = _predicted_alignment(
            synthesis.model, source_features, pair.speed, torch
        )
        neutral_features = _text_features(
            synthesis.model,
            _input_ids(synthesis.model, pair.neutral_phonemes, torch),
            ref_s,
            torch,
        )
        predicted_f0, noise = _f0_noise(
            synthesis.model, neutral_features, alignment, torch
        )
        neutral_state = neutral_features["t_en"]
        torch.manual_seed(RNG_SEED)
        neutral = synthesis._decode(
            neutral_state, alignment, predicted_f0, noise, ref_s
        )
        torch.manual_seed(RNG_SEED)
        identity = synthesis._decode(
            neutral_state, alignment, predicted_f0, noise, ref_s
        )
        full_lens_by_rule: dict[str, np.ndarray] = {}
        for rule_id in sorted(state_plans):
            state_plan = state_plans[rule_id]
            lens_features = _text_features(
                synthesis.model,
                _input_ids(synthesis.model, state_plan.lens_phonemes, torch),
                ref_s,
                torch,
            )
            if lens_features["t_en"].shape != neutral_state.shape:
                raise KokoroSynthesisError(
                    "a separated lens state changed the model-token count"
                )
            lens_state = neutral_state.clone()
            columns = list(state_plan.decoder_columns)
            lens_state[:, :, columns] = lens_features["t_en"][:, :, columns]
            torch.manual_seed(RNG_SEED)
            full_lens_by_rule[rule_id] = synthesis._decode(
                lens_state, alignment, predicted_f0, noise, ref_s
            )
    if neutral.shape != identity.shape or any(
        values.shape != neutral.shape for values in full_lens_by_rule.values()
    ):
        raise KokoroSynthesisError("separated listener outputs have unequal samples")
    if pcm16_bytes(neutral) != pcm16_bytes(identity):
        raise KokoroSynthesisError("separated listener identity is not bit-exact")
    return RuleSeparatedControlledRender(
        neutral=neutral,
        identity=identity,
        full_lens_by_rule=full_lens_by_rule,
        predicted_durations=tuple(
            int(value) for value in predicted_durations.cpu().tolist()
        ),
        rule_decoder_columns={
            rule_id: state_plan.decoder_columns
            for rule_id, state_plan in state_plans.items()
        },
    )


def render_rule_separated_v8(
    *,
    synthesis: Any,
    combined_plan: BilingualVowelPlan,
    rule_ids: tuple[str, ...],
) -> BilingualVowelRender:
    from .kokoro_output_domain_splice import (
        boundary_artifact_report,
        output_domain_splice,
    )
    from .kokoro_typed_diagnostic import localization_report

    controlled = render_rule_separated_triplet(
        synthesis=synthesis,
        combined_plan=combined_plan,
        rule_ids=rule_ids,
    )
    neutral = _pcm(controlled.neutral)
    identity = _pcm(controlled.identity)
    full_lens_by_rule = {
        rule_id: _pcm(values)
        for rule_id, values in controlled.full_lens_by_rule.items()
    }
    alignment = bilingual_alignment_record_v8(
        model=synthesis.model,
        plan=combined_plan,
        durations=controlled.predicted_durations,
        sample_count=neutral.size,
    )
    if any(row["segment_type"] != "vowel" for row in alignment["target_occurrences"]):
        raise BilingualVowelEngineError(
            "separated_decode_nonvowel_rule",
            "The v3 correction received a non-vowel target occurrence.",
        )
    target_intervals = [
        row["measurement_interval"] for row in alignment["target_occurrences"]
    ]
    lens = neutral.copy()
    full_lens = neutral.copy()
    combined_weights = np.zeros(neutral.size, dtype=np.float64)
    all_windows: list[dict[str, Any]] = []
    per_rule: dict[str, dict[str, Any]] = {}
    for rule_id in sorted(rule_ids):
        intervals = [
            row["measurement_interval"]
            for row in alignment["target_occurrences"]
            if row["rule_id"] == rule_id
        ]
        if not intervals:
            raise BilingualVowelEngineError(
                "separated_decode_rule_missing",
                "A separated rule has no aligned target occurrence.",
            )
        candidate_windows = [
            {
                "start_sample": int(interval["start_sample"])
                - SEGMENT_SPLICE_CONTEXT_SAMPLES,
                "end_sample_exclusive": int(interval["end_sample_exclusive"])
                + SEGMENT_SPLICE_CONTEXT_SAMPLES,
            }
            for interval in intervals
        ]
        windows = BilingualListenerRuntimeV8._merge_windows(
            candidate_windows, neutral.size
        )
        rule_full_lens = full_lens_by_rule[rule_id]
        rule_lens, rule_weights = output_domain_splice(
            neutral, rule_full_lens, windows
        )
        active = rule_weights > 0.0
        if np.any(active & (combined_weights > 0.0)):
            raise BilingualVowelEngineError(
                "separated_decode_window_overlap",
                "Two rule-specific output windows overlap.",
            )
        full_lens[active] = rule_full_lens[active]
        lens[active] = rule_lens[active]
        combined_weights[active] = rule_weights[active]
        all_windows.extend(windows)
        rule_boundary = boundary_artifact_report(
            neutral, rule_full_lens, rule_lens, windows
        )
        per_rule[rule_id] = {
            "decoder_columns": list(controlled.rule_decoder_columns[rule_id]),
            "full_lens_pcm_sha256": _pcm_hash(rule_full_lens),
            "windows": list(windows),
            "boundary_metrics_pass": bool(rule_boundary.get("pass")),
        }
    windows = tuple(sorted(all_windows, key=lambda row: int(row["start_sample"])))
    boundary = boundary_artifact_report(neutral, full_lens, lens, windows)
    localization = localization_report(neutral, lens, target_intervals)
    arrays = (neutral, identity, full_lens, lens, *full_lens_by_rule.values())
    clipped = [
        float(np.mean(np.abs(values.astype(np.int64)) >= 32767))
        for values in arrays
    ]
    equal_nonempty = bool(
        neutral.size
        and neutral.size == identity.size == full_lens.size == lens.size
        and all(values.size == neutral.size for values in full_lens_by_rule.values())
    )
    finite = all(np.isfinite(values.astype(np.float64)).all() for values in arrays)
    outside_exact = bool(
        np.array_equal(
            lens[combined_weights == 0.0], neutral[combined_weights == 0.0]
        )
    )
    interior_exact = bool(
        np.any(combined_weights == 1.0)
        and np.array_equal(
            lens[combined_weights == 1.0], full_lens[combined_weights == 1.0]
        )
    )
    per_rule_boundary_pass = all(
        row["boundary_metrics_pass"] for row in per_rule.values()
    )
    integrity_pass = bool(
        np.array_equal(neutral, identity)
        and equal_nonempty
        and finite
        and all(value < 0.001 for value in clipped)
        and outside_exact
        and interior_exact
        and per_rule_boundary_pass
        and boundary.get("pass") is True
        and localization.get("pass") is True
    )
    verification = BilingualRenderVerification(
        neutral_identity_bit_exact=bool(np.array_equal(neutral, identity)),
        equal_nonempty_samples=equal_nonempty,
        finite=finite,
        unclipped=all(value < 0.001 for value in clipped),
        outside_splice_exact_neutral=outside_exact,
        full_weight_interior_exact_lens=interior_exact,
        boundary_metrics_pass=bool(per_rule_boundary_pass and boundary.get("pass")),
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
    return BilingualVowelRender(
        plan=combined_plan,
        neutral_pcm=neutral,
        identity_pcm=identity,
        full_lens_pcm=full_lens,
        lens_pcm=lens,
        alignment=alignment,
        lens_alignment=alignment,
        splice_windows=windows,
        verification=verification,
        prosody={
            "version": SEPARATED_DECODE_COMPOSITION_VERSION,
            "per_rule": per_rule,
            "neutral_alignment_frames": sum(controlled.predicted_durations),
            "lens_alignment_frames": sum(controlled.predicted_durations),
            "candidate_decoder_render_count": 2 + len(rule_ids),
            "input_not_sent_to_openai": True,
        },
    )


def render_separated_decode_composition_candidate(
    runtime: BilingualCandidateRuntime, text: str
) -> BilingualCompositionCandidate:
    source_plan = runtime.base_planner.plan(text)
    changed_rule_ids = active_changed_rule_ids(source_plan)
    passing_cells = tuple(
        cell
        for rule_id in changed_rule_ids
        if (
            (cell := runtime.registry.cell(
                source_plan.profile_id, source_plan.voice_id, rule_id
            ))
            is not None
            and cell.automatic_pass
        )
    )
    if not 2 <= len(passing_cells) <= 3:
        raise BilingualCandidateScopeError(
            "unsupported_rule_composition",
            "The separated-decode spike requires two or three passing rules.",
        )
    if any(cell.candidate_rung != "v8" for cell in passing_cells):
        raise BilingualCandidateScopeError(
            "unsupported_mixed_rung_composition",
            "The separated-decode correction cannot mix candidate rungs.",
        )
    selected_rule_ids = tuple(sorted(cell.rule_id for cell in passing_cells))
    cells_by_id = {cell.rule_id: cell for cell in passing_cells}
    cells = tuple(cells_by_id[rule_id] for rule_id in selected_rule_ids)
    combined_plan = runtime._composition_planner(selected_rule_ids).plan(text)
    if active_changed_rule_ids(combined_plan) != selected_rule_ids:
        raise BilingualCandidateRuntimeError(
            "composition_plan_rule_drift",
            "The separated combined plan lost its exact selected rules.",
        )
    for cell in cells:
        counts = (
            _count_rule_occurrences(source_plan, cell.rule_id),
            _count_rule_occurrences(combined_plan, cell.rule_id),
        )
        if counts[0] <= 0 or len(set(counts)) != 1:
            raise BilingualCandidateRuntimeError(
                "composition_plan_occurrence_drift",
                "Separated decoding changed a selected rule occurrence count.",
            )
    rendered = render_rule_separated_v8(
        synthesis=runtime.synthesis,
        combined_plan=combined_plan,
        rule_ids=selected_rule_ids,
    )
    acoustic = evaluate_current_context_composition_acoustics(
        cells=cells,
        render=rendered,
        synthesis=runtime.synthesis,
        scaler=runtime.scaler,
    )
    return BilingualCompositionCandidate(
        source_plan=source_plan,
        isolated_plan=combined_plan,
        cells=cells,
        omitted_rule_ids=tuple(
            rule_id
            for rule_id in changed_rule_ids
            if rule_id not in selected_rule_ids
        ),
        render=rendered,
        acoustic=acoustic,
    )
