from __future__ import annotations

from dataclasses import dataclass
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
from .bilingual_vowel_engine import (
    BilingualRenderVerification,
    BilingualVowelEngineError,
    BilingualVowelPlan,
    BilingualVowelRender,
)
from .controlled_listener_synthesis import _validate_listener_plan
from .controlled_vowel_synthesis_v2 import vowel_stress_context_columns
from .kokoro_synthesis import (
    RNG_SEED,
    KokoroSynthesisError,
    PairPlan,
    _f0_noise,
    _filtered_symbols,
    _INFERENCE_LOCK,
    _input_ids,
    _predicted_alignment,
    _text_features,
    _word_column_spans,
    pcm16_bytes,
)


FACTORIZED_COMPOSITION_VERSION = "v8-rule-factorized-state-composition-v2"


@dataclass(frozen=True)
class FactorizedControlledRender:
    neutral: np.ndarray
    identity: np.ndarray
    full_lens: np.ndarray
    predicted_durations: tuple[int, ...]
    replaced_columns: tuple[int, ...]
    rule_decoder_columns: dict[str, tuple[int, ...]]


@dataclass(frozen=True)
class FactorizedRuleStatePlan:
    lens_phonemes: str
    target_word_indexes: tuple[int, ...]
    decoder_columns: tuple[int, ...]


def _pcm(audio: np.ndarray) -> np.ndarray:
    return np.frombuffer(pcm16_bytes(audio), dtype="<i2").copy()


def _rule_vowel_model_columns(
    model: Any, plan: BilingualVowelPlan, rule_id: str
) -> tuple[int, ...]:
    spans = _word_column_spans(model, plan.neutral_phonemes)
    if len(spans) != len(plan.words):
        raise BilingualVowelEngineError(
            "word_alignment_drift", "Factorized vowel columns lost word alignment."
        )
    columns: list[int] = []
    for word, word_columns in zip(plan.words, spans, strict=True):
        if len(word_columns) != len(word.neutral_phone):
            raise BilingualVowelEngineError(
                "word_column_drift",
                "Factorized vowel columns differ from the phone plan.",
            )
        for occurrence in word.vowel_occurrences:
            if not occurrence.changed or occurrence.rule_id != rule_id:
                continue
            start = occurrence.phone_offset
            stop = start + occurrence.phone_length
            if (
                word.neutral_phone[start:stop] != occurrence.source
                or word.lens_phone[start:stop] != occurrence.target
            ):
                raise BilingualVowelEngineError(
                    "segment_alignment_drift",
                    "A factorized vowel occurrence no longer matches its columns.",
                )
            columns.extend(word_columns[start:stop])
    return tuple(sorted(set(columns)))


def _rule_state_plan(
    synthesis: Any, combined_plan: BilingualVowelPlan, rule_id: str
) -> FactorizedRuleStatePlan:
    combined_pair = combined_plan.pair_plan()
    if combined_pair is None:
        raise BilingualVowelEngineError(
            "factorized_pair_missing", "The combined plan has no changed pair."
        )
    neutral_symbols = list(
        _filtered_symbols(synthesis.model, combined_pair.neutral_phonemes)
    )
    combined_lens_symbols = _filtered_symbols(
        synthesis.model, combined_pair.lens_phonemes
    )
    vowel_columns = _rule_vowel_model_columns(
        synthesis.model, combined_plan, rule_id
    )
    if not vowel_columns:
        raise BilingualVowelEngineError(
            "factorized_atomic_rule_drift",
            "A selected factorized rule has no changed vowel occurrence.",
        )
    target_word_indexes = tuple(
        word.word_index
        for word in combined_plan.words
        if any(
            occurrence.changed and occurrence.rule_id == rule_id
            for occurrence in word.vowel_occurrences
        )
    )
    for column in vowel_columns:
        neutral_symbols[column - 1] = combined_lens_symbols[column - 1]
    lens_phonemes = "".join(neutral_symbols)
    pair = PairPlan(
        source_phonemes=combined_pair.source_phonemes,
        neutral_phonemes=combined_pair.neutral_phonemes,
        lens_phonemes=lens_phonemes,
        target_word_indexes=target_word_indexes,
        speed=combined_pair.speed,
    )
    target_word_columns = _validate_listener_plan(
        synthesis, pair, allow_prosody_only=False
    )
    state_columns = vowel_stress_context_columns(
        tuple(_filtered_symbols(synthesis.model, pair.neutral_phonemes)),
        tuple(_filtered_symbols(synthesis.model, pair.lens_phonemes)),
        vowel_columns,
        target_word_columns,
    )
    return FactorizedRuleStatePlan(
        lens_phonemes=lens_phonemes,
        target_word_indexes=target_word_indexes,
        decoder_columns=tuple(
            sorted(set(target_word_columns).union(state_columns))
        ),
    )


def _validate_factorized_plans(
    *,
    synthesis: Any,
    combined_plan: BilingualVowelPlan,
    rule_ids: tuple[str, ...],
) -> dict[str, FactorizedRuleStatePlan]:
    combined_rules = active_changed_rule_ids(combined_plan)
    if tuple(sorted(rule_ids)) != combined_rules:
        raise BilingualVowelEngineError(
            "factorized_rule_set_drift",
            "Factorized plans do not cover the exact combined rule set.",
        )
    combined_pair = combined_plan.pair_plan()
    if combined_pair is None:
        raise BilingualVowelEngineError(
            "factorized_pair_missing", "The combined plan has no changed pair."
        )
    if (
        combined_plan.active_prosody_rule_ids
        or combined_plan.coverage.changed_consonant_occurrences
        or combined_plan.coverage.changed_insertion_occurrences
        or combined_plan.coverage.changed_prosody_occurrences
    ):
        raise BilingualVowelEngineError(
            "factorized_nonvowel_rule",
            "The v2 correction supports oral-vowel state composition only.",
        )

    state_plans: dict[str, FactorizedRuleStatePlan] = {}
    occupied: set[int] = set()
    for rule_id in sorted(rule_ids):
        state_plan = _rule_state_plan(synthesis, combined_plan, rule_id)
        columns = state_plan.decoder_columns
        overlap = occupied.intersection(columns)
        if overlap:
            raise BilingualVowelEngineError(
                "factorized_column_overlap",
                "Two factorized rules require the same decoder-state column.",
            )
        occupied.update(columns)
        state_plans[rule_id] = state_plan

    combined_target_word_columns = _validate_listener_plan(
        synthesis, combined_pair, allow_prosody_only=False
    )
    combined_vowel_columns = tuple(
        sorted(
            column
            for rule_id in rule_ids
            for column in _rule_vowel_model_columns(
                synthesis.model, combined_plan, rule_id
            )
        )
    )
    combined_state_columns = vowel_stress_context_columns(
        tuple(_filtered_symbols(synthesis.model, combined_pair.neutral_phonemes)),
        tuple(_filtered_symbols(synthesis.model, combined_pair.lens_phonemes)),
        combined_vowel_columns,
        combined_target_word_columns,
    )
    combined_columns = set(combined_target_word_columns).union(
        combined_state_columns
    )
    if occupied != combined_columns:
        raise BilingualVowelEngineError(
            "factorized_column_union_drift",
            "Atomic rule columns do not reconstruct the combined intervention.",
        )
    return state_plans


def render_rule_factorized_triplet(
    *,
    synthesis: Any,
    combined_plan: BilingualVowelPlan,
    rule_ids: tuple[str, ...],
) -> FactorizedControlledRender:
    state_plans = _validate_factorized_plans(
        synthesis=synthesis,
        combined_plan=combined_plan,
        rule_ids=rule_ids,
    )
    pair = combined_plan.pair_plan()
    if pair is None:
        raise BilingualVowelEngineError(
            "factorized_pair_missing", "The combined plan has no changed pair."
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
        lens_state = neutral_state.clone()
        for rule_id in sorted(state_plans):
            lens_phonemes = state_plans[rule_id].lens_phonemes
            lens_features = _text_features(
                synthesis.model,
                _input_ids(synthesis.model, lens_phonemes, torch),
                ref_s,
                torch,
            )
            if lens_features["t_en"].shape != neutral_state.shape:
                raise KokoroSynthesisError(
                    "a factorized lens state changed the model-token count"
                )
            columns = list(state_plans[rule_id].decoder_columns)
            lens_state[:, :, columns] = lens_features["t_en"][:, :, columns]

        torch.manual_seed(RNG_SEED)
        neutral = synthesis._decode(
            neutral_state, alignment, predicted_f0, noise, ref_s
        )
        torch.manual_seed(RNG_SEED)
        identity = synthesis._decode(
            neutral_state, alignment, predicted_f0, noise, ref_s
        )
        torch.manual_seed(RNG_SEED)
        full_lens = synthesis._decode(
            lens_state, alignment, predicted_f0, noise, ref_s
        )
    if neutral.shape != identity.shape or neutral.shape != full_lens.shape:
        raise KokoroSynthesisError("factorized listener pair has unequal samples")
    if pcm16_bytes(neutral) != pcm16_bytes(identity):
        raise KokoroSynthesisError("factorized listener identity is not bit-exact")
    return FactorizedControlledRender(
        neutral=neutral,
        identity=identity,
        full_lens=full_lens,
        predicted_durations=tuple(
            int(value) for value in predicted_durations.cpu().tolist()
        ),
        replaced_columns=tuple(
            sorted(
                column
                for state_plan in state_plans.values()
                for column in state_plan.decoder_columns
            )
        ),
        rule_decoder_columns={
            rule_id: state_plan.decoder_columns
            for rule_id, state_plan in state_plans.items()
        },
    )


def render_rule_factorized_v8(
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

    controlled = render_rule_factorized_triplet(
        synthesis=synthesis,
        combined_plan=combined_plan,
        rule_ids=rule_ids,
    )
    neutral = _pcm(controlled.neutral)
    identity = _pcm(controlled.identity)
    full_lens = _pcm(controlled.full_lens)
    alignment = bilingual_alignment_record_v8(
        model=synthesis.model,
        plan=combined_plan,
        durations=controlled.predicted_durations,
        sample_count=neutral.size,
    )
    if any(row["segment_type"] != "vowel" for row in alignment["target_occurrences"]):
        raise BilingualVowelEngineError(
            "factorized_nonvowel_rule",
            "The v2 correction received a non-vowel target occurrence.",
        )
    target_intervals = [
        row["measurement_interval"] for row in alignment["target_occurrences"]
    ]
    candidate_windows = [
        {
            "start_sample": int(interval["start_sample"])
            - SEGMENT_SPLICE_CONTEXT_SAMPLES,
            "end_sample_exclusive": int(interval["end_sample_exclusive"])
            + SEGMENT_SPLICE_CONTEXT_SAMPLES,
        }
        for interval in target_intervals
    ]
    windows = BilingualListenerRuntimeV8._merge_windows(
        candidate_windows, neutral.size
    )
    if not windows:
        raise BilingualVowelEngineError(
            "splice_window_missing", "Factorized listener changes produced no window."
        )
    lens, weights = output_domain_splice(neutral, full_lens, windows)
    boundary = boundary_artifact_report(neutral, full_lens, lens, windows)
    localization = localization_report(neutral, lens, target_intervals)
    arrays = (neutral, identity, full_lens, lens)
    clipped = [
        float(np.mean(np.abs(values.astype(np.int64)) >= 32767))
        for values in arrays
    ]
    equal_nonempty = bool(
        neutral.size
        and neutral.size == identity.size == full_lens.size == lens.size
    )
    finite = all(np.isfinite(values.astype(np.float64)).all() for values in arrays)
    outside_exact = bool(
        np.array_equal(lens[weights == 0.0], neutral[weights == 0.0])
    )
    interior_exact = bool(
        np.any(weights == 1.0)
        and np.array_equal(lens[weights == 1.0], full_lens[weights == 1.0])
    )
    integrity_pass = bool(
        np.array_equal(neutral, identity)
        and equal_nonempty
        and finite
        and all(value < 0.001 for value in clipped)
        and outside_exact
        and interior_exact
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
            "version": FACTORIZED_COMPOSITION_VERSION,
            "rule_decoder_columns": {
                rule_id: list(columns)
                for rule_id, columns in controlled.rule_decoder_columns.items()
            },
            "neutral_alignment_frames": sum(controlled.predicted_durations),
            "lens_alignment_frames": sum(controlled.predicted_durations),
            "input_not_sent_to_openai": True,
        },
    )


def render_factorized_composition_candidate(
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
            "The factorized spike requires two or three passing rules.",
        )
    if any(cell.candidate_rung != "v8" for cell in passing_cells):
        raise BilingualCandidateScopeError(
            "unsupported_mixed_rung_composition",
            "The factorized correction cannot mix candidate rungs.",
        )
    selected_rule_ids = tuple(sorted(cell.rule_id for cell in passing_cells))
    cells_by_id = {cell.rule_id: cell for cell in passing_cells}
    cells = tuple(cells_by_id[rule_id] for rule_id in selected_rule_ids)
    combined_planner = runtime._composition_planner(selected_rule_ids)
    combined_plan = combined_planner.plan(text)
    if active_changed_rule_ids(combined_plan) != selected_rule_ids:
        raise BilingualCandidateRuntimeError(
            "composition_plan_rule_drift",
            "The factorized combined plan lost its exact selected rules.",
        )
    for cell in cells:
        counts = (
            _count_rule_occurrences(source_plan, cell.rule_id),
            _count_rule_occurrences(combined_plan, cell.rule_id),
        )
        if counts[0] <= 0 or len(set(counts)) != 1:
            raise BilingualCandidateRuntimeError(
                "composition_plan_occurrence_drift",
                "Factorization changed a selected rule occurrence count.",
            )
    rendered = render_rule_factorized_v8(
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
