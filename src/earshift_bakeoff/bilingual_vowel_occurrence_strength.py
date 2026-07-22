from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np

from .bilingual_vowel_engine import BilingualVowelEngineError, BilingualVowelPlan
from .controlled_listener_synthesis import _validate_listener_plan
from .kokoro_synthesis import (
    RNG_SEED,
    SAMPLE_RATE_HZ,
    KokoroSynthesisError,
    KokoroSynthesisRuntime,
    _f0_noise,
    _INFERENCE_LOCK,
    _input_ids,
    _predicted_alignment,
    _text_features,
    _word_column_spans,
    pcm16_bytes,
)


OCCURRENCE_STRENGTH_CANDIDATE_VERSION = "v8-occurrence-strength-v1"


@dataclass(frozen=True)
class OccurrenceStrengthSpec:
    rule_id: str
    rule_occurrence_ordinal: int
    expected_occurrence_index: int
    expected_word_index: int
    expected_source: str
    expected_target: str
    strength: float


@dataclass(frozen=True)
class OccurrenceStrengthRender:
    neutral: np.ndarray
    identity: np.ndarray
    full_lens: np.ndarray
    predicted_durations: tuple[int, ...]
    decoder_column_strengths: tuple[tuple[int, float], ...]
    sample_rate_hz: int = SAMPLE_RATE_HZ
    version: str = OCCURRENCE_STRENGTH_CANDIDATE_VERSION


def occurrence_strength_columns(
    *,
    model: Any,
    plan: BilingualVowelPlan,
    specs: tuple[OccurrenceStrengthSpec, ...],
) -> dict[int, float]:
    """Map frozen changed-vowel occurrences to their v8 target-word columns.

    Baseline v8 replaces the complete decoder state for every changed target
    word. This candidate preserves that intervention family and only scales the
    state delta for an explicitly bound, single-change target word.
    """

    if not specs:
        return {}
    if len({(spec.rule_id, spec.rule_occurrence_ordinal) for spec in specs}) != len(
        specs
    ):
        raise BilingualVowelEngineError(
            "duplicate_occurrence_strength_spec",
            "Occurrence-strength specifications must be unique.",
        )
    if any(
        spec.rule_occurrence_ordinal < 0
        or spec.expected_occurrence_index < 0
        or spec.expected_word_index < 0
        or not np.isfinite(spec.strength)
        or spec.strength <= 0.0
        for spec in specs
    ):
        raise BilingualVowelEngineError(
            "invalid_occurrence_strength_spec",
            "Occurrence-strength specifications must be finite and positive.",
        )
    word_spans = _word_column_spans(model, plan.neutral_phonemes)
    if len(word_spans) != len(plan.words):
        raise BilingualVowelEngineError(
            "word_alignment_drift",
            "Occurrence-strength columns cannot be mapped to the plan.",
        )
    requested = {
        (spec.rule_id, spec.rule_occurrence_ordinal): spec for spec in specs
    }
    found: set[tuple[str, int]] = set()
    strengths: dict[int, float] = {}
    rule_ordinals: defaultdict[str, int] = defaultdict(int)
    occurrence_index = 0
    for word_index in plan.target_word_indexes:
        word = plan.words[word_index]
        word_columns = word_spans[word_index]
        if len(word_columns) != len(word.neutral_phone):
            raise BilingualVowelEngineError(
                "word_column_drift",
                "Occurrence-strength word columns differ from the phone plan.",
            )
        typed_occurrences = (
            *(("vowel", occurrence) for occurrence in word.vowel_occurrences),
            *(("consonant", occurrence) for occurrence in word.consonant_occurrences),
            *(("prosody", occurrence) for occurrence in word.prosody_occurrences),
            *(("insertion", occurrence) for occurrence in word.insertion_occurrences),
        )
        changed = tuple(
            (segment_type, occurrence)
            for segment_type, occurrence in typed_occurrences
            if occurrence.changed
        )
        for segment_type, occurrence in changed:
            ordinal = rule_ordinals[occurrence.rule_id]
            key = (occurrence.rule_id, ordinal)
            rule_ordinals[occurrence.rule_id] += 1
            spec = requested.get(key)
            if spec is not None:
                if (
                    segment_type != "vowel"
                    or len(changed) != 1
                    or occurrence_index != spec.expected_occurrence_index
                    or word_index != spec.expected_word_index
                    or occurrence.source != spec.expected_source
                    or occurrence.target != spec.expected_target
                ):
                    raise BilingualVowelEngineError(
                        "occurrence_strength_binding_drift",
                        "The requested occurrence no longer matches its frozen binding.",
                    )
                for column in word_columns:
                    previous = strengths.get(column)
                    if previous is not None and previous != spec.strength:
                        raise BilingualVowelEngineError(
                            "occurrence_strength_column_conflict",
                            "One decoder column received conflicting strengths.",
                        )
                    strengths[column] = float(spec.strength)
                found.add(key)
            occurrence_index += 1
    missing = sorted(set(requested) - found)
    if missing:
        raise BilingualVowelEngineError(
            "occurrence_strength_binding_missing",
            "A requested occurrence-strength binding was not found.",
        )
    return dict(sorted(strengths.items()))


def render_occurrence_strength_full_lens(
    *,
    runtime: KokoroSynthesisRuntime,
    plan: BilingualVowelPlan,
    specs: tuple[OccurrenceStrengthSpec, ...],
) -> OccurrenceStrengthRender:
    """Render a vowel-only v8 state with bound target-word delta strengths.

    This is deliberately isolated from the hash-frozen baseline v8 source.
    The caller must prove that an all-1.0 render reproduces the frozen v8 PCM
    before interpreting any alternative strength.
    """

    pair = plan.pair_plan()
    if pair is None:
        raise BilingualVowelEngineError(
            "occurrence_strength_pair_missing",
            "The occurrence-strength candidate requires a comparison plan.",
        )
    if (
        plan.coverage.changed_consonant_occurrences
        or plan.coverage.changed_prosody_occurrences
        or plan.coverage.changed_insertion_occurrences
        or plan.active_prosody_rule_ids
    ):
        raise BilingualVowelEngineError(
            "occurrence_strength_nonvowel_scope",
            "The occurrence-strength candidate is limited to vowel-only plans.",
        )
    torch = runtime.torch
    with _INFERENCE_LOCK, torch.no_grad():
        decoder_columns = _validate_listener_plan(
            runtime, pair, allow_prosody_only=False
        )
        strengths = occurrence_strength_columns(
            model=runtime.model,
            plan=plan,
            specs=specs,
        )
        if not set(strengths).issubset(decoder_columns):
            raise BilingualVowelEngineError(
                "occurrence_strength_column_escape",
                "A strength override escaped the frozen v8 decoder columns.",
            )
        ref_s = runtime._reference_style(pair.source_phonemes)
        source_features = _text_features(
            runtime.model,
            _input_ids(runtime.model, pair.source_phonemes, torch),
            ref_s,
            torch,
        )
        predicted_durations, alignment = _predicted_alignment(
            runtime.model, source_features, pair.speed, torch
        )
        neutral_features = _text_features(
            runtime.model,
            _input_ids(runtime.model, pair.neutral_phonemes, torch),
            ref_s,
            torch,
        )
        lens_features = _text_features(
            runtime.model,
            _input_ids(runtime.model, pair.lens_phonemes, torch),
            ref_s,
            torch,
        )
        predicted_f0, noise = _f0_noise(
            runtime.model, neutral_features, alignment, torch
        )
        neutral_state = neutral_features["t_en"]
        lens_state = neutral_state.clone()
        direct_columns = tuple(
            column
            for column in decoder_columns
            if float(strengths.get(column, 1.0)) == 1.0
        )
        if direct_columns:
            lens_state[:, :, list(direct_columns)] = lens_features["t_en"][
                :, :, list(direct_columns)
            ]
        for column, strength in strengths.items():
            if strength == 1.0:
                continue
            lens_state[:, :, column] = neutral_state[:, :, column] + strength * (
                lens_features["t_en"][:, :, column]
                - neutral_state[:, :, column]
            )
        torch.manual_seed(RNG_SEED)
        neutral = runtime._decode(
            neutral_state, alignment, predicted_f0, noise, ref_s
        )
        torch.manual_seed(RNG_SEED)
        identity = runtime._decode(
            neutral_state, alignment, predicted_f0, noise, ref_s
        )
        torch.manual_seed(RNG_SEED)
        full_lens = runtime._decode(
            lens_state, alignment, predicted_f0, noise, ref_s
        )
    if neutral.shape != identity.shape or neutral.shape != full_lens.shape:
        raise KokoroSynthesisError(
            "occurrence-strength triplet has unequal samples"
        )
    if pcm16_bytes(neutral) != pcm16_bytes(identity):
        raise KokoroSynthesisError(
            "occurrence-strength identity is not bit-exact"
        )
    return OccurrenceStrengthRender(
        neutral=neutral,
        identity=identity,
        full_lens=full_lens,
        predicted_durations=tuple(
            int(value) for value in predicted_durations.cpu().tolist()
        ),
        decoder_column_strengths=tuple(strengths.items()),
    )
