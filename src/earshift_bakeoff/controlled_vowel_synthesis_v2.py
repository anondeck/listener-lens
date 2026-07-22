from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .controlled_listener_synthesis import (
    ConsonantContextMode,
    F0InterventionReport,
    F0Operation,
    StressDurationIntervention,
    _apply_stress_duration_transfers,
    _apply_stress_intensity,
    _consonant_state_columns,
    _copy_column_frames,
    _expanded_segment_context_columns,
    _force_voiced_insertion_f0,
    _set_fixed_durations,
    _stress_duration_specs,
    _validate_listener_plan,
    apply_final_f0_operation,
    EPENTHETIC_VOWEL_FRAMES,
)
from .kokoro_synthesis import (
    RNG_SEED,
    SAMPLE_RATE_HZ,
    KokoroSynthesisError,
    KokoroSynthesisRuntime,
    PairPlan,
    _filtered_symbols,
    _f0_noise,
    _INFERENCE_LOCK,
    _input_ids,
    _predicted_alignment,
    _text_features,
    _WORD_BOUNDARIES,
    pcm16_bytes,
)


CONTROLLED_VOWEL_SYNTHESIS_VERSION = "controlled-vowel-stress-context-v2"
_STRESS_MARKERS = frozenset(("ˈ", "ˌ"))
_VOWEL_SYMBOLS = frozenset("AIOSQTWYᵊaeiouyɑɐɒæɔəɚɛɜɨɪɯʊʌᵻɤøœ")
_COMBINING_TILDE = "̃"


@dataclass(frozen=True)
class ControlledListenerRenderV2:
    neutral: np.ndarray
    identity: np.ndarray
    full_lens: np.ndarray
    predicted_durations: tuple[int, ...]
    lens_predicted_durations: tuple[int, ...]
    replaced_columns: tuple[int, ...]
    vowel_columns: tuple[int, ...]
    vowel_state_columns: tuple[int, ...]
    insertion_columns: tuple[int, ...]
    consonant_columns: tuple[int, ...]
    insertion_excitation_frame_count: int
    consonant_excitation_frame_count: int
    stress_duration_interventions: tuple[StressDurationIntervention, ...]
    neutral_f0: np.ndarray
    lens_f0: np.ndarray
    neutral_prosody: F0InterventionReport
    lens_prosody: F0InterventionReport
    sample_rate_hz: int = SAMPLE_RATE_HZ
    version: str = CONTROLLED_VOWEL_SYNTHESIS_VERSION


def vowel_stress_context_columns(
    neutral_symbols: tuple[str, ...],
    lens_symbols: tuple[str, ...],
    vowel_columns: tuple[int, ...],
    changed_columns: tuple[int, ...],
) -> tuple[int, ...]:
    """Expand complete changed vowel units to their preceding stress state.

    Kokoro assigns duration and context-conditioned content to stress-marker
    tokens. The selected controlled candidate therefore replaces the complete
    vowel unit (including a combining nasal marker) plus an immediately
    preceding primary or secondary stress marker. It never crosses a word
    boundary and it never broadens an unstressed vowel to unrelated context.
    """

    if len(neutral_symbols) != len(lens_symbols):
        raise KokoroSynthesisError("vowel state plans have unequal token counts")
    requested = tuple(sorted(set(vowel_columns)))
    changed = set(changed_columns)
    if not requested:
        return ()
    if not changed.intersection(requested):
        raise KokoroSynthesisError("vowel unit contains no changed model column")
    expanded: set[int] = set()
    for model_column in requested:
        index = model_column - 1
        if not 0 <= index < len(neutral_symbols):
            raise KokoroSynthesisError("vowel column is outside the phone plan")
        neutral = neutral_symbols[index]
        lens = lens_symbols[index]
        if neutral in _WORD_BOUNDARIES or lens in _WORD_BOUNDARIES:
            raise KokoroSynthesisError("vowel state expansion crossed a word boundary")
        if not (
            neutral in _VOWEL_SYMBOLS
            or lens in _VOWEL_SYMBOLS
            or neutral == lens == _COMBINING_TILDE
        ):
            raise KokoroSynthesisError("vowel unit contains a non-vowel model token")
        expanded.add(model_column)
        if index > 0 and neutral_symbols[index - 1] in _STRESS_MARKERS:
            if lens_symbols[index - 1] != neutral_symbols[index - 1]:
                raise KokoroSynthesisError(
                    "vowel stress context changed outside the vowel intervention"
                )
            expanded.add(model_column - 1)
    return tuple(sorted(expanded))


def render_controlled_listener_triplet_v2(
    runtime: KokoroSynthesisRuntime,
    plan: PairPlan,
    *,
    neutral_f0_operation: F0Operation = "identity",
    lens_f0_operation: F0Operation = "identity",
    allow_prosody_only: bool = False,
    vowel_columns: tuple[int, ...] = (),
    insertion_columns: tuple[int, ...] = (),
    consonant_columns: tuple[int, ...] = (),
    consonant_context_mode: ConsonantContextMode = "adjacent",
) -> ControlledListenerRenderV2:
    torch = runtime.torch
    with _INFERENCE_LOCK, torch.no_grad():
        changed_columns = _validate_listener_plan(
            runtime, plan, allow_prosody_only=allow_prosody_only
        )
        if any(column not in changed_columns for column in insertion_columns):
            raise KokoroSynthesisError(
                "an insertion column is not a changed neutral/lens column"
            )
        if any(column not in changed_columns for column in consonant_columns):
            raise KokoroSynthesisError(
                "a consonant column is not a changed neutral/lens column"
            )
        neutral_symbols = _filtered_symbols(runtime.model, plan.neutral_phonemes)
        lens_symbols = _filtered_symbols(runtime.model, plan.lens_phonemes)
        vowel_state_columns = vowel_stress_context_columns(
            neutral_symbols,
            lens_symbols,
            vowel_columns,
            changed_columns,
        )
        ref_s = runtime._reference_style(plan.source_phonemes)
        source_features = _text_features(
            runtime.model,
            _input_ids(runtime.model, plan.source_phonemes, torch),
            ref_s,
            torch,
        )
        pred_dur, alignment = _predicted_alignment(
            runtime.model, source_features, plan.speed, torch
        )
        if insertion_columns:
            pred_dur, alignment = _set_fixed_durations(
                pred_dur,
                insertion_columns,
                frames=EPENTHETIC_VOWEL_FRAMES,
                model=runtime.model,
                torch=torch,
            )
        stress_specs = _stress_duration_specs(runtime, plan)
        if stress_specs:
            (
                lens_pred_dur,
                lens_alignment,
                stress_duration_interventions,
            ) = _apply_stress_duration_transfers(
                pred_dur,
                stress_specs,
                model=runtime.model,
                torch=torch,
            )
            if any(not report.eligible for report in stress_duration_interventions):
                raise KokoroSynthesisError(
                    "a stress recategorization lacks a transferable duration frame"
                )
        else:
            lens_pred_dur = pred_dur.clone()
            lens_alignment = alignment
            stress_duration_interventions = ()
        if int(pred_dur.sum().item()) != int(lens_pred_dur.sum().item()):
            raise KokoroSynthesisError(
                "stress duration transfer changed total alignment duration"
            )
        insertion_context_columns = _expanded_segment_context_columns(
            neutral_symbols,
            insertion_columns,
        )
        consonant_context_columns = _consonant_state_columns(
            neutral_symbols,
            consonant_columns,
            mode=consonant_context_mode,
        )
        decoder_columns = tuple(
            sorted(
                set(changed_columns).union(
                    vowel_state_columns,
                    insertion_context_columns,
                    consonant_context_columns,
                    {
                        column
                        for report in stress_duration_interventions
                        for column in report.replacement_columns
                    },
                )
            )
        )
        neutral_features = _text_features(
            runtime.model,
            _input_ids(runtime.model, plan.neutral_phonemes, torch),
            ref_s,
            torch,
        )
        lens_features = _text_features(
            runtime.model,
            _input_ids(runtime.model, plan.lens_phonemes, torch),
            ref_s,
            torch,
        )
        predicted_f0, noise = _f0_noise(
            runtime.model, neutral_features, alignment, torch
        )
        lens_base_f0 = predicted_f0.clone()
        lens_noise = noise.clone()
        insertion_excitation_frame_count = 0
        consonant_excitation_frame_count = 0
        if insertion_columns or consonant_context_columns:
            lens_conditioned_f0, lens_conditioned_noise = _f0_noise(
                runtime.model, lens_features, lens_alignment, torch
            )
        if consonant_context_columns:
            lens_base_f0, consonant_excitation_frame_count = _copy_column_frames(
                lens_base_f0,
                lens_conditioned_f0,
                lens_pred_dur,
                consonant_context_columns,
            )
            lens_noise, consonant_noise_frame_count = _copy_column_frames(
                lens_noise,
                lens_conditioned_noise,
                lens_pred_dur,
                consonant_context_columns,
            )
            if consonant_noise_frame_count != consonant_excitation_frame_count:
                raise KokoroSynthesisError(
                    "consonant F0 and noise frame spans diverged"
                )
        if insertion_columns:
            lens_base_f0, insertion_excitation_frame_count = _copy_column_frames(
                lens_base_f0,
                lens_conditioned_f0,
                lens_pred_dur,
                insertion_columns,
            )
            lens_base_f0, forced_frame_count = _force_voiced_insertion_f0(
                lens_base_f0, lens_pred_dur, insertion_columns
            )
            if forced_frame_count != insertion_excitation_frame_count:
                raise KokoroSynthesisError(
                    "insertion excitation and forced-F0 spans diverged"
                )
            lens_noise, noise_frame_count = _copy_column_frames(
                lens_noise,
                lens_conditioned_noise,
                lens_pred_dur,
                insertion_columns,
            )
            if noise_frame_count != insertion_excitation_frame_count:
                raise KokoroSynthesisError(
                    "insertion F0 and noise frame spans diverged"
                )
        neutral_f0, neutral_report = apply_final_f0_operation(
            predicted_f0, neutral_f0_operation
        )
        lens_f0, lens_report = apply_final_f0_operation(lens_base_f0, lens_f0_operation)
        if not neutral_report.eligible or not lens_report.eligible:
            raise KokoroSynthesisError(
                "the requested final-contour operation lacks enough voiced frames"
            )
        neutral_state = neutral_features["t_en"]
        lens_state = neutral_state.clone()
        lens_state[:, :, list(decoder_columns)] = lens_features["t_en"][
            :, :, list(decoder_columns)
        ]
        torch.manual_seed(RNG_SEED)
        neutral = runtime._decode(neutral_state, alignment, neutral_f0, noise, ref_s)
        torch.manual_seed(RNG_SEED)
        identity = runtime._decode(neutral_state, alignment, neutral_f0, noise, ref_s)
        torch.manual_seed(RNG_SEED)
        full_lens = runtime._decode(
            lens_state, lens_alignment, lens_f0, lens_noise, ref_s
        )
        full_lens = _apply_stress_intensity(
            full_lens,
            lens_pred_dur,
            stress_duration_interventions,
            sample_rate_hz=SAMPLE_RATE_HZ,
        )
    if neutral.shape != identity.shape or neutral.shape != full_lens.shape:
        raise KokoroSynthesisError("controlled listener pair has unequal samples")
    if pcm16_bytes(neutral) != pcm16_bytes(identity):
        raise KokoroSynthesisError("controlled listener identity is not bit-exact")
    return ControlledListenerRenderV2(
        neutral=neutral,
        identity=identity,
        full_lens=full_lens,
        predicted_durations=tuple(int(value) for value in pred_dur.cpu().tolist()),
        lens_predicted_durations=tuple(
            int(value) for value in lens_pred_dur.cpu().tolist()
        ),
        replaced_columns=decoder_columns,
        vowel_columns=vowel_columns,
        vowel_state_columns=vowel_state_columns,
        insertion_columns=insertion_columns,
        consonant_columns=consonant_columns,
        insertion_excitation_frame_count=insertion_excitation_frame_count,
        consonant_excitation_frame_count=consonant_excitation_frame_count,
        stress_duration_interventions=stress_duration_interventions,
        neutral_f0=neutral_f0.detach().cpu().numpy(),
        lens_f0=lens_f0.detach().cpu().numpy(),
        neutral_prosody=neutral_report,
        lens_prosody=lens_report,
    )
