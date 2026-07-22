from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .controlled_listener_synthesis import _validate_listener_plan
from .kokoro_synthesis import (
    RNG_SEED,
    SAMPLE_RATE_HZ,
    KokoroSynthesisError,
    KokoroSynthesisRuntime,
    PairPlan,
    _f0_noise,
    _filtered_symbols,
    _INFERENCE_LOCK,
    _input_ids,
    _predicted_alignment,
    _text_features,
    pcm16_bytes,
)


CONTROLLED_VOWEL_FULL_CONTEXT_VERSION = "controlled-vowel-full-context-v1"


@dataclass(frozen=True)
class ControlledFullContextVowelRender:
    neutral: np.ndarray
    identity: np.ndarray
    full_lens: np.ndarray
    predicted_durations: tuple[int, ...]
    replaced_columns: tuple[int, ...]
    validated_target_word_columns: tuple[int, ...]
    textually_changed_columns: tuple[int, ...]
    neutral_f0: np.ndarray
    sample_rate_hz: int = SAMPLE_RATE_HZ
    version: str = CONTROLLED_VOWEL_FULL_CONTEXT_VERSION


def render_full_context_vowel_triplet(
    runtime: KokoroSynthesisRuntime,
    plan: PairPlan,
) -> ControlledFullContextVowelRender:
    """Decode the complete lens text state over neutral timing and excitation."""

    torch = runtime.torch
    with _INFERENCE_LOCK, torch.no_grad():
        target_word_columns = _validate_listener_plan(
            runtime, plan, allow_prosody_only=False
        )
        neutral_symbols = _filtered_symbols(runtime.model, plan.neutral_phonemes)
        lens_symbols = _filtered_symbols(runtime.model, plan.lens_phonemes)
        textually_changed_columns = tuple(
            index + 1
            for index, (neutral_symbol, lens_symbol) in enumerate(
                zip(neutral_symbols, lens_symbols, strict=True)
            )
            if neutral_symbol != lens_symbol
        )
        if not textually_changed_columns or not set(textually_changed_columns).issubset(
            target_word_columns
        ):
            raise KokoroSynthesisError(
                "full-context textual changes escaped the validated target words"
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
        neutral_state = neutral_features["t_en"]
        lens_state = lens_features["t_en"]
        if neutral_state.shape != lens_state.shape or neutral_state.ndim != 3:
            raise KokoroSynthesisError(
                "full-context neutral and lens states have unequal shapes"
            )
        predicted_f0, noise = _f0_noise(
            runtime.model, neutral_features, alignment, torch
        )
        torch.manual_seed(RNG_SEED)
        neutral = runtime._decode(neutral_state, alignment, predicted_f0, noise, ref_s)
        torch.manual_seed(RNG_SEED)
        identity = runtime._decode(neutral_state, alignment, predicted_f0, noise, ref_s)
        torch.manual_seed(RNG_SEED)
        full_lens = runtime._decode(lens_state, alignment, predicted_f0, noise, ref_s)
    if neutral.shape != identity.shape or neutral.shape != full_lens.shape:
        raise KokoroSynthesisError("full-context vowel triplet has unequal samples")
    if pcm16_bytes(neutral) != pcm16_bytes(identity):
        raise KokoroSynthesisError("full-context vowel identity is not bit-exact")
    return ControlledFullContextVowelRender(
        neutral=neutral,
        identity=identity,
        full_lens=full_lens,
        predicted_durations=tuple(int(value) for value in pred_dur.cpu().tolist()),
        replaced_columns=tuple(range(int(neutral_state.shape[-1]))),
        validated_target_word_columns=target_word_columns,
        textually_changed_columns=textually_changed_columns,
        neutral_f0=predicted_f0.detach().cpu().numpy(),
    )
