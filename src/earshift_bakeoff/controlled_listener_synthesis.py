from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Literal

import numpy as np
import torch

from .kokoro_synthesis import (
    MAX_PHONEME_CHARACTERS,
    RNG_SEED,
    SAMPLE_RATE_HZ,
    SPEED,
    KokoroSynthesisError,
    KokoroSynthesisRuntime,
    PairPlan,
    _INFERENCE_LOCK,
    _f0_noise,
    _input_ids,
    _predicted_alignment,
    _text_features,
    _validate_plan,
    _filtered_symbols,
    _WORD_BOUNDARIES,
    pcm16_bytes,
)


F0Operation = Literal["identity", "canonical_bp_rise_fall", "statement_fall"]
ConsonantContextMode = Literal["adjacent", "word"]
PROSODY_CONTROL_VERSION = "listener-prosody-control-v3"
SEGMENT_EXCITATION_CONTROL_VERSION = "listener-segment-excitation-v2"
FINAL_CONTOUR_START_FRACTION = 0.65
RISE_FALL_PEAK_FRACTION = 0.60
RISE_FALL_PEAK_RATIO = 1.18
FINAL_F0_RATIO = 0.78
VOICED_F0_THRESHOLD_HZ = 40.0
MINIMUM_FINAL_VOICED_FRAMES = 8
EPENTHETIC_VOWEL_FRAMES = 3
STRESS_DURATION_TRANSFER_FRAMES = 2
STRESS_INTENSITY_GAIN_DB = 3.0
_VOWEL_SYMBOLS = frozenset("AIOSQTWYᵊaeiouyɑɐɒæɔəɚɛɜɨɪɯʊʌᵻɤøœ")


@dataclass(frozen=True)
class F0InterventionReport:
    version: str
    operation: str
    eligible: bool
    reason: str | None
    start_index: int | None
    end_index_exclusive: int | None
    voiced_frame_count: int
    start_hz: float | None
    peak_hz: float | None
    end_hz: float | None
    curve_sha256: str


@dataclass(frozen=True)
class StressDurationIntervention:
    promoted_marker_column: int
    promoted_vowel_column: int
    demoted_marker_column: int
    demoted_vowel_column: int
    transferred_frames: int
    transferred_ms: float
    intensity_gain_db: float
    eligible: bool
    reason: str | None
    replacement_columns: tuple[int, ...]
    duration_donor_column: int | None = None
    duration_donor_kind: str | None = None


@dataclass(frozen=True)
class ControlledListenerRender:
    neutral: np.ndarray
    identity: np.ndarray
    full_lens: np.ndarray
    predicted_durations: tuple[int, ...]
    lens_predicted_durations: tuple[int, ...]
    replaced_columns: tuple[int, ...]
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


@dataclass(frozen=True)
class NaturalConditionRender:
    audio: np.ndarray
    predicted_durations: tuple[int, ...]
    f0: np.ndarray
    sample_rate_hz: int = SAMPLE_RATE_HZ


def _curve_hash(values: Any) -> str:
    array = values.detach().cpu().numpy().astype("<f4", copy=False)
    return hashlib.sha256(array.tobytes()).hexdigest()


def _identity_report(f0: Any) -> F0InterventionReport:
    return F0InterventionReport(
        version=PROSODY_CONTROL_VERSION,
        operation="identity",
        eligible=True,
        reason=None,
        start_index=None,
        end_index_exclusive=None,
        voiced_frame_count=int((f0[0] > VOICED_F0_THRESHOLD_HZ).sum().item()),
        start_hz=None,
        peak_hz=None,
        end_hz=None,
        curve_sha256=_curve_hash(f0),
    )


def apply_final_f0_operation(
    f0: Any, operation: F0Operation
) -> tuple[Any, F0InterventionReport]:
    if f0.ndim != 2 or f0.shape[0] != 1:
        raise KokoroSynthesisError("controlled prosody requires one F0 curve")
    if operation == "identity":
        values = f0.clone()
        return values, _identity_report(values)
    values = f0.clone()
    voiced = (values[0] > VOICED_F0_THRESHOLD_HZ).nonzero().flatten()
    if voiced.numel() < MINIMUM_FINAL_VOICED_FRAMES:
        report = F0InterventionReport(
            version=PROSODY_CONTROL_VERSION,
            operation=operation,
            eligible=False,
            reason="insufficient_voiced_frames",
            start_index=None,
            end_index_exclusive=None,
            voiced_frame_count=int(voiced.numel()),
            start_hz=None,
            peak_hz=None,
            end_hz=None,
            curve_sha256=_curve_hash(values),
        )
        return values, report
    first = int(voiced[0].item())
    last = int(voiced[-1].item())
    threshold = first + round((last - first) * FINAL_CONTOUR_START_FRACTION)
    final_voiced = voiced[voiced >= threshold]
    if final_voiced.numel() < MINIMUM_FINAL_VOICED_FRAMES:
        final_voiced = voiced[-MINIMUM_FINAL_VOICED_FRAMES:]
    start = int(final_voiced[0].item())
    end = int(final_voiced[-1].item()) + 1
    base_count = max(2, min(6, int(final_voiced.numel()) // 3))
    start_hz = float(values[0, final_voiced[:base_count]].median().item())
    count = int(final_voiced.numel())
    positions = values.new_tensor([index / max(1, count - 1) for index in range(count)])
    if operation == "canonical_bp_rise_fall":
        rising = 1.0 + (RISE_FALL_PEAK_RATIO - 1.0) * (
            positions / RISE_FALL_PEAK_FRACTION
        )
        falling = RISE_FALL_PEAK_RATIO + (FINAL_F0_RATIO - RISE_FALL_PEAK_RATIO) * (
            (positions - RISE_FALL_PEAK_FRACTION) / (1.0 - RISE_FALL_PEAK_FRACTION)
        )
        ratios = torch.where(positions <= RISE_FALL_PEAK_FRACTION, rising, falling)
    elif operation == "statement_fall":
        ratios = 1.0 + (FINAL_F0_RATIO - 1.0) * positions
    else:  # pragma: no cover - Literal plus callers prevent this
        raise KokoroSynthesisError(f"unsupported F0 operation: {operation}")
    values[0, final_voiced] = start_hz * ratios
    selected = values[0, final_voiced]
    report = F0InterventionReport(
        version=PROSODY_CONTROL_VERSION,
        operation=operation,
        eligible=True,
        reason=None,
        start_index=start,
        end_index_exclusive=end,
        voiced_frame_count=count,
        start_hz=round(float(selected[0].item()), 3),
        peak_hz=round(float(selected.max().item()), 3),
        end_hz=round(float(selected[-1].item()), 3),
        curve_sha256=_curve_hash(values),
    )
    return values, report


def _validate_listener_plan(
    runtime: KokoroSynthesisRuntime,
    plan: PairPlan,
    *,
    allow_prosody_only: bool,
) -> tuple[int, ...]:
    if not allow_prosody_only:
        return _validate_plan(runtime.model, plan)
    if plan.speed != SPEED:
        raise KokoroSynthesisError(f"only the frozen speed {SPEED} is supported")
    filtered: list[tuple[str, ...]] = []
    for label, phonemes in (
        ("source", plan.source_phonemes),
        ("neutral", plan.neutral_phonemes),
        ("lens", plan.lens_phonemes),
    ):
        if not phonemes or len(phonemes) > MAX_PHONEME_CHARACTERS:
            raise KokoroSynthesisError(
                f"{label} phoneme plan must contain 1-{MAX_PHONEME_CHARACTERS} characters"
            )
        unsupported = sorted(set(phonemes) - set(runtime.model.vocab))
        if unsupported:
            raise KokoroSynthesisError(
                f"{label} phoneme plan contains unsupported symbols: "
                + "".join(unsupported)
            )
        filtered.append(_filtered_symbols(runtime.model, phonemes))
    if len({len(values) for values in filtered}) != 1:
        raise KokoroSynthesisError(
            "source, neutral, and lens plans must have equal model-token counts"
        )
    return tuple(
        index + 1
        for index, (neutral, lens) in enumerate(
            zip(filtered[1], filtered[2], strict=True)
        )
        if neutral != lens
    )


def _set_fixed_durations(
    pred_dur: Any,
    columns: tuple[int, ...],
    *,
    frames: int,
    model: Any,
    torch: Any,
) -> tuple[Any, Any]:
    """Set only controlled latent-slot durations and rebuild alignment."""

    adjusted = pred_dur.clone()
    for column in columns:
        adjusted[column] = frames
    return adjusted, _alignment_from_durations(adjusted, model=model, torch=torch)


def _alignment_from_durations(durations: Any, *, model: Any, torch: Any) -> Any:
    indices = torch.repeat_interleave(
        torch.arange(durations.shape[0], device=model.device), durations
    )
    alignment = torch.zeros((durations.shape[0], indices.shape[0]), device=model.device)
    alignment[indices, torch.arange(indices.shape[0])] = 1
    return alignment.unsqueeze(0)


def _stress_duration_specs(
    runtime: KokoroSynthesisRuntime, plan: PairPlan
) -> tuple[tuple[int, int, int, int, tuple[int, ...]], ...]:
    neutral = _filtered_symbols(runtime.model, plan.neutral_phonemes)
    lens = _filtered_symbols(runtime.model, plan.lens_phonemes)
    words: list[list[int]] = []
    current: list[int] = []
    for index, symbol in enumerate(neutral):
        if symbol in _WORD_BOUNDARIES:
            if current:
                words.append(current)
                current = []
            continue
        current.append(index)
    if current:
        words.append(current)

    def following_vowel(word: list[int], marker_index: int) -> int:
        for index in word:
            if index > marker_index and neutral[index] in _VOWEL_SYMBOLS:
                return index
        raise KokoroSynthesisError("a changed stress marker has no following vowel")

    specs: list[tuple[int, int, int, int, tuple[int, ...]]] = []
    for word in words:
        promoted = [
            index for index in word if neutral[index] == "ˌ" and lens[index] == "ˈ"
        ]
        demoted = [
            index for index in word if neutral[index] == "ˈ" and lens[index] == "ˌ"
        ]
        if not promoted and not demoted:
            continue
        if len(promoted) != len(demoted):
            raise KokoroSynthesisError("stress promotion and demotion are not paired")
        for promoted_marker, demoted_marker in zip(promoted, demoted, strict=True):
            promoted_vowel = following_vowel(word, promoted_marker)
            demoted_vowel = following_vowel(word, demoted_marker)
            replacement = tuple(
                sorted(
                    {
                        *(
                            index + 1
                            for index in range(promoted_marker, promoted_vowel + 1)
                        ),
                        *(
                            index + 1
                            for index in range(demoted_marker, demoted_vowel + 1)
                        ),
                    }
                )
            )
            specs.append(
                (
                    promoted_marker + 1,
                    promoted_vowel + 1,
                    demoted_marker + 1,
                    demoted_vowel + 1,
                    replacement,
                )
            )
    return tuple(specs)


def _apply_stress_duration_transfers(
    pred_dur: Any,
    specs: tuple[tuple[int, int, int, int, tuple[int, ...]], ...],
    *,
    model: Any,
    torch: Any,
) -> tuple[Any, Any, tuple[StressDurationIntervention, ...]]:
    adjusted = pred_dur.clone()
    reports: list[StressDurationIntervention] = []
    for (
        promoted_marker,
        promoted_vowel,
        demoted_marker,
        demoted_vowel,
        replacement_columns,
    ) in specs:
        vowel_available = max(0, int(adjusted[demoted_vowel].item()) - 1)
        marker_available = max(0, int(adjusted[demoted_marker].item()) - 1)
        if vowel_available:
            donor_column = demoted_vowel
            donor_kind = "demoted_vowel"
            available = vowel_available
        elif marker_available:
            # A one-frame vowel cannot donate without disappearing. The stress
            # marker and following vowel form one decoder stress unit, so the
            # marker is the bounded fallback donor for that same demoted unit.
            donor_column = demoted_marker
            donor_kind = "demoted_stress_marker"
            available = marker_available
        else:
            donor_column = None
            donor_kind = None
            available = 0
        transfer = min(STRESS_DURATION_TRANSFER_FRAMES, available)
        eligible = transfer >= 1
        if eligible:
            adjusted[promoted_vowel] += transfer
            adjusted[donor_column] -= transfer
        reports.append(
            StressDurationIntervention(
                promoted_marker_column=promoted_marker,
                promoted_vowel_column=promoted_vowel,
                demoted_marker_column=demoted_marker,
                demoted_vowel_column=demoted_vowel,
                transferred_frames=transfer,
                transferred_ms=transfer * 25.0,
                intensity_gain_db=STRESS_INTENSITY_GAIN_DB,
                eligible=eligible,
                reason=(
                    None
                    if eligible
                    else "demoted_stress_unit_has_no_transferable_frame"
                ),
                replacement_columns=replacement_columns,
                duration_donor_column=donor_column,
                duration_donor_kind=donor_kind,
            )
        )
    return (
        adjusted,
        _alignment_from_durations(adjusted, model=model, torch=torch),
        tuple(reports),
    )


def _column_frame_interval(durations: Any, column: int) -> tuple[int, int]:
    start = int(durations[:column].sum().item())
    end = start + int(durations[column].item())
    return start, end


def _copy_column_frames(
    target: Any, source: Any, durations: Any, columns: tuple[int, ...]
) -> tuple[Any, int]:
    values = target.clone()
    frame_count = 0
    for column in columns:
        start, end = _column_frame_interval(durations, column)
        values[..., start:end] = source[..., start:end]
        frame_count += end - start
    return values, frame_count


def _expanded_segment_context_columns(
    neutral_symbols: tuple[str, ...], insertion_columns: tuple[int, ...]
) -> tuple[int, ...]:
    columns: set[int] = set()
    for model_column in insertion_columns:
        symbol_index = model_column - 1
        for candidate_index in range(symbol_index - 1, symbol_index + 2):
            if (
                0 <= candidate_index < len(neutral_symbols)
                and neutral_symbols[candidate_index] not in _WORD_BOUNDARIES
            ):
                columns.add(candidate_index + 1)
    return tuple(sorted(columns))


def _consonant_state_columns(
    neutral_symbols: tuple[str, ...],
    consonant_columns: tuple[int, ...],
    *,
    mode: ConsonantContextMode,
) -> tuple[int, ...]:
    if mode == "adjacent":
        return _expanded_segment_context_columns(neutral_symbols, consonant_columns)
    if mode != "word":  # pragma: no cover - Literal plus public validation
        raise KokoroSynthesisError(f"unsupported consonant context mode: {mode}")
    columns: set[int] = set()
    for model_column in consonant_columns:
        symbol_index = model_column - 1
        if not 0 <= symbol_index < len(neutral_symbols):
            raise KokoroSynthesisError("consonant column is outside the phone plan")
        start = symbol_index
        while start > 0 and neutral_symbols[start - 1] not in _WORD_BOUNDARIES:
            start -= 1
        end = symbol_index + 1
        while (
            end < len(neutral_symbols) and neutral_symbols[end] not in _WORD_BOUNDARIES
        ):
            end += 1
        columns.update(range(start + 1, end + 1))
    return tuple(sorted(columns))


def _force_voiced_insertion_f0(
    f0: Any, durations: Any, columns: tuple[int, ...]
) -> tuple[Any, int]:
    values = f0.clone()
    voiced = values[0] > VOICED_F0_THRESHOLD_HZ
    global_voiced = values[0, voiced]
    if columns and global_voiced.numel() == 0:
        raise KokoroSynthesisError(
            "insertion F0 control has no voiced context to interpolate"
        )
    forced = 0
    for column in columns:
        start, end = _column_frame_interval(durations, column)
        previous = values[0, :start][values[0, :start] > VOICED_F0_THRESHOLD_HZ][-1:]
        following = values[0, end:][values[0, end:] > VOICED_F0_THRESHOLD_HZ][:1]
        if previous.numel() and following.numel():
            curve = torch.linspace(
                float(previous.item()),
                float(following.item()),
                end - start,
                device=values.device,
                dtype=values.dtype,
            )
        else:
            anchor = (
                float(previous.item())
                if previous.numel()
                else (
                    float(following.item())
                    if following.numel()
                    else float(global_voiced.median().item())
                )
            )
            curve = torch.full(
                (end - start,), anchor, device=values.device, dtype=values.dtype
            )
        values[0, start:end] = curve
        forced += end - start
    return values, forced


def _apply_stress_intensity(
    audio: np.ndarray,
    durations: Any,
    reports: tuple[StressDurationIntervention, ...],
    *,
    sample_rate_hz: int,
) -> np.ndarray:
    if not reports:
        return audio
    values = np.asarray(audio, dtype=np.float64).reshape(-1).copy()
    total_frames = int(durations.sum().item())
    if total_frames <= 0 or values.size % total_frames:
        raise KokoroSynthesisError(
            "stress intensity control lost decoder-frame alignment"
        )
    samples_per_frame = values.size // total_frames
    taper_samples = round(sample_rate_hz * 0.005)

    def apply_gain(column: int, gain_db: float) -> None:
        frame_start, frame_end = _column_frame_interval(durations, column)
        start = frame_start * samples_per_frame
        end = frame_end * samples_per_frame
        count = end - start
        if count <= 0:
            raise KokoroSynthesisError("a stressed vowel has no decoded samples")
        gain = 10.0 ** (gain_db / 20.0)
        edge = min(taper_samples, count // 2)
        envelope = np.full(count, gain, dtype=np.float64)
        if edge:
            phase = np.linspace(0.0, np.pi, edge, endpoint=False)
            ramp = 1.0 + (gain - 1.0) * (0.5 - 0.5 * np.cos(phase))
            envelope[:edge] = ramp
            envelope[-edge:] = ramp[::-1]
        values[start:end] *= envelope

    for report in reports:
        apply_gain(report.promoted_vowel_column, report.intensity_gain_db)
        apply_gain(report.demoted_vowel_column, -report.intensity_gain_db)
    if not np.isfinite(values).all():
        raise KokoroSynthesisError("stress intensity control produced nonfinite audio")
    return values.astype(np.asarray(audio).dtype, copy=False)


def render_controlled_listener_triplet(
    runtime: KokoroSynthesisRuntime,
    plan: PairPlan,
    *,
    neutral_f0_operation: F0Operation = "identity",
    lens_f0_operation: F0Operation = "identity",
    allow_prosody_only: bool = False,
    insertion_columns: tuple[int, ...] = (),
    consonant_columns: tuple[int, ...] = (),
    consonant_context_mode: ConsonantContextMode = "adjacent",
) -> ControlledListenerRender:
    torch = runtime.torch
    with _INFERENCE_LOCK, torch.no_grad():
        columns = _validate_listener_plan(
            runtime, plan, allow_prosody_only=allow_prosody_only
        )
        if any(column not in columns for column in insertion_columns):
            raise KokoroSynthesisError(
                "an insertion column is not a changed neutral/lens column"
            )
        if any(column not in columns for column in consonant_columns):
            raise KokoroSynthesisError(
                "a consonant column is not a changed neutral/lens column"
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
            _filtered_symbols(runtime.model, plan.neutral_phonemes),
            insertion_columns,
        )
        consonant_context_columns = _consonant_state_columns(
            _filtered_symbols(runtime.model, plan.neutral_phonemes),
            consonant_columns,
            mode=consonant_context_mode,
        )
        decoder_columns = tuple(
            sorted(
                set(columns).union(
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
    return ControlledListenerRender(
        neutral=neutral,
        identity=identity,
        full_lens=full_lens,
        predicted_durations=tuple(int(value) for value in pred_dur.cpu().tolist()),
        lens_predicted_durations=tuple(
            int(value) for value in lens_pred_dur.cpu().tolist()
        ),
        replaced_columns=decoder_columns,
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


def render_natural_condition(
    runtime: KokoroSynthesisRuntime,
    *,
    phonemes: str,
    reference_phonemes: str,
) -> NaturalConditionRender:
    """Render a fully conditioned comparison anchor, not a controlled pair.

    This is a diagnostic anchor with its own duration, F0, noise, and decoder
    state. It must never be presented as the prosody-controlled product side.
    """

    if not phonemes or len(phonemes) > MAX_PHONEME_CHARACTERS:
        raise KokoroSynthesisError(
            f"phoneme plan must contain 1-{MAX_PHONEME_CHARACTERS} characters"
        )
    unsupported = sorted(set(phonemes) - set(runtime.model.vocab))
    if unsupported:
        raise KokoroSynthesisError(
            "phoneme plan contains unsupported symbols: " + "".join(unsupported)
        )
    torch = runtime.torch
    with _INFERENCE_LOCK, torch.no_grad():
        ref_s = runtime._reference_style(reference_phonemes)
        features = _text_features(
            runtime.model,
            _input_ids(runtime.model, phonemes, torch),
            ref_s,
            torch,
        )
        durations, alignment = _predicted_alignment(
            runtime.model, features, SPEED, torch
        )
        f0, noise = _f0_noise(runtime.model, features, alignment, torch)
        torch.manual_seed(RNG_SEED)
        audio = runtime._decode(features["t_en"], alignment, f0, noise, ref_s)
    return NaturalConditionRender(
        audio=audio,
        predicted_durations=tuple(int(value) for value in durations.cpu().tolist()),
        f0=f0.detach().cpu().numpy(),
    )
