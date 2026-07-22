from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np

from .bilingual_vowel_engine import (
    BilingualVowelEngineError,
    BilingualVowelPlan,
)
from .controlled_listener_synthesis import (
    ControlledListenerRender,
    render_controlled_listener_triplet,
)
from .kokoro_output_domain_splice import (
    boundary_artifact_report,
    output_domain_splice,
)
from .kokoro_synthesis import (
    PairPlan,
    SAMPLE_RATE_HZ,
    _word_column_spans,
    pcm16_bytes,
)
from .kokoro_typed_diagnostic import localization_report


PROSODY_COMPONENT_VERSION = "prosody-component-v2"
STRESS_RULE_ID = "enpt.lexical_stress_initial_bias"
QUESTION_RULE_ID = "pten.polar_rise_fall_statement"
SUPPORTED_RULE_IDS = frozenset({STRESS_RULE_ID, QUESTION_RULE_ID})
STRESS_SPLICE_CONTEXT_MS = 25.0
STRESS_SPLICE_CONTEXT_SAMPLES = round(STRESS_SPLICE_CONTEXT_MS * SAMPLE_RATE_HZ / 1000)


@dataclass(frozen=True)
class ProsodyComponentVerification:
    neutral_identity_bit_exact: bool
    equal_nonempty_samples: bool
    finite: bool
    unclipped: bool
    outside_splice_exact_neutral: bool
    full_weight_interior_exact_lens: bool
    boundary_metrics_pass: bool
    localization_pass: bool
    localization_fraction: float
    control_pass: bool
    integrity_pass: bool


@dataclass(frozen=True)
class ProsodyComponentRender:
    plan: BilingualVowelPlan
    rule_id: str
    neutral_phonemes: str
    lens_phonemes: str
    target_word_indexes: tuple[int, ...]
    neutral_pcm: np.ndarray
    identity_pcm: np.ndarray
    full_lens_pcm: np.ndarray
    lens_pcm: np.ndarray
    neutral_durations: tuple[int, ...]
    lens_durations: tuple[int, ...]
    splice_windows: tuple[dict[str, Any], ...]
    target_intervals: tuple[dict[str, Any], ...]
    stress_interventions: tuple[dict[str, Any], ...]
    neutral_prosody: dict[str, Any]
    lens_prosody: dict[str, Any]
    boundary: dict[str, Any]
    localization: dict[str, Any]
    verification: ProsodyComponentVerification
    sample_rate_hz: int = SAMPLE_RATE_HZ
    version: str = PROSODY_COMPONENT_VERSION


def _pcm(audio: np.ndarray) -> np.ndarray:
    return np.frombuffer(pcm16_bytes(audio), dtype="<i2").copy()


def _replace_words(
    complete: str, originals: Sequence[str], replacements: Sequence[str]
) -> str:
    if len(originals) != len(replacements):
        raise BilingualVowelEngineError(
            "prosody_word_alignment_drift",
            "Prosody-only words do not match the carrier word count.",
        )
    pieces: list[str] = []
    cursor = 0
    for original, replacement in zip(originals, replacements, strict=True):
        start = complete.find(original, cursor)
        if start < cursor:
            raise BilingualVowelEngineError(
                "prosody_word_alignment_drift",
                "A carrier word could not be found in the neutral phone plan.",
            )
        pieces.extend((complete[cursor:start], replacement))
        cursor = start + len(original)
    pieces.append(complete[cursor:])
    return "".join(pieces)


def prosody_only_lens_phonemes(
    plan: BilingualVowelPlan,
) -> tuple[str, tuple[int, ...]]:
    """Apply only changed stress markers while holding every segment constant."""

    replacements: list[str] = []
    target_indexes: list[int] = []
    for word in plan.words:
        symbols = list(word.neutral_phone)
        changed = False
        for occurrence in word.prosody_occurrences:
            if not occurrence.changed:
                continue
            start = occurrence.phone_offset
            end = start + occurrence.phone_length
            if "".join(symbols[start:end]) != occurrence.source:
                raise BilingualVowelEngineError(
                    "prosody_alignment_drift",
                    "A stress occurrence no longer matches the neutral phone plan.",
                )
            if len(occurrence.target) != occurrence.phone_length:
                raise BilingualVowelEngineError(
                    "prosody_token_length_drift",
                    "Prosody-only replacement must preserve model-token length.",
                )
            symbols[start:end] = occurrence.target
            changed = True
        replacements.append("".join(symbols))
        if changed:
            target_indexes.append(word.word_index)
    if not target_indexes:
        raise BilingualVowelEngineError(
            "prosody_rule_missing",
            "The plan contains no changed lexical-stress occurrence.",
        )
    complete = _replace_words(
        plan.neutral_phonemes,
        [word.neutral_phone for word in plan.words],
        replacements,
    )
    return complete, tuple(target_indexes)


def _sample_interval(
    columns: Sequence[int], durations: Sequence[int], sample_count: int
) -> dict[str, Any]:
    selected = tuple(int(column) for column in columns)
    if not selected or selected != tuple(range(selected[0], selected[-1] + 1)):
        raise BilingualVowelEngineError(
            "prosody_interval_drift", "Prosody columns must be contiguous."
        )
    total_frames = sum(int(value) for value in durations)
    if total_frames <= 0 or sample_count <= 0 or sample_count % total_frames:
        raise BilingualVowelEngineError(
            "prosody_sample_alignment_drift",
            "Prosody samples do not map to integral decoder frames.",
        )
    samples_per_frame = sample_count // total_frames
    start = sum(int(value) for value in durations[: selected[0]]) * samples_per_frame
    end = sum(int(value) for value in durations[: selected[-1] + 1]) * samples_per_frame
    if end <= start:
        raise BilingualVowelEngineError(
            "prosody_empty_interval", "A prosody interval has no decoded samples."
        )
    return {
        "columns": list(selected),
        "start_sample": start,
        "end_sample_exclusive": end,
        "start_s": start / SAMPLE_RATE_HZ,
        "end_s": end / SAMPLE_RATE_HZ,
    }


def _merge_windows(
    windows: Sequence[dict[str, Any]], sample_count: int
) -> tuple[dict[str, Any], ...]:
    pairs = sorted(
        (
            max(0, int(row["start_sample"])),
            min(sample_count, int(row["end_sample_exclusive"])),
        )
        for row in windows
    )
    merged: list[tuple[int, int]] = []
    for start, end in pairs:
        if end <= start:
            continue
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return tuple(
        {
            "start_sample": start,
            "end_sample_exclusive": end,
            "start_s": start / SAMPLE_RATE_HZ,
            "end_s": end / SAMPLE_RATE_HZ,
        }
        for start, end in merged
    )


def _stress_target_intervals(
    model: Any,
    plan: BilingualVowelPlan,
    target_word_indexes: Sequence[int],
    durations: Sequence[int],
    sample_count: int,
) -> tuple[dict[str, Any], ...]:
    spans = _word_column_spans(model, plan.neutral_phonemes)
    if len(spans) != len(plan.words):
        raise BilingualVowelEngineError(
            "prosody_word_alignment_drift",
            "Prosody-only word spans differ from the carrier plan.",
        )
    return tuple(
        _sample_interval(spans[index], durations, sample_count)
        for index in target_word_indexes
    )


def _question_target_interval(
    report: Any, f0_frame_count: int, sample_count: int
) -> dict[str, Any]:
    if report.start_index is None or report.end_index_exclusive is None:
        raise BilingualVowelEngineError(
            "prosody_window_missing",
            "The controlled question contour has no sample window.",
        )
    start = round(report.start_index * sample_count / f0_frame_count)
    end = round(report.end_index_exclusive * sample_count / f0_frame_count)
    if not 0 <= start < end <= sample_count:
        raise BilingualVowelEngineError(
            "prosody_window_drift",
            "The controlled question contour is outside the decoded audio.",
        )
    return {
        "start_sample": start,
        "end_sample_exclusive": end,
        "start_s": start / SAMPLE_RATE_HZ,
        "end_s": end / SAMPLE_RATE_HZ,
    }


def render_prosody_component(runtime: Any, text: str) -> ProsodyComponentRender:
    """Render one isolated listener-prosody rule with segment plans held fixed."""

    plan = runtime.planner.plan(text)
    active = set(plan.active_prosody_rule_ids)
    unsupported = active - SUPPORTED_RULE_IDS
    if unsupported or len(active) != 1:
        raise BilingualVowelEngineError(
            "unsupported_prosody_component",
            "Prosody-component rendering requires exactly one supported active rule.",
        )
    rule_id = next(iter(active))
    if rule_id == STRESS_RULE_ID:
        lens_phonemes, target_word_indexes = prosody_only_lens_phonemes(plan)
        neutral_operation = "identity"
        lens_operation = "identity"
        allow_prosody_only = False
    else:
        lens_phonemes = plan.neutral_phonemes
        target_word_indexes = (len(plan.words) - 1,)
        neutral_operation = "canonical_bp_rise_fall"
        lens_operation = "statement_fall"
        allow_prosody_only = True
    pair = PairPlan(
        source_phonemes=plan.render_reference_phonemes,
        neutral_phonemes=plan.neutral_phonemes,
        lens_phonemes=lens_phonemes,
        target_word_indexes=target_word_indexes,
    )
    controlled: ControlledListenerRender = render_controlled_listener_triplet(
        runtime.synthesis,
        pair,
        neutral_f0_operation=neutral_operation,
        lens_f0_operation=lens_operation,
        allow_prosody_only=allow_prosody_only,
    )
    neutral = _pcm(controlled.neutral)
    identity = _pcm(controlled.identity)
    full_lens = _pcm(controlled.full_lens)
    if rule_id == STRESS_RULE_ID:
        target_intervals = _stress_target_intervals(
            runtime.synthesis.model,
            plan,
            target_word_indexes,
            controlled.predicted_durations,
            neutral.size,
        )
        candidate_windows = tuple(
            {
                "start_sample": row["start_sample"] - STRESS_SPLICE_CONTEXT_SAMPLES,
                "end_sample_exclusive": row["end_sample_exclusive"]
                + STRESS_SPLICE_CONTEXT_SAMPLES,
            }
            for row in target_intervals
        )
    else:
        question_interval = _question_target_interval(
            controlled.lens_prosody,
            int(controlled.lens_f0.shape[-1]),
            neutral.size,
        )
        target_intervals = (question_interval,)
        candidate_windows = target_intervals
    windows = _merge_windows(candidate_windows, neutral.size)
    if not windows:
        raise BilingualVowelEngineError(
            "prosody_window_missing", "Prosody-component rendering produced no window."
        )
    lens, weights = output_domain_splice(neutral, full_lens, windows)
    boundary = boundary_artifact_report(neutral, full_lens, lens, windows)
    localization = localization_report(neutral, lens, target_intervals)
    arrays = (neutral, identity, full_lens, lens)
    equal_nonempty = bool(
        neutral.size and neutral.size == identity.size == full_lens.size == lens.size
    )
    finite = all(np.isfinite(values.astype(np.float64)).all() for values in arrays)
    unclipped = all(
        float(np.mean(np.abs(values.astype(np.int64)) >= 32767)) < 0.001
        for values in arrays
    )
    outside_exact = bool(np.array_equal(lens[weights == 0.0], neutral[weights == 0.0]))
    interior_exact = bool(
        np.any(weights == 1.0)
        and np.array_equal(lens[weights == 1.0], full_lens[weights == 1.0])
    )
    if rule_id == STRESS_RULE_ID:
        control_pass = bool(
            controlled.stress_duration_interventions
            and all(
                report.eligible and report.transferred_frames >= 1
                for report in controlled.stress_duration_interventions
            )
            and sum(controlled.predicted_durations)
            == sum(controlled.lens_predicted_durations)
        )
    else:
        control_pass = bool(
            controlled.neutral_prosody.eligible
            and controlled.lens_prosody.eligible
            and controlled.neutral_prosody.peak_hz
            > controlled.neutral_prosody.start_hz
            > controlled.neutral_prosody.end_hz
            and controlled.lens_prosody.peak_hz
            == controlled.lens_prosody.start_hz
            > controlled.lens_prosody.end_hz
        )
    integrity_pass = bool(
        np.array_equal(neutral, identity)
        and equal_nonempty
        and finite
        and unclipped
        and outside_exact
        and interior_exact
        and boundary.get("pass") is True
        and localization.get("pass") is True
        and control_pass
    )
    verification = ProsodyComponentVerification(
        neutral_identity_bit_exact=bool(np.array_equal(neutral, identity)),
        equal_nonempty_samples=equal_nonempty,
        finite=finite,
        unclipped=unclipped,
        outside_splice_exact_neutral=outside_exact,
        full_weight_interior_exact_lens=interior_exact,
        boundary_metrics_pass=bool(boundary.get("pass")),
        localization_pass=bool(localization.get("pass")),
        localization_fraction=float(
            localization.get("inside_difference_energy_fraction", 0.0)
        ),
        control_pass=control_pass,
        integrity_pass=integrity_pass,
    )
    return ProsodyComponentRender(
        plan=plan,
        rule_id=rule_id,
        neutral_phonemes=plan.neutral_phonemes,
        lens_phonemes=lens_phonemes,
        target_word_indexes=target_word_indexes,
        neutral_pcm=neutral,
        identity_pcm=identity,
        full_lens_pcm=full_lens,
        lens_pcm=lens,
        neutral_durations=controlled.predicted_durations,
        lens_durations=controlled.lens_predicted_durations,
        splice_windows=windows,
        target_intervals=target_intervals,
        stress_interventions=tuple(
            asdict(report) for report in controlled.stress_duration_interventions
        ),
        neutral_prosody=asdict(controlled.neutral_prosody),
        lens_prosody=asdict(controlled.lens_prosody),
        boundary=boundary,
        localization=localization,
        verification=verification,
    )
