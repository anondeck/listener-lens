from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
from typing import Any, Sequence

import numpy as np

from .bilingual_listener_engine import (
    BilingualListenerPlanner,
    BilingualListenerRuntime,
    SEGMENT_SPLICE_CONTEXT_SAMPLES,
)
from .bilingual_vowel_engine import (
    BilingualRenderVerification,
    BilingualVowelEngineError,
    BilingualVowelPlan,
    BilingualVowelRender,
    bilingual_alignment_record,
)
from .config import stable_json
from .controlled_vowel_synthesis_v2 import (
    CONTROLLED_VOWEL_SYNTHESIS_VERSION,
    render_controlled_listener_triplet_v2,
)
from .kokoro_synthesis import (
    SAMPLE_RATE_HZ,
    ParityRender,
    _word_column_spans,
)


BILINGUAL_LISTENER_CANDIDATE_VERSION_V8 = "listener-candidate-v8"
VOWEL_MEASUREMENT_ALIGNMENT_VERSION = "kokoro-stress-plus-vowel-alignment-v2"
_STRESS_MARKERS = frozenset(("ˈ", "ˌ"))


class BilingualListenerPlannerV8(BilingualListenerPlanner):
    """Immutable v8 carrier planner binding the corrected vowel intervention."""

    def plan(self, text: str) -> BilingualVowelPlan:
        parent = super().plan(text)
        payload = {
            "parent_plan_sha256": parent.plan_sha256,
            "candidate_version": BILINGUAL_LISTENER_CANDIDATE_VERSION_V8,
            "controlled_vowel_synthesis_version": CONTROLLED_VOWEL_SYNTHESIS_VERSION,
            "vowel_measurement_alignment_version": VOWEL_MEASUREMENT_ALIGNMENT_VERSION,
        }
        return replace(
            parent,
            plan_sha256=hashlib.sha256(
                stable_json(payload).encode("utf-8")
            ).hexdigest(),
        )


def _interval(
    durations: Sequence[int], columns: Sequence[int], *, sample_count: int
) -> dict[str, Any]:
    values = tuple(int(value) for value in durations)
    selected = tuple(int(value) for value in columns)
    if not selected or selected != tuple(range(selected[0], selected[-1] + 1)):
        raise BilingualVowelEngineError(
            "empty_alignment_interval", "A v8 interval is empty or noncontiguous."
        )
    total_frames = sum(values)
    if total_frames <= 0 or sample_count <= 0 or sample_count % total_frames:
        raise BilingualVowelEngineError(
            "sample_alignment_drift", "V8 samples lost the decoder frame grid."
        )
    if selected[0] < 0 or selected[-1] >= len(values):
        raise BilingualVowelEngineError(
            "segment_alignment_drift", "A v8 interval is outside the duration plan."
        )
    samples_per_frame = sample_count // total_frames
    start = sum(values[: selected[0]]) * samples_per_frame
    end = sum(values[: selected[-1] + 1]) * samples_per_frame
    return {
        "columns": list(selected),
        "start_sample": start,
        "end_sample_exclusive": end,
        "start_s": start / SAMPLE_RATE_HZ,
        "end_s": end / SAMPLE_RATE_HZ,
    }


def bilingual_alignment_record_v8(
    *,
    model: Any,
    plan: BilingualVowelPlan,
    durations: Sequence[int],
    sample_count: int,
) -> dict[str, Any]:
    """Expand changed-vowel measurement spans to stress plus complete vowel."""

    record = bilingual_alignment_record(
        model=model,
        plan=plan,
        durations=durations,
        sample_count=sample_count,
    )
    word_spans = _word_column_spans(model, plan.neutral_phonemes)
    rows: list[dict[str, Any]] = []
    for row in record["target_occurrences"]:
        if row["segment_type"] != "vowel":
            rows.append(row)
            continue
        word = plan.words[row["word_index"]]
        word_columns = word_spans[row["word_index"]]
        candidates = [
            occurrence
            for occurrence in word.vowel_occurrences
            if occurrence.changed
            and occurrence.rule_id == row["rule_id"]
            and occurrence.source == row["source"]
            and occurrence.target == row["target"]
        ]
        if len(candidates) != 1:
            raise BilingualVowelEngineError(
                "segment_alignment_drift",
                "A v8 vowel occurrence cannot be identified uniquely.",
            )
        occurrence = candidates[0]
        start = occurrence.phone_offset
        stop = start + occurrence.phone_length
        selected = list(word_columns[start:stop])
        stress_column: int | None = None
        if start > 0 and word.neutral_phone[start - 1] in _STRESS_MARKERS:
            if word.lens_phone[start - 1] != word.neutral_phone[start - 1]:
                raise BilingualVowelEngineError(
                    "segment_alignment_drift",
                    "A v8 vowel stress marker changed unexpectedly.",
                )
            stress_column = word_columns[start - 1]
            selected.insert(0, stress_column)
        rows.append(
            {
                **row,
                "measurement_interval": _interval(
                    durations, selected, sample_count=sample_count
                ),
                "measurement_alignment_version": VOWEL_MEASUREMENT_ALIGNMENT_VERSION,
                "stress_context_column": stress_column,
                "vowel_state_columns": list(word_columns[start:stop]),
            }
        )
    return {
        **record,
        "target_occurrences": rows,
        "alignment_version": VOWEL_MEASUREMENT_ALIGNMENT_VERSION,
    }


class BilingualListenerRuntimeV8(BilingualListenerRuntime):
    """V8 local runtime preserving the frozen v7 path as historical evidence."""

    def _vowel_model_columns(self, plan: BilingualVowelPlan) -> tuple[int, ...]:
        spans = _word_column_spans(self.synthesis.model, plan.neutral_phonemes)
        if len(spans) != len(plan.words):
            raise BilingualVowelEngineError(
                "word_alignment_drift", "Vowel columns cannot be mapped to v8."
            )
        columns: list[int] = []
        for word, word_columns in zip(plan.words, spans, strict=True):
            if len(word_columns) != len(word.neutral_phone):
                raise BilingualVowelEngineError(
                    "word_column_drift", "V8 vowel columns differ from the phone plan."
                )
            for occurrence in word.vowel_occurrences:
                if not occurrence.changed:
                    continue
                start = occurrence.phone_offset
                stop = start + occurrence.phone_length
                if (
                    word.neutral_phone[start:stop] != occurrence.source
                    or word.lens_phone[start:stop] != occurrence.target
                ):
                    raise BilingualVowelEngineError(
                        "segment_alignment_drift",
                        "A v8 vowel occurrence no longer matches its model columns.",
                    )
                columns.extend(word_columns[start:stop])
        return tuple(sorted(set(columns)))

    def render(self, text: str) -> BilingualVowelRender | BilingualVowelPlan:
        from .kokoro_output_domain_splice import (
            boundary_artifact_report,
            output_domain_splice,
        )
        from .kokoro_typed_diagnostic import localization_report

        plan = self.planner.plan(text)
        pair = plan.pair_plan()
        if pair is None:
            return plan
        polar_rule = "pten.polar_rise_fall_statement"
        polar_active = polar_rule in plan.active_prosody_rule_ids
        controlled = render_controlled_listener_triplet_v2(
            self.synthesis,
            pair,
            neutral_f0_operation=(
                "canonical_bp_rise_fall" if polar_active else "identity"
            ),
            lens_f0_operation="statement_fall" if polar_active else "identity",
            allow_prosody_only=polar_active,
            vowel_columns=self._vowel_model_columns(plan),
            insertion_columns=self._insertion_model_columns(plan),
            consonant_columns=self._consonant_model_columns(plan),
        )
        rendered = ParityRender(
            neutral=controlled.neutral,
            identity=controlled.identity,
            lens=controlled.full_lens,
            predicted_durations=controlled.predicted_durations,
            replaced_columns=controlled.replaced_columns,
        )
        neutral = self._pcm(rendered.neutral)
        identity = self._pcm(rendered.identity)
        full_lens = self._pcm(rendered.lens)
        alignment = bilingual_alignment_record_v8(
            model=self.synthesis.model,
            plan=plan,
            durations=rendered.predicted_durations,
            sample_count=neutral.size,
        )
        lens_alignment = bilingual_alignment_record_v8(
            model=self.synthesis.model,
            plan=plan,
            durations=controlled.lens_predicted_durations,
            sample_count=neutral.size,
        )
        word_intervals = {
            row["word_index"]: row["interval"] for row in alignment["target_words"]
        }
        stress_word_indexes = {
            row["word_index"]
            for row in alignment["target_occurrences"]
            if row["segment_type"] == "prosody"
        }
        target_intervals = [
            row["measurement_interval"]
            for row in alignment["target_occurrences"]
            if row["segment_type"] != "prosody"
        ] + [word_intervals[index] for index in sorted(stress_word_indexes)]
        prosody_window: dict[str, Any] | None = None
        if polar_active:
            report = controlled.lens_prosody
            if report.start_index is None or report.end_index_exclusive is None:
                raise BilingualVowelEngineError(
                    "prosody_window_missing",
                    "The active question contour has no controlled F0 window.",
                )
            f0_frames = int(controlled.lens_f0.shape[-1])
            start = round(report.start_index * neutral.size / f0_frames)
            end = round(report.end_index_exclusive * neutral.size / f0_frames)
            prosody_window = {
                "start_sample": start,
                "end_sample_exclusive": end,
                "start_s": start / SAMPLE_RATE_HZ,
                "end_s": end / SAMPLE_RATE_HZ,
            }
            target_intervals.append(prosody_window)
        candidate_windows = []
        for row in alignment["target_occurrences"]:
            interval = (
                word_intervals[row["word_index"]]
                if row["segment_type"] == "prosody"
                else row["measurement_interval"]
            )
            candidate_windows.append(
                {
                    "start_sample": int(interval["start_sample"])
                    - SEGMENT_SPLICE_CONTEXT_SAMPLES,
                    "end_sample_exclusive": int(interval["end_sample_exclusive"])
                    + SEGMENT_SPLICE_CONTEXT_SAMPLES,
                }
            )
        if prosody_window is not None:
            candidate_windows.append(prosody_window)
        windows = self._merge_windows(candidate_windows, neutral.size)
        if not windows:
            raise BilingualVowelEngineError(
                "splice_window_missing",
                "V8 listener changes produced no splice window.",
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
        stress_active = (
            "enpt.lexical_stress_initial_bias" in plan.active_prosody_rule_ids
        )
        stress_control_pass = bool(
            not stress_active
            or (
                controlled.stress_duration_interventions
                and all(
                    report.eligible and report.transferred_frames >= 1
                    for report in controlled.stress_duration_interventions
                )
                and sum(controlled.predicted_durations)
                == sum(controlled.lens_predicted_durations)
            )
        )
        prosody_control_pass = bool(
            controlled.neutral_prosody.eligible
            and controlled.lens_prosody.eligible
            and stress_control_pass
            and (
                not polar_active
                or (
                    controlled.neutral_prosody.peak_hz
                    > controlled.neutral_prosody.start_hz
                    > controlled.neutral_prosody.end_hz
                    and controlled.lens_prosody.peak_hz
                    == controlled.lens_prosody.start_hz
                    > controlled.lens_prosody.end_hz
                )
            )
        )
        acoustic_ready = bool(
            plan.coverage.pending_acoustic_changed_occurrences == 0
            and plan.coverage.pending_acoustic_changed_consonant_occurrences == 0
            and plan.coverage.pending_acoustic_changed_prosody_occurrences == 0
            and plan.coverage.pending_acoustic_changed_insertion_occurrences == 0
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
            and prosody_control_pass
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
            changed_rules_acoustically_validated=acoustic_ready,
            evidence_status=(
                "integrity_pass_all_changed_rules_acoustically_validated"
                if integrity_pass and acoustic_ready
                else "integrity_pass_acoustic_validation_pending"
                if integrity_pass
                else "automatic_integrity_failed"
            ),
            prosody_control_pass=prosody_control_pass,
            active_prosody_rule_ids=plan.active_prosody_rule_ids,
        )
        return BilingualVowelRender(
            plan=plan,
            neutral_pcm=neutral,
            identity_pcm=identity,
            full_lens_pcm=full_lens,
            lens_pcm=lens,
            alignment=alignment,
            lens_alignment=lens_alignment,
            splice_windows=windows,
            verification=verification,
            prosody={
                "version": "listener-prosody-control-v3",
                "controlled_vowel_synthesis_version": (
                    CONTROLLED_VOWEL_SYNTHESIS_VERSION
                ),
                "vowel_measurement_alignment_version": (
                    VOWEL_MEASUREMENT_ALIGNMENT_VERSION
                ),
                "vowel_columns": list(controlled.vowel_columns),
                "vowel_state_columns": list(controlled.vowel_state_columns),
                "neutral": asdict(controlled.neutral_prosody),
                "lens": asdict(controlled.lens_prosody),
                "sample_window": prosody_window,
                "f0_frame_count": int(controlled.lens_f0.shape[-1]),
                "stress_duration_interventions": [
                    asdict(report)
                    for report in controlled.stress_duration_interventions
                ],
                "insertion_excitation_frame_count": (
                    controlled.insertion_excitation_frame_count
                ),
                "consonant_excitation_frame_count": (
                    controlled.consonant_excitation_frame_count
                ),
                "neutral_alignment_frames": sum(controlled.predicted_durations),
                "lens_alignment_frames": sum(controlled.lens_predicted_durations),
                "input_not_sent_to_openai": True,
            },
        )
