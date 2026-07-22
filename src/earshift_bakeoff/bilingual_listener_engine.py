from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .bilingual_vowel_engine import (
    BILINGUAL_RULES_PATH,
    BilingualVowelEngineError,
    BilingualVowelPlan,
    BilingualVowelPlanner,
    BilingualVowelRender,
    BilingualVowelRuntime,
    BilingualRenderVerification,
    bilingual_alignment_record,
    _load_pinned_synthesis_voice,
    load_profiles as load_vowel_profiles,
)
from .config import ROOT, stable_json
from .kokoro_synthesis import ParityRender, SAMPLE_RATE_HZ, _word_column_spans


BILINGUAL_LISTENER_ENGINE_VERSION = 2
BILINGUAL_LISTENER_CANDIDATE_VERSION = "listener-candidate-v7"
BILINGUAL_LISTENER_RULES_PATH = ROOT / "rules" / "bilingual-listener-lenses-v2.json"

SEGMENT_SPLICE_CONTEXT_MS = 25.0
SEGMENT_SPLICE_CONTEXT_SAMPLES = round(
    SEGMENT_SPLICE_CONTEXT_MS * SAMPLE_RATE_HZ / 1000
)


def load_listener_profiles(
    path: Path = BILINGUAL_LISTENER_RULES_PATH,
) -> dict[str, dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("engine_version") != BILINGUAL_LISTENER_ENGINE_VERSION:
        raise BilingualVowelEngineError(
            "listener_rules_version_mismatch",
            "The bilingual listener rules do not match the listener engine.",
        )
    base_profiles = load_vowel_profiles(BILINGUAL_RULES_PATH)
    profiles: dict[str, dict[str, Any]] = {}
    for overlay in data.get("profiles", []):
        base_id = overlay.get("base_profile_id")
        try:
            base = base_profiles[base_id]
        except KeyError as exc:
            raise BilingualVowelEngineError(
                "unknown_base_profile", f"Unknown base vowel profile: {base_id!r}"
            ) from exc
        merged = dict(base)
        merged.update(
            {key: value for key, value in overlay.items() if key != "base_profile_id"}
        )
        merged["base_profile_id"] = base_id
        merged["engine_version"] = BILINGUAL_LISTENER_ENGINE_VERSION
        merged["profile_sources"] = tuple(data.get("sources", ()))
        if merged["id"] in profiles:
            raise BilingualVowelEngineError(
                "duplicate_listener_profile", f"Duplicate profile: {merged['id']}"
            )
        profiles[merged["id"]] = merged
    if not profiles:
        raise BilingualVowelEngineError(
            "empty_listener_profiles", "No bilingual listener profiles are configured."
        )
    return profiles


def _insertion_eligibility(
    plan: BilingualVowelPlan, profile: dict[str, Any]
) -> tuple[dict[str, Any], ...]:
    configured = {rule["id"]: rule for rule in profile.get("insertion_rules", ())}
    rule = configured.get("enpt.illicit_coda_epenthetic_i")
    if rule is None:
        return ()
    rows: list[dict[str, Any]] = []
    for word in plan.words:
        for occurrence in word.insertion_occurrences:
            rows.append(
                {
                    "rule_id": rule["id"],
                    "word_index": word.word_index,
                    "carrier_phone_offset": occurrence.phone_offset,
                    "context": occurrence.context,
                    "neutral_placeholder": occurrence.neutral_placeholder,
                    "target": occurrence.target,
                    "quality_status": "coarticulation_conditioning_required",
                    "architecture_status": rule["architecture_status"],
                    "evidence_tier": rule["evidence_tier"],
                }
            )
    return tuple(rows)


class BilingualListenerPlanner(BilingualVowelPlanner):
    """Versioned all-segment planner layered on the broad-vowel checkpoint."""

    @classmethod
    def load(
        cls, profile_id: str, *, voice_id: str | None = None
    ) -> "BilingualListenerPlanner":
        profiles = load_listener_profiles()
        try:
            profile = profiles[profile_id]
        except KeyError as exc:
            raise BilingualVowelEngineError(
                "unknown_listener_profile", f"Unknown listener profile: {profile_id}"
            ) from exc
        base = BilingualVowelPlanner.load(
            profile["base_profile_id"], voice_id=voice_id
        )
        return cls(
            profile={
                **profile,
                "voice_id": base.profile["voice_id"],
                "voice_registry_version": base.profile["voice_registry_version"],
                "voice_registry_sha256": base.profile["voice_registry_sha256"],
            },
            adapter=base.adapter,
            model_vocab=set(base.model_vocab),
            nonce_checker=base.nonce_checker,
            phone_indexes=base.phone_indexes,
            rules_path=BILINGUAL_LISTENER_RULES_PATH,
        )

    def plan(self, text: str) -> BilingualVowelPlan:
        plan = super().plan(text)
        eligible = _insertion_eligibility(plan, self.profile)
        active_question_rules = tuple(
            rule["id"]
            for rule in self.profile.get("prosody_rules", ())
            if plan.normalized_text.rstrip().endswith("?")
            and "polar_question" in rule.get("contexts", ())
            and rule.get("operation") != "identity"
        )
        active_prosody = tuple(
            sorted(set(plan.coverage.prosody_rules_used) | set(active_question_rules))
        )
        comparison_available = bool(plan.comparison_available or active_question_rules)
        target_indexes = plan.target_word_indexes
        if active_question_rules and not target_indexes:
            target_indexes = (len(plan.words) - 1,)
        payload = {
            "base_plan_sha256": plan.plan_sha256,
            "engine_version": self.engine_version,
            "candidate_version": BILINGUAL_LISTENER_CANDIDATE_VERSION,
            "profile_id": plan.profile_id,
            "insertion_eligibility": eligible,
            "active_prosody_rule_ids": active_prosody,
            "comparison_available": comparison_available,
            "target_word_indexes": target_indexes,
        }
        # The base plan remains a complete render contract. This derived hash
        # additionally binds the not-yet-applied insertion and prosody evidence
        # so callers cannot mistake a v1 vowel receipt for the broader v2 plan.
        return replace(
            plan,
            target_word_indexes=target_indexes,
            comparison_available=comparison_available,
            insertion_eligibilities=eligible,
            active_prosody_rule_ids=active_prosody,
            plan_sha256=hashlib.sha256(
                stable_json(payload).encode("utf-8")
            ).hexdigest(),
        )

    def insertion_eligibility(
        self, plan: BilingualVowelPlan
    ) -> tuple[dict[str, Any], ...]:
        return _insertion_eligibility(plan, self.profile)


class BilingualListenerRuntime(BilingualVowelRuntime):
    @classmethod
    def load(
        cls, profile_id: str, *, voice_id: str | None = None
    ) -> "BilingualListenerRuntime":
        planner = BilingualListenerPlanner.load(profile_id, voice_id=voice_id)
        synthesis = _load_pinned_synthesis_voice(planner.profile["voice_id"])
        return cls(planner=planner, synthesis=synthesis)

    @staticmethod
    def _merge_windows(
        windows: list[dict[str, Any]], sample_count: int
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

    def _insertion_model_columns(self, plan: BilingualVowelPlan) -> tuple[int, ...]:
        spans = _word_column_spans(self.synthesis.model, plan.neutral_phonemes)
        if len(spans) != len(plan.words):
            raise BilingualVowelEngineError(
                "word_alignment_drift",
                "Insertion columns cannot be mapped to the controlled phone plan.",
            )
        columns: list[int] = []
        for word, word_columns in zip(plan.words, spans, strict=True):
            if len(word_columns) != len(word.neutral_phone):
                raise BilingualVowelEngineError(
                    "word_column_drift",
                    "Insertion columns differ from the controlled word plan.",
                )
            for occurrence in word.insertion_occurrences:
                column = word_columns[occurrence.phone_offset]
                if (
                    word.neutral_phone[occurrence.phone_offset]
                    != occurrence.neutral_placeholder
                    or word.lens_phone[occurrence.phone_offset] != occurrence.target
                ):
                    raise BilingualVowelEngineError(
                        "insertion_alignment_drift",
                        "An insertion occurrence no longer matches its model column.",
                    )
                columns.append(column)
        return tuple(columns)

    def _consonant_model_columns(self, plan: BilingualVowelPlan) -> tuple[int, ...]:
        spans = _word_column_spans(self.synthesis.model, plan.neutral_phonemes)
        if len(spans) != len(plan.words):
            raise BilingualVowelEngineError(
                "word_alignment_drift",
                "Consonant columns cannot be mapped to the controlled phone plan.",
            )
        columns: list[int] = []
        for word, word_columns in zip(plan.words, spans, strict=True):
            if len(word_columns) != len(word.neutral_phone):
                raise BilingualVowelEngineError(
                    "word_column_drift",
                    "Consonant columns differ from the controlled word plan.",
                )
            for occurrence in word.consonant_occurrences:
                if not occurrence.changed:
                    continue
                start = occurrence.phone_offset
                end = start + occurrence.phone_length
                if (
                    word.neutral_phone[start:end] != occurrence.source
                    or word.lens_phone[start:end] != occurrence.target
                ):
                    raise BilingualVowelEngineError(
                        "segment_alignment_drift",
                        "A consonant occurrence no longer matches its model columns.",
                    )
                columns.extend(
                    word_columns[index]
                    for index in range(start, end)
                    if word.neutral_phone[index] != word.lens_phone[index]
                )
        return tuple(columns)

    def render(self, text: str) -> BilingualVowelRender | BilingualVowelPlan:
        from .controlled_listener_synthesis import (
            render_controlled_listener_triplet,
        )
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
        controlled = render_controlled_listener_triplet(
            self.synthesis,
            pair,
            neutral_f0_operation=(
                "canonical_bp_rise_fall" if polar_active else "identity"
            ),
            lens_f0_operation="statement_fall" if polar_active else "identity",
            allow_prosody_only=polar_active,
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
        alignment = bilingual_alignment_record(
            model=self.synthesis.model,
            plan=plan,
            durations=rendered.predicted_durations,
            sample_count=neutral.size,
        )
        lens_alignment = bilingual_alignment_record(
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
                "splice_window_missing", "Listener changes produced no splice window."
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
                else (
                    "integrity_pass_acoustic_validation_pending"
                    if integrity_pass
                    else "automatic_integrity_failed"
                )
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
