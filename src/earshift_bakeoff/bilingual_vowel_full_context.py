from __future__ import annotations

import numpy as np

from .bilingual_listener_engine import SEGMENT_SPLICE_CONTEXT_SAMPLES
from .bilingual_listener_engine_v8 import (
    BilingualListenerRuntimeV8,
    VOWEL_MEASUREMENT_ALIGNMENT_VERSION,
    bilingual_alignment_record_v8,
)
from .bilingual_vowel_engine import (
    BilingualRenderVerification,
    BilingualVowelEngineError,
    BilingualVowelPlan,
    BilingualVowelRender,
)
from .bilingual_vowel_word_context import BilingualVowelWordContextRuntime
from .controlled_vowel_full_context import (
    CONTROLLED_VOWEL_FULL_CONTEXT_VERSION,
    render_full_context_vowel_triplet,
)
from .kokoro_output_domain_splice import (
    boundary_artifact_report,
    output_domain_splice,
)
from .kokoro_typed_diagnostic import localization_report


VOWEL_FULL_CONTEXT_CANDIDATE_VERSION = "vowel-full-context-neutral-excitation-v1"


class BilingualVowelFullContextRuntime(BilingualListenerRuntimeV8):
    """Use complete lens text context but neutral timing/excitation and local output."""

    def render(self, text: str) -> BilingualVowelRender | BilingualVowelPlan:
        plan = self.planner.plan(text)
        pair = plan.pair_plan()
        if pair is None:
            return plan
        BilingualVowelWordContextRuntime._require_atomic_vowel_plan(plan)
        controlled = render_full_context_vowel_triplet(self.synthesis, pair)
        neutral = self._pcm(controlled.neutral)
        identity = self._pcm(controlled.identity)
        full_lens = self._pcm(controlled.full_lens)
        alignment = bilingual_alignment_record_v8(
            model=self.synthesis.model,
            plan=plan,
            durations=controlled.predicted_durations,
            sample_count=neutral.size,
        )
        rows = alignment["target_occurrences"]
        if not rows or any(row["segment_type"] != "vowel" for row in rows):
            raise BilingualVowelEngineError(
                "non_atomic_vowel_alignment",
                "The full-context candidate received a non-vowel target span.",
            )
        target_intervals = [row["measurement_interval"] for row in rows]
        windows = self._merge_windows(
            [
                {
                    "start_sample": int(interval["start_sample"])
                    - SEGMENT_SPLICE_CONTEXT_SAMPLES,
                    "end_sample_exclusive": int(interval["end_sample_exclusive"])
                    + SEGMENT_SPLICE_CONTEXT_SAMPLES,
                }
                for interval in target_intervals
            ],
            neutral.size,
        )
        if not windows:
            raise BilingualVowelEngineError(
                "splice_window_missing",
                "The full-context candidate produced no local output window.",
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
            prosody_control_pass=True,
            active_prosody_rule_ids=(),
        )
        return BilingualVowelRender(
            plan=plan,
            neutral_pcm=neutral,
            identity_pcm=identity,
            full_lens_pcm=full_lens,
            lens_pcm=lens,
            alignment=alignment,
            lens_alignment=alignment,
            splice_windows=windows,
            verification=verification,
            prosody={
                "version": VOWEL_FULL_CONTEXT_CANDIDATE_VERSION,
                "controlled_version": CONTROLLED_VOWEL_FULL_CONTEXT_VERSION,
                "vowel_measurement_alignment_version": (
                    VOWEL_MEASUREMENT_ALIGNMENT_VERSION
                ),
                "state_context": "complete_lens_text_state",
                "duration_alignment_f0_noise": "neutral_source_controlled",
                "validated_target_word_columns": list(
                    controlled.validated_target_word_columns
                ),
                "textually_changed_columns": list(controlled.textually_changed_columns),
                "replaced_state_columns": list(controlled.replaced_columns),
                "neutral_alignment_frames": sum(controlled.predicted_durations),
                "lens_alignment_frames": sum(controlled.predicted_durations),
                "input_not_sent_to_openai": True,
            },
        )
