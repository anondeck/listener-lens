from __future__ import annotations

from dataclasses import asdict

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
from .controlled_vowel_synthesis_v2 import (
    CONTROLLED_VOWEL_SYNTHESIS_VERSION,
    render_controlled_listener_triplet_v2,
)
from .kokoro_output_domain_splice import (
    boundary_artifact_report,
    output_domain_splice,
)
from .kokoro_typed_diagnostic import localization_report


VOWEL_WORD_CONTEXT_CANDIDATE_VERSION = "vowel-word-context-plus-excitation-v1"


class BilingualVowelWordContextRuntime(BilingualListenerRuntimeV8):
    """Atomic vowel candidate with word-context state and excitation.

    This runtime is intentionally restricted to isolated vowel-rule plans. It
    reuses the frozen v8 stress-plus-vowel alignment and output splice, while
    asking the existing word-context mechanism to condition the complete target
    carrier word. The stronger full-lens signal is still exposed only inside
    the same local output-domain windows.
    """

    @staticmethod
    def _require_atomic_vowel_plan(plan: BilingualVowelPlan) -> None:
        coverage = plan.coverage
        if (
            coverage.changed_vowel_occurrences < 1
            or coverage.changed_consonant_occurrences
            or coverage.changed_prosody_occurrences
            or coverage.changed_insertion_occurrences
            or plan.active_prosody_rule_ids
        ):
            raise BilingualVowelEngineError(
                "non_atomic_vowel_candidate",
                "The word-context candidate accepts one isolated vowel rule only.",
            )

    def render(self, text: str) -> BilingualVowelRender | BilingualVowelPlan:
        plan = self.planner.plan(text)
        pair = plan.pair_plan()
        if pair is None:
            return plan
        self._require_atomic_vowel_plan(plan)
        vowel_columns = self._vowel_model_columns(plan)
        controlled = render_controlled_listener_triplet_v2(
            self.synthesis,
            pair,
            vowel_columns=vowel_columns,
            # Reusing the existing word-context branch is deliberate: these are
            # the only changed columns, and that branch expands their contextual
            # text state plus target-conditioned F0/noise to the containing word.
            consonant_columns=vowel_columns,
            consonant_context_mode="word",
        )
        neutral = self._pcm(controlled.neutral)
        identity = self._pcm(controlled.identity)
        full_lens = self._pcm(controlled.full_lens)
        alignment = bilingual_alignment_record_v8(
            model=self.synthesis.model,
            plan=plan,
            durations=controlled.predicted_durations,
            sample_count=neutral.size,
        )
        lens_alignment = bilingual_alignment_record_v8(
            model=self.synthesis.model,
            plan=plan,
            durations=controlled.lens_predicted_durations,
            sample_count=neutral.size,
        )
        rows = alignment["target_occurrences"]
        if not rows or any(row["segment_type"] != "vowel" for row in rows):
            raise BilingualVowelEngineError(
                "non_atomic_vowel_alignment",
                "The word-context candidate received a non-vowel target span.",
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
                "The word-context candidate produced no local output window.",
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
        prosody_control_pass = bool(
            controlled.neutral_prosody.eligible
            and controlled.lens_prosody.eligible
            and controlled.neutral_prosody.operation == "identity"
            and controlled.lens_prosody.operation == "identity"
            and not controlled.stress_duration_interventions
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
            changed_rules_acoustically_validated=False,
            evidence_status=(
                "integrity_pass_acoustic_validation_pending"
                if integrity_pass
                else "automatic_integrity_failed"
            ),
            prosody_control_pass=prosody_control_pass,
            active_prosody_rule_ids=(),
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
                "version": VOWEL_WORD_CONTEXT_CANDIDATE_VERSION,
                "controlled_vowel_synthesis_version": (
                    CONTROLLED_VOWEL_SYNTHESIS_VERSION
                ),
                "vowel_measurement_alignment_version": (
                    VOWEL_MEASUREMENT_ALIGNMENT_VERSION
                ),
                "vowel_context_mode": "target_word_state_plus_excitation",
                "vowel_columns": list(controlled.vowel_columns),
                "stress_vowel_state_columns": list(controlled.vowel_state_columns),
                "word_context_columns": list(controlled.replaced_columns),
                "word_context_excitation_frame_count": (
                    controlled.consonant_excitation_frame_count
                ),
                "neutral": asdict(controlled.neutral_prosody),
                "lens": asdict(controlled.lens_prosody),
                "neutral_alignment_frames": sum(controlled.predicted_durations),
                "lens_alignment_frames": sum(controlled.lens_predicted_durations),
                "input_not_sent_to_openai": True,
            },
        )
