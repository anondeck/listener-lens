from __future__ import annotations

from dataclasses import dataclass
import unicodedata
from typing import Any

from .bilingual_candidate_registry import BilingualCandidateCell
from .bilingual_candidate_runtime import (
    evaluate_current_context_composition_acoustics,
)
from .bilingual_listener_engine_v8 import BilingualListenerPlannerV8
from .bilingual_product_isolation import active_changed_rule_ids
from .bilingual_v8_carrier_retry import (
    BilingualListenerPlannerV8CarrierRetry,
    CarrierRetrySpec,
)
from .bilingual_vowel_engine import (
    BilingualVowelEngineError,
    BilingualVowelPlan,
    BilingualVowelRender,
    MappingKey,
)
from .bilingual_listener_engine_v8 import BilingualListenerRuntimeV8


ADAPTIVE_CARRIER_CANDIDATE_VERSION = "v8-adaptive-carrier-v1"


@dataclass(frozen=True)
class AdaptiveCarrierAttempt:
    round_index: int
    retry_specs: tuple[CarrierRetrySpec, ...]
    plan: BilingualVowelPlan
    render: BilingualVowelRender
    acoustic: dict[str, Any]
    automatic_pass: bool
    failed_mapping_keys: tuple[MappingKey, ...]


@dataclass(frozen=True)
class AdaptiveCarrierResult:
    version: str
    automatic_pass: bool
    rescued_after_retry: bool
    selected_round_index: int | None
    attempts: tuple[AdaptiveCarrierAttempt, ...]
    failure_reason: str | None

    @property
    def selected_attempt(self) -> AdaptiveCarrierAttempt | None:
        if self.selected_round_index is None:
            return None
        return next(
            attempt
            for attempt in self.attempts
            if attempt.round_index == self.selected_round_index
        )


def _mapping_key(plan: BilingualVowelPlan, word_index: int) -> MappingKey:
    word = plan.words[word_index]
    return MappingKey(
        source_casefold=word.source.casefold(),
        source_phone=unicodedata.normalize("NFD", word.source_phone),
        carrier_role=word.carrier_role,
        profile_id=plan.profile_id,
    )


def failed_vowel_mapping_keys(
    *,
    plan: BilingualVowelPlan,
    render: BilingualVowelRender,
    acoustic: dict[str, Any],
) -> tuple[MappingKey, ...]:
    """Return retryable mappings for failed per-occurrence acoustic gates."""

    if (
        not render.verification.integrity_pass
        or acoustic.get("integrity_pass") is not True
        or acoustic.get("identity_false_positive_count") != 0
    ):
        return ()
    failed_indexes = {
        int(occurrence["occurrence_index"])
        for cell in acoustic.get("cells", ())
        for occurrence in cell.get("occurrences", ())
        if occurrence.get("aggregate", {}).get("directional_pass") is False
    }
    if not failed_indexes:
        return ()
    rows = {
        int(row["occurrence_index"]): row
        for row in render.alignment["target_occurrences"]
    }
    if not failed_indexes.issubset(rows):
        raise BilingualVowelEngineError(
            "adaptive_carrier_occurrence_drift",
            "A failed acoustic occurrence is absent from the alignment.",
        )
    ordered: list[MappingKey] = []
    for occurrence_index in sorted(failed_indexes):
        row = rows[occurrence_index]
        if row["segment_type"] != "vowel":
            raise BilingualVowelEngineError(
                "adaptive_carrier_nonvowel_failure",
                "Carrier retry is limited to failed vowel occurrences.",
            )
        key = _mapping_key(plan, int(row["word_index"]))
        if key not in ordered:
            ordered.append(key)
    return tuple(ordered)


class BilingualAdaptiveCarrierRuntime:
    """Fail-closed v8 composition with bounded deterministic carrier retries."""

    def __init__(
        self,
        *,
        base_planner: BilingualListenerPlannerV8,
        synthesis: Any,
        cells: tuple[BilingualCandidateCell, ...],
        scaler: dict[str, Any],
        maximum_retry_rounds: int,
    ) -> None:
        if not 1 <= maximum_retry_rounds <= 8:
            raise ValueError("adaptive carrier retries must be bounded to 1-8")
        if (
            len(cells) < 2
            or len({cell.rule_id for cell in cells}) != len(cells)
            or any(not cell.automatic_pass or cell.candidate_rung != "v8" for cell in cells)
        ):
            raise ValueError("adaptive carrier requires unique passing v8 cells")
        self.base_planner = base_planner
        self.synthesis = synthesis
        self.cells = cells
        self.scaler = scaler
        self.maximum_retry_rounds = maximum_retry_rounds

    def _planner(
        self, minimums: dict[MappingKey, int]
    ) -> BilingualListenerPlannerV8:
        if not minimums:
            return self.base_planner
        specs = tuple(
            CarrierRetrySpec(
                source_casefold=key.source_casefold,
                source_phone=key.source_phone,
                carrier_role=key.carrier_role,
                minimum_attempt=minimums[key],
            )
            for key in sorted(
                minimums,
                key=lambda value: (
                    value.source_casefold,
                    value.source_phone,
                    value.carrier_role,
                ),
            )
        )
        return BilingualListenerPlannerV8CarrierRetry.from_planner(
            self.base_planner, retry_specs=specs
        )

    def render(self, text: str) -> AdaptiveCarrierResult:
        expected_rule_ids = tuple(sorted(cell.rule_id for cell in self.cells))
        expected_occurrence_counts: dict[str, int] | None = None
        minimums: dict[MappingKey, int] = {}
        attempts: list[AdaptiveCarrierAttempt] = []
        failure_reason: str | None = None
        for round_index in range(self.maximum_retry_rounds + 1):
            planner = self._planner(minimums)
            plan = planner.plan(text)
            if active_changed_rule_ids(plan) != expected_rule_ids:
                raise BilingualVowelEngineError(
                    "adaptive_carrier_rule_drift",
                    "Carrier retry changed the isolated rule set.",
                )
            occurrence_counts = {
                rule_id: sum(
                    occurrence.changed and occurrence.rule_id == rule_id
                    for word in plan.words
                    for occurrence in word.vowel_occurrences
                )
                for rule_id in expected_rule_ids
            }
            if expected_occurrence_counts is None:
                expected_occurrence_counts = occurrence_counts
            elif occurrence_counts != expected_occurrence_counts:
                raise BilingualVowelEngineError(
                    "adaptive_carrier_occurrence_count_drift",
                    "Carrier retry changed the selected occurrence denominator.",
                )
            render = BilingualListenerRuntimeV8(
                planner=planner, synthesis=self.synthesis
            ).render(text)
            if not isinstance(render, BilingualVowelRender):
                raise BilingualVowelEngineError(
                    "adaptive_carrier_render_missing",
                    "Carrier retry produced no controlled pair.",
                )
            acoustic = evaluate_current_context_composition_acoustics(
                cells=self.cells,
                render=render,
                synthesis=self.synthesis,
                scaler=self.scaler,
            )
            automatic_pass = bool(
                render.verification.integrity_pass and acoustic["pass"]
            )
            failed_keys = (
                ()
                if automatic_pass
                else failed_vowel_mapping_keys(
                    plan=plan, render=render, acoustic=acoustic
                )
            )
            retry_specs = tuple(
                CarrierRetrySpec(
                    source_casefold=key.source_casefold,
                    source_phone=key.source_phone,
                    carrier_role=key.carrier_role,
                    minimum_attempt=minimums[key],
                )
                for key in minimums
            )
            attempts.append(
                AdaptiveCarrierAttempt(
                    round_index=round_index,
                    retry_specs=retry_specs,
                    plan=plan,
                    render=render,
                    acoustic=acoustic,
                    automatic_pass=automatic_pass,
                    failed_mapping_keys=failed_keys,
                )
            )
            if automatic_pass:
                return AdaptiveCarrierResult(
                    version=ADAPTIVE_CARRIER_CANDIDATE_VERSION,
                    automatic_pass=True,
                    rescued_after_retry=round_index > 0,
                    selected_round_index=round_index,
                    attempts=tuple(attempts),
                    failure_reason=None,
                )
            if not failed_keys:
                failure_reason = "nonretryable_integrity_identity_or_aggregate_failure"
                break
            if round_index == self.maximum_retry_rounds:
                failure_reason = "retry_round_limit_reached"
                break
            for key in failed_keys:
                matching_attempts = {
                    word.candidate_attempt
                    for word in plan.words
                    if _mapping_key(plan, word.word_index) == key
                }
                if len(matching_attempts) != 1:
                    raise BilingualVowelEngineError(
                        "adaptive_carrier_repetition_drift",
                        "Repeated source words no longer share one carrier attempt.",
                    )
                minimums[key] = max(minimums.get(key, 0), matching_attempts.pop() + 1)
        return AdaptiveCarrierResult(
            version=ADAPTIVE_CARRIER_CANDIDATE_VERSION,
            automatic_pass=False,
            rescued_after_retry=False,
            selected_round_index=None,
            attempts=tuple(attempts),
            failure_reason=failure_reason or "retry_exhausted",
        )
