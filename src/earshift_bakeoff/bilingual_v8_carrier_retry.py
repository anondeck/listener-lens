from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, replace
import hashlib
import unicodedata
from typing import Any, Sequence

from .bilingual_listener_engine_v8 import BilingualListenerPlannerV8
from .bilingual_vowel_engine import (
    MAX_CANDIDATE_ATTEMPTS,
    MAX_RESOLUTION_ROUNDS,
    BilingualVowelEngineError,
    BilingualVowelPlan,
    CarrierAssignment,
    MappingKey,
)
from .config import stable_json


CARRIER_RETRY_CANDIDATE_VERSION = "v8-carrier-retry-v1"


@dataclass(frozen=True)
class CarrierRetrySpec:
    source_casefold: str
    source_phone: str
    carrier_role: str
    minimum_attempt: int

    def mapping_key(self, profile_id: str) -> MappingKey:
        return MappingKey(
            source_casefold=self.source_casefold.casefold(),
            source_phone=unicodedata.normalize("NFD", self.source_phone),
            carrier_role=self.carrier_role,
            profile_id=profile_id,
        )


class BilingualListenerPlannerV8CarrierRetry(BilingualListenerPlannerV8):
    """V8 planner that advances selected mappings before the bounded search."""

    def __init__(self, *, retry_specs: tuple[CarrierRetrySpec, ...], **kwargs: Any):
        super().__init__(**kwargs)
        if not retry_specs:
            raise BilingualVowelEngineError(
                "carrier_retry_missing",
                "Carrier retry requires at least one mapping binding.",
            )
        keys = tuple(spec.mapping_key(self.profile["id"]) for spec in retry_specs)
        if len(set(keys)) != len(keys) or any(
            spec.minimum_attempt < 1 for spec in retry_specs
        ):
            raise BilingualVowelEngineError(
                "carrier_retry_invalid",
                "Carrier retry bindings must be unique with positive minima.",
            )
        self.retry_specs = retry_specs
        self.minimum_attempts = dict(
            zip(keys, (spec.minimum_attempt for spec in retry_specs), strict=True)
        )

    @classmethod
    def from_planner(
        cls,
        planner: BilingualListenerPlannerV8,
        *,
        retry_specs: tuple[CarrierRetrySpec, ...],
    ) -> BilingualListenerPlannerV8CarrierRetry:
        return cls(
            retry_specs=retry_specs,
            profile=planner.profile,
            adapter=planner.adapter,
            model_vocab=set(planner.model_vocab),
            nonce_checker=planner.nonce_checker,
            phone_indexes=planner.phone_indexes,
            rules_path=planner.rules_path,
        )

    def _resolve_assignments(
        self, keys: Sequence[MappingKey], rejection_counts: Counter[str]
    ) -> tuple[dict[MappingKey, CarrierAssignment], int, int]:
        unique_keys = tuple(dict.fromkeys(keys))
        missing = sorted(set(self.minimum_attempts) - set(unique_keys), key=repr)
        if missing:
            raise BilingualVowelEngineError(
                "carrier_retry_binding_missing",
                "A carrier retry mapping is absent from the analyzed utterance.",
            )
        assignments: dict[MappingKey, CarrierAssignment] = {}
        next_attempt = {
            key: self.minimum_attempts.get(key, 0) for key in unique_keys
        }
        implicated = set(unique_keys)
        total_attempts = 0
        adjacency_checks = 0
        language = self.profile["source_gate_language"]

        for _ in range(MAX_RESOLUTION_ROUNDS):
            for key in unique_keys:
                if key not in implicated:
                    continue
                accepted: CarrierAssignment | None = None
                for attempt in range(
                    next_attempt[key], next_attempt[key] + MAX_CANDIDATE_ATTEMPTS
                ):
                    total_attempts += 1
                    candidate = self._candidate(key, attempt)
                    reasons = self._isolated_reasons(candidate)
                    if not reasons:
                        accepted = candidate
                        break
                    rejection_counts.update(reasons)
                if accepted is None:
                    raise BilingualVowelEngineError(
                        "candidate_search_exhausted",
                        "No gate-clean carrier exists after the frozen retry minimum.",
                    )
                assignments[key] = accepted

            conflicts: set[MappingKey] = set()
            for previous_key, current_key in zip(keys, keys[1:]):
                adjacency_checks += 1
                previous = assignments[previous_key]
                current = assignments[current_key]
                for (
                    side,
                    previous_surface,
                    current_surface,
                    previous_phone,
                    current_phone,
                ) in (
                    (
                        "neutral",
                        previous.neutral_surface,
                        current.neutral_surface,
                        previous.neutral_phone,
                        current.neutral_phone,
                    ),
                    (
                        "lens",
                        previous.lens_surface,
                        current.lens_surface,
                        previous.lens_phone,
                        current.lens_phone,
                    ),
                ):
                    decision = self.nonce_checker.check(
                        current_surface, language, previous_surface
                    )
                    if not decision.accepted:
                        rejection_counts[
                            f"{side}_written_espeak_"
                            f"{decision.rejection_reason or 'adjacency_rejected'}"
                        ] += 1
                        conflicts.update((previous_key, current_key))
                    for index, phone_index in enumerate(self.phone_indexes):
                        if phone_index.phone_match(previous_phone + current_phone):
                            rejection_counts[
                                f"{side}_phone_index_{index}_"
                                "adjacency_predicted_homophone"
                            ] += 1
                            conflicts.update((previous_key, current_key))
            if not conflicts:
                return assignments, total_attempts, adjacency_checks
            implicated = conflicts
            for key in conflicts:
                next_attempt[key] = assignments[key].candidate_attempt + 1
        raise BilingualVowelEngineError(
            "adjacency_search_exhausted",
            "No globally gate-clean carrier exists after the frozen retry minimum.",
        )

    def plan(self, text: str) -> BilingualVowelPlan:
        parent = super().plan(text)
        payload = {
            "parent_plan_sha256": parent.plan_sha256,
            "candidate_version": CARRIER_RETRY_CANDIDATE_VERSION,
            "retry_specs": [asdict(spec) for spec in self.retry_specs],
        }
        return replace(
            parent,
            plan_sha256=hashlib.sha256(
                stable_json(payload).encode("utf-8")
            ).hexdigest(),
        )
