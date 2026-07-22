from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from .bilingual_listener_engine import (
    BilingualListenerPlanner,
    BilingualListenerRuntime,
)
from .bilingual_product_matrix import (
    BilingualProductMatrix,
    PlanEvidenceReport,
    load_bilingual_product_matrix,
)


class ListenerPlanner(Protocol):
    def plan(self, text: str) -> Any: ...


class ListenerRuntime(Protocol):
    def render(self, text: str) -> Any: ...


@dataclass(frozen=True)
class BilingualProductPlan:
    plan: Any
    evidence: PlanEvidenceReport

    @property
    def product_ready(self) -> bool:
        return self.evidence.product_ready

    def safe_metadata(self) -> dict[str, Any]:
        return {
            "plan": self.plan.safe_metadata(),
            "evidence": self.evidence.safe_metadata(),
        }


@dataclass(frozen=True)
class BilingualResearchRender:
    product_plan: BilingualProductPlan
    render: Any
    classification: str = "research_only_not_product_enabled"
    api_calls_made: int = 0


class BilingualProductPlanner:
    """One planner boundary for both directions and all four selected voices."""

    def __init__(
        self,
        *,
        planner: ListenerPlanner,
        matrix: BilingualProductMatrix,
    ) -> None:
        self.planner = planner
        self.matrix = matrix

    @classmethod
    def load(
        cls, profile_id: str, *, voice_id: str | None = None
    ) -> "BilingualProductPlanner":
        return cls(
            planner=BilingualListenerPlanner.load(
                profile_id, voice_id=voice_id
            ),
            matrix=load_bilingual_product_matrix(),
        )

    def plan(self, text: str) -> BilingualProductPlan:
        plan = self.planner.plan(text)
        return BilingualProductPlan(
            plan=plan,
            evidence=self.matrix.evaluate_plan(plan),
        )


class BilingualProductRuntime:
    """Research rendering and fail-closed product rendering share one engine."""

    def __init__(
        self,
        *,
        planner: BilingualProductPlanner,
        runtime: ListenerRuntime,
    ) -> None:
        self.planner = planner
        self.runtime = runtime

    @classmethod
    def load(
        cls, profile_id: str, *, voice_id: str | None = None
    ) -> "BilingualProductRuntime":
        product_planner = BilingualProductPlanner.load(
            profile_id, voice_id=voice_id
        )
        runtime = BilingualListenerRuntime.load(
            profile_id, voice_id=voice_id
        )
        return cls(planner=product_planner, runtime=runtime)

    def render_for_validation(self, text: str) -> BilingualResearchRender:
        product_plan = self.planner.plan(text)
        return BilingualResearchRender(
            product_plan=product_plan,
            render=self.runtime.render(text),
        )

    def render_for_product(self, text: str) -> Any:
        product_plan = self.planner.plan(text)
        self.planner.matrix.require_product_ready(product_plan.plan)
        return self.runtime.render(text)
