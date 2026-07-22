from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
from typing import Any

from .bilingual_candidate_runtime import (
    BilingualCandidateAcousticGateError,
    BilingualCandidateRuntime,
    _audio_record,
    _count_rule_occurrences,
    _pcm_hash,
)
from .bilingual_composed_candidate_runtime import BilingualComposedCandidateRuntime
from .bilingual_composed_candidate_runtime_v2 import _load_rule_display
from .bilingual_product_isolation import active_changed_rule_ids
from .bilingual_v8_adaptive_carrier import (
    ADAPTIVE_CARRIER_CANDIDATE_VERSION,
    BilingualAdaptiveCarrierRuntime,
)
from .config import ROOT, stable_json
from .util import sha256_file


COMPOSITION_STATE_V3_PATH = (
    ROOT / "rules" / "bilingual-kokoro-composition-candidate-v3.json"
)
COMPOSITION_STATE_V3_SHA256 = (
    "b6fcede002209be9a3d7b2fb2c2449e3bdb85bbc8be201c4cb2ca2cc7ce1d449"
)
COMPOSITION_CONTRACT_V4 = "bilingual-kokoro-controlled-pair-service-v4"
MULTI_RULE_ADAPTIVE_STATUS = "ready_automatic_only"
MULTI_RULE_ADAPTIVE_CLAIM = (
    "runtime_adaptive_composition_acoustic_pass_unseen_algorithm_pass_"
    "human_qc_pending"
)


def _semantic_hash(value: dict[str, Any]) -> str:
    payload = dict(value)
    payload.pop("record_sha256", None)
    return hashlib.sha256(stable_json(payload).encode("utf-8")).hexdigest()


def _load_composition_state_v3(
    path: Path = COMPOSITION_STATE_V3_PATH,
) -> dict[str, Any]:
    if sha256_file(path) != COMPOSITION_STATE_V3_SHA256:
        raise RuntimeError("the bilingual composition v3 state hash drifted")
    value = json.loads(path.read_text(encoding="utf-8"))
    evidence = value.get("evidence", {}) if isinstance(value, dict) else {}
    policy = value.get("runtime_policy", {}) if isinstance(value, dict) else {}
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != 1
        or value.get("candidate_id")
        != "bilingual-kokoro-vowel-composition-candidate-v3"
        or value.get("feature_flag") != "KOKORO_BILINGUAL_CANDIDATE_ENABLED"
        or value.get("enabled_by_default") is not False
        or value.get("production_enabled") is not False
        or value.get("service_contract_version") != COMPOSITION_CONTRACT_V4
        or evidence.get("unseen_composition_status")
        != "adaptive_algorithm_automatic_pass_3_of_3_two_rescues"
        or evidence.get("human_status") != "pending"
        or evidence.get("production_promotion") is not False
        or policy.get("composition_synthesis")
        != "combined_v8_with_deterministic_adaptive_carrier_retry"
        or policy.get("maximum_carrier_retry_rounds") != 5
        or policy.get("failed_or_exhausted_contexts_fail_closed") is not True
        or policy.get("human_composition_qc_eligible") is not True
        or policy.get("production_promotion_allowed") is not False
    ):
        raise RuntimeError("the bilingual composition v3 state contract drifted")
    bindings = (
        (value["base_candidate_state"]["path"], value["base_candidate_state"]["sha256"]),
        (
            value["prior_composition_state"]["path"],
            value["prior_composition_state"]["sha256"],
        ),
        (value["rule_display"]["path"], value["rule_display"]["sha256"]),
        (
            evidence["prior_unseen_result_path"],
            evidence["prior_unseen_result_sha256"],
        ),
        (
            evidence["known_correction_result_path"],
            evidence["known_correction_result_sha256"],
        ),
        (
            evidence["adaptive_unseen_result_path"],
            evidence["adaptive_unseen_result_sha256"],
        ),
    )
    if any(sha256_file(ROOT / bound_path) != digest for bound_path, digest in bindings):
        raise RuntimeError("a bilingual composition v3 evidence binding drifted")
    adaptive = json.loads(
        (ROOT / evidence["adaptive_unseen_result_path"]).read_text(encoding="utf-8")
    )
    if (
        adaptive.get("record_sha256")
        != evidence.get("adaptive_unseen_result_record_sha256")
        or _semantic_hash(adaptive)
        != evidence.get("adaptive_unseen_result_record_sha256")
        or adaptive.get("classification")
        != evidence.get("adaptive_unseen_classification")
        or adaptive.get("automatic_pass_count")
        != evidence.get("adaptive_unseen_pass_count")
        or adaptive.get("fixture_count")
        != evidence.get("adaptive_unseen_fixture_count")
        or adaptive.get("rescued_fixture_count")
        != evidence.get("adaptive_unseen_rescued_fixture_count")
        or adaptive.get("total_attempt_count")
        != evidence.get("adaptive_unseen_total_attempt_count")
        or adaptive.get("production_enabled") is not False
        or adaptive.get("api_calls_made") != 0
    ):
        raise RuntimeError("the adaptive unseen v3 result contract drifted")
    return value


class BilingualComposedCandidateRuntimeV3(BilingualComposedCandidateRuntime):
    """Disabled adaptive-carrier composition candidate with v4 contract."""

    def __init__(
        self,
        base: BilingualCandidateRuntime,
        *,
        state: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(base, state=state or _load_composition_state_v3())
        self.rule_display = _load_rule_display()

    @classmethod
    def load(
        cls, profile_id: str, voice_id: str
    ) -> BilingualComposedCandidateRuntimeV3:
        return cls(BilingualCandidateRuntime.load(profile_id, voice_id))

    def contract(self) -> dict[str, Any]:
        base = self.base.contract()
        evidence = self.state["evidence"]
        return {
            **base,
            "service_contract_version": COMPOSITION_CONTRACT_V4,
            "composition_candidate_id": self.state["candidate_id"],
            "composition_state_sha256": COMPOSITION_STATE_V3_SHA256,
            "composition_human_qc_status": evidence["human_status"],
            "composition_unseen_status": evidence["unseen_composition_status"],
        }

    def _display_result(self, result: dict[str, Any]) -> dict[str, Any]:
        transform = result.get("transform")
        if not isinstance(transform, dict) or "applied_rules" not in transform:
            return result
        display_rules = []
        for row in transform["applied_rules"]:
            display = self.rule_display.get(row["rule_id"])
            if (
                display is None
                or row["source_ipa"] != display["renderer_source"]
                or row["target_ipa"] != display["renderer_target"]
            ):
                raise RuntimeError("a bilingual applied-rule display mapping drifted")
            display_rules.append(
                {
                    **row,
                    "source_ipa": display["display_source"],
                    "target_ipa": display["display_target"],
                    "display_label": display["display_label"],
                }
            )
        return {
            **result,
            "transform": {**transform, "applied_rules": display_rules},
        }

    def render(self, text: str) -> dict[str, Any]:
        source_plan = self.base.base_planner.plan(text)
        changed_rule_ids = active_changed_rule_ids(source_plan)
        passing_cells = tuple(
            cell
            for rule_id in changed_rule_ids
            if (
                (cell := self.base.registry.cell(
                    source_plan.profile_id, source_plan.voice_id, rule_id
                ))
                is not None
                and cell.automatic_pass
            )
        )
        if not 2 <= len(passing_cells) <= 3 or any(
            cell.candidate_rung != "v8" for cell in passing_cells
        ):
            return self._display_result(
                self._extend_base_result(self.base.render(text))
            )

        started = time.perf_counter()
        selected_rule_ids = tuple(sorted(cell.rule_id for cell in passing_cells))
        cells_by_id = {cell.rule_id: cell for cell in passing_cells}
        cells = tuple(cells_by_id[rule_id] for rule_id in selected_rule_ids)
        planner = self.base._composition_planner(selected_rule_ids)
        adaptive = BilingualAdaptiveCarrierRuntime(
            base_planner=planner,
            synthesis=self.base.synthesis,
            cells=cells,
            scaler=self.base.scaler,
            maximum_retry_rounds=self.state["runtime_policy"][
                "maximum_carrier_retry_rounds"
            ],
        ).render(text)
        selected = adaptive.selected_attempt
        if not adaptive.automatic_pass or selected is None:
            raise BilingualCandidateAcousticGateError(
                "runtime_acoustic_gate_rejected",
                "The adaptive composition exhausted its bounded carrier retries.",
            )
        selected_counts = {
            cell.rule_id: _count_rule_occurrences(selected.plan, cell.rule_id)
            for cell in cells
        }
        source_counts = {
            cell.rule_id: _count_rule_occurrences(source_plan, cell.rule_id)
            for cell in cells
        }
        if (
            any(count <= 0 for count in selected_counts.values())
            or selected_counts != source_counts
        ):
            raise RuntimeError("an adaptive candidate changed its occurrence denominator")
        omitted_rule_ids = tuple(
            rule_id for rule_id in changed_rule_ids if rule_id not in selected_rule_ids
        )
        target_occurrence_count = sum(selected_counts.values())
        rendered = selected.render
        result = {
            "schema_version": 1,
            "status": MULTI_RULE_ADAPTIVE_STATUS,
            "claim_tier": MULTI_RULE_ADAPTIVE_CLAIM,
            "candidate_contract": self.contract(),
            "transform": {
                "schema_version": 1,
                "profile_id": selected.plan.profile_id,
                "voice_id": selected.plan.voice_id,
                "original_text": selected.plan.normalized_text,
                "neutral_script": selected.plan.neutral_script,
                "lens_script": selected.plan.lens_script,
                "comparison_available": True,
                "plan_sha256": selected.plan.plan_sha256,
                "composition_mode": "multi_rule_v8",
                "applied_rules": [
                    {
                        "rule_id": cell.rule_id,
                        "source_ipa": cell.source,
                        "target_ipa": cell.target,
                        "occurrences": selected_counts[cell.rule_id],
                    }
                    for cell in cells
                ],
                "omitted_rule_ids": list(omitted_rule_ids),
                "partial_profile_coverage": bool(omitted_rule_ids),
            },
            "audio": {
                "neutral": _audio_record(rendered.neutral_pcm),
                "lens": _audio_record(rendered.lens_pcm),
            },
            "verification": {
                "status": "runtime_acoustic_gates_passed",
                "plan_sha256": selected.plan.plan_sha256,
                "target_occurrence_count": target_occurrence_count,
                "neutral_pcm_sha256": _pcm_hash(rendered.neutral_pcm),
                "identity_pcm_sha256": _pcm_hash(rendered.identity_pcm),
                "lens_pcm_sha256": _pcm_hash(rendered.lens_pcm),
                "render_integrity": asdict(rendered.verification),
                "acoustic": selected.acoustic,
                "adaptive_carrier": {
                    "version": ADAPTIVE_CARRIER_CANDIDATE_VERSION,
                    "attempt_count": len(adaptive.attempts),
                    "selected_round_index": adaptive.selected_round_index,
                    "rescued_after_retry": adaptive.rescued_after_retry,
                    "maximum_retry_rounds": self.state["runtime_policy"][
                        "maximum_carrier_retry_rounds"
                    ],
                },
                "elapsed_ms": (time.perf_counter() - started) * 1_000.0,
                "api_calls_made": 0,
            },
            "cache_hit": False,
            "api_calls_made": 0,
        }
        return self._display_result(result)
