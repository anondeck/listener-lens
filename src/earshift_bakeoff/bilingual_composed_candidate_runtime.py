from __future__ import annotations

from dataclasses import asdict
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
from .bilingual_product_isolation import active_changed_rule_ids
from .config import ROOT
from .util import sha256_file


COMPOSITION_STATE_PATH = (
    ROOT / "rules" / "bilingual-kokoro-composition-candidate-v1.json"
)
COMPOSITION_STATE_SHA256 = (
    "eb37f63da7e27357ea1f3032af0f92bad207809c214809de743d0e2f5b83ed12"
)
COMPOSITION_CONTRACT_VERSION = "bilingual-kokoro-controlled-pair-service-v2"


def _load_composition_state(path: Path = COMPOSITION_STATE_PATH) -> dict[str, Any]:
    if sha256_file(path) != COMPOSITION_STATE_SHA256:
        raise RuntimeError("the bilingual composition state hash drifted")
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != 1
        or value.get("candidate_id")
        != "bilingual-kokoro-vowel-composition-candidate-v1"
        or value.get("feature_flag") != "KOKORO_BILINGUAL_CANDIDATE_ENABLED"
        or value.get("enabled_by_default") is not False
        or value.get("production_enabled") is not False
        or value.get("service_contract_version") != COMPOSITION_CONTRACT_VERSION
        or value.get("runtime_policy", {}).get("composition_synthesis")
        != "combined_v8_one_decode"
        or value.get("runtime_policy", {}).get("failed_contexts_fail_closed")
        is not True
    ):
        raise RuntimeError("the bilingual composition state contract drifted")
    bindings = (
        (value["base_candidate_state"]["path"], value["base_candidate_state"]["sha256"]),
        (
            value["evidence"]["composition_v1_result_path"],
            value["evidence"]["composition_v1_result_sha256"],
        ),
        (
            value["evidence"]["separated_decoder_v3_result_path"],
            value["evidence"]["separated_decoder_v3_result_sha256"],
        ),
    )
    if any(sha256_file(ROOT / path_value) != digest for path_value, digest in bindings):
        raise RuntimeError("a bilingual composition evidence binding drifted")
    return value


class BilingualComposedCandidateRuntime:
    """Disabled, coverage-gated single/multi-rule candidate response runtime."""

    def __init__(
        self,
        base: BilingualCandidateRuntime,
        *,
        state: dict[str, Any] | None = None,
    ) -> None:
        self.base = base
        self.state = state or _load_composition_state()

    @classmethod
    def load(
        cls, profile_id: str, voice_id: str
    ) -> BilingualComposedCandidateRuntime:
        return cls(BilingualCandidateRuntime.load(profile_id, voice_id))

    def contract(self) -> dict[str, Any]:
        base = self.base.contract()
        evidence = self.state["evidence"]
        return {
            **base,
            "service_contract_version": COMPOSITION_CONTRACT_VERSION,
            "composition_candidate_id": self.state["candidate_id"],
            "composition_state_sha256": COMPOSITION_STATE_SHA256,
            "composition_human_qc_status": evidence["human_status"],
            "composition_unseen_status": evidence["unseen_composition_status"],
        }

    def _extend_base_result(self, result: dict[str, Any]) -> dict[str, Any]:
        extended = {**result, "candidate_contract": self.contract()}
        if result.get("status") == "ready_pending_human_qc":
            extended["transform"] = {
                **result["transform"],
                "composition_mode": "single_rule",
            }
        return extended

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
            return self._extend_base_result(self.base.render(text))

        started = time.perf_counter()
        candidate = self.base.render_v8_composition_candidate(text)
        if not candidate.acoustic["pass"]:
            raise BilingualCandidateAcousticGateError(
                "runtime_acoustic_gate_rejected",
                "One or more composed occurrences failed the current-context gate.",
            )
        selected_counts = {
            cell.rule_id: _count_rule_occurrences(
                candidate.isolated_plan, cell.rule_id
            )
            for cell in candidate.cells
        }
        if any(count <= 0 for count in selected_counts.values()):
            raise RuntimeError("a composed candidate lost a selected occurrence")
        target_occurrence_count = sum(selected_counts.values())
        rendered = candidate.render
        return {
            "schema_version": 1,
            "status": "ready_pending_human_qc",
            "claim_tier": (
                "runtime_composition_acoustic_pass_human_qc_pending"
            ),
            "candidate_contract": self.contract(),
            "transform": {
                "schema_version": 1,
                "profile_id": candidate.isolated_plan.profile_id,
                "voice_id": candidate.isolated_plan.voice_id,
                "original_text": candidate.isolated_plan.normalized_text,
                "neutral_script": candidate.isolated_plan.neutral_script,
                "lens_script": candidate.isolated_plan.lens_script,
                "comparison_available": True,
                "plan_sha256": candidate.isolated_plan.plan_sha256,
                "composition_mode": "multi_rule_v8",
                "applied_rules": [
                    {
                        "rule_id": cell.rule_id,
                        "source_ipa": cell.source,
                        "target_ipa": cell.target,
                        "occurrences": selected_counts[cell.rule_id],
                    }
                    for cell in candidate.cells
                ],
                "omitted_rule_ids": list(candidate.omitted_rule_ids),
                "partial_profile_coverage": bool(candidate.omitted_rule_ids),
            },
            "audio": {
                "neutral": _audio_record(rendered.neutral_pcm),
                "lens": _audio_record(rendered.lens_pcm),
            },
            "verification": {
                "status": "runtime_acoustic_gates_passed",
                "plan_sha256": candidate.isolated_plan.plan_sha256,
                "target_occurrence_count": target_occurrence_count,
                "neutral_pcm_sha256": _pcm_hash(rendered.neutral_pcm),
                "identity_pcm_sha256": _pcm_hash(rendered.identity_pcm),
                "lens_pcm_sha256": _pcm_hash(rendered.lens_pcm),
                "render_integrity": asdict(rendered.verification),
                "acoustic": candidate.acoustic,
                "elapsed_ms": (time.perf_counter() - started) * 1_000.0,
                "api_calls_made": 0,
            },
            "cache_hit": False,
            "api_calls_made": 0,
        }
