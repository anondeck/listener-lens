from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .bilingual_candidate_runtime import BilingualCandidateRuntime
from .bilingual_composed_candidate_runtime import BilingualComposedCandidateRuntime
from .config import ROOT, stable_json
from .util import sha256_file


COMPOSITION_STATE_V2_PATH = (
    ROOT / "rules" / "bilingual-kokoro-composition-candidate-v2.json"
)
COMPOSITION_STATE_V2_SHA256 = (
    "d74f4bc22db405390b7aa2d1320100adab7c15ae1e15bb6dc1bddd683de725ab"
)
RULE_DISPLAY_PATH = ROOT / "rules" / "bilingual-rule-display-v1.json"
RULE_DISPLAY_SHA256 = (
    "c1ca4651ac9efef22a37605f1e96d7e8eaec945551705cf876b008d55af1113e"
)
COMPOSITION_CONTRACT_V3 = "bilingual-kokoro-controlled-pair-service-v3"
MULTI_RULE_AUTOMATIC_ONLY_STATUS = "ready_automatic_only"
MULTI_RULE_AUTOMATIC_ONLY_CLAIM = (
    "runtime_composition_acoustic_pass_unseen_aggregate_failed_no_human_claim"
)


def _semantic_hash(value: dict[str, Any]) -> str:
    payload = dict(value)
    payload.pop("record_sha256", None)
    return hashlib.sha256(stable_json(payload).encode("utf-8")).hexdigest()


def _load_composition_state_v2(
    path: Path = COMPOSITION_STATE_V2_PATH,
) -> dict[str, Any]:
    if sha256_file(path) != COMPOSITION_STATE_V2_SHA256:
        raise RuntimeError("the bilingual composition v2 state hash drifted")
    value = json.loads(path.read_text(encoding="utf-8"))
    evidence = value.get("evidence", {}) if isinstance(value, dict) else {}
    policy = value.get("runtime_policy", {}) if isinstance(value, dict) else {}
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != 1
        or value.get("candidate_id")
        != "bilingual-kokoro-vowel-composition-candidate-v2"
        or value.get("feature_flag") != "KOKORO_BILINGUAL_CANDIDATE_ENABLED"
        or value.get("enabled_by_default") is not False
        or value.get("production_enabled") is not False
        or value.get("service_contract_version") != COMPOSITION_CONTRACT_V3
        or evidence.get("unseen_composition_status") != "automatic_failed_2_of_3"
        or evidence.get("human_status")
        != "not_eligible_after_unseen_aggregate_failure"
        or evidence.get("production_promotion") is not False
        or policy.get("composition_synthesis") != "combined_v8_one_decode"
        or policy.get("failed_contexts_fail_closed") is not True
        or policy.get("human_composition_qc_eligible") is not False
        or policy.get("production_promotion_allowed") is not False
    ):
        raise RuntimeError("the bilingual composition v2 state contract drifted")
    bindings = (
        (value["base_candidate_state"]["path"], value["base_candidate_state"]["sha256"]),
        (
            value["prior_composition_state"]["path"],
            value["prior_composition_state"]["sha256"],
        ),
        (value["rule_display"]["path"], value["rule_display"]["sha256"]),
        (
            evidence["known_composition_result_path"],
            evidence["known_composition_result_sha256"],
        ),
        (
            evidence["unseen_composition_result_path"],
            evidence["unseen_composition_result_sha256"],
        ),
    )
    if any(sha256_file(ROOT / bound_path) != digest for bound_path, digest in bindings):
        raise RuntimeError("a bilingual composition v2 evidence binding drifted")
    result = json.loads(
        (ROOT / evidence["unseen_composition_result_path"]).read_text(
            encoding="utf-8"
        )
    )
    if (
        result.get("record_sha256")
        != evidence.get("unseen_composition_result_record_sha256")
        or _semantic_hash(result)
        != evidence.get("unseen_composition_result_record_sha256")
        or result.get("classification")
        != evidence.get("unseen_composition_classification")
        or result.get("automatic_pass_count")
        != evidence.get("unseen_composition_pass_count")
        or result.get("fixture_count")
        != evidence.get("unseen_composition_fixture_count")
        or result.get("production_enabled") is not False
        or result.get("api_calls_made") != 0
    ):
        raise RuntimeError("the unseen composition v2 result contract drifted")
    return value


def _load_rule_display(path: Path = RULE_DISPLAY_PATH) -> dict[str, dict[str, str]]:
    if sha256_file(path) != RULE_DISPLAY_SHA256:
        raise RuntimeError("the bilingual rule-display hash drifted")
    value = json.loads(path.read_text(encoding="utf-8"))
    rows = value.get("rules") if isinstance(value, dict) else None
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != 1
        or value.get("display_version") != "bilingual-rule-display-v1"
        or not isinstance(rows, list)
        or len(rows) != 13
    ):
        raise RuntimeError("the bilingual rule-display contract drifted")
    expected_keys = {
        "display_label",
        "display_source",
        "display_target",
        "renderer_source",
        "renderer_target",
        "rule_id",
    }
    by_id = {row.get("rule_id"): row for row in rows if isinstance(row, dict)}
    if (
        len(by_id) != len(rows)
        or None in by_id
        or any(set(row) != expected_keys for row in rows)
        or any(
            not isinstance(item, str) or not item
            for row in rows
            for item in row.values()
        )
    ):
        raise RuntimeError("the bilingual rule-display rows drifted")
    return by_id


class BilingualComposedCandidateRuntimeV2(BilingualComposedCandidateRuntime):
    """Truthful post-confirmation wrapper for the disabled composition path."""

    def __init__(
        self,
        base: BilingualCandidateRuntime,
        *,
        state: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(base, state=state or _load_composition_state_v2())
        self.rule_display = _load_rule_display()

    @classmethod
    def load(
        cls, profile_id: str, voice_id: str
    ) -> BilingualComposedCandidateRuntimeV2:
        return cls(BilingualCandidateRuntime.load(profile_id, voice_id))

    def contract(self) -> dict[str, Any]:
        base = self.base.contract()
        evidence = self.state["evidence"]
        return {
            **base,
            "service_contract_version": COMPOSITION_CONTRACT_V3,
            "composition_candidate_id": self.state["candidate_id"],
            "composition_state_sha256": COMPOSITION_STATE_V2_SHA256,
            "composition_human_qc_status": evidence["human_status"],
            "composition_unseen_status": evidence["unseen_composition_status"],
        }

    def render(self, text: str) -> dict[str, Any]:
        result = super().render(text)
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
        result = {
            **result,
            "transform": {**transform, "applied_rules": display_rules},
        }
        if transform.get("composition_mode") != "multi_rule_v8":
            return result
        return {
            **result,
            "status": MULTI_RULE_AUTOMATIC_ONLY_STATUS,
            "claim_tier": MULTI_RULE_AUTOMATIC_ONLY_CLAIM,
        }
