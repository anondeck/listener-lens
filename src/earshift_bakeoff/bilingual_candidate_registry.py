from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from .bilingual_product_isolation import active_changed_rule_ids
from .config import ROOT, stable_json
from .util import sha256_file


BILINGUAL_CANDIDATE_STATE_PATH = (
    ROOT / "rules" / "bilingual-kokoro-candidate-state-v1.json"
)
BILINGUAL_CANDIDATE_FEATURE_FLAG = "KOKORO_BILINGUAL_CANDIDATE_ENABLED"
EXPECTED_RESULT_CLASSIFICATION = (
    "unseen_typed_confirmation_complete_no_product_promotion"
)
SUPPORTED_CANDIDATE_RUNGS = frozenset(
    {"v8", "word_context", "full_context", "adaptive_strength"}
)


class BilingualCandidateRegistryError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class BilingualCandidateCell:
    cell_id: str
    profile_id: str
    voice_id: str
    rule_id: str
    source: str
    target: str
    candidate_rung: str
    automatic_classification: str
    automatic_pass: bool
    human_status: str
    product_enabled: bool

    def safe_metadata(self) -> dict[str, Any]:
        return {
            "cell_id": self.cell_id,
            "profile_id": self.profile_id,
            "voice_id": self.voice_id,
            "rule_id": self.rule_id,
            "candidate_rung": self.candidate_rung,
            "automatic_classification": self.automatic_classification,
            "automatic_pass": self.automatic_pass,
            "human_status": self.human_status,
            "product_enabled": self.product_enabled,
        }


@dataclass(frozen=True)
class CandidatePlanDecision:
    status: str
    profile_id: str
    voice_id: str
    changed_rule_ids: tuple[str, ...]
    omitted_rule_ids: tuple[str, ...]
    cell: BilingualCandidateCell | None
    blockers: tuple[str, ...]

    @property
    def render_eligible(self) -> bool:
        return self.status == "eligible_automatic_candidate"

    def safe_metadata(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "profile_id": self.profile_id,
            "voice_id": self.voice_id,
            "changed_rule_ids": self.changed_rule_ids,
            "omitted_rule_ids": self.omitted_rule_ids,
            "cell": None if self.cell is None else self.cell.safe_metadata(),
            "blockers": self.blockers,
            "render_eligible": self.render_eligible,
        }


@dataclass(frozen=True)
class BilingualCandidateRegistry:
    candidate_id: str
    feature_flag: str
    service_contract_version: str
    state_sha256: str
    result_sha256: str
    result_record_sha256: str
    runtime_gate_result_sha256: str
    runtime_gate_scaler_sha256: str
    production_enabled: bool
    runtime_policy: dict[str, Any]
    cells: tuple[BilingualCandidateCell, ...]

    def cell(
        self, profile_id: str, voice_id: str, rule_id: str
    ) -> BilingualCandidateCell | None:
        return next(
            (
                cell
                for cell in self.cells
                if cell.profile_id == profile_id
                and cell.voice_id == voice_id
                and cell.rule_id == rule_id
            ),
            None,
        )

    def evaluate_plan(self, plan: Any) -> CandidatePlanDecision:
        profile_id = str(plan.profile_id)
        voice_id = str(plan.voice_id)
        changed_rule_ids = active_changed_rule_ids(plan)
        if not changed_rule_ids:
            return CandidatePlanDecision(
                status="no_supported_sounds",
                profile_id=profile_id,
                voice_id=voice_id,
                changed_rule_ids=(),
                omitted_rule_ids=(),
                cell=None,
                blockers=("no_changed_listener_rules",),
            )
        matching_cells = tuple(
            cell
            for rule_id in changed_rule_ids
            if (cell := self.cell(profile_id, voice_id, rule_id)) is not None
        )
        passing_cells = tuple(cell for cell in matching_cells if cell.automatic_pass)
        if len(passing_cells) > 1:
            return CandidatePlanDecision(
                status="unsupported_rule_composition",
                profile_id=profile_id,
                voice_id=voice_id,
                changed_rule_ids=changed_rule_ids,
                omitted_rule_ids=(),
                cell=None,
                blockers=("multiple_supported_vowel_rules_not_composition_validated",),
            )
        if not passing_cells:
            failed_cells = tuple(cell for cell in matching_cells if not cell.automatic_pass)
            if failed_cells:
                cell = failed_cells[0]
                return CandidatePlanDecision(
                    status="automatic_evidence_failed",
                    profile_id=profile_id,
                    voice_id=voice_id,
                    changed_rule_ids=changed_rule_ids,
                    omitted_rule_ids=changed_rule_ids,
                    cell=cell,
                    blockers=(
                        f"unseen_automatic_{cell.automatic_classification}",
                    ),
                )
            return CandidatePlanDecision(
                status="unsupported_rule_or_voice",
                profile_id=profile_id,
                voice_id=voice_id,
                changed_rule_ids=changed_rule_ids,
                omitted_rule_ids=changed_rule_ids,
                cell=None,
                blockers=("no_unseen_oral_confirmation_cell",),
            )
        cell = passing_cells[0]
        omitted_rule_ids = tuple(
            rule_id for rule_id in changed_rule_ids if rule_id != cell.rule_id
        )
        return CandidatePlanDecision(
            status="eligible_automatic_candidate",
            profile_id=profile_id,
            voice_id=voice_id,
            changed_rule_ids=changed_rule_ids,
            omitted_rule_ids=omitted_rule_ids,
            cell=cell,
            blockers=(
                "per_request_acoustic_gate_required",
                "blind_human_qc_pending",
                *(("partial_profile_coverage",) if omitted_rule_ids else ()),
                "production_disabled",
            ),
        )

    def safe_catalog(self) -> dict[str, Any]:
        passing = [cell for cell in self.cells if cell.automatic_pass]
        return {
            "schema_version": 1,
            "candidate_id": self.candidate_id,
            "service_contract_version": self.service_contract_version,
            "production_enabled": self.production_enabled,
            "oral_confirmed_cell_count": len(self.cells),
            "unseen_automatic_pass_count": len(passing),
            "runtime_gate_pass_count": len(passing),
            "human_qc_pass_count": sum(
                cell.human_status == "pass" for cell in passing
            ),
            "product_enabled_cell_count": sum(
                cell.product_enabled for cell in passing
            ),
            "voices": {
                voice_id: {
                    "automatic_pass_rule_ids": sorted(
                        cell.rule_id
                        for cell in passing
                        if cell.voice_id == voice_id
                    ),
                    "automatic_pass_count": sum(
                        cell.voice_id == voice_id for cell in passing
                    ),
                }
                for voice_id in sorted({cell.voice_id for cell in self.cells})
            },
            "runtime_policy": self.runtime_policy,
        }


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BilingualCandidateRegistryError(
            "candidate_state_unreadable", f"Cannot read {path}."
        ) from exc
    if not isinstance(value, dict):
        raise BilingualCandidateRegistryError(
            "candidate_state_invalid", f"Expected an object in {path}."
        )
    return value


def _semantic_hash(payload: dict[str, Any]) -> str:
    semantic = dict(payload)
    semantic.pop("record_sha256", None)
    return hashlib.sha256(stable_json(semantic).encode("utf-8")).hexdigest()


def load_bilingual_candidate_registry(
    path: Path = BILINGUAL_CANDIDATE_STATE_PATH,
) -> BilingualCandidateRegistry:
    state = _load_object(path)
    expected_keys = {
        "schema_version",
        "candidate_id",
        "feature_flag",
        "enabled_by_default",
        "production_enabled",
        "service_contract_version",
        "evidence",
        "runtime_policy",
    }
    if set(state) != expected_keys:
        raise BilingualCandidateRegistryError(
            "candidate_state_schema_drift", "Candidate state has unexpected fields."
        )
    evidence = state["evidence"]
    runtime_policy = state["runtime_policy"]
    if (
        state["schema_version"] != 1
        or state["feature_flag"] != BILINGUAL_CANDIDATE_FEATURE_FLAG
        or state["enabled_by_default"] is not False
        or state["production_enabled"] is not False
        or evidence.get("classification") != EXPECTED_RESULT_CLASSIFICATION
        or evidence.get("human_status") != "pending"
        or evidence.get("production_promotion") is not False
        or runtime_policy.get("family") != "vowel"
        or runtime_policy.get("isolate_one_changed_rule") is not True
        or runtime_policy.get("multiple_occurrences_of_same_rule_allowed") is not True
        or runtime_policy.get("multiple_changed_rules_allowed") is not False
        or runtime_policy.get("source_plan_may_contain_omitted_unvalidated_rules")
        is not True
        or runtime_policy.get("omitted_rules_must_be_reported") is not True
        or runtime_policy.get("consonants_enabled") is not False
        or runtime_policy.get("insertions_enabled") is not False
        or runtime_policy.get("prosody_enabled") is not False
        or runtime_policy.get("nasal_vowels_enabled") is not False
        or runtime_policy.get("arbitrary_text_acoustic_gate")
        != "frozen_runtime_instrument_required_before_audio_response"
        or runtime_policy.get("human_qc_required_before_product_promotion") is not True
    ):
        raise BilingualCandidateRegistryError(
            "candidate_state_policy_drift", "Candidate policy is not fail-closed."
        )
    protocol_path = ROOT / str(evidence["protocol_path"])
    result_path = ROOT / str(evidence["result_path"])
    runtime_protocol_path = ROOT / str(evidence["runtime_gate_protocol_path"])
    runtime_result_path = ROOT / str(evidence["runtime_gate_result_path"])
    runtime_scaler_path = ROOT / str(evidence["runtime_gate_scaler_path"])
    if sha256_file(protocol_path) != evidence.get("protocol_sha256"):
        raise BilingualCandidateRegistryError(
            "candidate_protocol_hash_mismatch", "Frozen protocol hash changed."
        )
    if sha256_file(result_path) != evidence.get("result_sha256"):
        raise BilingualCandidateRegistryError(
            "candidate_result_hash_mismatch", "Frozen result hash changed."
        )
    if (
        sha256_file(runtime_protocol_path)
        != evidence.get("runtime_gate_protocol_sha256")
        or sha256_file(runtime_result_path)
        != evidence.get("runtime_gate_result_sha256")
        or sha256_file(runtime_scaler_path)
        != evidence.get("runtime_gate_scaler_sha256")
    ):
        raise BilingualCandidateRegistryError(
            "candidate_runtime_gate_hash_mismatch",
            "Frozen runtime-gate evidence changed.",
        )
    result = _load_object(result_path)
    runtime_result = _load_object(runtime_result_path)
    runtime_scalers = _load_object(runtime_scaler_path)
    if (
        result.get("record_sha256") != evidence.get("result_record_sha256")
        or _semantic_hash(result) != evidence.get("result_record_sha256")
        or result.get("classification") != evidence.get("classification")
        or result.get("production_enabled") is not False
        or result.get("api_calls_made") != 0
        or result.get("oral_candidate_cell_count")
        != evidence.get("oral_confirmed_cell_count")
        or result.get("unseen_automatic_pass_count")
        != evidence.get("unseen_automatic_pass_count")
    ):
        raise BilingualCandidateRegistryError(
            "candidate_result_contract_mismatch",
            "Frozen unseen-confirmation result does not match candidate state.",
        )
    if (
        runtime_result.get("record_sha256")
        != evidence.get("runtime_gate_result_record_sha256")
        or _semantic_hash(runtime_result)
        != evidence.get("runtime_gate_result_record_sha256")
        or runtime_result.get("classification")
        != "runtime_gate_complete_no_product_promotion"
        or runtime_result.get("production_enabled") is not False
        or runtime_result.get("api_calls_made") != 0
        or runtime_result.get("runtime_gate_pass_count")
        != evidence.get("runtime_gate_pass_count")
        or runtime_result.get("lost_prior_pass_cell_ids") != []
        or runtime_scalers.get("record_sha256")
        != evidence.get("runtime_gate_scaler_record_sha256")
        or _semantic_hash(runtime_scalers)
        != evidence.get("runtime_gate_scaler_record_sha256")
        or runtime_result.get("voice_scaler_sha256")
        != evidence.get("runtime_gate_scaler_sha256")
        or set(runtime_scalers.get("voice_scalers", {}))
        != {"af_heart", "am_michael", "pm_alex", "pf_dora"}
    ):
        raise BilingualCandidateRegistryError(
            "candidate_runtime_gate_contract_mismatch",
            "Frozen runtime-gate result does not match candidate state.",
        )
    summaries = result.get("cell_summaries")
    if not isinstance(summaries, list) or len(summaries) != evidence.get(
        "oral_confirmed_cell_count"
    ):
        raise BilingualCandidateRegistryError(
            "candidate_cell_denominator_mismatch", "Candidate cells are incomplete."
        )
    cells: list[BilingualCandidateCell] = []
    runtime_rows = runtime_result.get("cell_results")
    if not isinstance(runtime_rows, list) or len(runtime_rows) != len(summaries):
        raise BilingualCandidateRegistryError(
            "candidate_runtime_cell_denominator_mismatch",
            "Runtime-gate cells are incomplete.",
        )
    runtime_cells = {row.get("cell_id"): row for row in runtime_rows}
    summary_cell_ids = {row.get("cell_id") for row in summaries}
    if (
        len(runtime_cells) != len(runtime_rows)
        or set(runtime_cells) != summary_cell_ids
        or None in runtime_cells
    ):
        raise BilingualCandidateRegistryError(
            "candidate_runtime_cell_identity_mismatch",
            "Runtime-gate cells are duplicated, missing, or unexpected.",
        )
    for row in summaries:
        rung = row.get("candidate_rung")
        classification = row.get("replicated_anchor", {}).get("classification")
        automatic_pass = row.get("unseen_automatic_pass")
        runtime_cell = runtime_cells.get(row.get("cell_id"))
        if (
            rung not in SUPPORTED_CANDIDATE_RUNGS
            or classification
            not in {
                "exact_category_pass",
                "directional_only_pass",
                "fail",
                "anchor_validation_fail",
            }
            or not isinstance(automatic_pass, bool)
            or row.get("product_enabled") is not False
            or bool(classification in {"exact_category_pass", "directional_only_pass"})
            != automatic_pass
            or runtime_cell is None
            or bool(runtime_cell.get("runtime_gate_pass")) != automatic_pass
            or runtime_cell.get("product_enabled") is not False
        ):
            raise BilingualCandidateRegistryError(
                "candidate_cell_contract_mismatch",
                "An unseen-confirmation cell has an invalid state.",
            )
        cells.append(
            BilingualCandidateCell(
                cell_id=str(row["cell_id"]),
                profile_id=str(row["profile_id"]),
                voice_id=str(row["voice_id"]),
                rule_id=str(row["rule_id"]),
                source=str(row["source"]),
                target=str(row["target"]),
                candidate_rung=str(rung),
                automatic_classification=str(classification),
                automatic_pass=automatic_pass,
                human_status="pending" if automatic_pass else "not_eligible",
                product_enabled=False,
            )
        )
    keys = [(cell.profile_id, cell.voice_id, cell.rule_id) for cell in cells]
    if len(keys) != len(set(keys)) or sum(cell.automatic_pass for cell in cells) != 18:
        raise BilingualCandidateRegistryError(
            "candidate_cell_identity_mismatch", "Candidate cells are duplicated or lost."
        )
    if (
        evidence.get("practical_core_cell_count") != 36
        or evidence.get("unseen_automatic_nonpass_count") != 10
        or evidence.get("nasal_pending_cell_count") != 3
        or evidence.get("earlier_blocked_cell_count") != 5
        or 18 + 10 + 3 + 5 != 36
    ):
        raise BilingualCandidateRegistryError(
            "candidate_denominator_mismatch", "Practical-core accounting drifted."
        )
    passing_rule_ids_by_voice = {
        voice_id: sorted(
            cell.rule_id
            for cell in cells
            if cell.voice_id == voice_id and cell.automatic_pass
        )
        for voice_id in sorted({cell.voice_id for cell in cells})
    }
    if evidence.get("runtime_gate_pass_rule_ids_by_voice") != (
        passing_rule_ids_by_voice
    ):
        raise BilingualCandidateRegistryError(
            "candidate_runtime_catalog_mismatch",
            "The safe per-voice runtime catalog does not match frozen evidence.",
        )
    return BilingualCandidateRegistry(
        candidate_id=str(state["candidate_id"]),
        feature_flag=str(state["feature_flag"]),
        service_contract_version=str(state["service_contract_version"]),
        state_sha256=sha256_file(path),
        result_sha256=str(evidence["result_sha256"]),
        result_record_sha256=str(evidence["result_record_sha256"]),
        runtime_gate_result_sha256=str(evidence["runtime_gate_result_sha256"]),
        runtime_gate_scaler_sha256=str(evidence["runtime_gate_scaler_sha256"]),
        production_enabled=False,
        runtime_policy=dict(runtime_policy),
        cells=tuple(cells),
    )
