from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from .config import ROOT, stable_json
from .product_voices import ProductVoiceRegistry, load_product_voice_registry
from .util import sha256_file


BILINGUAL_PRODUCT_MATRIX_PATH = (
    ROOT / "rules" / "bilingual-product-matrix-v1.json"
)
BILINGUAL_PRODUCT_MATRIX_VERSION = "bilingual-product-matrix-v1"
BILINGUAL_PRODUCT_STRUCTURAL_STATE_PATH = (
    ROOT / "rules" / "bilingual-product-structural-state-v1.json"
)
BILINGUAL_PRODUCT_STRUCTURAL_STATE_VERSION = (
    "bilingual-product-structural-state-v1"
)


class BilingualProductMatrixError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class ProductRule:
    profile_id: str
    source_language: str
    listener_language: str
    family: str
    rule_id: str
    source: str | None
    target: str | None
    operation: str
    contexts: tuple[str, ...]
    evidence_tier: str
    acoustic_status: str
    source_ids: tuple[str, ...]
    changed: bool


@dataclass(frozen=True)
class ValidationCell:
    cell_id: str
    profile_id: str
    source_language: str
    listener_language: str
    voice_id: str
    family: str
    rule_id: str
    source: str | None
    target: str | None
    operation: str
    contexts: tuple[str, ...]
    evidence_tier: str
    acoustic_status: str
    source_ids: tuple[str, ...]
    changed: bool
    runtime_status: str
    human_status: str
    product_enabled: bool
    claim_tier: str
    reference_rule_id: str | None = None
    binding_id: str | None = None
    note: str | None = None

    def safe_metadata(self) -> dict[str, Any]:
        return {
            "cell_id": self.cell_id,
            "profile_id": self.profile_id,
            "source_language": self.source_language,
            "listener_language": self.listener_language,
            "voice_id": self.voice_id,
            "family": self.family,
            "rule_id": self.rule_id,
            "source": self.source,
            "target": self.target,
            "operation": self.operation,
            "contexts": self.contexts,
            "evidence_tier": self.evidence_tier,
            "acoustic_status": self.acoustic_status,
            "changed": self.changed,
            "runtime_status": self.runtime_status,
            "human_status": self.human_status,
            "product_enabled": self.product_enabled,
            "claim_tier": self.claim_tier,
        }


@dataclass(frozen=True)
class PlanEvidenceReport:
    matrix_version: str
    matrix_sha256: str
    profile_id: str
    voice_id: str
    changed_rule_ids: tuple[str, ...]
    cells: tuple[ValidationCell, ...]
    product_ready: bool
    blockers: tuple[str, ...]

    def safe_metadata(self) -> dict[str, Any]:
        return {
            "matrix_version": self.matrix_version,
            "matrix_sha256": self.matrix_sha256,
            "profile_id": self.profile_id,
            "voice_id": self.voice_id,
            "changed_rule_ids": self.changed_rule_ids,
            "cells": [cell.safe_metadata() for cell in self.cells],
            "product_ready": self.product_ready,
            "blockers": self.blockers,
        }


@dataclass(frozen=True)
class BilingualProductMatrix:
    matrix_version: str
    matrix_sha256: str
    rules_sha256: str
    production_enabled: bool
    cells: tuple[ValidationCell, ...]
    fixture_policy: dict[str, Any]

    def __post_init__(self) -> None:
        keys = [
            (cell.profile_id, cell.voice_id, cell.rule_id) for cell in self.cells
        ]
        if len(keys) != len(set(keys)):
            raise BilingualProductMatrixError(
                "duplicate_validation_cell", "Validation cells are duplicated."
            )

    def cell(
        self, profile_id: str, voice_id: str, rule_id: str
    ) -> ValidationCell:
        for cell in self.cells:
            if (
                cell.profile_id == profile_id
                and cell.voice_id == voice_id
                and cell.rule_id == rule_id
            ):
                return cell
        raise BilingualProductMatrixError(
            "unknown_validation_cell",
            f"No validation cell exists for {profile_id}/{voice_id}/{rule_id}.",
        )

    def evaluate_plan(self, plan: Any) -> PlanEvidenceReport:
        profile_id = str(plan.profile_id)
        voice_id = str(plan.voice_id)
        changed_rule_ids: set[str] = set()
        for word in plan.words:
            for attribute in (
                "vowel_occurrences",
                "consonant_occurrences",
                "insertion_occurrences",
                "prosody_occurrences",
            ):
                for occurrence in getattr(word, attribute, ()):
                    if occurrence.changed:
                        changed_rule_ids.add(occurrence.rule_id)
        changed_rule_ids.update(getattr(plan, "active_prosody_rule_ids", ()))
        ordered_rule_ids = tuple(sorted(changed_rule_ids))
        cells = tuple(
            self.cell(profile_id, voice_id, rule_id)
            for rule_id in ordered_rule_ids
        )
        blockers: list[str] = []
        if not cells:
            blockers.append("no_changed_listener_rules")
        for cell in cells:
            if not cell.product_enabled:
                blockers.append(
                    f"{cell.rule_id}@{cell.voice_id}:"
                    f"runtime={cell.runtime_status},human={cell.human_status}"
                )
        product_ready = bool(
            cells
            and self.production_enabled
            and all(cell.product_enabled for cell in cells)
        )
        if cells and not self.production_enabled:
            blockers.append("matrix_production_disabled")
        return PlanEvidenceReport(
            matrix_version=self.matrix_version,
            matrix_sha256=self.matrix_sha256,
            profile_id=profile_id,
            voice_id=voice_id,
            changed_rule_ids=ordered_rule_ids,
            cells=cells,
            product_ready=product_ready,
            blockers=tuple(blockers),
        )

    def require_product_ready(self, plan: Any) -> PlanEvidenceReport:
        report = self.evaluate_plan(plan)
        if not report.product_ready:
            raise BilingualProductMatrixError(
                "plan_evidence_incomplete",
                "The requested voice/rule cells have not all passed product gates: "
                + "; ".join(report.blockers),
            )
        return report

    def safe_catalog(self) -> dict[str, Any]:
        directions: list[dict[str, Any]] = []
        profile_ids = tuple(dict.fromkeys(cell.profile_id for cell in self.cells))
        for profile_id in profile_ids:
            profile_cells = [
                cell for cell in self.cells if cell.profile_id == profile_id
            ]
            sample = profile_cells[0]
            voices = []
            for voice_id in dict.fromkeys(
                cell.voice_id for cell in profile_cells
            ):
                voice_cells = [
                    cell for cell in profile_cells if cell.voice_id == voice_id
                ]
                changed = [cell for cell in voice_cells if cell.changed]
                voices.append(
                    {
                        "voice_id": voice_id,
                        "rule_cell_count": len(voice_cells),
                        "changed_rule_cell_count": len(changed),
                        "product_enabled_cell_count": sum(
                            cell.product_enabled for cell in changed
                        ),
                        "pending_cell_count": sum(
                            not cell.product_enabled for cell in changed
                        ),
                    }
                )
            directions.append(
                {
                    "profile_id": profile_id,
                    "source_language": sample.source_language,
                    "listener_language": sample.listener_language,
                    "voices": voices,
                }
            )
        changed_cells = [cell for cell in self.cells if cell.changed]
        return {
            "schema_version": 1,
            "matrix_version": self.matrix_version,
            "production_enabled": self.production_enabled,
            "evidence_transfer_between_voices": False,
            "evidence_transfer_between_rules": False,
            "rule_cell_count": len(self.cells),
            "changed_rule_cell_count": len(changed_cells),
            "product_enabled_cell_count": sum(
                cell.product_enabled for cell in changed_cells
            ),
            "directions": directions,
        }

    def validation_manifest(self) -> dict[str, Any]:
        slots: list[dict[str, Any]] = []
        for cell in self.cells:
            if not cell.changed:
                continue
            for context in _fixture_contexts(cell, self.fixture_policy):
                slots.append(
                    {
                        "logical_slot_id": (
                            f"{cell.profile_id}__{cell.voice_id}__"
                            f"{cell.rule_id}__{context}"
                        ),
                        "cell_id": cell.cell_id,
                        "profile_id": cell.profile_id,
                        "voice_id": cell.voice_id,
                        "family": cell.family,
                        "rule_id": cell.rule_id,
                        "source": cell.source,
                        "target": cell.target,
                        "context": context,
                        "fixture_builder": _fixture_builder(cell.family),
                        "fixture_spec": _fixture_spec(cell, context),
                        "render_sides": tuple(
                            self.fixture_policy["render_sides"]
                        ),
                        "status": "pending",
                    }
                )
        return {
            "schema_version": 1,
            "matrix_version": self.matrix_version,
            "matrix_sha256": self.matrix_sha256,
            "rules_sha256": self.rules_sha256,
            "classification": "structural_manifest_ready_no_audio_validation",
            "api_calls_made": 0,
            "audio_renders_made": 0,
            "replacement_fixtures_allowed": False,
            "selective_rerender_allowed": False,
            "cell_count": len(self.cells),
            "changed_cell_count": sum(cell.changed for cell in self.cells),
            "logical_slot_count": len(slots),
            "slots": slots,
        }


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise BilingualProductMatrixError(
            "invalid_matrix_source", f"Expected an object in {path}."
        )
    return value


def _fixture_builder(family: str) -> str:
    return {
        "vowel": "controlled_vowel_shell_v1",
        "consonant": "controlled_consonant_context_v1",
        "insertion": "controlled_latent_slot_context_v1",
        "prosody": "bounded_structure_prosody_v1",
    }[family]


def _fixture_contexts(
    cell: ValidationCell, fixture_policy: dict[str, Any]
) -> tuple[str, ...]:
    if cell.family == "vowel":
        return tuple(fixture_policy["segment_context_order"])
    if cell.family == "consonant" and cell.contexts == ("any",):
        return tuple(fixture_policy["segment_context_order"])
    return cell.contexts


def _fixture_spec(cell: ValidationCell, context: str) -> dict[str, Any]:
    identity_vowel = "ɛ" if cell.source_language == "en-US" else "i"
    filler_phone = f"mˈ{identity_vowel}v"
    if cell.family == "vowel":
        if cell.source is None:  # pragma: no cover - schema construction guards it
            raise BilingualProductMatrixError(
                "missing_fixture_source", "Vowel fixture has no source category."
            )
        target_phone = f"bˈ{cell.source}z"
    elif cell.family == "consonant":
        if cell.source is None:  # pragma: no cover - schema construction guards it
            raise BilingualProductMatrixError(
                "missing_fixture_source", "Consonant fixture has no source category."
            )
        if context == "word_initial":
            target_phone = f"{cell.source}ˈ{identity_vowel}b"
        elif context in {"word_final", "phrase_final_new_context"}:
            target_phone = f"bˈ{identity_vowel}{cell.source}"
        else:
            target_phone = f"bˈ{identity_vowel}{cell.source}{identity_vowel}b"
    elif cell.family == "insertion":
        target_phone = (
            f"bˈ{identity_vowel}p"
            if context == "word_final_obstruent"
            else f"bˈ{identity_vowel}ps"
        )
    elif cell.family == "prosody":
        target_phone = (
            "bˌanˈa"
            if cell.rule_id == "enpt.lexical_stress_initial_bias"
            else f"bˈ{identity_vowel}b"
        )
    else:  # pragma: no cover - ProductRule.family is closed above
        raise BilingualProductMatrixError(
            "unknown_fixture_family", f"Unknown fixture family: {cell.family}."
        )

    if context == "phrase_medial_continuous_speech":
        source_words = ("mora", "tavi", "nelo")
        source_phones = (filler_phone, target_phone, filler_phone)
        expected_target_word_indexes = (1,)
    elif context == "phrase_final_new_context":
        source_words = ("mora", "tavi")
        source_phones = (filler_phone, target_phone)
        expected_target_word_indexes = (1,)
    elif context == "repeated_multi_target":
        source_words = ("tavi", "tavi")
        source_phones = (target_phone, target_phone)
        expected_target_word_indexes = (0, 1)
    else:
        source_words = ("tavi",)
        source_phones = (target_phone,)
        expected_target_word_indexes = (0,)
    punctuation = "?" if cell.contexts == ("polar_question",) else "."
    return {
        "source_words": source_words,
        "source_phones": source_phones,
        "punctuation": punctuation,
        "text": " ".join(source_words) + punctuation,
        "expected_target_word_indexes": expected_target_word_indexes,
    }


def _rules_for_profiles(
    vowels: dict[str, Any], listeners: dict[str, Any]
) -> tuple[ProductRule, ...]:
    vowel_profiles = {
        profile["id"]: profile for profile in vowels.get("profiles", ())
    }
    source_ids = {
        row["id"]
        for row in (*vowels.get("sources", ()), *listeners.get("sources", ()))
    }
    rules: list[ProductRule] = []
    for overlay in listeners.get("profiles", ()):
        try:
            base = vowel_profiles[overlay["base_profile_id"]]
        except KeyError as exc:
            raise BilingualProductMatrixError(
                "unknown_base_profile", "Listener profile has no vowel base."
            ) from exc
        common = {
            "profile_id": overlay["id"],
            "source_language": base["source_language"],
            "listener_language": base["listener_language"],
        }
        for raw in base.get("vowel_rules", ()):
            rules.append(
                ProductRule(
                    **common,
                    family="vowel",
                    rule_id=raw["id"],
                    source=raw["source"],
                    target=raw["target"],
                    operation="substitute",
                    contexts=("any",),
                    evidence_tier=raw["evidence_tier"],
                    acoustic_status=raw["acoustic_status"],
                    source_ids=tuple(raw["source_ids"]),
                    changed=raw["source"] != raw["target"],
                )
            )
        for raw in overlay.get("consonant_rules", ()):
            rules.append(
                ProductRule(
                    **common,
                    family="consonant",
                    rule_id=raw["id"],
                    source=raw["source"],
                    target=raw["target"],
                    operation="substitute",
                    contexts=tuple(raw["contexts"]),
                    evidence_tier=raw["evidence_tier"],
                    acoustic_status=raw["acoustic_status"],
                    source_ids=tuple(raw["source_ids"]),
                    changed=raw["source"] != raw["target"],
                )
            )
        for raw in overlay.get("insertion_rules", ()):
            rules.append(
                ProductRule(
                    **common,
                    family="insertion",
                    rule_id=raw["id"],
                    source=None,
                    target=raw["target"],
                    operation=raw["operation"],
                    contexts=tuple(raw["contexts"]),
                    evidence_tier=raw["evidence_tier"],
                    acoustic_status=(
                        "pending_controlled_epenthesis_acoustic_and_listener_qc"
                    ),
                    source_ids=tuple(raw["source_ids"]),
                    changed=True,
                )
            )
        for raw in overlay.get("prosody_rules", ()):
            changed = raw["operation"] != "identity"
            rules.append(
                ProductRule(
                    **common,
                    family="prosody",
                    rule_id=raw["id"],
                    source=None,
                    target=None,
                    operation=raw["operation"],
                    contexts=tuple(raw["contexts"]),
                    evidence_tier=raw["evidence_tier"],
                    acoustic_status=(
                        raw["architecture_status"]
                        if changed
                        else "not_required_identity"
                    ),
                    source_ids=tuple(raw["source_ids"]),
                    changed=changed,
                )
            )
    if not rules:
        raise BilingualProductMatrixError(
            "empty_rule_inventory", "The bilingual rule inventory is empty."
        )
    for rule in rules:
        unknown = set(rule.source_ids) - source_ids
        if unknown:
            raise BilingualProductMatrixError(
                "unknown_rule_source",
                f"{rule.rule_id} cites unknown sources: {sorted(unknown)}",
            )
    identities = [(rule.profile_id, rule.rule_id) for rule in rules]
    if len(identities) != len(set(identities)):
        raise BilingualProductMatrixError(
            "duplicate_rule_id", "Rule IDs are duplicated within a profile."
        )
    return tuple(rules)


def _default_cell(
    rule: ProductRule,
    voice_id: str,
    policy: dict[str, Any],
) -> ValidationCell:
    if rule.changed:
        runtime_status = policy["changed_rule_default_runtime_status"]
        human_status = policy["changed_rule_default_human_status"]
        claim_tier = "research_candidate_pending_validation"
    else:
        runtime_status = "not_required_identity"
        human_status = "not_required_identity"
        claim_tier = "identity_coverage_no_listener_intervention"
    return ValidationCell(
        cell_id=f"{rule.profile_id}::{voice_id}::{rule.family}::{rule.rule_id}",
        profile_id=rule.profile_id,
        source_language=rule.source_language,
        listener_language=rule.listener_language,
        voice_id=voice_id,
        family=rule.family,
        rule_id=rule.rule_id,
        source=rule.source,
        target=rule.target,
        operation=rule.operation,
        contexts=rule.contexts,
        evidence_tier=rule.evidence_tier,
        acoustic_status=rule.acoustic_status,
        source_ids=rule.source_ids,
        changed=rule.changed,
        runtime_status=runtime_status,
        human_status=human_status,
        product_enabled=False,
        claim_tier=claim_tier,
    )


def _apply_overrides(
    cells: Iterable[ValidationCell],
    overrides: list[dict[str, Any]],
    bindings: dict[str, dict[str, str]],
    production_enabled: bool,
) -> tuple[ValidationCell, ...]:
    by_key = {
        (cell.profile_id, cell.voice_id, cell.rule_id): cell for cell in cells
    }
    expected_keys = {
        "profile_id",
        "voice_id",
        "rule_id",
        "reference_rule_id",
        "runtime_status",
        "human_status",
        "product_enabled",
        "claim_tier",
        "binding_id",
        "note",
    }
    for override in overrides:
        if set(override) != expected_keys:
            raise BilingualProductMatrixError(
                "invalid_evidence_override", "Evidence override schema drifted."
            )
        key = (
            override["profile_id"],
            override["voice_id"],
            override["rule_id"],
        )
        try:
            cell = by_key[key]
        except KeyError as exc:
            raise BilingualProductMatrixError(
                "unknown_evidence_override", f"Unknown evidence cell: {key!r}."
            ) from exc
        if not cell.changed:
            raise BilingualProductMatrixError(
                "identity_evidence_override", "Identity cells cannot be promoted."
            )
        binding_id = override["binding_id"]
        if binding_id not in bindings:
            raise BilingualProductMatrixError(
                "unknown_evidence_binding", f"Unknown binding: {binding_id}."
            )
        if override["product_enabled"] and not production_enabled:
            raise BilingualProductMatrixError(
                "unsafe_matrix_promotion",
                "A product cell cannot be enabled while the matrix is disabled.",
            )
        by_key[key] = replace(
            cell,
            runtime_status=override["runtime_status"],
            human_status=override["human_status"],
            product_enabled=override["product_enabled"],
            claim_tier=override["claim_tier"],
            reference_rule_id=override["reference_rule_id"],
            binding_id=binding_id,
            note=override["note"],
        )
    return tuple(by_key[key] for key in sorted(by_key))


def _validate_policy(policy: dict[str, Any]) -> None:
    if set(policy) != {
        "cell_unit",
        "evidence_transfer_between_voices",
        "evidence_transfer_between_rules",
        "identity_rules_require_intervention_validation",
        "changed_rule_default_runtime_status",
        "changed_rule_default_human_status",
        "all_changed_rules_required_for_unqualified_profile_claim",
        "failed_or_pending_cells_fail_closed",
    }:
        raise BilingualProductMatrixError(
            "invalid_validation_policy", "Validation policy schema drifted."
        )
    if (
        policy["cell_unit"] != "listener_profile_x_source_voice_x_rule"
        or policy["evidence_transfer_between_voices"] is not False
        or policy["evidence_transfer_between_rules"] is not False
        or policy["identity_rules_require_intervention_validation"] is not False
        or policy["all_changed_rules_required_for_unqualified_profile_claim"]
        is not True
        or policy["failed_or_pending_cells_fail_closed"] is not True
    ):
        raise BilingualProductMatrixError(
            "unsafe_validation_policy", "Validation policy permits evidence transfer."
        )


def _validate_fixture_policy(policy: dict[str, Any]) -> None:
    if set(policy) != {
        "segment_context_order",
        "render_sides",
        "same_voice_pair_required",
        "common_rng_required",
        "replacement_fixtures_allowed",
        "selective_rerender_allowed",
    }:
        raise BilingualProductMatrixError(
            "invalid_fixture_policy", "Fixture policy schema drifted."
        )
    if (
        len(policy["segment_context_order"]) != 3
        or policy["render_sides"]
        != [
            "neutral",
            "identity",
            "full_lens_diagnostic",
            "spliced_lens",
        ]
        or policy["same_voice_pair_required"] is not True
        or policy["common_rng_required"] is not True
        or policy["replacement_fixtures_allowed"] is not False
        or policy["selective_rerender_allowed"] is not False
    ):
        raise BilingualProductMatrixError(
            "unsafe_fixture_policy", "Fixture policy permits uncontrolled selection."
        )


def _validate_heart_reference(
    candidate: dict[str, Any], overrides: list[dict[str, Any]]
) -> None:
    if len(overrides) != 1:
        raise BilingualProductMatrixError(
            "invalid_reference_override_count",
            "The initial matrix binds exactly one narrow reference cell.",
        )
    override = overrides[0]
    if (
        candidate.get("profile_id") != "en-to-pt-BR-vowel-lens"
        or candidate.get("renderer", {}).get("voice") != "af_heart"
        or candidate.get("rule_ids") != [override.get("reference_rule_id")]
        or candidate.get("evidence", {}).get("automatic_status") != "pass"
        or candidate.get("evidence", {}).get("human_status") != "pending"
        or candidate.get("evidence", {}).get("production_promotion") is not False
        or override.get("profile_id") != "en-US-to-pt-BR-listener-v2"
        or override.get("voice_id") != "af_heart"
        or override.get("rule_id") != "enpt.ae_eh"
        or override.get("product_enabled") is not False
    ):
        raise BilingualProductMatrixError(
            "heart_reference_binding_drift",
            "The narrow Heart reference no longer matches its frozen evidence.",
        )


def load_bilingual_product_matrix(
    path: Path = BILINGUAL_PRODUCT_MATRIX_PATH,
) -> BilingualProductMatrix:
    data = _load_object(path)
    if set(data) != {
        "schema_version",
        "matrix_version",
        "status",
        "production_enabled",
        "source_bindings",
        "validation_policy",
        "fixture_policy",
        "evidence_overrides",
    }:
        raise BilingualProductMatrixError(
            "invalid_matrix_schema", "Product matrix schema drifted."
        )
    if (
        data["schema_version"] != 1
        or data["matrix_version"] != BILINGUAL_PRODUCT_MATRIX_VERSION
        or data["production_enabled"] is not False
        or data["status"]
        != "four_voice_bidirectional_research_matrix_all_product_cells_disabled"
    ):
        raise BilingualProductMatrixError(
            "unsafe_matrix_state", "Product matrix safety state drifted."
        )
    _validate_policy(data["validation_policy"])
    _validate_fixture_policy(data["fixture_policy"])
    bindings = data["source_bindings"]
    if set(bindings) != {
        "vowel_rules",
        "listener_rules",
        "voice_registry",
        "heart_reference_candidate",
    }:
        raise BilingualProductMatrixError(
            "incomplete_matrix_bindings", "Matrix source bindings are incomplete."
        )
    loaded: dict[str, dict[str, Any]] = {}
    for binding_id, binding in bindings.items():
        if set(binding) != {"path", "sha256"}:
            raise BilingualProductMatrixError(
                "invalid_matrix_binding", f"Malformed binding: {binding_id}."
            )
        source_path = (ROOT / binding["path"]).resolve()
        if not source_path.is_relative_to(ROOT.resolve()):
            raise BilingualProductMatrixError(
                "unsafe_matrix_path", "A matrix binding escapes the repository."
            )
        if sha256_file(source_path) != binding["sha256"]:
            raise BilingualProductMatrixError(
                "matrix_binding_drift", f"Matrix binding drifted: {binding_id}."
            )
        loaded[binding_id] = _load_object(source_path)
    _validate_heart_reference(
        loaded["heart_reference_candidate"], data["evidence_overrides"]
    )
    registry: ProductVoiceRegistry = load_product_voice_registry(
        ROOT / bindings["voice_registry"]["path"]
    )
    rules = _rules_for_profiles(
        loaded["vowel_rules"], loaded["listener_rules"]
    )
    cells = []
    for rule in rules:
        for voice in registry.voices_for(rule.source_language):
            cells.append(
                _default_cell(rule, voice.voice_id, data["validation_policy"])
            )
    cells = list(
        _apply_overrides(
            cells,
            data["evidence_overrides"],
            bindings,
            data["production_enabled"],
        )
    )
    rules_sha256 = hashlib.sha256(
        stable_json([asdict(rule) for rule in rules]).encode("utf-8")
    ).hexdigest()
    matrix_sha256 = hashlib.sha256(
        stable_json(
            {
                "matrix_config_sha256": sha256_file(path),
                "rules_sha256": rules_sha256,
                "voice_registry_sha256": registry.registry_sha256,
                "cells": [asdict(cell) for cell in cells],
            }
        ).encode("utf-8")
    ).hexdigest()
    return BilingualProductMatrix(
        matrix_version=data["matrix_version"],
        matrix_sha256=matrix_sha256,
        rules_sha256=rules_sha256,
        production_enabled=False,
        cells=tuple(cells),
        fixture_policy=data["fixture_policy"],
    )


def load_bilingual_structural_state(
    matrix: BilingualProductMatrix | None = None,
    path: Path = BILINGUAL_PRODUCT_STRUCTURAL_STATE_PATH,
    *,
    verify_result_artifact: bool = True,
) -> dict[str, Any]:
    matrix = matrix or load_bilingual_product_matrix()
    state = _load_object(path)
    if set(state) != {
        "schema_version",
        "state_version",
        "matrix_version",
        "matrix_config_sha256",
        "matrix_sha256",
        "result_binding",
        "classification",
        "planner_slot_count",
        "planner_pass_count",
        "planner_fail_count",
        "planner_gate_yield",
        "api_calls_made",
        "audio_renders_made",
        "audio_validation_status",
        "production_enabled",
    }:
        raise BilingualProductMatrixError(
            "invalid_structural_state", "Structural state schema drifted."
        )
    if (
        state["schema_version"] != 1
        or state["state_version"]
        != BILINGUAL_PRODUCT_STRUCTURAL_STATE_VERSION
        or state["matrix_version"] != matrix.matrix_version
        or state["matrix_config_sha256"]
        != sha256_file(BILINGUAL_PRODUCT_MATRIX_PATH)
        or state["matrix_sha256"] != matrix.matrix_sha256
        or state["classification"] != "all_structural_slots_pass"
        or state["planner_slot_count"] != 280
        or state["planner_pass_count"] != 280
        or state["planner_fail_count"] != 0
        or state["planner_gate_yield"] != 1.0
        or state["api_calls_made"] != 0
        or state["audio_renders_made"] != 0
        or state["audio_validation_status"] != "pending"
        or state["production_enabled"] is not False
    ):
        raise BilingualProductMatrixError(
            "structural_state_drift", "Structural state no longer binds its pass."
        )
    binding = state["result_binding"]
    if set(binding) != {"path", "sha256", "record_sha256"}:
        raise BilingualProductMatrixError(
            "invalid_structural_result_binding",
            "Structural result binding schema drifted.",
        )
    if not verify_result_artifact:
        return state
    result_path = (ROOT / binding["path"]).resolve()
    if (
        not result_path.is_relative_to(ROOT.resolve())
        or sha256_file(result_path) != binding["sha256"]
    ):
        raise BilingualProductMatrixError(
            "structural_result_binding_drift",
            "Structural result file no longer matches its binding.",
        )
    result = _load_object(result_path)
    if (
        result.get("record_sha256") != binding["record_sha256"]
        or result.get("matrix_sha256") != matrix.matrix_sha256
        or result.get("classification") != state["classification"]
        or result.get("planner_slot_count") != state["planner_slot_count"]
        or result.get("planner_pass_count") != state["planner_pass_count"]
        or result.get("planner_fail_count") != state["planner_fail_count"]
        or result.get("planner_gate_yield") != state["planner_gate_yield"]
        or result.get("api_calls_made") != 0
        or result.get("audio_renders_made") != 0
        or result.get("production_enabled") is not False
    ):
        raise BilingualProductMatrixError(
            "structural_result_semantic_drift",
            "Structural result semantics no longer match the state record.",
        )
    return state
