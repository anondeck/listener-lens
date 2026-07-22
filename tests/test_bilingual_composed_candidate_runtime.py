from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

import earshift_bakeoff.bilingual_composed_candidate_runtime_v3 as runtime_v3_module
from earshift_bakeoff.bilingual_candidate_runtime import (
    BilingualCandidateAcousticGateError,
)
from earshift_bakeoff.bilingual_composed_candidate_runtime import (
    BilingualComposedCandidateRuntime,
    COMPOSITION_CONTRACT_VERSION,
    COMPOSITION_STATE_SHA256,
    _load_composition_state,
)
from earshift_bakeoff.bilingual_composed_candidate_runtime_v2 import (
    BilingualComposedCandidateRuntimeV2,
    COMPOSITION_CONTRACT_V3,
    COMPOSITION_STATE_V2_SHA256,
    MULTI_RULE_AUTOMATIC_ONLY_CLAIM,
    MULTI_RULE_AUTOMATIC_ONLY_STATUS,
    RULE_DISPLAY_SHA256,
    _load_composition_state_v2,
    _load_rule_display,
)
from earshift_bakeoff.bilingual_composed_candidate_runtime_v3 import (
    BilingualComposedCandidateRuntimeV3,
    COMPOSITION_CONTRACT_V4,
    COMPOSITION_STATE_V3_SHA256,
    MULTI_RULE_ADAPTIVE_CLAIM,
    MULTI_RULE_ADAPTIVE_STATUS,
    _load_composition_state_v3,
)
from earshift_bakeoff.bilingual_vowel_engine import BilingualRenderVerification


@dataclass(frozen=True)
class _Occurrence:
    rule_id: str
    changed: bool = True


def _word(*rule_ids: str) -> SimpleNamespace:
    return SimpleNamespace(
        vowel_occurrences=tuple(_Occurrence(rule_id) for rule_id in rule_ids),
        consonant_occurrences=(),
        insertion_occurrences=(),
        prosody_occurrences=(),
    )


def _plan(*rule_ids: str) -> SimpleNamespace:
    return SimpleNamespace(
        profile_id="en-US-to-pt-BR-listener-v2",
        voice_id="af_heart",
        normalized_text="We once took books.",
        neutral_script="Wibp wungsh kuhtv guupf.",
        lens_script="Wibp wangsh kootv goopf.",
        plan_sha256="a" * 64,
        active_prosody_rule_ids=(),
        words=(_word(*rule_ids),),
    )


def _cell(rule_id: str, source: str, target: str) -> SimpleNamespace:
    return SimpleNamespace(
        rule_id=rule_id,
        source=source,
        target=target,
        automatic_pass=True,
        candidate_rung="v8",
    )


class _Registry:
    def __init__(self, cells: tuple[SimpleNamespace, ...]) -> None:
        self.cells = {cell.rule_id: cell for cell in cells}

    def cell(self, _profile_id: str, _voice_id: str, rule_id: str):
        return self.cells.get(rule_id)


class _Base:
    def __init__(self, cells: tuple[SimpleNamespace, ...], *, acoustic_pass=True):
        self.cells = cells
        self.registry = _Registry(cells)
        self.base_planner = SimpleNamespace(plan=lambda _text: _plan(*(c.rule_id for c in cells)))
        self.acoustic_pass = acoustic_pass
        self.synthesis = object()
        self.scaler = {}
        self.single_calls = 0
        self.composition_calls = 0

    def _composition_planner(self, _rule_ids: tuple[str, ...]):
        return self.base_planner

    def contract(self) -> dict:
        return {
            "service_contract_version": "bilingual-kokoro-controlled-pair-service-v1",
            "candidate_id": "bilingual-kokoro-vowel-candidate-v1",
            "candidate_state_sha256": "b" * 64,
            "runtime_gate_result_sha256": "c" * 64,
            "runtime_gate_scaler_sha256": "d" * 64,
            "profile_id": "en-US-to-pt-BR-listener-v2",
            "voice_id": "af_heart",
            "voice_registry_version": "kokoro-product-voices-v1",
            "voice_registry_sha256": "e" * 64,
            "production_enabled": False,
            "human_qc_status": "pending",
        }

    def render(self, _text: str) -> dict:
        self.single_calls += 1
        return {
            "status": "ready_pending_human_qc",
            "candidate_contract": self.contract(),
            "transform": {
                "applied_rules": [
                    {
                        "rule_id": self.cells[0].rule_id,
                        "source_ipa": self.cells[0].source,
                        "target_ipa": self.cells[0].target,
                    }
                ]
            },
        }

    def render_v8_composition_candidate(self, _text: str) -> SimpleNamespace:
        self.composition_calls += 1
        plan = _plan(*(cell.rule_id for cell in self.cells))
        verification = BilingualRenderVerification(
            neutral_identity_bit_exact=True,
            equal_nonempty_samples=True,
            finite=True,
            unclipped=True,
            outside_splice_exact_neutral=True,
            full_weight_interior_exact_lens=True,
            boundary_metrics_pass=True,
            localization_pass=True,
            localization_fraction=1.0,
            integrity_pass=True,
            changed_rules_acoustically_validated=False,
            evidence_status="integrity_pass_acoustic_validation_pending",
        )
        audio = np.array([0, 100, -100, 0], dtype="<i2")
        render = SimpleNamespace(
            neutral_pcm=audio,
            identity_pcm=audio.copy(),
            lens_pcm=np.array([0, 200, -200, 0], dtype="<i2"),
            verification=verification,
        )
        return SimpleNamespace(
            isolated_plan=plan,
            cells=self.cells,
            omitted_rule_ids=("enpt.illicit_coda_epenthetic_i",),
            render=render,
            acoustic={
                "version": "bilingual-candidate-v8-composition-gate-v1",
                "pass": self.acoustic_pass,
            },
        )


def test_composition_state_is_hash_bound_disabled_and_nonpromotional() -> None:
    state = _load_composition_state()

    assert state["production_enabled"] is False
    assert state["service_contract_version"] == COMPOSITION_CONTRACT_VERSION
    assert state["evidence"]["composition_v1_pass_count"] == 2
    assert state["evidence"]["human_status"] == "pending"
    assert state["evidence"]["unseen_composition_status"] == "pending"
    assert len(COMPOSITION_STATE_SHA256) == 64


def test_single_rule_response_keeps_base_path_and_extends_contract() -> None:
    base = _Base((_cell("enpt.ae_eh", "æ", "ɛ"),))
    runtime = BilingualComposedCandidateRuntime(base)

    result = runtime.render("The cat naps.")

    assert base.single_calls == 1
    assert base.composition_calls == 0
    assert result["transform"]["composition_mode"] == "single_rule"
    assert result["candidate_contract"]["service_contract_version"] == (
        COMPOSITION_CONTRACT_VERSION
    )
    assert result["candidate_contract"]["composition_state_sha256"] == (
        COMPOSITION_STATE_SHA256
    )


def test_two_rule_v8_response_is_composed_only_after_current_gate_passes() -> None:
    cells = (
        _cell("enpt.ah_a", "ʌ", "a"),
        _cell("enpt.uh_u", "ʊ", "u"),
    )
    base = _Base(cells)
    runtime = BilingualComposedCandidateRuntime(base)

    result = runtime.render("We once took books.")

    assert base.single_calls == 0
    assert base.composition_calls == 1
    assert result["claim_tier"] == (
        "runtime_composition_acoustic_pass_human_qc_pending"
    )
    assert result["transform"]["composition_mode"] == "multi_rule_v8"
    assert [row["rule_id"] for row in result["transform"]["applied_rules"]] == [
        "enpt.ah_a",
        "enpt.uh_u",
    ]
    assert result["verification"]["target_occurrence_count"] == 2
    assert result["verification"]["acoustic"]["pass"] is True
    assert result["api_calls_made"] == 0


def test_composed_current_context_failure_is_rejected_without_fallback() -> None:
    cells = (
        _cell("pten.final_e_i", "e", "i"),
        _cell("pten.o_goat", "o", "O"),
    )
    base = _Base(cells, acoustic_pass=False)
    runtime = BilingualComposedCandidateRuntime(base)

    with pytest.raises(BilingualCandidateAcousticGateError) as caught:
        runtime.render("O local fica em frente.")

    assert caught.value.code == "runtime_acoustic_gate_rejected"
    assert base.single_calls == 0
    assert base.composition_calls == 1


def test_composition_v2_state_preserves_failed_unseen_aggregate() -> None:
    state = _load_composition_state_v2()

    assert state["production_enabled"] is False
    assert state["service_contract_version"] == COMPOSITION_CONTRACT_V3
    assert state["evidence"]["unseen_composition_pass_count"] == 2
    assert state["evidence"]["unseen_composition_fixture_count"] == 3
    assert state["evidence"]["unseen_composition_status"] == (
        "automatic_failed_2_of_3"
    )
    assert state["evidence"]["human_status"] == (
        "not_eligible_after_unseen_aggregate_failure"
    )
    assert state["runtime_policy"]["human_composition_qc_eligible"] is False
    assert len(COMPOSITION_STATE_V2_SHA256) == 64


def test_rule_display_hides_renderer_tokens_without_changing_rule_identity() -> None:
    display = _load_rule_display()

    assert len(display) == 13
    assert len(RULE_DISPLAY_SHA256) == 64
    assert display["enpt.goat_o"]["display_label"] == "/oʊ/ → /o/"
    assert display["pten.o_goat"]["display_label"] == "/o/ → /oʊ/"
    assert display["pten.final_e_i"] == {
        "rule_id": "pten.final_e_i",
        "renderer_source": "y",
        "renderer_target": "i",
        "display_source": "e",
        "display_target": "i",
        "display_label": "word-final /e/ → /i/ (renderer projection)",
    }


def test_composition_v2_single_rule_keeps_its_separate_human_queue() -> None:
    base = _Base((_cell("enpt.ae_eh", "æ", "ɛ"),))
    runtime = BilingualComposedCandidateRuntimeV2(base)

    result = runtime.render("The cat naps.")

    assert result["status"] == "ready_pending_human_qc"
    assert result["transform"]["composition_mode"] == "single_rule"
    assert result["candidate_contract"]["service_contract_version"] == (
        COMPOSITION_CONTRACT_V3
    )
    assert result["candidate_contract"]["composition_state_sha256"] == (
        COMPOSITION_STATE_V2_SHA256
    )
    assert result["candidate_contract"]["composition_unseen_status"] == (
        "automatic_failed_2_of_3"
    )
    assert result["transform"]["applied_rules"][0]["display_label"] == (
        "/æ/ → /ɛ/"
    )


def test_composition_v2_multi_rule_result_is_automatic_only() -> None:
    cells = (
        _cell("enpt.ah_a", "ʌ", "a"),
        _cell("enpt.uh_u", "ʊ", "u"),
    )
    base = _Base(cells)
    runtime = BilingualComposedCandidateRuntimeV2(base)

    result = runtime.render("We once took books.")

    assert result["status"] == MULTI_RULE_AUTOMATIC_ONLY_STATUS
    assert result["claim_tier"] == MULTI_RULE_AUTOMATIC_ONLY_CLAIM
    assert result["transform"]["composition_mode"] == "multi_rule_v8"
    assert [row["display_label"] for row in result["transform"]["applied_rules"]] == [
        "/ʌ/ → /a/",
        "/ʊ/ → /u/",
    ]
    assert result["candidate_contract"]["composition_human_qc_status"] == (
        "not_eligible_after_unseen_aggregate_failure"
    )
    assert result["api_calls_made"] == 0


def _adaptive_result(
    base: _Base,
    *,
    automatic_pass: bool,
    selected_round_index: int | None = 1,
) -> SimpleNamespace:
    candidate = base.render_v8_composition_candidate("We once took books.")
    base.composition_calls = 0
    selected = (
        SimpleNamespace(
            plan=candidate.isolated_plan,
            render=candidate.render,
            acoustic=candidate.acoustic,
        )
        if automatic_pass
        else None
    )
    attempt_count = (selected_round_index + 1) if selected_round_index is not None else 2
    return SimpleNamespace(
        automatic_pass=automatic_pass,
        selected_attempt=selected,
        selected_round_index=selected_round_index,
        rescued_after_retry=bool(selected_round_index),
        attempts=tuple(object() for _ in range(attempt_count)),
    )


def test_composition_v3_state_binds_adaptive_unseen_pass_without_promotion() -> None:
    state = _load_composition_state_v3()

    assert state["production_enabled"] is False
    assert state["service_contract_version"] == COMPOSITION_CONTRACT_V4
    assert state["evidence"]["adaptive_unseen_pass_count"] == 3
    assert state["evidence"]["adaptive_unseen_fixture_count"] == 3
    assert state["evidence"]["adaptive_unseen_rescued_fixture_count"] == 2
    assert state["evidence"]["adaptive_unseen_total_attempt_count"] == 5
    assert state["evidence"]["human_status"] == "pending"
    assert state["runtime_policy"]["maximum_carrier_retry_rounds"] == 5
    assert state["runtime_policy"]["failed_or_exhausted_contexts_fail_closed"] is True
    assert len(COMPOSITION_STATE_V3_SHA256) == 64


def test_composition_v3_single_rule_preserves_nonadaptive_base_path() -> None:
    base = _Base((_cell("enpt.ae_eh", "æ", "ɛ"),))
    runtime = BilingualComposedCandidateRuntimeV3(base)

    result = runtime.render("The cat naps.")

    assert base.single_calls == 1
    assert base.composition_calls == 0
    assert result["status"] == "ready_pending_human_qc"
    assert result["transform"]["composition_mode"] == "single_rule"
    assert "adaptive_carrier" not in result.get("verification", {})
    assert result["candidate_contract"]["service_contract_version"] == (
        COMPOSITION_CONTRACT_V4
    )
    assert result["candidate_contract"]["composition_state_sha256"] == (
        COMPOSITION_STATE_V3_SHA256
    )


def test_composition_v3_multi_rule_uses_bounded_adaptive_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cells = (
        _cell("enpt.ah_a", "ʌ", "a"),
        _cell("enpt.uh_u", "ʊ", "u"),
    )
    base = _Base(cells)
    adaptive = _adaptive_result(base, automatic_pass=True, selected_round_index=1)

    class _AdaptiveRuntime:
        def __init__(self, **kwargs) -> None:
            assert kwargs["maximum_retry_rounds"] == 5

        def render(self, _text: str) -> SimpleNamespace:
            return adaptive

    monkeypatch.setattr(
        runtime_v3_module, "BilingualAdaptiveCarrierRuntime", _AdaptiveRuntime
    )
    runtime = BilingualComposedCandidateRuntimeV3(base)

    result = runtime.render("We once took books.")

    assert base.single_calls == 0
    assert base.composition_calls == 0
    assert result["status"] == MULTI_RULE_ADAPTIVE_STATUS
    assert result["claim_tier"] == MULTI_RULE_ADAPTIVE_CLAIM
    assert result["transform"]["composition_mode"] == "multi_rule_v8"
    assert result["verification"]["adaptive_carrier"] == {
        "version": "v8-adaptive-carrier-v1",
        "attempt_count": 2,
        "selected_round_index": 1,
        "rescued_after_retry": True,
        "maximum_retry_rounds": 5,
    }
    assert result["candidate_contract"]["composition_human_qc_status"] == (
        "pending"
    )
    assert result["api_calls_made"] == 0


def test_composition_v3_exhausted_adaptive_retries_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cells = (
        _cell("enpt.ah_a", "ʌ", "a"),
        _cell("enpt.uh_u", "ʊ", "u"),
    )
    base = _Base(cells)
    adaptive = _adaptive_result(
        base, automatic_pass=False, selected_round_index=None
    )

    class _AdaptiveRuntime:
        def __init__(self, **_kwargs) -> None:
            pass

        def render(self, _text: str) -> SimpleNamespace:
            return adaptive

    monkeypatch.setattr(
        runtime_v3_module, "BilingualAdaptiveCarrierRuntime", _AdaptiveRuntime
    )
    runtime = BilingualComposedCandidateRuntimeV3(base)

    with pytest.raises(BilingualCandidateAcousticGateError) as caught:
        runtime.render("We once took books.")

    assert caught.value.code == "runtime_acoustic_gate_rejected"
    assert base.single_calls == 0
