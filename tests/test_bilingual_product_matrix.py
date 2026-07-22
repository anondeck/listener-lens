from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from earshift_bakeoff.bilingual_product_engine import (
    BilingualProductPlanner,
    BilingualProductRuntime,
)
from earshift_bakeoff.bilingual_product_matrix import (
    BILINGUAL_PRODUCT_MATRIX_PATH,
    BilingualProductMatrixError,
    load_bilingual_product_matrix,
    load_bilingual_structural_state,
)


def _occurrence(rule_id: str, *, changed: bool = True) -> SimpleNamespace:
    return SimpleNamespace(rule_id=rule_id, changed=changed)


def _word(
    *,
    vowels: tuple[SimpleNamespace, ...] = (),
    consonants: tuple[SimpleNamespace, ...] = (),
    insertions: tuple[SimpleNamespace, ...] = (),
    prosody: tuple[SimpleNamespace, ...] = (),
) -> SimpleNamespace:
    return SimpleNamespace(
        vowel_occurrences=vowels,
        consonant_occurrences=consonants,
        insertion_occurrences=insertions,
        prosody_occurrences=prosody,
    )


def _plan(
    profile_id: str,
    voice_id: str,
    words: tuple[SimpleNamespace, ...],
    *,
    active_prosody_rule_ids: tuple[str, ...] = (),
) -> SimpleNamespace:
    return SimpleNamespace(
        profile_id=profile_id,
        voice_id=voice_id,
        words=words,
        active_prosody_rule_ids=active_prosody_rule_ids,
        safe_metadata=lambda: {
            "profile_id": profile_id,
            "voice_id": voice_id,
        },
    )


def test_matrix_covers_both_directions_all_four_voices_and_every_rule() -> None:
    matrix = load_bilingual_product_matrix()

    assert len(matrix.cells) == 166
    assert sum(cell.changed for cell in matrix.cells) == 98
    assert sum(not cell.changed for cell in matrix.cells) == 68
    assert {cell.voice_id for cell in matrix.cells} == {
        "af_heart",
        "am_michael",
        "pm_alex",
        "pf_dora",
    }
    assert {cell.profile_id for cell in matrix.cells} == {
        "en-US-to-pt-BR-listener-v2",
        "pt-BR-to-en-US-listener-v2",
    }
    assert {
        cell.family for cell in matrix.cells
    } == {"vowel", "consonant", "insertion", "prosody"}
    assert sum(cell.product_enabled for cell in matrix.cells) == 0


def test_matrix_keeps_heart_reference_narrow_and_transfers_nothing() -> None:
    matrix = load_bilingual_product_matrix()
    heart = matrix.cell(
        "en-US-to-pt-BR-listener-v2", "af_heart", "enpt.ae_eh"
    )
    michael = matrix.cell(
        "en-US-to-pt-BR-listener-v2", "am_michael", "enpt.ae_eh"
    )

    assert heart.runtime_status == (
        "strict_shell_automatic_pass_pending_broad_confirmation"
    )
    assert heart.claim_tier == "narrow_reference_only"
    assert heart.reference_rule_id == "ptbr.vowel.ae_to_eh"
    assert heart.product_enabled is False
    assert michael.runtime_status == "pending"
    assert michael.reference_rule_id is None
    assert michael.product_enabled is False


def test_manifest_batches_every_changed_cell_without_selective_replacement() -> None:
    matrix = load_bilingual_product_matrix()
    manifest = matrix.validation_manifest()

    assert manifest["cell_count"] == 166
    assert manifest["changed_cell_count"] == 98
    assert manifest["logical_slot_count"] == 280
    assert manifest["api_calls_made"] == 0
    assert manifest["audio_renders_made"] == 0
    assert manifest["replacement_fixtures_allowed"] is False
    assert manifest["selective_rerender_allowed"] is False
    assert len({row["logical_slot_id"] for row in manifest["slots"]}) == 280
    assert all(
        row["render_sides"]
        == (
            "neutral",
            "identity",
            "full_lens_diagnostic",
            "spliced_lens",
        )
        for row in manifest["slots"]
    )


def test_structural_state_binds_the_complete_zero_render_pass() -> None:
    matrix = load_bilingual_product_matrix()
    state = load_bilingual_structural_state(matrix)

    assert state["classification"] == "all_structural_slots_pass"
    assert state["planner_slot_count"] == 280
    assert state["planner_pass_count"] == 280
    assert state["planner_fail_count"] == 0
    assert state["planner_gate_yield"] == 1.0
    assert state["api_calls_made"] == 0
    assert state["audio_renders_made"] == 0
    assert state["audio_validation_status"] == "pending"
    assert state["production_enabled"] is False


def test_plan_evidence_collects_every_changed_family_and_fails_closed() -> None:
    matrix = load_bilingual_product_matrix()
    plan = _plan(
        "en-US-to-pt-BR-listener-v2",
        "af_heart",
        (
            _word(
                vowels=(_occurrence("enpt.ae_eh"),),
                consonants=(_occurrence("enpt.theta_t"),),
                insertions=(_occurrence("enpt.illicit_coda_epenthetic_i"),),
                prosody=(_occurrence("enpt.lexical_stress_initial_bias"),),
            ),
        ),
    )

    report = matrix.evaluate_plan(plan)

    assert report.changed_rule_ids == (
        "enpt.ae_eh",
        "enpt.illicit_coda_epenthetic_i",
        "enpt.lexical_stress_initial_bias",
        "enpt.theta_t",
    )
    assert report.product_ready is False
    assert len(report.cells) == 4
    assert "matrix_production_disabled" in report.blockers
    with pytest.raises(BilingualProductMatrixError) as exc_info:
        matrix.require_product_ready(plan)
    assert exc_info.value.code == "plan_evidence_incomplete"


def test_question_prosody_rule_is_included_even_without_a_word_occurrence() -> None:
    matrix = load_bilingual_product_matrix()
    plan = _plan(
        "pt-BR-to-en-US-listener-v2",
        "pm_alex",
        (_word(vowels=(_occurrence("pten.i_i", changed=False),)),),
        active_prosody_rule_ids=("pten.polar_rise_fall_statement",),
    )

    report = matrix.evaluate_plan(plan)

    assert report.changed_rule_ids == ("pten.polar_rise_fall_statement",)
    assert report.cells[0].family == "prosody"
    assert report.product_ready is False


def test_product_runtime_stops_before_rendering_unvalidated_cells() -> None:
    matrix = load_bilingual_product_matrix()
    raw_plan = _plan(
        "en-US-to-pt-BR-listener-v2",
        "af_heart",
        (_word(vowels=(_occurrence("enpt.ae_eh"),)),),
    )

    class _Planner:
        def plan(self, text: str) -> SimpleNamespace:
            assert text == "cat"
            return raw_plan

    class _Runtime:
        calls = 0

        def render(self, text: str) -> None:
            self.calls += 1

    runtime = _Runtime()
    product = BilingualProductRuntime(
        planner=BilingualProductPlanner(planner=_Planner(), matrix=matrix),
        runtime=runtime,
    )

    with pytest.raises(BilingualProductMatrixError):
        product.render_for_product("cat")
    assert runtime.calls == 0


def test_safe_catalog_exposes_counts_but_no_hashes_or_binding_paths() -> None:
    catalog = load_bilingual_product_matrix().safe_catalog()
    serialized = json.dumps(catalog)

    assert catalog["changed_rule_cell_count"] == 98
    assert catalog["product_enabled_cell_count"] == 0
    assert len(catalog["directions"]) == 2
    assert "sha256" not in serialized
    assert "binding" not in serialized
    assert "path" not in serialized


def test_matrix_fails_if_a_bound_rule_source_drifts(tmp_path) -> None:
    source = json.loads(
        BILINGUAL_PRODUCT_MATRIX_PATH.read_text(encoding="utf-8")
    )
    source["source_bindings"]["vowel_rules"]["sha256"] = "0" * 64
    path = tmp_path / "matrix.json"
    path.write_text(json.dumps(source), encoding="utf-8")

    with pytest.raises(BilingualProductMatrixError) as exc_info:
        load_bilingual_product_matrix(path)
    assert exc_info.value.code == "matrix_binding_drift"
