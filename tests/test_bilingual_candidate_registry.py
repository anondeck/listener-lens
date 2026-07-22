from __future__ import annotations

from dataclasses import dataclass

from earshift_bakeoff.bilingual_candidate_registry import (
    load_bilingual_candidate_registry,
)


@dataclass(frozen=True)
class _Occurrence:
    rule_id: str
    changed: bool = True


@dataclass(frozen=True)
class _Word:
    vowel_occurrences: tuple[_Occurrence, ...] = ()
    consonant_occurrences: tuple[_Occurrence, ...] = ()
    insertion_occurrences: tuple[_Occurrence, ...] = ()
    prosody_occurrences: tuple[_Occurrence, ...] = ()


@dataclass(frozen=True)
class _Plan:
    profile_id: str
    voice_id: str
    words: tuple[_Word, ...]
    active_prosody_rule_ids: tuple[str, ...] = ()


def _plan(*rule_ids: str, voice_id: str = "af_heart") -> _Plan:
    return _Plan(
        profile_id="en-US-to-pt-BR-listener-v2",
        voice_id=voice_id,
        words=tuple(
            _Word(vowel_occurrences=(_Occurrence(rule_id),))
            for rule_id in rule_ids
        ),
    )


def test_registry_binds_complete_unseen_result_without_promoting_it() -> None:
    registry = load_bilingual_candidate_registry()

    catalog = registry.safe_catalog()
    assert registry.feature_flag == "KOKORO_BILINGUAL_CANDIDATE_ENABLED"
    assert registry.production_enabled is False
    assert len(registry.cells) == 28
    assert catalog["unseen_automatic_pass_count"] == 18
    assert catalog["runtime_gate_pass_count"] == 18
    assert catalog["human_qc_pass_count"] == 0
    assert catalog["product_enabled_cell_count"] == 0
    assert catalog["voices"]["af_heart"]["automatic_pass_count"] == 6
    assert catalog["voices"]["am_michael"]["automatic_pass_count"] == 8
    assert catalog["voices"]["pm_alex"]["automatic_pass_count"] == 2
    assert catalog["voices"]["pf_dora"]["automatic_pass_count"] == 2
    assert catalog["voices"]["af_heart"]["automatic_pass_rule_ids"] == [
        "enpt.aa_a",
        "enpt.ae_eh",
        "enpt.ah_a",
        "enpt.goat_o",
        "enpt.nurse_eh",
        "enpt.uh_u",
    ]


def test_registry_allows_one_passed_rule_with_repeated_occurrences() -> None:
    registry = load_bilingual_candidate_registry()

    decision = registry.evaluate_plan(_plan("enpt.ae_eh", "enpt.ae_eh"))

    assert decision.render_eligible is True
    assert decision.changed_rule_ids == ("enpt.ae_eh",)
    assert decision.omitted_rule_ids == ()
    assert decision.cell is not None
    assert decision.cell.automatic_classification == "exact_category_pass"
    assert decision.blockers == (
        "per_request_acoustic_gate_required",
        "blind_human_qc_pending",
        "production_disabled",
    )


def test_registry_rejects_failed_voice_rule_and_rule_composition() -> None:
    registry = load_bilingual_candidate_registry()

    failed = registry.evaluate_plan(_plan("enpt.ih_i"))
    composed = registry.evaluate_plan(_plan("enpt.ae_eh", "enpt.ah_a"))
    unsupported_voice = registry.evaluate_plan(
        _plan("enpt.ih_i", voice_id="am_not_a_product_voice")
    )

    assert failed.status == "automatic_evidence_failed"
    assert failed.cell is not None
    assert failed.cell.automatic_classification == "fail"
    assert composed.status == "unsupported_rule_composition"
    assert unsupported_voice.status == "unsupported_rule_or_voice"


def test_registry_allows_one_supported_rule_and_reports_every_omission() -> None:
    registry = load_bilingual_candidate_registry()

    decision = registry.evaluate_plan(
        _plan("enpt.ae_eh", "enpt.eth_d", "enpt.illicit_coda_epenthetic_i")
    )

    assert decision.render_eligible is True
    assert decision.cell is not None and decision.cell.rule_id == "enpt.ae_eh"
    assert decision.omitted_rule_ids == (
        "enpt.eth_d",
        "enpt.illicit_coda_epenthetic_i",
    )
    assert "partial_profile_coverage" in decision.blockers


def test_registry_reports_no_rule_without_fabricating_support() -> None:
    registry = load_bilingual_candidate_registry()
    decision = registry.evaluate_plan(_plan())

    assert decision.status == "no_supported_sounds"
    assert decision.render_eligible is False
    assert decision.cell is None
