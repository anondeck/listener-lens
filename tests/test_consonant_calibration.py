from __future__ import annotations

import pytest

from earshift_bakeoff.consonant_calibration import (
    aggregate_rule_instrument,
    calibration_fixtures,
    labels_support_expected,
    overlapping_upr_labels,
    parse_allosaurus_timestamps,
)


def test_manifest_has_three_contexts_for_each_frozen_rule() -> None:
    fixtures = calibration_fixtures()
    counts: dict[str, int] = {}
    for fixture in fixtures:
        counts[fixture.rule_id] = counts.get(fixture.rule_id, 0) + 1
        assert len(fixture.neutral_phonemes) == len(fixture.lens_phonemes)

    assert len(fixtures) == 18
    assert set(counts.values()) == {3}
    assert set(counts) == {
        "enpt.theta_t",
        "enpt.eth_d",
        "pten.palatal_lateral_yod",
        "pten.palatal_nasal_n",
        "pten.dorsal_r_h",
        "pten.tap_flap",
    }


def test_allosaurus_timestamp_parser_and_overlap_are_bounded() -> None:
    rows = parse_allosaurus_timestamps(
        "0.100 0.045 m\n0.160 0.045 θ\n0.220 0.045 a\n"
    )

    actual = overlapping_upr_labels(
        rows, start_s=0.17, end_s=0.20, context_s=0.0
    )

    assert actual == ("θ",)
    assert labels_support_expected(actual, ("θ",)) is True
    assert labels_support_expected(actual, ("t", "tʰ")) is False


def test_allosaurus_parser_rejects_malformed_rows() -> None:
    with pytest.raises(ValueError, match="unexpected"):
        parse_allosaurus_timestamps("0.1 θ")


def test_direct_support_is_only_eligible_for_human_qc() -> None:
    rows = [
        {
            "source_anchor_upr_match": index != 2,
            "target_anchor_upr_match": index != 0,
            "engineering_integrity_pass": True,
        }
        for index in range(3)
    ]

    actual = aggregate_rule_instrument(rows, evidence_tier="direct_listener_result")

    assert actual["auxiliary_instrument_status"] == "supportive"
    assert actual["claim_status"] == "eligible_for_blind_human_qc_not_promoted"


def test_derived_support_never_promotes_rule() -> None:
    rows = [
        {
            "source_anchor_upr_match": True,
            "target_anchor_upr_match": True,
            "engineering_integrity_pass": True,
        }
        for _ in range(3)
    ]

    actual = aggregate_rule_instrument(rows, evidence_tier="derived_projection")

    assert actual["auxiliary_instrument_status"] == "supportive"
    assert actual["claim_status"] == "research_only_derived_rule_not_promoted"
