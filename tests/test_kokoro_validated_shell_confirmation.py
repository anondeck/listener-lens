from __future__ import annotations

from earshift_bakeoff.kokoro_validated_shell import (
    VALIDATED_LENS_SHELL,
    VALIDATED_NEUTRAL_SHELL,
)
from earshift_bakeoff.kokoro_validated_shell_confirmation import (
    EXPECTED,
    FIXTURE_INVENTORY,
    _layout,
    protocol_record,
)


def test_corrective_protocol_binds_exactly_one_carrier_mechanism_change() -> None:
    protocol = protocol_record()
    assert protocol["status"] == "frozen_before_any_corrective_decode"
    assert protocol["change"]["mechanism"] == "rule-bearing carrier allocation"
    assert protocol["change"]["neutral_immediate_shell"] == VALIDATED_NEUTRAL_SHELL
    assert protocol["change"]["lens_immediate_shell"] == VALIDATED_LENS_SHELL
    assert protocol["automatic_gate"]["unchanged_from_unseen_v1"] is True
    assert protocol["scope"]["decoder_attempt_ceiling"] == 9
    assert protocol["scope"]["api_calls"] == 0


def test_corrective_protocol_preserves_failed_parent_and_diagnoses_upstream() -> None:
    parent = protocol_record()["failed_parent"]
    assert parent["classification_preserved"] == "unseen_output_splice_automatic_failed"
    assert len(parent["fixtures"]) == 3
    assert all(
        row["untouched_full_state_lens_primary_pass"] is False
        for row in parent["fixtures"]
    )


def test_first_gate_clean_corrective_fixtures_are_frozen_and_unheard() -> None:
    protocol = protocol_record()
    assert len(protocol["fixtures"]) == len(FIXTURE_INVENTORY) == 3
    for row in protocol["fixtures"]:
        assert row["selected_order"] == 1
        assert (row["text"], row["plan_sha256"]) == EXPECTED[row["fixture_id"]]
        for index in row["target_word_indexes"]:
            word = row["words"][index]
            target = word["target_offsets"][0]
            assert word["neutral_phone"][target - 2 : target + 2] == VALIDATED_NEUTRAL_SHELL
            assert word["lens_phone"][target - 2 : target + 2] == VALIDATED_LENS_SHELL


def test_blind_layout_has_six_balanced_hidden_trials() -> None:
    layout = _layout()
    assert len(layout) == 6
    for spec in FIXTURE_INVENTORY:
        rows = [row for row in layout if row["fixture_id"] == spec["fixture_id"]]
        assert {row["condition"] for row in rows} == {
            "identity-control",
            "spliced-lens",
        }
