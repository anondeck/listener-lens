from __future__ import annotations

from earshift_bakeoff.kokoro_output_domain_splice import (
    TAPER_MS,
    TAPER_SAMPLES,
)
from earshift_bakeoff.kokoro_output_splice_unseen import (
    EXPECTED_SELECTIONS,
    FIXTURE_INVENTORIES,
    automatic_outcome,
    blinded_trial_plan,
    phrase_medial_edge_gate,
    protocol_record,
)


def test_protocol_selects_first_gate_clean_unheard_fixture_in_each_inventory() -> None:
    protocol = protocol_record()
    assert protocol["status"] == "frozen_before_any_unseen_decode"
    assert len(protocol["fixtures"]) == 3
    for fixture in protocol["fixtures"]:
        expected = EXPECTED_SELECTIONS[fixture["fixture_id"]]
        assert fixture["selected_order"] == 1
        assert fixture["text"] == expected["text"]
        assert fixture["plan_sha256"] == expected["plan_sha256"]
        assert fixture["previously_unheard_artifact_check"] == {
            "source_text_absent": True,
            "plan_hash_absent": True,
        }


def test_protocol_keeps_unchanged_splice_and_exact_decode_ceiling() -> None:
    protocol = protocol_record()
    assert protocol["scope"]["decoder_attempt_ceiling"] == 9
    assert len(protocol["render_manifest"]) == 9
    assert protocol["intervention"]["taper"] == {
        "kind": "raised cosine",
        "milliseconds_each_edge": TAPER_MS,
        "samples_each_edge": TAPER_SAMPLES,
    }
    assert protocol["scope"]["api_calls"] == 0
    assert protocol["scope"]["replacement_fixtures"] == 0


def test_fixture_roles_and_anchor_maps_cover_the_required_shapes() -> None:
    protocol = protocol_record()
    rows = {row["fixture_id"]: row for row in protocol["fixtures"]}
    assert rows["phrase-medial-continuous"]["anchor_occurrence_map"] == [0]
    assert rows["phrase-final-new-context"]["anchor_occurrence_map"] == [1]
    assert rows["multiple-repeated-target"]["anchor_occurrence_map"] == [0, 1]
    repeated = rows["multiple-repeated-target"]
    assert repeated["target_occurrence_count"] == 2
    assert len(set(repeated["target_word_indexes"])) == 2
    assert len(FIXTURE_INVENTORIES) == 3


def test_descriptive_windows_neither_fail_nor_rescue_primary() -> None:
    assert automatic_outcome(True, False, False) is True
    assert automatic_outcome(False, True, True) is False


def test_phrase_medial_edge_rule_requires_both_edges_inside_neighbors() -> None:
    words = [
        {"start_sample": 0, "end_sample_exclusive": 100},
        {"start_sample": 100, "end_sample_exclusive": 200},
        {"start_sample": 200, "end_sample_exclusive": 300},
    ]
    passing = phrase_medial_edge_gate(
        1, words, {"start_sample": 50, "end_sample_exclusive": 250}
    )
    assert passing["pass"] is True
    failing = phrase_medial_edge_gate(
        1, words, {"start_sample": 0, "end_sample_exclusive": 250}
    )
    assert failing["pass"] is False


def test_blind_layout_has_one_identity_and_one_lens_trial_per_fixture() -> None:
    layout = blinded_trial_plan()
    assert len(layout) == 6
    assert [row["trial_id"] for row in layout] == [
        f"comparison-{index:02d}" for index in range(1, 7)
    ]
    for inventory in FIXTURE_INVENTORIES:
        trials = [row for row in layout if row["fixture_id"] == inventory.fixture_id]
        assert {row["condition"] for row in trials} == {
            "identity-control",
            "spliced-lens",
        }
        assert all(set(row["side_roles"]) == {"A", "B"} for row in trials)
