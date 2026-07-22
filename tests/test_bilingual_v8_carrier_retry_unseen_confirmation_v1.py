from __future__ import annotations

import json

from earshift_bakeoff.config import ROOT
from earshift_bakeoff.util import sha256_file


PROTOCOL = (
    ROOT / "rules" / "bilingual-v8-carrier-retry-unseen-confirmation-v1.json"
)


def _protocol() -> dict:
    return json.loads(PROTOCOL.read_text(encoding="utf-8"))


def test_adaptive_carrier_unseen_protocol_is_zero_api_bound_and_novel() -> None:
    protocol = _protocol()

    assert protocol["schema_version"] == 1
    assert protocol["protocol_version"] == (
        "bilingual-v8-carrier-retry-unseen-confirmation-v1"
    )
    assert protocol["status"] == "frozen_before_first_unseen_audio"
    assert protocol["candidate_version"] == "v8-adaptive-carrier-v1"
    assert protocol["api_calls_allowed"] == 0
    assert protocol["production_enabled"] is False
    assert protocol["novelty_freeze"] == {
        "parent_commit": "38e73e6",
        "all_nine_candidate_texts_absent_from_parent_tree": True,
        "selection_used_audio": False,
        "selection_used_acoustic_outcomes": False,
        "selection_rule": "first_planner_gate_clean_candidate_under_written_order",
    }
    for binding in protocol["bindings"]:
        assert sha256_file(ROOT / binding["path"]) == binding["sha256"]


def test_adaptive_carrier_unseen_denominator_and_algorithm_are_frozen() -> None:
    protocol = _protocol()
    fixtures = protocol["fixture_groups"]

    assert [row["fixture_id"] for row in fixtures] == [
        "heart_adaptive_unseen",
        "michael_adaptive_unseen",
        "dora_adaptive_unseen",
    ]
    assert all(len(row["candidate_inventory"]) == 3 for row in fixtures)
    assert all(
        row["selected_fixture"]["text"] == row["candidate_inventory"][0]
        for row in fixtures
    )
    assert sum(
        sum(row["selected_fixture"]["selected_rule_occurrences"].values())
        for row in fixtures
    ) == 17
    assert protocol["maximum_retry_rounds_per_fixture"] == 5
    assert protocol["algorithm"]["maximum_attempts_per_fixture"] == 6
    assert protocol["algorithm"]["no_threshold_or_renderer_change"] is True
    assert protocol["algorithm"]["no_listening_selection"] is True
    assert protocol["algorithm"]["no_fixture_replacement"] is True
    assert all(
        row["selected_fixture"]["initial_plan_receipt"]["gates"][
            "written_and_espeak_gate_pass"
        ]
        and row["selected_fixture"]["initial_plan_receipt"]["gates"][
            "supplemental_phone_gates_pass"
        ]
        and row["selected_fixture"]["initial_plan_receipt"]["gates"][
            "repeated_word_invariant_pass"
        ]
        for row in fixtures
    )
