from __future__ import annotations

import json

from earshift_bakeoff.config import ROOT
from earshift_bakeoff.util import sha256_file


PROTOCOL = ROOT / "rules" / "bilingual-v8-carrier-retry-correction-v1.json"


def _protocol() -> dict:
    return json.loads(PROTOCOL.read_text(encoding="utf-8"))


def test_carrier_retry_protocol_is_frozen_zero_api_and_bound() -> None:
    protocol = _protocol()

    assert protocol["schema_version"] == 1
    assert protocol["protocol_version"] == (
        "bilingual-v8-carrier-retry-correction-v1"
    )
    assert protocol["status"] == "frozen_before_first_candidate_audio"
    assert protocol["candidate_version"] == "v8-carrier-retry-v1"
    assert protocol["api_calls_allowed"] == 0
    assert protocol["production_enabled"] is False
    for binding in protocol["bindings"]:
        assert sha256_file(ROOT / binding["path"]) == binding["sha256"]


def test_carrier_retry_protocol_freezes_first_pass_search_and_plans() -> None:
    protocol = _protocol()
    plans = protocol["candidate_plans"]

    assert protocol["selection_rule"] == (
        "first_complete_composition_gate_pass_under_frozen_plan_order"
    )
    assert [row["round"] for row in plans] == [1, 2, 3, 4, 5]
    assert [row["minimum_attempt"] for row in plans] == [4, 5, 6, 7, 8]
    assert [row["selected_attempt"] for row in plans] == [4, 5, 6, 7, 9]
    assert all(
        row["receipt"]["word_candidate_attempts"][:3] == [0, 0, 0]
        and row["receipt"]["word_candidate_attempts"][4] == 0
        and row["receipt"]["retried_word"]["source_casefold"] == "took"
        and row["receipt"]["gates"]["written_and_espeak_gate_pass"]
        and row["receipt"]["gates"]["supplemental_phone_gates_pass"]
        and row["receipt"]["gates"]["repeated_word_invariant_pass"]
        for row in plans
    )
    assert protocol["execution"] == {
        "candidate_order_frozen_before_audio": True,
        "stop_at_first_complete_pass": True,
        "maximum_candidate_pair_renders": 5,
        "current_context_anchor_decodes_per_attempt": 6,
        "replacement_fixture_allowed": False,
        "threshold_changes_allowed": False,
        "manual_or_listening_selection_allowed": False,
        "human_review_generated": False,
    }
