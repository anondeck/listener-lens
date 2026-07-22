from __future__ import annotations

import json

from earshift_bakeoff.config import Paths
from earshift_bakeoff.listener_lens import LENS_RULES_PATH, TRANSFORM_ALGORITHM_VERSION
from earshift_bakeoff.util import sha256_file


REPORT = (
    Paths().artifacts
    / "weak-carrier"
    / "20260716-typed-weak-carrier-v1"
    / "fixture-report.json"
)


def test_frozen_weak_carrier_fixture_report_matches_runtime_inputs() -> None:
    payload = json.loads(REPORT.read_text(encoding="utf-8"))

    assert payload["transform_algorithm_version"] == TRANSFORM_ALGORITHM_VERSION == 5
    assert payload["rules_sha256"] == sha256_file(LENS_RULES_PATH)
    assert payload["gate_database_sha256"] == sha256_file(Paths().gate_db)
    assert [fixture["fixture_id"] for fixture in payload["fixtures"]] == [
        "repeated_function_words",
        "adjacency",
        "punctuation",
        "no_enabled_rule",
        "rule_bearing_function_word",
        "current_smoke",
    ]
    assert payload["aggregate"] == {
        "fixture_count": 6,
        "fixture_candidate_attempts": 14,
        "fixture_selected_mappings": 14,
        "fixture_gate_yield": 1.0,
        "fixture_rejection_reason_counts": {},
        "inventory_accepted": 33,
        "inventory_attempted": 33,
        "inventory_gate_yield": 1.0,
        "inventory_rejection_reason_counts": {},
    }


def test_fixture_report_keeps_rule_bearing_function_words_out_of_weak_role() -> None:
    payload = json.loads(REPORT.read_text(encoding="utf-8"))
    rule_fixture = next(
        fixture
        for fixture in payload["fixtures"]
        if fixture["fixture_id"] == "rule_bearing_function_word"
    )
    smoke = next(
        fixture
        for fixture in payload["fixtures"]
        if fixture["fixture_id"] == "current_smoke"
    )

    assert rule_fixture["roles"][0] == "content"
    assert rule_fixture["slot_count"] == 3
    assert smoke["roles"] == [
        "content", "weak", "content", "content", "weak",
        "weak", "weak", "content", "weak", "content",
    ]
    assert smoke["slot_count"] == 1
