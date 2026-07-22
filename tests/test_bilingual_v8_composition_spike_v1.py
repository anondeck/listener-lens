from __future__ import annotations

import json

from earshift_bakeoff.config import ROOT
from earshift_bakeoff.util import sha256_file


PROTOCOL = ROOT / "rules" / "bilingual-v8-composition-spike-v1.json"


def _protocol() -> dict:
    return json.loads(PROTOCOL.read_text(encoding="utf-8"))


def test_composition_protocol_is_frozen_before_local_rendering() -> None:
    protocol = _protocol()

    assert protocol["schema_version"] == 1
    assert protocol["protocol_version"] == "bilingual-v8-composition-spike-v1"
    assert protocol["production_enabled"] is False
    assert protocol["api_calls_allowed"] == 0
    assert protocol["candidate_constraints"] == {
        "candidate_rung": "v8",
        "minimum_rule_count": 2,
        "maximum_rule_count": 3,
        "same_voice_required": True,
        "one_render_set_per_fixture": True,
        "replacement_fixtures_allowed": False,
        "listening_selection_allowed": False,
        "threshold_changes_allowed": False,
        "shared_current_context_anchor_decodes_per_fixture": 6,
        "omitted_rules_must_be_reported": True,
    }
    for binding in protocol["bindings"].values():
        assert sha256_file(ROOT / binding["path"]) == binding["sha256"]


def test_composition_protocol_has_exact_bounded_fixture_denominator() -> None:
    fixtures = _protocol()["fixtures"]

    assert [fixture["fixture_id"] for fixture in fixtures] == [
        "heart_two_v8_rules",
        "michael_three_v8_rules",
        "dora_two_v8_rules",
    ]
    assert [fixture["voice_id"] for fixture in fixtures] == [
        "af_heart",
        "am_michael",
        "pf_dora",
    ]
    assert sum(
        sum(fixture["selected_rule_occurrences"].values())
        for fixture in fixtures
    ) == 9
    assert all(
        2 <= len(fixture["selected_rule_occurrences"]) <= 3
        for fixture in fixtures
    )
