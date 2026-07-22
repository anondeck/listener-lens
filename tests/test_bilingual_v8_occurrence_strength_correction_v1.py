from __future__ import annotations

import json

from earshift_bakeoff.config import ROOT
from earshift_bakeoff.util import sha256_file


PROTOCOL = ROOT / "rules" / "bilingual-v8-occurrence-strength-correction-v1.json"


def _protocol() -> dict:
    return json.loads(PROTOCOL.read_text(encoding="utf-8"))


def test_occurrence_strength_protocol_is_bound_zero_api_and_nonpromotional() -> None:
    protocol = _protocol()

    assert protocol["schema_version"] == 1
    assert (
        protocol["protocol_version"]
        == "bilingual-v8-occurrence-strength-correction-v1"
    )
    assert protocol["status"] == "frozen_before_first_candidate_render"
    assert protocol["api_calls_allowed"] == 0
    assert protocol["production_enabled"] is False
    assert protocol["parent_evidence"] == {
        "protocol_version": "bilingual-v8-composition-unseen-confirmation-v2",
        "classification": "unseen_v8_composition_automatic_failed_preserve_exact_result",
        "automatic_pass_count": 2,
        "fixture_count": 3,
        "preservation_rule": (
            "The frozen 2/3 unseen result and exact failed measurements remain "
            "unchanged regardless of this exploratory correction outcome."
        ),
    }
    for binding in protocol["bindings"]:
        assert sha256_file(ROOT / binding["path"]) == binding["sha256"]


def test_occurrence_strength_protocol_freezes_exact_failure_and_search() -> None:
    protocol = _protocol()

    assert protocol["failed_occurrence_binding"] == {
        "fixture_id": "heart_unseen_continuous",
        "text": "One good cook took books.",
        "rule_id": "enpt.uh_u",
        "rule_occurrence_ordinal_zero_based": 2,
        "global_occurrence_index": 3,
        "word_index": 3,
        "source": "ʊ",
        "target": "u",
        "failure_mechanism": "target_gain_gate_only",
    }
    intervention = protocol["intervention"]
    assert intervention["baseline_equivalence_strength"] == 1.0
    assert intervention["alternative_strength_order"] == [
        0.75,
        1.25,
        0.5,
        1.5,
        2.0,
    ]
    assert intervention["maximum_alternative_render_count"] == 5
    assert intervention["selection_rule"] == (
        "first_complete_composition_gate_pass_under_frozen_order"
    )
    assert protocol["automatic_gates"]["all_five_occurrences_must_pass"] is True
    assert protocol["selection_and_exclusion"] == {
        "listening_selection_allowed": False,
        "manual_review_generated": False,
        "fixture_replacement_allowed": False,
        "selective_rerender_allowed": False,
        "threshold_change_allowed": False,
        "strength_reordering_allowed": False,
        "posthoc_best_candidate_selection_allowed": False,
        "failed_attempts_remain_in_denominator": True,
    }
