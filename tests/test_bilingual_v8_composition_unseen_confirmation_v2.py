from __future__ import annotations

import json

from earshift_bakeoff.config import ROOT
from earshift_bakeoff.util import sha256_file


PROTOCOL = (
    ROOT / "rules" / "bilingual-v8-composition-unseen-confirmation-v2.json"
)
KNOWN_PROTOCOL = ROOT / "rules" / "bilingual-v8-composition-spike-v1.json"


def _protocol() -> dict:
    return json.loads(PROTOCOL.read_text(encoding="utf-8"))


def test_unseen_composition_protocol_is_hash_bound_and_zero_api() -> None:
    protocol = _protocol()

    assert protocol["schema_version"] == 1
    assert (
        protocol["protocol_version"]
        == "bilingual-v8-composition-unseen-confirmation-v2"
    )
    assert protocol["production_enabled"] is False
    assert protocol["api_calls_allowed"] == 0
    assert protocol["novelty_freeze"] == {
        "parent_commit": "42859d0",
        "candidate_texts_absent_from_parent_worktree": True,
        "selection_used_listening": False,
        "selection_used_acoustic_outcomes": False,
        "note": (
            "All candidate strings were absent from the repository at the parent "
            "commit. Dry planning inspected only source G2P, candidate-rung "
            "eligibility, nonce/adjacency validity, plan identity, and occurrence "
            "counts before this protocol was frozen."
        ),
    }
    for binding in protocol["bindings"].values():
        assert sha256_file(ROOT / binding["path"]) == binding["sha256"]


def test_unseen_composition_denominator_and_first_candidate_are_frozen() -> None:
    protocol = _protocol()
    fixtures = protocol["fixture_groups"]

    assert [fixture["fixture_id"] for fixture in fixtures] == [
        "heart_unseen_continuous",
        "michael_unseen_repeated",
        "dora_unseen_phrase_final",
    ]
    assert [fixture["voice_id"] for fixture in fixtures] == [
        "af_heart",
        "am_michael",
        "pf_dora",
    ]
    assert all(len(fixture["candidate_inventory"]) == 3 for fixture in fixtures)
    assert all(
        fixture["selected_fixture"]["text"] == fixture["candidate_inventory"][0]
        for fixture in fixtures
    )
    assert sum(
        len(fixture["selected_fixture"]["selected_rule_occurrences"])
        for fixture in fixtures
    ) == 7
    assert sum(
        sum(fixture["selected_fixture"]["selected_rule_occurrences"].values())
        for fixture in fixtures
    ) == 16
    assert fixtures[1]["fixture_role"] == (
        "repeated_source_words_and_multiple_targets"
    )
    michael_text = fixtures[1]["selected_fixture"]["text"].casefold()
    assert michael_text.count("black") == 2
    assert michael_text.count("cats") == 2


def test_unseen_fixture_texts_do_not_reuse_known_composition_texts() -> None:
    protocol = _protocol()
    known = json.loads(KNOWN_PROTOCOL.read_text(encoding="utf-8"))
    known_texts = {fixture["text"] for fixture in known["fixtures"]}
    candidate_texts = {
        text
        for fixture in protocol["fixture_groups"]
        for text in fixture["candidate_inventory"]
    }

    assert known_texts.isdisjoint(candidate_texts)
    assert protocol["candidate_constraints"] == {
        "candidate_rung": "v8",
        "minimum_rule_count": 2,
        "maximum_rule_count": 3,
        "same_voice_required": True,
        "first_gate_clean_candidate_under_ordering": True,
        "one_render_set_per_fixture": True,
        "replacement_fixtures_allowed": False,
        "selective_rerenders_allowed": False,
        "listening_selection_allowed": False,
        "acoustic_selection_allowed": False,
        "threshold_changes_allowed": False,
        "shared_current_context_anchor_decodes_per_fixture": 6,
        "omitted_rules_must_be_reported": True,
        "integrated_synthesis_path": "combined_v8_one_decode",
    }
