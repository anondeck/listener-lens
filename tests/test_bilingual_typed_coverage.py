from __future__ import annotations

from types import SimpleNamespace

from earshift_bakeoff import bilingual_typed_coverage as coverage
from earshift_bakeoff.config import sha256_json


class _Registry:
    def __init__(self, cells):
        self.cells = {
            (cell.profile_id, cell.voice_id, cell.rule_id): cell for cell in cells
        }

    def cell(self, profile_id, voice_id, rule_id):
        return self.cells.get((profile_id, voice_id, rule_id))


def _plan(rule_ids):
    occurrences = tuple(
        SimpleNamespace(rule_id=rule_id, changed=True) for rule_id in rule_ids
    )
    word = SimpleNamespace(
        vowel_occurrences=occurrences,
        consonant_occurrences=(),
        prosody_occurrences=(),
        insertion_occurrences=(),
    )
    return SimpleNamespace(
        profile_id="profile",
        voice_id="voice",
        words=(word,),
        active_prosody_rule_ids=(),
    )


def _cell(rule_id, *, passed=True, rung="v8"):
    return SimpleNamespace(
        profile_id="profile",
        voice_id="voice",
        rule_id=rule_id,
        automatic_pass=passed,
        candidate_rung=rung,
    )


def test_protocol_is_hash_bound_and_zero_render() -> None:
    protocol = coverage.protocol_record()
    payload = dict(protocol)
    digest = payload.pop("protocol_sha256")
    assert digest == sha256_json(payload)
    assert protocol["inventory"]["lexical_limit_per_case"] == 5_000
    assert protocol["scope"] == {
        "audio_renders": 0,
        "api_calls": 0,
        "production_enabled": False,
    }


def test_classification_separates_full_partial_failed_and_unsupported() -> None:
    registry = _Registry((_cell("a"), _cell("b"), _cell("failed", passed=False)))
    assert coverage._classification(_plan(()), registry)[0] == "no_listener_change"
    assert coverage._classification(_plan(("a",)), registry) == (
        "automatic_candidate_full",
        ("a",),
        (),
    )
    assert coverage._classification(_plan(("a", "missing")), registry) == (
        "automatic_candidate_partial",
        ("a",),
        ("missing",),
    )
    assert coverage._classification(_plan(("failed",)), registry)[0] == (
        "automatic_evidence_failed"
    )
    non_v8 = _Registry((_cell("a"), _cell("b", rung="word_context")))
    assert coverage._classification(_plan(("a", "b")), non_v8)[0] == (
        "unsupported_rule_or_composition"
    )
