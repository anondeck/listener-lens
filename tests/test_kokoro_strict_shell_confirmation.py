from __future__ import annotations

import json

from earshift_bakeoff.config import Paths, sha256_json
from earshift_bakeoff.kokoro_strict_shell_confirmation import (
    EXPECTED_PLAN_SHA256,
    EXPECTED_TEXT,
    NEW_FIXTURE_ID,
    REUSED_FIXTURE_IDS,
    _layout,
)


def _frozen_protocol() -> dict:
    path = (
        Paths().artifacts
        / "typed-engine"
        / "20260717-kokoro-strict-shell-confirmation-v1"
        / "protocol.json"
    )
    protocol = json.loads(path.read_text(encoding="utf-8"))
    payload = dict(protocol)
    digest = payload.pop("protocol_sha256")
    assert digest == sha256_json(payload)
    return protocol


def test_protocol_changes_only_target_word_coverage_boundary() -> None:
    protocol = _frozen_protocol()
    assert protocol["status"] == "frozen_before_strict_shell_decode"
    assert protocol["change"]["mechanism"] == (
        "coverage boundary for rule-bearing source words"
    )
    assert protocol["change"]["supported_shape"] == (
        "exact C-stress-/ae/-C source phone plan"
    )
    assert protocol["change"]["renderer_or_gate_changes"] == "none"
    assert protocol["scope"]["new_decoder_attempt_ceiling"] == 3
    assert protocol["scope"]["rerendered_successful_cells"] == 0
    assert protocol["scope"]["api_calls"] == 0


def test_protocol_reuses_exactly_two_hash_bound_parent_successes() -> None:
    parent = _frozen_protocol()["reused_parent_successes"]
    assert parent["classification_preserved"] == "validated_shell_automatic_failed"
    rows = parent["reused_without_rerender"]
    assert tuple(row["fixture_id"] for row in rows) == REUSED_FIXTURE_IDS
    assert all(row["automatic_pass"] is True for row in rows)


def test_first_gate_clean_medial_fixture_is_frozen_and_unheard() -> None:
    fixture = _frozen_protocol()["new_fixture"]
    assert fixture["fixture_id"] == NEW_FIXTURE_ID
    assert fixture["selected_order"] == 1
    assert fixture["text"] == EXPECTED_TEXT
    assert fixture["plan_sha256"] == EXPECTED_PLAN_SHA256
    assert fixture["target_word_indexes"] == [2]


def test_review_layout_covers_new_and_reused_fixtures_without_labels() -> None:
    layout = _layout()
    assert len(layout) == 6
    assert {row["fixture_id"] for row in layout} == {
        NEW_FIXTURE_ID,
        *REUSED_FIXTURE_IDS,
    }
