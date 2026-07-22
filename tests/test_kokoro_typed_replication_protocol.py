from __future__ import annotations

import json
from pathlib import Path

from earshift_bakeoff.config import sha256_json
from earshift_bakeoff.kokoro_typed_replication_protocol import (
    ENGINE_COMMIT,
    FIXTURES,
    LOCALIZATION_MINIMUM,
    SELECTED_SPAN,
    blinded_trial_plan,
    protocol_record,
)

_RUN_DIR = (
    Path(__file__).resolve().parents[1]
    / "artifacts"
    / "typed-engine"
    / "20260716-kokoro-typed-replication-v1"
)


def test_fixture_plans_are_frozen_gate_clean_and_cover_three_roles() -> None:
    protocol = protocol_record()
    records = protocol["fixtures"]
    assert len(records) == 3
    assert [row["fixture_id"] for row in records] == [
        "single-target",
        "multi-target-repeated",
        "rhythm-punctuation-weak",
    ]
    assert [row["plan_sha256"] for row in records] == [
        fixture.expected_plan_sha256 for fixture in FIXTURES
    ]
    assert [row["target_occurrence_count"] for row in records] == [1, 2, 1]
    assert all(row["gate_summary"]["espeak_gate_pass"] for row in records)
    assert all(row["gate_summary"]["kokoro_phone_gate_pass"] for row in records)

    repeated = records[1]
    target_words = [
        repeated["words"][index] for index in repeated["target_word_indexes"]
    ]
    assert target_words[0]["neutral_phone"] == target_words[1]["neutral_phone"]
    assert target_words[0]["lens_phone"] == target_words[1]["lens_phone"]
    assert target_words[0]["neutral_surface"] == target_words[1]["neutral_surface"]
    assert target_words[0]["lens_surface"] == target_words[1]["lens_surface"]


def test_protocol_is_zero_api_hash_bound_and_has_exact_manifest() -> None:
    protocol = protocol_record()
    assert protocol["engine"]["commit"] == ENGINE_COMMIT
    assert protocol["scope"]["api_calls"] == 0
    assert protocol["scope"]["paid_calls"] == 0
    assert protocol["scope"]["logical_wav_outputs"] == 9
    assert protocol["scope"]["candidate_span"] == SELECTED_SPAN
    assert len(protocol["render_manifest"]) == 9
    assert [row["role"] for row in protocol["render_manifest"]] == [
        "neutral",
        "identity",
        "lens",
    ] * 3
    assert protocol["parents"]["engine_parity"]["pass"] is True
    assert protocol["parents"]["v4"]["selected_span_acoustic_pass"] is True
    assert protocol["parents"]["creator_product_qc"]["selected_candidate"] == (
        SELECTED_SPAN
    )
    assert len(protocol["protocol_sha256"]) == 64


def test_acceptance_rules_are_conjunctive_and_replication_only_is_separate() -> None:
    protocol = protocol_record()
    assert protocol["replication_only_acoustic_gate"]["never_a_runtime_gate"] is True
    assert protocol["replication_only_localization_gate"]["never_a_runtime_gate"] is (
        True
    )
    assert (
        protocol["replication_only_localization_gate"][
            "minimum_inside_squared_difference_energy_fraction"
        ]
        == LOCALIZATION_MINIMUM
    )
    assert protocol["blind_listener_protocol"]["run_pass"] == (
        "all three fixtures pass"
    )
    assert (
        "every runtime, acoustic, localization"
        in protocol["predetermined_outcomes"]["replication_pass"]
    )
    assert "no replacement" in protocol["stopping_rule"]


def test_blind_plan_is_repeatable_and_has_both_branches_for_every_fixture() -> None:
    first = blinded_trial_plan()
    assert first == blinded_trial_plan()
    assert len(first) == 6
    assert [row["trial_id"] for row in first] == [
        f"comparison-{index:02d}" for index in range(1, 7)
    ]
    for fixture in FIXTURES:
        rows = [row for row in first if row["fixture_id"] == fixture.fixture_id]
        assert {row["condition"] for row in rows} == {
            "identity-catch",
            "lens-candidate",
        }
        assert all(set(row["side_roles"]) == {"A", "B"} for row in rows)


def test_frozen_protocol_artifact_is_intact() -> None:
    # The 2026-07-16 freeze predates later uv.lock changes, so prepare() can no
    # longer reproduce it byte-for-byte; the durable invariant is the frozen
    # artifact's internal hash and its binding to the recorded analysis.
    frozen = json.loads((_RUN_DIR / "protocol.json").read_text(encoding="utf-8"))
    payload = dict(frozen)
    digest = payload.pop("protocol_sha256")
    assert digest == sha256_json(payload)
    analysis = json.loads((_RUN_DIR / "analysis.json").read_text(encoding="utf-8"))
    assert analysis["protocol_sha256"] == digest
