from __future__ import annotations

import json

from earshift_bakeoff.config import ROOT
from earshift_bakeoff.util import sha256_file


PROTOCOL = ROOT / "rules" / "bilingual-v8-factorized-composition-v2.json"


def _protocol() -> dict:
    return json.loads(PROTOCOL.read_text(encoding="utf-8"))


def test_factorized_composition_protocol_is_frozen_before_rendering() -> None:
    protocol = _protocol()

    assert protocol["schema_version"] == 1
    assert protocol["protocol_version"] == "bilingual-v8-factorized-composition-v2"
    assert protocol["production_enabled"] is False
    assert protocol["api_calls_allowed"] == 0
    assert protocol["intervention"]["version"] == (
        "v8-rule-factorized-state-composition-v2"
    )
    assert protocol["intervention"]["neutral_pcm_must_be_bit_exact_to_v1"] is True
    assert protocol["pre_audio_amendment"] == {
        "sequence": 1,
        "reason": (
            "The first execution attempt stopped during structural validation "
            "before decoding or writing audio because independently replanning one "
            "Heart rule changed an unrelated opaque-carrier consonant. That planner "
            "coupling would violate the frozen neutral-carrier invariant."
        ),
        "candidate_audio_decoded": False,
        "candidate_audio_files_written": 0,
        "result_files_written": 0,
        "correction": (
            "Derive each atomic lens phoneme string directly from the frozen combined "
            "neutral/lens plan by activating only the named rule. This preserves the "
            "exact combined carrier and the existing complete target-word v8 state "
            "scope while changing only cross-rule encoder factorization."
        ),
        "threshold_fixture_and_audio_contract_changes": False,
    }
    for binding in protocol["bindings"].values():
        assert sha256_file(ROOT / binding["path"]) == binding["sha256"]


def test_factorized_protocol_reuses_the_exact_v1_denominator() -> None:
    protocol = _protocol()
    baseline = json.loads(
        (
            ROOT
            / "artifacts"
            / "product-matrix"
            / "20260718-bilingual-v8-composition-spike-v1"
            / "results.json"
        ).read_text(encoding="utf-8")
    )
    baseline_rows = {row["fixture_id"]: row for row in baseline["fixtures"]}

    assert [row["fixture_id"] for row in protocol["fixtures"]] == [
        "heart_two_v8_rules",
        "michael_three_v8_rules",
        "dora_two_v8_rules",
    ]
    for fixture in protocol["fixtures"]:
        prior = baseline_rows[fixture["fixture_id"]]
        assert fixture["text"] == prior["text"]
        assert fixture["profile_id"] == prior["profile_id"]
        assert fixture["voice_id"] == prior["voice_id"]
        assert fixture["selected_rule_occurrences"] == prior[
            "selected_rule_occurrences"
        ]
        assert fixture["expected_omitted_rule_ids"] == prior["omitted_rule_ids"]
        assert fixture["plan_sha256"] == prior["plan_sha256"]
        assert fixture["baseline_neutral_pcm_sha256"] == prior["audio"]["neutral"][
            "pcm_sha256"
        ]
        assert fixture["baseline_automatic_pass"] is prior["automatic_pass"]


def test_factorized_outcome_requires_no_regression_and_dora_rescue() -> None:
    criteria = _protocol()["automatic_pass_criteria"]

    assert criteria["both_v1_passing_fixtures_must_remain_passes"] is True
    assert criteria["dora_v1_failure_must_pass_without_threshold_or_fixture_change"] is True
    assert criteria["every_rule_directional_pass"] is True
    assert criteria["identity_false_positive_count"] == 0
