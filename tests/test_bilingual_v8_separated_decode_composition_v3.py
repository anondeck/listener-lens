from __future__ import annotations

import json

from earshift_bakeoff.config import ROOT
from earshift_bakeoff.util import sha256_file


PROTOCOL = (
    ROOT / "rules" / "bilingual-v8-separated-decode-composition-v3.json"
)


def _protocol() -> dict:
    return json.loads(PROTOCOL.read_text(encoding="utf-8"))


def test_separated_decode_protocol_is_frozen_before_rendering() -> None:
    protocol = _protocol()

    assert protocol["schema_version"] == 1
    assert protocol["protocol_version"] == (
        "bilingual-v8-separated-decode-composition-v3"
    )
    assert protocol["production_enabled"] is False
    assert protocol["api_calls_allowed"] == 0
    assert protocol["intervention"]["version"] == (
        "v8-rule-separated-decode-composition-v3"
    )
    assert protocol["intervention"]["rule_output_windows_must_not_overlap"] is True
    for binding in protocol["bindings"].values():
        assert sha256_file(ROOT / binding["path"]) == binding["sha256"]


def test_separated_decode_protocol_preserves_the_exact_denominator() -> None:
    protocol = _protocol()
    baseline = json.loads(
        (
            ROOT
            / "artifacts"
            / "product-matrix"
            / "20260718-bilingual-v8-factorized-composition-v2"
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


def test_separated_decode_outcome_disambiguates_the_remaining_mechanism() -> None:
    protocol = _protocol()
    criteria = protocol["automatic_pass_criteria"]

    assert criteria["both_existing_passes_must_remain_passes"] is True
    assert criteria["dora_o_goat_must_pass_without_threshold_or_fixture_change"] is True
    assert protocol["outcomes"]["all_fixtures_pass"].endswith(
        "eligible_for_unseen_confirmation"
    )
    assert "current-carrier realization instability" in protocol["outcomes"][
        "interpretation_limit"
    ]
