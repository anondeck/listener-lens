from __future__ import annotations

from earshift_bakeoff.bilingual_candidate_isolation import (
    isolate_listener_profile_set,
)
from earshift_bakeoff.bilingual_product_isolation import (
    ISOLATED_VALIDATION_PROFILE_VERSION,
    isolate_listener_profile,
)
from earshift_bakeoff.bilingual_listener_engine import load_listener_profiles


def test_isolated_profile_retains_exactly_one_changed_vowel_rule() -> None:
    profile = load_listener_profiles()["en-US-to-pt-BR-listener-v2"]
    isolated = isolate_listener_profile(profile, "enpt.ae_eh")

    changed_vowels = [
        row["id"]
        for row in isolated["vowel_rules"]
        if row["source"] != row["target"]
    ]
    assert changed_vowels == ["enpt.ae_eh"]
    assert all(
        row["source"] == row["target"]
        for row in isolated["consonant_rules"]
    )
    assert isolated["insertion_rules"] == []
    assert all(row["operation"] == "identity" for row in isolated["prosody_rules"])
    assert (
        isolated["validation_profile_version"]
        == ISOLATED_VALIDATION_PROFILE_VERSION
    )


def test_isolated_profile_retains_exactly_one_insertion_or_prosody_rule() -> None:
    profile = load_listener_profiles()["en-US-to-pt-BR-listener-v2"]
    insertion = isolate_listener_profile(
        profile, "enpt.illicit_coda_epenthetic_i"
    )
    prosody = isolate_listener_profile(
        profile, "enpt.lexical_stress_initial_bias"
    )

    assert [row["id"] for row in insertion["insertion_rules"]] == [
        "enpt.illicit_coda_epenthetic_i"
    ]
    assert all(
        row["source"] == row["target"] for row in insertion["vowel_rules"]
    )
    assert prosody["insertion_rules"] == []
    assert [
        row["id"]
        for row in prosody["prosody_rules"]
        if row["operation"] != "identity"
    ] == ["enpt.lexical_stress_initial_bias"]


def test_isolated_profile_rejects_unknown_rule() -> None:
    profile = load_listener_profiles()["pt-BR-to-en-US-listener-v2"]
    try:
        isolate_listener_profile(profile, "missing.rule")
    except ValueError as exc:
        assert "unknown listener rule" in str(exc)
    else:  # pragma: no cover - assertion guard
        raise AssertionError("unknown rule was accepted")


def test_isolated_profile_set_retains_exact_selected_rules() -> None:
    profile = load_listener_profiles()["en-US-to-pt-BR-listener-v2"]
    isolated = isolate_listener_profile_set(
        profile, ("enpt.ae_eh", "enpt.ih_i")
    )

    changed = {
        rule["id"]
        for rule in isolated["vowel_rules"]
        if rule["source"] != rule["target"]
    }
    assert changed == {"enpt.ae_eh", "enpt.ih_i"}
    assert isolated["isolated_validation_rule_ids"] == (
        "enpt.ae_eh",
        "enpt.ih_i",
    )
    assert "isolated_validation_rule_id" not in isolated


def test_isolated_profile_set_rejects_empty_and_duplicate_ids() -> None:
    profile = load_listener_profiles()["en-US-to-pt-BR-listener-v2"]

    for rule_ids in ((), ("enpt.ae_eh", "enpt.ae_eh")):
        try:
            isolate_listener_profile_set(profile, rule_ids)
        except ValueError as exc:
            assert "unique and nonempty" in str(exc)
        else:  # pragma: no cover - assertion guard
            raise AssertionError("invalid isolated rule set was accepted")
