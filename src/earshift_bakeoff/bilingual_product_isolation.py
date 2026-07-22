from __future__ import annotations

from copy import deepcopy
from typing import Any

from .bilingual_vowel_engine import BilingualVowelPlan


ISOLATED_VALIDATION_PROFILE_VERSION = "isolated-validation-profile-v1"


def isolate_listener_profile(
    profile: dict[str, Any], rule_id: str
) -> dict[str, Any]:
    """Return a validation-only profile in which exactly one changed rule remains.

    Product synthesis still composes every enabled rule. This profile exists only
    to obtain atomic evidence for one matrix cell without carrier/filler rules
    contaminating its acoustic endpoint.
    """

    isolated = deepcopy(profile)
    known = False
    vowel_rules = []
    for rule in isolated.get("vowel_rules", ()):
        row = dict(rule)
        if row["id"] == rule_id:
            known = True
        else:
            row["target"] = row["source"]
            row["acoustic_status"] = "validation_identity_control"
        vowel_rules.append(row)
    isolated["vowel_rules"] = vowel_rules

    consonant_rules = []
    for rule in isolated.get("consonant_rules", ()):
        row = dict(rule)
        if row["id"] == rule_id:
            known = True
        else:
            row["target"] = row["source"]
            row["acoustic_status"] = "validation_identity_control"
        consonant_rules.append(row)
    isolated["consonant_rules"] = consonant_rules

    insertion_rules = []
    for rule in isolated.get("insertion_rules", ()):
        if rule["id"] == rule_id:
            insertion_rules.append(dict(rule))
            known = True
    isolated["insertion_rules"] = insertion_rules

    prosody_rules = []
    for rule in isolated.get("prosody_rules", ()):
        row = dict(rule)
        if row["id"] == rule_id:
            known = True
        else:
            row["operation"] = "identity"
            row["architecture_status"] = "validation_identity_control"
        prosody_rules.append(row)
    isolated["prosody_rules"] = prosody_rules

    if not known:
        raise ValueError(f"unknown listener rule for isolated validation: {rule_id}")
    isolated["validation_profile_version"] = ISOLATED_VALIDATION_PROFILE_VERSION
    isolated["isolated_validation_rule_id"] = rule_id
    return isolated


def active_changed_rule_ids(plan: BilingualVowelPlan) -> tuple[str, ...]:
    rule_ids = set(plan.active_prosody_rule_ids)
    for word in plan.words:
        for attribute in (
            "vowel_occurrences",
            "consonant_occurrences",
            "insertion_occurrences",
            "prosody_occurrences",
        ):
            rule_ids.update(
                occurrence.rule_id
                for occurrence in getattr(word, attribute, ())
                if occurrence.changed
            )
    return tuple(sorted(rule_ids))
