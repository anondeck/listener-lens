from __future__ import annotations

from copy import deepcopy
from typing import Any

from .bilingual_product_isolation import ISOLATED_VALIDATION_PROFILE_VERSION


def isolate_listener_profile_set(
    profile: dict[str, Any], rule_ids: tuple[str, ...]
) -> dict[str, Any]:
    """Keep an explicit candidate rule set changed without mutating frozen code."""

    selected = frozenset(rule_ids)
    if not selected or len(selected) != len(rule_ids):
        raise ValueError("isolated listener rule IDs must be unique and nonempty")
    isolated = deepcopy(profile)
    known: set[str] = set()

    for family in ("vowel_rules", "consonant_rules"):
        rows = []
        for rule in isolated.get(family, ()):
            row = dict(rule)
            if row["id"] in selected:
                known.add(row["id"])
            else:
                row["target"] = row["source"]
                row["acoustic_status"] = "validation_identity_control"
            rows.append(row)
        isolated[family] = rows

    isolated["insertion_rules"] = [
        dict(rule)
        for rule in isolated.get("insertion_rules", ())
        if rule["id"] in selected
    ]
    known.update(rule["id"] for rule in isolated["insertion_rules"])

    prosody_rules = []
    for rule in isolated.get("prosody_rules", ()):
        row = dict(rule)
        if row["id"] in selected:
            known.add(row["id"])
        else:
            row["operation"] = "identity"
            row["architecture_status"] = "validation_identity_control"
        prosody_rules.append(row)
    isolated["prosody_rules"] = prosody_rules

    missing = selected - known
    if missing:
        raise ValueError(
            "unknown listener rule for isolated validation: "
            + ",".join(sorted(missing))
        )
    isolated["validation_profile_version"] = ISOLATED_VALIDATION_PROFILE_VERSION
    isolated["isolated_validation_rule_ids"] = tuple(sorted(selected))
    return isolated
