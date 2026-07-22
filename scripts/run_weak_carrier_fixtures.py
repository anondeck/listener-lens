from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from earshift_bakeoff.config import Paths, stable_json
from earshift_bakeoff.listener_lens import (
    LENS_RULES_PATH,
    TRANSFORM_ALGORITHM_VERSION,
    DatabaseNonceChecker,
    ListenerLensEngine,
    WORD_RE,
    _vowel_units,
)
from earshift_bakeoff.util import atomic_write_json, sha256_file


RUN_ID = "20260716-typed-weak-carrier-v1"
PROFILE_ID = "en-to-pt-BR-vowel-lens"
FIXTURES = (
    ("repeated_function_words", "The cat and the cat.", ("weak", "content", "content", "weak", "content")),
    ("adjacency", "The cat is in the back.", ("weak", "content", "weak", "weak", "weak", "content")),
    ("punctuation", "The cat, and the back!", ("weak", "content", "content", "weak", "content")),
    ("no_enabled_rule", "The day is in the sun.", ("weak", "content", "weak", "weak", "weak", "content")),
    ("rule_bearing_function_word", "That cat is back.", ("content", "content", "weak", "content")),
    ("current_smoke", "What a great day it is to catch some sun.", ("content", "weak", "content", "content", "weak", "weak", "weak", "content", "weak", "content")),
)


def punctuation_skeleton(text: str) -> str:
    return WORD_RE.sub("", text)


def main() -> None:
    engine = ListenerLensEngine()
    if not isinstance(engine.nonce_checker, DatabaseNonceChecker):
        raise RuntimeError("frozen fixtures require the pinned database nonce checker")
    report: dict = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "transform_algorithm_version": TRANSFORM_ALGORITHM_VERSION,
        "rules_sha256": sha256_file(LENS_RULES_PATH),
        "gate_database_sha256": sha256_file(Paths().gate_db),
        "fixtures": [],
    }
    all_rejections: Counter[str] = Counter()
    total_candidate_attempts = 0
    total_selected_mappings = 0
    for fixture_id, text, expected_roles in FIXTURES:
        result = engine.transform(text, PROFILE_ID)
        roles = tuple(word.carrier_role for word in result.words)
        if roles != expected_roles:
            raise RuntimeError(f"{fixture_id} role drift: {roles!r}")
        if len(result.words) != len(WORD_RE.findall(text)):
            raise RuntimeError(f"{fixture_id} word-count drift")
        if punctuation_skeleton(result.neutral_script) != punctuation_skeleton(text):
            raise RuntimeError(f"{fixture_id} neutral punctuation drift")
        if punctuation_skeleton(result.lens_script) != punctuation_skeleton(text):
            raise RuntimeError(f"{fixture_id} lens punctuation drift")
        for word in result.words:
            if word.syllables != len(_vowel_units(word.source_ipa)):
                raise RuntimeError(f"{fixture_id} source syllable drift")
            if word.carrier_role == "weak" and word.slots:
                raise RuntimeError(f"{fixture_id} weak word carries an enabled slot")
            if word.carrier_role == "weak" and word.neutral_surface != word.lens_surface:
                raise RuntimeError(f"{fixture_id} weak word broke neutral/lens alignment")
        for side in ("neutral", "lens"):
            surfaces = [getattr(word, f"{side}_surface") for word in result.words]
            previous = None
            for surface in surfaces:
                decision = engine.nonce_checker.check(surface, "en", previous)
                if not decision.accepted:
                    raise RuntimeError(
                        f"{fixture_id} {side} adjacency gate failed: {decision.rejection_reason}"
                    )
                previous = surface
        if fixture_id == "repeated_function_words":
            if result.words[0].neutral_surface != result.words[3].neutral_surface:
                raise RuntimeError("repeated weak source word changed carrier")
            if result.words[1].neutral_surface != result.words[4].neutral_surface:
                raise RuntimeError("repeated rule-bearing source word changed carrier")
        if fixture_id == "no_enabled_rule":
            if result.comparison_available or result.slots:
                raise RuntimeError("no-rule fixture exposed an audio comparison")
        if fixture_id == "rule_bearing_function_word":
            if result.words[0].carrier_role != "content" or not result.words[0].slots:
                raise RuntimeError("rule-bearing function word lost its enabled slot")
        weak_report = result.weak_form_report
        total_candidate_attempts += weak_report.candidate_attempt_count
        total_selected_mappings += weak_report.selected_mapping_count
        all_rejections.update(weak_report.rejection_reason_counts)
        payload = result.to_dict()
        report["fixtures"].append(
            {
                "fixture_id": fixture_id,
                "source_text": text,
                "result_sha256": hashlib.sha256(
                    stable_json(payload).encode("utf-8")
                ).hexdigest(),
                "neutral_script": result.neutral_script,
                "lens_script": result.lens_script,
                "comparison_available": result.comparison_available,
                "roles": list(roles),
                "slot_count": len(result.slots),
                "weak_form_report": payload["weak_form_report"],
            }
        )

    inventory_audit = []
    inventory_rejections: Counter[str] = Counter()
    for index, candidate in enumerate(engine.rules["weak_carrier_policy"]["candidate_inventory"]):
        decision = engine.nonce_checker.check(candidate["surface"], "en", None)
        predicted_syllables = (
            len(_vowel_units(decision.predicted_ipa)) if decision.predicted_ipa else 0
        )
        reason = decision.rejection_reason
        if decision.accepted and predicted_syllables != candidate["syllables"]:
            reason = "predicted_syllable_mismatch"
        accepted = decision.accepted and reason is None
        if reason:
            inventory_rejections[reason] += 1
        inventory_audit.append(
            {
                "candidate_index": index,
                "surface": candidate["surface"],
                "declared_syllables": candidate["syllables"],
                "predicted_syllables": predicted_syllables,
                "predicted_ipa": decision.predicted_ipa,
                "accepted": accepted,
                "rejection_reason": reason,
            }
        )
    report["aggregate"] = {
        "fixture_count": len(FIXTURES),
        "fixture_candidate_attempts": total_candidate_attempts,
        "fixture_selected_mappings": total_selected_mappings,
        "fixture_gate_yield": (
            total_selected_mappings / total_candidate_attempts
            if total_candidate_attempts
            else None
        ),
        "fixture_rejection_reason_counts": dict(sorted(all_rejections.items())),
        "inventory_accepted": sum(item["accepted"] for item in inventory_audit),
        "inventory_attempted": len(inventory_audit),
        "inventory_gate_yield": sum(item["accepted"] for item in inventory_audit) / len(inventory_audit),
        "inventory_rejection_reason_counts": dict(sorted(inventory_rejections.items())),
    }
    report["inventory_audit"] = inventory_audit
    output = Paths().artifacts / "weak-carrier" / RUN_ID / "fixture-report.json"
    atomic_write_json(output, report)
    print(json.dumps(report["aggregate"], indent=2))


if __name__ == "__main__":
    main()
