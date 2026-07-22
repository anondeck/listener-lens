#!/usr/bin/env python3
from __future__ import annotations

from collections import Counter
import hashlib
import json
from typing import Any

from earshift_bakeoff.bilingual_listener_engine import (
    BilingualListenerPlanner,
    load_listener_profiles,
)
from earshift_bakeoff.bilingual_product_matrix import (
    load_bilingual_product_matrix,
    load_bilingual_structural_state,
)
from earshift_bakeoff.config import Paths, stable_json
from earshift_bakeoff.kokoro_gate_bridge import KokoroGateIndex
from earshift_bakeoff.kokoro_synthesis import CONFIG_FILE, verify_model_files
from earshift_bakeoff.listener_lens import DatabaseNonceChecker
from earshift_bakeoff.portuguese_carrier_planner_v1 import (
    PortuguesePositiveOnlyIndexV1,
)
from earshift_bakeoff.util import atomic_write_json, sha256_file

from run_bilingual_product_audio_integrity_screen_v1 import FixtureAdapter, _tuple


RUN_ID = "20260717-bilingual-product-rule-isolation-audit-v1"
RUN_DIR = Paths().artifacts / "product-matrix" / RUN_ID
MANIFEST_PATH = (
    Paths().artifacts
    / "product-matrix"
    / "20260717-bilingual-product-matrix-v1"
    / "manifest.json"
)
AUDIO_RESULT_PATH = (
    Paths().artifacts
    / "product-matrix"
    / "20260717-bilingual-product-audio-integrity-screen-v1"
    / "results.json"
)


def _active_changed_rule_ids(plan: Any) -> tuple[str, ...]:
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


def _semantic_hash(payload: dict[str, Any]) -> str:
    semantic = dict(payload)
    semantic.pop("record_sha256", None)
    return hashlib.sha256(stable_json(semantic).encode("utf-8")).hexdigest()


def main() -> None:
    if RUN_DIR.exists():
        raise RuntimeError(f"refusing to overwrite rule-isolation audit: {RUN_DIR}")
    matrix = load_bilingual_product_matrix()
    load_bilingual_structural_state(matrix)
    manifest_record = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    slots = manifest_record["validation_manifest"]["slots"]
    profiles = load_listener_profiles()
    files = verify_model_files(download=False)
    model_vocab = set(
        json.loads(files[CONFIG_FILE].read_text(encoding="utf-8"))["vocab"]
    )
    nonce_checker = DatabaseNonceChecker()
    phone_indexes = (KokoroGateIndex(), PortuguesePositiveOnlyIndexV1())
    outcomes: list[dict[str, Any]] = []
    for slot in slots:
        fixture = slot["fixture_spec"]
        profile = profiles[slot["profile_id"]]
        planner = BilingualListenerPlanner(
            profile={**profile, "voice_id": slot["voice_id"]},
            adapter=FixtureAdapter(
                language_id=profile["source_language"],
                source_words=_tuple(fixture["source_words"]),
                source_phones=_tuple(fixture["source_phones"]),
                punctuation=fixture["punctuation"],
            ),
            model_vocab=model_vocab,
            nonce_checker=nonce_checker,
            phone_indexes=phone_indexes,
        )
        plan = planner.plan(fixture["text"])
        active = _active_changed_rule_ids(plan)
        isolated = active == (slot["rule_id"],)
        outcomes.append(
            {
                "logical_slot_id": slot["logical_slot_id"],
                "cell_id": slot["cell_id"],
                "profile_id": slot["profile_id"],
                "voice_id": slot["voice_id"],
                "family": slot["family"],
                "named_rule_id": slot["rule_id"],
                "context": slot["context"],
                "plan_sha256": plan.plan_sha256,
                "active_changed_rule_ids": active,
                "rule_isolated": isolated,
                "additional_active_rule_ids": tuple(
                    rule_id for rule_id in active if rule_id != slot["rule_id"]
                ),
                "safe_word_count": len(plan.words),
                "product_enabled": False,
            }
        )
    isolated_count = sum(row["rule_isolated"] for row in outcomes)
    coactivated = len(outcomes) - isolated_count
    families = {}
    for family in sorted({row["family"] for row in outcomes}):
        selected = [row for row in outcomes if row["family"] == family]
        passed = sum(row["rule_isolated"] for row in selected)
        families[family] = {
            "slot_count": len(selected),
            "isolated_count": passed,
            "coactivated_count": len(selected) - passed,
            "isolation_yield": passed / len(selected),
        }
    voices = {}
    for voice_id in ("af_heart", "am_michael", "pm_alex", "pf_dora"):
        selected = [row for row in outcomes if row["voice_id"] == voice_id]
        passed = sum(row["rule_isolated"] for row in selected)
        voices[voice_id] = {
            "slot_count": len(selected),
            "isolated_count": passed,
            "coactivated_count": len(selected) - passed,
            "isolation_yield": passed / len(selected),
        }
    coactivation_sets = Counter(
        row["active_changed_rule_ids"]
        for row in outcomes
        if not row["rule_isolated"]
    )
    result = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "classification": (
            "rule_isolation_incomplete_acoustic_fixtures_require_isolated_plans"
            if coactivated
            else "all_rules_isolated"
        ),
        "scope": "all_280_frozen_structural_manifest_slots",
        "matrix_version": matrix.matrix_version,
        "matrix_sha256": matrix.matrix_sha256,
        "source_manifest_sha256": sha256_file(MANIFEST_PATH),
        "source_manifest_record_sha256": manifest_record["record_sha256"],
        "audio_integrity_result_sha256": sha256_file(AUDIO_RESULT_PATH),
        "slot_count": len(outcomes),
        "isolated_slot_count": isolated_count,
        "coactivated_slot_count": coactivated,
        "isolation_yield": isolated_count / len(outcomes),
        "families": families,
        "voices": voices,
        "coactivation_sets": [
            {"active_changed_rule_ids": key, "slot_count": count}
            for key, count in sorted(
                coactivation_sets.items(), key=lambda item: (-item[1], item[0])
            )
        ],
        "interpretation": (
            "The prior 98/98 result remains a universal renderer/splice-integrity "
            "pass. A named matrix cell is not clean per-rule acoustic evidence when "
            "its carrier activates additional changed rules. Family acoustic work "
            "must use isolated plans or explicitly modeled multi-rule trials."
        ),
        "api_calls_made": 0,
        "audio_renders_made": 0,
        "production_enabled": False,
        "outcomes": outcomes,
    }
    result["record_sha256"] = _semantic_hash(result)
    RUN_DIR.mkdir(parents=True, exist_ok=False)
    atomic_write_json(RUN_DIR / "results.json", result)
    print(
        json.dumps(
            {
                "output": str(RUN_DIR / "results.json"),
                "classification": result["classification"],
                "slot_count": len(outcomes),
                "isolated_slot_count": isolated_count,
                "coactivated_slot_count": coactivated,
                "isolation_yield": result["isolation_yield"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
