#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from typing import Any

from earshift_bakeoff.bilingual_listener_engine import (
    BilingualListenerPlanner,
    load_listener_profiles,
)
from earshift_bakeoff.bilingual_product_isolation import (
    ISOLATED_VALIDATION_PROFILE_VERSION,
    active_changed_rule_ids,
    isolate_listener_profile,
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


RUN_ID = "20260717-bilingual-product-isolated-acoustic-manifest-v1"
RUN_DIR = Paths().artifacts / "product-matrix" / RUN_ID
SOURCE_MANIFEST_PATH = (
    Paths().artifacts
    / "product-matrix"
    / "20260717-bilingual-product-matrix-v1"
    / "manifest.json"
)
ISOLATION_AUDIT_PATH = (
    Paths().artifacts
    / "product-matrix"
    / "20260717-bilingual-product-rule-isolation-audit-v1"
    / "results.json"
)


def _semantic_hash(payload: dict[str, Any]) -> str:
    semantic = dict(payload)
    semantic.pop("record_sha256", None)
    return hashlib.sha256(stable_json(semantic).encode("utf-8")).hexdigest()


def main() -> None:
    if RUN_DIR.exists():
        raise RuntimeError(f"refusing to overwrite isolated manifest: {RUN_DIR}")
    matrix = load_bilingual_product_matrix()
    load_bilingual_structural_state(matrix)
    source_manifest = json.loads(SOURCE_MANIFEST_PATH.read_text(encoding="utf-8"))
    audit = json.loads(ISOLATION_AUDIT_PATH.read_text(encoding="utf-8"))
    if audit["classification"] != (
        "rule_isolation_incomplete_acoustic_fixtures_require_isolated_plans"
    ):
        raise RuntimeError("rule-isolation audit no longer requires this manifest")
    profiles = load_listener_profiles()
    files = verify_model_files(download=False)
    model_vocab = set(
        json.loads(files[CONFIG_FILE].read_text(encoding="utf-8"))["vocab"]
    )
    nonce_checker = DatabaseNonceChecker()
    phone_indexes = (KokoroGateIndex(), PortuguesePositiveOnlyIndexV1())
    outcomes: list[dict[str, Any]] = []
    for source_slot in source_manifest["validation_manifest"]["slots"]:
        fixture = source_slot["fixture_spec"]
        base_profile = profiles[source_slot["profile_id"]]
        profile = isolate_listener_profile(base_profile, source_slot["rule_id"])
        planner = BilingualListenerPlanner(
            profile={**profile, "voice_id": source_slot["voice_id"]},
            adapter=FixtureAdapter(
                language_id=base_profile["source_language"],
                source_words=_tuple(fixture["source_words"]),
                source_phones=_tuple(fixture["source_phones"]),
                punctuation=fixture["punctuation"],
            ),
            model_vocab=model_vocab,
            nonce_checker=nonce_checker,
            phone_indexes=phone_indexes,
        )
        plan = planner.plan(fixture["text"])
        active = active_changed_rule_ids(plan)
        status = "pass" if active == (source_slot["rule_id"],) else "fail"
        outcomes.append(
            {
                **source_slot,
                "source_manifest_plan_status": source_slot["status"],
                "isolated_validation_profile_version": (
                    ISOLATED_VALIDATION_PROFILE_VERSION
                ),
                "isolated_plan_sha256": plan.plan_sha256,
                "isolated_active_changed_rule_ids": active,
                "isolated_target_word_indexes": plan.target_word_indexes,
                "isolated_plan_gate_pass": bool(
                    plan.gates.written_and_espeak_gate_pass
                    and plan.gates.supplemental_phone_gates_pass
                    and plan.gates.model_representable
                    and plan.gates.punctuation_preserved
                    and plan.gates.repeated_word_invariant_pass
                    and plan.comparison_available
                ),
                "status": status,
                "product_enabled": False,
            }
        )
    passed = sum(
        row["status"] == "pass" and row["isolated_plan_gate_pass"]
        for row in outcomes
    )
    result = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "classification": (
            "all_acoustic_slots_atomically_isolated"
            if passed == len(outcomes)
            else "isolated_acoustic_manifest_incomplete"
        ),
        "matrix_version": matrix.matrix_version,
        "matrix_sha256": matrix.matrix_sha256,
        "isolated_validation_profile_version": (
            ISOLATED_VALIDATION_PROFILE_VERSION
        ),
        "source_manifest_sha256": sha256_file(SOURCE_MANIFEST_PATH),
        "source_manifest_record_sha256": source_manifest["record_sha256"],
        "source_isolation_audit_sha256": sha256_file(ISOLATION_AUDIT_PATH),
        "source_isolation_audit_record_sha256": audit["record_sha256"],
        "logical_slot_count": len(outcomes),
        "isolated_plan_pass_count": passed,
        "isolated_plan_fail_count": len(outcomes) - passed,
        "isolated_plan_gate_yield": passed / len(outcomes),
        "api_calls_made": 0,
        "audio_renders_made": 0,
        "production_enabled": False,
        "slots": outcomes,
    }
    result["record_sha256"] = _semantic_hash(result)
    RUN_DIR.mkdir(parents=True, exist_ok=False)
    atomic_write_json(RUN_DIR / "manifest.json", result)
    print(
        json.dumps(
            {
                "output": str(RUN_DIR / "manifest.json"),
                "classification": result["classification"],
                "logical_slot_count": len(outcomes),
                "isolated_plan_pass_count": passed,
                "isolated_plan_fail_count": len(outcomes) - passed,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
