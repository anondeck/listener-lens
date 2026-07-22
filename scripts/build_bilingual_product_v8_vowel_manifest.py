#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from typing import Any

from earshift_bakeoff.bilingual_listener_engine import load_listener_profiles
from earshift_bakeoff.bilingual_listener_engine_v8 import (
    BILINGUAL_LISTENER_CANDIDATE_VERSION_V8,
    BilingualListenerPlannerV8,
    BilingualListenerRuntimeV8,
    VOWEL_MEASUREMENT_ALIGNMENT_VERSION,
)
from earshift_bakeoff.bilingual_product_isolation import (
    active_changed_rule_ids,
    isolate_listener_profile,
)
from earshift_bakeoff.bilingual_product_matrix import (
    load_bilingual_product_matrix,
    load_bilingual_structural_state,
)
from earshift_bakeoff.bilingual_vowel_engine import _load_pinned_synthesis_voice
from earshift_bakeoff.config import Paths, stable_json
from earshift_bakeoff.controlled_vowel_synthesis_v2 import (
    CONTROLLED_VOWEL_SYNTHESIS_VERSION,
    vowel_stress_context_columns,
)
from earshift_bakeoff.kokoro_gate_bridge import KokoroGateIndex
from earshift_bakeoff.kokoro_synthesis import (
    CONFIG_FILE,
    _filtered_symbols,
    verify_model_files,
)
from earshift_bakeoff.listener_lens import DatabaseNonceChecker
from earshift_bakeoff.portuguese_carrier_planner_v1 import (
    PortuguesePositiveOnlyIndexV1,
)
from earshift_bakeoff.util import atomic_write_json, sha256_file

from run_bilingual_product_audio_integrity_screen_v1 import FixtureAdapter, _tuple


RUN_ID = "20260717-bilingual-product-v8-vowel-manifest"
RUN_DIR = Paths().artifacts / "product-matrix" / RUN_ID
SOURCE_MANIFEST_PATH = (
    Paths().artifacts
    / "product-matrix"
    / "20260717-bilingual-product-isolated-acoustic-manifest-v1"
    / "manifest.json"
)
VOICE_ORDER = ("af_heart", "am_michael", "pm_alex", "pf_dora")


def _semantic_hash(payload: Any) -> str:
    return hashlib.sha256(stable_json(payload).encode("utf-8")).hexdigest()


def main() -> None:
    if RUN_DIR.exists():
        raise RuntimeError(f"refusing to overwrite v8 vowel manifest: {RUN_DIR}")
    matrix = load_bilingual_product_matrix()
    load_bilingual_structural_state(matrix)
    source = json.loads(SOURCE_MANIFEST_PATH.read_text(encoding="utf-8"))
    source_slots = [slot for slot in source["slots"] if slot["family"] == "vowel"]
    if len(source_slots) != 240:
        raise RuntimeError("source vowel manifest no longer contains 240 slots")
    profiles = load_listener_profiles()
    files = verify_model_files(download=False)
    model_vocab = set(
        json.loads(files[CONFIG_FILE].read_text(encoding="utf-8"))["vocab"]
    )
    nonce_checker = DatabaseNonceChecker()
    phone_indexes = (KokoroGateIndex(), PortuguesePositiveOnlyIndexV1())
    outcomes = []
    for voice_id in VOICE_ORDER:
        synthesis = _load_pinned_synthesis_voice(voice_id)
        for slot in (row for row in source_slots if row["voice_id"] == voice_id):
            fixture = slot["fixture_spec"]
            base_profile = profiles[slot["profile_id"]]
            profile = isolate_listener_profile(base_profile, slot["rule_id"])
            planner = BilingualListenerPlannerV8(
                profile={**profile, "voice_id": voice_id},
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
            runtime = BilingualListenerRuntimeV8(
                planner=planner,
                synthesis=synthesis,
            )
            plan = planner.plan(fixture["text"])
            active = active_changed_rule_ids(plan)
            vowel_columns = runtime._vowel_model_columns(plan)
            neutral_symbols = _filtered_symbols(synthesis.model, plan.neutral_phonemes)
            lens_symbols = _filtered_symbols(synthesis.model, plan.lens_phonemes)
            changed_columns = tuple(
                index + 1
                for index, (neutral, lens) in enumerate(
                    zip(neutral_symbols, lens_symbols, strict=True)
                )
                if neutral != lens
            )
            state_columns = vowel_stress_context_columns(
                neutral_symbols,
                lens_symbols,
                vowel_columns,
                changed_columns,
            )
            gate_pass = bool(
                active == (slot["rule_id"],)
                and plan.gates.written_and_espeak_gate_pass
                and plan.gates.supplemental_phone_gates_pass
                and plan.gates.model_representable
                and plan.gates.punctuation_preserved
                and plan.gates.repeated_word_invariant_pass
                and plan.comparison_available
                and vowel_columns
                and state_columns
                and set(vowel_columns).issubset(state_columns)
            )
            outcomes.append(
                {
                    **slot,
                    "source_v7_plan_sha256": slot["isolated_plan_sha256"],
                    "v8_plan_sha256": plan.plan_sha256,
                    "v8_candidate_version": BILINGUAL_LISTENER_CANDIDATE_VERSION_V8,
                    "controlled_vowel_synthesis_version": (
                        CONTROLLED_VOWEL_SYNTHESIS_VERSION
                    ),
                    "measurement_alignment_version": (
                        VOWEL_MEASUREMENT_ALIGNMENT_VERSION
                    ),
                    "v8_active_changed_rule_ids": active,
                    "changed_model_columns": changed_columns,
                    "vowel_unit_columns": vowel_columns,
                    "vowel_state_columns": state_columns,
                    "stress_context_added": tuple(state_columns)
                    != tuple(vowel_columns),
                    "v8_plan_gate_pass": gate_pass,
                    "status": "pass" if gate_pass else "fail",
                    "product_enabled": False,
                }
            )
    passed = sum(row["status"] == "pass" for row in outcomes)
    result = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "classification": (
            "all_v8_vowel_slots_frozen_with_stress_context"
            if passed == len(outcomes)
            else "v8_vowel_manifest_incomplete"
        ),
        "matrix_version": matrix.matrix_version,
        "matrix_sha256": matrix.matrix_sha256,
        "source_manifest_sha256": sha256_file(SOURCE_MANIFEST_PATH),
        "source_manifest_record_sha256": source["record_sha256"],
        "v8_candidate_version": BILINGUAL_LISTENER_CANDIDATE_VERSION_V8,
        "controlled_vowel_synthesis_version": CONTROLLED_VOWEL_SYNTHESIS_VERSION,
        "measurement_alignment_version": VOWEL_MEASUREMENT_ALIGNMENT_VERSION,
        "logical_slot_count": len(outcomes),
        "pass_count": passed,
        "fail_count": len(outcomes) - passed,
        "stress_context_slot_count": sum(
            row["stress_context_added"] for row in outcomes
        ),
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
                "pass_count": passed,
                "fail_count": len(outcomes) - passed,
                "stress_context_slot_count": result["stress_context_slot_count"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
