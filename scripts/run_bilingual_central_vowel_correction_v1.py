#!/usr/bin/env python3
from __future__ import annotations

from collections import Counter, defaultdict
import argparse
import hashlib
import json
import time
from typing import Any

from earshift_bakeoff.bilingual_listener_engine import load_listener_profiles
from earshift_bakeoff.bilingual_vowel_engine import _load_pinned_synthesis_voice
from earshift_bakeoff.config import Paths, sha256_json, stable_json
from earshift_bakeoff.kokoro_gate_bridge import KokoroGateIndex
from earshift_bakeoff.kokoro_synthesis import CONFIG_FILE, verify_model_files
from earshift_bakeoff.listener_lens import DatabaseNonceChecker
from earshift_bakeoff.portuguese_carrier_planner_v1 import (
    PortuguesePositiveOnlyIndexV1,
)
from earshift_bakeoff.util import atomic_write_json, sha256_file

import run_bilingual_product_v8_vowel_acoustic_screen as v8_screen
import run_bilingual_vowel_adaptive_strength_screen_v1 as adaptive
import run_bilingual_vowel_word_context_screen_v1 as word_screen


VERSION = "bilingual-central-vowel-correction-v1"
RUN_ID = "20260718-bilingual-central-vowel-correction-v1"
PROTOCOL_PATH = Paths().root / "rules" / f"{VERSION}.json"
RUN_DIR = Paths().artifacts / "product-matrix" / RUN_ID
RESULT_PATH = RUN_DIR / "results.json"
V8_MANIFEST_PATH = v8_screen.V8_MANIFEST_PATH
V8_RESULT_PATH = adaptive.V8_RESULT_PATH
STRENGTHS = (1.0, 0.75, 1.25, 0.5, 1.5, 2.0, 2.5, 3.0)
RULE_IDS = (
    "enpt.reduced_schwa_a",
    "enpt.schwa_reduced_a",
    "pten.final_a_schwa",
)
VOICE_ORDER = ("af_heart", "am_michael", "pm_alex", "pf_dora")


def _semantic_hash(payload: Any) -> str:
    return hashlib.sha256(stable_json(payload).encode("utf-8")).hexdigest()


def _selected_slots(manifest: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    return tuple(slot for slot in manifest["slots"] if slot["rule_id"] in RULE_IDS)


def _cell_ids(slots: tuple[dict[str, Any], ...]) -> tuple[str, ...]:
    return tuple(sorted({slot["cell_id"] for slot in slots}))


def _source_bindings() -> dict[str, str]:
    paths = (
        "scripts/run_bilingual_central_vowel_correction_v1.py",
        "scripts/run_bilingual_vowel_adaptive_strength_screen_v1.py",
        "src/earshift_bakeoff/bilingual_vowel_state_strength.py",
        "src/earshift_bakeoff/controlled_vowel_state_strength.py",
        "src/earshift_bakeoff/kokoro_output_domain_splice.py",
    )
    return {path: sha256_file(Paths().root / path) for path in paths}


def protocol_record() -> dict[str, Any]:
    manifest = json.loads(V8_MANIFEST_PATH.read_text(encoding="utf-8"))
    v8_result = json.loads(V8_RESULT_PATH.read_text(encoding="utf-8"))
    slots = _selected_slots(manifest)
    payload = {
        "schema_version": 1,
        "version": VERSION,
        "run_id": RUN_ID,
        "status": "frozen_before_local_synthesis",
        "purpose": (
            "Evaluate one shared complete-context latent-delta strength mechanism "
            "for English central-vowel reduction and Portuguese final-a reduction "
            "across all four product voices."
        ),
        "parents": {
            "v8_manifest_path": str(V8_MANIFEST_PATH.relative_to(Paths().root)),
            "v8_manifest_sha256": sha256_file(V8_MANIFEST_PATH),
            "v8_manifest_record_sha256": manifest["record_sha256"],
            "v8_result_path": str(V8_RESULT_PATH.relative_to(Paths().root)),
            "v8_result_sha256": sha256_file(V8_RESULT_PATH),
            "v8_result_record_sha256": v8_result["record_sha256"],
        },
        "scope": {
            "rule_ids": list(RULE_IDS),
            "cell_ids": list(_cell_ids(slots)),
            "cell_count": 6,
            "logical_slot_ids": [slot["logical_slot_id"] for slot in slots],
            "logical_slot_count": 18,
            "contexts_per_cell": 3,
            "expected_occurrence_count_per_cell": 4,
            "voice_order": list(VOICE_ORDER),
        },
        "intervention": {
            "candidate": "complete-context-state-delta-with-output-domain-splice",
            "strength_order": list(STRENGTHS),
            "selection": (
                "For each occurrence choose the first exact-category pass in "
                "frozen order, otherwise the first directional-only pass; never "
                "select using listening or effect magnitude."
            ),
            "candidate_render_set_count": 18 * len(STRENGTHS),
            "replacement_slots": 0,
        },
        "gates": {
            "inheritance": (
                "Reuse the frozen v8 plan, neutral PCM, source/target anchors, "
                "three-ceiling analysis family, classification formula, boundary "
                "metrics, localization gate, and complete three-context/four-"
                "occurrence aggregation unchanged."
            ),
            "regression_control": (
                "Michael enpt.reduced_schwa_a is included despite its later unseen "
                "pass; this run may not weaken or erase that evidence."
            ),
        },
        "outcomes": {
            "mechanism_pass": (
                "All six cells complete with at least directional evidence and the "
                "Michael control remains exact-category. Eligible for a separately "
                "frozen unseen confirmation, not production promotion."
            ),
            "mixed": "Preserve per-cell outcomes and redesign only failed mechanism classes.",
            "fail": "No production promotion and no threshold change.",
        },
        "stopping_rule": (
            "Run every frozen slot and strength exactly once locally. Do not replace "
            "fixtures, rerender a valid result, alter thresholds, or promote a cell."
        ),
        "scope_controls": {
            "api_calls": 0,
            "paid_calls": 0,
            "production_enabled": False,
            "deployment": False,
        },
        "source_bindings": _source_bindings(),
    }
    return {**payload, "protocol_sha256": sha256_json(payload)}


def prepare() -> dict[str, Any]:
    protocol = protocol_record()
    if PROTOCOL_PATH.exists():
        existing = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        if stable_json(existing) != stable_json(protocol):
            raise RuntimeError("central-vowel protocol differs from frozen record")
        return existing
    if RUN_DIR.exists():
        raise RuntimeError("central-vowel run exists before its protocol")
    atomic_write_json(PROTOCOL_PATH, protocol)
    return protocol


def _cell_summaries(outcomes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for outcome in outcomes:
        groups[outcome["cell_id"]].append(outcome)
    rows = []
    for cell_id, slots in sorted(groups.items()):
        measured = [slot for slot in slots if slot["status"] == "measured"]
        complete = bool(
            len(slots) == 3
            and len(measured) == 3
            and all(slot["selection_complete"] for slot in measured)
            and sum(len(slot["occurrence_outcomes"]) for slot in measured) == 4
        )
        classification = adaptive._aggregate(measured) if complete else "fail"
        rows.append(
            {
                "cell_id": cell_id,
                "profile_id": slots[0]["profile_id"],
                "voice_id": slots[0]["voice_id"],
                "rule_id": slots[0]["rule_id"],
                "source": slots[0]["source"],
                "target": slots[0]["target"],
                "complete_three_context_four_occurrence_yield": complete,
                "classification": classification,
                "exact_category_pass": classification == "exact_category_pass",
                "directional_pass": classification
                in {"exact_category_pass", "directional_only_pass"},
                "selected_strength_counts": dict(
                    Counter(
                        selection["label"]
                        for slot in measured
                        for selection in slot["occurrence_selections"]
                        if selection is not None
                    )
                ),
                "production_enabled": False,
            }
        )
    return rows


def run() -> dict[str, Any]:
    if RESULT_PATH.exists():
        return json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if stable_json(protocol) != stable_json(protocol_record()):
        raise RuntimeError("central-vowel protocol or bound sources drifted")
    manifest = json.loads(V8_MANIFEST_PATH.read_text(encoding="utf-8"))
    v8_result = json.loads(V8_RESULT_PATH.read_text(encoding="utf-8"))
    slots = _selected_slots(manifest)
    if len(slots) != 18 or len(_cell_ids(slots)) != 6:
        raise RuntimeError("central-vowel manifest denominator drifted")
    baseline_by_id = {
        outcome["logical_slot_id"]: outcome for outcome in v8_result["outcomes"]
    }
    profiles = load_listener_profiles()
    files = verify_model_files(download=False)
    model_vocab = set(
        json.loads(files[CONFIG_FILE].read_text(encoding="utf-8"))["vocab"]
    )
    nonce_checker = DatabaseNonceChecker()
    phone_indexes = (KokoroGateIndex(), PortuguesePositiveOnlyIndexV1())
    outcomes: list[dict[str, Any]] = []
    RUN_DIR.mkdir(parents=True, exist_ok=False)
    original_run_dir = adaptive.RUN_DIR
    original_strengths = adaptive.STRENGTHS
    adaptive.RUN_DIR = RUN_DIR
    adaptive.STRENGTHS = STRENGTHS
    started = time.perf_counter()
    try:
        for voice_id in VOICE_ORDER:
            synthesis = _load_pinned_synthesis_voice(voice_id)
            for slot in (item for item in slots if item["voice_id"] == voice_id):
                try:
                    planner = word_screen._planner_v8(
                        slot=slot,
                        profiles=profiles,
                        model_vocab=model_vocab,
                        nonce_checker=nonce_checker,
                        phone_indexes=phone_indexes,
                    )
                    outcomes.append(
                        adaptive._adaptive_slot(
                            slot=slot,
                            baseline=baseline_by_id[slot["logical_slot_id"]],
                            planner=planner,
                            synthesis=synthesis,
                        )
                    )
                except Exception as exc:
                    outcomes.append(
                        {
                            "logical_slot_id": slot["logical_slot_id"],
                            "cell_id": slot["cell_id"],
                            "profile_id": slot["profile_id"],
                            "voice_id": slot["voice_id"],
                            "rule_id": slot["rule_id"],
                            "context": slot["context"],
                            "source": slot["source"],
                            "target": slot["target"],
                            "status": "render_or_measurement_error",
                            "candidate_classification": "fail",
                            "error_code": getattr(exc, "code", type(exc).__name__),
                            "error": str(exc),
                            "api_calls_made": 0,
                            "production_enabled": False,
                        }
                    )
    finally:
        adaptive.RUN_DIR = original_run_dir
        adaptive.STRENGTHS = original_strengths
    cells = _cell_summaries(outcomes)
    counts = Counter(row["classification"] for row in cells)
    all_directional = bool(cells and all(row["directional_pass"] for row in cells))
    michael_control = next(
        row
        for row in cells
        if row["voice_id"] == "am_michael" and row["rule_id"] == "enpt.reduced_schwa_a"
    )
    classification = (
        "central_vowel_mechanism_pass_pending_unseen_confirmation"
        if all_directional and michael_control["exact_category_pass"]
        else "central_vowel_mechanism_mixed_no_promotion"
    )
    payload = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "version": VERSION,
        "protocol_sha256": protocol["protocol_sha256"],
        "classification": classification,
        "cell_classification_counts": dict(sorted(counts.items())),
        "cell_count": len(cells),
        "logical_slot_count": len(outcomes),
        "measured_slot_count": sum(row["status"] == "measured" for row in outcomes),
        "error_slot_count": sum(row["status"] != "measured" for row in outcomes),
        "candidate_render_set_count": len(outcomes) * len(STRENGTHS),
        "api_calls_made": 0,
        "production_enabled": False,
        "elapsed_s": time.perf_counter() - started,
        "cell_summaries": cells,
        "outcomes": outcomes,
    }
    result = {**payload, "record_sha256": _semantic_hash(payload)}
    atomic_write_json(RESULT_PATH, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "run"))
    args = parser.parse_args()
    result = prepare() if args.command == "prepare" else run()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
