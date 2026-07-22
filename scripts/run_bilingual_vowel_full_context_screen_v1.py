#!/usr/bin/env python3
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import time
from typing import Any, Sequence

from earshift_bakeoff.bilingual_listener_engine import load_listener_profiles
from earshift_bakeoff.bilingual_product_matrix import load_bilingual_product_matrix
from earshift_bakeoff.bilingual_vowel_acoustics_v2 import VOWEL_ACOUSTIC_VERSION_V2
from earshift_bakeoff.bilingual_vowel_engine import _load_pinned_synthesis_voice
from earshift_bakeoff.bilingual_vowel_full_context import (
    BilingualVowelFullContextRuntime,
    VOWEL_FULL_CONTEXT_CANDIDATE_VERSION,
)
from earshift_bakeoff.config import Paths, stable_json
from earshift_bakeoff.controlled_vowel_full_context import (
    CONTROLLED_VOWEL_FULL_CONTEXT_VERSION,
)
from earshift_bakeoff.kokoro_gate_bridge import KokoroGateIndex
from earshift_bakeoff.kokoro_synthesis import CONFIG_FILE, verify_model_files
from earshift_bakeoff.listener_lens import DatabaseNonceChecker
from earshift_bakeoff.portuguese_carrier_planner_v1 import (
    PortuguesePositiveOnlyIndexV1,
)
from earshift_bakeoff.util import atomic_write_json, sha256_file

import run_bilingual_vowel_word_context_screen_v1 as word_screen
from run_bilingual_product_v8_vowel_acoustic_screen import V8_MANIFEST_PATH


PROTOCOL_VERSION = "bilingual-vowel-full-context-screen-v1"
PROTOCOL_PATH = Paths().root / "rules" / f"{PROTOCOL_VERSION}.json"
RUN_ID = "20260717-bilingual-vowel-full-context-screen-v1"
RUN_DIR = Paths().artifacts / "product-matrix" / RUN_ID
V8_RESULT_PATH = word_screen.V8_RESULT_PATH
WORD_RESULT_PATH = (
    Paths().artifacts
    / "product-matrix"
    / "20260717-bilingual-vowel-word-context-screen-v1"
    / "results.json"
)
VOICE_ORDER = word_screen.VOICE_ORDER


def _semantic_hash(payload: Any) -> str:
    return hashlib.sha256(stable_json(payload).encode("utf-8")).hexdigest()


def _candidate_cell_ids(word_result: dict[str, Any]) -> tuple[str, ...]:
    return tuple(
        sorted(
            row["cell_id"]
            for row in word_result["cell_summaries"]
            if row["candidate_classification"] == "fail"
        )
    )


def _load_protocol(
    *,
    matrix_sha256: str,
    manifest: dict[str, Any],
    v8_result: dict[str, Any],
    word_result: dict[str, Any],
    candidate_cell_ids: tuple[str, ...],
) -> dict[str, Any]:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    expected_keys = {
        "schema_version",
        "protocol_version",
        "status",
        "production_enabled",
        "parent_bindings",
        "selection_basis",
        "scope",
        "candidate_intervention",
        "instrument_and_gates",
        "aggregation_policy",
        "claim_limits",
        "stopping_rule",
        "source_bindings",
    }
    if set(protocol) != expected_keys:
        raise RuntimeError("full-context protocol schema drifted")
    parents = protocol["parent_bindings"]
    if (
        protocol["schema_version"] != 1
        or protocol["protocol_version"] != PROTOCOL_VERSION
        or protocol["status"] != "frozen_before_first_full_context_matrix_render"
        or protocol["production_enabled"] is not False
        or parents["matrix_sha256"] != matrix_sha256
        or parents["v8_manifest_sha256"] != sha256_file(V8_MANIFEST_PATH)
        or parents["v8_manifest_record_sha256"] != manifest["record_sha256"]
        or parents["v8_result_sha256"] != sha256_file(V8_RESULT_PATH)
        or parents["v8_result_record_sha256"] != v8_result["record_sha256"]
        or parents["word_result_sha256"] != sha256_file(WORD_RESULT_PATH)
        or parents["word_result_record_sha256"] != word_result["record_sha256"]
        or tuple(protocol["scope"]["cell_ids_in_order"]) != candidate_cell_ids
        or protocol["scope"]["voice_rule_cell_count"] != 13
        or protocol["scope"]["logical_slot_count"] != 39
        or protocol["scope"]["candidate_render_set_count"] != 39
        or protocol["candidate_intervention"]["api_calls_allowed"] != 0
        or protocol["candidate_intervention"]["replacement_slots_allowed"] is not False
        or protocol["aggregation_policy"]["product_promotion_allowed"] is not False
    ):
        raise RuntimeError("full-context protocol binding drifted")
    for binding in protocol["source_bindings"]:
        if sha256_file(Paths().root / binding["path"]) != binding["sha256"]:
            raise RuntimeError(f"full-context source drifted: {binding['path']}")
    return protocol


def _evaluate_slot(
    *,
    slot: dict[str, Any],
    baseline: dict[str, Any],
    word_parent: dict[str, Any],
    planner: Any,
    synthesis: Any,
) -> dict[str, Any]:
    original_run_dir = word_screen.RUN_DIR
    original_runtime = word_screen.BilingualVowelWordContextRuntime
    word_screen.RUN_DIR = RUN_DIR
    word_screen.BilingualVowelWordContextRuntime = BilingualVowelFullContextRuntime
    try:
        outcome = word_screen._slot_outcome(
            slot=slot,
            baseline=baseline,
            planner=planner,
            synthesis=synthesis,
        )
    finally:
        word_screen.RUN_DIR = original_run_dir
        word_screen.BilingualVowelWordContextRuntime = original_runtime

    audio = dict(outcome["audio"])
    record = dict(audio.pop("word_context_lens"))
    old_path = RUN_DIR / record["relative_path"]
    new_path = old_path.with_name(
        old_path.name.replace("word-context-lens", "full-context-lens")
    )
    old_path.replace(new_path)
    record["relative_path"] = str(new_path.relative_to(RUN_DIR))
    audio["full_context_lens"] = record
    return {
        **outcome,
        "parent_word_context_classification": word_parent["candidate_classification"],
        "candidate_classification": outcome["candidate_classification"],
        "candidate_version": VOWEL_FULL_CONTEXT_CANDIDATE_VERSION,
        "audio": audio,
    }


def _cell_summaries(outcomes: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for outcome in outcomes:
        groups[outcome["cell_id"]].append(outcome)
    summaries = []
    for cell_id, rows in groups.items():
        measured = [row for row in rows if row["status"] == "measured"]
        occurrence_count = sum(len(row["occurrence_outcomes"]) for row in measured)
        complete = bool(len(rows) == 3 and len(measured) == 3 and occurrence_count == 4)
        classification = word_screen._aggregate(measured) if complete else "fail"
        nasal = any("̃" in str(row[key]) for row in rows for key in ("source", "target"))
        eligible = bool(
            complete
            and classification in {"exact_category_pass", "directional_only_pass"}
            and not nasal
        )
        summaries.append(
            {
                "cell_id": cell_id,
                "profile_id": rows[0]["profile_id"],
                "voice_id": rows[0]["voice_id"],
                "rule_id": rows[0]["rule_id"],
                "source": rows[0]["source"],
                "target": rows[0]["target"],
                "slot_count": len(rows),
                "occurrence_count": occurrence_count,
                "complete_three_context_four_occurrence_yield": complete,
                "parent_word_context_classification": "fail",
                "candidate_classification": classification,
                "first_passing_context_mode": (
                    "complete_context_state_neutral_excitation"
                    if classification
                    in {"exact_category_pass", "directional_only_pass"}
                    else None
                ),
                "automatic_human_qc_eligible": eligible,
                "human_qc_status": "pending" if eligible else "not_eligible",
                "product_enabled": False,
            }
        )
    return sorted(summaries, key=lambda row: row["cell_id"])


def main() -> None:
    if RUN_DIR.exists():
        raise RuntimeError(f"refusing to overwrite frozen full-context run: {RUN_DIR}")
    matrix = load_bilingual_product_matrix()
    manifest = json.loads(V8_MANIFEST_PATH.read_text(encoding="utf-8"))
    v8_result = json.loads(V8_RESULT_PATH.read_text(encoding="utf-8"))
    word_result = json.loads(WORD_RESULT_PATH.read_text(encoding="utf-8"))
    candidate_cell_ids = _candidate_cell_ids(word_result)
    protocol = _load_protocol(
        matrix_sha256=matrix.matrix_sha256,
        manifest=manifest,
        v8_result=v8_result,
        word_result=word_result,
        candidate_cell_ids=candidate_cell_ids,
    )
    slots = [
        slot for slot in manifest["slots"] if slot["cell_id"] in candidate_cell_ids
    ]
    if len(slots) != 39:
        raise RuntimeError("full-context candidate manifest is not exactly 39 slots")
    baseline_by_id = {
        outcome["logical_slot_id"]: outcome for outcome in v8_result["outcomes"]
    }
    word_by_id = {
        outcome["logical_slot_id"]: outcome for outcome in word_result["outcomes"]
    }
    profiles = load_listener_profiles()
    files = verify_model_files(download=False)
    model_vocab = set(
        json.loads(files[CONFIG_FILE].read_text(encoding="utf-8"))["vocab"]
    )
    nonce_checker = DatabaseNonceChecker()
    phone_indexes = (KokoroGateIndex(), PortuguesePositiveOnlyIndexV1())
    outcomes = []
    RUN_DIR.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    for voice_id in VOICE_ORDER:
        synthesis = _load_pinned_synthesis_voice(voice_id)
        for slot in (row for row in slots if row["voice_id"] == voice_id):
            try:
                planner = word_screen._planner_v8(
                    slot=slot,
                    profiles=profiles,
                    model_vocab=model_vocab,
                    nonce_checker=nonce_checker,
                    phone_indexes=phone_indexes,
                )
                outcomes.append(
                    _evaluate_slot(
                        slot=slot,
                        baseline=baseline_by_id[slot["logical_slot_id"]],
                        word_parent=word_by_id[slot["logical_slot_id"]],
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
                        "parent_word_context_classification": "fail",
                        "candidate_classification": "fail",
                        "error_code": getattr(exc, "code", type(exc).__name__),
                        "error": str(exc),
                        "api_calls_made": 0,
                        "product_enabled": False,
                    }
                )
    cells = _cell_summaries(outcomes)
    counts = {
        label: sum(row["candidate_classification"] == label for row in cells)
        for label in ("exact_category_pass", "directional_only_pass", "fail")
    }
    result = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "protocol_version": PROTOCOL_VERSION,
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "candidate_version": VOWEL_FULL_CONTEXT_CANDIDATE_VERSION,
        "controlled_version": CONTROLLED_VOWEL_FULL_CONTEXT_VERSION,
        "acoustic_version": VOWEL_ACOUSTIC_VERSION_V2,
        "matrix_sha256": matrix.matrix_sha256,
        "v8_result_sha256": sha256_file(V8_RESULT_PATH),
        "word_result_sha256": sha256_file(WORD_RESULT_PATH),
        "classification": "full_context_candidate_screen_complete_no_product_promotion",
        "candidate_cell_count": len(cells),
        "logical_slot_count": len(outcomes),
        "measured_slot_count": sum(row["status"] == "measured" for row in outcomes),
        "error_slot_count": sum(row["status"] != "measured" for row in outcomes),
        "cell_classification_counts": counts,
        "rescued_cell_count": counts["exact_category_pass"]
        + counts["directional_only_pass"],
        "automatic_human_qc_eligible_cell_count": sum(
            row["automatic_human_qc_eligible"] for row in cells
        ),
        "api_calls_made": 0,
        "replacement_slots_used": 0,
        "elapsed_s": time.perf_counter() - started,
        "production_enabled": False,
        "protocol": protocol,
        "cell_summaries": cells,
        "outcomes": outcomes,
    }
    result["record_sha256"] = _semantic_hash(result)
    atomic_write_json(RUN_DIR / "results.json", result)
    print(
        json.dumps(
            {
                "output": str(RUN_DIR / "results.json"),
                "classification": result["classification"],
                "logical_slot_count": result["logical_slot_count"],
                "measured_slot_count": result["measured_slot_count"],
                "error_slot_count": result["error_slot_count"],
                "cell_classification_counts": counts,
                "rescued_cell_count": result["rescued_cell_count"],
                "automatic_human_qc_eligible_cell_count": result[
                    "automatic_human_qc_eligible_cell_count"
                ],
                "api_calls_made": 0,
                "elapsed_s": result["elapsed_s"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
