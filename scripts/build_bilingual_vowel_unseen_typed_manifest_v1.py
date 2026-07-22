#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from typing import Any

from earshift_bakeoff.bilingual_vowel_unseen_fixtures import (
    UNSEEN_TYPED_FIXTURE_VERSION,
    select_unseen_typed_fixtures,
)
from earshift_bakeoff.config import Paths, stable_json
from earshift_bakeoff.util import atomic_write_json, sha256_file


PROTOCOL_VERSION = "bilingual-vowel-unseen-typed-fixture-selection-v1"
PROTOCOL_PATH = Paths().root / "rules" / f"{PROTOCOL_VERSION}.json"
RUN_ID = "20260718-bilingual-vowel-unseen-typed-manifest-v1"
RUN_DIR = Paths().artifacts / "product-matrix" / RUN_ID
CALIBRATION_RESULT_PATH = (
    Paths().artifacts
    / "product-matrix"
    / "20260717-bilingual-vowel-replicated-anchor-calibration-v1"
    / "results.json"
)
FAILURE_RESULT_PATH = (
    Paths().artifacts
    / "product-matrix"
    / "20260718-bilingual-vowel-replicated-anchor-failure-screen-v1"
    / "results.json"
)
REACHABILITY_RESULT_PATH = (
    Paths().artifacts
    / "product-matrix"
    / "20260717-bilingual-g2p-reachability-v1"
    / "results.json"
)
EXPECTED_CELL_COUNT = 28
EXPECTED_RULE_GROUP_COUNT = 15
EXPECTED_SLOT_COUNT = 84
EXPECTED_OCCURRENCE_COUNT = 112


def _load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _semantic_hash(payload: dict[str, Any]) -> str:
    semantic = dict(payload)
    semantic.pop("record_sha256", None)
    return hashlib.sha256(stable_json(semantic).encode("utf-8")).hexdigest()


def _candidate_cells(calibration: dict, failure: dict) -> list[dict[str, str]]:
    cells: dict[str, dict[str, str]] = {}
    for row in calibration["cell_summaries"]:
        aggregate = row["replicated_anchor"]
        if row["reference_rung"] and aggregate["directional_pass"]:
            cells[row["cell_id"]] = {
                "cell_id": row["cell_id"],
                "profile_id": row["profile_id"],
                "voice_id": row["voice_id"],
                "rule_id": row["rule_id"],
                "source": row["source"],
                "target": row["target"],
                "candidate_rung": row["reference_rung"],
            }
    for row in failure["cell_summaries"]:
        aggregate = row["replicated_anchor"]
        if aggregate["directional_pass"]:
            cells[row["cell_id"]] = {
                "cell_id": row["cell_id"],
                "profile_id": row["profile_id"],
                "voice_id": row["voice_id"],
                "rule_id": row["rule_id"],
                "source": row["source"],
                "target": row["target"],
                "candidate_rung": row["candidate_rung"],
            }
    oral = [
        row
        for row in cells.values()
        if "̃" not in row["source"] and "̃" not in row["target"]
    ]
    return sorted(oral, key=lambda row: row["cell_id"])


def _load_protocol(calibration: dict, failure: dict, reachability: dict) -> dict:
    protocol = _load(PROTOCOL_PATH)
    parents = protocol["parent_bindings"]
    for label, path, result in (
        ("calibration", CALIBRATION_RESULT_PATH, calibration),
        ("failure", FAILURE_RESULT_PATH, failure),
        ("reachability", REACHABILITY_RESULT_PATH, reachability),
    ):
        if (
            parents[f"{label}_result_sha256"] != sha256_file(path)
            or parents[f"{label}_record_sha256"] != result["record_sha256"]
        ):
            raise RuntimeError(f"unseen fixture parent drifted: {label}")
    if (
        protocol["schema_version"] != 1
        or protocol["protocol_version"] != PROTOCOL_VERSION
        or protocol["status"] != "frozen_before_first_unseen_fixture_selection"
        or protocol["production_enabled"] is not False
        or protocol["scope"]["oral_candidate_cell_count"] != EXPECTED_CELL_COUNT
        or protocol["scope"]["rule_group_count"] != EXPECTED_RULE_GROUP_COUNT
        or protocol["scope"]["logical_slot_count"] != EXPECTED_SLOT_COUNT
        or protocol["scope"]["expected_occurrence_count"] != EXPECTED_OCCURRENCE_COUNT
        or protocol["selection"]["fixture_selection_version"]
        != UNSEEN_TYPED_FIXTURE_VERSION
        or protocol["stopping_rule"]["api_calls_allowed"] != 0
        or protocol["stopping_rule"]["audio_renders_allowed"] != 0
    ):
        raise RuntimeError("unseen fixture selection protocol drifted")
    for binding in protocol["source_bindings"]:
        if sha256_file(Paths().root / binding["path"]) != binding["sha256"]:
            raise RuntimeError(f"unseen fixture source drifted: {binding['path']}")
    return protocol


def main() -> int:
    if RUN_DIR.exists():
        raise RuntimeError(f"refusing to overwrite unseen manifest: {RUN_DIR}")
    calibration = _load(CALIBRATION_RESULT_PATH)
    failure = _load(FAILURE_RESULT_PATH)
    reachability = _load(REACHABILITY_RESULT_PATH)
    protocol = _load_protocol(calibration, failure, reachability)
    candidates = _candidate_cells(calibration, failure)
    if len(candidates) != EXPECTED_CELL_COUNT:
        raise RuntimeError("unseen typed oral candidate count drifted")
    selection = select_unseen_typed_fixtures(candidates)
    if (
        selection["cell_count"] != EXPECTED_CELL_COUNT
        or selection["rule_group_count"] != EXPECTED_RULE_GROUP_COUNT
        or selection["logical_slot_count"] != EXPECTED_SLOT_COUNT
        or selection["expected_occurrence_count"] != EXPECTED_OCCURRENCE_COUNT
    ):
        raise RuntimeError("unseen typed selection count drifted")
    result: dict[str, Any] = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "protocol_version": PROTOCOL_VERSION,
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "classification": "unseen_real_g2p_typed_fixtures_frozen_before_audio",
        "production_enabled": False,
        "api_calls_made": 0,
        "audio_renders_made": 0,
        "candidate_cells": candidates,
        "parent_bindings": protocol["parent_bindings"],
        **selection,
    }
    result["record_sha256"] = _semantic_hash(result)
    RUN_DIR.mkdir(parents=True, exist_ok=False)
    atomic_write_json(RUN_DIR / "manifest.json", result)
    print(
        stable_json(
            {
                key: result[key]
                for key in (
                    "classification",
                    "cell_count",
                    "rule_group_count",
                    "logical_slot_count",
                    "expected_occurrence_count",
                    "api_calls_made",
                    "audio_renders_made",
                    "record_sha256",
                )
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
