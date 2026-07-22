from __future__ import annotations

import json

from earshift_bakeoff.config import ROOT
from earshift_bakeoff.util import sha256_file


PROTOCOL = ROOT / "rules" / "bilingual-candidate-runtime-gate-v1.json"


def test_runtime_gate_protocol_is_frozen_fail_closed_and_zero_api() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))

    assert protocol["status"] == "frozen_before_runtime_gate_execution"
    assert protocol["production_enabled"] is False
    assert protocol["scope"]["prior_unseen_pass_cell_count"] == 18
    assert protocol["scope"]["logical_slot_count"] == 84
    assert protocol["scope"]["target_occurrence_count"] == 112
    assert protocol["rendering"]["natural_decoder_render_count"] == 504
    assert protocol["rendering"]["candidate_audio_rerender_count"] == 0
    assert protocol["stopping_rule"] == {
        "execute_once": True,
        "api_calls_allowed": 0,
        "candidate_audio_rerenders_allowed": 0,
        "replacement_slots_allowed": 0,
        "threshold_changes_allowed": 0,
        "product_promotion_allowed": False,
    }


def test_runtime_gate_protocol_binds_every_parent_and_source() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    parents = protocol["parent_bindings"]

    assert sha256_file(ROOT / parents["unseen_manifest_path"]) == parents[
        "unseen_manifest_sha256"
    ]
    assert sha256_file(ROOT / parents["unseen_result_path"]) == parents[
        "unseen_result_sha256"
    ]
    for binding in protocol["source_bindings"]:
        assert sha256_file(ROOT / binding["path"]) == binding["sha256"]


def test_runtime_gate_cannot_rescue_or_promote_a_cell() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))

    assert "cannot add a new cell" in protocol["scope"]["eligibility"]
    assert protocol["decision_rule"]["human_qc"].startswith(
        "No automatic outcome"
    )
    assert "multi-rule composition" in protocol["claim_limits"]["coverage"]
