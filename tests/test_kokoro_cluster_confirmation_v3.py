from __future__ import annotations

from earshift_bakeoff import kokoro_cluster_confirmation_v3 as v3
from earshift_bakeoff.config import sha256_json


def test_v3_parent_and_protocol_hash_are_bound() -> None:
    protocol = v3.protocol_record()
    payload = dict(protocol)
    digest = payload.pop("protocol_sha256")
    assert digest == sha256_json(payload)
    assert protocol["parent"]["classification"] == (
        "cluster_shell_v2_anchor_calibration_failed"
    )
    assert protocol["scope"] == {
        "api_calls": 0,
        "reused_anchor_decodes": 42,
        "new_anchor_decodes": 0,
        "candidate_decoder_slots": 9,
        "production_enabled": False,
    }


def test_v3_selects_only_stable_coherent_families() -> None:
    protocol = v3.protocol_record()
    selected = protocol["analysis_family_selection"]["selected_by_fixture_position"]
    medial = selected["phrase-medial-cluster-v2"]["0"]
    assert medial["selected_ceilings_hz"] == [5500, 5750]
    assert medial["excluded_ceilings_hz"] == [6000]
    assert min(medial["cross_family_vector_cosines"].values()) >= 0.75
    assert set(medial["anchors"]) == {"5500", "5750"}

    final = selected["phrase-final-cluster-v2"]["0"]
    assert final["selected_ceilings_hz"] == [5500, 5750, 6000]
    assert final["excluded_ceilings_hz"] == []

    repeated_first = selected["repeated-cluster-v2"]["0"]
    assert repeated_first["individually_passing_ceilings_hz"] == [5500, 5750, 6000]
    assert repeated_first["selected_ceilings_hz"] == [5750, 6000]
    assert repeated_first["excluded_ceilings_hz"] == [5500]

    repeated_second = selected["repeated-cluster-v2"]["1"]
    assert repeated_second["selected_ceilings_hz"] == [5500, 5750, 6000]
    assert repeated_second["excluded_ceilings_hz"] == []

    for positions in selected.values():
        for row in positions.values():
            assert min(row["cross_family_vector_cosines"].values()) >= 0.75


def test_v3_fixtures_are_exactly_v2_fixtures() -> None:
    parent = v3._verified_v2_parent()
    assert v3.protocol_record()["fixtures"] == parent["protocol"]["fixtures"]


def test_frozen_v3_result_is_path_plumbing_inconclusive() -> None:
    result_path = v3.run_dir() / v3.ANALYSIS_FILE
    records_path = v3.run_dir() / v3.RECORDS_FILE
    if not result_path.is_file() or not records_path.is_file():
        return
    result = v3._load_json(result_path)
    records = v3._load_json(records_path)
    assert result["classification"] == "cluster_shell_v3_runtime_inconclusive"
    assert "is not in the subpath" in result["failure"]
    assert result["automatic_pass"] is False
    assert result["api_calls_made"] == 0
    assert records["decoder_attempt_count"] == 0
    assert records["fixtures"] == []
    assert records["status"] == "runtime_failure_no_retry"
