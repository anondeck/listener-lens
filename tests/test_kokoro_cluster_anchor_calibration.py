from __future__ import annotations

import json

import pytest

from earshift_bakeoff import kokoro_cluster_anchor_calibration as calibration
from earshift_bakeoff.config import sha256_json
from earshift_bakeoff.util import sha256_file


def test_conditions_swap_only_the_target_word() -> None:
    conditions = calibration.anchor_conditions()
    assert len(conditions) == 2 * len(calibration.LEGACY_CLUSTER_EXTRA_CONSONANTS)
    assert [row["condition_id"] for row in conditions] == [
        "ae-t",
        "ae-k",
        "ae-p",
        "eh-t",
        "eh-k",
        "eh-p",
    ]
    frame = calibration.EXPECTED_FRAME_NEUTRAL_PHONEMES.split(" ")
    for row in conditions:
        words = row["phonemes"].split(" ")
        assert len(words) == len(frame)
        for index, word in enumerate(words):
            if index == calibration.EXPECTED_TARGET_WORD_INDEX:
                assert word == row["target_phone"]
            elif index == len(words) - 1:
                assert word == frame[index]
            else:
                assert word == frame[index]
    real = {row["target_phone"] for row in conditions if row["is_real_english_word"]}
    assert real == set(calibration.REAL_WORD_ANCHORS)


def test_protocol_is_hash_bound_and_scoped() -> None:
    protocol = calibration.protocol_record()
    payload = dict(protocol)
    digest = payload.pop("protocol_sha256")
    assert digest == sha256_json(payload)
    assert protocol["scope"]["api_calls"] == 0
    assert protocol["scope"]["production_enabled"] is False
    assert protocol["scope"]["logical_decodes"] == 30
    assert protocol["parent"]["run_id"] == calibration.PARENT_RUN_ID
    assert len(protocol["parent"]["fixture_plan_sha256"]) == 64
    assert protocol["gates"]["endpoint_definition"].startswith("mean")
    assert protocol["seeds"]["baseline_double_decode_bit_identity_required"] is True


def test_legacy_protocol_builders_reproduce_frozen_artifacts() -> None:
    v1 = json.loads((calibration.run_dir() / "protocol.json").read_text("utf-8"))
    v2 = json.loads((calibration.run_dir_v2() / "protocol.json").read_text("utf-8"))
    assert calibration.protocol_record() == v1
    assert calibration.protocol_record_v2() == v2
    assert v1["cluster_shell_version"] == 2
    assert v2["cluster_shell_version"] == 2
    assert {row["target_phone"] for row in v2["conditions"]} == {
        "vˈæst",
        "vˈæsk",
        "vˈæsp",
        "vˈɛst",
        "vˈɛsk",
        "vˈɛsp",
    }


def test_v3_protocol_is_new_shell_and_binds_frozen_v2() -> None:
    protocol = calibration.protocol_record_v3()
    payload = dict(protocol)
    digest = payload.pop("protocol_sha256")
    assert digest == sha256_json(payload)
    assert protocol["cluster_shell_version"] == calibration.V3_CLUSTER_SHELL_VERSION == 3
    assert protocol["v2_parent"]["protocol_file_sha256"] == sha256_file(
        calibration.run_dir_v2() / "protocol.json"
    )
    assert protocol["v2_parent"]["analysis_file_sha256"] == sha256_file(
        calibration.run_dir_v2() / "analysis.json"
    )
    assert protocol["isolated_gate_probe"]["all_forms_pass"] is True
    assert all(
        not row["rejection_reasons"] for row in protocol["isolated_gate_probe"]["rows"]
    )
    assert {row["target_phone"] for row in protocol["conditions"]} == {
        "ʒˈæst",
        "ʒˈæsk",
        "ʒˈæsp",
        "ʒˈɛst",
        "ʒˈɛsk",
        "ʒˈɛsp",
    }


def test_v3_protocol_builder_reproduces_frozen_artifact() -> None:
    frozen = json.loads(
        (calibration.run_dir_v3() / "protocol.json").read_text("utf-8")
    )
    assert calibration.protocol_record_v3() == frozen
    assert frozen["cluster_shell_version"] == 3


def test_v4_protocol_narrows_pool_and_binds_frozen_v3() -> None:
    from earshift_bakeoff.kokoro_cluster_shell import CLUSTER_SHELL_VERSION

    protocol = calibration.protocol_record_v4()
    payload = dict(protocol)
    digest = payload.pop("protocol_sha256")
    assert digest == sha256_json(payload)
    assert protocol["cluster_shell_version"] == CLUSTER_SHELL_VERSION == 4
    parent = protocol["v3_parent"]
    assert parent["classification"] == "cluster_anchor_calibration_v3_fail"
    assert parent["protocol_file_sha256"] == sha256_file(
        calibration.run_dir_v3() / "protocol.json"
    )
    assert parent["analysis_file_sha256"] == sha256_file(
        calibration.run_dir_v3() / "analysis.json"
    )
    assert [row["condition_id"] for row in protocol["conditions"]] == [
        "ae-t",
        "ae-p",
        "eh-t",
        "eh-p",
    ]
    assert {row["target_phone"] for row in protocol["conditions"]} == {
        "ʒˈæst",
        "ʒˈæsp",
        "ʒˈɛst",
        "ʒˈɛsp",
    }
    assert not any(row["is_real_english_word"] for row in protocol["conditions"])
    assert protocol["pool_change"]["from_extras"] == ["t", "k", "p"]
    assert protocol["pool_change"]["to_extras"] == ["t", "p"]
    assert protocol["isolated_gate_probe"]["all_forms_pass"] is True
    assert all(
        not row["rejection_reasons"]
        for row in protocol["isolated_gate_probe"]["rows"]
    )
    assert protocol["seeds"]["training"] == list(calibration.V2_TRAINING_SEEDS)
    assert protocol["scope"]["api_calls"] == 0
    assert protocol["scope"]["logical_decodes"] == 28
    assert protocol["scope"]["production_enabled"] is False
    assert protocol["gates"]["seed_spread_statistic"] == "median_pairwise_distance"


def test_prepare_v4_is_write_once(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(calibration, "run_dir_v4", lambda: tmp_path / "run")
    first = calibration.prepare_v4()
    second = calibration.prepare_v4()
    assert first == second
    stored = json.loads((tmp_path / "run" / "protocol.json").read_text("utf-8"))
    assert stored == first
    tampered = dict(stored)
    tampered["status"] = "tampered"
    (tmp_path / "run" / "protocol.json").write_text(
        json.dumps(tampered), encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="differs from freeze"):
        calibration.prepare_v4()


def test_prepare_is_write_once(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(calibration, "run_dir", lambda: tmp_path / "run")
    first = calibration.prepare()
    second = calibration.prepare()
    assert first == second
    stored = json.loads((tmp_path / "run" / "protocol.json").read_text("utf-8"))
    assert stored == first
    tampered = dict(stored)
    tampered["status"] = "tampered"
    (tmp_path / "run" / "protocol.json").write_text(
        json.dumps(tampered), encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="differs from freeze"):
        calibration.prepare()
