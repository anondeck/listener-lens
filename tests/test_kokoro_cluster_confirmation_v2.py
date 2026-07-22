from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from earshift_bakeoff import kokoro_cluster_confirmation_v2 as v2
from earshift_bakeoff.config import sha256_json, stable_json


def _word(neutral: str) -> SimpleNamespace:
    return SimpleNamespace(neutral_phone=neutral)


def test_medial_neighbor_floor() -> None:
    plan = SimpleNamespace(
        target_word_indexes=(2,),
        words=[
            _word("ðˈW"),
            _word("ˈɛvɹi"),
            _word("ʒˈæst"),
            _word("hˈɪdən"),
            _word("wˈɛl"),
        ],
    )
    assert v2._medial_neighbor_reasons(plan) == []
    plan.words[1] = _word("ðə")
    assert v2._medial_neighbor_reasons(plan) == ["medial_neighbor_too_short"]
    plan.words[1] = _word("ˈɛvɹi")
    plan.words[3] = _word("ɪn")
    assert v2._medial_neighbor_reasons(plan) == ["medial_neighbor_too_short"]


def test_anchor_phonemes_swap_targets_only() -> None:
    fixture = {
        "neutral_phonemes": "ðə ʒˈæst ɡˈɑɹdz ðə ʒˈæst.",
        "target_word_indexes": [1, 4],
        "words": [
            {"neutral_phone": "ðə"},
            {"neutral_phone": "ʒˈæst"},
            {"neutral_phone": "ɡˈɑɹdz"},
            {"neutral_phone": "ðə"},
            {"neutral_phone": "ʒˈæst"},
        ],
    }
    ae = v2._anchor_phonemes(fixture, "ʒˈæs")
    eh = v2._anchor_phonemes(fixture, "ʒˈɛs")
    assert ae == "ðə ʒˈæst ɡˈɑɹdz ðə ʒˈæst."
    assert eh == "ðə ʒˈɛst ɡˈɑɹdz ðə ʒˈɛst."
    assert eh.endswith(".")


def test_v1_parent_must_be_frozen_failure() -> None:
    parent = v2._verified_v1_failure()
    assert parent["run_id"] == v2.V1_RUN_ID
    assert parent["classification"] == "cluster_shell_aggregate_automatic_failed"
    assert len(parent["protocol_file_sha256"]) == 64
    assert len(parent["analysis_file_sha256"]) == 64


def test_layout_is_deterministic_and_balanced() -> None:
    layout = v2._layout()
    assert layout == v2._layout()
    assert len(layout) == 6
    by_fixture: dict[str, set[str]] = {}
    for trial in layout:
        by_fixture.setdefault(trial["fixture_id"], set()).add(trial["condition"])
    assert by_fixture == {
        fixture_id: {"identity-control", "spliced-lens"}
        for fixture_id, _, _, _ in v2.FIXTURE_SPECS
    }


@pytest.fixture(scope="module")
def protocol() -> dict:
    return v2.protocol_record()


def test_protocol_reroutes_all_three_mechanisms(protocol) -> None:
    payload = dict(protocol)
    digest = payload.pop("protocol_sha256")
    assert digest == sha256_json(payload)
    assert protocol["cluster_shell_version"] == 4
    assert (
        protocol["parents"]["v1_failed_confirmation"]["classification"]
        == "cluster_shell_aggregate_automatic_failed"
    )
    fixtures = {row["fixture_id"]: row for row in protocol["fixtures"]}
    assert set(fixtures) == {
        "phrase-medial-cluster-v2",
        "phrase-final-cluster-v2",
        "repeated-cluster-v2",
    }
    medial = fixtures["phrase-medial-cluster-v2"]
    assert medial["inventory"] == list(v2.MEDIAL_INVENTORY_V2)
    assert fixtures["phrase-final-cluster-v2"]["inventory"] == list(
        v2.FINAL_INVENTORY_V2
    )
    assert fixtures["repeated-cluster-v2"]["inventory"] == list(
        v2.REPEATED_INVENTORY_V2
    )
    v1_texts = {
        "They kept the task hidden well.",
        "The song was about the past.",
        "The camp guards the camp.",
    }
    for fixture in fixtures.values():
        assert fixture["text"] not in v1_texts
        for extras in fixture["target_extras"]:
            assert set(extras) <= {"t", "p"}
    assert fixtures["repeated-cluster-v2"]["target_occurrence_count"] == 2

    anchors = protocol["same_context_anchors"]
    assert len(anchors["plan"]) == 6
    for row in anchors["plan"]:
        fixture = fixtures[row["fixture_id"]]
        assert row["target_word_indexes"] == fixture["target_word_indexes"]
        for index, word_index in enumerate(row["target_word_indexes"]):
            token = row["phonemes"].split(" ")[word_index].rstrip(".")
            shell = "ʒˈæs" if row["endpoint"] == "ae" else "ʒˈɛs"
            assert token == shell + fixture["target_extras"][index]
    assert anchors["training_seeds"] == list(v2.ANCHOR_TRAINING_SEEDS)
    assert protocol["scope"]["anchor_logical_decodes"] == 42
    assert protocol["scope"]["candidate_decoder_slots"] == 9
    assert protocol["scope"]["api_calls"] == 0


def test_prepare_is_write_once(protocol, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(v2, "run_dir", lambda: tmp_path / "run")
    monkeypatch.setattr(v2, "protocol_record", lambda: protocol)
    first = v2.prepare()
    second = v2.prepare()
    assert stable_json(first) == stable_json(second) == stable_json(protocol)
    stored = json.loads((tmp_path / "run" / "protocol.json").read_text("utf-8"))
    assert stable_json(stored) == stable_json(protocol)
    tampered = dict(stored)
    tampered["status"] = "tampered"
    (tmp_path / "run" / "protocol.json").write_text(
        json.dumps(tampered), encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="protocol differs"):
        v2.prepare()


def test_frozen_v2_result_stops_before_candidate_on_one_analysis_family() -> None:
    result_path = v2.run_dir() / v2.ANALYSIS_FILE
    records_path = v2.run_dir() / v2.RECORDS_FILE
    if not result_path.is_file() or not records_path.is_file():
        pytest.skip("frozen v2 result has not been recorded yet")

    result = json.loads(result_path.read_text(encoding="utf-8"))
    records = json.loads(records_path.read_text(encoding="utf-8"))
    assert result["classification"] == "cluster_shell_v2_anchor_calibration_failed"
    assert result["automatic_pass"] is False
    assert result["same_context_anchor_pass"] is False
    assert result["candidate_decodes_attempted"] is False
    assert result["api_calls_made"] == 0
    assert records["decoder_attempt_count"] == 0
    assert records["fixtures"] == []
    assert records["status"] == "anchor_calibration_failed"

    gates = records["same_context_anchors"]["gates"]
    failed = []
    for fixture_id, positions in gates.items():
        for position in positions:
            for ceiling in v2.CEILINGS_HZ:
                if not position[str(ceiling)]["pass"]:
                    failed.append((fixture_id, position["position"], ceiling))
    assert failed == [("phrase-medial-cluster-v2", 0, 6000)]

    cell = gates["phrase-medial-cluster-v2"][0]["6000"]
    assert cell["ae"]["spread_pass"] is True
    assert cell["eh"]["all_training_seeds_valid"] is True
    assert cell["eh"]["spread_pass"] is False
    assert cell["eh"]["median_pairwise_spread_bark"] == pytest.approx(
        0.3534182735965269
    )


def test_descendant_artifacts_do_not_retroactively_change_v2_protocol() -> None:
    frozen = json.loads((v2.run_dir() / v2.PROTOCOL_FILE).read_text(encoding="utf-8"))
    assert stable_json(v2.protocol_record()) == stable_json(frozen)


def test_audio_metadata_uses_explicit_versioned_base_directory(tmp_path) -> None:
    base = tmp_path / "successor-run"
    path = base / "audio" / "fixture__neutral.wav"
    values = np.asarray([0, 1, -1, 2], dtype="<i2")
    v2._write_wav(path, values)
    record = v2._audio(path, values, base_dir=base)
    assert record["relative_path"] == "audio/fixture__neutral.wav"
    assert record["sample_count"] == 4
