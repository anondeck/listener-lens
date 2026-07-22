from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from earshift_bakeoff import kokoro_cluster_confirmation as confirmation
from earshift_bakeoff.config import sha256_json


def _word(neutral: str, lens: str) -> SimpleNamespace:
    return SimpleNamespace(neutral_phone=neutral, lens_phone=lens)


def _plan(
    *,
    words: list[SimpleNamespace],
    indexes: tuple[int, ...],
    occurrences: int,
) -> SimpleNamespace:
    return SimpleNamespace(
        words=words,
        target_word_indexes=indexes,
        target_occurrence_count=occurrences,
    )


def test_cluster_target_reasons_accept_and_reject() -> None:
    good = _plan(
        words=[_word("hˈɪd", "hˈɪd"), _word("ʒˈæst", "ʒˈɛst")],
        indexes=(1,),
        occurrences=1,
    )
    assert confirmation._cluster_target_reasons(good, 1) == []
    wrong_shell = _plan(
        words=[_word("vˈæst", "vˈɛst")], indexes=(0,), occurrences=1
    )
    reasons = confirmation._cluster_target_reasons(wrong_shell, 1)
    assert "neutral_not_cluster_shell" in reasons
    assert "lens_not_cluster_shell" in reasons
    bad_extra = _plan(
        words=[_word("ʒˈæsk", "ʒˈɛsk")], indexes=(0,), occurrences=1
    )
    assert "extras_outside_preserving_pool" in confirmation._cluster_target_reasons(
        bad_extra, 1
    )
    differing = _plan(
        words=[_word("ʒˈæst", "ʒˈɛsp")], indexes=(0,), occurrences=1
    )
    assert "lens_extras_differ" in confirmation._cluster_target_reasons(differing, 1)
    wrong_count = confirmation._cluster_target_reasons(good, 2)
    assert "wrong_target_occurrence_count" in wrong_count
    assert "wrong_target_word_count" in wrong_count


def test_role_reasons_cover_all_three_roles() -> None:
    words = [_word("a", "a"), _word("ʒˈæst", "ʒˈɛst"), _word("b", "b")]
    medial = _plan(words=words, indexes=(1,), occurrences=1)
    assert confirmation._role_reasons(medial, "medial") == []
    assert confirmation._role_reasons(medial, "final") == [
        "target_not_phrase_final"
    ]
    final = _plan(words=words, indexes=(2,), occurrences=1)
    assert confirmation._role_reasons(final, "final") == []
    assert confirmation._role_reasons(final, "medial") == [
        "target_not_phrase_medial"
    ]
    repeated = _plan(
        words=[
            _word("a", "a"),
            _word("ʒˈæst", "ʒˈɛst"),
            _word("b", "b"),
            _word("ʒˈæst", "ʒˈɛst"),
        ],
        indexes=(1, 3),
        occurrences=2,
    )
    assert confirmation._role_reasons(repeated, "repeated") == []
    drifted = _plan(
        words=[
            _word("ʒˈæst", "ʒˈɛst"),
            _word("ʒˈæsp", "ʒˈɛsp"),
        ],
        indexes=(0, 1),
        occurrences=2,
    )
    assert confirmation._role_reasons(drifted, "repeated") == [
        "repeated_carriers_differ"
    ]


def test_calibration_parent_requires_v4_pass(monkeypatch) -> None:
    parent = confirmation._verified_calibration_parent()
    assert parent["run_id"] == confirmation.CALIBRATION_RUN_ID
    assert parent["classification"] == "cluster_anchor_calibration_v4_pass"
    assert set(parent["endpoints_by_extra"]) == {"t", "p"}
    for rows in parent["endpoints_by_extra"].values():
        assert set(rows) == {"5500", "5750", "6000"}
        for endpoint in rows.values():
            assert len(endpoint["ae_bark"]) == 2
            assert len(endpoint["eh_bark"]) == 2

    from earshift_bakeoff.kokoro_cluster_anchor_calibration import run_dir_v3

    monkeypatch.setattr(confirmation, "calibration_dir", run_dir_v3)
    with pytest.raises(RuntimeError, match="requires the v4 calibration pass"):
        confirmation._verified_calibration_parent()


def test_strict_parent_is_bound_automatic_pass() -> None:
    parent = confirmation._verified_strict_parent()
    assert parent["run_id"] == confirmation.STRICT_PARENT_RUN_ID
    assert (
        parent["classification"]
        == "strict_shell_aggregate_automatic_pass_pending_human_qc"
    )
    assert len(parent["analysis_sha256"]) == 64


def test_layout_is_deterministic_and_balanced() -> None:
    layout = confirmation._layout()
    assert layout == confirmation._layout()
    assert len(layout) == 6
    assert [row["trial_id"] for row in layout] == [
        f"comparison-{index:02d}" for index in range(1, 7)
    ]
    by_fixture: dict[str, set[str]] = {}
    for trial in layout:
        by_fixture.setdefault(trial["fixture_id"], set()).add(trial["condition"])
        assert set(trial["side_roles"]) == {"A", "B"}
    assert by_fixture == {
        fixture_id: {"identity-control", "spliced-lens"}
        for fixture_id, _, _, _ in confirmation.FIXTURE_SPECS
    }


@pytest.fixture(scope="module")
def protocol() -> dict:
    return confirmation.protocol_record()


def test_protocol_selects_all_three_fixtures(protocol) -> None:
    payload = dict(protocol)
    digest = payload.pop("protocol_sha256")
    assert digest == sha256_json(payload)
    assert protocol["cluster_shell_version"] == 4
    assert protocol["parents"]["calibration_v4"]["classification"] == (
        "cluster_anchor_calibration_v4_pass"
    )
    assert set(protocol["endpoints_by_extra"]) == {"t", "p"}
    fixtures = protocol["fixtures"]
    assert [row["fixture_id"] for row in fixtures] == [
        "phrase-medial-cluster",
        "phrase-final-cluster",
        "repeated-cluster",
    ]
    for fixture in fixtures:
        for extras in fixture["target_extras"]:
            assert extras
            assert set(extras) <= {"t", "p"}
        for index in fixture["target_word_indexes"]:
            word = fixture["words"][index]
            assert word["neutral_phone"].startswith("ʒˈæs")
            assert word["lens_phone"].startswith("ʒˈɛs")
    assert fixtures[0]["target_occurrence_count"] == 1
    assert fixtures[1]["target_occurrence_count"] == 1
    assert fixtures[2]["target_occurrence_count"] == 2
    assert protocol["scope"] == {
        "api_calls": 0,
        "logical_render_pairs": 4,
        "production_enabled": False,
    }
    assert protocol["gates"]["localization_minimum"] == 0.80
    assert protocol["measurement"]["primary_window_percent"] == 50


def test_prepare_is_write_once(protocol, tmp_path, monkeypatch) -> None:
    from earshift_bakeoff.config import stable_json

    monkeypatch.setattr(confirmation, "run_dir", lambda: tmp_path / "run")
    monkeypatch.setattr(confirmation, "protocol_record", lambda: protocol)
    first = confirmation.prepare()
    second = confirmation.prepare()
    assert stable_json(first) == stable_json(second) == stable_json(protocol)
    stored = json.loads(
        (tmp_path / "run" / "protocol.json").read_text("utf-8")
    )
    assert stable_json(stored) == stable_json(protocol)
    tampered = dict(stored)
    tampered["status"] = "tampered"
    (tmp_path / "run" / "protocol.json").write_text(
        json.dumps(tampered), encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="protocol differs"):
        confirmation.prepare()
