from __future__ import annotations

from earshift_bakeoff import kokoro_cluster_confirmation_v4 as v4
from earshift_bakeoff.config import sha256_json


def test_v4_binds_v3_runtime_failure_and_preserves_design() -> None:
    protocol = v4.protocol_record()
    payload = dict(protocol)
    digest = payload.pop("protocol_sha256")
    assert digest == sha256_json(payload)
    assert protocol["parent"]["classification"] == (
        "cluster_shell_v3_runtime_inconclusive"
    )
    assert protocol["parent"]["completed_decoder_slots"] == 0
    assert protocol["path_correction"] == {
        "mechanism": "explicit_versioned_base_directory_for_audio_metadata",
        "temporary_directory_regression_required": True,
        "v3_candidate_evidence_reused": False,
    }
    assert protocol["scope"]["candidate_decoder_slots"] == 9
    assert protocol["scope"]["api_calls"] == 0
    assert protocol["scope"]["production_enabled"] is False


def test_v4_keeps_v3_fixtures_and_family_selection() -> None:
    parent = v4._verified_v3_parent()["protocol"]
    protocol = v4.protocol_record()
    assert protocol["fixtures"] == parent["fixtures"]
    assert protocol["analysis_family_selection"] == parent["analysis_family_selection"]


def test_frozen_v4_result_passes_every_fixture_and_selected_family() -> None:
    result_path = v4.run_dir() / v4.ANALYSIS_FILE
    records_path = v4.run_dir() / v4.RECORDS_FILE
    if not result_path.is_file() or not records_path.is_file():
        return
    result = v4._load_json(result_path)
    records = v4._load_json(records_path)
    assert result["classification"] == (
        "cluster_shell_v4_aggregate_automatic_pass_pending_human_qc"
    )
    assert result["automatic_pass"] is True
    assert result["pending_human_review"] is True
    assert result["api_calls_made"] == 0
    assert result["decoder_attempt_count"] == 9
    assert records["decoder_attempt_count"] == 9
    assert records["status"] == "render_complete"
    assert len(result["fixtures"]) == 3
    assert (
        sum(
            len(row["acoustic"]["windows"]["50"]["occurrences"])
            for row in result["fixtures"]
        )
        == 4
    )
    for fixture in result["fixtures"]:
        assert fixture["automatic_pass"] is True
        assert all(fixture["automatic_checks"].values())
        assert fixture["acoustic"]["window_sensitive"] is False
        assert (
            fixture["spliced_localization"]["inside_difference_energy_fraction"] == 1.0
        )
        assert fixture["localization_runtime_benchmark"]["p95_ms"] < 1.0
        for occurrence in fixture["acoustic"]["windows"]["50"]["occurrences"]:
            assert occurrence["pass"] is True
            assert all(family["pass"] for family in occurrence["families"].values())


def test_v4_blind_layout_is_balanced_and_hides_roles() -> None:
    protocol = v4.protocol_record()
    layout = v4._layout(protocol)
    assert layout == v4._layout(protocol)
    assert len(layout) == 6
    by_fixture: dict[str, set[str]] = {}
    for trial in layout:
        assert "roles" not in trial
        assert set(trial["side_roles"]) == {"A", "B"}
        by_fixture.setdefault(trial["fixture_id"], set()).add(trial["condition"])
    assert all(
        conditions == {"identity-control", "spliced-lens"}
        for conditions in by_fixture.values()
    )
