from __future__ import annotations

import json
import subprocess
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import earshift_bakeoff.ptbr_listener_lens_feasibility as feasibility
from earshift_bakeoff.ptbr_listener_lens_feasibility import (
    classify_local_acoustics,
    generate_review,
    localization_report,
    verify_protocol_committed_at_head,
    verify_repo_bound_inputs_at_head,
)
from earshift_bakeoff.ptbr_listener_lens_feasibility_protocol import (
    CEILINGS_HZ,
    ProfileTargetOccurrence,
    RESPONSE_FILENAME,
    ReciprocalFeasibilityProtocolError,
)


def _measurement(f1_bark: float, f2_bark: float) -> dict[str, float | bool]:
    return {
        "f1_bark": f1_bark,
        "f2_bark": f2_bark,
        "plausibility_pass": True,
    }


def _families(f1_bark: float, f2_bark: float) -> dict[str, dict[str, object]]:
    return {str(ceiling): _measurement(f1_bark, f2_bark) for ceiling in CEILINGS_HZ}


def test_local_gate_uses_same_ceiling_anchors_category_direction_and_magnitude() -> (
    None
):
    result = classify_local_acoustics(
        anchor_neutral=_families(3.0, 12.0),
        anchor_lens=_families(4.0, 13.0),
        controlled_neutral=_families(3.1, 12.1),
        controlled_lens=_families(4.1, 13.1),
    )

    assert result["all_three_ceilings_pass"] is True
    for family in result["families"].values():
        assert family["anchor_valid"] is True
        assert family["neutral_category_pass"] is True
        assert family["lens_category_pass"] is True
        assert family["direction_cosine"] == pytest.approx(1.0)
        assert family["pass"] is True


def test_local_gate_cannot_rescue_reversed_candidate_direction() -> None:
    result = classify_local_acoustics(
        anchor_neutral=_families(3.0, 12.0),
        anchor_lens=_families(4.0, 13.0),
        controlled_neutral=_families(4.1, 13.1),
        controlled_lens=_families(3.1, 12.1),
    )

    assert result["all_three_ceilings_pass"] is False
    assert all(
        family["direction_cosine"] == pytest.approx(-1.0)
        for family in result["families"].values()
    )


def test_localization_rejects_identity_and_passes_target_local_change() -> None:
    neutral = np.zeros(24_000, dtype=np.int16)
    identity = neutral.copy()
    lens = neutral.copy()
    lens[11_000:13_000] = 100
    intervals = [
        {
            "target_interval": {
                "start_sample": 11_500,
                "end_sample_exclusive": 12_500,
            }
        }
    ]

    zero = localization_report(neutral, identity, intervals)
    localized = localization_report(neutral, lens, intervals)

    assert zero["zero_total_difference"] is True
    assert zero["inside_energy_fraction"] == 0
    assert zero["pass"] is False
    assert localized["zero_total_difference"] is False
    assert localized["inside_energy_fraction"] == pytest.approx(1.0)
    assert localized["pass"] is True


def _persist_eligible_review_inputs(root: Path) -> tuple[Path, Path]:
    audio = root / "audio"
    audio.mkdir(parents=True)
    records = []
    for index, slot in enumerate(
        ("controlled-neutral", "controlled-identity", "controlled-lens"),
        start=1,
    ):
        path = audio / f"{slot}.wav"
        feasibility._write_wav(
            path,
            np.linspace(-0.1 * index, 0.1 * index, 240, dtype=np.float32),
        )
        records.append(
            {
                "slot_id": slot,
                "audio": {
                    "relative_path": str(path.relative_to(root)),
                    "wav_sha256": feasibility.sha256_file(path),
                },
            }
        )
    render_path = root / "render-records.json"
    render_path.write_text(
        json.dumps({"protocol_sha256": "p" * 64, "records": records}),
        encoding="utf-8",
    )
    analysis_path = root / "analysis.json"
    analysis_path.write_text(
        json.dumps(
            {
                "protocol_sha256": "p" * 64,
                "classification": (
                    "automatic_acoustic_feasibility_pass__blind_prototype_review_pending"
                ),
                "automatic_acoustic_feasibility_pass": True,
                "render_evidence_verified": True,
                "render_records_sha256": feasibility.sha256_file(render_path),
            }
        ),
        encoding="utf-8",
    )
    return analysis_path, render_path


def test_blind_review_publication_is_opaque_hash_identical_and_private(
    tmp_path: Path,
) -> None:
    analysis_path, render_path = _persist_eligible_review_inputs(tmp_path)

    result = generate_review(
        analysis_path=analysis_path,
        render_records_path=render_path,
        destination=tmp_path,
    )

    public_root = tmp_path / "public" / "review"
    manifest_path = public_root / "review-manifest.json"
    html_path = public_root / "review.html"
    private_path = tmp_path / "private" / "blind-key.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    private = json.loads(private_path.read_text(encoding="utf-8"))
    public_text = manifest_path.read_text(encoding="utf-8") + html_path.read_text(
        encoding="utf-8"
    )
    for leaked in (
        "controlled-neutral",
        "controlled-identity",
        "controlled-lens",
        '"role"',
        "source_relative_path",
    ):
        assert leaked not in public_text
    assert RESPONSE_FILENAME in html_path.read_text(encoding="utf-8")
    assert 'id="download" disabled' in html_path.read_text(encoding="utf-8")
    assert len(manifest["clips"]) == result["opaque_copy_count"] == 3
    mapping_by_id = {item["blind_id"]: item for item in private["mappings"]}
    for clip in manifest["clips"]:
        assert len(clip["blind_id"]) == 24
        assert clip["audio"] == f"audio/{clip['blind_id']}.wav"
        public_audio = public_root / clip["audio"]
        source_audio = (
            tmp_path / mapping_by_id[clip["blind_id"]]["source_relative_path"]
        )
        assert feasibility.sha256_file(public_audio) == feasibility.sha256_file(
            source_audio
        )
    assert len(private["blind_secret"]) == 64


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )


def test_repo_bound_input_gate_rejects_unstaged_and_staged_drift(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.invalid")
    _git(repo, "config", "user.name", "Test")
    protocol_path = repo / "artifacts" / "chain" / "protocol.json"
    evidence_path = repo / "artifacts" / "parent" / "evidence.json"
    protocol_path.parent.mkdir(parents=True)
    evidence_path.parent.mkdir(parents=True)
    protocol_path.write_text('{"protocol":"one"}\n', encoding="utf-8")
    evidence_path.write_text('{"evidence":"one"}\n', encoding="utf-8")

    with pytest.raises(ReciprocalFeasibilityProtocolError, match="tracked"):
        verify_protocol_committed_at_head(protocol_path, repository=repo)

    _git(repo, "add", "artifacts/chain/protocol.json", "artifacts/parent/evidence.json")
    _git(repo, "commit", "-q", "-m", "freeze protocol")
    bindings = {
        "artifacts/chain/protocol.json": feasibility.sha256_file(protocol_path),
        "artifacts/parent/evidence.json": feasibility.sha256_file(evidence_path),
    }
    receipt = verify_repo_bound_inputs_at_head(bindings, repository=repo)
    assert receipt["verified_input_count"] == 2
    assert receipt["all_inputs_tracked_clean_and_byte_identical"] is True

    protocol_path.write_text('{"protocol":"dirty"}\n', encoding="utf-8")
    dirty_bindings = {
        **bindings,
        "artifacts/chain/protocol.json": feasibility.sha256_file(protocol_path),
    }
    with pytest.raises(ReciprocalFeasibilityProtocolError, match="unstaged"):
        verify_repo_bound_inputs_at_head(dirty_bindings, repository=repo)
    protocol_path.write_text('{"protocol":"one"}\n', encoding="utf-8")

    evidence_path.write_text('{"evidence":"staged"}\n', encoding="utf-8")
    _git(repo, "add", "artifacts/parent/evidence.json")
    staged_bindings = {
        **bindings,
        "artifacts/parent/evidence.json": feasibility.sha256_file(evidence_path),
    }
    with pytest.raises(ReciprocalFeasibilityProtocolError, match="staged"):
        verify_repo_bound_inputs_at_head(staged_bindings, repository=repo)


def test_repo_bound_approval_inputs_reject_staged_and_unstaged_drift(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    research = repo / "docs" / "research"
    research.mkdir(parents=True)
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.invalid")
    _git(repo, "config", "user.name", "Test")
    approval_paths = [
        research / "track-d-selection-claims-independent-recheck-v2.json",
        research / "track-d-acoustic-instrument-independent-recheck-v2.json",
        research / "track-d-reciprocal-feasibility-final-approval-resolution-v1.json",
    ]
    for index, path in enumerate(approval_paths):
        path.write_text(json.dumps({"version": index}) + "\n", encoding="utf-8")
    _git(repo, "add", "docs/research")
    _git(repo, "commit", "-q", "-m", "bind approval inputs")

    bindings = {
        path.relative_to(repo).as_posix(): feasibility.sha256_file(path)
        for path in approval_paths
    }
    receipt = verify_repo_bound_inputs_at_head(bindings, repository=repo)
    assert receipt["verified_input_count"] == 3

    selection = approval_paths[0]
    selection.write_text('{"version":"unstaged"}\n', encoding="utf-8")
    unstaged = {
        **bindings,
        selection.relative_to(repo).as_posix(): feasibility.sha256_file(selection),
    }
    with pytest.raises(ReciprocalFeasibilityProtocolError, match="unstaged"):
        verify_repo_bound_inputs_at_head(unstaged, repository=repo)
    selection.write_text(json.dumps({"version": 0}) + "\n", encoding="utf-8")

    acoustic = approval_paths[1]
    acoustic.write_text('{"version":"staged"}\n', encoding="utf-8")
    _git(repo, "add", acoustic.relative_to(repo).as_posix())
    staged = {
        **bindings,
        acoustic.relative_to(repo).as_posix(): feasibility.sha256_file(acoustic),
    }
    with pytest.raises(ReciprocalFeasibilityProtocolError, match="staged"):
        verify_repo_bound_inputs_at_head(staged, repository=repo)


def test_runtime_integrity_failure_short_circuits_measurement_and_localization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    render_payload = {"runtime_integrity": {"pass": False}, "records": []}
    (tmp_path / "render-records.json").write_text(
        json.dumps(render_payload), encoding="utf-8"
    )
    (tmp_path / "audio").mkdir()
    committed = {"repository_head": "h" * 40, "inputs": []}
    (tmp_path / "render-attempt.json").write_text(
        json.dumps({"protocol_sha256": "p" * 64, "committed_inputs": committed}),
        encoding="utf-8",
    )
    monkeypatch.setattr(feasibility, "run_dir", lambda: tmp_path)
    monkeypatch.setattr(
        feasibility,
        "verify_frozen_protocol",
        lambda: {"protocol_sha256": "p" * 64},
    )
    monkeypatch.setattr(
        feasibility,
        "_verify_current_repo_inputs",
        lambda protocol: committed,
    )
    monkeypatch.setattr(
        feasibility,
        "_verify_render_evidence",
        lambda payload, destination, protocol: (
            {},
            {"pass": False, "identity": False},
        ),
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("downstream analysis must not run")

    monkeypatch.setattr(feasibility, "_measure_record", forbidden)
    monkeypatch.setattr(feasibility, "_read_pcm", forbidden)
    monkeypatch.setattr(feasibility, "localization_report", forbidden)

    result = feasibility.analyze()

    assert result["classification"] == "automatic_acoustic_feasibility_failed"
    assert result["measurement_status"] == "unavailable_runtime_integrity_failure"
    assert result["measurements"] == {}
    assert result["localization"] == {
        "status": "skipped_runtime_integrity_failure",
        "pass": False,
    }
    assert result["automatic_acoustic_feasibility_pass"] is False


def test_target_only_interval_is_not_distorted_by_stress_duration() -> None:
    occurrence = ProfileTargetOccurrence(
        occurrence_index=0,
        source_word_index=1,
        source_phone_offset=3,
        profile_character_index=2,
        model_column=3,
        stress_model_column=2,
    )
    plan = SimpleNamespace(
        equal_model_token_count=4,
        target_occurrences=(occurrence,),
    )
    durations = (1, 1, 100, 2, 1, 1)
    intervals = feasibility._alignment_intervals(
        plan, durations=durations, sample_count=sum(durations) * 10
    )
    record = intervals[0]
    assert record["target_interval"]["columns"] == [3]
    assert record["target_interval"]["start_sample"] == 1020
    assert record["target_interval"]["end_sample_exclusive"] == 1040
    assert record["stress_plus_target_descriptive_interval"]["columns"] == [2, 3]
    assert record["stress_plus_target_descriptive_interval"]["start_sample"] == 20


def test_real_profile_target_symbol_and_offset_mapping() -> None:
    _, plan = feasibility._profile_plan()
    occurrence = plan.target_occurrences[0]
    assert plan.neutral_phonemes[occurrence.profile_character_index] == "ɔ"
    assert plan.lens_phonemes[occurrence.profile_character_index] == "ɑ"
    assert occurrence.source_phone_offset == 3
    durations = (1,) * (plan.equal_model_token_count + 2)
    intervals = feasibility._alignment_intervals(
        plan, durations=durations, sample_count=len(durations) * 10
    )
    assert intervals[0]["target_interval"]["columns"] == [occurrence.model_column]
    assert intervals[0]["stress_plus_target_descriptive_interval"]["columns"] == [
        occurrence.stress_model_column,
        occurrence.model_column,
    ]


def test_anchor_floor_collapsed_near_exact_and_single_ceiling() -> None:
    controlled_neutral = _families(0.0, 0.0)

    collapsed = classify_local_acoustics(
        anchor_neutral=_families(0.0, 0.0),
        anchor_lens=_families(0.0, 0.0),
        controlled_neutral=controlled_neutral,
        controlled_lens=_families(0.25, 0.0),
    )
    assert collapsed["all_three_ceilings_pass"] is False
    assert all(not item["anchor_valid"] for item in collapsed["families"].values())

    near = classify_local_acoustics(
        anchor_neutral=_families(0.0, 0.0),
        anchor_lens=_families(0.249, 0.0),
        controlled_neutral=controlled_neutral,
        controlled_lens=_families(0.249, 0.0),
    )
    assert near["all_three_ceilings_pass"] is False

    exact = classify_local_acoustics(
        anchor_neutral=_families(0.0, 0.0),
        anchor_lens=_families(0.25, 0.0),
        controlled_neutral=controlled_neutral,
        controlled_lens=_families(0.25, 0.0),
    )
    assert exact["all_three_ceilings_pass"] is True
    assert all(
        item["anchor_distance_bark"] == 0.25 for item in exact["families"].values()
    )

    one_bad_anchor = _families(0.25, 0.0)
    one_bad_anchor[str(CEILINGS_HZ[1])] = _measurement(0.24, 0.0)
    single = classify_local_acoustics(
        anchor_neutral=_families(0.0, 0.0),
        anchor_lens=one_bad_anchor,
        controlled_neutral=controlled_neutral,
        controlled_lens=_families(0.25, 0.0),
    )
    assert single["all_three_ceilings_pass"] is False
    assert single["families"][str(CEILINGS_HZ[1])]["anchor_valid"] is False


def test_cosine_and_magnitude_exact_boundaries_pass() -> None:
    root_three = np.sqrt(3.0)
    neutral = (0.1875, -root_three / 16)
    lens = (0.3125, root_three / 16)
    result = classify_local_acoustics(
        anchor_neutral=_families(0.0, 0.0),
        anchor_lens=_families(0.5, 0.0),
        controlled_neutral=_families(*neutral),
        controlled_lens=_families(*lens),
    )
    assert result["all_three_ceilings_pass"] is True
    for family in result["families"].values():
        assert family["direction_cosine"] == pytest.approx(0.5)
        assert family["controlled_magnitude_bark"] == pytest.approx(0.25)
        assert family["pass"] is True


def test_localization_boundary_and_multiple_window_union(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    neutral = np.zeros(5, dtype=np.int16)
    lens = np.ones(5, dtype=np.int16)
    monkeypatch.setattr(feasibility, "LOCALIZATION_PADDING_S", 0.0)
    at = localization_report(
        neutral,
        lens,
        [{"target_interval": {"start_sample": 0, "end_sample_exclusive": 4}}],
    )
    below = localization_report(
        neutral,
        lens,
        [{"target_interval": {"start_sample": 0, "end_sample_exclusive": 3}}],
    )
    assert at["inside_energy_fraction"] == pytest.approx(0.8)
    assert at["pass"] is True
    assert below["inside_energy_fraction"] == pytest.approx(0.6)
    assert below["pass"] is False

    monkeypatch.setattr(
        feasibility, "LOCALIZATION_PADDING_S", 1 / feasibility.SAMPLE_RATE_HZ
    )
    union = localization_report(
        neutral,
        lens,
        [
            {"target_interval": {"start_sample": 0, "end_sample_exclusive": 1}},
            {"target_interval": {"start_sample": 4, "end_sample_exclusive": 5}},
        ],
    )
    assert union["windows"] == [
        {"start_sample": 0, "end_sample_exclusive": 2},
        {"start_sample": 3, "end_sample_exclusive": 5},
    ]
    assert union["inside_energy_fraction"] == pytest.approx(0.8)


def test_frame_retention_rejects_invalid_formant_domain_before_bark() -> None:
    valid = [{"time_s": "0.5", "f1_hz": "500", "f2_hz": "1500"} for _ in range(10)]
    invalid = [
        {"time_s": "0.5", "f1_hz": "0", "f2_hz": "1000"},
        {"time_s": "0.5", "f1_hz": "-10", "f2_hz": "1000"},
        {"time_s": "0.5", "f1_hz": "1000", "f2_hz": "900"},
        {"time_s": "0.5", "f1_hz": "--undefined--", "f2_hz": "900"},
    ]
    result = feasibility._measure_rows(
        valid + invalid, {"start_s": 0.0, "end_s": 1.0}, 5500
    )
    assert result["queried_frame_count"] == 14
    assert result["finite_f1_f2_frame_count"] == 13
    assert result["positive_ordered_f1_f2_frame_count"] == 10
    assert result["retained_f1_f2_frame_count"] == 10
    assert result["f1_bark"] == pytest.approx(feasibility._bark(500))

    with pytest.raises(ReciprocalFeasibilityProtocolError, match="retention failed"):
        feasibility._measure_rows(invalid * 2, {"start_s": 0.0, "end_s": 1.0}, 5500)


def test_measure_parses_mocked_praat_tsv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_run(command, **kwargs):
        output = Path(command[4])
        output.write_text(
            "time_s\tf1_hz\tf2_hz\n"
            "0.30\t500\t1500\n"
            "0.40\t510\t1510\n"
            "0.50\t520\t1520\n"
            "0.60\t530\t1530\n"
            "0.70\t540\t1540\n",
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(feasibility.subprocess, "run", fake_run)
    result = feasibility._measure(
        tmp_path / "probe.wav", {"start_s": 0.0, "end_s": 1.0}, 5500
    )
    assert result["queried_frame_count"] == 5
    assert result["retained_f1_f2_frame_count"] == 5
    assert result["f1_hz"] == 520
    assert result["f2_hz"] == 1520


def test_audio_clipping_and_real_wav_integrity(tmp_path: Path) -> None:
    clean = np.linspace(-0.5, 0.5, 1000, dtype=np.float32)
    clean_path = tmp_path / "clean.wav"
    feasibility._write_wav(clean_path, clean)
    pcm = feasibility._read_pcm(clean_path)
    clean_record = feasibility._audio_record(clean, clean_path, tmp_path)
    assert pcm.shape == (1000,)
    assert clean_record["finite"] is True
    assert clean_record["clipping_pass"] is True

    clipped = clean.copy()
    clipped[:2] = 1.0
    clipped_path = tmp_path / "clipped.wav"
    feasibility._write_wav(clipped_path, clipped)
    clipped_record = feasibility._audio_record(clipped, clipped_path, tmp_path)
    assert clipped_record["clipped_fraction"] == pytest.approx(0.002)
    assert clipped_record["clipping_pass"] is False


@pytest.mark.parametrize(
    (
        "runtime",
        "measurement_error",
        "acoustic",
        "localization_error",
        "localized",
        "expected",
    ),
    [
        (False, None, False, None, None, "automatic_acoustic_feasibility_failed"),
        (True, "instrument", False, None, None, "automatic_measurement_inconclusive"),
        (True, None, False, "ignored", None, "automatic_acoustic_feasibility_failed"),
        (True, None, True, "tool", None, "automatic_measurement_inconclusive"),
        (
            True,
            None,
            True,
            None,
            True,
            "automatic_acoustic_feasibility_pass__blind_prototype_review_pending",
        ),
        (True, None, True, None, False, "automatic_acoustic_feasibility_failed"),
    ],
)
def test_automatic_branch_truth_table_is_exact(
    runtime: bool,
    measurement_error: str | None,
    acoustic: bool,
    localization_error: str | None,
    localized: bool | None,
    expected: str,
) -> None:
    assert (
        feasibility._automatic_branch(
            runtime_pass=runtime,
            measurement_error=measurement_error,
            acoustic_pass=acoustic,
            localization_error=localization_error,
            localization_pass=localized,
        )
        == expected
    )


def test_attempt_marker_precedes_decoder_and_second_attempt_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol = {"protocol_sha256": "p" * 64}
    committed = {"repository_head": "h" * 40, "inputs": []}
    monkeypatch.setattr(feasibility, "run_dir", lambda: tmp_path)
    monkeypatch.setattr(feasibility, "verify_frozen_protocol", lambda: protocol)
    monkeypatch.setattr(
        feasibility, "_verify_current_repo_inputs", lambda value: committed
    )

    def stop_before_decoder():
        marker = tmp_path / "render-attempt.json"
        assert marker.is_file()
        persisted = json.loads(marker.read_text(encoding="utf-8"))
        assert persisted["committed_inputs"] == committed
        raise RuntimeError("stop-before-decoder")

    monkeypatch.setattr(feasibility, "_profile_plan", stop_before_decoder)
    with pytest.raises(RuntimeError, match="stop-before-decoder"):
        feasibility.render()
    with pytest.raises(ReciprocalFeasibilityProtocolError, match="stale"):
        feasibility.render()


def test_response_validator_requires_exact_clips_enums_and_session_bindings(
    tmp_path: Path,
) -> None:
    analysis_path, render_path = _persist_eligible_review_inputs(tmp_path)
    generate_review(
        analysis_path=analysis_path,
        render_records_path=render_path,
        destination=tmp_path,
    )
    manifest_path = tmp_path / "public" / "review" / "review-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    response = {
        "schema_version": 1,
        "run_id": feasibility.RUN_ID,
        "review_kind": "blind_prototype_qc_not_perceptual_validation",
        "complete": True,
        "bindings": {
            "protocol_sha256": "p" * 64,
            "analysis_file_sha256": feasibility.sha256_file(analysis_path),
            "public_manifest_file_sha256": feasibility.sha256_file(manifest_path),
        },
        "ratings": {
            clip["blind_id"]: {
                "naturalness": "5",
                "artifact": "none",
                "meaning": "none",
                "notes": "",
            }
            for clip in manifest["clips"]
        },
    }
    assert (
        feasibility.validate_review_response(
            response,
            analysis_path=analysis_path,
            public_manifest_path=manifest_path,
        )
        == response
    )

    malformed = deepcopy(response)
    malformed["ratings"].pop(next(iter(malformed["ratings"])))
    with pytest.raises(ReciprocalFeasibilityProtocolError, match="clip set"):
        feasibility.validate_review_response(
            malformed,
            analysis_path=analysis_path,
            public_manifest_path=manifest_path,
        )
    wrong_enum = deepcopy(response)
    next(iter(wrong_enum["ratings"].values()))["naturalness"] = "6"
    with pytest.raises(ReciprocalFeasibilityProtocolError, match="enum"):
        feasibility.validate_review_response(
            wrong_enum,
            analysis_path=analysis_path,
            public_manifest_path=manifest_path,
        )
    wrong_session = deepcopy(response)
    wrong_session["bindings"]["analysis_file_sha256"] = "0" * 64
    with pytest.raises(ReciprocalFeasibilityProtocolError, match="session"):
        feasibility.validate_review_response(
            wrong_session,
            analysis_path=analysis_path,
            public_manifest_path=manifest_path,
        )
    extra = deepcopy(response)
    extra["ratings"]["extra-clip"] = next(iter(response["ratings"].values()))
    with pytest.raises(ReciprocalFeasibilityProtocolError, match="clip set"):
        feasibility.validate_review_response(
            extra,
            analysis_path=analysis_path,
            public_manifest_path=manifest_path,
        )


def test_review_rejects_ineligible_stale_or_overwrite_states(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ineligible = tmp_path / "ineligible"
    ineligible.mkdir()
    analysis_path, render_path = _persist_eligible_review_inputs(ineligible)
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    analysis["classification"] = "automatic_acoustic_feasibility_failed"
    analysis["automatic_acoustic_feasibility_pass"] = False
    analysis_path.write_text(json.dumps(analysis), encoding="utf-8")
    with pytest.raises(ReciprocalFeasibilityProtocolError, match="persisted verified"):
        generate_review(
            analysis_path=analysis_path,
            render_records_path=render_path,
            destination=ineligible,
        )
    assert not (ineligible / "public").exists()
    assert not (ineligible / "private").exists()

    eligible = tmp_path / "eligible"
    eligible.mkdir()
    eligible_analysis, eligible_render = _persist_eligible_review_inputs(eligible)
    generate_review(
        analysis_path=eligible_analysis,
        render_records_path=eligible_render,
        destination=eligible,
    )
    with pytest.raises(ReciprocalFeasibilityProtocolError, match="overwrite"):
        generate_review(
            analysis_path=eligible_analysis,
            render_records_path=eligible_render,
            destination=eligible,
        )

    failure = tmp_path / "failure"
    failure.mkdir()
    failure_analysis, failure_render = _persist_eligible_review_inputs(failure)
    monkeypatch.setattr(
        feasibility.shutil,
        "copy2",
        lambda *args: (_ for _ in ()).throw(OSError("copy failed")),
    )
    with pytest.raises(OSError, match="copy failed"):
        generate_review(
            analysis_path=failure_analysis,
            render_records_path=failure_render,
            destination=failure,
        )
    assert not (failure / "public").exists()
    assert not (failure / "private").exists()
    marker = json.loads(
        (failure / "review-generation-failure.json").read_text(encoding="utf-8")
    )
    assert marker["review_published"] is False
    assert marker["classification"].endswith("review_generation_failed")


def test_truth_table_failure_and_inconclusive_branches_publish_no_review(
    tmp_path: Path,
) -> None:
    classifications = {
        feasibility._automatic_branch(
            runtime_pass=False,
            measurement_error=None,
            acoustic_pass=False,
            localization_error=None,
            localization_pass=None,
        ),
        feasibility._automatic_branch(
            runtime_pass=True,
            measurement_error="tool",
            acoustic_pass=False,
            localization_error=None,
            localization_pass=None,
        ),
    }
    assert classifications == {
        "automatic_acoustic_feasibility_failed",
        "automatic_measurement_inconclusive",
    }
    assert not (tmp_path / "public").exists()
    assert not (tmp_path / "private").exists()
