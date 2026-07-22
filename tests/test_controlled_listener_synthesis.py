from __future__ import annotations

import numpy as np
import pytest
import torch
from types import SimpleNamespace

from earshift_bakeoff.controlled_listener_synthesis import (
    FINAL_F0_RATIO,
    RISE_FALL_PEAK_RATIO,
    StressDurationIntervention,
    _apply_stress_intensity,
    _apply_stress_duration_transfers,
    _copy_column_frames,
    _consonant_state_columns,
    _expanded_segment_context_columns,
    _force_voiced_insertion_f0,
    _set_fixed_durations,
    apply_final_f0_operation,
)


def _curve() -> torch.Tensor:
    values = np.full(60, -1.0, dtype=np.float32)
    values[8:52] = np.linspace(150.0, 180.0, 44, dtype=np.float32)
    return torch.from_numpy(values).unsqueeze(0)


def test_identity_operation_is_exact_and_hash_bound() -> None:
    source = _curve()

    actual, report = apply_final_f0_operation(source, "identity")

    assert torch.equal(actual, source)
    assert actual.data_ptr() != source.data_ptr()
    assert report.eligible is True
    assert report.operation == "identity"
    assert len(report.curve_sha256) == 64


def test_canonical_bp_question_has_frozen_rise_fall_shape() -> None:
    source = _curve()

    actual, report = apply_final_f0_operation(source, "canonical_bp_rise_fall")

    assert report.eligible is True
    assert report.start_hz is not None
    assert report.peak_hz == pytest.approx(
        report.start_hz * RISE_FALL_PEAK_RATIO, abs=1.0
    )
    assert report.end_hz == pytest.approx(report.start_hz * FINAL_F0_RATIO, abs=1.0)
    assert report.peak_hz > report.start_hz > report.end_hz
    assert torch.equal(actual[:, : report.start_index], source[:, : report.start_index])


def test_statement_operation_is_monotonic_fall_over_voiced_frames() -> None:
    actual, report = apply_final_f0_operation(_curve(), "statement_fall")

    assert report.eligible is True
    selected = actual[0, report.start_index : report.end_index_exclusive]
    selected = selected[selected > 40.0]
    assert torch.all(selected[1:] <= selected[:-1])
    assert report.peak_hz == report.start_hz
    assert report.end_hz == pytest.approx(report.start_hz * FINAL_F0_RATIO, abs=1.0)


def test_nonvoiced_curve_fails_eligibility_without_mutation() -> None:
    source = torch.full((1, 30), -1.0)

    actual, report = apply_final_f0_operation(source, "statement_fall")

    assert report.eligible is False
    assert report.reason == "insufficient_voiced_frames"
    assert torch.equal(actual, source)


def test_latent_insertion_fixed_duration_rebuilds_alignment_deterministically() -> None:
    durations = torch.tensor([1, 2, 8, 4], dtype=torch.long)
    model = SimpleNamespace(device=torch.device("cpu"))

    adjusted, alignment = _set_fixed_durations(
        durations,
        (2,),
        frames=3,
        model=model,
        torch=torch,
    )

    assert durations.tolist() == [1, 2, 8, 4]
    assert adjusted.tolist() == [1, 2, 3, 4]
    assert alignment.shape == (1, 4, 10)
    assert torch.equal(alignment.sum(dim=1), torch.ones((1, 10)))
    assert alignment[0, 2].sum().item() == 3


def test_stress_duration_transfer_preserves_total_frames() -> None:
    durations = torch.tensor([1, 1, 2, 1, 1, 4, 2], dtype=torch.long)
    model = SimpleNamespace(device=torch.device("cpu"))
    specs = ((1, 2, 4, 5, (1, 2, 4, 5)),)

    adjusted, alignment, reports = _apply_stress_duration_transfers(
        durations,
        specs,
        model=model,
        torch=torch,
    )

    assert adjusted.tolist() == [1, 1, 4, 1, 1, 2, 2]
    assert adjusted.sum().item() == durations.sum().item()
    assert alignment.shape[-1] == durations.sum().item()
    assert reports[0].eligible is True
    assert reports[0].transferred_frames == 2
    assert reports[0].transferred_ms == 50.0
    assert reports[0].duration_donor_column == 5
    assert reports[0].duration_donor_kind == "demoted_vowel"


def test_stress_duration_uses_marker_when_one_frame_vowel_cannot_donate() -> None:
    durations = torch.tensor([1, 1, 1, 1, 2, 1, 2], dtype=torch.long)
    model = SimpleNamespace(device=torch.device("cpu"))
    specs = ((1, 2, 4, 5, (1, 2, 4, 5)),)

    adjusted, alignment, reports = _apply_stress_duration_transfers(
        durations,
        specs,
        model=model,
        torch=torch,
    )

    assert adjusted.tolist() == [1, 1, 2, 1, 1, 1, 2]
    assert adjusted.sum().item() == durations.sum().item()
    assert alignment.shape[-1] == durations.sum().item()
    assert reports[0].eligible is True
    assert reports[0].transferred_frames == 1
    assert reports[0].duration_donor_column == 4
    assert reports[0].duration_donor_kind == "demoted_stress_marker"


def test_target_local_excitation_copy_leaves_other_frames_exact() -> None:
    target = torch.arange(10, dtype=torch.float32).reshape(1, 10)
    source = target + 100
    durations = torch.tensor([1, 2, 3, 4], dtype=torch.long)

    actual, count = _copy_column_frames(target, source, durations, (2,))

    assert count == 3
    assert torch.equal(actual[..., :3], target[..., :3])
    assert torch.equal(actual[..., 3:6], source[..., 3:6])
    assert torch.equal(actual[..., 6:], target[..., 6:])


def test_consonant_excitation_copy_can_cover_a_complete_word_state() -> None:
    target = torch.zeros((1, 12), dtype=torch.float32)
    source = torch.arange(12, dtype=torch.float32).reshape(1, 12)
    durations = torch.tensor([1, 2, 3, 2, 4], dtype=torch.long)

    actual, count = _copy_column_frames(target, source, durations, (1, 2, 3))

    assert count == 7
    assert torch.equal(actual[..., :1], target[..., :1])
    assert torch.equal(actual[..., 1:8], source[..., 1:8])
    assert torch.equal(actual[..., 8:], target[..., 8:])


def test_insertion_context_expands_one_phone_without_crossing_word_boundary() -> None:
    symbols = ("a", "p", "p", " ", "k")

    assert _expanded_segment_context_columns(symbols, (3,)) == (2, 3)


def test_consonant_word_context_selects_only_the_containing_word() -> None:
    symbols = tuple("mal θaf nav")

    assert _consonant_state_columns(symbols, (5,), mode="word") == (5, 6, 7)
    assert _consonant_state_columns(symbols, (5,), mode="adjacent") == (5, 6)


def test_insertion_f0_is_interpolated_only_over_reserved_frames() -> None:
    durations = torch.tensor([1, 2, 3, 2], dtype=torch.long)
    source = torch.tensor([[150.0, 155.0, 160.0, -1.0, -1.0, -1.0, 170.0, 175.0]])

    actual, count = _force_voiced_insertion_f0(source, durations, (2,))

    assert count == 3
    assert torch.equal(actual[..., :3], source[..., :3])
    assert torch.equal(actual[..., 6:], source[..., 6:])
    assert torch.all(actual[..., 3:6] > 40.0)


def test_stress_intensity_gain_is_tapered_and_target_local() -> None:
    durations = torch.tensor([1, 3, 1, 3], dtype=torch.long)
    audio = np.full(8 * 300, 0.1, dtype=np.float32)
    report = StressDurationIntervention(
        promoted_marker_column=0,
        promoted_vowel_column=1,
        demoted_marker_column=2,
        demoted_vowel_column=3,
        transferred_frames=1,
        transferred_ms=25.0,
        intensity_gain_db=3.0,
        eligible=True,
        reason=None,
        replacement_columns=(0, 1, 2, 3),
    )

    actual = _apply_stress_intensity(
        audio,
        durations,
        (report,),
        sample_rate_hz=24_000,
    )

    assert actual[0] == audio[0]
    assert actual[450] > audio[450]
    assert actual[1650] < audio[1650]
    assert actual[-1] == pytest.approx(audio[-1])
