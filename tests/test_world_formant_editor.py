from __future__ import annotations

import numpy as np
import pytest

from earshift_bakeoff.world_formant_editor import (
    WorldFormantSpec,
    warp_world_spectral_envelope,
)


def _spec(start: int = 240, end: int = 720) -> WorldFormantSpec:
    return WorldFormantSpec(
        start_sample=start,
        end_sample_exclusive=end,
        source_f1_hz=700.0,
        source_f2_hz=1700.0,
        source_f3_hz=2800.0,
        target_f1_hz=550.0,
        target_f2_hz=1950.0,
    )


def test_world_envelope_warp_is_local_in_time_and_deterministic() -> None:
    spectral = np.tile(np.linspace(1.0, 3.0, 513), (9, 1))
    times = np.arange(9, dtype=np.float64) * 0.005
    first, record = warp_world_spectral_envelope(
        spectral,
        times,
        sample_rate_hz=24_000,
        specs=(_spec(),),
    )
    second, _ = warp_world_spectral_envelope(
        spectral,
        times,
        sample_rate_hz=24_000,
        specs=(_spec(),),
    )
    assert np.array_equal(first, second)
    assert record["affected_frame_count"] == 3
    assert np.array_equal(first[0], spectral[0])
    assert not np.array_equal(first[4], spectral[4])
    assert np.array_equal(first[-1], spectral[-1])


def test_world_envelope_warp_rejects_overlapping_specs() -> None:
    spectral = np.ones((9, 513), dtype=np.float64)
    times = np.arange(9, dtype=np.float64) * 0.005
    with pytest.raises(ValueError, match="overlap"):
        warp_world_spectral_envelope(
            spectral,
            times,
            sample_rate_hz=24_000,
            specs=(_spec(), _spec(300, 800)),
        )


def test_world_envelope_warp_rejects_target_past_f3() -> None:
    invalid = WorldFormantSpec(
        start_sample=240,
        end_sample_exclusive=720,
        source_f1_hz=700.0,
        source_f2_hz=1700.0,
        source_f3_hz=1800.0,
        target_f1_hz=550.0,
        target_f2_hz=1900.0,
    )
    with pytest.raises(ValueError, match="invalid"):
        warp_world_spectral_envelope(
            np.ones((9, 513), dtype=np.float64),
            np.arange(9, dtype=np.float64) * 0.005,
            sample_rate_hz=24_000,
            specs=(invalid,),
        )
