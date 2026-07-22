from __future__ import annotations

import numpy as np

from earshift_bakeoff.same_take_world import shift_envelope, warp_spectral_envelope


def test_shift_envelope_uses_frozen_linear_taper() -> None:
    assert shift_envelope(0.334, 0.335, 0.415) == 0
    assert shift_envelope(0.335, 0.335, 0.415) == 0
    assert abs(shift_envelope(0.345, 0.335, 0.415) - 0.5) < 1e-12
    assert shift_envelope(0.375, 0.335, 0.415) == 1
    assert abs(shift_envelope(0.405, 0.335, 0.415) - 0.5) < 1e-12


def test_identity_warp_is_bit_identical_in_float_domain() -> None:
    spectral = np.arange(5 * 513, dtype=np.float64).reshape(5, 513) + 1
    times = np.arange(5) * 0.005
    output, record = warp_spectral_envelope(
        spectral, times, 24_000, 0.0, 0.0, 0.02
    )
    assert np.array_equal(output, spectral)
    assert record["affected_frame_count"] == 0
