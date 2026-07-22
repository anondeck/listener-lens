from __future__ import annotations

import numpy as np

from earshift_bakeoff.same_take_followup import band_power_db, signed_rms_db


def test_signed_rms_db_retains_direction() -> None:
    reference = np.array([1000, -1000, 1000, -1000], dtype=np.int16)
    louder = reference.astype(np.int32) * 2
    quieter = reference.astype(np.float64) * 0.5
    assert 6.01 < signed_rms_db(reference, louder) < 6.03
    assert -6.03 < signed_rms_db(reference, quieter) < -6.01


def test_high_band_measure_separates_low_and_high_tones() -> None:
    sample_rate = 24_000
    time = np.arange(2_400) / sample_rate
    low = np.sin(2 * np.pi * 1_000 * time)
    high = np.sin(2 * np.pi * 7_000 * time)
    assert band_power_db(high, sample_rate) > band_power_db(low, sample_rate) + 80
