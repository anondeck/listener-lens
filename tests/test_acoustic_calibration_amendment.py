from earshift_bakeoff.acoustic_calibration_amendment import CANDIDATE_INVENTORY


def test_amendment_candidate_inventory_is_bounded_and_frozen() -> None:
    assert len(CANDIDATE_INVENTORY) == 12
    assert [candidate.rank for candidate in CANDIDATE_INVENTORY] == list(range(1, 13))
    assert [candidate.shell for candidate in CANDIDATE_INVENTORY] == [
        "b_V_v",
        "d_V_v",
        "v_V_d",
        "z_V_d",
        "b_V_vd",
        "d_V_vd",
        "g_V_vd",
        "z_V_vd",
        "b_V_th",
        "d_V_th",
        "g_V_th",
        "v_V_th",
    ]
    assert "z_V_b" not in {candidate.shell for candidate in CANDIDATE_INVENTORY}
    assert all(len(candidate.surfaces) == 4 for candidate in CANDIDATE_INVENTORY)
    assert all(len(set(candidate.surfaces)) == 4 for candidate in CANDIDATE_INVENTORY)


def test_amendment_surfaces_encode_the_four_frozen_vowel_spellings() -> None:
    for candidate in CANDIDATE_INVENTORY:
        assert "ih" in candidate.ih
        assert "ee" in candidate.ee
        assert "a" in candidate.a
        assert "eh" in candidate.eh
