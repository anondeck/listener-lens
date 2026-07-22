from earshift_bakeoff.report import _truth


def test_truth_accepts_csv_text_and_computed_booleans() -> None:
    assert _truth("True")
    assert _truth("yes")
    assert _truth(True)
    assert not _truth("False")
    assert not _truth(False)
