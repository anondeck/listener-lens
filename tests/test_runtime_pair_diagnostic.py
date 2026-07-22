from earshift_bakeoff.runtime_pair_diagnostic import classify_points
from earshift_bakeoff.sentence_pair_v2 import ANCHOR_GATE


def _points(fraction: float) -> dict[str, tuple[float, float]]:
    result = {}
    for key, anchor in ANCHOR_GATE["families"].items():
        source = anchor["source_centroid_bark"]
        vector = anchor["anchor_vector_bark"]
        result[key] = (
            source[0] + fraction * vector[0],
            source[1] + fraction * vector[1],
        )
    return result


def test_runtime_pair_category_and_direction_pass() -> None:
    result = classify_points(_points(0.0), _points(0.9))
    assert result["classification"] == "category_and_direction_diagnostic_pass"
    assert all(
        family["category_and_direction_pass"]
        for family in result["families"].values()
    )


def test_runtime_pair_directional_only_when_lens_stays_source_near() -> None:
    result = classify_points(_points(0.0), _points(0.3))
    assert result["classification"] == "directional_only_diagnostic"


def test_runtime_pair_fails_on_opposite_movement() -> None:
    result = classify_points(_points(0.0), _points(-0.5))
    assert result["classification"] == "diagnostic_fail"
