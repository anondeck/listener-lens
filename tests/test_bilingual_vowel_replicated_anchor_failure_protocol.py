from __future__ import annotations

import json

from earshift_bakeoff.config import ROOT
from earshift_bakeoff.util import sha256_file


PROTOCOL = ROOT / "rules" / "bilingual-vowel-replicated-anchor-failure-screen-v1.json"


def _load() -> dict:
    return json.loads(PROTOCOL.read_text(encoding="utf-8"))


def test_failure_protocol_is_hash_bound_zero_api_and_nonpromotional() -> None:
    protocol = _load()

    assert protocol["production_enabled"] is False
    assert protocol["stopping_rule"]["api_calls_allowed"] == 0
    assert protocol["stopping_rule"]["new_candidate_decoder_renders_allowed"] == 0
    assert protocol["stopping_rule"]["product_promotion_allowed"] is False
    for binding in protocol["source_bindings"]:
        assert sha256_file(ROOT / binding["path"]) == binding["sha256"]
    for label in ("calibration", "v8", "adaptive"):
        path = ROOT / protocol["parent_bindings"][f"{label}_result_path"]
        assert (
            sha256_file(path) == protocol["parent_bindings"][f"{label}_result_sha256"]
        )


def test_failure_protocol_uses_only_anchor_eligible_failures() -> None:
    protocol = _load()
    scope = protocol["scope"]
    hierarchy = protocol["candidate_hierarchy"]

    assert scope["calibrated_core_cell_count"] == 36
    assert scope["eligible_failure_cell_count"] == 23
    assert scope["eligible_failure_slot_count"] == 69
    assert scope["eligible_failure_occurrence_count"] == 92
    assert hierarchy["v8_cell_count"] == 21
    assert hierarchy["adaptive_strength_cell_count"] == 2
    assert len(hierarchy["cells_in_order"]) == 23
    assert len({row["cell_id"] for row in hierarchy["cells_in_order"]}) == 23
    assert scope["ineligible_failure_cell"]["cell_id"] not in {
        row["cell_id"] for row in hierarchy["cells_in_order"]
    }


def test_failure_protocol_keeps_calibrated_instrument_and_strength_order() -> None:
    protocol = _load()

    assert protocol["anchor_reproduction"]["decoder_render_count"] == 864
    assert protocol["anchor_reproduction"]["new_anchor_wav_retention"] == 0
    assert protocol["candidate_hierarchy"]["adaptive_strength_order"] == [
        1.0,
        0.75,
        1.25,
        0.5,
        1.5,
        2.0,
    ]
    assert (
        "first exact-category"
        in protocol["candidate_hierarchy"]["adaptive_occurrence_selection"]
    )
    assert protocol["classification_and_aggregation"]["cell"].startswith(
        "All four occurrences"
    )
