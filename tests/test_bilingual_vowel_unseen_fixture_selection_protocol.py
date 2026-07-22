from __future__ import annotations

import json

from earshift_bakeoff.config import ROOT
from earshift_bakeoff.util import sha256_file


PROTOCOL = ROOT / "rules" / "bilingual-vowel-unseen-typed-fixture-selection-v1.json"


def _load() -> dict:
    return json.loads(PROTOCOL.read_text(encoding="utf-8"))


def test_unseen_selection_protocol_is_bound_zero_render_and_nonpromotional() -> None:
    protocol = _load()

    assert protocol["production_enabled"] is False
    assert protocol["stopping_rule"]["api_calls_allowed"] == 0
    assert protocol["stopping_rule"]["audio_renders_allowed"] == 0
    assert protocol["stopping_rule"]["product_promotion_allowed"] is False
    for binding in protocol["source_bindings"]:
        assert sha256_file(ROOT / binding["path"]) == binding["sha256"]
    for label in ("calibration", "failure", "reachability"):
        path = ROOT / protocol["parent_bindings"][f"{label}_result_path"]
        assert (
            sha256_file(path) == protocol["parent_bindings"][f"{label}_result_sha256"]
        )


def test_unseen_selection_protocol_covers_every_oral_candidate_in_three_frames() -> (
    None
):
    protocol = _load()
    scope = protocol["scope"]
    selection = protocol["selection"]

    assert scope["oral_candidate_cell_count"] == 28
    assert scope["rule_group_count"] == 15
    assert scope["logical_slot_count"] == 84
    assert scope["expected_occurrence_count"] == 112
    assert sum(scope["voice_cell_counts"].values()) == 28
    assert sum(scope["candidate_rung_counts"].values()) == 28
    assert len(selection["context_order"]) == 3
    assert len(selection["english_frames"]) == 3
    assert len(selection["portuguese_frames"]) == 3
    assert selection["minimum_canonical_rank"] == 256
    assert selection["maximum_canonical_rank"] == 75_000
    assert selection["candidates_retained_per_rule"] == 128
