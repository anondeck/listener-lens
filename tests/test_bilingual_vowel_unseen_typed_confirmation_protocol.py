from __future__ import annotations

import json

from earshift_bakeoff.config import ROOT
from earshift_bakeoff.util import sha256_file


PROTOCOL = ROOT / "rules" / "bilingual-vowel-unseen-typed-confirmation-v1.json"


def _load() -> dict:
    return json.loads(PROTOCOL.read_text(encoding="utf-8"))


def test_unseen_confirmation_is_bound_zero_api_and_nonpromotional() -> None:
    protocol = _load()

    assert protocol["production_enabled"] is False
    assert protocol["rendering"]["api_calls_allowed"] == 0
    assert protocol["stopping_rule"]["api_calls_allowed"] == 0
    assert protocol["stopping_rule"]["product_promotion_allowed"] is False
    for binding in protocol["source_bindings"]:
        assert sha256_file(ROOT / binding["path"]) == binding["sha256"]
    for label in ("manifest", "calibration", "failure"):
        path = ROOT / protocol["parent_bindings"][f"{label}_path"]
        assert sha256_file(path) == protocol["parent_bindings"][f"{label}_sha256"]


def test_unseen_confirmation_freezes_complete_render_denominator() -> None:
    protocol = _load()
    scope = protocol["scope"]
    rendering = protocol["rendering"]

    assert scope["oral_candidate_cell_count"] == 28
    assert scope["rule_group_count"] == 15
    assert scope["logical_slot_count"] == 84
    assert scope["target_occurrence_count"] == 112
    assert sum(scope["candidate_rung_counts"].values()) == 28
    assert rendering["natural_anchor_decoder_render_count"] == 672
    assert rendering["candidate_render_set_count"] == 174
    assert sum(rendering["candidate_render_set_breakdown"].values()) == 174
    assert rendering["adaptive_strength_order"] == [
        1.0,
        0.75,
        1.25,
        0.5,
        1.5,
        2.0,
    ]


def test_unseen_confirmation_requires_every_occurrence_and_identity_control() -> None:
    protocol = _load()
    gates = protocol["instrument_and_gates"]

    assert gates["minimum_anchor_separation_scaled_rms"] == 0.25
    assert gates["minimum_direction_cosine"] == 0.5
    assert gates["minimum_directional_movement_fraction"] == 0.25
    assert gates["minimum_exact_movement_fraction"] == 0.5
    assert "All four target occurrences" in gates["cell_gate"]
    assert "All 112" in gates["identity_gate"]
    assert protocol["stopping_rule"]["selective_rerender_allowed"] is False
