from __future__ import annotations

from dataclasses import asdict
import json

from earshift_bakeoff.bilingual_vowel_spectral_category import (
    DEFAULT_FEATURE_CONFIG,
    MAXIMUM_REVERSED_HELDOUT_ANCHOR_PAIRS,
    MINIMUM_HELDOUT_EXACT_ANCHOR_PAIRS,
    SPECTRAL_CATEGORY_VERSION,
)
from earshift_bakeoff.config import ROOT
from earshift_bakeoff.util import sha256_file


PROTOCOL = ROOT / "rules" / "bilingual-vowel-spectral-category-screen-v1.json"


def _load() -> dict:
    return json.loads(PROTOCOL.read_text(encoding="utf-8"))


def test_spectral_protocol_is_zero_api_hash_bound_and_nonpromotional() -> None:
    protocol = _load()

    assert protocol["production_enabled"] is False
    assert protocol["stopping_rule"]["api_calls_allowed"] == 0
    assert protocol["stopping_rule"]["new_audio_renders_allowed"] == 0
    assert protocol["stopping_rule"]["post_result_threshold_tuning_allowed"] is False
    assert protocol["stopping_rule"]["product_promotion_allowed"] is False
    for binding in protocol["source_bindings"]:
        assert sha256_file(ROOT / binding["path"]) == binding["sha256"]
    for label in ("v8", "v1", "reachability"):
        path = ROOT / protocol["parent_bindings"][f"{label}_result_path"]
        assert (
            sha256_file(path) == protocol["parent_bindings"][f"{label}_result_sha256"]
        )


def test_spectral_protocol_freezes_feature_and_validation_contract() -> None:
    protocol = _load()
    validation = protocol["leave_context_out_validation"]

    assert protocol["feature_extraction"]["version"] == SPECTRAL_CATEGORY_VERSION
    assert protocol["feature_extraction"]["config"] == json.loads(
        json.dumps(asdict(DEFAULT_FEATURE_CONFIG))
    )
    assert validation["heldout_unit"].startswith("Complete logical carrier slot")
    assert (
        validation["minimum_exact_heldout_pairs"]
        == MINIMUM_HELDOUT_EXACT_ANCHOR_PAIRS
        == 3
    )
    assert (
        validation["maximum_reversed_heldout_pairs"]
        == MAXIMUM_REVERSED_HELDOUT_ANCHOR_PAIRS
        == 0
    )
    assert protocol["candidate_classification"]["identity_negative_control"].endswith(
        "all 320 occurrences."
    )


def test_spectral_protocol_uses_complete_matrix_and_typed_core_denominator() -> None:
    scope = _load()["scope"]

    assert scope["voice_rule_cell_count"] == 80
    assert scope["logical_slot_count"] == 240
    assert scope["occurrence_count"] == 320
    assert scope["typed_core_cell_count"] == 36
    assert len(scope["typed_core_cell_ids_in_order"]) == 36
    assert len(set(scope["typed_core_cell_ids_in_order"])) == 36
