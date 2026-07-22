from __future__ import annotations

from dataclasses import asdict
import json

from earshift_bakeoff.bilingual_vowel_replicated_anchors import (
    MAXIMUM_REVERSED_SEED_PAIRS_PER_OCCURRENCE,
    MINIMUM_EXACT_SEED_PAIRS_PER_OCCURRENCE,
    TRAINING_SEEDS,
)
from earshift_bakeoff.bilingual_vowel_spectral_category import (
    DEFAULT_FEATURE_CONFIG,
)
from earshift_bakeoff.config import ROOT
from earshift_bakeoff.kokoro_synthesis import RNG_SEED
from earshift_bakeoff.util import sha256_file


PROTOCOL = ROOT / "rules" / "bilingual-vowel-replicated-anchor-calibration-v1.json"


def _load() -> dict:
    return json.loads(PROTOCOL.read_text(encoding="utf-8"))


def test_replicated_anchor_protocol_is_hash_bound_zero_api_and_nonpromotional() -> None:
    protocol = _load()

    assert protocol["production_enabled"] is False
    assert protocol["stopping_rule"]["api_calls_allowed"] == 0
    assert protocol["stopping_rule"]["failure_cell_evaluation_allowed"] is False
    assert protocol["stopping_rule"]["product_promotion_allowed"] is False
    for binding in protocol["source_bindings"]:
        assert sha256_file(ROOT / binding["path"]) == binding["sha256"]
    for label in ("v8", "v1", "word", "full", "adaptive", "reachability"):
        path = ROOT / protocol["parent_bindings"][f"{label}_result_path"]
        assert (
            sha256_file(path) == protocol["parent_bindings"][f"{label}_result_sha256"]
        )


def test_replicated_anchor_protocol_freezes_seeds_features_and_gates() -> None:
    protocol = _load()
    rendering = protocol["replicated_anchor_rendering"]
    validation = protocol["context_matched_validation"]

    assert rendering["baseline_parity_seed"] == RNG_SEED
    assert tuple(rendering["training_seeds_in_order"]) == TRAINING_SEEDS
    assert rendering["decoder_render_count"] == 864
    assert rendering["spectral_feature_config"] == json.loads(
        json.dumps(asdict(DEFAULT_FEATURE_CONFIG))
    )
    assert (
        validation["minimum_exact_seed_pairs_per_occurrence"]
        == MINIMUM_EXACT_SEED_PAIRS_PER_OCCURRENCE
        == 2
    )
    assert (
        validation["maximum_reversed_seed_pairs_per_occurrence"]
        == MAXIMUM_REVERSED_SEED_PAIRS_PER_OCCURRENCE
        == 0
    )


def test_replicated_anchor_protocol_calibrates_only_current_reference_cells() -> None:
    protocol = _load()
    scope = protocol["scope"]
    ladder = protocol["reference_candidate_ladder"]

    assert scope["typed_core_cell_count"] == 36
    assert scope["logical_slot_count"] == 108
    assert scope["occurrence_count"] == 144
    assert len(scope["typed_core_cell_ids_in_order"]) == 36
    assert ladder["reference_candidate_cell_count"] == 12
    assert len(ladder["cells_in_order"]) == 12
    assert len({row["cell_id"] for row in ladder["cells_in_order"]}) == 12
    assert (
        _load()["global_instrument_sanity"]["minimum_reference_concordant_cell_count"]
        == 10
    )
