from __future__ import annotations

import hashlib
import json

from earshift_bakeoff.bilingual_product_matrix import (
    BILINGUAL_PRODUCT_MATRIX_PATH,
    load_bilingual_product_matrix,
    load_bilingual_structural_state,
)
from earshift_bakeoff.config import ROOT, stable_json
from earshift_bakeoff.util import sha256_file


PROTOCOL = ROOT / "rules" / "bilingual-product-audio-integrity-screen-v1.json"
MANIFEST = (
    ROOT
    / "artifacts"
    / "product-matrix"
    / "20260717-bilingual-product-matrix-v1"
    / "manifest.json"
)


def _load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_audio_integrity_screen_is_frozen_to_one_slot_per_changed_cell() -> None:
    protocol = _load(PROTOCOL)
    manifest = _load(MANIFEST)["validation_manifest"]
    selected = {}
    for slot in manifest["slots"]:
        selected.setdefault(slot["cell_id"], slot)
    logical_ids = [slot["logical_slot_id"] for slot in selected.values()]

    assert len(logical_ids) == 98
    assert protocol["status"] == "frozen_before_first_audio_render"
    assert protocol["slot_selection"]["logical_slot_count"] == 98
    assert protocol["slot_selection"]["logical_slots_sha256"] == hashlib.sha256(
        stable_json(logical_ids).encode("utf-8")
    ).hexdigest()
    assert protocol["source_manifest_binding"]["sha256"] == sha256_file(
        MANIFEST
    )


def test_audio_screen_preserves_no_replacement_and_no_promotion_boundaries() -> None:
    protocol = _load(PROTOCOL)
    renderer = protocol["renderer_policy"]
    classification = protocol["classification_policy"]

    assert renderer["voices_in_order"] == [
        "af_heart",
        "am_michael",
        "pm_alex",
        "pf_dora",
    ]
    assert renderer["same_voice_neutral_lens_required"] is True
    assert renderer["common_rng_required"] is True
    assert renderer["one_valid_render_set_per_slot"] is True
    assert renderer["replacement_slots_allowed"] is False
    assert renderer["selective_rerender_allowed"] is False
    assert renderer["api_calls_allowed"] == 0
    assert classification["product_promotion_allowed"] is False
    assert classification["human_qc"] == "not_generated_by_this_screen"
    assert all(
        value == "pending_separate_family_analysis"
        for key, value in classification.items()
        if key.endswith("_acoustic_classification")
    )


def test_audio_screen_binds_the_matrix_state_and_every_source_file() -> None:
    protocol = _load(PROTOCOL)
    matrix = load_bilingual_product_matrix()
    load_bilingual_structural_state(matrix)

    assert protocol["matrix_binding"]["path"] == str(
        BILINGUAL_PRODUCT_MATRIX_PATH.relative_to(ROOT)
    )
    assert protocol["matrix_binding"]["sha256"] == sha256_file(
        BILINGUAL_PRODUCT_MATRIX_PATH
    )
    assert protocol["matrix_binding"]["matrix_sha256"] == matrix.matrix_sha256
    for binding in protocol["source_bindings"]:
        assert sha256_file(ROOT / binding["path"]) == binding["sha256"]
