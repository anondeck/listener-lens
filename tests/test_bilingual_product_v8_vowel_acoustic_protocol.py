from __future__ import annotations

import json

from earshift_bakeoff.config import ROOT
from earshift_bakeoff.util import sha256_file


PROTOCOL = ROOT / "rules" / "bilingual-product-v8-vowel-acoustic-screen.json"
MANIFEST = (
    ROOT
    / "artifacts"
    / "product-matrix"
    / "20260717-bilingual-product-v8-vowel-manifest"
    / "manifest.json"
)
V7_AUDIO = (
    ROOT
    / "artifacts"
    / "product-matrix"
    / "20260717-bilingual-product-isolated-audio-screen-v1"
    / "results.json"
)
V1_ACOUSTIC = (
    ROOT
    / "artifacts"
    / "product-matrix"
    / "20260717-bilingual-product-vowel-acoustic-screen-v1"
    / "results.json"
)


def _load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_v8_vowel_protocol_binds_every_stress_context_slot_before_render() -> None:
    protocol = _load(PROTOCOL)
    manifest = _load(MANIFEST)
    slots = manifest["slots"]

    assert protocol["status"] == ("frozen_before_first_broad_v8_vowel_audio_render")
    assert len(slots) == protocol["scope"]["logical_slot_count"] == 240
    assert len({slot["cell_id"] for slot in slots}) == 80
    assert len({slot["rule_id"] for slot in slots}) == 40
    assert all(slot["stress_context_added"] for slot in slots)
    assert all(slot["v8_plan_gate_pass"] for slot in slots)
    bindings = protocol["source_data_bindings"]
    assert bindings["v8_manifest_sha256"] == sha256_file(MANIFEST)
    assert bindings["v7_audio_result_sha256"] == sha256_file(V7_AUDIO)
    assert bindings["v1_acoustic_result_sha256"] == sha256_file(V1_ACOUSTIC)


def test_v8_vowel_protocol_freezes_core_measurement_and_cross_family_gates() -> None:
    protocol = _load(PROTOCOL)
    measurement = protocol["measurement_protocol"]
    gates = protocol["analysis_gates"]
    aggregation = protocol["aggregation_policy"]

    assert protocol["instrument"]["formant_ceilings_hz"] == [5500, 5750, 6000]
    assert measurement["monophthong_mode"]["core_fraction"] == [0.25, 0.75]
    assert measurement["diphthong_mode"]["core_bins"] == [
        [0.25, 0.5],
        [0.5, 0.75],
    ]
    assert gates["measurement_retention"]["minimum_valid_frames_per_bin"] == 3
    assert gates["base_vowel_endpoint"] == {
        "minimum_anchor_separation_bark_rms": 0.18,
        "minimum_controlled_movement_bark_rms": 0.18,
        "minimum_controlled_movement_fraction_of_anchor_for_exact": 0.5,
        "minimum_direction_cosine": 0.5,
        "directional_requires_target_distance_gain": True,
        "directional_requires_source_distance_increase": True,
        "exact_requires_neutral_on_source_side_of_anchor_bisector": True,
        "exact_requires_lens_on_target_side_of_anchor_bisector": True,
    }
    assert gates["analysis_family"]["one_ceiling_cannot_rescue_another"] is True
    assert aggregation["missing_or_excluded_measurement_fails_the_cell"] is True
    assert aggregation["product_promotion_allowed"] is False


def test_v8_vowel_protocol_forbids_rerender_and_preserves_claim_boundaries() -> None:
    protocol = _load(PROTOCOL)
    render = protocol["render_policy"]

    assert render["candidate_render_sets"] == 240
    assert render["natural_anchor_wav_count_reused_from_v1"] == 480
    assert render["replacement_slots_allowed"] is False
    assert render["selective_rerender_allowed"] is False
    assert render["api_calls_allowed"] == 0
    assert (
        protocol["defect_and_intervention"]["v1_result_reclassification_allowed"]
        is False
    )
    assert (
        "Not evaluated or solved"
        in protocol["claim_limits"]["consonants_insertions_and_prosody"]
    )
    assert protocol["production_enabled"] is False
    for binding in protocol["source_bindings"]:
        assert sha256_file(ROOT / binding["path"]) == binding["sha256"]
