from __future__ import annotations

import json

from earshift_bakeoff.config import ROOT
from earshift_bakeoff.util import sha256_file


PROTOCOL = ROOT / "rules" / "bilingual-vowel-unseen-blind-qc-v1.json"


def _load() -> dict:
    return json.loads(PROTOCOL.read_text(encoding="utf-8"))


def test_blind_qc_protocol_is_bound_bounded_and_nonpromotional() -> None:
    protocol = _load()
    bindings = protocol["parent_bindings"]

    assert protocol["status"] == "frozen_before_first_human_review"
    assert protocol["production_enabled"] is False
    assert protocol["scope"] == {
        "automatic_candidate_cell_count": 18,
        "candidate_context_count_per_cell": 3,
        "neutral_lens_trial_count": 54,
        "bit_identical_identity_trial_count": 18,
        "total_trial_count": 72,
        "voice_session_counts": {
            "af_heart": 24,
            "am_michael": 32,
            "pf_dora": 8,
            "pm_alex": 8,
        },
        "retained_wavs_only": True,
        "new_audio_renders_allowed": 0,
        "api_calls_allowed": 0,
    }
    for label in ("unseen_confirmation", "typed_manifest", "automatic_protocol"):
        assert sha256_file(ROOT / bindings[f"{label}_path"]) == bindings[
            f"{label}_sha256"
        ]
    assert protocol["stopping_rule"]["production_enabled_after_review"] is False


def test_blind_qc_protocol_hides_every_condition_identifier() -> None:
    presentation = _load()["presentation"]

    for field in (
        "session_identity_hidden",
        "condition_hidden",
        "source_text_hidden",
        "carrier_script_hidden",
        "source_word_hidden",
        "rule_id_hidden",
        "source_and_target_ipa_hidden",
        "original_filename_hidden",
        "shared_target_cues_shown_in_every_condition",
    ):
        assert presentation[field] is True


def test_blind_qc_protocol_requires_every_context_without_reclassification() -> None:
    protocol = _load()

    assert protocol["cell_gate"]["all_three_neutral_lens_contexts_must_pass"]
    assert not protocol["cell_gate"]["favorable_context_selection_allowed"]
    assert not protocol["cell_gate"]["wrong_direction_allowed"]
    assert not protocol["cell_gate"]["automatic_result_reclassification_allowed"]
    assert protocol["identity_control_handling"][
        "clean_identity_required_before_product_promotion"
    ]
    assert len(protocol["direction_prompts"]) == 13
