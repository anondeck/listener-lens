from __future__ import annotations

from earshift_bakeoff.kokoro_specs import (
    ENGLISH_SCREEN_SHORTLIST,
    LANGUAGE_SPECS,
    PORTUGUESE_SCREEN_VOICES,
    VOICE_SPECS,
    VOICE_SPECS_BY_ID,
    voice_inventory_receipt,
)


def test_exact_language_voice_inventory_and_disabled_candidates() -> None:
    assert len(VOICE_SPECS) == 23
    assert len(LANGUAGE_SPECS["en-US"].voice_ids) == 20
    assert PORTUGUESE_SCREEN_VOICES == ("pf_dora", "pm_alex", "pm_santa")
    assert LANGUAGE_SPECS["pt-BR"].voice_ids == PORTUGUESE_SCREEN_VOICES
    assert all(not spec.renderer_candidate_enabled for spec in LANGUAGE_SPECS.values())


def test_frozen_english_shortlist_is_grade_bounded_and_preserves_anchor() -> None:
    assert ENGLISH_SCREEN_SHORTLIST == (
        "af_heart",
        "af_bella",
        "af_nicole",
        "am_fenrir",
        "am_michael",
        "am_puck",
    )
    assert VOICE_SPECS_BY_ID["af_heart"].evidence_anchor is True
    assert set(ENGLISH_SCREEN_SHORTLIST).issubset(
        set(LANGUAGE_SPECS["en-US"].voice_ids)
    )


def test_all_pinned_voice_packs_are_locally_hash_and_shape_verified() -> None:
    receipt = voice_inventory_receipt(download=False)
    assert receipt["license"] == "Apache-2.0"
    assert len(receipt["voices"]) == 23
    assert all(item["model_compatible"] for item in receipt["voices"])
    assert all(item["shape"] == [510, 1, 256] for item in receipt["voices"])
    assert receipt["receipt_sha256"]
