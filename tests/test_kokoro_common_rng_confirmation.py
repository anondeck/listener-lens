from __future__ import annotations

from earshift_bakeoff.kokoro_common_rng_confirmation import RNG_SEED, _review, manifest, protocol_record


def test_manifest_has_seven_product_rows_and_six_context_anchors() -> None:
    slots = manifest()
    assert len(slots) == 13
    assert sum(slot.kind == "shared_state" for slot in slots) == 7
    assert sum(slot.kind == "context_anchor" for slot in slots) == 6
    assert [slot.request_order for slot in slots] == list(range(1, 14))


def test_protocol_freezes_common_random_numbers_and_context_gate() -> None:
    protocol = protocol_record()
    assert protocol["common_random_contract"]["seed"] == RNG_SEED
    assert protocol["common_random_contract"]["execution"].startswith("separate decoder calls")
    assert protocol["anchor_gate"]["endpoints"].startswith("full-carrier")
    assert protocol["renderer"]["api_calls"] == 0
    assert len(protocol["protocol_sha256"]) == 64


def test_review_download_newline_is_valid_javascript_escape(tmp_path) -> None:
    records = [
        {"slot_id": "common-neutral", "condition": "neutral", "audio_relative_path": "neutral.wav"},
        {
            "slot_id": "common-neutral-identity",
            "condition": "identity",
            "audio_relative_path": "identity.wav",
        },
        {"slot_id": "lens", "condition": "stress-plus-target", "audio_relative_path": "lens.wav"},
    ]

    _review(records, "stress-plus-target", tmp_path)

    html = (tmp_path / "review.html").read_text(encoding="utf-8")
    assert "JSON.stringify(payload,null,2)+'\\n'" in html
    assert "JSON.stringify(payload,null,2)+'\n'" not in html
