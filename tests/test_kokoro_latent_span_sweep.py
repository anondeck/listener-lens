from __future__ import annotations

from earshift_bakeoff.kokoro_latent_span_sweep import VARIANT_ORDER, manifest, protocol_record


def test_manifest_is_one_seven_row_batch() -> None:
    slots = manifest()
    assert len(slots) == 7
    assert [slot.request_order for slot in slots] == list(range(1, 8))
    assert [slot.condition for slot in slots[2:]] == list(VARIANT_ORDER)


def test_protocol_freezes_smallest_span_selection_and_zero_api() -> None:
    protocol = protocol_record()
    assert protocol["renderer"]["api_calls"] == 0
    assert "first passing span" in protocol["acoustic_gate"]["selection"]
    assert protocol["fixed_inputs"]["input_delta"] == "one raw phoneme: /ae/ becomes /eh/"
    assert len(protocol["protocol_sha256"]) == 64

