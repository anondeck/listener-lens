from earshift_bakeoff.sentence_pair_v2 import (
    ANCHOR_GATE,
    CARRIERS,
    build_manifest,
    prompt_contract_fingerprint,
    protocol_record,
)


def test_sentence_pair_v2_uses_only_three_validated_ae_shells() -> None:
    assert {carrier.shell for carrier in CARRIERS} == {"z_V_f", "v_V_p", "b_V_vd"}
    assert all(carrier.target_word_index_zero_based == 2 for carrier in CARRIERS)
    assert all(len(carrier.neutral_script.split()) == 5 for carrier in CARRIERS)
    assert all(len(carrier.lens_script.split()) == 5 for carrier in CARRIERS)


def test_sentence_pair_v2_manifest_is_exactly_four_by_two_by_three() -> None:
    manifest = build_manifest()

    assert len(manifest) == 24
    assert len({slot.slot_id for slot in manifest}) == 24
    for carrier in CARRIERS:
        selected = [slot for slot in manifest if slot.carrier_id == carrier.carrier_id]
        assert sum(slot.side == "neutral" for slot in selected) == 4
        assert sum(slot.side == "lens" for slot in selected) == 4
        assert {slot.take_index for slot in selected if slot.side == "neutral"} == {1, 2, 3, 4}
        assert {slot.take_index for slot in selected if slot.side == "lens"} == {1, 2, 3, 4}


def test_each_request_block_balances_carrier_and_side() -> None:
    manifest = build_manifest()

    for start in range(0, 24, 6):
        block = manifest[start : start + 6]
        assert {slot.carrier_id for slot in block} == {carrier.carrier_id for carrier in CARRIERS}
        for carrier in CARRIERS:
            sides = {slot.side for slot in block if slot.carrier_id == carrier.carrier_id}
            assert sides == {"neutral", "lens"}


def test_anchor_family_passed_before_rendering() -> None:
    assert ANCHOR_GATE["passed_before_rendering"] is True
    assert ANCHOR_GATE["maximum_formant_hz_family"] == [5500, 5750, 6000]
    assert min(ANCHOR_GATE["pairwise_anchor_vector_cosines"]) >= 0.75
    assert all(
        family["magnitude_bark"] > family["magnitude_threshold_bark"]
        and family["minimum_cross_take_cosine"] >= 0.50
        and family["anchor_vector_bark"][0] < 0
        and family["anchor_vector_bark"][1] > 0
        for family in ANCHOR_GATE["families"].values()
    )


def test_prompt_and_protocol_are_frozen_and_zero_call() -> None:
    record = protocol_record()

    assert prompt_contract_fingerprint() == "56849071dc1b30123b0292c4a3fa63a530f549bb5fde5e38982d427c5ad14026"
    assert record["manifest"]["logical_slots"] == 24
    assert record["request_policy"]["successful_audio_ceiling"] == 24
    assert record["request_policy"]["maximum_attempts_per_slot"] == 2
    assert record["cost"]["api_calls_already_made_for_this_run"] == 0
    assert record["downstream_work_held"] == [
        "typed-audio connection",
        "rule-profile expansion",
        "Portuguese GPT Audio nonce-renderer study",
    ]


def test_pre_render_amendment_hides_stimulus_identity_and_separates_external_failure() -> None:
    record = protocol_record()
    blinding = record["listener_pilot"]["blinding"]

    assert all(
        phrase in blinding
        for phrase in ("spelling", "script", "filename", "condition label", "token identity")
    )
    assert record["listener_pilot"]["visible_structure"][2] == "highlighted position 3"
    assert record["request_policy"]["evidentiary_failure_after_audio_never_triggers_retry"] is True
    assert "inconclusive_external_failure" in record["stopping_rule"]["yield"]
    assert "do not by themselves prove" in record["listener_decision_rule"]["interpretation"]
