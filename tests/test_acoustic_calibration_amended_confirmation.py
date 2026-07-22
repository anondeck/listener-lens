from __future__ import annotations

from earshift_bakeoff.acoustic_calibration_amended_confirmation import (
    AMENDED_RULE_SPECS,
    EXPECTED_PROTOCOL_SHA256,
    MANIFEST_SEED,
    SELECTED_SHELL,
    SELECTED_TOKENS,
    amendment_protocol_record,
    build_amendment_manifest,
    candidate_gate_audit_record,
    reused_evidence_record,
)


def test_amended_manifest_has_exactly_the_twelve_frozen_slots() -> None:
    manifest = build_amendment_manifest()

    assert len(manifest) == 12
    assert len({item.slot_id for item in manifest}) == 12
    assert all(item.kind == "contrast" for item in manifest)
    assert {item.rule_id for item in manifest} == {
        "ptbr.vowel.ih_to_i",
        "ptbr.vowel.ae_to_eh",
    }
    assert {item.shell for item in manifest} == {SELECTED_SHELL}
    assert {item.take for item in manifest} == {1, 2, 3}
    assert MANIFEST_SEED == "calibration-v3-amended-confirmation-20260715"
    assert build_amendment_manifest() == manifest
    assert [item.slot_id for item in manifest] == [
        "contrast__ae_to_eh__b_V_vd__neutral__take-2",
        "contrast__ih_to_i__b_V_vd__neutral__take-1",
        "contrast__ae_to_eh__b_V_vd__lens__take-1",
        "contrast__ih_to_i__b_V_vd__lens__take-2",
        "contrast__ih_to_i__b_V_vd__lens__take-3",
        "contrast__ae_to_eh__b_V_vd__neutral__take-3",
        "contrast__ih_to_i__b_V_vd__lens__take-1",
        "contrast__ae_to_eh__b_V_vd__lens__take-2",
        "contrast__ae_to_eh__b_V_vd__neutral__take-1",
        "contrast__ih_to_i__b_V_vd__neutral__take-2",
        "contrast__ae_to_eh__b_V_vd__lens__take-3",
        "contrast__ih_to_i__b_V_vd__neutral__take-3",
    ]


def test_amended_manifest_uses_the_selected_tokens_and_spans() -> None:
    for item in build_amendment_manifest():
        if item.rule_id == "ptbr.vowel.ih_to_i":
            expected = SELECTED_TOKENS["ih" if item.side == "neutral" else "ee"]
            span = (
                item.neutral_character_span
                if item.side == "neutral"
                else item.lens_character_span
            )
            grapheme = "ih" if item.side == "neutral" else "ee"
        else:
            expected = SELECTED_TOKENS["a" if item.side == "neutral" else "eh"]
            span = (
                item.neutral_character_span
                if item.side == "neutral"
                else item.lens_character_span
            )
            grapheme = "a" if item.side == "neutral" else "eh"
        assert item.token == expected
        assert span is not None
        assert item.token[slice(*span)] == grapheme


def test_gate_audit_freezes_the_first_passing_candidate() -> None:
    audit = candidate_gate_audit_record()

    assert audit["selected"]["rank"] == 5
    assert audit["selected"]["shell"] == SELECTED_SHELL
    assert audit["audit_sha256"] == (
        "e0378d1653b6e553f3e8ee85d272742553891cfbe12bdc0fdec428d59640a30c"
    )
    assert [
        candidate["rank"]
        for candidate in audit["candidates"]
        if candidate["passed"]
    ] == [5, 6, 8]


def test_reused_evidence_is_exactly_twelve_anchors_and_twenty_four_cells() -> None:
    binding, records = reused_evidence_record()

    assert binding["reused_audio_count"] == 36
    assert binding["binding_sha256"] == (
        "fd8083a36d935d4074771e2d4ebefd37c17895beaf9fe4c8f0b4905b4bc59b36"
    )
    assert sum(record["stimulus"]["kind"] == "reference" for record in records) == 12
    assert sum(record["stimulus"]["kind"] == "contrast" for record in records) == 24
    assert {
        record["stimulus"].get("shell")
        for record in records
        if record["stimulus"]["kind"] == "contrast"
    } == {"z_V_f", "v_V_p"}
    assert {rule["rule_id"] for rule in AMENDED_RULE_SPECS} == {
        "ptbr.vowel.ih_to_i",
        "ptbr.vowel.ae_to_eh",
    }


def test_complete_amendment_protocol_hash_is_frozen() -> None:
    assert EXPECTED_PROTOCOL_SHA256 == (
        "3c55d9d2c5d30a399829b7b0559f00ec2c9eadb62b59293d44f950d75a3a1814"
    )
    assert amendment_protocol_record()["protocol_sha256"] == EXPECTED_PROTOCOL_SHA256
