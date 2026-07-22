from earshift_bakeoff.curated_matched_pair import (
    EXPECTED_PROTOCOL_SHA256,
    LENS_SCRIPT,
    NEUTRAL_SCRIPT,
    SOURCE_SENTENCE,
    build_manifest,
    frozen_transform_record,
    protocol_record,
)


def test_flagship_transform_is_frozen_to_one_enabled_rule() -> None:
    record = frozen_transform_record()

    assert record["source_sentence"] == SOURCE_SENTENCE
    assert record["neutral_script"] == NEUTRAL_SCRIPT
    assert record["lens_script"] == LENS_SCRIPT
    assert [rule["rule_id"] for rule in record["applied_rules"]] == [
        "ptbr.vowel.ae_to_eh"
    ]
    assert len(record["slots"]) == 3


def test_curated_manifest_is_exactly_four_plus_four_in_frozen_order() -> None:
    manifest = build_manifest()

    assert len(manifest) == 8
    assert sum(slot.side == "neutral" for slot in manifest) == 4
    assert sum(slot.side == "lens" for slot in manifest) == 4
    assert [slot.slot_id for slot in manifest] == [
        "lens__take-2",
        "neutral__take-3",
        "neutral__take-1",
        "lens__take-1",
        "lens__take-3",
        "neutral__take-2",
        "lens__take-4",
        "neutral__take-4",
    ]
    assert build_manifest() == manifest


def test_curated_protocol_hash_is_frozen() -> None:
    assert EXPECTED_PROTOCOL_SHA256
    assert protocol_record()["protocol_sha256"] == EXPECTED_PROTOCOL_SHA256
