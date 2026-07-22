from __future__ import annotations

import sqlite3

from earshift_bakeoff.portuguese_kokoro_gate import (
    CHALLENGE_WORDS,
    LANGUAGE_ID,
    NORMALIZATION_VERSION,
    PortugueseKokoroExtractor,
    PortugueseKokoroGateIndex,
    _phone_hash,
    protocol_record,
)


def test_pinned_portuguese_g2p_challenge_inventory_is_exact() -> None:
    extractor = PortugueseKokoroExtractor()
    records = extractor.extract(tuple(CHALLENGE_WORDS))
    assert {record.word: record.phone for record in records} == CHALLENGE_WORDS
    assert all(record.rejection_reason is None for record in records)


def test_portuguese_protocol_is_language_scoped_and_claim_bounded() -> None:
    protocol = protocol_record()
    assert protocol["language_id"] == LANGUAGE_ID
    assert protocol["extraction"]["normalization_version"] == NORMALIZATION_VERSION
    assert "does not enumerate contextual variants" in protocol["negative_lookup_scope"]
    assert protocol["status"] == "frozen_before_full_index_build"
    assert len(protocol["protocol_sha256"]) == 64


def test_portuguese_index_lookup_uses_language_and_normalization_domain(
    tmp_path,
) -> None:
    database = tmp_path / "index.sqlite3"
    with sqlite3.connect(database) as conn:
        conn.execute(
            "CREATE TABLE word_phone(word_sha256 BLOB NOT NULL, "
            "phone_sha256 BLOB NOT NULL, PRIMARY KEY(word_sha256, phone_sha256)) "
            "WITHOUT ROWID"
        )
        conn.execute(
            "INSERT INTO word_phone VALUES (?, ?)", (b"w" * 32, _phone_hash("pˈɐ̃ʊ̃"))
        )
    index = PortugueseKokoroGateIndex(database)
    assert index.phone_match("pˈɐ̃ʊ̃") is True
    assert index.phone_match("pˈaʊ") is False
