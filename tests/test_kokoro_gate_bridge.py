from __future__ import annotations

from pathlib import Path

from earshift_bakeoff.kokoro_gate_bridge import (
    KokoroGateIndex,
    MANDATORY_WORDS,
    NORMALIZATION_VERSION,
    SAMPLE_SIZE,
    WordVariants,
    build_full_index,
    normalize_kokoro_phone,
    select_sample,
)


def test_sample_selection_is_bounded_unique_and_deterministic() -> None:
    inventory = tuple(
        dict.fromkeys((*MANDATORY_WORDS, *(f"word{index}" for index in range(10_000))))
    )

    first = select_sample(inventory)
    second = select_sample(inventory)

    assert len(first) == SAMPLE_SIZE
    assert len(set(first)) == SAMPLE_SIZE
    assert first == second
    assert set(MANDATORY_WORDS) <= set(first)


def test_normalization_matches_default_misaki_token_contract() -> None:
    assert NORMALIZATION_VERSION == "kokoro-phone-normalization-v1"
    assert normalize_kokoro_phone(" ɾæʔ ") == "Tæt"


def test_word_variants_fail_closed_on_unrepresentable_symbols() -> None:
    bundle = WordVariants("test")
    bundle.add("tɛst", "gold:DEFAULT", set("tɛs"))
    bundle.add("t☃", "gold:OTHER", set("tɛs"))

    assert sorted(bundle.phones) == ["tɛst"]
    assert bundle.rejections == ["unrepresentable:☃:gold:OTHER"]


class _FakeExtractor:
    def __init__(self) -> None:
        self.vocab = set("abcdefghijklmnopqrstuvwxyzɛæˈ")

    def extract(self, words: tuple[str, ...]) -> list[WordVariants]:
        bundles: list[WordVariants] = []
        for word in words:
            bundle = WordVariants(word)
            if word == "secretlexeme":
                bundle.add("ˈrɛkord", "gold:NOUN", self.vocab)
                bundle.add("rɛˈkord", "gold:VERB", self.vocab)
            else:
                bundle.add(word, "espeak_fallback:isolated", self.vocab)
            bundles.append(bundle)
        return bundles


def test_full_index_preserves_variants_and_contains_no_plaintext(
    tmp_path: Path,
) -> None:
    database = tmp_path / "kokoro.sqlite3"
    receipt = build_full_index(
        database,
        inventory=("secretlexeme", "test"),
        extractor=_FakeExtractor(),  # type: ignore[arg-type]
        chunk_size=1,
        require_frozen_feasibility=False,
        receipt_destination=tmp_path / "receipt.json",
    )
    index = KokoroGateIndex(database)

    assert receipt["counts"]["word_phone_variants"] == 3
    assert receipt["contains_plaintext_words_or_phones"] is False
    assert index.phone_match("ˈrɛkord") is True
    assert index.phone_match("rɛˈkord") is True
    assert index.phone_match("nonsense") is False
    assert index.source_mask("secretlexeme", "rɛˈkord") == 1
    assert b"secretlexeme" not in database.read_bytes()
