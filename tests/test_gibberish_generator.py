from __future__ import annotations

import json
import unicodedata

import pytest

from earshift_bakeoff import azure_lens_builder as lane
from earshift_bakeoff import gibberish_generator as gib


def test_every_map_symbol_is_classified_as_vowel_consonant_or_mark() -> None:
    """No codepoint the map can emit may fall between the two classes.

    The vowel test is a positive set, so a locale added later could introduce
    a symbol that is neither a listed vowel nor a known consonant and would
    then be silently treated as a consonant — landing in an onset forever and
    never carrying a nucleus. This is the check that would have caught ʋ, c,
    ɟ, ʈ and ɖ being read as vowels in the first place.
    """

    marks = set(gib.NUCLEUS_MARKS) | set(gib.STRESS_MARKS) | {"ʰ", "ʲ", " "}
    unclassified: set[str] = set()
    for table in lane.load_ipa_map().values():
        for row in table.values():
            for char in row["azure_ipa"]:
                if char in gib.VOWELS or char in lane._GENERAL_CONSONANTS:
                    continue
                if char in marks or unicodedata.category(char) == "Mn":
                    continue
                unclassified.add(char)
    # Consonants the lens builder's post-vocalic set never needed to list.
    # Named here so a genuinely new symbol still fails this test.
    known_consonants = set("cqɕɖɟɫɬɭɳʂʈʋʑχ")
    assert unclassified <= known_consonants, (
        f"unclassified map symbols: {sorted(unclassified - known_consonants)}"
    )
    assert not (gib.VOWELS & lane._GENERAL_CONSONANTS)


def test_syllabify_keeps_modifiers_out_of_the_nucleus() -> None:
    # Palatalisation and aspiration ride on the consonant; a syllable built
    # around one has no vowel in it and cannot be pronounced.
    syllables, stress = gib.syllabify("nʲiˈkʰa")
    assert syllables == ["nʲi", "kʰa"]
    assert stress == 1
    for syllable in syllables:
        assert any(gib.is_vowel(char) for char in syllable)


def test_syllabify_reports_no_stress_when_the_string_carries_none() -> None:
    syllables, stress = gib.syllabify("kata")
    assert syllables == ["ka", "ta"]
    assert stress is None


def test_syllabify_gives_trailing_consonants_to_the_last_syllable() -> None:
    syllables, _ = gib.syllabify("plˈæŋks")
    assert len(syllables) == 1
    assert syllables[0].endswith("ŋks")


def test_reduce_collapses_the_nucleus_and_drops_its_length() -> None:
    assert gib._reduce("kaː") == "kə"
    assert gib._reduce("stroː") == "strə"
    # A syllable that somehow arrives with no vowel still yields something
    # pronounceable rather than an empty ph fragment.
    assert gib._reduce("") == gib.SCHWA


@pytest.mark.parametrize("space", sorted(gib.BANKS))
def test_every_banked_syllable_has_a_nucleus_and_no_whitespace(space: str) -> None:
    path, _ = gib.BANKS[space]
    bank = json.loads(path.read_text(encoding="utf-8"))
    for locale, row in bank["locales"].items():
        assert row["syllables"], locale
        vowels = gib.vowels_for(locale, space)
        for entry in row["syllables"]:
            syllable = entry["syllable"]
            assert syllable, locale
            assert " " not in syllable, f"{locale}: {syllable!r} would break the ph attribute"
            assert any(char in vowels for char in syllable), f"{locale}: {syllable!r}"
            assert entry["weight"] >= 1


@pytest.mark.parametrize("space", sorted(gib.BANKS))
def test_bank_covers_every_lens_locale(space: str) -> None:
    """Parity is the ship gate: no language may be offered the lens but not
    this mode, or the product would be more complete in one language than
    another for no reason the user can see."""

    lens_locales = {
        profile["source_locale"]
        for profile in lane.load_azure_profiles().values()
        if profile.get("source_locale")
    }
    assert lens_locales <= gib.supported_locales(space)
    assert len(gib.supported_locales(space)) == 30


def test_adapter_vowels_come_from_the_map_and_lose_no_nucleus() -> None:
    """The ligature problem, pinned.

    misaki writes the English diphthongs as single uppercase letters, so an
    adapter-space vowel set written by hand would miss them and every ligature
    would become an onset consonant that never carries a nucleus. Deriving the
    set from the map is what catches them; this checks the derivation covers
    what the map can actually emit, and that filtering to single codepoints
    throws no nucleus away.
    """

    ligatures = set("AIOQWY")
    assert ligatures <= gib.adapter_vowels("en-US")
    # None of them is a vowel once mapped, so VOWELS — which is correct for
    # the mapped bank — files every one of them as a consonant. That is the
    # whole reason this set is derived rather than reused.
    assert not (ligatures & gib.VOWELS)

    for locale, table in lane.load_ipa_map().items():
        vowels = gib.adapter_vowels(locale)
        assert all(len(symbol) == 1 for symbol in vowels), locale
        for symbol, row in table.items():
            maps_to_vowel = bool(row["azure_ipa"]) and row["azure_ipa"][0] in gib.VOWELS
            if len(symbol) == 1:
                assert (symbol in vowels) is maps_to_vowel, f"{locale}: {symbol!r}"
            elif maps_to_vowel:
                # A dropped multi-codepoint row must not be the only way its
                # nucleus is reachable, or the walk would lose it entirely.
                assert symbol[0] in vowels, f"{locale}: {symbol!r} head is not a vowel"


def test_every_adapter_syllable_survives_the_map() -> None:
    """build_pair maps the lens output, so an unmappable bank symbol would
    surface as a failed request rather than a failed build. The harvest checks
    this before freezing; this is the same check on the frozen artifact."""

    bank = json.loads(gib.ADAPTER_CORES_PATH.read_text(encoding="utf-8"))
    for locale, row in bank["locales"].items():
        table = lane.load_ipa_map()[locale]
        normalization = lane._MAP_NORMALIZATION_BY_LOCALE.get(locale, "NFD")
        for entry in row["syllables"]:
            mapped = lane._map_symbols(
                entry["syllable"], table, context=locale, normalization=normalization
            )
            assert mapped and " " not in mapped, f"{locale}: {entry['syllable']!r}"


def test_the_two_banks_are_distinct_where_the_map_is() -> None:
    """Both banks exist precisely because the map is not the identity.

    If they were the same artifact the adapter one would be dead weight; the
    locales where they differ are the locales where the lens could not see
    mapped syllables at all.
    """

    azure = json.loads(gib.SYLLABLE_CORES_PATH.read_text(encoding="utf-8"))["locales"]
    adapter = json.loads(gib.ADAPTER_CORES_PATH.read_text(encoding="utf-8"))["locales"]
    assert set(azure) == set(adapter)
    differing = {
        locale
        for locale in azure
        if {entry["syllable"] for entry in azure[locale]["syllables"]}
        != {entry["syllable"] for entry in adapter[locale]["syllables"]}
    }
    # en-US is the loudest case: every diphthong is a ligature upstream.
    assert "en-US" in differing
    assert len(differing) >= 20, sorted(differing)


def test_generation_is_deterministic_and_sentence_sensitive() -> None:
    first = gib.build_gibberish("The cat naps.", "en-US")
    again = gib.build_gibberish("The cat naps.", "en-US")
    assert first["ssml_gibberish"] == again["ssml_gibberish"]
    other = gib.build_gibberish("The cat naps!", "en-US")
    assert other["ssml_gibberish"] != first["ssml_gibberish"]


def test_generation_keeps_the_source_syllable_count_per_word() -> None:
    result = gib.build_gibberish("The birch canoe slid on the smooth planks.", "en-US")
    for row in result["words"]:
        source_syllables, _ = gib.syllabify(row["source_phone"])
        built, _ = gib.syllabify(row["gibberish_phone"])
        assert row["syllable_count"] == max(1, len(source_syllables))
        assert len(built) == row["syllable_count"], row


def test_both_sides_are_phoneme_tagged_and_differ_only_in_their_phones() -> None:
    result = gib.build_gibberish("The cat naps.", "en-US")
    neutral, gibberish = result["ssml_neutral"], result["ssml_gibberish"]
    # A pair where one side is tagged and the other is plain text would differ
    # in more than its phones, and the comparison would stop being about them.
    assert neutral.count("<phoneme") == gibberish.count("<phoneme")
    assert neutral.count("<phoneme") == len(result["words"])
    assert neutral != gibberish
    for row in result["words"]:
        assert row["source_phone"] != "" and row["gibberish_phone"] != ""


def test_the_stress_mark_is_withheld_without_a_positive_receipt() -> None:
    """Every voice ignores IPA stress position, so the mark must not ship.

    Requiring a positive verdict rather than the absence of a negative one
    also keeps the deploy container — which has no probe artifact — from
    emitting a mark the local build withholds.
    """

    assert set(lane.load_stress_honour().values()) == {"ignored"}
    result = gib.build_gibberish("The birch canoe slid on the planks.", "en-US")
    assert result["stress_mark_emitted"] is False
    assert "ˈ" not in result["ssml_gibberish"]


def test_stress_timed_table_covers_every_supported_locale() -> None:
    assert set(gib.STRESS_TIMED) == gib.supported_locales()
    assert gib.STRESS_TIMED["en-US"] is True
    assert gib.STRESS_TIMED["es-ES"] is False


def test_reduction_only_runs_where_the_table_says_so() -> None:
    english = gib.build_gibberish("The birch canoe slid on the planks.", "en-US")
    spanish = gib.build_gibberish("El veloz murciélago hindú comía feliz.", "es-ES")
    assert english["vowel_reduction"] is True
    assert spanish["vowel_reduction"] is False


def test_unsupported_locale_and_shape_fail_closed() -> None:
    with pytest.raises(gib.GibberishError):
        gib.build_gibberish("hello", "xx-XX")
    with pytest.raises(gib.GibberishError):
        gib.build_gibberish("hello", "en-US", syllable_shape="freestyle")


def test_empty_text_fails_closed_rather_than_rendering_silence() -> None:
    for text in ("", "   "):
        with pytest.raises(gib.GibberishError):
            gib.build_gibberish(text, "en-US")


def test_match_source_shape_tracks_the_source_syllable() -> None:
    result = gib.build_gibberish(
        "The birch canoe slid on the smooth planks.",
        "en-US",
        syllable_shape="match_source",
    )
    assert result["syllable_shape"] == "match_source"
    matched = total = 0
    for row in result["words"]:
        source_syllables, _ = gib.syllabify(row["source_phone"])
        built, _ = gib.syllabify(row["gibberish_phone"])
        for source, drawn in zip(source_syllables, built):
            total += 1
            matched += gib._is_open(source) is gib._is_open(drawn)
    # Reduction can close an open draw, so this is a strong tendency rather
    # than an invariant; the default strategy does not track shape at all.
    assert matched / total > 0.6


@pytest.fixture(scope="module")
def inventory_words() -> dict[str, str]:
    """Each locale's hand coverage list, which is per-locale text the runtime
    does not otherwise carry. Loaded from the scripts data module the way the
    scripts themselves do, rather than restating thirty sentences here."""

    import sys

    sys.path.insert(0, str(lane.ROOT / "scripts"))
    from lens_language_data_v1 import INVENTORY_WORDS  # type: ignore[import-not-found]

    return INVENTORY_WORDS


@pytest.mark.parametrize("locale", sorted(gib.supported_locales()))
def test_every_locale_generates_a_renderable_pair(
    locale: str, inventory_words: dict[str, str]
) -> None:
    """Parity, checked per language rather than asserted once.

    Each locale must produce a pair from its own words whose phone strings are
    non-empty, whitespace-free (whitespace in a ph attribute is rejected by
    Azure) and different from the source.
    """

    text = " ".join(inventory_words[locale].split()[:8])
    result = gib.build_gibberish(text, locale)
    assert result["locale"] == locale
    assert result["words"]
    for row in result["words"]:
        assert row["gibberish_phone"]
        assert " " not in row["gibberish_phone"]
    joined_source = "".join(row["source_phone"] for row in result["words"])
    joined_built = "".join(row["gibberish_phone"] for row in result["words"])
    assert joined_source != joined_built


@pytest.mark.parametrize("locale", sorted(gib.supported_locales(gib.ADAPTER_SPACE)))
def test_every_locale_can_be_handed_to_a_listener(
    locale: str, inventory_words: dict[str, str]
) -> None:
    """The same parity question asked of the listener path.

    A SourceAnalysis the lens cannot consume would fail at request time in one
    language and not another, which is exactly the asymmetry the mode is not
    allowed to have. Building the analysis is the check: alignment, separator
    count and non-empty phones are what build_pair reads before any rule runs.
    """

    text = " ".join(inventory_words[locale].split()[:8])
    built = gib.gibberish_analysis(text, locale)
    analysis = built["analysis"]
    assert analysis.language_id == locale
    assert len(analysis.words) == len(built["words"])
    assert len(analysis.phone_separators) == len(analysis.words) + 1
    for word in analysis.words:
        assert word.phone and " " not in word.phone
    # compose() is what the engine uses to reassemble a sentence, and it
    # raises on drift rather than returning something subtly misaligned.
    assert analysis.compose([word.phone for word in analysis.words])


def test_plain_gibberish_does_not_read_the_adapter_bank() -> None:
    """The parity guarantee, made structural.

    Plain gibberish shipped before any of this existed and must keep rendering
    the same audio for the same input. The way that breaks is a shared code
    path quietly starting to read the wrong bank, so the check is that the
    source-only path never touches the adapter artifact at all.
    """

    read: list[str] = []
    original = gib.load_syllable_cores

    def watched(space: str = gib.AZURE_SPACE):
        read.append(space)
        return original(space)

    gib.load_syllable_cores = watched  # type: ignore[assignment]
    gib._core.cache_clear()
    try:
        gib.build_gibberish("The quick brown fox jumps", "en-US")
    finally:
        gib.load_syllable_cores = original  # type: ignore[assignment]
        gib._core.cache_clear()
    assert read, "the bank was never read at all"
    assert gib.ADAPTER_SPACE not in read
