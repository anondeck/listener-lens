"""Gibberish mode — a language's own sound with its meaning removed.

The listener lens answers "how does a Portuguese ear re-hear your English?"
This answers a different question: "what does my language sound like to
someone who does not speak it?" Same voice, same rhythm, real syllables,
no recoverable words — the Celentano trick.

It is a second mode, not a listener direction: it needs a source language and
nothing else, and it makes no claim about anybody's perception. The lens
models a cited re-hearing; this removes lexical meaning and keeps the sound.
Those are different claims and the copy must not blur them.

How a pseudoword is built, and what part of it the renderer actually carries:

  * The syllable bank is harvested offline per language and frozen in
    ``rules/gibberish-syllable-cores-v1.json``. Concentrating on that
    language's most common syllables is what keeps the output from drifting
    into a neighbouring language, and it is the parameter that matters most.
  * The typed sentence supplies the skeleton: each source word contributes
    its syllable count and which syllable carried primary stress.
  * Syllable count is audible — it sets the word lengths and therefore the
    rhythm. So is schwa reduction on the stress-timed languages, which is the
    single strongest cue that English sounds English.
  * The primary-stress *mark* is not audible. All thirty voices ignore IPA
    stress position (artifacts/azure-stress-probe-v1), so it is emitted only
    where a voice is receipted as honouring it — today, nowhere. The stress
    index still does real work through reduction; the glyph does not, and is
    gated rather than shipped as decoration.

Deterministic: the RNG is seeded from sha256(locale|text), so one input
always yields one output and the render caches like any other pair.
"""

from __future__ import annotations

import hashlib
import json
import random
from functools import lru_cache
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape, quoteattr

from .config import ROOT

GIBBERISH_LANE_VERSION = "gibberish-lane-v1"
SYLLABLE_CORES_PATH = ROOT / "rules" / "gibberish-syllable-cores-v1.json"
ADAPTER_CORES_PATH = ROOT / "rules" / "gibberish-syllable-cores-adapter-v1.json"

# The two phone spaces a syllable bank can live in.
#
# "azure" is what plain gibberish ships on: syllables harvested from the
# language's own words and already carried through the Azure IPA map, so they
# drop straight into a ph attribute.
#
# "adapter" is the identical harvest with that mapping step left off. It
# exists for one reason — the lens matches its rules *upstream* of the map,
# against the adapter's own alphabet, which is the wider one (misaki writes
# the English diphthongs as ligatures). Mapped syllables are invisible to
# those rules, so gibberish a listener can re-hear has to stay in the
# alphabet the rules are written in. build_pair maps at the end regardless,
# so the audio is reached the same way either route.
#
# Two banks rather than one converted at runtime because the map is lossy in
# the direction that would matter: several adapter symbols share an Azure
# target, so nothing can be walked back.
AZURE_SPACE = "azure"
ADAPTER_SPACE = "adapter"
BANKS = {
    AZURE_SPACE: (SYLLABLE_CORES_PATH, "gibberish-syllable-cores-v1"),
    ADAPTER_SPACE: (ADAPTER_CORES_PATH, "gibberish-syllable-cores-adapter-v1"),
}

# How many of the ranked bank's syllables the generator draws from. The bank
# stores more than this so the size is tunable without a rebuild. Ninety is
# the value the recipe was auditioned at; widening it admits rarer syllables
# and loosens the concentration that holds a language in place.
DEFAULT_CORE_SIZE = 90

# How a slot picks its syllable.
#
#   prefer_open   Vowel-final everywhere but the last slot. The audition
#                 default: this is the rule that was listened to and approved.
#   match_source  Mirror whether the source syllable in that slot was open or
#                 closed.
#
# The difference is measurable rather than a matter of taste. Syllable *count*
# transfers to the pseudoword; syllable *weight* does not, because the bank is
# ranked by frequency and a language's most frequent syllables are its
# lightest. Over the thirty example sentences the generated phone string runs
# to a median 62% of the source's length under prefer_open and 71% under
# match_source, English 43% and 69% — so the rhythm is systematically lighter
# and quicker than the sentence it was built from, most of all in English.
#
# Left at the audited default because which one sounds right is an ears
# question and has not been asked yet. Named and exposed so asking it is a
# parameter change rather than an edit.
DEFAULT_SYLLABLE_SHAPE = "prefer_open"
SYLLABLE_SHAPES = ("prefer_open", "match_source")


class GibberishError(RuntimeError):
    pass


def _bank(space: str) -> tuple[Path, str]:
    try:
        return BANKS[space]
    except KeyError as exc:
        raise GibberishError(f"unknown phone space {space!r}") from exc


# Whether unstressed vowels reduce toward schwa in this language.
#
# This is a typological default, not a measurement: no probe backs it and no
# native speaker has reviewed it. It is separated out and named so it can be
# corrected per language without touching the generator.
#
# Reduction is what makes English sound English, and applying it to a
# syllable-timed language stops that language sounding like itself — so the
# uncertain cases default to False, where the error is a flatter rhythm
# rather than a different language's rhythm.
STRESS_TIMED: dict[str, bool] = {
    "en-US": True, "de-DE": True, "nl-NL": True, "sv-SE": True,
    "nb-NO": True, "ru-RU": True, "bg-BG": True, "pt-BR": True,
    "pt-PT": True, "ca-ES": True,
    "es-ES": False, "es-MX": False, "fr-FR": False, "it-IT": False,
    "ro-RO": False, "el-GR": False, "pl-PL": False, "cs-CZ": False,
    "sk-SK": False, "sl-SI": False, "hr-HR": False, "uk-UA": False,
    "hu-HU": False, "tr-TR": False, "id-ID": False, "ms-MY": False,
    "hi-IN": False, "mr-IN": False, "gu-IN": False, "te-IN": False,
}

# Every vowel the Azure IPA map can emit, enumerated from the map itself.
#
# Stated positively on purpose. The obvious test — "not in the lens builder's
# consonant set" — is wrong here, because that set exists to decide whether a
# segment is post-vocalic and a few gaps cost it nothing. Reused as a vowel
# test it misfiles fourteen consonants as nuclei, ʋ alone in 24 locales, and
# the palatal and retroflex stops c ɟ ʈ ɖ across the Indic ones. Each became a
# standalone "syllable" of pure consonant in those banks.
#
# The vowel inventory is closed and small, so it can be written down; a
# consonant inventory is open-ended and cannot. test_gibberish_generator pins
# every codepoint the map can emit against this set so a new locale cannot
# quietly add a symbol that lands in neither class.
VOWELS = frozenset("aeiouyæøœɐɑɒɔəɚɛɜɨɪɵɯʉʊʌʏ")

# Marks that ride on a nucleus rather than being one. ʰ and ʲ are modifier
# letters, not vowels: treating them as nuclei split Russian tʲ and Hindi kʰ
# into syllables with no vowel in them.
NUCLEUS_MARKS = ("ː", "̃")
STRESS_MARKS = "ˈˌ"

SCHWA = "ə"


def is_vowel(char: str) -> bool:
    """A codepoint that can carry a syllable nucleus, in Azure IPA."""

    return char in VOWELS


@lru_cache(maxsize=64)
def adapter_vowels(locale: str) -> frozenset[str]:
    """The same question asked one step upstream, in *adapter* space.

    The lens matches its rules against the adapter's phones, before the Azure
    IPA map runs; the gibberish bank above is mapped. So a pseudoword that the
    lens is meant to re-hear has to be built from adapter symbols, and that
    alphabet is the wider one — misaki writes the English diphthongs as the
    ligatures A I O Q W Y and carries ᵻ ᵊ, none of which survive the map.

    Derived from each locale's own map rather than written down: a symbol is a
    nucleus exactly when what it maps to opens with one. Enumerating it by
    hand is what put fourteen consonants in the banks the first time, and the
    ligature set is precisely where a hand-written list would go wrong again,
    since none of A I O Q W Y is a vowel once mapped. Per locale because the
    inventories differ sharply — 38 nuclei in en-US against 19 in hi-IN.
    """

    from .azure_lens_builder import load_ipa_map

    try:
        table = load_ipa_map()[locale]
    except KeyError as exc:
        raise GibberishError(f"no Azure IPA map for locale: {locale}") from exc
    return frozenset(
        symbol
        for symbol, row in table.items()
        # One codepoint, because every consumer walks a phone string one
        # codepoint at a time — as _map_symbols itself does, which is why the
        # table's handful of two-codepoint rows (aː, ɛi, ɔʏ) are never looked
        # up there either. Dropping them loses no nucleus: each one's leading
        # codepoint is a vowel row in the same table, verified in the tests.
        if len(symbol) == 1 and row["azure_ipa"] and row["azure_ipa"][0] in VOWELS
    )


def syllabify(
    ipa: str, vowels: frozenset[str] = VOWELS
) -> tuple[list[str], int | None]:
    """Split a phone string into syllables and locate its primary stress.

    Returns the syllables with stress marks stripped, plus the index of the
    syllable the primary mark preceded (None when the string carried none).
    Onset-maximal apart from the final syllable, which keeps the trailing
    consonants because there is no following onset to give them to.

    ``vowels`` selects the phone space: the default is Azure IPA, and callers
    working upstream of the map pass ``adapter_vowels(locale)``.
    """

    syllables: list[str] = []
    stressed_index: int | None = None
    onset = ""
    pending_stress = False
    chars = list(ipa)
    total = len(chars)
    index = 0

    def push(syllable: str) -> None:
        nonlocal stressed_index
        if pending_stress and stressed_index is None:
            stressed_index = len(syllables)
        syllables.append(syllable)

    while index < total:
        char = chars[index]
        if char in STRESS_MARKS:
            pending_stress = char == "ˈ"
            index += 1
            continue
        if char in vowels:
            nucleus = char
            index += 1
            # Diphthongs, length and nasalisation belong to the same nucleus.
            while index < total and (
                chars[index] in vowels or chars[index] in NUCLEUS_MARKS
            ):
                nucleus += chars[index]
                index += 1
            ahead = index
            while (
                ahead < total
                and chars[ahead] not in vowels
                and chars[ahead] not in STRESS_MARKS
            ):
                ahead += 1
            if ahead >= total:
                push(onset + nucleus + "".join(chars[index:ahead]))
                index = ahead
            else:
                push(onset + nucleus)
            onset = ""
            pending_stress = False
        else:
            onset += char
            index += 1
    if onset and syllables:
        syllables[-1] += onset
    return syllables, stressed_index


def _reduce(syllable: str, vowels: frozenset[str] = VOWELS) -> str:
    """Collapse a syllable's nucleus toward schwa, keeping its consonants."""

    out = ""
    seen_nucleus = False
    for char in syllable:
        if char in vowels:
            if not seen_nucleus:
                out += SCHWA
                seen_nucleus = True
            continue
        if char in NUCLEUS_MARKS:
            # Length and nasality belong to the vowel that was just dropped.
            continue
        out += char
    return out or SCHWA


@lru_cache(maxsize=len(BANKS))
def load_syllable_cores(space: str = AZURE_SPACE) -> dict[str, Any]:
    path, bank_id = _bank(space)
    if not path.is_file():
        raise GibberishError(f"{space} gibberish syllable bank is missing")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("bank_id") != bank_id:
        raise GibberishError("unexpected gibberish syllable bank identity")
    return data["locales"]


def supported_locales(space: str = AZURE_SPACE) -> frozenset[str]:
    """Locales with a usable syllable bank.

    Parity is the ship gate for this mode: a language either has a bank built
    from its own words or it is not offered. The builder refuses to write a
    thin bank, so membership here is the same question as "did the harvest
    succeed", asked of the frozen artifact rather than re-derived.
    """

    return frozenset(load_syllable_cores(space))


def vowels_for(locale: str, space: str = AZURE_SPACE) -> frozenset[str]:
    """The nucleus alphabet a phone string in this space is read against."""

    _bank(space)
    return adapter_vowels(locale) if space == ADAPTER_SPACE else VOWELS


def _is_open(syllable: str, vowels: frozenset[str] = VOWELS) -> bool:
    return bool(syllable) and syllable[-1] in vowels


@lru_cache(maxsize=128)
def _core(
    locale: str, core_size: int, space: str = AZURE_SPACE
) -> dict[str, tuple[tuple[str, ...], tuple[int, ...]]]:
    """The ranked bank sliced to `core_size`, split by syllable shape.

    Open syllables join more smoothly than closed ones, so a slot that has a
    neighbour after it prefers them; without that the pseudowords collect
    consonant pile-ups at their seams that no real word would have.
    """

    cores = load_syllable_cores(space)
    row = cores.get(locale)
    if row is None:
        raise GibberishError(f"no {space} gibberish syllable bank for {locale}")
    vowels = vowels_for(locale, space)
    ranked = row["syllables"][:core_size]
    syllables = tuple(entry["syllable"] for entry in ranked)
    weights = tuple(entry["weight"] for entry in ranked)
    pools: dict[str, tuple[tuple[str, ...], tuple[int, ...]]] = {
        "any": (syllables, weights)
    }
    for name, wanted_open in (("open", True), ("closed", False)):
        pairs = [
            (syllable, weight)
            for syllable, weight in zip(syllables, weights, strict=True)
            if _is_open(syllable, vowels) is wanted_open
        ]
        # A bank can be all one shape — Italian's is almost entirely open — so
        # an empty pool falls back to the whole core rather than failing.
        pools[name] = (
            (tuple(s for s, _ in pairs), tuple(w for _, w in pairs))
            if pairs
            else (syllables, weights)
        )
    return pools


def _voice_for(locale: str, space: str = AZURE_SPACE) -> str:
    cores = load_syllable_cores(space)
    row = cores.get(locale)
    if row is None:
        raise GibberishError(f"no {space} gibberish syllable bank for {locale}")
    return row["voice"]


def _pseudowords(
    text: str,
    locale: str,
    *,
    space: str,
    core_size: int,
    syllable_shape: str,
) -> tuple[str, list[dict[str, Any]], bool, bool]:
    """Draw one pseudoword per typed word, in the requested phone space.

    The skeleton is the same either way — the source sentence supplies each
    word's syllable count and stress position, the bank supplies the syllables
    — so the two spaces differ in exactly two places: whether the source phone
    is mapped before being read, and which alphabet counts as a nucleus.

    Returns the normalized text, the per-word rows, and the two per-locale
    switches the caller reports.
    """

    from .azure_lens_builder import (
        AzureLensBuilderError,
        _adapter_for,
        _map_symbols,
        _MAP_NORMALIZATION_BY_LOCALE,
        _normalize_source_text,
        load_ipa_map,
        load_stress_honour,
    )
    from .bilingual_vowel_engine import BilingualVowelEngineError

    if locale not in supported_locales(space):
        raise GibberishError(f"unsupported gibberish locale {locale!r}")
    if syllable_shape not in SYLLABLE_SHAPES:
        raise GibberishError(f"unknown syllable shape {syllable_shape!r}")
    normalized_text = _normalize_source_text(locale, text)
    adapter = _adapter_for(locale, None)
    try:
        analysis = adapter.analyze(normalized_text)
    except BilingualVowelEngineError as exc:
        raise GibberishError(f"cannot read this sentence: {exc}") from exc

    table = load_ipa_map()[locale]
    normalization = _MAP_NORMALIZATION_BY_LOCALE.get(locale, "NFD")
    pools = _core(locale, core_size, space)
    vowels = vowels_for(locale, space)
    reduce_unstressed = STRESS_TIMED.get(locale, False)
    # Same receipts the lens reads, but requiring a positive verdict rather
    # than the absence of a negative one. The probe artifact is not copied
    # into the deploy container, where load_stress_honour() then returns an
    # empty map — so "not ignored" would emit the mark in production and
    # withhold it locally, giving one input two different SSML strings and two
    # different cache keys. No receipt means no mark.
    stress_honoured = load_stress_honour().get(locale) == "honoured"

    # A local RNG, not the module-global one: the service is threaded, and
    # seeding the global generator lets two concurrent requests interleave
    # draws and produce output neither of them would produce alone — which
    # would quietly break the determinism the cache depends on.
    seed = int(
        hashlib.sha256(f"{locale}|{normalized_text}".encode("utf-8")).hexdigest(), 16
    )
    rng = random.Random(seed)

    rows: list[dict[str, Any]] = []
    for word in analysis.words:
        if space == ADAPTER_SPACE:
            # Already the alphabet the bank and the lens both speak.
            source_ph = word.phone
        else:
            try:
                source_ph = _map_symbols(
                    word.phone, table, context=f"gibberish {word.source!r}",
                    normalization=normalization,
                )
            except AzureLensBuilderError as exc:
                raise GibberishError(str(exc)) from exc
        source_syllables, stressed_index = syllabify(source_ph, vowels)
        count = max(1, len(source_syllables))
        built: list[str] = []
        for position in range(count):
            if syllable_shape == "match_source":
                source_syllable = (
                    source_syllables[position]
                    if position < len(source_syllables)
                    else ""
                )
                pool = pools["open" if _is_open(source_syllable, vowels) else "closed"]
            elif position < count - 1:
                pool = pools["open"]
            else:
                pool = pools["any"]
            drawn = rng.choices(pool[0], pool[1], k=1)[0]
            if (
                reduce_unstressed
                and stressed_index is not None
                and position != stressed_index
            ):
                drawn = _reduce(drawn, vowels)
            built.append(drawn)
        token = "".join(built)
        if stress_honoured and stressed_index is not None and stressed_index < len(built):
            offset = sum(len(built[index]) for index in range(stressed_index))
            for index in range(offset, len(token)):
                if token[index] in vowels:
                    token = token[:index] + "ˈ" + token[index:]
                    break
        if not token:
            raise GibberishError(f"generated an empty pseudoword for {word.source!r}")
        rows.append({
            "word_index": word.word_index,
            "written": word.source,
            "source_phone": source_ph,
            "gibberish_phone": token,
            "syllable_count": count,
            "stressed_syllable": stressed_index,
        })

    if not rows:
        raise GibberishError("no pronounceable words in this text")
    # The whole point of the mode is that the two sides differ. Identical
    # phone strings would render identical audio and the pair would be a lie;
    # it cannot happen with a bank of this size, so it is a guard, not a case.
    if all(row["source_phone"] == row["gibberish_phone"] for row in rows):
        raise GibberishError("gibberish matched the source exactly")
    return normalized_text, rows, reduce_unstressed, stress_honoured


def gibberish_analysis(
    text: str,
    locale: str,
    *,
    core_size: int = DEFAULT_CORE_SIZE,
    syllable_shape: str = DEFAULT_SYLLABLE_SHAPE,
) -> dict[str, Any]:
    """The same pseudowords, dressed as a source reading for the lens.

    Plain gibberish asks what a language sounds like with its meaning gone.
    Handing that to a listener asks the next question: what does *that* sound
    like to an ear tuned to some other language? The two are worth putting
    side by side only if it is demonstrably the same nonsense both times, so
    nothing is redrawn per listener — the seed is the source and the text, as
    it always was, and the listener only ever re-hears what was already there.

    A SourceAnalysis is exactly what build_pair asks its G2P for, so a
    prepared one substitutes for the reading of a sentence that was never
    typed. The words keep their real written forms: they are what the page
    shows and what decides the sentence's contour, and only the phones are
    invented.
    """

    from .bilingual_vowel_engine import SourceAnalysis, SourceWord

    normalized_text, rows, reduce_unstressed, stress_honoured = _pseudowords(
        text,
        locale,
        space=ADAPTER_SPACE,
        core_size=core_size,
        syllable_shape=syllable_shape,
    )
    words = tuple(
        SourceWord(
            word_index=row["word_index"],
            source=row["written"],
            phone=row["gibberish_phone"],
        )
        for row in rows
    )
    analysis = SourceAnalysis(
        language_id=locale,
        normalized_text=normalized_text,
        source_phonemes=" ".join(word.phone for word in words),
        words=words,
        # compose() wants one more separator than there are words: the pair
        # that brackets the sentence, plus the space between each two.
        phone_separators=("",) + (" ",) * (len(words) - 1) + ("",),
    )
    return {
        "lane_version": GIBBERISH_LANE_VERSION,
        "locale": locale,
        "normalized_text": normalized_text,
        "analysis": analysis,
        "words": rows,
        "core_size": core_size,
        "syllable_shape": syllable_shape,
        "vowel_reduction": reduce_unstressed,
        "stress_mark_emitted": stress_honoured,
    }


def build_gibberish(
    text: str,
    locale: str,
    *,
    voice: str | None = None,
    core_size: int = DEFAULT_CORE_SIZE,
    syllable_shape: str = DEFAULT_SYLLABLE_SHAPE,
) -> dict[str, Any]:
    """A neutral/gibberish SSML pair for typed text in one language.

    Both sides carry a per-word ``<phoneme alphabet="ipa">`` tag, exactly as
    the lens pair does. A pair where one side is tagged and the other is plain
    text would differ in more than its phones, and the comparison would stop
    being about the phones.
    """

    normalized_text, rows, reduce_unstressed, stress_honoured = _pseudowords(
        text,
        locale,
        space=AZURE_SPACE,
        core_size=core_size,
        syllable_shape=syllable_shape,
    )
    selected_voice = voice or _voice_for(locale)

    def ssml(side: str) -> str:
        pieces: list[str] = []
        for row in rows:
            phone = row["source_phone"] if side == "neutral" else row["gibberish_phone"]
            if " " in phone:
                raise GibberishError(
                    f"ph attribute contains whitespace for {row['written']!r}"
                )
            pieces.append(
                f"<phoneme alphabet=\"ipa\" ph={quoteattr(phone)}>"
                f"{escape(row['written'])}</phoneme>"
            )
        final = "?" if normalized_text.strip().endswith("?") else "."
        body = " ".join(pieces) + final
        return (
            '<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" '
            f'xml:lang="{locale}"><voice name="{selected_voice}">{body}</voice></speak>'
        )

    return {
        "lane_version": GIBBERISH_LANE_VERSION,
        "locale": locale,
        "voice": selected_voice,
        "normalized_text": normalized_text,
        "ssml_neutral": ssml("neutral"),
        "ssml_gibberish": ssml("gibberish"),
        "words": rows,
        "core_size": core_size,
        "syllable_shape": syllable_shape,
        "vowel_reduction": reduce_unstressed,
        "stress_mark_emitted": stress_honoured,
        "api_calls_made": 0,
    }
