#!/usr/bin/env python3
"""Harvest the per-language syllable bank the gibberish mode draws from.

Gibberish keeps a language's sound and drops its meaning. What holds it in
place — what stops Spanish gibberish drifting into Italian — is drawing every
syllable from a tight bank of that language's *most ordinary* syllables. So
the bank is harvested from ordinary words, phonemised by that language's own
converter, mapped through the same Azure IPA table the lens uses, syllabified
by the runtime's own splitter, and ranked by frequency.

Run offline; the result is frozen into rules/gibberish-syllable-cores-v1.json
and the runtime only reads it. Keeping it a build step means the request path
carries no word list, no wordfreq, and no per-locale cold start, and it means
the bank is a reviewable object rather than something re-derived per process.

Where the words come from, and why it is not one source:

  wordfreq covers 27 of the 30 locales. It does not cover Marathi, Telugu or
  Gujarati, and — this is the trap — it does not say so. `top_n_list("te")`
  silently returns *English*, and "mr" silently returns Hindi, so the obvious
  implementation would have built the Telugu bank out of English words and
  shipped a language that could not possibly sound like Telugu. The mapping
  here is explicit per locale and never falls back.

  For those three the words come from espeak-ng's own pronunciation
  dictionaries, which hold roughly 1,900 real words each in the language's
  own script. That is a weaker source than a frequency corpus and is recorded
  as such in the artifact: espeak's dictionaries are exception lists plus
  common words, so they are not frequency-ranked and they over-represent the
  irregular. It is nonetheless the difference between a bank of ~65 syllables
  drawn from a 30-word coverage list and one of ~1,300 drawn from ordinary
  vocabulary, and the syllables are unambiguously that language's own.

Parity is the ship gate: a locale that cannot produce a real bank is not
written, and the mode is not offered for it. MIN_BANK_SYLLABLES is what
"real" means here.

Two banks, one recipe, selected by --space:

  azure     (default) syllables carried through the Azure IPA map, ready to
            drop into a ph attribute. What plain gibberish ships on.
  adapter   the same harvest with the mapping step left off, for gibberish
            that a listener lens has to be able to re-hear: the lens matches
            its rules upstream of that map, so mapped syllables are invisible
            to it. Each syllable is still mapped once here — not to store it,
            but to prove it *can* be, since build_pair will do exactly that
            downstream and an unmapped symbol would surface on a request
            rather than on a build.

Writes rules/gibberish-syllable-cores-v1.json
   and rules/gibberish-syllable-cores-adapter-v1.json.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import unicodedata
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from earshift_bakeoff.azure_lens_builder import (  # noqa: E402
    _MAP_NORMALIZATION_BY_LOCALE,
    AzureLensBuilderError,
    _adapter_for,
    _map_symbols,
    load_ipa_map,
)
from earshift_bakeoff.gibberish_generator import (  # noqa: E402
    ADAPTER_SPACE,
    AZURE_SPACE,
    BANKS,
    syllabify,
    vowels_for,
)
from discover_surface_symbols_v1 import CORPUS_LANGUAGE, SENTENCE_COVERAGE  # noqa: E402
from lens_language_data_v1 import AZURE_VOICE, INVENTORY_WORDS, LOCALES  # noqa: E402

ACCEPTANCE_PATH = REPO / "artifacts" / "azure-matrix-probe-v1" / "receipts.json"

# Ranked entries kept per locale. The runtime slices the top ~90 of these; the
# surplus is stored so core size stays tunable without a rebuild.
BANK_SIZE = 200
CORPUS_WORDS = 2000

# Below this many distinct syllables the harvest has not found a language, it
# has found a word list. The three corpus-less locales sat at 61-70 on the
# hand coverage lists, which is what sent this script looking for a better
# source; every locale now clears it by an order of magnitude.
MIN_BANK_SYLLABLES = 200

# Script ranges for the locales served from espeak's dictionaries. Matching on
# the script is what separates the language's own words from the ASCII noise
# in a compiled dictionary file.
DICTIONARY_SCRIPT = {
    "mr-IN": r"[ऀ-ॿ]",
    "te-IN": r"[ఀ-౿]",
    "gu-IN": r"[઀-૿]",
}


def espeak_data_path() -> Path:
    """Where this machine's espeak-ng keeps its dictionaries."""

    output = subprocess.run(
        ["espeak-ng", "--version"], capture_output=True, text=True, check=True
    ).stdout
    match = re.search(r"Data at:\s*(\S+)", output)
    if not match:
        raise SystemExit("could not read the espeak-ng data path from --version")
    return Path(match.group(1))


def dictionary_words(locale: str) -> list[str]:
    path = espeak_data_path() / f"{locale.split('-')[0]}_dict"
    if not path.is_file():
        raise SystemExit(f"espeak dictionary missing for {locale}: {path}")
    raw = path.read_bytes().decode("utf-8", "ignore")
    return sorted(set(re.findall(DICTIONARY_SCRIPT[locale] + r"{2,}", raw)))


def source_words(locale: str) -> tuple[list[str], str]:
    """Ordinary words of the language, and where they came from."""

    hand = (INVENTORY_WORDS[locale] + " " + SENTENCE_COVERAGE.get(locale, "")).split()
    language = CORPUS_LANGUAGE.get(locale)
    if language is None:
        return dictionary_words(locale) + hand, "espeak_dictionary"
    from wordfreq import top_n_list

    return hand + top_n_list(language, CORPUS_WORDS), "wordfreq"


def accepted_symbols(locale: str) -> set[str]:
    """Symbols with an acceptance receipt on this locale's own voice."""

    data = json.loads(ACCEPTANCE_PATH.read_text(encoding="utf-8"))
    return {
        row["azure_ipa"]
        for row in data["locales"].get(locale, {}).values()
        if row["probe"] == "accepted"
    }


def harvest(locale: str, space: str = AZURE_SPACE) -> dict:
    words, provenance = source_words(locale)
    adapter = _adapter_for(locale, None)
    normalization = _MAP_NORMALIZATION_BY_LOCALE.get(locale, "NFD")
    table = load_ipa_map()[locale]
    vowels = vowels_for(locale, space)
    language = CORPUS_LANGUAGE.get(locale)
    if language:
        from wordfreq import zipf_frequency

    counts: Counter[str] = Counter()
    unreadable = 0
    vowelless = 0
    for word in words:
        # Punctuation never survives into a phone string, and a token carrying
        # it is a dictionary artefact rather than a word.
        if any(unicodedata.category(char).startswith("P") for char in word):
            continue
        try:
            rows = adapter.analyze(word).words
            phones = "".join(
                row.phone
                if space == ADAPTER_SPACE
                else _map_symbols(
                    row.phone, table, context="gibberish harvest",
                    normalization=normalization,
                )
                for row in rows
            )
        except (AzureLensBuilderError, Exception):  # noqa: BLE001
            unreadable += 1
            continue
        syllables, _ = syllabify(phones, vowels)
        # A frequency corpus is ranked; a dictionary is not, so those three
        # locales weight every word equally and rely on repetition across
        # ~1,900 words to surface the common syllables.
        weight = max(1, int(zipf_frequency(word, language))) if language else 1
        for syllable in syllables:
            if not any(char in vowels for char in syllable):
                # No nucleus, so not a syllable: an aspirated or palatalised
                # consonant that the splitter had nothing to attach to.
                vowelless += 1
                continue
            counts[syllable] += weight

    receipts = accepted_symbols(locale)
    ranked: list[dict] = []
    unreceipted = 0
    for syllable, weight in counts.most_common():
        if len(ranked) >= BANK_SIZE:
            break
        # Every symbol the bank can emit must already be receipted as
        # renderable on this voice. The syllables come from this locale's own
        # mapped phones, so this should hold by construction; checking it is
        # what turns "should" into "does" before the bank is frozen.
        #
        # An adapter-space syllable has to be carried through the map first,
        # because that is what build_pair will do to it downstream. Mapping
        # here is therefore two checks in one: the receipt, and the fact that
        # the symbol has a map row at all — an unmapped one would raise at
        # render time, on a request, rather than here on a build.
        if space == ADAPTER_SPACE:
            try:
                probe = _map_symbols(
                    syllable, table, context="gibberish bank",
                    normalization=normalization,
                )
            except AzureLensBuilderError:
                unreceipted += 1
                continue
        else:
            probe = syllable
        if not _covered_by_receipts(probe, receipts):
            unreceipted += 1
            continue
        ranked.append({"syllable": syllable, "weight": weight})

    total = sum(counts.values())
    top = sum(entry["weight"] for entry in ranked[:90])
    return {
        "voice": AZURE_VOICE[locale],
        "word_source": provenance,
        "word_count": len(words),
        "distinct_syllables": len(counts),
        "unreadable_words": unreadable,
        "vowelless_fragments_dropped": vowelless,
        "unreceipted_syllables_dropped": unreceipted,
        "top90_token_share": round(top / total, 4) if total else 0.0,
        "syllables": ranked,
    }


def _covered_by_receipts(syllable: str, receipts: set[str]) -> bool:
    """Whether every codepoint run of the syllable has an acceptance receipt.

    Receipts are per mapped symbol, and a symbol can be more than one
    codepoint (a length mark or a tilde rides with its vowel), so the check
    walks the syllable greedily against the receipted set rather than
    per character.
    """

    index = 0
    while index < len(syllable):
        for span in (3, 2, 1):
            if syllable[index : index + span] in receipts:
                index += span
                break
        else:
            return False
    return True


def main() -> None:
    args = sys.argv[1:]
    space = next(
        (arg.split("=", 1)[1] for arg in args if arg.startswith("--space=")),
        AZURE_SPACE,
    )
    if space not in BANKS:
        raise SystemExit(f"--space must be one of {', '.join(BANKS)}")
    out_path, bank_id = BANKS[space]
    selected = [arg for arg in args if arg in LOCALES] or list(LOCALES)
    report: dict = {"bank_id": bank_id, "locales": {}}
    if out_path.is_file():
        report = json.loads(out_path.read_text(encoding="utf-8"))
        report.setdefault("locales", {})

    refused: list[str] = []
    for locale in selected:
        row = harvest(locale, space)
        if row["distinct_syllables"] < MIN_BANK_SYLLABLES or len(row["syllables"]) < 90:
            refused.append(locale)
            report["locales"].pop(locale, None)
            print(
                f"{locale:<7} REFUSED distinct={row['distinct_syllables']} "
                f"kept={len(row['syllables'])} ({row['word_source']})",
                flush=True,
            )
            continue
        report["locales"][locale] = row
        print(
            f"{locale:<7} kept={len(row['syllables']):>3} "
            f"distinct={row['distinct_syllables']:>4} "
            f"top90={row['top90_token_share']:.0%} "
            f"unreadable={row['unreadable_words']:>4} "
            f"vowelless={row['vowelless_fragments_dropped']:>4} "
            f"{row['word_source']}",
            flush=True,
        )

    report["bank_size"] = BANK_SIZE
    report["min_bank_syllables"] = MIN_BANK_SYLLABLES
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"\n{len(report['locales'])} locales banked; {len(refused)} refused")
    if refused:
        print(f"refused: {' '.join(refused)}")
    print(f"written to {out_path}")


if __name__ == "__main__":
    main()
