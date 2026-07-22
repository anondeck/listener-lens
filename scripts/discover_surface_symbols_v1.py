#!/usr/bin/env python3
"""Record the surface symbol inventory each locale's G2P can actually emit.

The phonemic inventory says what a language contrasts; it does not say what
the adapter puts in a phone string. espeak emits allophones (Spanish /b d ɡ/
surface as β ð ɣ), ligatures (ʧ for tʃ), shorthand capitals (A I O W Y), and
occasional artifacts (a bare "." from the Hindi and Gujarati voices). The
Azure IPA map has to cover exactly this set, so it is discovered once from
the coverage word lists and committed rather than recomputed at build time.

Writes artifacts/lens-surface-symbols-v1.json.
"""

from __future__ import annotations

import json
import sys
import unicodedata
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from lens_language_data_v1 import (  # noqa: E402
    ESPEAK_LANGUAGE,
    INVENTORY_WORDS,
    LOCALES,
)

OUT_PATH = REPO / "artifacts" / "lens-surface-symbols-v1" / "symbols.json"

# Adapters that are not espeak. These keep their pinned Kokoro-era G2P so the
# validated en/pt pair stays byte-for-byte unchanged.
MISAKI_LOCALES = {"en-US", "pt-BR"}

STRESS_AND_LENGTH = set("ˈˌː")


# wordfreq language for each locale. The original hand-written coverage lists
# averaged 28 words, which is far too few to surface an inventory: Catalan's
# list contains no /ɲ/ or /ʒ/ and Turkish's no /ʤ/, so those phones never
# reached the map and ordinary words like "llibre" and "Cocuk" failed closed
# at build time. A frequency corpus finds them because it contains the
# ordinary words themselves.
CORPUS_LANGUAGE = {
    "en-US": "en", "pt-BR": "pt", "pt-PT": "pt", "es-ES": "es", "es-MX": "es",
    "ca-ES": "ca", "fr-FR": "fr", "it-IT": "it", "ro-RO": "ro", "de-DE": "de",
    "nl-NL": "nl", "sv-SE": "sv", "nb-NO": "nb", "pl-PL": "pl", "cs-CZ": "cs",
    "sk-SK": "sk", "hr-HR": "hr", "sl-SI": "sl", "bg-BG": "bg", "ru-RU": "ru",
    "uk-UA": "uk", "el-GR": "el", "hu-HU": "hu", "tr-TR": "tr", "id-ID": "id",
    "ms-MY": "ms", "hi-IN": "hi",
}
CORPUS_SIZE = 3000


# Pangrams and sample sentences. A frequency list is ranked by how often a
# word occurs, so it under-covers exactly the phones that live in rare and
# borrowed words: 3,000 Croatian words contain no /ʤ/ and 3,000 Norwegian no
# /w/. Pangrams are built for the opposite property. They also give the three
# locales with no wordfreq corpus something wider than a 28-word hand list.
SENTENCE_COVERAGE = {
    "hr-HR": "Gojazni đačić s biciklom drži hmelj i finu vatu u džepu.",
    "nb-NO": "Vår sære Zulu fra badeøya spilte jo whist og quickstep.",
    "gu-IN": "આજે હવામાન ખૂબ સરસ છે અને પક્ષીઓ ખુશીથી ગાય છે. જ્ઞાન ચિત્ર વૃક્ષ ઝરણું",
    "mr-IN": "आज हवामान खूप छान आहे आणि पक्षी आनंदाने गात आहेत. ज्ञान चित्र वृक्ष झरा",
    "te-IN": "ఈరోజు వాతావరణం చాలా బాగుంది మరియు పక్షులు పాడుతున్నాయి. జ్ఞానం చిత్రం వృక్షం",
}


def coverage_words(locale: str) -> str:
    """Hand list plus a frequency corpus where one exists.

    The hand list is kept rather than replaced: it was chosen to hit specific
    contrasts and still does. Gujarati, Marathi, and Telugu have no wordfreq
    corpus, so those three keep hand-list coverage only and remain the
    locales most likely to hold a gap.
    """

    words = INVENTORY_WORDS[locale] + " " + SENTENCE_COVERAGE.get(locale, "")
    language = CORPUS_LANGUAGE.get(locale)
    if language is None:
        return words
    from wordfreq import top_n_list

    return words + " " + " ".join(top_n_list(language, CORPUS_SIZE))


CHUNK = 40


def _phonemize(locale: str, text: str) -> str:
    if locale == "en-US":
        from misaki.en import G2P
        from misaki.espeak import EspeakFallback

        g2p = G2P(british=False, fallback=EspeakFallback(british=False))
        phones, _ = g2p(text)
        return phones
    if locale == "pt-BR":
        from earshift_bakeoff.bilingual_vowel_engine import PortugueseMisakiAdapter

        adapter = PortugueseMisakiAdapter.load(voice_id="pf_dora")
        return " ".join(word.phone for word in adapter.analyze(text).words)
    from earshift_bakeoff.azure_source_adapters import _PHONE_REPAIRS
    from misaki.espeak import EspeakG2P

    g2p = EspeakG2P(language=ESPEAK_LANGUAGE[locale])
    phones, _ = g2p(text)
    # The runtime adapter repairs a locale's known converter defects before
    # anything downstream sees the phones, so discovery has to apply the same
    # repairs or the map is built for symbols the lane no longer emits — and
    # the ones it now does emit go unmapped and fail closed. Imported from the
    # adapter rather than restated so the two cannot drift.
    for wrong, right in _PHONE_REPAIRS.get(locale, ()):
        phones = phones.replace(wrong, right)
    return phones


def discover(locale: str) -> tuple[list[str], int]:
    """Phonemize the corpus in chunks, skipping the ones that will not align.

    Some words break the adapters' one-phone-group-per-word invariant — the
    same failure that made three locales unrenderable on an ordinary sentence.
    A whole-corpus call therefore fails outright and yields nothing, so the
    corpus is chunked and a failing chunk is dropped rather than losing the
    other 2,960 words with it. Dropped chunks are counted, not hidden.
    """

    # Not str.isalpha(): an Indic word carries its vowels as combining marks,
    # which are category Mn rather than letters, so isalpha() is False for
    # ordinary words like కావాలి and silently discarded all 25 Telugu entries.
    # Require a letter somewhere and no digits instead.
    words = [
        word
        for word in coverage_words(locale).split()
        if any(ch.isalpha() for ch in word) and not any(ch.isdigit() for ch in word)
    ]
    symbols: set[str] = set()
    skipped = 0
    for start in range(0, len(words), CHUNK):
        chunk = words[start : start + CHUNK]
        try:
            phones = _phonemize(locale, " ".join(chunk))
        except Exception:
            # Degrade to one word at a time rather than losing the chunk. The
            # Indic hand lists are a single chunk, so a phrase-level failure
            # there discarded the whole inventory and reported zero symbols
            # for Telugu — a silent regression that looks like a clean run.
            phones = ""
            for word in chunk:
                try:
                    phones += " " + _phonemize(locale, word)
                except Exception:
                    skipped += 1
        symbols.update(symbol for symbol in phones if symbol != " ")
    return sorted(symbols), skipped


def classify(symbol: str) -> str:
    if symbol in STRESS_AND_LENGTH:
        return "structural"
    category = unicodedata.category(symbol)
    if category == "Mn":
        return "combining"
    if category.startswith("L") or symbol in "ʰʲ":
        return "phone"
    return "artifact"


def main() -> None:
    # Re-running one locale beats re-running thirty: the sweep takes about
    # twelve minutes and most fixes touch a single language.
    only = set(sys.argv[1:])
    previous_report = (
        json.loads(OUT_PATH.read_text(encoding="utf-8"))["locales"]
        if OUT_PATH.is_file()
        else {}
    )
    report: dict[str, object] = {"schema_version": 1, "locales": dict(previous_report)}
    for locale in LOCALES:
        if only and locale not in only:
            continue
        symbols, skipped = discover(locale)
        rows = {symbol: classify(symbol) for symbol in symbols}
        artifacts = sorted(s for s, kind in rows.items() if kind == "artifact")
        report["locales"][locale] = {
            "adapter": "misaki" if locale in MISAKI_LOCALES else "espeak",
            "symbol_count": len(symbols),
            "symbols": symbols,
            "artifacts": artifacts,
            "skipped_chunks": skipped,
        }
        flag = f"  artifacts={artifacts}" if artifacts else ""
        drop = f"  skipped={skipped}" if skipped else ""
        print(f"{locale:<6} n={len(symbols):>3}{drop}{flag}")
    # Discovery only ever widens: the corpus is a superset of the hand list,
    # so losing a symbol means the run broke rather than that the phone went
    # away. Two separate bugs — a chunk-level failure and an isalpha() filter
    # that discarded every Indic word — each reported a clean exit while
    # emptying a locale's inventory, so this is checked rather than trusted.
    # A declared repair is the one legitimate way a symbol leaves an
    # inventory: rewriting the Gujarati "r." to ɭ means bare r genuinely stops
    # appearing. Only the characters a repair consumes are excused, so the
    # guard still catches a run that broke.
    from earshift_bakeoff.azure_source_adapters import _PHONE_REPAIRS

    if OUT_PATH.is_file():
        previous = json.loads(OUT_PATH.read_text(encoding="utf-8"))["locales"]
        shrunk = {}
        for locale, row in report["locales"].items():
            if locale not in previous:
                continue
            excused = {
                char
                for wrong, _ in _PHONE_REPAIRS.get(locale, ())
                for char in wrong
            }
            lost = set(previous[locale]["symbols"]) - set(row["symbols"]) - excused
            if lost:
                shrunk[locale] = sorted(lost)
        if shrunk:
            for locale, lost in shrunk.items():
                print(f"LOST {locale}: {' '.join(lost)}")
            raise SystemExit(
                f"{len(shrunk)} locale(s) lost symbols; refusing to overwrite "
                "the artifact with a narrower inventory"
            )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"written to {OUT_PATH}")


if __name__ == "__main__":
    main()
