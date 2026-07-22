#!/usr/bin/env python3
"""Audit the neutral track: does each locale's G2P produce its own phonemes?

Every listener-lens rule carries a per-voice audibility receipt. The neutral
side — "as you say it" — carries nothing: it trusts espeak, which no native
speaker has audited language by language.

One check is mechanical and needs no native speaker. A language's contrastive
phonemes are known (PHONEMIC_INVENTORY). If a phoneme the language contrasts
never appears anywhere in a 3,000-word frequency corpus plus pangrams, the
converter is not producing that category at all — which is exactly the
confirmed Gujarati defect, where the retroflex lateral ɭ is emitted as r.

What this can and cannot find:

  * finds   a phoneme the converter never produces (a whole category missing)
  * misses  a phoneme produced in the wrong places, a wrong vowel in a
            particular word, a missed assimilation. Those need a native ear.

So a clean report here is not a clean bill of health; it means no category is
wholly absent. Absences are ranked, not asserted as defects: a genuinely rare
phoneme can be missing from a corpus honestly, so each hit names the phoneme
and lets a human judge.

Writes artifacts/neutral-g2p-audit-v1/report.json.
"""

from __future__ import annotations

import json
import sys
import unicodedata
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from discover_surface_symbols_v1 import coverage_words  # noqa: E402
from lens_language_data_v1 import (  # noqa: E402
    ESPEAK_LANGUAGE,
    LOCALES,
    PHONEMIC_INVENTORY,
)

OUT_PATH = REPO / "artifacts" / "neutral-g2p-audit-v1" / "report.json"

# Below this much phonemized text, "never produced" means "the sample was too
# small", not "the converter cannot produce it". Three locales (Gujarati,
# Marathi, Telugu) have no wordfreq corpus and fall back to a hand list of
# roughly thirty words — about 310 phone characters against ~25,000 for every
# other locale. An early run read that as nine missing Telugu phonemes,
# including two this session had already watched Telugu produce correctly.
# Those locales are reported inconclusive rather than accused.
MIN_CORPUS_CHARS = 5000

# Spellings the converter may use for the same category. Notational variance
# is not a defect, and counting it as one buries the real signal: an early
# run reported 34 absences across three Indic locales, nearly all of them
# this — espeak writes breathy voice with ʰ rather than ʱ, prefers v to ʋ and
# ɾ to r, and renders the Indic palatal affricates as the palatal stops c/ɟ.
# These correspondences were read off actual converter output, not assumed.
EQUIVALENT = {
    "tʃ": ("ʧ", "c"), "dʒ": ("ʤ", "ɟ", "z"), "ts": ("ʦ",), "dz": ("ʣ",),
    "tɕ": ("ʨ", "ʧ"), "dʑ": ("ʥ", "ʤ"), "tʂ": ("ʈʂ",), "dʐ": ("ɖʐ",),
    "eɪ": ("A",), "aɪ": ("I",), "oʊ": ("O",), "aʊ": ("W",), "ɔɪ": ("Y",),
    "ɜ": ("ɚ",), "ə": ("ᵊ",), "ɪ": ("ᵻ",),
    # A dental diacritic is often simply absent from the converter's output.
    "t̪": ("t",), "d̪": ("d",), "t̪ʰ": ("tʰ",), "d̪ʱ": ("dʰ", "dʱ"),
    # Breathy voice: ʰ (modifier h) stands in for ʱ (modifier h with hook).
    "bʱ": ("bʰ",), "ɖʱ": ("ɖʰ",), "ɡʱ": ("ɡʰ",), "dʒʱ": ("dʒʰ", "ɟʰ"),
    "tʃʰ": ("cʰ", "ʧʰ"),
    # Rhotic and approximant notation.
    "ʋ": ("v", "w"), "r": ("ɾ", "ɽ"), "ɽ": ("ɾ", "r"),
    # Sibilant notation.
    "ʂ": ("ʃ",), "ʐ": ("ʒ",),
    # Length written as a separate mark rather than a composed vowel.
    "aː": ("aː", "ɑː"), "ɔː": ("ɔː", "oː"), "ɛː": ("ɛː", "eː"),
}


def phonemize(locale: str) -> str:
    text = coverage_words(locale)
    if locale == "en-US":
        from misaki.en import G2P
        from misaki.espeak import EspeakFallback

        g2p = G2P(british=False, fallback=EspeakFallback(british=False))
    elif locale == "pt-BR":
        from earshift_bakeoff.bilingual_vowel_engine import PortugueseMisakiAdapter

        adapter = PortugueseMisakiAdapter.load(voice_id="pf_dora")
        # The Portuguese adapter aligns per word and refuses a chunk whose
        # groups do not line up, so it is fed word by word: one refusal must
        # not empty the whole corpus and read as a missing inventory.
        collected: list[str] = []
        for word in text.split():
            try:
                collected.extend(row.phone for row in adapter.analyze(word).words)
            except Exception:  # noqa: BLE001
                continue
        return " ".join(collected)
    else:
        from misaki.espeak import EspeakG2P

        g2p = EspeakG2P(language=ESPEAK_LANGUAGE[locale])
    out: list[str] = []
    words = text.split()
    for start in range(0, len(words), 200):
        chunk = " ".join(words[start:start + 200])
        try:
            phones, _ = g2p(chunk)
        except Exception:  # noqa: BLE001 — a chunk that will not align is skipped
            continue
        out.append(phones)
    return " ".join(out)


def produced(phoneme: str, corpus: str) -> bool:
    candidates = (phoneme, *EQUIVALENT.get(phoneme, ()))
    forms: set[str] = set()
    for candidate in candidates:
        forms.add(candidate)
        forms.add(unicodedata.normalize("NFD", candidate))
        forms.add(unicodedata.normalize("NFC", candidate))
    return any(form and form in corpus for form in forms)


def main() -> None:
    selected = [loc for loc in sys.argv[1:] if loc in LOCALES] or list(LOCALES)
    # Merge into whatever is already on disk. Writing a fresh report meant a
    # scoped re-run of three locales replaced the artifact with three locales,
    # discarding the other twenty-seven and the hand-written triage alongside
    # them — the same shape of loss the discovery sweep grew a guard for.
    report: dict = {"schema_version": 1, "locales": {}}
    if OUT_PATH.is_file():
        report = json.loads(OUT_PATH.read_text(encoding="utf-8"))
        report.setdefault("locales", {})
    total_missing = 0

    for locale in selected:
        corpus = phonemize(locale)
        nfd = unicodedata.normalize("NFD", corpus)
        spec = PHONEMIC_INVENTORY[locale]
        missing = {"vowels": [], "consonants": []}
        for kind in ("vowels", "consonants"):
            for phoneme in spec[kind]:
                if not (produced(phoneme, corpus) or produced(phoneme, nfd)):
                    missing[kind].append(phoneme)
        count = len(missing["vowels"]) + len(missing["consonants"])
        conclusive = len(corpus) >= MIN_CORPUS_CHARS
        if conclusive:
            total_missing += count
        report["locales"][locale] = {
            "corpus_phone_chars": len(corpus),
            "inventory_size": len(spec["vowels"]) + len(spec["consonants"]),
            "conclusive": conclusive,
            "never_produced": missing if conclusive else {"vowels": [], "consonants": []},
            "inconclusive_absences": None if conclusive else missing,
        }
        if not conclusive:
            print(f"{locale:<7} inventory={report['locales'][locale]['inventory_size']:>3}"
                  f"  INCONCLUSIVE (corpus {len(corpus)} chars < "
                  f"{MIN_CORPUS_CHARS}; no frequency list for this language)",
                  flush=True)
            continue
        flag = ""
        if count:
            flag = "  ABSENT: " + " ".join(
                missing["vowels"] + missing["consonants"]
            )
        print(f"{locale:<7} inventory={report['locales'][locale]['inventory_size']:>3}"
              f" absent={count:>2}{flag}", flush=True)

    # Recomputed over every locale on file, not just the ones re-run, so a
    # scoped run cannot leave a total that describes a subset.
    report["total_absent"] = sum(
        len(row["never_produced"]["vowels"]) + len(row["never_produced"]["consonants"])
        for row in report["locales"].values()
    )
    report["min_corpus_chars"] = MIN_CORPUS_CHARS
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"\n{total_missing} inventory phonemes never produced across "
          f"{len(selected)} locales")
    print(f"written to {OUT_PATH}")


if __name__ == "__main__":
    main()
