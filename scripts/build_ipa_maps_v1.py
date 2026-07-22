#!/usr/bin/env python3
"""Extend the Azure IPA map to cover every locale in the matrix.

A source locale's map must contain two things: every symbol its own G2P can
emit (the neutral side), and every symbol any lens rule can target on it (the
lens side). The second set is cross-inventory — a Greek voice has to render
/ɹ/ for the English-listener rule, and an English voice has to render /ɣ/ for
the Greek-listener one — which is why the required set is computed from the
generated rules rather than guessed.

Existing entries are never rewritten. The five tables that already carry
hand-written notes and per-symbol Azure acceptance receipts (en-US, pt-BR,
es-ES, it-IT, de-DE) keep exactly what they have; this only fills gaps.

Every symbol added here is unverified until the probe records an accept or
reject receipt against that locale's own voice. Fidelity starts at
"approximate" for a cross-inventory target because it is, by construction, a
category the locale does not own.
"""

from __future__ import annotations

import json
import sys
import unicodedata
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from lens_language_data_v1 import LOCALES  # noqa: E402

MAP_PATH = REPO / "rules" / "azure-ipa-map-v1.json"
SURFACE_PATH = REPO / "artifacts" / "lens-surface-symbols-v1" / "symbols.json"
REQUIRED_PATH = (
    REPO / "artifacts" / "lens-surface-symbols-v1" / "required-map-symbols.json"
)

STRUCTURAL = {
    " ": "word separator; never inside a ph attribute",
    "ˈ": None,
    "ˌ": None,
}

# Ligature and shorthand spellings espeak/Misaki emit, written out to the
# digraph Azure expects.
EXPANSIONS = {
    "ʧ": "tʃ", "ʤ": "dʒ", "ʦ": "ts", "ʣ": "dz", "ʨ": "tɕ", "ʥ": "dʑ",
    "A": "eɪ", "I": "aɪ", "O": "oʊ", "W": "aʊ", "Y": "ɔɪ", "Q": "əʊ",
    "ᵊ": "ə", "ᵻ": "ɪ",
}

# Adapter artifacts that are not phones and must never reach a ph attribute.
ARTIFACTS = {
    ".": "espeak artifact, not a phone",
    # The Russian voice emits a bare quote after some /u/ vowels (людей ->
    # ɭʲu"dʲˈej). It is not a segment and Azure rejects it outright, which
    # failed every Russian word carrying one.
    '"': "espeak artifact, not a phone",
}

# Combining marks ride attached to the preceding segment.
COMBINING = {
    "̃": "combining tilde stays attached to its vowel",
    "̪": "dental diacritic stays attached to its consonant",
    "̝": "raised diacritic (Czech ř)",
    "̩": "syllabic diacritic",
    # Only ever arrives composed onto its base as part of ç; no G2P emits it
    # alone. Mapping it to itself made it look like a standalone phone, which
    # Azure rejects.
    "̧": "cedilla stays attached to its consonant",
    "ʲ": "palatalisation stays attached to its consonant",
    "ʰ": "aspiration stays attached to its stop",
    "ː": "length mark stays attached to its vowel",
}


def entry(symbol: str, *, owned: bool) -> dict[str, str]:
    if symbol in STRUCTURAL:
        row = {"azure_ipa": symbol, "fidelity": "structural"}
        if STRUCTURAL[symbol]:
            row["note"] = STRUCTURAL[symbol]
        return row
    if symbol in ARTIFACTS:
        return {"azure_ipa": "", "fidelity": "drop", "note": ARTIFACTS[symbol]}
    if symbol in COMBINING:
        return {
            "azure_ipa": symbol,
            "fidelity": "exact",
            "note": COMBINING[symbol],
        }
    if symbol in EXPANSIONS:
        return {
            "azure_ipa": EXPANSIONS[symbol],
            "fidelity": "exact",
            "note": "ligature to digraph",
        }
    if owned:
        return {"azure_ipa": symbol, "fidelity": "exact"}
    return {
        "azure_ipa": symbol,
        "fidelity": "approximate",
        "note": "cross-inventory lens target; this locale does not own the "
                "category, so it appears only on the lens side",
    }


def main() -> None:
    data = json.loads(MAP_PATH.read_text(encoding="utf-8"))
    locales = data["locales"]
    surface = {
        loc: set(row["symbols"])
        for loc, row in json.loads(
            SURFACE_PATH.read_text(encoding="utf-8")
        )["locales"].items()
    }
    required = json.loads(REQUIRED_PATH.read_text(encoding="utf-8"))

    created = added = 0
    for locale in LOCALES:
        table = locales.setdefault(locale, {})
        was_new = not table
        owned = surface.get(locale, set()) | {" ", "ˈ", "ˌ"}
        targets = set(required.get(locale, ()))
        # Targets are inserted into the phone string whole, but the mapper
        # walks it one character at a time, so a multi-character target needs
        # its parts covered too: a rule writing "aː" onto a locale whose own
        # inventory has no length mark left ː unmapped and failed the render.
        targets |= {char for target in targets for char in target}
        wanted = owned | targets
        # The mapper walks the phone string after NFD normalisation, so a
        # precomposed symbol arrives as its decomposition: espeak emits the
        # Greek ç precomposed, NFD splits it into c + U+0327, and the bare
        # combining cedilla was unmapped — which failed every Greek-source
        # direction on the word χέρι. Cover the decomposed forms too.
        wanted |= {
            char
            for symbol in tuple(wanted)
            for char in unicodedata.normalize("NFD", symbol)
        }
        new_here = 0
        for symbol in sorted(wanted):
            if symbol in table:
                continue
            table[symbol] = entry(symbol, owned=symbol in owned)
            new_here += 1
        added += new_here
        if was_new:
            created += 1
        print(f"{locale:<7} entries={len(table):>3} added={new_here:>3}"
              f"{'  (new table)' if was_new else ''}")

    MAP_PATH.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"\n{created} tables created, {added} entries added")
    print(f"written to {MAP_PATH}")


if __name__ == "__main__":
    main()
