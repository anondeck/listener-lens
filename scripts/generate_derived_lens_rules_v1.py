#!/usr/bin/env python3
"""Derive a baseline listener-lens rule table for a language pair.

A listener rule table says: when a speaker of X is heard by a listener whose
sound system is Y, which of X's categories collapse onto Y's? That mapping is
computable rather than authored — take X's phoneme inventory, take Y's, and
map each of X's segments to its nearest Y category by articulatory feature
distance (panphon).

Two guards keep the output honest:

  * distance threshold — beyond MAX_DISTANCE the generator emits no rule and
    records a no-near-category note. Without it the nearest match for English
    /h/ in Spanish comes back as /l/ at distance 2.38, which is noise, not a
    perceptual claim.
  * overrides — some attested mappings are orthographic or historical rather
    than featural. Spanish maps English /v/ to /b/, which no feature vector
    predicts (feature distance picks /f/). Overrides win and carry their own
    evidence tier.

Everything emitted here is a *derived* claim: reproducible from these
inventories plus panphon, and explicitly not a cited listener result. Rules
are upgraded to a direct tier only when a real published source is attached
by hand.

Build-time only: panphon is not a runtime dependency. The committed JSON is
what the lane reads.
"""

from __future__ import annotations

import json
import sys
import unicodedata
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

# Draft output: articulatory derivation is reliable for vowels and not for
# consonants (see the module docstring), so this is a review artifact, not
# a rules/ table the lane may load.
OUT_PATH = REPO / "artifacts" / "derived-lens-draft-v1" / "en-es-draft.json"

# Beyond this feature distance the "nearest" category is not a perceptual
# neighbour at all, so no rule is emitted.
MAX_DISTANCE = 1.0

# Misaki/espeak shorthand for English diphthongs, expanded before segmentation.
SHORTHAND = {"A": "eɪ", "I": "aɪ", "O": "oʊ", "W": "aʊ", "Y": "ɔɪ"}
STRIP = set("ˈˌː^ ")

# Curated *phonemic* inventories. Scraping an inventory from espeak output
# does not work: espeak emits surface allophones, so Spanish /b d ɡ/ appear
# only as [β ð ɣ] and the deriver concludes Spanish lacks voiced stops —
# which produced b→p, d→t, ɡ→k, i.e. devoicing English stops. A listener
# rule table has to compare phoneme systems, not surface realisations.
# These are standard, uncontested descriptions of each system.
PHONEMIC_INVENTORY = {
    "en-US": {
        "vowels": ["i", "ɪ", "eɪ", "ɛ", "æ", "ɑ", "ɔ", "oʊ", "ʊ", "u", "ʌ",
                   "ɜ", "ə", "aɪ", "aʊ", "ɔɪ"],
        "consonants": ["p", "b", "t", "d", "k", "ɡ", "tʃ", "dʒ", "f", "v",
                       "θ", "ð", "s", "z", "ʃ", "ʒ", "h", "m", "n", "ŋ",
                       "l", "ɹ", "j", "w"],
    },
    # European Spanish: keeps /θ/ (distinción) and conservative /ʎ/.
    "es-ES": {
        "vowels": ["a", "e", "i", "o", "u"],
        "consonants": ["p", "b", "t", "d", "k", "ɡ", "tʃ", "f", "θ", "s",
                       "x", "m", "n", "ɲ", "l", "ʎ", "ɾ", "r", "j", "w"],
    },
    # Brazilian Portuguese: seven oral vowels plus the five contrastive nasals
    # (lã/lã~la), and the palatalised /tʃ dʒ/ that ti/di surface as.
    "pt-BR": {
        "vowels": ["a", "e", "ɛ", "i", "o", "ɔ", "u",
                   "ɐ̃", "ẽ", "ĩ", "õ", "ũ"],
        "consonants": ["p", "b", "t", "d", "k", "ɡ", "tʃ", "dʒ", "f", "v",
                       "s", "z", "ʃ", "ʒ", "m", "n", "ɲ", "l", "ʎ", "ɾ",
                       "ʁ", "j", "w"],
    },
    # Mexican Spanish: seseo (no /θ/) and yeísmo (no /ʎ/). This is the
    # majority-variety system; es-ES above is the conservative Castilian one.
    # The two differ by exactly those two categories, which is why English
    # /θ/ survives into es-ES and collapses onto /s/ here.
    "es-MX": {
        "vowels": ["a", "e", "i", "o", "u"],
        "consonants": ["p", "b", "t", "d", "k", "ɡ", "tʃ", "f", "s",
                       "x", "m", "n", "ɲ", "l", "ɾ", "r", "j", "w"],
    },
    "fr-FR": {
        "vowels": ["i", "e", "ɛ", "a", "ɔ", "o", "u", "y", "ø", "œ", "ə",
                   "ɛ̃", "ɑ̃", "ɔ̃"],
        "consonants": ["p", "b", "t", "d", "k", "ɡ", "f", "v", "s", "z",
                       "ʃ", "ʒ", "m", "n", "ɲ", "l", "ʁ", "j", "w", "ɥ"],
    },
    "de-DE": {
        "vowels": ["i", "ɪ", "e", "ɛ", "a", "ɔ", "o", "ʊ", "u", "y", "ʏ",
                   "ø", "œ", "ə", "aɪ", "aʊ", "ɔʏ"],
        "consonants": ["p", "b", "t", "d", "k", "ɡ", "f", "v", "s", "z",
                       "ʃ", "ʒ", "ç", "x", "h", "m", "n", "ŋ", "l", "ʁ",
                       "j", "ts", "pf"],
    },
    "it-IT": {
        "vowels": ["i", "e", "ɛ", "a", "ɔ", "o", "u"],
        "consonants": ["p", "b", "t", "d", "k", "ɡ", "tʃ", "dʒ", "ts", "dz",
                       "f", "v", "s", "z", "ʃ", "m", "n", "ɲ", "l", "ʎ",
                       "r", "j", "w"],
    },
}

# Word lists retained only for the espeak surface-inventory comparison that
# exposed the allophone problem; they are not used to derive rules.
INVENTORY_WORDS = {
    "en-US": (
        "beat bit bait bet bat bot bought boat book boot but bird about "
        "pat bad tack dog cat gap fat vat thin then sat zap ship measure "
        "chat jam mat nap sing lap rat yes wet hat house buy boy now"
    ),
    "es-ES": (
        "padre madre gato dedo casa fuego jamón hijo llave año caro carro "
        "cinco zapato chico mucho sol luz mano nada peso vino bueno agua "
        "tierra silla queso guerra piso puerta"
    ),
    "es-MX": (
        "padre madre gato dedo casa fuego jamón hijo llave año caro carro "
        "cinco zapato chico mucho sol luz mano nada peso vino bueno agua "
        "tierra silla queso guerra piso puerta"
    ),
    "fr-FR": (
        "papa bébé table donner chat gare face vase sac zéro chien jaune "
        "manger nous lit rue oui huile pain bon brun vin peu peur beau "
        "port pur lune fille agneau"
    ),
    "de-DE": (
        "Vater Mutter Tag Kind Gast Fisch Wasser Sohn Zeit Katze Bach ich "
        "Buch Loch machen Menschen nein Ring lang rot Haus Bein Leute "
        "schön Bücher Käse Meer bitte Hand"
    ),
    "it-IT": (
        "padre madre gatto dito casa fuoco gente figlio chiave anno caro "
        "carro cinque zappa cielo gioco sole luce mano nulla peso vino "
        "buono acqua terra sedia gnomo aglio pesce"
    ),
}


def segments(phone_string: str, feature_table) -> list[str]:
    """Split an espeak phone string into feature-bearing segments."""

    text = phone_string
    for shorthand, expansion in SHORTHAND.items():
        text = text.replace(shorthand, expansion)
    text = "".join(ch for ch in text if ch not in STRIP)
    text = unicodedata.normalize("NFC", text)
    return [seg for seg in feature_table.ipa_segs(text) if seg]


def inventory(locale: str, feature_table) -> list[str]:
    from earshift_bakeoff.azure_source_adapters import ESPEAK_LANGUAGE_BY_LOCALE
    from misaki.espeak import EspeakG2P

    if locale == "en-US":
        from misaki.en import G2P
        from misaki.espeak import EspeakFallback

        g2p = G2P(british=False, fallback=EspeakFallback(british=False))
        phones, _ = g2p(INVENTORY_WORDS[locale])
    else:
        g2p = EspeakG2P(language=ESPEAK_LANGUAGE_BY_LOCALE[locale])
        phones, _ = g2p(INVENTORY_WORDS[locale])
    found: list[str] = []
    for seg in segments(phones, feature_table):
        if seg not in found:
            found.append(seg)
    return sorted(found)


def derive(source_locale: str, target_locale: str, overrides: dict[str, dict]) -> dict:
    from panphon.distance import Distance

    distance = Distance()
    source_spec = PHONEMIC_INVENTORY[source_locale]
    target_spec = PHONEMIC_INVENTORY[target_locale]
    source_vowels = set(source_spec["vowels"])
    source_inventory = [*source_spec["vowels"], *source_spec["consonants"]]
    target_inventory = [*target_spec["vowels"], *target_spec["consonants"]]
    # A vowel may only recategorise onto a vowel and a consonant onto a
    # consonant; feature distance alone will happily pair /ʒ/ with /ð/ across
    # that boundary, which is not a listener category judgement.
    target_by_class = {
        True: target_spec["vowels"],
        False: target_spec["consonants"],
    }

    vowel_rules: list[dict] = []
    consonant_rules: list[dict] = []
    unmapped: list[dict] = []
    for symbol in source_inventory:
        is_vowel = symbol in source_vowels
        if symbol in overrides:
            spec = overrides[symbol]
            target, tier, note = spec["target"], spec["evidence_tier"], spec["note"]
            reason = "override"
            score = None
        else:
            if symbol in target_inventory:
                continue  # the listener already owns this category
            ranked = sorted(
                target_by_class[is_vowel],
                key=lambda candidate: distance.weighted_feature_edit_distance(
                    symbol, candidate
                ),
            )
            if not ranked:
                continue
            target = ranked[0]
            score = round(
                distance.weighted_feature_edit_distance(symbol, target), 3
            )
            if score > MAX_DISTANCE:
                unmapped.append(
                    {
                        "source": symbol,
                        "nearest": target,
                        "distance": score,
                        "reason": "no_near_category_within_threshold",
                    }
                )
                continue
            tier = "derived_feature_distance_nearest_category"
            note = (
                f"Nearest {target_locale} category to {source_locale} /{symbol}/ by "
                f"panphon weighted feature edit distance ({score})."
            )
            reason = "derived"
        if target == symbol:
            continue
        rule = {
            "id": f"{source_locale.split('-')[0]}{target_locale.split('-')[0]}"
            f".{symbol}_{target}",
            "source": symbol,
            "target": target,
            "contexts": ["any"],
            "evidence_tier": tier,
            "acoustic_status": "pending_azure_qc",
            "source_ids": [],
            "derivation": reason,
            "feature_distance": score,
            "note": note,
        }
        (vowel_rules if is_vowel else consonant_rules).append(rule)
    return {
        "source_locale": source_locale,
        "target_locale": target_locale,
        "source_inventory": source_inventory,
        "target_inventory": target_inventory,
        "vowel_rules": vowel_rules,
        "consonant_rules": consonant_rules,
        "insertion_rules": [],
        "prosody_rules": [],
        "unmapped": unmapped,
    }


# Attested mappings that articulatory distance does not predict. Each needs a
# real source before it may claim a direct tier.
OVERRIDES = {
    ("en-US", "es-ES"): {
        "v": {
            "target": "b",
            "evidence_tier": "attested_orthographic_merger_uncited",
            "note": "Spanish merges orthographic v with /b/; feature distance "
            "wrongly prefers /f/. Needs a citation before claiming a direct tier.",
        },
    },
    # The b/v merger is pan-Spanish, not Castilian-specific, so it carries
    # over to the Mexican inventory unchanged.
    ("en-US", "es-MX"): {
        "v": {
            "target": "b",
            "evidence_tier": "attested_orthographic_merger_uncited",
            "note": "Spanish merges orthographic v with /b/; feature distance "
            "wrongly prefers /f/. Needs a citation before claiming a direct tier.",
        },
    },
}


def main() -> None:
    pairs = [("en-US", "es-ES"), ("es-ES", "en-US")]
    if len(sys.argv) > 2:
        pairs = [(sys.argv[1], sys.argv[2])]
    out_path = Path(sys.argv[3]) if len(sys.argv) > 3 else OUT_PATH
    report = {"schema_version": 1, "max_distance": MAX_DISTANCE, "profiles": {}}
    for source, target in pairs:
        profile = derive(source, target, OVERRIDES.get((source, target), {}))
        report["profiles"][f"{source}-to-{target}-listener-derived-v1"] = profile
        print(
            f"{source} -> {target}: {len(profile['vowel_rules'])} vowel, "
            f"{len(profile['consonant_rules'])} consonant, "
            f"{len(profile['unmapped'])} unmapped "
            f"(source inv {len(profile['source_inventory'])}, "
            f"target inv {len(profile['target_inventory'])})"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    print(f"written to {out_path}")


if __name__ == "__main__":
    main()
