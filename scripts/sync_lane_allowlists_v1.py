#!/usr/bin/env python3
"""Regenerate the Worker allowlist and the site's direction menu.

The Worker allowlist is a fail-closed security boundary: a profile id the
Worker does not know is rejected at the edge before any upstream call. That
property is worth keeping at 863 directions, but the list can no longer be
maintained by hand without drifting from the service. So it is generated from
the same registry the builder loads, which is the single source of truth
(`supported_profile_ids`).

Writes worker/azure-lens-profiles.generated.js and
site/listener-directions.generated.js.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from earshift_bakeoff.azure_lens_builder import (  # noqa: E402
    AzureLensBuilderError,
    PROFILE_LOCALES,
    audible_rule_count,
    build_pair,
    load_azure_profiles,
    supported_profile_ids,
)
from lens_language_data_v1 import LOCALES  # noqa: E402

WORKER_OUT = REPO / "worker" / "azure-lens-profiles.generated.js"
GIBBERISH_OUT = REPO / "worker" / "gibberish-locales.generated.js"
SITE_OUT = REPO / "site" / "listener-directions.generated.js"

DISPLAY_NAME = {
    "en-US": "English", "pt-BR": "Portuguese (Brazil)",
    "pt-PT": "Portuguese (Portugal)", "es-ES": "Spanish (Spain)",
    "es-MX": "Spanish (Mexico)", "fr-FR": "French", "it-IT": "Italian",
    "de-DE": "German", "el-GR": "Greek", "ru-RU": "Russian", "hi-IN": "Hindi",
    "nl-NL": "Dutch", "pl-PL": "Polish", "tr-TR": "Turkish",
    "sv-SE": "Swedish", "uk-UA": "Ukrainian", "id-ID": "Indonesian",
    "cs-CZ": "Czech", "ro-RO": "Romanian", "hu-HU": "Hungarian",
    "nb-NO": "Norwegian", "ca-ES": "Catalan", "hr-HR": "Croatian",
    "sk-SK": "Slovak", "sl-SI": "Slovenian", "bg-BG": "Bulgarian",
    "ms-MY": "Malay", "mr-IN": "Marathi", "te-IN": "Telugu",
    "gu-IN": "Gujarati",
}

# Endonyms. A picker for a language tool should let someone find their own
# language written the way they write it, rather than only under its English
# exonym.
NATIVE_NAME = {
    "en-US": "English", "pt-BR": "Português (BR)", "pt-PT": "Português (PT)",
    "es-ES": "Español", "es-MX": "Español (MX)", "fr-FR": "Français",
    "it-IT": "Italiano", "de-DE": "Deutsch", "el-GR": "Ελληνικά",
    "ru-RU": "Русский", "hi-IN": "हिन्दी", "nl-NL": "Nederlands",
    "pl-PL": "Polski", "tr-TR": "Türkçe", "sv-SE": "Svenska",
    "uk-UA": "Українська", "id-ID": "Bahasa Indonesia", "cs-CZ": "Čeština",
    "ro-RO": "Română", "hu-HU": "Magyar", "nb-NO": "Norsk",
    "ca-ES": "Català", "hr-HR": "Hrvatski", "sk-SK": "Slovenčina",
    "sl-SI": "Slovenščina", "bg-BG": "Български", "ms-MY": "Bahasa Melayu",
    "mr-IN": "मराठी", "te-IN": "తెలుగు", "gu-IN": "ગુજરાતી",
}

# Genetic family, which for this product is not decoration: relatives share
# large parts of an inventory, so they behave alike as listeners. Grouping by
# family means the nearest alternative to a disappointing lens is adjacent to
# it rather than scattered across an alphabetical list.
FAMILY = {
    "Germanic": ["en-US", "de-DE", "nl-NL", "sv-SE", "nb-NO"],
    "Romance": ["fr-FR", "it-IT", "es-ES", "pt-BR", "es-MX", "pt-PT",
                "ca-ES", "ro-RO"],
    "Slavic": ["pl-PL", "cs-CZ", "sk-SK", "hr-HR", "sl-SI", "bg-BG",
               "ru-RU", "uk-UA"],
    "Indo-Aryan": ["hi-IN", "gu-IN", "mr-IN"],
    "Austronesian": ["id-ID", "ms-MY"],
    "Other families": ["el-GR", "hu-HU", "tr-TR", "te-IN"],
}

# One example per source language, not one per direction. A per-direction
# example would be tuned to flatter its own lens, so two lenses could never
# be compared on equal input. These are pangrams and standard phonetically
# balanced sentences, each verified offline to fire rules across that
# language's listeners rather than chosen for any single one.
EXAMPLE_SENTENCE = {
    "en-US": "The happy cat sat back and laughed at the bad joke.",
    "pt-BR": "Um pequeno jabuti xereta viu dez cegonhas felizes.",
    "pt-PT": "Um pequeno jabuti xereta viu dez cegonhas felizes.",
    "es-ES": "El veloz murciélago hindú comía feliz cardillo y kiwi.",
    "es-MX": "El veloz murciélago hindú comía feliz cardillo y kiwi mientras el niño pequeño juega en el jardín.",
    "ca-ES": "Jove xef, porti whisky amb quinze glaçons d'hidrogen.",
    "fr-FR": "Portez ce vieux whisky au juge blond qui fume.",
    "it-IT": "Ma la volpe, col suo balzo, ha raggiunto il quieto Fido.",
    "ro-RO": "Bând whisky, jazzmanul vorbea șugubăț despre existența fizică.",
    "de-DE": "Victor jagt zwölf Boxkämpfer quer über den großen Deich.",
    "nl-NL": "Pa's wijze lynx bezag vroom het fikse aquaduct.",
    "sv-SE": "Flygande bäckasiner söka hvila på mjuka tuvor medan sju sjuka sjömän körde över ängen.",
    "nb-NO": "Vår sære Zulu fra badeøya spilte jo whist og quickstep.",
    "pl-PL": "Pchnąć w tę łódź jeża lub ośm skrzyń fig.",
    "cs-CZ": "Příliš žluťoučký kůň úpěl ďábelské ódy.",
    "sk-SK": "Kŕdeľ šťastných ďatľov učí pri ústí Váhu mĺkveho koňa.",
    "hr-HR": "Gojazni đačić s biciklom drži hmelj i finu vatu u džepu.",
    "sl-SI": "V kožuščku hudobnega fanta stopiclja mizar.",
    "bg-BG": "Жълтата дюля беше щастлива, че пухът цъфна и замръзна.",
    "ru-RU": "Съешь же ещё этих мягких французских булок да выпей чаю.",
    "uk-UA": "Чуєш їх, доцю, га? Кумедна ж ти, прощайся без ґольфів!",
    "el-GR": "Ξεσκεπάζω την ψυχοφθόρα βδελυγμία.",
    "hu-HU": "Egy hűtlen vejét fülöncsípő, dühös mexikói úr mázol.",
    "tr-TR": "Pijamalı hasta yağız şoföre çabucak güvendi.",
    "id-ID": "Muharjo seorang xenofobia universal mengelak pesta di kota.",
    "ms-MY": "Seorang pemuda gigih menjual buah cempedak di kaki lima sambil menyanyi lagu yang nyaring.",
    "hi-IN": "आज मौसम बहुत सुहावना है और पक्षी खुशी से गा रहे हैं। लड़का बड़ी ट्रेन और ठंडा पानी देखता है।",
    "gu-IN": "આજે હવામાન ખૂબ સરસ છે અને પક્ષીઓ ખુશીથી ગાય છે. મોટા છોકરાએ ઠંડા પાણીમાં ટ્રેન અને ઢોલ જોયા.",
    "mr-IN": "आज हवामान खूप छान आहे आणि पक्षी आनंदाने गात आहेत. मोठा मुलगा ठाण्यात ट्रक आणि ढग पाहतो.",
    "te-IN": "ఈరోజు వాతావరణం చాలా బాగుంది మరియు పక్షులు పాడుతున్నాయి."
}


BANNER = (
    "// GENERATED by scripts/sync_lane_allowlists_v1.py — do not edit.\n"
    "// Source of truth: azure_lens_builder.supported_profile_ids().\n"
)


def main() -> None:
    ids = sorted(supported_profile_ids())
    WORKER_OUT.write_text(
        BANNER
        + "export const AZURE_LENS_PROFILES = Object.freeze("
        + json.dumps(ids, indent=2)
        + ");\n",
        encoding="utf-8",
    )

    # Gibberish is source-only, so its allowlist is locales rather than
    # directions, and it comes from the frozen syllable bank rather than the
    # lens registry: a language is offered exactly when the harvest produced a
    # bank for it. Generated here so the edge cannot drift from the artifact.
    from earshift_bakeoff.gibberish_generator import supported_locales

    gibberish_locales = sorted(supported_locales())
    GIBBERISH_OUT.write_text(
        "// GENERATED by scripts/sync_lane_allowlists_v1.py — do not edit.\n"
        "// Source of truth: gibberish_generator.supported_locales().\n"
        + "export const GIBBERISH_LOCALES = Object.freeze("
        + json.dumps(gibberish_locales, indent=2)
        + ");\n",
        encoding="utf-8",
    )

    profiles = load_azure_profiles()
    supported = supported_profile_ids()
    # The curated registry carries hand-authored rules with the deriver's
    # errors corrected; the matrix does not. The picker marks the difference
    # rather than presenting 822 directions as equally well evidenced.
    curated_ids = {
        profile["id"]
        for profile in json.loads(
            (REPO / "rules" / "azure-listener-lenses-v1.json").read_text(
                encoding="utf-8"
            )
        )["profiles"]
    }
    by_source: dict[str, list[dict[str, str]]] = {}
    for profile_id, profile in profiles.items():
        source = profile.get("source_locale")
        listener = profile.get("listener_locale")
        if not source or not listener:
            continue
        if profile_id not in supported:
            # Suppressed directions must leave the menu, not just fail at the
            # Worker: an option that always errors is worse than no option.
            continue
        by_source.setdefault(source, []).append({
            "profileId": profile_id,
            "listener": listener,
            "label": f"{DISPLAY_NAME.get(listener, listener)} listener",
            "curated": profile_id in curated_ids,
            # Lets the random-listener button weight toward directions that
            # audibly do something instead of landing on a thin pair.
            "audibleRules": audible_rule_count(profile),
        })
    # The frozen en<->pt pair lives in the bilingual engine, not the Azure
    # registry, so the matrix generator skips it entirely. Building the menu
    # from load_azure_profiles() alone therefore drops the best-evidenced
    # direction in the product; add it back explicitly.
    from earshift_bakeoff.bilingual_listener_engine import load_listener_profiles

    frozen = load_listener_profiles()
    for profile_id, source in PROFILE_LOCALES.items():
        listener = "pt-BR" if source == "en-US" else "en-US"
        by_source.setdefault(source, []).append({
            "profileId": profile_id,
            "listener": listener,
            "label": f"{DISPLAY_NAME.get(listener, listener)} listener",
            "curated": True,
            "audibleRules": audible_rule_count(
                {**frozen[profile_id], "source_locale": source}
            ),
        })

    # The default listener (first in each source's list) is what a user lands
    # on when they pick that source, so it must actually fire on that source's
    # example sentence. Plain alphabetical order put Bulgarian's thin bg->hr
    # pair first, which refused on the example — a broken first impression.
    # Float listeners that fire to the top; alphabetical within each group so
    # the picker's family layout (which reads this list) stays stable.
    def fires_on_example(source: str, profile_id: str) -> bool:
        sentence = EXAMPLE_SENTENCE.get(source)
        if not sentence:
            return False
        try:
            return bool(build_pair(sentence, profile_id)["applied_rule_ids"])
        except AzureLensBuilderError:
            return False

    for source, rows in by_source.items():
        rows.sort(
            key=lambda row: (
                not fires_on_example(source, row["profileId"]),
                DISPLAY_NAME.get(row["listener"], row["listener"]),
            )
        )

    ordered = {
        locale: by_source[locale]
        for locale in LOCALES
        if by_source.get(locale)
    }
    # Only families whose languages actually survived suppression, so the
    # picker never renders an empty group heading.
    families = [
        {"family": family,
         "locales": [locale for locale in locales if locale in ordered]}
        for family, locales in FAMILY.items()
    ]
    families = [group for group in families if group["locales"]]
    missing = set(ordered) - {
        locale for group in families for locale in group["locales"]
    }
    if missing:
        raise SystemExit(f"locales missing a FAMILY entry: {sorted(missing)}")

    suggestion_path = REPO / "artifacts" / "language-expansion" / "direction-suggestions-v1.json"
    suggestions = (
        json.loads(suggestion_path.read_text(encoding="utf-8"))
        if suggestion_path.is_file() else {}
    )
    SITE_OUT.write_text(
        BANNER
        + "export const LANGUAGE_NAMES = "
        + json.dumps(DISPLAY_NAME, indent=2, ensure_ascii=False)
        + ";\n\nexport const NATIVE_NAMES = "
        + json.dumps(NATIVE_NAME, indent=2, ensure_ascii=False)
        + ";\n\nexport const LANGUAGE_FAMILIES = "
        + json.dumps(families, indent=2, ensure_ascii=False)
        + ";\n\nexport const DIRECTION_SUGGESTIONS = "
        + json.dumps(suggestions, indent=2, ensure_ascii=False)
        + ";\n\nexport const EXAMPLE_SENTENCE = "
        + json.dumps(EXAMPLE_SENTENCE, indent=2, ensure_ascii=False)
        + ";\n\nexport const LISTENERS_BY_SOURCE = "
        + json.dumps(ordered, indent=2, ensure_ascii=False)
        + ";\n",
        encoding="utf-8",
    )

    print(f"worker allowlist : {len(ids)} profile ids -> {WORKER_OUT}")
    print(f"gibberish locales: {len(gibberish_locales)} -> {GIBBERISH_OUT}")
    print(f"site menu        : {len(ordered)} source languages -> {SITE_OUT}")
    for locale, rows in list(ordered.items())[:3]:
        print(f"  {locale}: {len(rows)} listeners")


if __name__ == "__main__":
    main()
