#!/usr/bin/env python3
"""Generate every ordered listener direction across the supported locales.

For a pair (source S, listener L) the rule table answers: when a speaker of S
is heard by someone whose sound system is L, which of S's categories collapse
onto L's? Vowels come from panphon feature distance, which is reliable for
them. Consonants come from the listener preference chains in
lens_language_data_v1, because feature distance rejects exactly the
perceptually salient consonant mappings (a trill is not "near" an approximant
in feature space, yet every English listener files one as /ɹ/).

Two pruning gates keep the output honest rather than merely large:

  * surface gate — a rule whose source never appears in the phone strings that
    locale's G2P actually emits can never fire, so it is dropped with a
    reason rather than shipped as decoration. This is why the source symbols
    are resolved against the committed surface inventory instead of the
    phonemic one: en-US emits ʧ and A, not tʃ and eɪ.
  * identity gate — a category the listener already owns produces no rule.

Everything emitted is a derived candidate: reproducible from these
inventories, and explicitly not a cited listener result.

Writes rules/azure-listener-lenses-v2.json and reports the symbols each
source locale's Azure IPA map must therefore cover.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from lens_language_data_v1 import (  # noqa: E402
    AZURE_VOICE,
    CORRECTION_CHAINS,
    DELETE,
    IPA_HONOURED,
    LISTENER_PHONOTACTICS,
    LISTENER_STRESS,
    LOCALES,
    PHONEMIC_INVENTORY,
    SOURCE_OVERRIDES,
)

SURFACE_PATH = REPO / "artifacts" / "lens-surface-symbols-v1" / "symbols.json"
OUT_PATH = REPO / "rules" / "azure-listener-lenses-v2.json"
REQUIRED_PATH = REPO / "artifacts" / "lens-surface-symbols-v1" / "required-map-symbols.json"

# Beyond this feature distance the "nearest" category is not a perceptual
# neighbour at all, so no vowel rule is emitted.
MAX_DISTANCE = 1.0

# The validated Kokoro-era pair keeps its own frozen profiles and evidence.
FROZEN_PAIRS = {("en-US", "pt-BR"), ("pt-BR", "en-US")}

# Ligatures and shorthand: what a phonemic symbol may surface as. Checked
# against the discovered inventory, so a locale only ever uses the spelling
# its own adapter emits.
SURFACE_ALIASES: dict[str, tuple[str, ...]] = {
    "tʃ": ("ʧ",), "dʒ": ("ʤ",), "ts": ("ʦ",), "dz": ("ʣ",),
    "tɕ": ("ʨ", "ʧ"), "dʑ": ("ʥ", "ʤ"), "tʂ": ("ʈʂ", "ʧ"), "dʐ": ("ɖʐ", "ʤ"),
    "eɪ": ("A",), "aɪ": ("I",), "oʊ": ("O",), "aʊ": ("W",), "ɔɪ": ("Y",),
    "ɜ": ("ɚ", "ɜ"), "ə": ("ᵊ", "ə"), "ɪ": ("ᵻ", "ɪ"),
    "r̝": ("r̝", "ř"),
}

# Two-letter tag per locale for rule ids; es-ES/es-MX and pt-BR/pt-PT must not
# collide.
TAG = {
    "en-US": "en", "pt-BR": "pt", "pt-PT": "pp", "es-ES": "es", "es-MX": "mx",
    "fr-FR": "fr", "it-IT": "it", "de-DE": "de", "el-GR": "el", "ru-RU": "ru",
    "hi-IN": "hi", "nl-NL": "nl", "pl-PL": "pl", "tr-TR": "tr", "sv-SE": "sv",
    "uk-UA": "uk", "id-ID": "id", "cs-CZ": "cs", "ro-RO": "ro", "hu-HU": "hu",
    "nb-NO": "nb", "ca-ES": "ca", "hr-HR": "hr", "sk-SK": "sk", "sl-SI": "sl",
    "bg-BG": "bg", "ms-MY": "ms", "mr-IN": "mr", "te-IN": "te", "gu-IN": "gu",
}

VOICED_TO_VOICELESS = {"b": "p", "d": "t", "ɡ": "k", "v": "f", "z": "s",
                       "ʒ": "ʃ", "dʒ": "tʃ", "dz": "ts", "ʐ": "ʂ", "ʑ": "ɕ",
                       "ɣ": "x", "ð": "θ"}


def load_surface() -> dict[str, set[str]]:
    data = json.loads(SURFACE_PATH.read_text(encoding="utf-8"))
    return {loc: set(row["symbols"]) for loc, row in data["locales"].items()}


DISTINCTNESS_PATH = (
    REPO / "artifacts" / "azure-rule-distinctness-v1" / "receipts.json"
)


def load_distinctness() -> dict[str, str]:
    """Per-rule receipts from probe_rule_distinctness_v1.

    Acceptance and audibility come apart: a voice can return 200 for both of
    a rule's phones and then render them byte-identically, so the lens runs,
    the SSML differs, and the listener hears nothing. Carrying the verdict on
    the rule lets the lane report that instead of counting it as applied.
    """

    if not DISTINCTNESS_PATH.is_file():
        return {}
    data = json.loads(DISTINCTNESS_PATH.read_text(encoding="utf-8"))
    return {key: row["verdict"] for key, row in data.get("rules", {}).items()}


# Ligature and digraph spellings of the same category. A rule whose source
# and target differ only by spelling is a no-op that renders identically on
# both sides, so the identity gate compares canonical forms.
CANONICAL = {"ʧ": "tʃ", "ʤ": "dʒ", "ʦ": "ts", "ʣ": "dz", "ʨ": "tɕ",
             "ʥ": "dʑ", "ɚ": "ɜ", "ᵊ": "ə", "ᵻ": "ɪ",
             "A": "eɪ", "I": "aɪ", "O": "oʊ", "W": "aʊ", "Y": "ɔɪ"}


def canonical(symbol: str) -> str:
    return CANONICAL.get(symbol, symbol)


FACTS_PATH = REPO / "artifacts" / "language-expansion" / "propagated-listener-facts-v1.json"


def load_propagated_facts() -> dict[str, dict[str, str]]:
    """(listener|phone) decisions extracted from the curated tables.

    Assimilation is keyed by listener: a correction curated once (Italian
    listeners stop /θ/) applies to every source that sends that phone to
    that listener. Conflicting or context-dependent donors were already
    excluded at extraction, so everything here is unambiguous.
    """

    if not FACTS_PATH.is_file():
        return {}
    return json.loads(FACTS_PATH.read_text(encoding="utf-8"))["facts"]


PROPAGATED = load_propagated_facts()


def resolve_surface(symbol: str, surface: set[str]) -> str | None:
    """The spelling this locale's adapter actually emits, or None."""

    if symbol in surface:
        return symbol
    for alias in SURFACE_ALIASES.get(symbol, ()):
        if alias in surface:
            return alias
    return None


def listener_category(symbol: str, listener_inventory: set[str]) -> str | None:
    """First category in the symbol's preference chain the listener owns."""

    for candidate in CORRECTION_CHAINS.get(symbol, ()):
        if candidate == DELETE:
            return DELETE
        if candidate in listener_inventory:
            return candidate
    return None


def derive_pair(
    source: str,
    listener: str,
    surface: dict[str, set[str]],
    distance,
) -> tuple[dict, set[str], list[dict]]:
    source_spec = PHONEMIC_INVENTORY[source]
    listener_spec = PHONEMIC_INVENTORY[listener]
    source_vowels = set(source_spec["vowels"])
    listener_inventory = set(listener_spec["vowels"]) | set(listener_spec["consonants"])
    listener_vowels = listener_spec["vowels"]
    source_surface = surface[source]
    overrides = SOURCE_OVERRIDES.get((listener, source), {})

    vowel_rules: list[dict] = []
    consonant_rules: list[dict] = []
    dropped: list[dict] = []
    targets: set[str] = set()
    tag = f"{TAG[source]}{TAG[listener]}"

    for symbol in (*source_spec["vowels"], *source_spec["consonants"]):
        is_vowel = symbol in source_vowels
        # The listener already owns this category: no rule.
        if symbol in listener_inventory and symbol not in overrides:
            continue
        spelled = resolve_surface(symbol, source_surface)
        if spelled is None:
            dropped.append({"source": symbol, "reason": "not_in_surface_inventory"})
            continue

        if symbol in overrides:
            target = overrides[symbol]
            tier = "attested_orthographic_merger_uncited"
            note = f"Attested {listener} treatment of {source} /{symbol}/."
        elif (fact := PROPAGATED.get(f"{listener}|{canonical(symbol)}")) is not None:
            if fact["target"] == "__NO_RULE__":
                # The curator deliberately ruled this phone out for this
                # listener; the deriver's rule for it does not survive.
                continue
            target = fact["target"]
            tier = "propagated_from_curated_pair"
            note = (
                f"Curated treatment of /{symbol}/ by a {listener} listener "
                f"(from {fact['donors']}), propagated across sources: "
                "assimilation is a property of the listener, not the pair."
            )
        elif is_vowel:
            ranked = sorted(
                listener_vowels,
                key=lambda c: distance.weighted_feature_edit_distance(symbol, c),
            )
            if not ranked:
                continue
            target = ranked[0]
            score = round(
                distance.weighted_feature_edit_distance(symbol, target), 3
            )
            if score > MAX_DISTANCE:
                chained = listener_category(symbol, listener_inventory)
                if chained in (None, DELETE):
                    dropped.append({
                        "source": symbol,
                        "nearest": target,
                        "distance": score,
                        "reason": "no_near_category_within_threshold",
                    })
                    continue
                target, tier = chained, "derived_nearest_listener_category"
                note = (
                    f"Feature distance found no {listener} vowel within "
                    f"{MAX_DISTANCE} of /{symbol}/; the listener preference "
                    f"chain resolves it to /{target}/."
                )
            else:
                tier = "derived_feature_distance_nearest_category"
                note = (
                    f"Nearest {listener} vowel to {source} /{symbol}/ by panphon "
                    f"weighted feature edit distance ({score})."
                )
        else:
            chained = listener_category(symbol, listener_inventory)
            if chained is None:
                dropped.append({"source": symbol, "reason": "no_chain_category"})
                continue
            if chained == DELETE:
                consonant_rules.append({
                    "id": f"{tag}.{symbol}_deleted",
                    "source": spelled,
                    "operation": "delete",
                    "contexts": ["any"],
                    "evidence_tier": "derived_structural_phonotactic",
                    "acoustic_status": "pending_azure_qc",
                    "source_ids": [],
                    "note": (
                        f"{listener} has no category for /{symbol}/ and no near "
                        f"neighbour; the segment is not perceived."
                    ),
                })
                continue
            target = chained
            tier = "derived_nearest_listener_category"
            note = (
                f"{listener} lacks /{symbol}/; the listener preference chain "
                f"resolves it to /{target}/. Hand-authored chain because "
                f"feature distance is unreliable across consonant manner."
            )

        if canonical(target) == canonical(spelled):
            continue
        targets.add(target)
        rule = {
            "id": f"{tag}.{symbol}_{target}",
            "source": spelled,
            "target": target,
            "contexts": ["any"],
            "evidence_tier": tier,
            "acoustic_status": "pending_azure_qc",
            "source_ids": [],
            "note": note,
        }
        (vowel_rules if is_vowel else consonant_rules).append(rule)

    return (
        {"vowel_rules": vowel_rules, "consonant_rules": consonant_rules},
        targets,
        dropped,
    )


def phonotactic_rules(
    source: str, listener: str, surface: dict[str, set[str]]
) -> tuple[list[dict], list[dict], set[str]]:
    """The listener's structural repairs, rendered against this source."""

    tag = f"{TAG[source]}{TAG[listener]}"
    source_surface = surface[source]
    insertion: list[dict] = []
    deletion: list[dict] = []
    targets: set[str] = set()

    for spec in LISTENER_PHONOTACTICS.get(listener, ()):
        op = spec["op"]
        if op in ("insert_before", "insert_after"):
            rule = {
                "id": f"{tag}.{op}_{spec['target']}",
                "operation": op,
                "target": spec["target"],
                "contexts": list(spec["contexts"]),
                "evidence_tier": "derived_structural_phonotactic",
                "acoustic_status": "pending_azure_qc",
                "source_ids": [],
                "note": spec["note"],
            }
            for key in ("onsets", "not_followed_by", "vowels", "legal_codas"):
                if key in spec:
                    rule[key] = list(spec[key])
            targets.add(spec["target"])
            insertion.append(rule)
        elif op == "delete":
            spelled = resolve_surface(str(spec["source"]), source_surface)
            if spelled is None:
                continue
            rule = {
                "id": f"{tag}.delete_{spec['source']}",
                "source": spelled,
                "operation": "delete",
                "contexts": list(spec["contexts"]),
                "evidence_tier": "derived_structural_phonotactic",
                "acoustic_status": "pending_azure_qc",
                "source_ids": [],
                "note": spec["note"],
            }
            if "followed_by" in spec:
                rule["followed_by"] = list(spec["followed_by"])
            deletion.append(rule)
        elif op == "final_devoicing":
            for voiced, voiceless in VOICED_TO_VOICELESS.items():
                spelled = resolve_surface(voiced, source_surface)
                if spelled is None:
                    continue
                if voiceless not in set(PHONEMIC_INVENTORY[listener]["consonants"]):
                    continue
                targets.add(voiceless)
                deletion.append({
                    "id": f"{tag}.final_devoice_{voiced}",
                    "source": spelled,
                    "target": voiceless,
                    "operation": "substitute",
                    "contexts": ["word_final"],
                    "evidence_tier": "derived_structural_phonotactic",
                    "acoustic_status": "pending_azure_qc",
                    "source_ids": [],
                    "note": spec["note"],
                })
    return insertion, deletion, targets


def prosody_rules(source: str, listener: str) -> list[dict]:
    tag = f"{TAG[source]}{TAG[listener]}"
    placement = LISTENER_STRESS.get(listener)
    rules: list[dict] = []
    if placement == "final":
        rules.append({
            "id": f"{tag}.stress_final",
            "operation": "shift_primary_stress_to_final",
            "evidence_tier": "derived_nearest_listener_category",
            "acoustic_status": "pending_azure_qc",
            "source_ids": [],
            "note": f"{listener} places prominence on the final syllable.",
        })
    elif placement in ("initial", "penultimate"):
        rules.append({
            "id": f"{tag}.stress_{placement}",
            "operation": "swap_primary_and_initial_secondary_stress",
            "evidence_tier": "derived_nearest_listener_category",
            "acoustic_status": "pending_azure_qc",
            "source_ids": [],
            "note": f"{listener} has a fixed {placement} stress bias.",
        })
    return rules


def main() -> None:
    from panphon.distance import Distance

    distance = Distance()
    surface = load_surface()
    distinctness = load_distinctness()
    profiles: list[dict] = []
    required: dict[str, set[str]] = {loc: set() for loc in LOCALES}
    stats: list[tuple[str, int, int, int, int]] = []

    for source in LOCALES:
        if source not in IPA_HONOURED:
            continue
        for listener in LOCALES:
            if listener == source or (source, listener) in FROZEN_PAIRS:
                continue
            segments, targets, dropped = derive_pair(
                source, listener, surface, distance
            )
            insertion, deletion, struct_targets = phonotactic_rules(
                source, listener, surface
            )
            prosody = prosody_rules(source, listener)
            # A listener can reach the same repair twice: the preference
            # chain resolves French /h/ to DELETE, and the phonotactic block
            # also deletes it. Keep the phonotactic statement, which carries
            # the cited context, and drop the chain's duplicate.
            structural_sources = {
                rule["source"] for rule in deletion if "source" in rule
            }
            consonants = [
                rule for rule in segments["consonant_rules"]
                if not (
                    rule.get("operation") == "delete"
                    and rule.get("source") in structural_sources
                )
            ] + deletion
            # The builder takes the first rule whose context matches, so a
            # narrow context has to be offered before the general one. A
            # word-final devoicing rule listed after an "any" substitution on
            # the same segment would never fire.
            consonants.sort(key=lambda rule: "any" in rule.get("contexts", ("any",)))
            if not segments["vowel_rules"] and not consonants:
                continue
            required[source] |= targets | struct_targets
            # Stamp the audibility receipt onto every substitution so the lane
            # can separate "this voice renders the shift" from "this voice
            # collapses the pair". Deletions and insertions change the segment
            # count, so they are audible by construction and are not probed.
            for rule in [*segments["vowel_rules"], *consonants]:
                if rule.get("operation") == "delete" or not rule.get("target"):
                    continue
                verdict = distinctness.get(
                    f"{source}|{rule['source']}|{rule['target']}"
                )
                if verdict:
                    rule["renderer_verdict"] = verdict
            profiles.append({
                "id": f"{source}-to-{listener}-listener-v1",
                "source_locale": source,
                "listener_locale": listener,
                "voice": AZURE_VOICE[source],
                "claim": (
                    f"Experimental {source}-through-{listener} listener profile. "
                    "Derived vowel baseline plus chain-resolved consonant "
                    "recategorisations; no cited listener result and no blind QC."
                ),
                "vowel_rules": segments["vowel_rules"],
                "consonant_rules": consonants,
                "insertion_rules": insertion,
                "prosody_rules": prosody,
                "dropped": dropped,
            })
            stats.append((
                f"{source}->{listener}",
                len(segments["vowel_rules"]),
                len(consonants),
                len(insertion),
                len(prosody),
            ))

    OUT_PATH.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "status": "derived_baseline_pending_citation_and_blind_qc",
                "method": {
                    "vowel_derivation": "panphon weighted feature edit distance.",
                    "consonant_derivation": "listener preference chains; feature "
                    "distance is unreliable across consonant manner.",
                    "evidence_policy": "No rule here claims a cited listener "
                    "result. Every rule is a derived candidate awaiting both a "
                    "published source and blind human QC.",
                },
                "profiles": profiles,
            },
            indent=2, ensure_ascii=False, sort_keys=False,
        ) + "\n",
        encoding="utf-8",
    )
    REQUIRED_PATH.write_text(
        json.dumps(
            {loc: sorted(syms) for loc, syms in sorted(required.items())},
            indent=2, ensure_ascii=False, sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )

    total_possible = len(LOCALES) * (len(LOCALES) - 1) - len(FROZEN_PAIRS)
    print(f"generated {len(profiles)} of {total_possible} possible directions")
    empty = [row for row in stats if row[1] + row[2] == 0]
    print(f"directions with no segment rule: {len(empty)}")
    print(f"written to {OUT_PATH}")
    print(f"required map symbols written to {REQUIRED_PATH}")


if __name__ == "__main__":
    main()
