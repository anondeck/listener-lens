#!/usr/bin/env python3
"""Turn the curated tables into (listener, phone) facts the matrix can reuse.

Assimilation is a property of the listener, not of the pair: Italian
listeners stop /θ/ no matter whether the θ arrives from English or Greek.
The ten curated and frozen tables therefore contain decisions that
generalize across sources — but the generator never read them, so a
correction made by hand in one direction left the same error standing in
every other direction with the same listener.

Extraction fails closed three ways:

  * a phone a donor rules on more than once is context-dependent and is
    excluded for that donor (the frozen tables condition some vowels on
    rhotic context);
  * donors that disagree produce no fact — the disagreement itself is
    recorded, because it marks either a genuinely source-dependent
    assimilation (en hears es /x/ as h but de /x/ as k) or an
    inconsistency worth human eyes;
  * a curated table's deliberate silence (the derived twin ruled on a
    phone, the curator removed it) becomes a NO_RULE fact only when that
    silence is unambiguous across donors.

Writes artifacts/language-expansion/propagated-listener-facts-v1.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

CURATED_PATH = REPO / "rules" / "azure-listener-lenses-v1.json"
MATRIX_PATH = REPO / "rules" / "azure-listener-lenses-v2.json"
OUT_PATH = REPO / "artifacts" / "language-expansion" / "propagated-listener-facts-v1.json"

NO_RULE = "__NO_RULE__"

CANONICAL = {"ʧ": "tʃ", "ʤ": "dʒ", "ʦ": "ts", "ʣ": "dz", "ʨ": "tɕ",
             "ʥ": "dʑ", "ɚ": "ɜ", "ᵊ": "ə", "ᵻ": "ɪ",
             "A": "eɪ", "I": "aɪ", "O": "oʊ", "W": "aʊ", "Y": "ɔɪ"}


def canon(symbol: str) -> str:
    return CANONICAL.get(symbol, symbol)


def main() -> None:
    curated = json.loads(CURATED_PATH.read_text(encoding="utf-8"))["profiles"]
    matrix = {
        profile["id"]: profile
        for profile in json.loads(MATRIX_PATH.read_text(encoding="utf-8"))["profiles"]
    }

    from earshift_bakeoff.bilingual_listener_engine import load_listener_profiles

    frozen = load_listener_profiles()
    donors = [
        (p["listener_locale"], p["source_locale"], p, matrix.get(p["id"]))
        for p in curated
    ]
    donors.append(("pt-BR", "en-US", frozen["en-US-to-pt-BR-listener-v2"], None))
    donors.append(("en-US", "pt-BR", frozen["pt-BR-to-en-US-listener-v2"], None))

    # facts[(listener, phone)] -> {value: [donor labels]}
    raw: dict[tuple[str, str], dict[str, list[str]]] = {}
    for listener, source, profile, twin in donors:
        label = f"{source}->{listener}"
        rules = list(profile.get("vowel_rules") or []) + list(
            profile.get("consonant_rules") or []
        )
        seen: dict[str, int] = {}
        for rule in rules:
            if rule.get("operation") == "delete" or not rule.get("source"):
                continue
            seen[canon(rule["source"])] = seen.get(canon(rule["source"]), 0) + 1
        for rule in rules:
            symbol = canon(rule.get("source") or "")
            if not symbol or rule.get("operation") == "delete":
                continue
            contexts = rule.get("contexts")
            # Context-gated or multiply-ruled phones are pair-specific
            # phonology this flat key cannot express: fail closed.
            if seen[symbol] > 1 or (contexts and contexts != ["any"]):
                raw.setdefault((listener, symbol), {}).setdefault(
                    "__CONTEXT_DEPENDENT__", []
                ).append(label)
                continue
            raw.setdefault((listener, symbol), {}).setdefault(
                canon(rule["target"]), []
            ).append(label)
        if twin is not None:
            ruled = {canon(r["source"]) for r in rules if r.get("source")}
            for rule in list(twin.get("vowel_rules") or []) + list(
                twin.get("consonant_rules") or []
            ):
                symbol = canon(rule.get("source") or "")
                if symbol and symbol not in ruled and rule.get("operation") != "delete":
                    raw.setdefault((listener, symbol), {}).setdefault(
                        NO_RULE, []
                    ).append(label)

    facts: dict[str, dict[str, str]] = {}
    conflicts: dict[str, dict[str, list[str]]] = {}
    for (listener, symbol), values in sorted(raw.items()):
        key = f"{listener}|{symbol}"
        if "__CONTEXT_DEPENDENT__" in values or len(values) > 1:
            conflicts[key] = values
            continue
        target, donors_for = next(iter(values.items()))
        facts[key] = {"target": target, "donors": ", ".join(donors_for)}

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(
        json.dumps(
            {"schema_version": 1, "facts": facts, "conflicts": conflicts},
            indent=2, ensure_ascii=False, sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    print(f"{len(facts)} clean facts, {len(conflicts)} withheld (conflict/context)")
    for key, values in conflicts.items():
        print(f"  withheld {key}: " + " vs ".join(
            f"{v}[{','.join(d)}]" for v, d in values.items()))
    print(f"written to {OUT_PATH}")


if __name__ == "__main__":
    main()
