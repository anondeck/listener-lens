"""Deterministic phonotactic corpus compiler for the renderer bake-off.

GPT-5.6, via Codex, authored this compiler. It emits candidate gibberish
scripts from the per-language phonotactic inventories below, constrained by
rules/phonotactics.yaml, and writes corpus/candidates.json with bound
provenance checksums. It calls no model API. The local wordfreq and eSpeak
gates — not this compiler and not any model — make all acceptance decisions.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROMPT = ROOT / "corpus" / "PROMPT.md"
RULES = ROOT / "rules" / "phonotactics.yaml"
OUTPUT = ROOT / "corpus" / "candidates.json"

PROFILES = {
    "en-US-mae": {
        "onsets": ["b", "br", "d", "dr", "f", "fl", "g", "gr", "k", "kl", "m", "n", "p", "pl", "s", "sk", "t", "tr", "v", "z"],
        "vowels": ["a", "e", "i", "o", "u", "ae", "oo"],
        "codas": ["m", "n", "p", "t", "k", "v", "z", "sh", "nd", "mp"],
        "rules": ["en.templates", "en.reduction", "en.rhotic", "en.prosody"],
    },
    "es-MX-cdmx": {
        "onsets": ["b", "ch", "d", "f", "g", "k", "l", "m", "n", "p", "r", "rr", "s", "t", "v", "y"],
        "vowels": ["a", "e", "i", "o", "u"],
        "codas": ["n", "s", "l", "r"],
        "rules": ["es.vowels", "es.templates", "es.rhotics", "es.prosody"],
    },
    "pt-BR-sp": {
        "onsets": ["b", "ch", "d", "f", "g", "j", "lh", "m", "n", "nh", "p", "r", "s", "t", "v", "z"],
        "vowels": ["a", "e", "i", "o", "u", "é", "ó"],
        "codas": ["m", "n", "r", "s", "l"],
        "rules": ["pt.vowels", "pt.templates", "pt.connected", "pt.prosody"],
    },
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def choose(values: list[str], digest: bytes, offset: int) -> str:
    return values[digest[offset] % len(values)]


def nonce(profile: str, seed: str) -> str:
    spec = PROFILES[profile]
    digest = hashlib.sha256(seed.encode("utf-8")).digest()
    first = (
        choose(spec["onsets"], digest, 0)
        + choose(spec["vowels"], digest, 1)
        + choose(spec["codas"], digest, 2)
    )
    second = (
        choose(spec["onsets"], digest, 3)
        + choose(spec["vowels"], digest, 4)
        + choose(spec["codas"], digest, 5)
    )
    return first + second


def token(profile: str, surface: str, role: str) -> dict[str, object]:
    stress = 0 if profile == "en-US-mae" else 1
    return {
        "surface": surface,
        "role": role,
        "intended_ipa": f"/{surface}/",
        "syllables": 2,
        "primary_stress_index": stress,
        "rule_ids": PROFILES[profile]["rules"],
    }


def candidate(profile: str, round_index: int, index: int) -> dict[str, object]:
    stem = f"{profile}:{round_index}:{index}"
    content = [nonce(profile, f"{stem}:content:{i}") for i in range(12)]
    fillers = [nonce(profile, f"{stem}:filler:{i}") for i in range(3)]
    surfaces: list[tuple[str, str]] = []
    for phrase_unit in range(6):
        surfaces.extend(
            [
                (content[phrase_unit * 2], "content"),
                (content[phrase_unit * 2 + 1], "content"),
                (fillers[phrase_unit % 3], "filler"),
            ]
        )
    language = {"en-US-mae": "en", "es-MX-cdmx": "es", "pt-BR-sp": "pt"}[profile]
    return {
        "candidate_id": f"codex-{language}-r{round_index}-{index + 1:02d}",
        "profile_id": profile,
        "tokens": [token(profile, surface, role) for surface, role in surfaces],
        "punctuation_after_token": {"8": ",", "17": "."},
    }


def main() -> None:
    rounds = []
    for round_index, count in enumerate((20, 10, 10)):
        rounds.append(
            {
                "languages": [
                    {
                        "profile_id": profile,
                        "candidates": [
                            candidate(profile, round_index, index)
                            for index in range(count)
                        ],
                    }
                    for profile in PROFILES
                ]
            }
        )
    payload = {
        "schema_version": 1,
        "provenance": {
            "source": "codex",
            "model_label": "gpt-5.6 via Codex",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "prompt_sha256": sha256(PROMPT),
            "rules_sha256": sha256(RULES),
            "notes": "Codex-authored deterministic phonotactic inventory compiler; local gates make acceptance decisions.",
        },
        "rounds": rounds,
    }
    OUTPUT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {OUTPUT} with {sum(len(x['candidates']) for r in rounds for x in r['languages'])} candidates")


if __name__ == "__main__":
    main()
