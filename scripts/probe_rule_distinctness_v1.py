#!/usr/bin/env python3
"""Prove each rule is audible, not merely accepted.

The per-symbol acceptance probe answers "will this voice render this phone?"
It does not answer the question that decides whether a lens does anything:
"does this voice render the rule's source and target *differently*?"

Those come apart. Azure's Hindi voice accepts both ʋ and v with HTTP 200 and
then produces byte-identical audio for them, so the hi-IN -> en-US rule
ʋ->v is a silent no-op — the lens ran, the SSML differed, and the listener
heard nothing. That is the fail-open mode this project cares most about,
because nothing in the pipeline reports an error.

So each distinct (voice, source, target) triple gets rendered twice in the
same carrier and the PCM compared. Identical bytes means the rule cannot be
perceived on that voice and must not be shipped as if it were.

Resumable; audio is not retained. Needs AZURE_SPEECH_KEY / AZURE_SPEECH_REGION.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from earshift_bakeoff.azure_lens_builder import (  # noqa: E402
    _MAP_NORMALIZATION_BY_LOCALE,
    AzureLensBuilderError,
    _map_symbols,
    load_ipa_map,
    load_local_env,
    render_ssml_bytes,
)
from lens_language_data_v1 import AZURE_VOICE  # noqa: E402

MATRIX_PATH = REPO / "rules" / "azure-listener-lenses-v2.json"
OUT_PATH = REPO / "artifacts" / "azure-rule-distinctness-v1" / "receipts.json"

CARRIER = "a"
MAX_RETRIES = 6


def render_hash(locale: str, voice: str, ph: str, key: str, region: str) -> str:
    ssml = (
        '<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" '
        f'xml:lang="{locale}"><voice name="{voice}">'
        f'<phoneme alphabet="ipa" ph="{ph}">a</phoneme>.</voice></speak>'
    )
    for attempt in range(MAX_RETRIES):
        result = render_ssml_bytes(ssml, key=key, region=region)
        if result["rendered"]:
            return hashlib.sha256(result["wav_bytes"]).hexdigest()
        if result["http_status"] != 429:
            return f"HTTP{result['http_status']}"
        time.sleep(2 * (attempt + 1))
    return "THROTTLED"


def main() -> None:
    import os

    environment = {**load_local_env(), **os.environ}
    key = environment.get("AZURE_SPEECH_KEY", "")
    region = environment.get("AZURE_SPEECH_REGION", "")
    if not key or not region:
        raise SystemExit("AZURE_SPEECH_KEY / AZURE_SPEECH_REGION are required")

    maps = load_ipa_map()
    matrix = json.loads(MATRIX_PATH.read_text(encoding="utf-8"))

    triples: dict[str, tuple[str, str, str]] = {}

    def collect(locale: str, rules) -> None:
        for rule in rules or ():
            source, target = rule.get("source"), rule.get("target")
            if rule.get("operation") == "delete" or not target or not source:
                continue
            if source == target:
                continue
            key_id = f"{locale}|{source}|{target}"
            triples.setdefault(key_id, (locale, source, target))

    for profile in matrix["profiles"]:
        locale = profile["source_locale"]
        collect(locale, profile["vowel_rules"] + profile["consonant_rules"])

    # The generated matrix is only one of three rule sources. The curated
    # registry and the frozen bilingual profiles serve the directions most
    # likely to be demonstrated — including en<->pt, whose th-stopping and
    # vowel lens had never been audibility-checked — so probe them too rather
    # than leaving the flagship pair unreceipted.
    curated = json.loads(
        (REPO / "rules" / "azure-listener-lenses-v1.json").read_text(encoding="utf-8")
    )
    for profile in curated.get("profiles", ()):
        collect(
            profile["source_locale"],
            (profile.get("vowel_rules") or []) + (profile.get("consonant_rules") or []),
        )

    from earshift_bakeoff.bilingual_listener_engine import load_listener_profiles

    frozen_locale = {
        "en-US-to-pt-BR-listener-v2": "en-US",
        "pt-BR-to-en-US-listener-v2": "pt-BR",
    }
    listener_profiles = load_listener_profiles()
    for profile_id, locale in frozen_locale.items():
        profile = listener_profiles[profile_id]
        collect(
            locale,
            list(profile.get("vowel_rules") or [])
            + list(profile.get("consonant_rules") or []),
        )

    report: dict = {"schema_version": 1, "rules": {}}
    if OUT_PATH.is_file():
        report = json.loads(OUT_PATH.read_text(encoding="utf-8"))
    rows = report["rules"]

    def mapped(symbol: str, locale: str) -> str | None:
        try:
            return _map_symbols(
                symbol,
                maps[locale],
                context="probe",
                normalization=_MAP_NORMALIZATION_BY_LOCALE.get(locale, "NFD"),
            )
        except AzureLensBuilderError:
            return None

    pending = [k for k in sorted(triples) if k not in rows]
    print(f"{len(rows)} receipts on file, {len(pending)} to probe")
    for index, key_id in enumerate(pending, 1):
        locale, source, target = triples[key_id]
        voice = AZURE_VOICE[locale]
        source_ph, target_ph = mapped(source, locale), mapped(target, locale)
        if source_ph is None or target_ph is None:
            rows[key_id] = {"verdict": "unmappable"}
        elif source_ph == target_ph:
            rows[key_id] = {"verdict": "map_neutralised", "ph": source_ph}
        else:
            a = render_hash(locale, voice, f"{CARRIER}{source_ph}{CARRIER}", key, region)
            time.sleep(0.2)
            b = render_hash(locale, voice, f"{CARRIER}{target_ph}{CARRIER}", key, region)
            time.sleep(0.2)
            if a.startswith("HTTP") or b.startswith("HTTP") or "THROTTLED" in (a, b):
                rows[key_id] = {"verdict": "error", "source": a, "target": b}
            else:
                rows[key_id] = {
                    "verdict": "audible" if a != b else "inaudible",
                    "source_ph": source_ph,
                    "target_ph": target_ph,
                }
        if index % 25 == 0 or index == len(pending):
            OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
            OUT_PATH.write_text(
                json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            counts: dict[str, int] = {}
            for row in rows.values():
                counts[row["verdict"]] = counts.get(row["verdict"], 0) + 1
            print(f"  {index}/{len(pending)} {counts}", flush=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    counts = {}
    for row in rows.values():
        counts[row["verdict"]] = counts.get(row["verdict"], 0) + 1
    print(f"\nfinal: {counts}")
    inaudible = [k for k, v in rows.items() if v["verdict"] == "inaudible"]
    print(f"inaudible rule pairs: {len(inaudible)}")
    for key_id in inaudible[:20]:
        print(f"   {key_id}")
    print(f"written to {OUT_PATH}")


if __name__ == "__main__":
    main()
