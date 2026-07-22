#!/usr/bin/env python3
"""Per-symbol Azure acceptance receipts for every locale in the matrix.

Renders each mapped symbol inside a minimal carrier on that locale's own
voice and records the HTTP verdict. A 200 is an acceptance receipt for that
symbol *on that voice only* — receipts never transfer across locales, because
each direction renders on its source language's voice.

A 400 is the useful outcome: Azure fails closed on a phone a voice cannot
produce, which is exactly how an unrenderable cross-inventory lens target
gets caught before it ships as a rule that silently does nothing.

Resumable: existing receipts are kept, so an interrupted run continues where
it stopped. Audio bytes are not retained — the receipt is the status.

Needs AZURE_SPEECH_KEY / AZURE_SPEECH_REGION (environment or .env.local).
"""

from __future__ import annotations

import json
import sys
import time
import unicodedata
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from earshift_bakeoff.azure_lens_builder import (  # noqa: E402
    load_ipa_map,
    load_local_env,
    render_ssml_bytes,
)
from lens_language_data_v1 import AZURE_VOICE, LOCALES  # noqa: E402

OUT_PATH = REPO / "artifacts" / "azure-matrix-probe-v1" / "receipts.json"

# A neutral carrier vowel the symbol can sit against. Anything the voice
# certainly owns; the point is to isolate the probed symbol.
CARRIER = "a"

MAX_RETRIES = 6


def carrier_ph(symbol: str) -> str:
    """Minimal renderable string containing the symbol."""

    # Combining marks and length need a vowel to attach to.
    if symbol in {"̃", "̪", "̝", "̩", "ʲ", "ʰ", "ː"}:
        return f"{CARRIER}{symbol}"
    return f"{CARRIER}{symbol}{CARRIER}"


def probe(locale: str, voice: str, ph: str, key: str, region: str) -> dict:
    ssml = (
        '<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" '
        f'xml:lang="{locale}"><voice name="{voice}">'
        f'<phoneme alphabet="ipa" ph="{ph}">a</phoneme>.</voice></speak>'
    )
    for attempt in range(MAX_RETRIES):
        result = render_ssml_bytes(ssml, key=key, region=region)
        if result["rendered"]:
            return {"probe": "accepted", "http_status": 200}
        if result["http_status"] != 429:
            return {
                "probe": "rejected",
                "http_status": result["http_status"],
                "error": result.get("error_body", "")[:160],
            }
        time.sleep(2 * (attempt + 1))
    return {"probe": "throttled", "http_status": 429}


def main() -> None:
    import os

    environment = {**load_local_env(), **os.environ}
    key = environment.get("AZURE_SPEECH_KEY", "")
    region = environment.get("AZURE_SPEECH_REGION", "")
    if not key or not region:
        raise SystemExit("AZURE_SPEECH_KEY / AZURE_SPEECH_REGION are required")

    report: dict = {"schema_version": 1, "locales": {}}
    if OUT_PATH.is_file():
        report = json.loads(OUT_PATH.read_text(encoding="utf-8"))

    table_by_locale = load_ipa_map()
    selected = [loc for loc in sys.argv[1:] if loc in LOCALES] or list(LOCALES)

    for locale in selected:
        table = table_by_locale.get(locale, {})
        voice = AZURE_VOICE[locale]
        rows = report["locales"].setdefault(locale, {})
        pending = [
            (symbol, spec)
            for symbol, spec in sorted(table.items())
            if spec["fidelity"] not in ("structural", "drop")
            and spec["azure_ipa"]
            and symbol not in rows
            # A combining mark is never sent on its own — it rides attached to
            # the segment before it. Rendering one alone in the carrier asks
            # Azure to pronounce a floating diacritic, which it rejects, so
            # probing them produced five standing rejections that described
            # the probe rather than the map.
            and not all(unicodedata.category(ch) == "Mn" for ch in symbol)
        ]
        if not pending:
            print(f"{locale:<7} already complete ({len(rows)} receipts)")
            continue
        for symbol, spec in pending:
            verdict = probe(
                locale, voice, carrier_ph(spec["azure_ipa"]), key, region
            )
            rows[symbol] = {
                "azure_ipa": spec["azure_ipa"],
                "fidelity": spec["fidelity"],
                **verdict,
            }
            OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
            OUT_PATH.write_text(
                json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True)
                + "\n",
                encoding="utf-8",
            )
            time.sleep(0.25)
        accepted = sum(1 for r in rows.values() if r["probe"] == "accepted")
        rejected = sum(1 for r in rows.values() if r["probe"] == "rejected")
        print(
            f"{locale:<7} accepted={accepted:>3} rejected={rejected:>3} "
            f"total={len(rows):>3}",
            flush=True,
        )

    total = sum(len(r) for r in report["locales"].values())
    acc = sum(
        1 for r in report["locales"].values() for v in r.values()
        if v["probe"] == "accepted"
    )
    print(f"\n{acc}/{total} symbols accepted across {len(report['locales'])} locales")
    print(f"written to {OUT_PATH}")


if __name__ == "__main__":
    main()
