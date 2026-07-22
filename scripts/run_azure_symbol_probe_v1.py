#!/usr/bin/env python3
"""Per-symbol Azure acceptance receipts for the IPA map.

Renders every distinct non-structural mapped output of
`rules/azure-ipa-map-v1.json` inside a minimal locale carrier and records
the HTTP verdict per symbol. A 200 is an acceptance receipt; a 400 marks a
mapping the fail-closed service could never render and therefore must be
fixed. Audio bytes are not retained — the receipt is the status.

Needs AZURE_SPEECH_KEY / AZURE_SPEECH_REGION (environment or .env.local).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from earshift_bakeoff.azure_lens_builder import (  # noqa: E402
    DEFAULT_VOICES,
    load_ipa_map,
    load_local_env,
    render_ssml_bytes,
)

OUT_PATH = REPO / "artifacts" / "azure-symbol-probe-v1" / "report.json"
OUT_PATH_BY_LOCALE = REPO / "artifacts" / "azure-symbol-probe-v1"
CARRIER_VOWEL = {"en-US": "ə", "pt-BR": "a", "it-IT": "a", "de-DE": "ə", "es-ES": "a"}


def main() -> None:
    import os

    environment = {**load_local_env(), **os.environ}
    key = environment.get("AZURE_SPEECH_KEY", "")
    region = environment.get("AZURE_SPEECH_REGION", "")
    if not key or not region:
        raise SystemExit("AZURE_SPEECH_KEY / AZURE_SPEECH_REGION are required")

    report: dict[str, object] = {"schema_version": 1, "locales": {}, "renders": 0}
    renders = 0
    selected = set(sys.argv[1:])
    for locale, table in load_ipa_map().items():
        if selected and locale not in selected:
            continue
        voice = DEFAULT_VOICES[locale]
        vowel = CARRIER_VOWEL[locale]
        rows: dict[str, object] = {}
        for symbol, entry in sorted(table.items()):
            if entry["fidelity"] in ("structural", "drop") or not entry["azure_ipa"]:
                rows[symbol] = {
                    "azure_ipa": entry["azure_ipa"],
                    "fidelity": entry["fidelity"],
                    "probe": "not_applicable",
                }
                continue
            ph = f"t{vowel}{entry['azure_ipa']}t{vowel}"
            ssml = (
                '<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" '
                f'xml:lang="{locale}"><voice name="{voice}">'
                f'<phoneme alphabet="ipa" ph="{ph}">test</phoneme>.</voice></speak>'
            )
            # A 429 is free-tier throttling, not a phone verdict; retry with
            # backoff so the receipt records accept (200) versus reject (400).
            for attempt in range(8):
                result = render_ssml_bytes(ssml, key=key, region=region)
                renders += 1
                if result["http_status"] != 429:
                    break
                time.sleep(5.0 + attempt * 5.0)
            time.sleep(1.0)
            rows[symbol] = {
                "azure_ipa": entry["azure_ipa"],
                "fidelity": entry["fidelity"],
                "carrier_ph": ph,
                "http_status": result["http_status"],
                "accepted": bool(result["rendered"]),
                "byte_count": len(result.get("wav_bytes", b"")),
            }
        accepted = sum(1 for row in rows.values() if row.get("accepted"))
        probed = sum(1 for row in rows.values() if "accepted" in row)
        report["locales"][locale] = {  # type: ignore[index]
            "voice": voice,
            "probed_symbol_count": probed,
            "accepted_symbol_count": accepted,
            "rejected_symbols": sorted(
                symbol
                for symbol, row in rows.items()
                if "accepted" in row and not row["accepted"]
            ),
            "symbols": rows,
        }
    report["renders"] = renders
    out = OUT_PATH if not selected else (
        OUT_PATH_BY_LOCALE / ("report-" + "-".join(sorted(selected)) + ".json")
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    for locale, row in report["locales"].items():  # type: ignore[union-attr]
        print(
            f"{locale}: {row['accepted_symbol_count']}/{row['probed_symbol_count']} accepted;"
            f" rejected: {row['rejected_symbols'] or 'none'}"
        )
    print(f"{renders} renders · report at {out}")


if __name__ == "__main__":
    main()
