#!/usr/bin/env python3
"""Do Azure voices honor IPA stress position at all?

Every segmental rule now carries a distinctness receipt, but the prosody
family rides on an untested assumption: that moving ˈ inside a ph attribute
moves the accent. 462 matrix rules (the stress-bias operations) claim exactly
that. If a voice ignores stress position, every one of them on that voice is
a silent no-op reported as applied — the same fail-open class the ʋ/v
discovery exposed for segments.

One minimal pair per voice: ˈama versus aˈma, identical phones, only the
stress mark moves. /a/ and /m/ exist in all thirty languages, so the carrier
is native everywhere. Byte-identical PCM means the voice does not render
stress position and the receipt says so.

Resumable; audio not retained. Needs AZURE_SPEECH_KEY / AZURE_SPEECH_REGION.
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
    load_local_env,
    render_ssml_bytes,
)
from lens_language_data_v1 import AZURE_VOICE  # noqa: E402

OUT_PATH = REPO / "artifacts" / "azure-stress-probe-v1" / "receipts.json"

PAIR = ("ˈama", "aˈma")
MAX_RETRIES = 6


def render_hash(locale: str, voice: str, ph: str, key: str, region: str) -> str:
    ssml = (
        '<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" '
        f'xml:lang="{locale}"><voice name="{voice}">'
        f'<phoneme alphabet="ipa" ph="{ph}">ama</phoneme>.</voice></speak>'
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

    report: dict = {
        "schema_version": 1,
        "pair": list(PAIR),
        "locales": {},
    }
    if OUT_PATH.is_file():
        report = json.loads(OUT_PATH.read_text(encoding="utf-8"))
    rows = report["locales"]

    for locale, voice in sorted(AZURE_VOICE.items()):
        if locale in rows:
            continue
        initial = render_hash(locale, voice, PAIR[0], key, region)
        time.sleep(0.2)
        second = render_hash(locale, voice, PAIR[1], key, region)
        time.sleep(0.2)
        if initial.startswith("HTTP") or second.startswith("HTTP") or "THROTTLED" in (initial, second):
            rows[locale] = {"voice": voice, "verdict": "error",
                            "initial": initial, "second": second}
        else:
            rows[locale] = {
                "voice": voice,
                "verdict": "honoured" if initial != second else "ignored",
            }
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUT_PATH.write_text(
            json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"{locale:<7} {rows[locale]['verdict']}", flush=True)

    counts: dict[str, int] = {}
    for row in rows.values():
        counts[row["verdict"]] = counts.get(row["verdict"], 0) + 1
    print(f"\nfinal: {counts}")
    ignored = sorted(loc for loc, row in rows.items() if row["verdict"] == "ignored")
    if ignored:
        print("voices ignoring stress position:", " ".join(ignored))
    print(f"written to {OUT_PATH}")


if __name__ == "__main__":
    main()
