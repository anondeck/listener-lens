#!/usr/bin/env python3
"""Azure locale feasibility probe for candidate listener-lens languages.

Answers two gating questions per candidate locale, before any rule-table or
IPA-map work is committed to it:

  1. acceptance — does the locale's neural voice return 200 for an
     ``<phoneme alphabet="ipa">`` override at all?
  2. honouring — does the override actually change the audio? A voice can
     return 200 and silently ignore the tag, which would make the whole lens
     architecture a no-op for that language. Two deliberately different ph
     strings must produce two different waveforms.

Also reports the espeak-ng phoneme inventory each language emits, so the
per-locale IPA-map work can be sized (and ligature/tie normalisation spotted)
without guessing.

Audio bytes are not retained — the receipt is the status plus a digest.
Needs AZURE_SPEECH_KEY / AZURE_SPEECH_REGION (environment or .env.local).
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from earshift_bakeoff.azure_lens_builder import (  # noqa: E402
    load_local_env,
    render_ssml_bytes,
)

OUT_PATH = REPO / "artifacts" / "azure-locale-feasibility-v1" / "report.json"

# Candidate locales: Azure voice + the espeak-ng language code that feeds it.
CANDIDATES = {
    "es-ES": {"voice": "es-ES-ElviraNeural", "espeak": "es", "sample": "El gato duerme en la casa"},
    "fr-FR": {"voice": "fr-FR-DeniseNeural", "espeak": "fr-fr", "sample": "Le chat dort dans la maison"},
    "de-DE": {"voice": "de-DE-KatjaNeural", "espeak": "de", "sample": "Die Katze schläft im Haus"},
    "it-IT": {"voice": "it-IT-ElsaNeural", "espeak": "it", "sample": "Il gatto dorme nella casa"},
}

# Deliberately distinct ph strings built from universally available symbols,
# so a null result means "ignored", never "unsupported exotic symbol".
PROBE_A = "tata"
PROBE_B = "titi"


def speak(body: str, locale: str, voice: str) -> str:
    return (
        '<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" '
        f'xml:lang="{locale}"><voice name="{voice}">{body}</voice></speak>'
    )


def render(ssml: str, key: str, region: str) -> dict[str, object]:
    for attempt in range(6):
        result = render_ssml_bytes(ssml, key=key, region=region)
        if result["http_status"] != 429:
            break
        time.sleep(2.0 + attempt * 2.0)
    time.sleep(0.25)
    payload = result.get("wav_bytes", b"")
    return {
        "http_status": result["http_status"],
        "rendered": bool(result["rendered"]),
        "byte_count": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest() if payload else None,
    }


def espeak_inventory(language: str, sample: str) -> dict[str, object]:
    try:
        from misaki.espeak import EspeakG2P

        phones, _ = EspeakG2P(language=language)(sample)
    except Exception as exc:  # pragma: no cover - reported, not raised
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    symbols = sorted({ch for ch in phones if not ch.isspace()})
    # Symbols that a per-locale map will have to normalise before Azure sees
    # them: ligatures and espeak's tie character.
    suspect = sorted(set(symbols) & set("ʦʧʤʣʥʨ^"))
    return {
        "available": True,
        "phones": phones,
        "distinct_symbol_count": len(symbols),
        "symbols": "".join(symbols),
        "needs_normalisation": suspect,
    }


def main() -> None:
    import os

    environment = {**load_local_env(), **os.environ}
    key = environment.get("AZURE_SPEECH_KEY", "")
    region = environment.get("AZURE_SPEECH_REGION", "")
    if not key or not region:
        raise SystemExit("AZURE_SPEECH_KEY / AZURE_SPEECH_REGION are required")

    report: dict[str, object] = {"schema_version": 1, "locales": {}, "renders": 0}
    renders = 0
    for locale, spec in CANDIDATES.items():
        voice = spec["voice"]
        plain = render(speak("teste.", locale, voice), key, region)
        a = render(
            speak(f'<phoneme alphabet="ipa" ph="{PROBE_A}">teste</phoneme>.', locale, voice),
            key,
            region,
        )
        b = render(
            speak(f'<phoneme alphabet="ipa" ph="{PROBE_B}">teste</phoneme>.', locale, voice),
            key,
            region,
        )
        renders += 3
        accepted = bool(a["rendered"] and b["rendered"])
        honoured = bool(accepted and a["sha256"] and a["sha256"] != b["sha256"])
        report["locales"][locale] = {  # type: ignore[index]
            "voice": voice,
            "voice_available": bool(plain["rendered"]),
            "ipa_accepted": accepted,
            "ipa_honoured": honoured,
            "verdict": (
                "viable"
                if honoured
                else "accepts_but_ignores_ipa"
                if accepted
                else "unavailable"
            ),
            "renders": {"plain": plain, "probe_a": a, "probe_b": b},
            "espeak": espeak_inventory(spec["espeak"], spec["sample"]),
        }
    report["renders"] = renders
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    for locale, row in report["locales"].items():  # type: ignore[union-attr]
        espeak = row["espeak"]
        extra = (
            f"{espeak['distinct_symbol_count']} espeak symbols"
            if espeak.get("available")
            else f"espeak unavailable ({espeak.get('error')})"
        )
        norm = espeak.get("needs_normalisation") or "none"
        print(
            f"{locale}: {row['verdict']} · voice={row['voice_available']} "
            f"accepted={row['ipa_accepted']} honoured={row['ipa_honoured']} · "
            f"{extra} · normalise: {norm}"
        )
    print(f"{renders} renders · report at {OUT_PATH}")


if __name__ == "__main__":
    main()
