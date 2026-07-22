#!/usr/bin/env python3
"""Azure SSML phoneme probe v1 — renderer survey, not frozen evidence.

Answers, in one run against a caller-supplied Azure Speech key:
  1. determinism — is the same SSML byte-identical across repeat renders?
  2. pair locality — does a one-word IPA swap change only that region?
  3. phoneme fidelity — do en-US and pt-BR voices honor in-inventory IPA,
     including a word-initial-/ʒ/ nonce carrier word?
  4. cross-inventory behavior — does an out-of-set phone (ɑ into pt-BR)
     render or fail closed with HTTP 400?
  5. acoustic direction — Praat F1/F2 movement of the differing region,
     using the repo's existing measurement stack.

Zero renders happen without AZURE_SPEECH_KEY and AZURE_SPEECH_REGION in the
environment; without them the script prints the exact SSML plan and exits.
Cost at current F0/S0 pricing rounds to zero: 13 short renders.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import urllib.error
import urllib.request
import wave
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

SAMPLE_RATE_HZ = 24_000
OUTPUT_FORMAT = "riff-24khz-16bit-mono-pcm"
OUT_DIR = REPO / "artifacts" / "azure-phoneme-probe-v1"
EN_VOICE = os.environ.get("AZURE_PROBE_EN_VOICE", "en-US-AvaNeural")
PT_VOICE = os.environ.get("AZURE_PROBE_PT_VOICE", "pt-BR-FranciscaNeural")
DIFF_PAD_S = 0.020


def _ssml(lang: str, voice: str, body: str) -> str:
    return (
        '<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" '
        f'xml:lang="{lang}"><voice name="{voice}">{body}</voice></speak>'
    )


def _ph(word: str, ph: str) -> str:
    return f'<phoneme alphabet="ipa" ph="{ph}">{word}</phoneme>'


def _conditions() -> list[dict[str, object]]:
    en = lambda body: _ssml("en-US", EN_VOICE, body)  # noqa: E731
    pt = lambda body: _ssml("pt-BR", PT_VOICE, body)  # noqa: E731
    return [
        {
            "condition_id": "en-real",
            "expectation": "ae->eh: F1 down, F2 up on both target words",
            "neutral": en(f"The {_ph('cat', 'kæt')} {_ph('naps', 'næps')}."),
            "lens": en(f"The {_ph('cat', 'kɛt')} {_ph('naps', 'nɛps')}."),
        },
        {
            "condition_id": "en-nonce",
            "expectation": "nonce word-initial ʒ carrier renders; ae->eh shift",
            "neutral": en(
                f"They kept the {_ph('zhast', 'ʒæst')} hidden well."
            ),
            "lens": en(f"They kept the {_ph('zhast', 'ʒɛst')} hidden well."),
        },
        {
            "condition_id": "pt-real",
            "expectation": "o->ɔ: F1 up on the target vowel",
            "neutral": pt(f"O {_ph('povo', 'povu')} corre."),
            "lens": pt(f"O {_ph('povo', 'pɔvu')} corre."),
        },
    ]


CROSS_INVENTORY = {
    "condition_id": "pt-cross-inventory",
    "expectation": "ɑ is not in the pt-BR phone set; expect HTTP 400 or a render",
    "ssml": _ssml("pt-BR", PT_VOICE, f"O {_ph('povo', 'pɑvu')} corre."),
}


def _render(key: str, region: str, ssml: str, destination: Path) -> dict[str, object]:
    request = urllib.request.Request(
        f"https://{region}.tts.speech.microsoft.com/cognitiveservices/v1",
        data=ssml.encode("utf-8"),
        headers={
            "Ocp-Apim-Subscription-Key": key,
            "Content-Type": "application/ssml+xml",
            "X-Microsoft-OutputFormat": OUTPUT_FORMAT,
            "User-Agent": "build-week-azure-phoneme-probe/1",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = response.read()
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", "replace")[:500]
        return {"http_status": error.code, "error_body": detail, "rendered": False}
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(payload)
    return {
        "http_status": 200,
        "rendered": True,
        "bytes": len(payload),
        "wav_sha256": hashlib.sha256(payload).hexdigest(),
    }


def _pcm(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as handle:
        if handle.getframerate() != SAMPLE_RATE_HZ or handle.getnchannels() != 1:
            raise RuntimeError(f"unexpected audio format in {path.name}")
        frames = handle.readframes(handle.getnframes())
    return np.frombuffer(frames, dtype="<i2")


def _diff_region(neutral: np.ndarray, lens: np.ndarray) -> dict[str, object]:
    length = min(neutral.size, lens.size)
    differing = np.nonzero(neutral[:length] != lens[:length])[0]
    tail = neutral.size != lens.size
    if differing.size == 0 and not tail:
        return {"identical": True}
    first = int(differing[0]) if differing.size else length
    last = int(differing[-1]) if differing.size else length
    if tail:
        last = max(last, length - 1)
    span = (last - first + 1) / SAMPLE_RATE_HZ
    return {
        "identical": False,
        "equal_length": not tail,
        "first_diff_s": first / SAMPLE_RATE_HZ,
        "last_diff_s": (last + 1) / SAMPLE_RATE_HZ,
        "diff_span_s": span,
        "diff_fraction_of_file": span / (neutral.size / SAMPLE_RATE_HZ),
    }


def _formants(path: Path, start_s: float, end_s: float) -> dict[str, object]:
    from earshift_bakeoff.kokoro_typed_confirmation_protocol import CEILINGS_HZ
    from earshift_bakeoff.kokoro_typed_diagnostic import measure_interval_windows

    interval = {"start_s": start_s, "end_s": end_s}
    rows: dict[str, object] = {}
    for ceiling in CEILINGS_HZ:
        windows = measure_interval_windows(path, interval, ceiling)
        primary = windows["50"]
        rows[str(ceiling)] = {
            "measurement_valid": bool(primary.get("measurement_valid")),
            "f1_hz": primary.get("f1_hz"),
            "f2_hz": primary.get("f2_hz"),
            "f1_bark": primary.get("f1_bark"),
            "f2_bark": primary.get("f2_bark"),
        }
    return rows


def main() -> None:
    key = os.environ.get("AZURE_SPEECH_KEY", "")
    region = os.environ.get("AZURE_SPEECH_REGION", "")
    conditions = _conditions()
    if not key or not region:
        print("DRY RUN — set AZURE_SPEECH_KEY and AZURE_SPEECH_REGION to render.")
        print(f"Planned renders: {len(conditions) * 4 + 1} "
              f"({len(conditions)} pairs x2 variants x2 repeats + 1 cross-inventory)")
        for condition in conditions:
            print(f"\n[{condition['condition_id']}] {condition['expectation']}")
            print(f"  neutral: {condition['neutral']}")
            print(f"  lens:    {condition['lens']}")
        print(f"\n[{CROSS_INVENTORY['condition_id']}] {CROSS_INVENTORY['expectation']}")
        print(f"  ssml:    {CROSS_INVENTORY['ssml']}")
        return

    report: dict[str, object] = {"conditions": {}, "output_format": OUTPUT_FORMAT}
    render_count = 0
    for condition in conditions:
        cid = str(condition["condition_id"])
        entry: dict[str, object] = {"expectation": condition["expectation"]}
        files: dict[str, Path] = {}
        failed = False
        for variant in ("neutral", "lens"):
            for take in (1, 2):
                destination = OUT_DIR / f"{cid}__{variant}-take{take}.wav"
                result = _render(key, region, str(condition[variant]), destination)
                render_count += 1
                entry[f"{variant}_take{take}"] = result
                if not result.get("rendered"):
                    failed = True
                files[f"{variant}{take}"] = destination
        if failed:
            entry["verdict"] = "render_failed_see_http_status"
            report["conditions"][cid] = entry  # type: ignore[index]
            continue
        neutral1, neutral2 = _pcm(files["neutral1"]), _pcm(files["neutral2"])
        lens1, lens2 = _pcm(files["lens1"]), _pcm(files["lens2"])
        entry["determinism"] = {
            "neutral_repeat_identical": bool(np.array_equal(neutral1, neutral2)),
            "lens_repeat_identical": bool(np.array_equal(lens1, lens2)),
        }
        region_info = _diff_region(neutral1, lens1)
        entry["pair_locality"] = region_info
        if not region_info.get("identical"):
            start = max(0.0, float(region_info["first_diff_s"]) - DIFF_PAD_S)
            end = float(region_info["last_diff_s"]) + DIFF_PAD_S
            try:
                entry["formants"] = {
                    "measured_interval_s": [start, end],
                    "neutral": _formants(files["neutral1"], start, end),
                    "lens": _formants(files["lens1"], start, end),
                }
            except Exception as exc:  # Praat optional for the probe
                entry["formants"] = {"unavailable": f"{type(exc).__name__}: {exc}"}
        report["conditions"][cid] = entry  # type: ignore[index]

    cross = _render(
        key,
        region,
        str(CROSS_INVENTORY["ssml"]),
        OUT_DIR / "pt-cross-inventory.wav",
    )
    render_count += 1
    report["cross_inventory"] = {
        "expectation": CROSS_INVENTORY["expectation"],
        **cross,
    }
    report["total_renders"] = render_count

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    print(f"\nWAVs + report.json in {OUT_DIR}")
    print("Listen to the pairs yourself — the script never rates audibility.")


if __name__ == "__main__":
    main()
