#!/usr/bin/env python3
"""Build (and optionally render) one Azure neutral/lens pair from typed text.

Examples:
  uv run python scripts/run_azure_lens_pair_v1.py --text "The cat naps" \
      --profile en-US-to-pt-BR-listener-v2
  ... --render   # also writes pair WAVs (needs AZURE_SPEECH_KEY/REGION,
                 # read from the environment or .env.local)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from earshift_bakeoff.azure_lens_builder import (  # noqa: E402
    build_pair,
    load_local_env,
    render_ssml,
)

OUT_DIR = REPO / "artifacts" / "azure-lens-pairs"


def main() -> None:
    parser = argparse.ArgumentParser(description="Azure lens pair v1")
    parser.add_argument("--text", required=True)
    parser.add_argument(
        "--profile",
        default="en-US-to-pt-BR-listener-v2",
        choices=("en-US-to-pt-BR-listener-v2", "pt-BR-to-en-US-listener-v2"),
    )
    parser.add_argument("--voice", default=None)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    pair = build_pair(args.text, args.profile, voice=args.voice)
    if args.render:
        env = {**load_local_env(), **os.environ}
        key = env.get("AZURE_SPEECH_KEY", "")
        region = env.get("AZURE_SPEECH_REGION", "")
        if not key or not region:
            parser.error("--render needs AZURE_SPEECH_KEY/AZURE_SPEECH_REGION")
        slug = "".join(
            ch if ch.isalnum() else "-" for ch in pair["normalized_text"].lower()
        ).strip("-")[:40]
        renders = {}
        for side in ("neutral", "lens"):
            destination = OUT_DIR / f"{slug}__{side}.wav"
            renders[side] = {
                **render_ssml(pair[f"ssml_{side}"], destination, key=key, region=region),
                "path": str(destination),
            }
        pair["renders"] = renders
        pair["api_calls_made"] = 2
    print(json.dumps(pair, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
