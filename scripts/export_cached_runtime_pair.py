from __future__ import annotations

import argparse
import base64
import hashlib
import json
import urllib.request
from pathlib import Path


EXPECTED = {
    "neutral": "2ed8f2023db0b61ae7996ce17194e7dd84762c8e49f7daaece965d8dc4873a41",
    "lens": "48e47fb1ce64a3322b7f1b92b7a7a625ec043d5a928df406e0829cf6a6dc7116",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export the frozen selected runtime pair from a zero-call local cache hit."
    )
    parser.add_argument("--url", default="http://127.0.0.1:8789/api/listener-lens")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payload = json.dumps(
        {
            "text": "What a great day it is to catch some sun.",
            "profile_id": "en-to-pt-BR-vowel-lens",
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        args.url,
        data=payload,
        headers={"content-type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        body = json.load(response)

    if body.get("status") != "ready" or body.get("cache_hit") is not True:
        raise RuntimeError("expected a ready cache hit")
    if body.get("api_calls_made") != 0:
        raise RuntimeError("refusing to export a response that made an API call")

    args.output.mkdir(parents=True, exist_ok=False)
    audio_dir = args.output / "audio"
    audio_dir.mkdir()
    manifest = {
        "schema_version": 1,
        "status": "frozen_cached_pair_exported_without_api_call",
        "transform_cache_key": body["transform"]["cache_key"],
        "scripts": {
            "neutral": body["transform"]["neutral_script"],
            "lens": body["transform"]["lens_script"],
        },
        "target_word_index_zero_based": 7,
        "audio": {},
        "selection": body["selection"],
        "renderer": body["renderer"],
        "verification": body["verification"],
        "cache_hit": True,
        "api_calls_made": 0,
    }
    for side in ("neutral", "lens"):
        encoded = body["audio"][side]["base64"]
        wav = base64.b64decode(encoded, validate=True)
        digest = hashlib.sha256(wav).hexdigest()
        if digest != EXPECTED[side] or digest != body["audio"][side]["sha256"]:
            raise RuntimeError(f"{side} audio hash mismatch")
        path = audio_dir / f"{side}.wav"
        path.write_bytes(wav)
        manifest["audio"][side] = {
            "path": str(path.relative_to(args.output)),
            "sha256": digest,
            "mime_type": body["audio"][side]["mime_type"],
        }

    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
