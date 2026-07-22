#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from earshift_bakeoff.kokoro_typed_confirmation import decode_response, run
from earshift_bakeoff.kokoro_typed_confirmation_protocol import prepare


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze, run, or decode Kokoro typed confirmation v1."
    )
    parser.add_argument("command", choices=("prepare", "run", "decode"))
    parser.add_argument("response", nargs="?", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare()
    elif args.command == "run":
        result = run()
    else:
        if args.response is None:
            parser.error("decode requires the downloaded response path")
        result = decode_response(args.response)
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
