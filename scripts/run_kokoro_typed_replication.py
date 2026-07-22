#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from earshift_bakeoff.kokoro_typed_replication import (
    analyze,
    build_review,
    decode_response,
    render,
)
from earshift_bakeoff.kokoro_typed_replication_protocol import prepare


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command", choices=("prepare", "render", "analyze", "review", "decode")
    )
    parser.add_argument("response", nargs="?", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare()
    elif args.command == "render":
        result = render()
    elif args.command == "analyze":
        result = analyze()
    elif args.command == "review":
        result = build_review()
    else:
        if args.response is None:
            parser.error("decode requires a downloaded response path")
        result = decode_response(args.response)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
