#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from earshift_bakeoff.kokoro_gate_bridge import build_full_index, measure, prepare


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "measure", "build-index"))
    args = parser.parse_args()
    if args.command == "prepare":
        payload = prepare()
    elif args.command == "measure":
        payload = measure()
    else:
        payload = build_full_index()
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
