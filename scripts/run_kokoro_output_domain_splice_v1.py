#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from earshift_bakeoff.kokoro_output_domain_splice import adjudicate, prepare, run


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze or run the bounded output-domain splice v1 spike."
    )
    parser.add_argument("command", choices=("prepare", "run", "adjudicate"))
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare()
    elif args.command == "run":
        result = run()
    else:
        result = adjudicate()
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
