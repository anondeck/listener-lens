#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from earshift_bakeoff.bilingual_typed_coverage import prepare, run


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "run"))
    args = parser.parse_args()
    result = prepare() if args.command == "prepare" else run()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
