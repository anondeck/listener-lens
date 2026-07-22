#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from earshift_bakeoff.kokoro_typed_diagnostic import run
from earshift_bakeoff.kokoro_typed_diagnostic_protocol import prepare


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "run"))
    args = parser.parse_args()
    result = prepare() if args.command == "prepare" else run()
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
