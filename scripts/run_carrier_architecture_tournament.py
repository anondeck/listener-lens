from __future__ import annotations

import argparse
import json

from earshift_bakeoff.carrier_architecture_tournament import (
    prepare_tournament,
    run_tournament,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "run"))
    args = parser.parse_args()
    result = prepare_tournament() if args.mode == "prepare" else run_tournament()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
