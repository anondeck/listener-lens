from __future__ import annotations

import argparse
import json

from earshift_bakeoff.realtime_transport_correction import prepare_probe, run_probe


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "run"))
    args = parser.parse_args()
    result = prepare_probe() if args.mode == "prepare" else run_probe()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
