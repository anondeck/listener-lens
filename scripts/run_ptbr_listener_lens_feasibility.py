from __future__ import annotations

import argparse
import json

from earshift_bakeoff.ptbr_listener_lens_feasibility import analyze, render


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the frozen reciprocal Portuguese feasibility chain."
    )
    parser.add_argument("stage", choices=("render", "analyze"))
    args = parser.parse_args()
    result = render() if args.stage == "render" else analyze()
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
