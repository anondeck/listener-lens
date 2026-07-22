from __future__ import annotations

import argparse
import json

from earshift_bakeoff.kokoro_source_aligned import (
    prepare,
    prepare_measurement_amendment,
    reanalyze_stress_spans,
    run,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "run", "prepare-reanalysis", "reanalyze"))
    args = parser.parse_args()
    if args.mode == "prepare":
        result = prepare()
    elif args.mode == "run":
        result = run()
    elif args.mode == "prepare-reanalysis":
        result = prepare_measurement_amendment()
    else:
        result = reanalyze_stress_spans()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
