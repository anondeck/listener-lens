from __future__ import annotations

import argparse
import json

from earshift_bakeoff.kokoro_latent_span_sweep import prepare, run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "run"))
    args = parser.parse_args()
    result = prepare() if args.mode == "prepare" else run()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

