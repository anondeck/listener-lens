from __future__ import annotations

import argparse
import json

from earshift_bakeoff.kokoro_voice_screen_v1 import prepare_screen, render_screen


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze or execute the zero-API bilingual Kokoro voice screen. "
            "Commit the prepared protocol before rendering."
        )
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--prepare",
        action="store_true",
        help="Freeze protocol.json without rendering audio.",
    )
    mode.add_argument(
        "--render",
        action="store_true",
        help="Execute each frozen render slot exactly once.",
    )
    args = parser.parse_args()
    result = prepare_screen() if args.prepare else render_screen()
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
