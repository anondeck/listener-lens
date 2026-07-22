#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from earshift_bakeoff.bilingual_vowel_unseen_blind_qc import (
    RUN_DIR,
    adjudicate_session_response,
)
from earshift_bakeoff.util import atomic_write_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("response", type=Path)
    args = parser.parse_args()
    result = adjudicate_session_response(args.response)
    target = RUN_DIR / "responses" / f"{result['session_id']}-adjudication.json"
    if target.exists():
        raise RuntimeError(f"refusing to overwrite adjudication: {target}")
    atomic_write_json(target, result)
    print(f"wrote {target}")
    print(f"human_qc_pass_cells={result['human_qc_pass_cell_count']}")
    print(f"record_sha256={result['record_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
