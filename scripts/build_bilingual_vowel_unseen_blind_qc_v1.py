#!/usr/bin/env python3
from __future__ import annotations

from earshift_bakeoff.bilingual_vowel_unseen_blind_qc import (
    RUN_DIR,
    build_review_package,
)


def main() -> int:
    result = build_review_package()
    print(f"wrote {RUN_DIR / 'index.html'}")
    print(f"trials={sum(result['trial_kind_counts'].values())}")
    print(f"record_sha256={result['record_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
