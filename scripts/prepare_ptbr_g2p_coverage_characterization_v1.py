#!/usr/bin/env python3
from __future__ import annotations

import json

from earshift_bakeoff.ptbr_g2p_coverage_characterization_v1 import prepare


def main() -> None:
    print(json.dumps(prepare(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
