from __future__ import annotations

import json

from earshift_bakeoff.runtime_pair_diagnostic import analyze_runtime_pair


if __name__ == "__main__":
    print(json.dumps(analyze_runtime_pair(), indent=2))
