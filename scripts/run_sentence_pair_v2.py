from __future__ import annotations

import json

from earshift_bakeoff.sentence_pair_v2_run import run_sentence_pair_v2


if __name__ == "__main__":
    print(json.dumps(run_sentence_pair_v2(), indent=2))
