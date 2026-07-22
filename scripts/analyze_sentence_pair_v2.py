from __future__ import annotations

import json

from earshift_bakeoff.sentence_pair_v2_analysis import analyze_sentence_pair_v2


if __name__ == "__main__":
    result = analyze_sentence_pair_v2()
    print(json.dumps({key: result[key] for key in ("status", "classification", "complete_carrier_blocks", "measurable_take_count", "individual_pairing_eligible_take_count")}, indent=2))
