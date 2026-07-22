#!/usr/bin/env python3
from __future__ import annotations

import json

from earshift_bakeoff.bilingual_g2p_reachability import write_characterization


def main() -> None:
    result = write_characterization()
    print(
        json.dumps(
            {
                "run_id": result["run_id"],
                "status": result["status"],
                "api_calls_made": result["api_calls_made"],
                "profiles": [
                    {
                        "language": row["language"],
                        "canonical_word_count": row["canonical_word_count"],
                        "analyzed_word_count": row["analyzed_word_count"],
                        "analysis_error_count": row["analysis_error_count"],
                        "observed_changed_rule_count": row[
                            "observed_changed_rule_count"
                        ],
                        "changed_rule_count": row["changed_rule_count"],
                        "elapsed_s": row["elapsed_s"],
                    }
                    for row in result["profiles"]
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
