from __future__ import annotations

import argparse
import json

from earshift_bakeoff.kokoro_specs import PORTUGUESE_SCREEN_VOICES
from earshift_bakeoff.portuguese_carrier_planner_v1 import (
    PortugueseCarrierPlannerV1,
    portuguese_smoke_screening_receipt_v1,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plan the frozen local pt-BR real-text/opaque-carrier smoke fixture. "
            "This does not render audio or enable a production candidate."
        )
    )
    parser.add_argument(
        "--voice-id",
        required=True,
        choices=PORTUGUESE_SCREEN_VOICES,
        help="Explicit pinned pt-BR VoiceSpec to bind to the plan.",
    )
    args = parser.parse_args()

    planner = PortugueseCarrierPlannerV1.load(voice_id=args.voice_id)
    receipt = portuguese_smoke_screening_receipt_v1(planner)
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
