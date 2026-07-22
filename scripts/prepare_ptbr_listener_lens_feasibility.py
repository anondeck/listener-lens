from __future__ import annotations

import json

from earshift_bakeoff.ptbr_listener_lens_feasibility_protocol import prepare


if __name__ == "__main__":
    print(json.dumps(prepare(), ensure_ascii=False, indent=2, sort_keys=True))
