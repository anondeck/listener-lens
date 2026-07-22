#!/usr/bin/env python3
from earshift_bakeoff.config import Paths
from earshift_bakeoff.kokoro_specs import voice_inventory_receipt
from earshift_bakeoff.util import atomic_write_json


RUN_ID = "20260717-kokoro-bilingual-voice-screen-v1"


def main() -> None:
    destination = Paths().artifacts / "voice-screen" / RUN_ID / "inventory.json"
    atomic_write_json(destination, voice_inventory_receipt(download=False))
    print(destination)


if __name__ == "__main__":
    main()
