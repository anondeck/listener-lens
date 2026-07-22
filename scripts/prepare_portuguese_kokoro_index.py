#!/usr/bin/env python3
from earshift_bakeoff.config import Paths
from earshift_bakeoff.portuguese_kokoro_gate import RUN_ID, protocol_record
from earshift_bakeoff.util import atomic_write_json


def main() -> None:
    destination = Paths().artifacts / "portuguese" / RUN_ID / "protocol.json"
    atomic_write_json(destination, protocol_record())
    print(destination)


if __name__ == "__main__":
    main()
