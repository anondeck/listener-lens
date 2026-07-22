#!/usr/bin/env python3
from earshift_bakeoff.portuguese_kokoro_gate import build_full_index


if __name__ == "__main__":
    receipt = build_full_index()
    print(receipt["status"])
