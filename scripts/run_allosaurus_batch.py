from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path
import sys

from allosaurus.app import read_recognizer


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: run_allosaurus_batch.py MANIFEST.json OUTPUT.json")
    manifest_path = Path(sys.argv[1]).resolve()
    output_path = Path(sys.argv[2]).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    recognizer = read_recognizer()
    rows = []
    for row in manifest["inputs"]:
        path = Path(row["path"]).resolve()
        rows.append(
            {
                "id": row["id"],
                "path": str(path),
                "timestamp_output": recognizer.recognize(
                    str(path), lang_id="ipa", timestamp=True
                ),
            }
        )
    payload = {
        "allosaurus_version": importlib.metadata.version("allosaurus"),
        "scipy_version": importlib.metadata.version("scipy"),
        "rows": rows,
    }
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
