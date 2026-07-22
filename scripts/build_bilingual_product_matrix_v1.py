#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from typing import Any

from earshift_bakeoff.bilingual_product_matrix import (
    BILINGUAL_PRODUCT_MATRIX_PATH,
    load_bilingual_product_matrix,
)
from earshift_bakeoff.config import Paths, stable_json
from earshift_bakeoff.util import atomic_write_json, sha256_file


RUN_ID = "20260717-bilingual-product-matrix-v1"
RUN_DIR = Paths().artifacts / "product-matrix" / RUN_ID


def _with_record_hash(payload: dict[str, Any]) -> dict[str, Any]:
    semantic = dict(payload)
    semantic.pop("record_sha256", None)
    return {
        **semantic,
        "record_sha256": hashlib.sha256(
            stable_json(semantic).encode("utf-8")
        ).hexdigest(),
    }


def main() -> None:
    matrix = load_bilingual_product_matrix()
    manifest = matrix.validation_manifest()
    record = _with_record_hash(
        {
            "schema_version": 1,
            "run_id": RUN_ID,
            "classification": (
                "four_voice_bidirectional_structural_matrix_ready_"
                "all_changed_cells_pending_validation"
            ),
            "matrix_config_path": str(
                BILINGUAL_PRODUCT_MATRIX_PATH.relative_to(Paths().root)
            ),
            "matrix_config_sha256": sha256_file(
                BILINGUAL_PRODUCT_MATRIX_PATH
            ),
            "matrix_catalog": matrix.safe_catalog(),
            "validation_manifest": manifest,
            "api_calls_made": 0,
            "audio_renders_made": 0,
            "production_enabled": False,
        }
    )
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    output = RUN_DIR / "manifest.json"
    atomic_write_json(output, record)
    print(
        json.dumps(
            {
                "output": str(output),
                "record_sha256": record["record_sha256"],
                "rule_cell_count": manifest["cell_count"],
                "changed_cell_count": manifest["changed_cell_count"],
                "logical_slot_count": manifest["logical_slot_count"],
                "api_calls_made": 0,
                "audio_renders_made": 0,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
