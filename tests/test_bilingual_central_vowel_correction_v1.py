from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

from earshift_bakeoff.config import Paths, sha256_json


SCRIPT = Paths().root / "scripts" / "run_bilingual_central_vowel_correction_v1.py"


def _module():
    scripts = str(SCRIPT.parent)
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    spec = importlib.util.spec_from_file_location("central_vowel_v1", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_protocol_has_fixed_bilingual_six_cell_denominator() -> None:
    module = _module()
    protocol = module.protocol_record()
    payload = dict(protocol)
    digest = payload.pop("protocol_sha256")
    assert digest == sha256_json(payload)
    assert protocol["scope"]["cell_count"] == 6
    assert protocol["scope"]["logical_slot_count"] == 18
    assert set(protocol["scope"]["rule_ids"]) == {
        "enpt.reduced_schwa_a",
        "enpt.schwa_reduced_a",
        "pten.final_a_schwa",
    }
    assert {cell.split("::")[0] for cell in protocol["scope"]["cell_ids"]} == {
        "en-US-to-pt-BR-listener-v2",
        "pt-BR-to-en-US-listener-v2",
    }
    assert protocol["scope_controls"] == {
        "api_calls": 0,
        "paid_calls": 0,
        "production_enabled": False,
        "deployment": False,
    }


def test_strength_order_is_bounded_and_regression_control_is_present() -> None:
    module = _module()
    protocol = module.protocol_record()
    assert protocol["intervention"]["strength_order"] == [
        1.0,
        0.75,
        1.25,
        0.5,
        1.5,
        2.0,
        2.5,
        3.0,
    ]
    assert "Michael" in protocol["gates"]["regression_control"]
    for path, digest in protocol["source_bindings"].items():
        assert Path(Paths().root / path).is_file()
        assert len(digest) == 64
