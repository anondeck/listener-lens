from __future__ import annotations

import importlib.util
import sys

from earshift_bakeoff.config import Paths, sha256_json


SCRIPT = (
    Paths().root
    / "scripts"
    / "run_english_central_vowel_spectral_correction_v1.py"
)


def _module():
    scripts = str(SCRIPT.parent)
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    spec = importlib.util.spec_from_file_location("central_spectral_v1", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_protocol_is_bounded_zero_render_and_hash_bound() -> None:
    protocol = _module().protocol_record()
    payload = dict(protocol)
    digest = payload.pop("protocol_sha256")
    assert digest == sha256_json(payload)
    assert protocol["scope"]["cell_count"] == 4
    assert protocol["scope"]["logical_slot_count"] == 12
    assert protocol["scope"]["candidate_count"] == 72
    assert protocol["scope_controls"] == {
        "kokoro_renders": 0,
        "api_calls": 0,
        "paid_calls": 0,
        "production_enabled": False,
        "deployment": False,
    }


def test_protocol_preserves_high_band_and_existing_acoustic_gate() -> None:
    protocol = _module().protocol_record()
    assert "above 4500 Hz" in protocol["intervention"]["processing"]
    assert "three-ceiling" in protocol["acoustic_gates"]
    assert protocol["engineering_gates"]["absolute_high_band_delta_db_maximum"] == 1.5
