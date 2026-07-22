from __future__ import annotations

import importlib.util
import sys

from earshift_bakeoff.config import Paths, sha256_json


SCRIPT = Paths().root / "scripts" / "run_broad_vowel_acoustic_feedback_v1.py"


def _module():
    scripts = str(SCRIPT.parent)
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    spec = importlib.util.spec_from_file_location("broad_vowel_feedback_v1", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_protocol_is_bounded_broad_and_zero_render() -> None:
    protocol = _module().protocol_record()
    payload = dict(protocol)
    digest = payload.pop("protocol_sha256")
    assert digest == sha256_json(payload)
    assert protocol["scope"]["cell_count"] == 45
    assert protocol["scope"]["logical_slot_count"] == 135
    assert protocol["scope"]["maximum_feedback_edits"] == 405
    assert protocol["controller"]["maximum_iterations"] == 3
    assert protocol["controller"]["feedback_gain"] == 0.75
    assert "no best-iteration selection" in protocol["stopping_rule"].lower()
    assert protocol["scope_controls"]["kokoro_renders"] == 0
    assert protocol["scope_controls"]["api_calls"] == 0
