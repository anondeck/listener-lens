from __future__ import annotations

import importlib.util
import sys

from earshift_bakeoff.config import Paths, sha256_json


SCRIPT = Paths().root / "scripts" / "run_broad_vowel_acoustic_feedback_v2.py"


def _module():
    scripts = str(SCRIPT.parent)
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    spec = importlib.util.spec_from_file_location("broad_vowel_feedback_v2", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_protocol_changes_only_slot_exception_boundary() -> None:
    protocol = _module().protocol_record()
    payload = dict(protocol)
    digest = payload.pop("protocol_sha256")
    assert digest == sha256_json(payload)
    assert protocol["invariants"]["cell_count"] == 45
    assert protocol["invariants"]["maximum_feedback_edits"] == 405
    assert protocol["invariants"]["mechanism_change"] is False
    assert protocol["invariants"]["target_change"] is False
    assert protocol["invariants"]["threshold_change"] is False
    assert protocol["invariants"]["aggregation_change"] is False
    assert protocol["scope_controls"]["api_calls"] == 0
