from __future__ import annotations

import importlib.util
import sys

import numpy as np

from earshift_bakeoff.config import Paths, sha256_json


SCRIPT = Paths().root / "scripts" / "run_broad_vowel_jacobian_controller_v1.py"


def _module():
    scripts = str(SCRIPT.parent)
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    spec = importlib.util.spec_from_file_location("broad_vowel_jacobian_v1", SCRIPT)
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
    assert protocol["scope"]["maximum_signal_edits"] == 405
    assert protocol["controller"]["selection"] == "none; one deterministic solution and one final edit"
    assert protocol["scope_controls"]["api_calls"] == 0


def test_solver_recovers_well_conditioned_linear_request() -> None:
    module = _module()
    neutral = {"feature_bark": [5.0, 10.0]}
    result = module._solve(
        neutral=neutral,
        f1_probe={"feature_bark": [5.5, 10.0]},
        f2_probe={"feature_bark": [5.0, 10.5]},
        target_delta=[0.8, -0.6],
    )
    assert result["pass"] is True
    assert np.allclose(result["predicted_displacement_bark"], [0.7619, -0.5714], atol=1e-3)
