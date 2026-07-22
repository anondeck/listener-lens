from __future__ import annotations

import importlib.util
import math
import sys

from earshift_bakeoff.config import Paths, sha256_json


SCRIPT = (
    Paths().root
    / "scripts"
    / "run_broad_oral_monophthong_contextual_warp_v1.py"
)


def _module():
    scripts = str(SCRIPT.parent)
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    spec = importlib.util.spec_from_file_location("broad_contextual_warp_v1", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_protocol_is_broad_bilingual_zero_render_and_hash_bound() -> None:
    protocol = _module().protocol_record()
    payload = dict(protocol)
    digest = payload.pop("protocol_sha256")
    assert digest == sha256_json(payload)
    assert protocol["scope"]["cell_count"] == 45
    assert protocol["scope"]["logical_slot_count"] == 135
    assert protocol["scope"]["target_occurrence_count"] == 180
    assert protocol["scope"]["candidate_count"] == 810
    assert any("en-US-to-pt-BR" in row for row in protocol["scope"]["cell_ids"])
    assert any("pt-BR-to-en-US" in row for row in protocol["scope"]["cell_ids"])
    assert protocol["scope_controls"]["kokoro_renders"] == 0
    assert protocol["scope_controls"]["api_calls"] == 0


def test_bark_inverse_round_trip_is_stable() -> None:
    module = _module()
    for hz in (250.0, 500.0, 1_000.0, 2_500.0):
        bark = 26.81 * hz / (1960.0 + hz) - 0.53
        assert math.isclose(module._bark_to_hz(bark), hz, rel_tol=1e-9)
