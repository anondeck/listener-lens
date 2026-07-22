from __future__ import annotations

import importlib.util
import sys

import numpy as np

from earshift_bakeoff.config import Paths, sha256_json


SCRIPT = Paths().root / "scripts" / "run_broad_vowel_world_differential_v1.py"


def _module():
    scripts = str(SCRIPT.parent)
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    spec = importlib.util.spec_from_file_location("broad_world_differential_v1", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_protocol_is_bounded_and_reuses_frozen_world_outputs() -> None:
    protocol = _module().protocol_record()
    payload = dict(protocol)
    digest = payload.pop("protocol_sha256")
    assert digest == sha256_json(payload)
    assert protocol["scope"]["available_input_pair_count"] == 133
    assert protocol["scope"]["maximum_derived_lens_count"] == 133
    assert protocol["scope"]["world_syntheses"] == 0
    assert protocol["intervention"]["selection"] == "none; coefficient fixed at exactly 1.0"


def test_differential_cancels_shared_identity_coloration() -> None:
    module = _module()
    original = np.asarray([100, 200, 300], dtype=np.int16)
    identity = np.asarray([110, 180, 330], dtype=np.int16)
    lens = np.asarray([115, 175, 350], dtype=np.int16)
    result, clipped = module._differential_lens(original, identity, lens)
    assert clipped == 0
    assert np.array_equal(result, np.asarray([105, 195, 320], dtype=np.int16))


def test_differential_reports_saturation_before_pcm_clipping() -> None:
    module = _module()
    result, clipped = module._differential_lens(
        np.asarray([32_760], dtype=np.int16),
        np.asarray([0], dtype=np.int16),
        np.asarray([100], dtype=np.int16),
    )
    assert clipped == 1
    assert result[0] == 32_767
