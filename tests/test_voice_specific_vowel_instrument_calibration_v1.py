from __future__ import annotations

import importlib.util
import sys

from earshift_bakeoff.config import Paths, sha256_json


SCRIPT = (
    Paths().root
    / "scripts"
    / "run_voice_specific_vowel_instrument_calibration_v1.py"
)


def _module():
    scripts = str(SCRIPT.parent)
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    spec = importlib.util.spec_from_file_location("voice_vowel_calibration_v1", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_protocol_is_anchor_only_broad_and_hash_bound() -> None:
    protocol = _module().protocol_record()
    payload = dict(protocol)
    digest = payload.pop("protocol_sha256")
    assert digest == sha256_json(payload)
    assert protocol["scope"]["voice_rule_cell_count"] == 80
    assert protocol["scope"]["natural_anchor_wav_count"] == 480
    assert protocol["selection"]["candidate_audio_excluded"] is True
    assert protocol["scope_controls"]["audio_renders"] == 0
    assert protocol["scope_controls"]["api_calls"] == 0


def test_voice_orders_distinguish_male_and_female_defaults() -> None:
    module = _module()
    assert module.CEILING_ORDER_BY_VOICE["af_heart"][0] == 5_500
    assert module.CEILING_ORDER_BY_VOICE["pf_dora"][0] == 5_500
    assert module.CEILING_ORDER_BY_VOICE["am_michael"][0] == 5_000
    assert module.CEILING_ORDER_BY_VOICE["pm_alex"][0] == 5_000
    assert set(module.FEMALE_CEILING_ORDER) == set(module.MALE_CEILING_ORDER)
