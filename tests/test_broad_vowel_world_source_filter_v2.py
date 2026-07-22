from __future__ import annotations

import importlib.util
import sys

from earshift_bakeoff.config import Paths, sha256_json


SCRIPT = Paths().root / "scripts" / "run_broad_vowel_world_source_filter_v2.py"


def _module():
    scripts = str(SCRIPT.parent)
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    spec = importlib.util.spec_from_file_location("broad_world_v2", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_recovery_protocol_is_hash_bound_and_signal_processing_free() -> None:
    protocol = _module().protocol_record()
    payload = dict(protocol)
    digest = payload.pop("protocol_sha256")
    assert digest == sha256_json(payload)
    assert protocol["scope"]["frozen_wav_count"] == 266
    assert protocol["scope"]["world_analyses"] == 0
    assert protocol["scope"]["world_syntheses"] == 0
    assert protocol["scope_controls"]["kokoro_renders"] == 0


def test_classification_uses_strings_without_legacy_boolean_adapter() -> None:
    module = _module()
    exact = [{"classification": "exact_category_pass"}] * 3
    mixed = exact[:2] + [{"classification": "directional_only_pass"}]
    failed = exact[:2] + [{"classification": "fail"}]
    assert module._classification(exact) == "exact_category_pass"
    assert module._classification(mixed) == "directional_only_pass"
    assert module._classification(failed) == "fail"
