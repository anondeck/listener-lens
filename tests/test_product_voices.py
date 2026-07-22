from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from earshift_bakeoff.kokoro_product_contract import PRODUCT_VOICE_PINS
from earshift_bakeoff.kokoro_specs import VOICE_SPECS_BY_ID
from earshift_bakeoff.product_voices import (
    PRODUCT_VOICE_REGISTRY_PATH,
    ProductVoiceError,
    load_product_voice_registry,
)


ROOT = Path(__file__).resolve().parents[1]


def test_product_registry_contains_exactly_the_four_selected_voices() -> None:
    registry = load_product_voice_registry()

    assert set(registry.voices) == {
        "af_heart",
        "am_michael",
        "pf_dora",
        "pm_alex",
    }
    assert registry.defaults == {"en-US": "af_heart", "pt-BR": "pm_alex"}
    assert registry.same_voice_pair_required is True
    assert registry.production_enabled is False
    assert len(registry.registry_sha256) == 64


def test_lightweight_product_pins_match_the_complete_research_inventory() -> None:
    for voice_id, pin in PRODUCT_VOICE_PINS.items():
        complete = VOICE_SPECS_BY_ID[voice_id]
        assert pin.language_id == complete.language_id
        assert pin.gender == complete.gender
        assert pin.sha256 == complete.sha256


def test_product_registry_loads_without_third_party_site_packages() -> None:
    environment = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    completed = subprocess.run(
        [
            sys.executable,
            "-S",
            "-c",
            (
                "from earshift_bakeoff.product_voices import "
                "load_product_voice_registry; "
                "assert len(load_product_voice_registry().voices) == 4"
            ),
        ],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_product_registry_rejects_cross_language_and_unselected_voices() -> None:
    registry = load_product_voice_registry()

    assert registry.resolve("en-US").voice_id == "af_heart"
    assert registry.resolve("en-US", "am_michael").gender == "male"
    assert registry.resolve("pt-BR").voice_id == "pm_alex"
    assert registry.resolve("pt-BR", "pf_dora").gender == "female"
    with pytest.raises(ProductVoiceError, match="incompatible"):
        registry.resolve("en-US", "pm_alex")
    with pytest.raises(ProductVoiceError, match="outside"):
        registry.resolve("en-US", "af_bella")


def test_safe_catalog_omits_voice_hashes_and_selection_record_paths() -> None:
    registry = load_product_voice_registry()
    catalog = registry.safe_catalog()
    serialized = json.dumps(catalog)

    assert catalog["renderer"] == "kokoro"
    assert catalog["production_enabled"] is False
    assert [row["language_id"] for row in catalog["languages"]] == [
        "en-US",
        "pt-BR",
    ]
    assert "voice_sha256" not in serialized
    assert "kokoro-en-voice-shortlist.json" not in serialized
    assert PRODUCT_VOICE_REGISTRY_PATH.is_file()
