from __future__ import annotations

from dataclasses import dataclass


KOKORO_VERSION = "0.9.4"
KOKORO_MODEL_REPO = "hexgrad/Kokoro-82M"
KOKORO_MODEL_REVISION = "f3ff3571791e39611d31c381e3a41a3af07b4987"
KOKORO_SAMPLE_RATE_HZ = 24_000
KOKORO_LICENSE = "Apache-2.0"


@dataclass(frozen=True)
class ProductVoicePin:
    language_id: str
    gender: str
    sha256: str


# This dependency-free subset is the boot-time product contract. The complete
# research inventory remains in kokoro_specs.py; keeping this subset free of
# Hugging Face, NumPy, and Torch lets the disabled slim service validate its
# public voice registry without importing the synthesis stack.
PRODUCT_VOICE_PINS = {
    "af_heart": ProductVoicePin(
        language_id="en-US",
        gender="female",
        sha256="0ab5709b8ffab19bfd849cd11d98f75b60af7733253ad0d67b12382a102cb4ff",
    ),
    "am_michael": ProductVoicePin(
        language_id="en-US",
        gender="male",
        sha256="9a443b79a4b22489a5b0ab7c651a0bcd1a30bef675c28333f06971abbd47bd37",
    ),
    "pf_dora": ProductVoicePin(
        language_id="pt-BR",
        gender="female",
        sha256="07e4ff987c5d5a8c3995efd15cc4f0db7c4c15e881b198d8ab7f67ecf51f5eb7",
    ),
    "pm_alex": ProductVoicePin(
        language_id="pt-BR",
        gender="male",
        sha256="cf0ba8c573c2480fc54123683a35cf1e2ae130428e441eb91f9149bdb188a526",
    ),
}
