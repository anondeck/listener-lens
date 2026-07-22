from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from .config import ROOT
from .kokoro_product_contract import (
    KOKORO_LICENSE,
    KOKORO_MODEL_REPO,
    KOKORO_MODEL_REVISION,
    KOKORO_SAMPLE_RATE_HZ,
    KOKORO_VERSION,
    PRODUCT_VOICE_PINS,
)
from .util import sha256_file


PRODUCT_VOICE_REGISTRY_PATH = ROOT / "rules" / "kokoro-product-voices.json"
PRODUCT_VOICE_REGISTRY_VERSION = "kokoro-product-voices-v1"


class ProductVoiceError(ValueError):
    """The selected product voice is absent, incompatible, or unverified."""


@dataclass(frozen=True)
class ProductVoice:
    voice_id: str
    language_id: str
    display_name: str
    gender: str
    style_label: str
    selection_role: str
    evidence_status: str
    voice_sha256: str

    def safe_metadata(self) -> dict[str, str]:
        return {
            "voice_id": self.voice_id,
            "language_id": self.language_id,
            "display_name": self.display_name,
            "gender": self.gender,
            "style_label": self.style_label,
            "selection_role": self.selection_role,
            "evidence_status": self.evidence_status,
        }


@dataclass(frozen=True)
class ProductVoiceRegistry:
    registry_version: str
    registry_sha256: str
    production_enabled: bool
    same_voice_pair_required: bool
    defaults: dict[str, str]
    profiles: dict[str, str]
    voices: dict[str, ProductVoice]

    def voices_for(self, language_id: str) -> tuple[ProductVoice, ...]:
        values = tuple(
            voice for voice in self.voices.values() if voice.language_id == language_id
        )
        if not values:
            raise ProductVoiceError(f"unsupported product language: {language_id}")
        return values

    def resolve(self, language_id: str, voice_id: str | None = None) -> ProductVoice:
        selected = voice_id or self.defaults.get(language_id)
        try:
            voice = self.voices[str(selected)]
        except KeyError as exc:
            raise ProductVoiceError("voice is outside the selected product registry") from exc
        if voice.language_id != language_id:
            raise ProductVoiceError("voice and source language are incompatible")
        return voice

    def safe_catalog(self) -> dict[str, Any]:
        languages = []
        for language_id, default_voice_id in self.defaults.items():
            languages.append(
                {
                    "language_id": language_id,
                    "default_voice_id": default_voice_id,
                    "voices": [
                        voice.safe_metadata() for voice in self.voices_for(language_id)
                    ],
                }
            )
        return {
            "schema_version": 1,
            "registry_version": self.registry_version,
            "registry_sha256": self.registry_sha256,
            "renderer": "kokoro",
            "same_voice_pair_required": self.same_voice_pair_required,
            "production_enabled": self.production_enabled,
            "languages": languages,
        }


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ProductVoiceError(f"expected an object in {path}")
    return value


def load_product_voice_registry(
    path: Path = PRODUCT_VOICE_REGISTRY_PATH,
) -> ProductVoiceRegistry:
    data = _load_object(path)
    if set(data) != {
        "schema_version",
        "registry_version",
        "renderer",
        "same_voice_pair_required",
        "production_enabled",
        "selection_bindings",
        "languages",
    }:
        raise ProductVoiceError("product voice registry has an unexpected schema")
    renderer = data["renderer"]
    if renderer != {
        "package": "kokoro",
        "version": KOKORO_VERSION,
        "model_repo": KOKORO_MODEL_REPO,
        "model_revision": KOKORO_MODEL_REVISION,
        "sample_rate_hz": KOKORO_SAMPLE_RATE_HZ,
        "license": KOKORO_LICENSE,
    }:
        raise ProductVoiceError("product voice renderer binding drifted")
    if (
        data["schema_version"] != 1
        or data["registry_version"] != PRODUCT_VOICE_REGISTRY_VERSION
        or data["same_voice_pair_required"] is not True
        or data["production_enabled"] is not False
    ):
        raise ProductVoiceError("product voice safety state drifted")

    bindings = data["selection_bindings"]
    if set(bindings) != {"en-US", "pt-BR"}:
        raise ProductVoiceError("product voice selection bindings are incomplete")
    selection_records: dict[str, dict[str, Any]] = {}
    for language_id, binding in bindings.items():
        if set(binding) != {"path", "sha256"}:
            raise ProductVoiceError("product voice selection binding is malformed")
        binding_path = ROOT / binding["path"]
        if sha256_file(binding_path) != binding["sha256"]:
            raise ProductVoiceError("product voice selection record hash mismatch")
        selection_records[language_id] = _load_object(binding_path)

    languages = data["languages"]
    if not isinstance(languages, list) or len(languages) != 2:
        raise ProductVoiceError("exactly two product languages are required")
    defaults: dict[str, str] = {}
    profiles: dict[str, str] = {}
    voices: dict[str, ProductVoice] = {}
    for language in languages:
        if set(language) != {
            "language_id",
            "display_name",
            "default_voice_id",
            "profile_ids",
            "voices",
        }:
            raise ProductVoiceError("product language entry has an unexpected schema")
        language_id = language["language_id"]
        if language_id in defaults or language_id not in {"en-US", "pt-BR"}:
            raise ProductVoiceError("product language IDs are invalid or duplicated")
        default_voice_id = language["default_voice_id"]
        defaults[language_id] = default_voice_id
        if not isinstance(language["profile_ids"], list) or not language["profile_ids"]:
            raise ProductVoiceError("product language has no bound profile")
        for profile_id in language["profile_ids"]:
            if profile_id in profiles:
                raise ProductVoiceError("product profile is assigned twice")
            profiles[profile_id] = language_id
        language_voices = language["voices"]
        if not isinstance(language_voices, list) or len(language_voices) != 2:
            raise ProductVoiceError("each product language requires two voices")
        for raw in language_voices:
            if set(raw) != {
                "voice_id",
                "display_name",
                "gender",
                "style_label",
                "selection_role",
                "evidence_status",
                "voice_sha256",
            }:
                raise ProductVoiceError("product voice entry has an unexpected schema")
            voice_id = raw["voice_id"]
            if voice_id in voices:
                raise ProductVoiceError("product voice IDs are duplicated")
            try:
                pinned = PRODUCT_VOICE_PINS[voice_id]
            except KeyError as exc:
                raise ProductVoiceError("product voice is not pinned") from exc
            if (
                pinned.language_id != language_id
                or pinned.gender != raw["gender"]
                or pinned.sha256 != raw["voice_sha256"]
            ):
                raise ProductVoiceError("product voice metadata disagrees with pin")
            voices[voice_id] = ProductVoice(language_id=language_id, **raw)
        if default_voice_id not in voices or voices[default_voice_id].language_id != language_id:
            raise ProductVoiceError("product default voice is missing")

    if set(voices) != {"af_heart", "am_michael", "pf_dora", "pm_alex"}:
        raise ProductVoiceError("product registry must contain exactly four selected voices")
    english_selection = selection_records["en-US"]
    if english_selection.get("selected_defaults") != {
        "female_voice_id": "af_heart",
        "male_voice_id": "am_michael",
    }:
        raise ProductVoiceError("English product defaults disagree with creator selection")
    portuguese_selection = selection_records["pt-BR"]
    if (
        portuguese_selection.get("selected_voice_id") != "pm_alex"
        or portuguese_selection.get("selection", {}).get("alternate_voice_ids")
        != ["pf_dora"]
    ):
        raise ProductVoiceError(
            "Portuguese product defaults disagree with creator selection"
        )
    return ProductVoiceRegistry(
        registry_version=data["registry_version"],
        registry_sha256=sha256_file(path),
        production_enabled=False,
        same_voice_pair_required=True,
        defaults=defaults,
        profiles=profiles,
        voices=voices,
    )
