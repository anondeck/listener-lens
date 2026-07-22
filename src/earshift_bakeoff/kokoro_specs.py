from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download

from .kokoro_synthesis import (
    CONFIG_FILE,
    MODEL_FILE,
    MODEL_HASHES,
    MODEL_REPO,
    MODEL_REVISION,
)
from .util import sha256_file


KOKORO_LICENSE = "Apache-2.0"
VOICE_PACK_SHAPE = (510, 1, 256)
VOICE_PACK_DTYPE = "torch.float32"
VOICES_MANIFEST_FILE = "VOICES.md"
VOICES_MANIFEST_SHA256 = (
    "ec7e4941ad7e194af61e3455928528a9ff5360c7c505e412efab27d6a69ea106"
)
MODEL_CARD_FILE = "README.md"
MODEL_CARD_SHA256 = "91dcabced89db6f109b8786642f50402d3ee87450e8189589b6f85520e7f4d78"
KOKORO_WHEEL_SHA256 = "a129dc6364a286bd6a92c396e9862459d3d3e45f2c15596ed5a94dcee5789efd"


@dataclass(frozen=True)
class VoiceSpec:
    voice_id: str
    language_id: str
    kokoro_lang_code: str
    gender: str
    sha256: str
    published_overall_grade: str | None
    in_frozen_screen: bool
    evidence_anchor: bool = False

    @property
    def filename(self) -> str:
        return f"voices/{self.voice_id}.pt"


@dataclass(frozen=True)
class LanguageSpec:
    language_id: str
    display_name: str
    kokoro_lang_code: str
    espeak_voice: str
    voice_ids: tuple[str, ...]
    renderer_candidate_flag: str
    renderer_candidate_enabled: bool
    listener_lens_status: str


_VOICE_ROWS = (
    (
        "af_heart",
        "en-US",
        "a",
        "female",
        "0ab5709b8ffab19bfd849cd11d98f75b60af7733253ad0d67b12382a102cb4ff",
        "A",
        True,
        True,
    ),
    (
        "af_alloy",
        "en-US",
        "a",
        "female",
        "6d877149dd8b348fbad12e5845b7e43d975390e9f3b68a811d1d86168bef5aa3",
        "C",
        False,
        False,
    ),
    (
        "af_aoede",
        "en-US",
        "a",
        "female",
        "c03bd1a4c3716c2d8eaa3d50022f62d5c31cfbd6e15933a00b17fefe13841cc4",
        "C+",
        False,
        False,
    ),
    (
        "af_bella",
        "en-US",
        "a",
        "female",
        "8cb64e02fcc8de0327a8e13817e49c76c945ecf0052ceac97d3081480e8e48d6",
        "A-",
        True,
        False,
    ),
    (
        "af_jessica",
        "en-US",
        "a",
        "female",
        "cdfdccb8cc975aa34ee6b89642963b0064237675de0e41a30ae64cc958dd4e87",
        "D",
        False,
        False,
    ),
    (
        "af_kore",
        "en-US",
        "a",
        "female",
        "8bfbc512321c3db49dff984ac675fa5ac7eaed5a96cc31104d3a9080e179d69d",
        "C+",
        False,
        False,
    ),
    (
        "af_nicole",
        "en-US",
        "a",
        "female",
        "c5561808bcf5250fe8c5f5de32caf2d94f27e57e95befdb098c5c85991d4c5da",
        "B-",
        True,
        False,
    ),
    (
        "af_nova",
        "en-US",
        "a",
        "female",
        "e0233676ddc21908c37a1f102f6b88a59e4e5c1bd764983616eb9eda629dbcd2",
        "C",
        False,
        False,
    ),
    (
        "af_river",
        "en-US",
        "a",
        "female",
        "e149459bd9c084416b74756b9bd3418256a8b839088abb07d463730c369dab8f",
        "D",
        False,
        False,
    ),
    (
        "af_sarah",
        "en-US",
        "a",
        "female",
        "49bd364ea3be9eb3e9685e8f9a15448c4883112a7c0ff7ab139fa4088b08cef9",
        "C+",
        False,
        False,
    ),
    (
        "af_sky",
        "en-US",
        "a",
        "female",
        "c799548aed06e0cb0d655a85a01b48e7f10484d71663f9a3045a5b9362e8512c",
        "C-",
        False,
        False,
    ),
    (
        "am_adam",
        "en-US",
        "a",
        "male",
        "ced7e284aba12472891be1da3ab34db84cc05cc02b5889535796dbf2d8b0cb34",
        "F+",
        False,
        False,
    ),
    (
        "am_echo",
        "en-US",
        "a",
        "male",
        "8bcfdc852bc985fb45c396c561e571ffb9183930071f962f1b50df5c97b161e8",
        "D",
        False,
        False,
    ),
    (
        "am_eric",
        "en-US",
        "a",
        "male",
        "ada66f0eefff34ec921b1d7474d7ac8bec00cd863c170f1c534916e9b8212aae",
        "D",
        False,
        False,
    ),
    (
        "am_fenrir",
        "en-US",
        "a",
        "male",
        "98e507eca1db08230ae3b6232d59c10aec9630022d19accac4f5d12fcec3c37a",
        "C+",
        True,
        False,
    ),
    (
        "am_liam",
        "en-US",
        "a",
        "male",
        "c82550757ddb31308b97f30040dda8c2d609a9e2de6135848d0a948368138518",
        "D",
        False,
        False,
    ),
    (
        "am_michael",
        "en-US",
        "a",
        "male",
        "9a443b79a4b22489a5b0ab7c651a0bcd1a30bef675c28333f06971abbd47bd37",
        "C+",
        True,
        False,
    ),
    (
        "am_onyx",
        "en-US",
        "a",
        "male",
        "e8452be16cd0f6da7b4579eaf7b1e4506e92524882053d86d72b96b9a7fed584",
        "D",
        False,
        False,
    ),
    (
        "am_puck",
        "en-US",
        "a",
        "male",
        "dd1d8973f4ce4b7d8ae407c77a435f485dabc052081b80ea75c4f30b84f36223",
        "C+",
        True,
        False,
    ),
    (
        "am_santa",
        "en-US",
        "a",
        "male",
        "7f2f7582fa2b1f160e90aafe6d0b442a685e773608b6667e545d743b073e97a7",
        "D-",
        False,
        False,
    ),
    (
        "pf_dora",
        "pt-BR",
        "p",
        "female",
        "07e4ff987c5d5a8c3995efd15cc4f0db7c4c15e881b198d8ab7f67ecf51f5eb7",
        None,
        True,
        False,
    ),
    (
        "pm_alex",
        "pt-BR",
        "p",
        "male",
        "cf0ba8c573c2480fc54123683a35cf1e2ae130428e441eb91f9149bdb188a526",
        None,
        True,
        False,
    ),
    (
        "pm_santa",
        "pt-BR",
        "p",
        "male",
        "d42103169c5c872abbafb9129133af7e942bb9d272c3cc3b95c203e7d7198c29",
        None,
        True,
        False,
    ),
)

VOICE_SPECS = tuple(VoiceSpec(*row) for row in _VOICE_ROWS)
VOICE_SPECS_BY_ID = {voice.voice_id: voice for voice in VOICE_SPECS}

ENGLISH_SCREEN_SHORTLIST = tuple(
    voice.voice_id
    for voice in VOICE_SPECS
    if voice.language_id == "en-US" and voice.in_frozen_screen
)
PORTUGUESE_SCREEN_VOICES = tuple(
    voice.voice_id for voice in VOICE_SPECS if voice.language_id == "pt-BR"
)

LANGUAGE_SPECS = {
    "en-US": LanguageSpec(
        language_id="en-US",
        display_name="American English",
        kokoro_lang_code="a",
        espeak_voice="en-us",
        voice_ids=tuple(
            voice.voice_id for voice in VOICE_SPECS if voice.language_id == "en-US"
        ),
        renderer_candidate_flag="KOKORO_ENGLISH_CANDIDATE_ENABLED",
        renderer_candidate_enabled=False,
        listener_lens_status="existing-English-to-BP-research-chain",
    ),
    "pt-BR": LanguageSpec(
        language_id="pt-BR",
        display_name="Brazilian Portuguese",
        kokoro_lang_code="p",
        espeak_voice="pt-br",
        voice_ids=PORTUGUESE_SCREEN_VOICES,
        renderer_candidate_flag="PORTUGUESE_RENDERER_CANDIDATE_ENABLED",
        renderer_candidate_enabled=False,
        listener_lens_status="renderer-and-carrier-only-no-listener-lens-claim",
    ),
}


def resolve_pinned_file(filename: str, *, download: bool = False) -> Path:
    return Path(
        hf_hub_download(
            repo_id=MODEL_REPO,
            revision=MODEL_REVISION,
            filename=filename,
            local_files_only=not download,
        )
    )


def verify_voice_pack(voice: VoiceSpec, *, download: bool = False) -> dict[str, Any]:
    import torch

    path = resolve_pinned_file(voice.filename, download=download)
    actual_hash = sha256_file(path)
    if actual_hash != voice.sha256:
        raise ValueError(
            f"voice hash mismatch for {voice.voice_id}: {actual_hash} != {voice.sha256}"
        )
    pack = torch.load(path, map_location="cpu", weights_only=True)
    shape = tuple(int(value) for value in pack.shape)
    finite = bool(torch.isfinite(pack).all())
    contiguous = bool(pack.is_contiguous())
    compatible = bool(
        shape == VOICE_PACK_SHAPE
        and str(pack.dtype) == VOICE_PACK_DTYPE
        and finite
        and contiguous
    )
    if not compatible:
        raise ValueError(
            f"voice pack is incompatible with Kokoro v1.0: {voice.voice_id}"
        )
    return {
        **asdict(voice),
        "filename": voice.filename,
        "bytes": path.stat().st_size,
        "shape": list(shape),
        "dtype": str(pack.dtype),
        "finite": finite,
        "contiguous": contiguous,
        "model_compatible": compatible,
    }


def voice_inventory_receipt(*, download: bool = False) -> dict[str, Any]:
    manifest_path = resolve_pinned_file(VOICES_MANIFEST_FILE, download=download)
    if sha256_file(manifest_path) != VOICES_MANIFEST_SHA256:
        raise ValueError("pinned VOICES.md hash mismatch")
    model_card_path = resolve_pinned_file(MODEL_CARD_FILE, download=download)
    if sha256_file(model_card_path) != MODEL_CARD_SHA256:
        raise ValueError("pinned model-card hash mismatch")
    if "license: apache-2.0" not in model_card_path.read_text(encoding="utf-8"):
        raise ValueError("pinned model card does not declare Apache-2.0")
    model_files = {
        filename: resolve_pinned_file(filename, download=download)
        for filename in (CONFIG_FILE, MODEL_FILE)
    }
    actual_model_hashes = {
        filename: sha256_file(path) for filename, path in model_files.items()
    }
    expected_model_hashes = {
        filename: MODEL_HASHES[filename] for filename in (CONFIG_FILE, MODEL_FILE)
    }
    if actual_model_hashes != expected_model_hashes:
        raise ValueError("pinned Kokoro model hashes mismatch")
    voices = [verify_voice_pack(voice, download=download) for voice in VOICE_SPECS]
    payload = {
        "schema_version": 1,
        "model_repo": MODEL_REPO,
        "model_revision": MODEL_REVISION,
        "license": KOKORO_LICENSE,
        "license_evidence": {
            "file": MODEL_CARD_FILE,
            "sha256": MODEL_CARD_SHA256,
            "declaration": "license: apache-2.0",
        },
        "kokoro_wheel_sha256": KOKORO_WHEEL_SHA256,
        "voices_manifest_sha256": VOICES_MANIFEST_SHA256,
        "model_hashes": actual_model_hashes,
        "voice_pack_contract": {
            "shape": list(VOICE_PACK_SHAPE),
            "dtype": VOICE_PACK_DTYPE,
            "finite": True,
            "contiguous": True,
        },
        "languages": {
            key: asdict(value) for key, value in sorted(LANGUAGE_SPECS.items())
        },
        "english_screen_shortlist": list(ENGLISH_SCREEN_SHORTLIST),
        "english_shortlist_rule": (
            "af_heart evidence anchor; the next two highest published female "
            "overall grades; and all three male voices tied for the highest "
            "published male overall grade at the pinned revision"
        ),
        "portuguese_screen_voices": list(PORTUGUESE_SCREEN_VOICES),
        "voices": voices,
    }
    payload["receipt_sha256"] = hashlib.sha256(
        json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    return payload
