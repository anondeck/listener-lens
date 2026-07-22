from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import json
import os
import random
import shutil
import subprocess
import wave
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

from .config import Paths, sha256_json, stable_json
from .kokoro_gate_bridge import KokoroGateIndex
from .kokoro_specs import (
    ENGLISH_SCREEN_SHORTLIST,
    KOKORO_WHEEL_SHA256,
    LANGUAGE_SPECS,
    PORTUGUESE_SCREEN_VOICES,
    VOICE_SPECS_BY_ID,
    resolve_pinned_file,
)
from .kokoro_synthesis import (
    CONFIG_FILE,
    KOKORO_VERSION,
    MAX_PHONEME_CHARACTERS,
    MODEL_FILE,
    MODEL_REPO,
    MODEL_REVISION,
    SAMPLE_RATE_HZ,
    SPEED,
    _INFERENCE_LOCK,
    pcm16_bytes,
    verify_model_files,
)
from .kokoro_typed_engine import KokoroTypedPlanner
from .listener_lens import DatabaseNonceChecker, WORD_RE
from .portuguese_carrier_planner_v1 import (
    PORTUGUESE_RENDERER_CANDIDATE_ENABLED,
    PORTUGUESE_SMOKE_FIXTURE_ID_V1,
    PORTUGUESE_SMOKE_TARGET_PHONE_V1,
    PORTUGUESE_SMOKE_TEXT_V1,
    PortugueseCarrierPlannerV1,
    plan_portuguese_smoke_fixture_v1,
)
from .util import atomic_write_json, atomic_write_text, sha256_bytes, sha256_file


RUN_ID = "20260717-kokoro-bilingual-voice-screen-v1"
SCHEMA_VERSION = 1
RENDER_SEED = 20_260_717_02
BLIND_SEED = "kokoro-bilingual-voice-screen-v1-blinding-20260717"
SAMPLES_PER_DURATION_FRAME = 600
MAX_CLIPPED_FRACTION = 0.001
MIN_DURATION_S = 0.25
MAX_DURATION_S = 45.0
MIN_RMS = 1e-5
MIN_PEAK = 1e-4
MAX_ABS_DC_OFFSET = 0.10

ENGLISH_RESPONSE_FILENAME = "kokoro-en-voice-screen-v1-response.json"
PORTUGUESE_RESPONSE_FILENAME = "kokoro-ptbr-voice-screen-v1-response.json"
ENGLISH_KOKORO_GATE_RUN_ID = "20260716-kokoro-gate-bridge-feasibility-v1"
PORTUGUESE_KOKORO_GATE_RUN_ID = "20260717-pt-kokoro-homophone-index-v1"

ENGLISH_REAL_PASSAGES = (
    (
        "en-real-harbor",
        "The harbor lights shimmered while distant rain softened the evening traffic.",
    ),
    (
        "en-real-notebook",
        "Mira tucked the blue notebook beside the kettle before the guests arrived.",
    ),
)
ENGLISH_OPAQUE_SOURCE = "The map rests beside the lamp while rain falls."
PORTUGUESE_REAL_PASSAGES = (
    (
        "ptbr-real-varanda",
        "A brisa atravessou a varanda enquanto a chuva caía devagar.",
    ),
    (
        "ptbr-real-caderno",
        ("Marta guardou o caderno azul ao lado da chaleira antes da visita chegar."),
    ),
)


class VoiceScreenError(RuntimeError):
    """The preregistered voice-screen contract could not be satisfied."""


@dataclass(frozen=True)
class FixturePlan:
    fixture_id: str
    language_id: str
    fixture_kind: str
    source_text: str
    render_text: str
    phonemes: str
    g2p_tokens: tuple[dict[str, Any], ...]
    gate_receipt: dict[str, Any] | None

    def record(self, model_vocab: Mapping[str, int] | set[str]) -> dict[str, Any]:
        if self.fixture_kind not in {"real-passage", "opaque-carrier"}:
            raise VoiceScreenError(f"unsupported fixture kind: {self.fixture_kind}")
        if self.language_id not in {"en-US", "pt-BR"}:
            raise VoiceScreenError(f"unsupported fixture language: {self.language_id}")
        if not self.source_text or not self.render_text or not self.phonemes:
            raise VoiceScreenError("fixture text and phoneme plans must be nonempty")
        if len(self.phonemes) > MAX_PHONEME_CHARACTERS:
            raise VoiceScreenError(f"{self.fixture_id} exceeds Kokoro context length")
        unsupported = sorted(set(self.phonemes) - set(model_vocab))
        if unsupported:
            raise VoiceScreenError(
                f"{self.fixture_id} contains unsupported symbols: {''.join(unsupported)}"
            )
        filtered_count = (
            sum(1 for symbol in self.phonemes if model_vocab.get(symbol) is not None)
            if isinstance(model_vocab, Mapping)
            else sum(1 for symbol in self.phonemes if symbol in model_vocab)
        )
        payload: dict[str, Any] = {
            "fixture_id": self.fixture_id,
            "language_id": self.language_id,
            "fixture_kind": self.fixture_kind,
            "source_text": self.source_text,
            "source_text_sha256": sha256_bytes(self.source_text.encode("utf-8")),
            "render_text": self.render_text,
            "render_text_sha256": sha256_bytes(self.render_text.encode("utf-8")),
            "review_reference_text": (
                self.render_text if self.fixture_kind == "real-passage" else None
            ),
            "phonemes": self.phonemes,
            "phonemes_sha256": sha256_bytes(self.phonemes.encode("utf-8")),
            "raw_phoneme_character_count": len(self.phonemes),
            "model_token_count_without_boundaries": filtered_count,
            "style_index": len(self.phonemes) - 1,
            "g2p_tokens": list(self.g2p_tokens),
            "g2p_route": (
                "KPipeline(lang_code='a').g2p"
                if self.language_id == "en-US"
                else "KPipeline(lang_code='p').g2p"
            ),
            "gate_receipt": self.gate_receipt,
            "renderer_candidate_enabled": False,
        }
        payload["fixture_plan_sha256"] = sha256_json(payload)
        return payload


@dataclass(frozen=True)
class RenderOutput:
    audio: np.ndarray
    predicted_durations: tuple[int, ...]


class OrdinaryRuntime(Protocol):
    model_vocab: dict[str, int]

    def render(self, *, voice_id: str, phonemes: str) -> RenderOutput: ...


def run_dir() -> Path:
    return Paths().artifacts / "voice-screen" / RUN_ID


def _sha256_payload(value: dict[str, Any]) -> str:
    return hashlib.sha256(stable_json(value).encode("utf-8")).hexdigest()


def _pipeline_plan(pipeline: Any, text: str) -> tuple[str, tuple[dict[str, Any], ...]]:
    phonemes, tokens = pipeline.g2p(text)
    phonemes = str(phonemes).strip()
    if not phonemes:
        raise VoiceScreenError("pinned Kokoro G2P returned an empty plan")
    token_rows: list[dict[str, Any]] = []
    if tokens is not None:
        for token in tokens:
            token_rows.append(
                {
                    "text": str(token.text),
                    "phonemes": (
                        None if token.phonemes is None else str(token.phonemes)
                    ),
                    "whitespace": str(token.whitespace),
                }
            )
        reconstructed = "".join(
            (row["phonemes"] or "") + row["whitespace"] for row in token_rows
        )
        if reconstructed != phonemes:
            raise VoiceScreenError("KPipeline token and phrase phone plans diverged")
    return phonemes, tuple(token_rows)


def _english_complete_index_receipt() -> dict[str, Any]:
    path = (
        Paths().artifacts
        / "typed-engine"
        / ENGLISH_KOKORO_GATE_RUN_ID
        / "full-index-receipt.json"
    )
    if not path.is_file():
        raise VoiceScreenError("frozen English Kokoro index receipt is missing")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    database_hash = sha256_file(Paths().kokoro_gate_db)
    if (
        receipt.get("run_id") != ENGLISH_KOKORO_GATE_RUN_ID
        or receipt.get("status") != "complete"
        or receipt.get("database_sha256") != database_hash
        or not receipt.get("negative_lookup_scope")
    ):
        raise VoiceScreenError("frozen English Kokoro index receipt changed")
    return {
        "run_id": receipt["run_id"],
        "status": receipt["status"],
        "receipt_file_sha256": sha256_file(path),
        "database_sha256": database_hash,
        "input_words": receipt["inventory"]["input_words"],
        "negative_lookup_scope": receipt["negative_lookup_scope"],
    }


def _english_pipeline_gate_receipt(
    *,
    carrier_script: str,
    pipeline_tokens: Sequence[dict[str, Any]],
    generator_plan: Any,
) -> dict[str, Any]:
    mandatory = DatabaseNonceChecker()
    phone_index = KokoroGateIndex()
    if not mandatory.enabled:
        raise VoiceScreenError("English mandatory nonce gate is unavailable")
    lexical = [
        row
        for row in pipeline_tokens
        if WORD_RE.fullmatch(str(row["text"])) is not None
    ]
    if not lexical:
        raise VoiceScreenError("English carrier pipeline produced no lexical tokens")
    checks: list[dict[str, Any]] = []
    previous_surface: str | None = None
    previous_phone: str | None = None
    for row in lexical:
        surface = str(row["text"])
        phone = str(row["phonemes"] or "")
        isolated = mandatory.check(surface, "en", None)
        adjacent = mandatory.check(surface, "en", previous_surface)
        phone_positive = phone_index.phone_match(phone)
        adjacency_phone_positive = bool(
            previous_phone is not None
            and phone_index.phone_match(previous_phone + phone)
        )
        checks.append(
            {
                "surface": surface,
                "phone": phone,
                "written_espeak_isolated_pass": isolated.accepted,
                "written_espeak_adjacency_pass": adjacent.accepted,
                "kokoro_phone_positive_homophone": phone_positive,
                "kokoro_adjacency_positive_homophone": adjacency_phone_positive,
            }
        )
        previous_surface = surface
        previous_phone = phone
    pass_all = all(
        row["written_espeak_isolated_pass"]
        and row["written_espeak_adjacency_pass"]
        and not row["kokoro_phone_positive_homophone"]
        and not row["kokoro_adjacency_positive_homophone"]
        for row in checks
    )
    if not pass_all:
        raise VoiceScreenError(
            "actual English KPipeline carrier plan failed opacity gates"
        )
    english_index = _english_complete_index_receipt()
    payload = {
        "schema_version": 1,
        "carrier_script": carrier_script,
        "carrier_script_sha256": sha256_bytes(carrier_script.encode("utf-8")),
        "generator": "KokoroTypedPlanner v1 neutral carrier surface",
        "generator_plan_sha256": generator_plan.plan_sha256,
        "generator_gate_summary": asdict(generator_plan.gate_summary),
        "actual_render_plan_gate": "KPipeline English token phones",
        "checks": checks,
        "mandatory_written_espeak_gate_pass": True,
        "actual_kokoro_phone_gate_pass": True,
        "english_complete_kokoro_index_negative_used_for_clearance": True,
        "english_complete_kokoro_index_scope": english_index["negative_lookup_scope"],
        "english_complete_kokoro_index_receipt": english_index,
        "pass": True,
    }
    payload["receipt_sha256"] = sha256_json(payload)
    return payload


def build_fixture_plans() -> tuple[FixturePlan, ...]:
    from kokoro import KPipeline

    english_pipeline = KPipeline(lang_code="a", repo_id=MODEL_REPO, model=False)
    portuguese_pipeline = KPipeline(lang_code="p", repo_id=MODEL_REPO, model=False)
    fixtures: list[FixturePlan] = []

    for fixture_id, text in ENGLISH_REAL_PASSAGES:
        phones, tokens = _pipeline_plan(english_pipeline, text)
        fixtures.append(
            FixturePlan(
                fixture_id=fixture_id,
                language_id="en-US",
                fixture_kind="real-passage",
                source_text=text,
                render_text=text,
                phonemes=phones,
                g2p_tokens=tokens,
                gate_receipt=None,
            )
        )

    english_generator_plan = KokoroTypedPlanner.load().plan(ENGLISH_OPAQUE_SOURCE)
    if not (
        english_generator_plan.gate_summary.espeak_gate_pass
        and english_generator_plan.gate_summary.kokoro_phone_gate_pass
        and english_generator_plan.gate_summary.exact_plan_representable
    ):
        raise VoiceScreenError("English opaque carrier generator gate failed")
    english_phones, english_tokens = _pipeline_plan(
        english_pipeline, english_generator_plan.neutral_script
    )
    english_gate = _english_pipeline_gate_receipt(
        carrier_script=english_generator_plan.neutral_script,
        pipeline_tokens=english_tokens,
        generator_plan=english_generator_plan,
    )
    fixtures.append(
        FixturePlan(
            fixture_id="en-opaque-carrier",
            language_id="en-US",
            fixture_kind="opaque-carrier",
            source_text=ENGLISH_OPAQUE_SOURCE,
            render_text=english_generator_plan.neutral_script,
            phonemes=english_phones,
            g2p_tokens=english_tokens,
            gate_receipt=english_gate,
        )
    )

    for fixture_id, text in PORTUGUESE_REAL_PASSAGES:
        phones, tokens = _pipeline_plan(portuguese_pipeline, text)
        fixtures.append(
            FixturePlan(
                fixture_id=fixture_id,
                language_id="pt-BR",
                fixture_kind="real-passage",
                source_text=text,
                render_text=text,
                phonemes=phones,
                g2p_tokens=tokens,
                gate_receipt=None,
            )
        )

    portuguese_plans = [
        plan_portuguese_smoke_fixture_v1(
            PortugueseCarrierPlannerV1.load(voice_id=voice_id)
        )
        for voice_id in PORTUGUESE_SCREEN_VOICES
    ]
    carrier_signatures = {
        (plan.carrier_script, plan.carrier_phonemes) for plan in portuguese_plans
    }
    if len(carrier_signatures) != 1:
        raise VoiceScreenError("Portuguese carrier changed across pinned voice specs")
    portuguese_plan = portuguese_plans[0]
    portuguese_phones, portuguese_tokens = _pipeline_plan(
        portuguese_pipeline, portuguese_plan.carrier_script
    )
    if portuguese_phones != portuguese_plan.carrier_phonemes:
        raise VoiceScreenError("Portuguese frozen planner and KPipeline plans diverged")
    if any(
        not (
            plan.gate_receipt.mandatory_written_espeak_gate_pass
            and plan.gate_receipt.native_positive_only_gate_pass
            and not plan.gate_receipt.native_negative_used_for_clearance
            and plan.gate_receipt.exact_native_phrase_plan
            and plan.gate_receipt.model_representable
            and not plan.candidate_enabled
            and not plan.production_route_available
        )
        for plan in portuguese_plans
    ):
        raise VoiceScreenError("Portuguese frozen carrier receipt failed")
    portuguese_gate: dict[str, Any] = {
        "schema_version": 1,
        "source_fixture_id": PORTUGUESE_SMOKE_FIXTURE_ID_V1,
        "source_text": PORTUGUESE_SMOKE_TEXT_V1,
        "target_phone": PORTUGUESE_SMOKE_TARGET_PHONE_V1,
        "same_carrier_across_voice_specs": True,
        "carrier_script": portuguese_plan.carrier_script,
        "carrier_phonemes": portuguese_plan.carrier_phonemes,
        "per_voice_receipts": {
            plan.voice_id: plan.screening_receipt() for plan in portuguese_plans
        },
        "candidate_enabled": False,
        "production_route_available": False,
        "pass": True,
    }
    portuguese_gate["receipt_sha256"] = sha256_json(portuguese_gate)
    fixtures.append(
        FixturePlan(
            fixture_id="ptbr-opaque-carrier",
            language_id="pt-BR",
            fixture_kind="opaque-carrier",
            source_text=PORTUGUESE_SMOKE_TEXT_V1,
            render_text=portuguese_plan.carrier_script,
            phonemes=portuguese_phones,
            g2p_tokens=portuguese_tokens,
            gate_receipt=portuguese_gate,
        )
    )
    return tuple(fixtures)


def _module_file(module: Any) -> Path:
    return Path(inspect.getfile(module))


def build_asset_bindings(screen_dir: Path | None = None) -> dict[str, Any]:
    import kokoro.pipeline
    import misaki.en
    import misaki.espeak
    import torch

    from . import (
        kokoro_gate_bridge,
        kokoro_specs,
        kokoro_synthesis,
        kokoro_typed_engine,
        listener_lens,
        portuguese_carrier_planner_v1,
        portuguese_kokoro_gate,
    )

    screen_dir = screen_dir or run_dir()
    inventory_path = screen_dir / "inventory.json"
    if not inventory_path.is_file():
        raise VoiceScreenError(f"missing frozen voice inventory: {inventory_path}")
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    if inventory.get("english_screen_shortlist") != list(ENGLISH_SCREEN_SHORTLIST):
        raise VoiceScreenError("frozen English voice shortlist changed")
    if inventory.get("portuguese_screen_voices") != list(PORTUGUESE_SCREEN_VOICES):
        raise VoiceScreenError("frozen Portuguese voice set changed")
    model_files = verify_model_files(download=False)
    selected_voices = (*ENGLISH_SCREEN_SHORTLIST, *PORTUGUESE_SCREEN_VOICES)
    voice_hashes: dict[str, str] = {}
    for voice_id in selected_voices:
        spec = VOICE_SPECS_BY_ID[voice_id]
        path = resolve_pinned_file(spec.filename)
        actual = sha256_file(path)
        if actual != spec.sha256:
            raise VoiceScreenError(f"pinned voice hash changed: {voice_id}")
        voice_hashes[voice_id] = actual

    gate_paths = {
        "written_word_and_espeak_db": Paths().gate_db,
        "english_kokoro_phone_db": Paths().kokoro_gate_db,
        "english_kokoro_full_index_receipt": (
            Paths().artifacts
            / "typed-engine"
            / ENGLISH_KOKORO_GATE_RUN_ID
            / "full-index-receipt.json"
        ),
        "portuguese_kokoro_positive_v1_db": Paths().portuguese_kokoro_gate_db,
        "portuguese_kokoro_positive_v1_receipt": (
            Paths().artifacts
            / "portuguese"
            / PORTUGUESE_KOKORO_GATE_RUN_ID
            / "full-index-receipt.json"
        ),
    }
    missing_gates = [name for name, path in gate_paths.items() if not path.is_file()]
    if missing_gates:
        raise VoiceScreenError(f"missing frozen gate assets: {missing_gates}")

    root = Paths().root
    script_path = root / "scripts" / "run_kokoro_voice_screen_v1.py"
    code_paths = {
        "voice_screen": Path(__file__),
        "runner": script_path,
        "kokoro_specs": _module_file(kokoro_specs),
        "kokoro_synthesis": _module_file(kokoro_synthesis),
        "kokoro_typed_engine": _module_file(kokoro_typed_engine),
        "portuguese_carrier_planner_v1": _module_file(portuguese_carrier_planner_v1),
        "listener_lens": _module_file(listener_lens),
        "kokoro_gate_bridge": _module_file(kokoro_gate_bridge),
        "portuguese_kokoro_gate": _module_file(portuguese_kokoro_gate),
    }
    missing_code = [name for name, path in code_paths.items() if not path.is_file()]
    if missing_code:
        raise VoiceScreenError(f"missing screen code assets: {missing_code}")
    return {
        "source": {
            "inventory_file_sha256": sha256_file(inventory_path),
            "inventory_receipt_sha256": inventory["receipt_sha256"],
            "dependency_lock_sha256": sha256_file(root / "uv.lock"),
        },
        "model": {
            "repo": MODEL_REPO,
            "revision": MODEL_REVISION,
            "kokoro_version": KOKORO_VERSION,
            "wheel_sha256": KOKORO_WHEEL_SHA256,
            "files": {
                name: sha256_file(path) for name, path in sorted(model_files.items())
            },
        },
        "voices": voice_hashes,
        "g2p": {
            "kokoro_version": importlib.metadata.version("kokoro"),
            "misaki_version": importlib.metadata.version("misaki"),
            "espeakng_loader_version": importlib.metadata.version("espeakng-loader"),
            "kokoro_pipeline_py_sha256": sha256_file(_module_file(kokoro.pipeline)),
            "misaki_en_py_sha256": sha256_file(_module_file(misaki.en)),
            "misaki_espeak_py_sha256": sha256_file(_module_file(misaki.espeak)),
        },
        "gates": {name: sha256_file(path) for name, path in sorted(gate_paths.items())},
        "code": {name: sha256_file(path) for name, path in sorted(code_paths.items())},
        "runtime": {
            "python": os.sys.version.split()[0],
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
    }


def render_manifest(
    fixtures: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    fixture_by_language = {
        language_id: [
            fixture for fixture in fixtures if fixture["language_id"] == language_id
        ]
        for language_id in ("en-US", "pt-BR")
    }
    voices = {
        "en-US": ENGLISH_SCREEN_SHORTLIST,
        "pt-BR": PORTUGUESE_SCREEN_VOICES,
    }
    rows: list[dict[str, Any]] = []
    order = 0
    for language_id in ("en-US", "pt-BR"):
        if len(fixture_by_language[language_id]) != 3:
            raise VoiceScreenError(f"{language_id} requires exactly three fixtures")
        for voice_id in voices[language_id]:
            for fixture in fixture_by_language[language_id]:
                group = f"{language_id}__{voice_id}__{fixture['fixture_id']}"
                for render_role in ("primary", "determinism-repeat"):
                    order += 1
                    rows.append(
                        {
                            "request_order": order,
                            "slot_id": f"{group}__{render_role}",
                            "determinism_group": group,
                            "language_id": language_id,
                            "voice_id": voice_id,
                            "fixture_id": fixture["fixture_id"],
                            "fixture_kind": fixture["fixture_kind"],
                            "render_role": render_role,
                            "phonemes_sha256": fixture["phonemes_sha256"],
                            "output_relative_path": f"audio/{order:03d}.wav",
                            "rng_seed": RENDER_SEED,
                            "speed": SPEED,
                            "maximum_attempts": 1,
                            "retry_allowed": False,
                            "replacement_allowed": False,
                        }
                    )
    return rows


def assemble_protocol(
    *,
    fixtures: Sequence[dict[str, Any]],
    asset_bindings: dict[str, Any],
) -> dict[str, Any]:
    if (
        any(spec.renderer_candidate_enabled for spec in LANGUAGE_SPECS.values())
        or PORTUGUESE_RENDERER_CANDIDATE_ENABLED
    ):
        raise VoiceScreenError("renderer candidate flags must remain disabled")
    fixtures = list(fixtures)
    manifest = render_manifest(fixtures)
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": RUN_ID,
        "status": "frozen-before-render",
        "question": (
            "Do the pinned Kokoro English shortlist and all pinned Portuguese "
            "voices pass objective render integrity and merit independent blinded "
            "human quality review on two real passages and one opaque carrier?"
        ),
        "interpretation": (
            "This is a voice-quality and renderer-feasibility screen. Automatic "
            "checks establish integrity only; they never rank or select a voice, "
            "enable a candidate, validate a listener lens, or authorize production."
        ),
        "candidate_flags": {
            "KOKORO_ENGLISH_CANDIDATE_ENABLED": False,
            "PORTUGUESE_RENDERER_CANDIDATE_ENABLED": False,
            "production_selection_performed": False,
        },
        "renderer": {
            "mode": "ordinary independent KModel forward",
            "same_voice_within_each_fixture_repeat": True,
            "model_repo": MODEL_REPO,
            "model_revision": MODEL_REVISION,
            "sample_rate_hz": SAMPLE_RATE_HZ,
            "speed": SPEED,
            "rng_seed_reset_before_every_forward": RENDER_SEED,
            "samples_per_duration_frame": SAMPLES_PER_DURATION_FRAME,
        },
        "scope": {
            "languages": ["en-US", "pt-BR"],
            "english_voices": list(ENGLISH_SCREEN_SHORTLIST),
            "portuguese_voices": list(PORTUGUESE_SCREEN_VOICES),
            "fixtures_per_language": 3,
            "real_passages_per_language": 2,
            "opaque_carriers_per_language": 1,
            "renders_per_voice_fixture": 2,
            "primary_renders_per_voice_fixture": 1,
            "retained_determinism_repeats_per_voice_fixture": 1,
            "total_render_attempts": len(manifest),
            "api_calls": 0,
            "paid_calls": 0,
        },
        "assets": asset_bindings,
        "fixtures": fixtures,
        "render_manifest": manifest,
        "automatic_integrity_gate": {
            "purpose": "file, signal, plan, and determinism integrity only",
            "must_pass_every_output": [
                "mono PCM16 WAV at exactly 24000 Hz",
                "nonempty finite samples",
                f"duration within {MIN_DURATION_S}-{MAX_DURATION_S} seconds",
                f"clipped sample fraction below {MAX_CLIPPED_FRACTION}",
                f"RMS at least {MIN_RMS} and peak at least {MIN_PEAK}",
                f"absolute DC offset at most {MAX_ABS_DC_OFFSET}",
                "finite nondegenerate spectral measurements",
                "current KPipeline plan exactly matches the frozen fixture plan",
                "duration vector matches model token count plus boundaries",
                f"decoder emits exactly {SAMPLES_PER_DURATION_FRAME} samples per duration frame",
                "written WAV PCM round-trips exactly and PCM/WAV hashes are recorded",
            ],
            "must_pass_every_repeat_pair": [
                "predicted duration vectors are identical",
                "PCM16 bytes are bit-identical",
                "complete WAV bytes are bit-identical",
            ],
            "forbidden": [
                "rerender",
                "retry",
                "fixture replacement",
                "clip replacement",
                "metric-based voice ranking",
                "automatic voice selection",
            ],
        },
        "human_review": {
            "created_only_after_all_automatic_integrity_gates_pass": True,
            "primary_clips_only": True,
            "single_clip_no_comparisons": True,
            "blind_order_seed": BLIND_SEED,
            "voice_identity_absent_from_public_manifest_and_review_html": True,
            "separate_private_keys": True,
            "required_fields": [
                "naturalness",
                "accent_fit",
                "sentence_flow",
                "clarity",
                "artifacts",
                "nonce_handling (opaque carrier only)",
            ],
            "english_response_filename": ENGLISH_RESPONSE_FILENAME,
            "portuguese_response_filename": PORTUGUESE_RESPONSE_FILENAME,
            "no_human_threshold_or_selection_preregistered": True,
        },
    }
    payload["protocol_sha256"] = _sha256_payload(payload)
    return payload


def protocol_record(screen_dir: Path | None = None) -> dict[str, Any]:
    screen_dir = screen_dir or run_dir()
    config = json.loads(resolve_pinned_file(CONFIG_FILE).read_text(encoding="utf-8"))
    model_vocab = config["vocab"]
    fixtures = [plan.record(model_vocab) for plan in build_fixture_plans()]
    return assemble_protocol(
        fixtures=fixtures,
        asset_bindings=build_asset_bindings(screen_dir),
    )


def verify_protocol(protocol: dict[str, Any]) -> str:
    if protocol.get("run_id") != RUN_ID:
        raise VoiceScreenError("voice-screen run id changed")
    expected = protocol.get("protocol_sha256")
    unsigned = dict(protocol)
    unsigned.pop("protocol_sha256", None)
    actual = _sha256_payload(unsigned)
    if expected != actual:
        raise VoiceScreenError("voice-screen protocol hash mismatch")
    if protocol.get("status") != "frozen-before-render":
        raise VoiceScreenError("voice-screen protocol is not frozen before render")
    return actual


def prepare_screen(screen_dir: Path | None = None) -> dict[str, Any]:
    screen_dir = screen_dir or run_dir()
    if (screen_dir / "render-started.json").exists():
        raise VoiceScreenError("render has already started; protocol cannot change")
    preexisting_render_outputs = [
        path
        for path in screen_dir.rglob("*")
        if path.is_file()
        and (
            path.suffix.casefold() == ".wav"
            or bool(
                set(path.relative_to(screen_dir).parts) & {"audio", "review", "private"}
            )
            or path.name
            in {
                "records.json",
                "records.in-progress.json",
                "summary.json",
                "review.html",
            }
        )
    ]
    if preexisting_render_outputs:
        raise VoiceScreenError("render outputs exist before the protocol freeze")
    protocol = protocol_record(screen_dir)
    path = screen_dir / "protocol.json"
    if path.exists():
        current = json.loads(path.read_text(encoding="utf-8"))
        if current != protocol:
            raise VoiceScreenError("existing frozen protocol differs from current plan")
    else:
        atomic_write_json(path, protocol)
    return {
        "run_id": RUN_ID,
        "status": protocol["status"],
        "protocol": str(path),
        "protocol_sha256": protocol["protocol_sha256"],
        "render_attempts": protocol["scope"]["total_render_attempts"],
        "novel_wavs": 0,
    }


def _spectral_metrics(values: np.ndarray, sample_rate_hz: int) -> dict[str, float]:
    centered = values - float(np.mean(values))
    if centered.size < 2:
        return {
            "spectral_centroid_hz": 0.0,
            "spectral_bandwidth_hz": 0.0,
            "spectral_rolloff_95_hz": 0.0,
            "spectral_flatness": 0.0,
            "zero_crossing_rate": 0.0,
        }
    spectrum = np.abs(np.fft.rfft(centered * np.hanning(centered.size)))
    power = np.square(spectrum)
    frequencies = np.fft.rfftfreq(centered.size, 1.0 / sample_rate_hz)
    total = float(power.sum())
    if total <= 0 or not np.isfinite(total):
        centroid = bandwidth = rolloff = flatness = 0.0
    else:
        centroid = float(np.sum(frequencies * power) / total)
        bandwidth = float(
            np.sqrt(np.sum(np.square(frequencies - centroid) * power) / total)
        )
        cumulative = np.cumsum(power)
        rolloff_index = min(
            int(np.searchsorted(cumulative, cumulative[-1] * 0.95)),
            len(frequencies) - 1,
        )
        rolloff = float(frequencies[rolloff_index])
        positive = spectrum[spectrum > 0]
        flatness = (
            float(np.exp(np.mean(np.log(positive))) / np.mean(positive))
            if positive.size
            else 0.0
        )
    crossings = float(np.mean(centered[:-1] * centered[1:] < 0))
    return {
        "spectral_centroid_hz": centroid,
        "spectral_bandwidth_hz": bandwidth,
        "spectral_rolloff_95_hz": rolloff,
        "spectral_flatness": flatness,
        "zero_crossing_rate": crossings,
    }


def inspect_audio(
    *,
    audio: np.ndarray,
    predicted_durations: Sequence[int],
    phonemes: str,
    model_vocab: dict[str, int],
    sample_rate_hz: int = SAMPLE_RATE_HZ,
) -> dict[str, Any]:
    values = np.asarray(audio, dtype=np.float64).reshape(-1)
    finite = bool(values.size and np.isfinite(values).all())
    safe_values = values if finite else np.zeros(max(1, values.size), dtype=np.float64)
    durations = tuple(int(value) for value in predicted_durations)
    model_token_count = sum(
        1 for symbol in phonemes if model_vocab.get(symbol) is not None
    )
    duration_count_pass = len(durations) == model_token_count + 2
    durations_positive = bool(durations and min(durations) >= 1)
    total_frames = sum(durations) if durations_positive else 0
    expected_samples = total_frames * SAMPLES_PER_DURATION_FRAME
    samples_per_frame = values.size / total_frames if total_frames else None
    duration_plan_pass = bool(
        duration_count_pass
        and durations_positive
        and values.size == expected_samples
        and samples_per_frame == SAMPLES_PER_DURATION_FRAME
    )
    duration_s = values.size / sample_rate_hz if sample_rate_hz else 0.0
    peak = float(np.max(np.abs(safe_values)))
    rms = float(np.sqrt(np.mean(np.square(safe_values))))
    dc_offset = float(np.mean(safe_values))
    clipped_fraction = float(np.mean(np.abs(safe_values) >= 1.0))
    spectral = _spectral_metrics(safe_values, sample_rate_hz)
    spectral_finite = bool(all(np.isfinite(value) for value in spectral.values()))
    checks = {
        "sample_rate_pass": sample_rate_hz == SAMPLE_RATE_HZ,
        "nonempty_pass": values.size > 0,
        "finite_pass": finite,
        "duration_bounds_pass": MIN_DURATION_S <= duration_s <= MAX_DURATION_S,
        "clipping_pass": clipped_fraction < MAX_CLIPPED_FRACTION,
        "rms_pass": rms >= MIN_RMS,
        "peak_pass": peak >= MIN_PEAK,
        "dc_offset_pass": abs(dc_offset) <= MAX_ABS_DC_OFFSET,
        "spectral_integrity_pass": spectral_finite
        and spectral["spectral_centroid_hz"] > 0,
        "duration_plan_pass": duration_plan_pass,
    }
    return {
        "sample_rate_hz": sample_rate_hz,
        "sample_count": int(values.size),
        "duration_s": duration_s,
        "finite": finite,
        "clipped_fraction": clipped_fraction,
        "rms": rms,
        "peak": peak,
        "dc_offset": dc_offset,
        **spectral,
        "model_token_count_without_boundaries": model_token_count,
        "predicted_durations": list(durations),
        "predicted_durations_sha256": sha256_json(list(durations)),
        "predicted_duration_frame_count": total_frames,
        "expected_sample_count_from_duration_plan": expected_samples,
        "samples_per_duration_frame": samples_per_frame,
        "checks": checks,
        "integrity_pass": all(checks.values()),
    }


def _write_wav(path: Path, audio: np.ndarray) -> tuple[bytes, str]:
    pcm = pcm16_bytes(audio)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".partial.wav")
    with wave.open(str(partial), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE_HZ)
        handle.writeframes(pcm)
    partial.replace(path)
    return pcm, sha256_file(path)


def _wav_receipt(path: Path) -> dict[str, Any]:
    with wave.open(str(path), "rb") as handle:
        payload = {
            "channels": handle.getnchannels(),
            "sample_width_bytes": handle.getsampwidth(),
            "sample_rate_hz": handle.getframerate(),
            "sample_count": handle.getnframes(),
            "compression_type": handle.getcomptype(),
            "pcm_sha256": sha256_bytes(handle.readframes(handle.getnframes())),
        }
    payload["wav_conformance_pass"] = bool(
        payload["channels"] == 1
        and payload["sample_width_bytes"] == 2
        and payload["sample_rate_hz"] == SAMPLE_RATE_HZ
        and payload["sample_count"] > 0
        and payload["compression_type"] == "NONE"
    )
    return payload


class KokoroOrdinaryRuntime:
    def __init__(self, model: Any, voice_packs: dict[str, Any], torch: Any) -> None:
        self.model = model
        self.model_vocab = model.vocab
        self.voice_packs = voice_packs
        self.torch = torch

    @classmethod
    def load(cls) -> KokoroOrdinaryRuntime:
        if importlib.metadata.version("kokoro") != KOKORO_VERSION:
            raise VoiceScreenError(f"Kokoro {KOKORO_VERSION} is required")
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        files = verify_model_files(download=False)
        import torch
        from kokoro import KModel

        with _INFERENCE_LOCK:
            torch.set_num_threads(1)
            try:
                torch.set_num_interop_threads(1)
            except RuntimeError:
                pass
            torch.backends.mkldnn.enabled = False
            torch.backends.nnpack.set_flags(False)
            torch.use_deterministic_algorithms(True)
            model = (
                KModel(
                    repo_id=MODEL_REPO,
                    config=str(files[CONFIG_FILE]),
                    model=str(files[MODEL_FILE]),
                )
                .to("cpu")
                .eval()
            )
            voice_packs = {
                voice_id: torch.load(
                    resolve_pinned_file(VOICE_SPECS_BY_ID[voice_id].filename),
                    map_location="cpu",
                    weights_only=True,
                )
                for voice_id in (*ENGLISH_SCREEN_SHORTLIST, *PORTUGUESE_SCREEN_VOICES)
            }
        return cls(model, voice_packs, torch)

    def render(self, *, voice_id: str, phonemes: str) -> RenderOutput:
        if voice_id not in self.voice_packs:
            raise VoiceScreenError(f"voice is outside the frozen screen: {voice_id}")
        if not 1 <= len(phonemes) <= MAX_PHONEME_CHARACTERS:
            raise VoiceScreenError("phoneme plan has no pinned voice style row")
        pack = self.voice_packs[voice_id]
        with _INFERENCE_LOCK, self.torch.no_grad():
            self.torch.manual_seed(RENDER_SEED)
            output = self.model(
                phonemes,
                pack[len(phonemes) - 1],
                SPEED,
                return_output=True,
            )
        if output.pred_dur is None:
            raise VoiceScreenError("ordinary Kokoro forward returned no durations")
        return RenderOutput(
            audio=output.audio.detach().cpu().numpy(),
            predicted_durations=tuple(
                int(value) for value in output.pred_dur.reshape(-1).tolist()
            ),
        )


def determinism_report(
    records: Sequence[dict[str, Any]], manifest: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    records_by_slot = {record["slot_id"]: record for record in records}
    groups: dict[str, list[dict[str, Any]]] = {}
    for slot in manifest:
        groups.setdefault(slot["determinism_group"], []).append(slot)
    pair_rows: list[dict[str, Any]] = []
    for group_id, slots in groups.items():
        if [slot["render_role"] for slot in slots] != [
            "primary",
            "determinism-repeat",
        ]:
            raise VoiceScreenError(f"invalid determinism pair: {group_id}")
        primary = records_by_slot.get(slots[0]["slot_id"])
        repeated = records_by_slot.get(slots[1]["slot_id"])
        both_pass = bool(
            primary
            and repeated
            and primary.get("integrity_pass")
            and repeated.get("integrity_pass")
        )
        durations_identical = bool(
            both_pass
            and primary["audio_metrics"]["predicted_durations"]
            == repeated["audio_metrics"]["predicted_durations"]
        )
        pcm_identical = bool(
            both_pass and primary["pcm_sha256"] == repeated["pcm_sha256"]
        )
        wav_identical = bool(
            both_pass and primary["wav_sha256"] == repeated["wav_sha256"]
        )
        pair_rows.append(
            {
                "determinism_group": group_id,
                "primary_slot_id": slots[0]["slot_id"],
                "repeat_slot_id": slots[1]["slot_id"],
                "durations_bit_identical": durations_identical,
                "pcm16_bit_identical": pcm_identical,
                "wav_bit_identical": wav_identical,
                "pass": durations_identical and pcm_identical and wav_identical,
            }
        )
    return {
        "pair_count": len(pair_rows),
        "pass_count": sum(row["pass"] for row in pair_rows),
        "pairs": pair_rows,
        "pass": bool(pair_rows and all(row["pass"] for row in pair_rows)),
    }


def _blind_id(protocol_sha256: str, slot_id: str) -> str:
    return hashlib.sha256(
        f"{BLIND_SEED}:{protocol_sha256}:{slot_id}".encode("utf-8")
    ).hexdigest()[:14]


def _review_html(public: dict[str, Any]) -> str:
    rows = json.dumps(public["clips"], ensure_ascii=False).replace("</", "<\\/")
    language_name = (
        "American English"
        if public["language_id"] == "en-US"
        else "Brazilian Portuguese"
    )
    values = {
        "__ROWS__": rows,
        "__RUN_ID__": RUN_ID,
        "__PROTOCOL__": public["protocol_sha256"],
        "__LANGUAGE__": language_name,
        "__LANGUAGE_ID__": public["language_id"],
        "__RESPONSE__": public["response_filename"],
        "__STORAGE_KEY__": f"{RUN_ID}-{public['language_id']}-review",
    }
    html = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="Content-Security-Policy" content="default-src 'self'; media-src 'self'; script-src 'unsafe-inline'; style-src 'unsafe-inline'"><title>Blind Kokoro voice screen</title><style>:root{font-family:ui-sans-serif,system-ui,sans-serif;color:#18221d;background:#f2efe7}*{box-sizing:border-box}body{margin:0}.wrap{max-width:860px;margin:auto;padding:28px 18px 80px}h1{font-size:clamp(2rem,7vw,4.4rem);line-height:.95;letter-spacing:-.05em}.lede{max-width:720px;color:#58635b}.panel,.clip{background:#fff;border:1px solid #d8d3c7;border-radius:18px;padding:20px;margin:18px 0;box-shadow:0 8px 24px #302d2410}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}label{display:block;font-weight:700;margin:11px 0}select,textarea,input{width:100%;font:inherit;padding:10px;border:1px solid #aaa79d;border-radius:9px;background:#fff}audio{width:100%}.hint{font-size:.9rem;color:#657067}.nonce{background:#eef3ec;border-radius:12px;padding:12px}.actions{position:sticky;bottom:0;padding:12px 0;background:#f2efe7e8;backdrop-filter:blur(7px)}button{font:inherit;font-weight:800;color:#fff;background:#174b38;border:0;border-radius:999px;padding:12px 18px}.error{color:#9b2c2c;font-weight:700}@media(max-width:620px){.grid{grid-template-columns:1fr}}</style></head><body><main class="wrap"><p><b>Blinded single-clip screen · __LANGUAGE__</b></p><h1>Judge each voice on its own.</h1><p class="lede">Use headphones at one comfortable volume. Clips are shuffled and voice identities are hidden. Do not compare clips or try to rank voices; rate the clip currently in front of you. The retained determinism repeats are not part of this review.</p><section class="panel"><div class="grid"><div><label for="reviewer">Reviewer code</label><input id="reviewer" autocomplete="off"></div><div><label for="background">Language background</label><input id="background" autocomplete="off"></div></div></section><div id="clips"></div><p id="error" class="error"></p><div class="actions"><button id="download">Download __RESPONSE__</button></div></main><script>const ROWS=__ROWS__;const KEY='__STORAGE_KEY__';let S=JSON.parse(localStorage.getItem(KEY)||'{"meta":{},"clips":{}}');const save=()=>localStorage.setItem(KEY,JSON.stringify(S));const esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'}[c]));const opt=(id,k,vs)=>'<option value="">—</option>'+vs.map(v=>`<option value="${v}" ${S.clips[id]?.[k]===v?'selected':''}>${v}</option>`).join('');document.getElementById('reviewer').value=S.meta.reviewer??'';document.getElementById('background').value=S.meta.background??'';document.getElementById('reviewer').oninput=e=>{S.meta.reviewer=e.target.value;save()};document.getElementById('background').oninput=e=>{S.meta.background=e.target.value;save()};document.getElementById('clips').innerHTML=ROWS.map((r,i)=>`<article class="clip"><h2>Clip ${i+1}</h2><audio controls preload="metadata" data-id="${r.blind_id}" src="${r.audio}"></audio>${r.reference_text?`<p class="hint">Expected passage: ${esc(r.reference_text)}</p>`:'<p class="nonce">This is an invented-word carrier. Judge whether the nonce material is handled fluently as one utterance.</p>'}<div class="grid"><label>Naturalness (1–5)<select data-id="${r.blind_id}" data-k="naturalness">${opt(r.blind_id,'naturalness',['1','2','3','4','5'])}</select></label><label>__LANGUAGE__ accent fit (1–5)<select data-id="${r.blind_id}" data-k="accent_fit">${opt(r.blind_id,'accent_fit',['1','2','3','4','5'])}</select></label><label>Sentence flow (1–5)<select data-id="${r.blind_id}" data-k="sentence_flow">${opt(r.blind_id,'sentence_flow',['1','2','3','4','5'])}</select></label><label>Clarity (1–5)<select data-id="${r.blind_id}" data-k="clarity">${opt(r.blind_id,'clarity',['1','2','3','4','5'])}</select></label><label>Artifacts<select data-id="${r.blind_id}" data-k="artifacts">${opt(r.blind_id,'artifacts',['none','minor','major','uncertain'])}</select></label>${r.opaque?`<label>Nonce handling<select data-id="${r.blind_id}" data-k="nonce_handling">${opt(r.blind_id,'nonce_handling',['fluent','mostly-fluent','word-list-like','broken','uncertain'])}</select></label>`:''}</div><label>Optional notes<textarea data-id="${r.blind_id}" data-k="notes">${esc(S.clips[r.blind_id]?.notes??'')}</textarea></label></article>`).join('');document.querySelectorAll('[data-id][data-k]').forEach(el=>el.oninput=()=>{const id=el.dataset.id;S.clips[id]??={plays:0};S.clips[id][el.dataset.k]=el.value;save()});document.querySelectorAll('audio').forEach(el=>el.onplay=()=>{S.clips[el.dataset.id]??={plays:0};S.clips[el.dataset.id].plays=(S.clips[el.dataset.id].plays??0)+1;save()});document.getElementById('download').onclick=()=>{const error=document.getElementById('error');error.textContent='';if(!S.meta.reviewer?.trim()||!S.meta.background?.trim()){error.textContent='Enter reviewer code and language background.';return}for(const r of ROWS){const x=S.clips[r.blind_id]??{};if(!x.naturalness||!x.accent_fit||!x.sentence_flow||!x.clarity||!x.artifacts||(r.opaque&&!x.nonce_handling)){error.textContent='Complete every required rating.';return}}const payload={schema_version:1,run_id:'__RUN_ID__',protocol_sha256:'__PROTOCOL__',language_id:'__LANGUAGE_ID__',saved_at:new Date().toISOString(),reviewer:S.meta,ratings:ROWS.map(r=>({blind_id:r.blind_id,...S.clips[r.blind_id]}))};const blob=new Blob([JSON.stringify(payload,null,2)+'\n'],{type:'application/json'});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='__RESPONSE__';a.click();URL.revokeObjectURL(a.href)};</script></body></html>"""
    for needle, replacement in values.items():
        html = html.replace(needle, replacement)
    return html


def build_review_session(
    *,
    screen_dir: Path,
    language_id: str,
    protocol: dict[str, Any],
    records: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    language_slug = "en" if language_id == "en-US" else "ptbr"
    response_filename = (
        ENGLISH_RESPONSE_FILENAME
        if language_id == "en-US"
        else PORTUGUESE_RESPONSE_FILENAME
    )
    fixture_by_id = {fixture["fixture_id"]: fixture for fixture in protocol["fixtures"]}
    primary = [
        record
        for record in records
        if record["language_id"] == language_id
        and record["render_role"] == "primary"
        and record["integrity_pass"]
    ]
    expected_count = (
        len(ENGLISH_SCREEN_SHORTLIST) * 3
        if language_id == "en-US"
        else len(PORTUGUESE_SCREEN_VOICES) * 3
    )
    if len(primary) != expected_count:
        raise VoiceScreenError("review cannot be built from incomplete primary clips")
    rows: list[dict[str, Any]] = []
    keys: dict[str, Any] = {}
    review_dir = screen_dir / "review" / language_slug
    audio_dir = review_dir / "audio"
    for record in primary:
        fixture = fixture_by_id[record["fixture_id"]]
        blind_id = _blind_id(protocol["protocol_sha256"], record["slot_id"])
        destination = audio_dir / f"{blind_id}.wav"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(screen_dir / record["audio_relative_path"], destination)
        if sha256_file(destination) != record["wav_sha256"]:
            raise VoiceScreenError("blinded WAV copy changed")
        rows.append(
            {
                "blind_id": blind_id,
                "audio": f"audio/{blind_id}.wav",
                "opaque": fixture["fixture_kind"] == "opaque-carrier",
                "reference_text": fixture["review_reference_text"],
            }
        )
        keys[blind_id] = {
            "voice_id": record["voice_id"],
            "fixture_id": record["fixture_id"],
            "fixture_kind": record["fixture_kind"],
            "slot_id": record["slot_id"],
            "source_audio_relative_path": record["audio_relative_path"],
            "wav_sha256": record["wav_sha256"],
            "pcm_sha256": record["pcm_sha256"],
        }
    rng = random.Random(f"{BLIND_SEED}:{protocol['protocol_sha256']}:{language_id}")
    rng.shuffle(rows)
    rows = [
        {"presentation_order": index, **row} for index, row in enumerate(rows, start=1)
    ]
    public = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "protocol_sha256": protocol["protocol_sha256"],
        "status": "pending-human-review",
        "language_id": language_id,
        "design": "blinded shuffled single-clip primary-only review",
        "response_filename": response_filename,
        "clips": rows,
    }
    private = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "protocol_sha256": protocol["protocol_sha256"],
        "language_id": language_id,
        "public_manifest_sha256": sha256_json(public),
        "mapping": keys,
    }
    public_path = review_dir / "public-manifest.json"
    private_path = screen_dir / "private" / f"{language_slug}-blind-key.json"
    html_path = review_dir / "review.html"
    atomic_write_json(public_path, public)
    atomic_write_json(private_path, private)
    html = _review_html(public)
    forbidden = set(
        ENGLISH_SCREEN_SHORTLIST if language_id == "en-US" else PORTUGUESE_SCREEN_VOICES
    )
    if any(
        identity in html or identity in stable_json(public) for identity in forbidden
    ):
        raise VoiceScreenError("voice identity leaked into public blind session")
    atomic_write_text(html_path, html)
    return {
        "language_id": language_id,
        "status": "pending-human-review",
        "clip_count": len(rows),
        "public_manifest": str(public_path.relative_to(screen_dir)),
        "public_manifest_file_sha256": sha256_file(public_path),
        "private_key": str(private_path.relative_to(screen_dir)),
        "private_key_file_sha256": sha256_file(private_path),
        "review_html": str(html_path.relative_to(screen_dir)),
        "review_html_sha256": sha256_file(html_path),
        "response_filename": response_filename,
    }


def _require_committed_protocol(screen_dir: Path, protocol_path: Path) -> None:
    try:
        relative = protocol_path.relative_to(Paths().root)
    except ValueError as exc:
        raise VoiceScreenError(
            "production render protocol must be inside the repo"
        ) from exc
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", str(relative)],
        cwd=Paths().root,
        capture_output=True,
        text=True,
        check=False,
    )
    clean = subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--", str(relative)],
        cwd=Paths().root,
        check=False,
    )
    if tracked.returncode or clean.returncode:
        raise VoiceScreenError(
            "protocol.json must be tracked and committed before --render"
        )
    if screen_dir != run_dir():
        raise VoiceScreenError(
            "committed-protocol enforcement applies to frozen run dir"
        )


def render_screen(
    *,
    screen_dir: Path | None = None,
    runtime: OrdinaryRuntime | None = None,
    require_committed_protocol: bool = True,
) -> dict[str, Any]:
    screen_dir = screen_dir or run_dir()
    protocol_path = screen_dir / "protocol.json"
    if not protocol_path.is_file():
        raise VoiceScreenError(
            "run --prepare and commit protocol.json before rendering"
        )
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    verify_protocol(protocol)
    if require_committed_protocol:
        _require_committed_protocol(screen_dir, protocol_path)
    current = protocol_record(screen_dir)
    if current != protocol:
        raise VoiceScreenError(
            "current assets, plans, or code differ from frozen protocol"
        )
    started_path = screen_dir / "render-started.json"
    if started_path.exists():
        raise VoiceScreenError("render already started; this protocol forbids reruns")

    runtime = runtime or KokoroOrdinaryRuntime.load()
    fixture_by_id = {fixture["fixture_id"]: fixture for fixture in protocol["fixtures"]}
    atomic_write_json(
        started_path,
        {
            "schema_version": 1,
            "run_id": RUN_ID,
            "protocol_sha256": protocol["protocol_sha256"],
            "status": "render-started-no-retry",
            "started_at_utc": datetime.now(UTC).isoformat(),
            "attempt_limit_per_manifest_slot": 1,
        },
    )
    records: list[dict[str, Any]] = []
    for slot in protocol["render_manifest"]:
        fixture = fixture_by_id[slot["fixture_id"]]
        record: dict[str, Any] = {
            **slot,
            "status": "render-error",
            "attempt_number": 1,
            "integrity_pass": False,
        }
        try:
            output = runtime.render(
                voice_id=slot["voice_id"], phonemes=fixture["phonemes"]
            )
            metrics = inspect_audio(
                audio=output.audio,
                predicted_durations=output.predicted_durations,
                phonemes=fixture["phonemes"],
                model_vocab=runtime.model_vocab,
            )
            path = screen_dir / slot["output_relative_path"]
            pcm, wav_hash = _write_wav(path, output.audio)
            wav_receipt = _wav_receipt(path)
            pcm_roundtrip_pass = wav_receipt["pcm_sha256"] == sha256_bytes(pcm)
            metrics["checks"]["wav_conformance_pass"] = wav_receipt[
                "wav_conformance_pass"
            ]
            metrics["checks"]["pcm_roundtrip_pass"] = pcm_roundtrip_pass
            metrics["integrity_pass"] = all(metrics["checks"].values())
            record.update(
                {
                    "status": (
                        "passed-integrity"
                        if metrics["integrity_pass"]
                        else "failed-integrity"
                    ),
                    "audio_relative_path": slot["output_relative_path"],
                    "wav_sha256": wav_hash,
                    "pcm_sha256": sha256_bytes(pcm),
                    "wav_receipt": wav_receipt,
                    "audio_metrics": metrics,
                    "integrity_pass": metrics["integrity_pass"],
                }
            )
        except Exception as exc:  # Preserve one failed attempt without retrying.
            record["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
        records.append(record)
        atomic_write_json(screen_dir / "records.in-progress.json", records)

    automatic_outputs_pass = bool(
        len(records) == len(protocol["render_manifest"])
        and all(record["integrity_pass"] for record in records)
    )
    determinism = determinism_report(records, protocol["render_manifest"])
    automatic_pass = automatic_outputs_pass and determinism["pass"]
    reviews: list[dict[str, Any]] = []
    if automatic_pass:
        reviews = [
            build_review_session(
                screen_dir=screen_dir,
                language_id=language_id,
                protocol=protocol,
                records=records,
            )
            for language_id in ("en-US", "pt-BR")
        ]
    summary = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "protocol_sha256": protocol["protocol_sha256"],
        "status": (
            "pending-human-review" if automatic_pass else "blocked-integrity-failure"
        ),
        "expected_render_attempts": len(protocol["render_manifest"]),
        "completed_render_attempts": len(records),
        "integrity_pass_count": sum(record["integrity_pass"] for record in records),
        "automatic_output_integrity_pass": automatic_outputs_pass,
        "determinism": determinism,
        "automatic_integrity_pass": automatic_pass,
        "voice_selection_performed": False,
        "production_candidate_enabled": False,
        "renderer_candidate_flags_unchanged": True,
        "human_reviews": reviews,
        "api_calls": 0,
        "paid_calls": 0,
    }
    atomic_write_json(screen_dir / "records.json", records)
    atomic_write_json(screen_dir / "summary.json", summary)
    return summary
