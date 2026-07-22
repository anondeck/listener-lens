from __future__ import annotations

import base64
import binascii
import hashlib
import json
import os
import re
import socket
import tempfile
import threading
import wave
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from .gates import EspeakPhonemizer
from .bilingual_product_matrix import (
    load_bilingual_product_matrix,
    load_bilingual_structural_state,
)
from .bilingual_product_audio_state import (
    load_bilingual_audio_integrity_state,
    load_bilingual_isolated_audio_state,
)
from .listener_lens import (
    TRANSFORM_ALGORITHM_VERSION,
    ListenerLensEngine,
    ListenerLensError,
)
from .runtime_audio import (
    analyze_audio_timing,
    analyze_prosody_fingerprint,
    check_transcript,
)
from .product_voices import ProductVoiceError, load_product_voice_registry
from .util import sha256_file


SERVICE_VERSION = "typed-transform-service-v1"
KOKORO_PROFILE_ID = "en-to-pt-BR-vowel-lens"
KOKORO_CURRENT_EVIDENCE_VOICE_ID = "af_heart"
BILINGUAL_PROFILE_SOURCE_LANGUAGES = {
    "en-US-to-pt-BR-listener-v2": "en-US",
    "pt-BR-to-en-US-listener-v2": "pt-BR",
}
MAX_REQUEST_BYTES = 6 * 1024 * 1024
MAX_AUDIO_BYTES = 3 * 1024 * 1024
MAX_SCRIPT_CHARS = 2_000
MAX_PROVIDER_TRANSCRIPT_CHARS = 4_000
REQUIRED_SAMPLE_RATE_HZ = 24_000
MIN_AUDIO_DURATION_S = 0.25
MAX_AUDIO_DURATION_S = 45.0
MAX_CLIPPED_FRACTION = 0.001
MAX_CONCURRENT_INFERENCE_REQUESTS = 2
MAX_QUEUED_INFERENCE_REQUESTS = 4
MAX_ADMITTED_INFERENCE_REQUESTS = (
    MAX_CONCURRENT_INFERENCE_REQUESTS + MAX_QUEUED_INFERENCE_REQUESTS
)
MAX_HTTP_HANDLER_THREADS = MAX_ADMITTED_INFERENCE_REQUESTS + 2
INFERENCE_QUEUE_WAIT_S = 2.0
SOCKET_READ_TIMEOUT_S = 10.0
HTTP_REQUEST_QUEUE_SIZE = 16
OVERLOAD_RETRY_AFTER_S = 2

_JSON_CONTENT_TYPE = re.compile(
    r"application/json(?:\s*;\s*charset\s*=\s*(?:utf-8|\"utf-8\"))?",
    re.IGNORECASE,
)


class DeployService:
    """Internal-only deterministic transform and renderer-output inspector."""

    def __init__(
        self,
        engine: ListenerLensEngine | None = None,
        *,
        kokoro_candidate: Any | None = None,
        kokoro_candidate_enabled: bool | None = None,
        bilingual_candidate: Any | None = None,
        bilingual_candidate_enabled: bool | None = None,
    ) -> None:
        self.engine = engine or ListenerLensEngine()
        self.kokoro_candidate_enabled = (
            os.environ.get("KOKORO_ENGLISH_CANDIDATE_ENABLED") == "true"
            if kokoro_candidate_enabled is None
            else kokoro_candidate_enabled
        )
        self._kokoro_candidate = kokoro_candidate
        self._kokoro_candidate_lock = threading.Lock()
        self.bilingual_candidate_enabled = (
            os.environ.get("KOKORO_BILINGUAL_CANDIDATE_ENABLED") == "true"
            if bilingual_candidate_enabled is None
            else bilingual_candidate_enabled
        )
        self._bilingual_candidate = bilingual_candidate
        self._bilingual_candidate_runtimes: dict[tuple[str, str], Any] = {}
        self._bilingual_candidate_lock = threading.Lock()
        self.azure_lens_enabled = (
            os.environ.get("AZURE_LENS_CANDIDATE_ENABLED") == "true"
        )
        self._azure_pair_cache: dict[str, dict[str, Any]] = {}
        self._azure_pair_lock = threading.Lock()
        # Gibberish is a second mode making a different claim, not another
        # listener direction, so it gates separately: the lens can ship
        # without it and it can be pulled without touching the lens.
        self.gibberish_enabled = (
            os.environ.get("GIBBERISH_CANDIDATE_ENABLED") == "true"
        )
        self._gibberish_cache: dict[str, dict[str, Any]] = {}
        self._gibberish_lock = threading.Lock()
        self.product_voices = load_product_voice_registry()
        self.bilingual_product_matrix = load_bilingual_product_matrix()
        self.bilingual_structural_state = load_bilingual_structural_state(
            self.bilingual_product_matrix,
            verify_result_artifact=False,
        )
        self.bilingual_audio_integrity_state = (
            load_bilingual_audio_integrity_state(
                matrix_version=self.bilingual_product_matrix.matrix_version,
                matrix_sha256=self.bilingual_product_matrix.matrix_sha256,
                verify_result_artifact=False,
            )
        )
        self.bilingual_isolated_audio_state = load_bilingual_isolated_audio_state(
            matrix_version=self.bilingual_product_matrix.matrix_version,
            matrix_sha256=self.bilingual_product_matrix.matrix_sha256,
            verify_result_artifact=False,
        )
        gate = self.engine.nonce_checker
        database = getattr(getattr(gate, "gate", None), "database", None)
        self.rules_sha256 = sha256_file(self.engine.rules_path)
        self.gate_database_sha256 = sha256_file(Path(database)) if database else None
        self.espeak_version = EspeakPhonemizer().version()

    def health(self) -> dict[str, Any]:
        gate = self.engine.nonce_checker
        return {
            "status": "ok" if gate.enabled else "unavailable",
            "service_version": SERVICE_VERSION,
            "transform_algorithm_version": TRANSFORM_ALGORITHM_VERSION,
            "rules_sha256": self.rules_sha256,
            "nonce_gate_enabled": gate.enabled,
            "gate_database_sha256": self.gate_database_sha256,
            "espeak_version": self.espeak_version,
            "kokoro_candidate_enabled": self.kokoro_candidate_enabled,
            "kokoro_candidate_loaded": self._kokoro_candidate is not None,
            "azure_lens_enabled": self.azure_lens_enabled,
            "gibberish_enabled": self.gibberish_enabled,
            "bilingual_candidate_enabled": self.bilingual_candidate_enabled,
            "bilingual_candidate_loaded": bool(
                self._bilingual_candidate is not None
                or self._bilingual_candidate_runtimes
            ),
            "voice_registry_version": self.product_voices.registry_version,
            "voice_registry_sha256": self.product_voices.registry_sha256,
            "configured_voice_ids": sorted(self.product_voices.voices),
            "bilingual_matrix_version": (
                self.bilingual_product_matrix.matrix_version
            ),
            "bilingual_matrix_sha256": (
                self.bilingual_product_matrix.matrix_sha256
            ),
            "bilingual_structural_classification": (
                self.bilingual_structural_state["classification"]
            ),
            "bilingual_audio_validation_status": (
                self.bilingual_audio_integrity_state[
                    "family_acoustic_validation_status"
                ]
            ),
            "bilingual_audio_integrity_classification": (
                self.bilingual_audio_integrity_state["classification"]
            ),
            "bilingual_isolated_audio_classification": (
                self.bilingual_isolated_audio_state["classification"]
            ),
        }

    def bilingual_capabilities(self) -> dict[str, Any]:
        catalog = self.bilingual_product_matrix.safe_catalog()
        return {
            **catalog,
            "structural_planner_gate_yield": self.bilingual_structural_state[
                "planner_gate_yield"
            ],
            "structural_planner_slot_count": self.bilingual_structural_state[
                "planner_slot_count"
            ],
            "audio_integrity_classification": (
                self.bilingual_audio_integrity_state["classification"]
            ),
            "audio_integrity_gate_yield": (
                self.bilingual_audio_integrity_state[
                    "universal_integrity_yield"
                ]
            ),
            "audio_integrity_slot_count": self.bilingual_audio_integrity_state[
                "slot_count"
            ],
            "isolated_audio_classification": (
                self.bilingual_isolated_audio_state["classification"]
            ),
            "isolated_audio_integrity_gate_yield": (
                self.bilingual_isolated_audio_state[
                    "isolated_universal_integrity_yield"
                ]
            ),
            "isolated_audio_slot_count": self.bilingual_isolated_audio_state[
                "slot_count"
            ],
            "audio_validation_status": self.bilingual_audio_integrity_state[
                "family_acoustic_validation_status"
            ],
        }

    def deploy_contract(self, profile_id: str) -> dict[str, Any]:
        profile = next(
            profile
            for profile in self.engine.rules["profiles"]
            if profile["id"] == profile_id
        )
        return {
            "service_version": SERVICE_VERSION,
            "transform_algorithm_version": TRANSFORM_ALGORITHM_VERSION,
            "rules_sha256": self.rules_sha256,
            "gate_database_sha256": self.gate_database_sha256,
            "schema_version": self.engine.rules["schema_version"],
            "profile_id": profile_id,
            "enabled_rule_ids": [
                rule["id"]
                for rule in profile["transformations"]
                if rule.get("enabled", True)
            ],
        }

    def transform(self, payload: object) -> tuple[int, dict[str, Any]]:
        if not isinstance(payload, dict) or set(payload) != {"text", "profile_id"}:
            return 422, {"error": "unsupported_transform_request"}
        if not isinstance(payload["text"], str) or not isinstance(
            payload["profile_id"], str
        ):
            return 422, {"error": "unsupported_transform_request"}
        if not self.engine.nonce_checker.enabled:
            return 503, {"error": "nonce_gate_unavailable"}
        try:
            result = self.engine.transform(payload["text"], payload["profile_id"])
        except ListenerLensError:
            return 422, {"error": "transform_rejected"}
        output = result.to_dict()
        output["deploy_contract"] = self.deploy_contract(payload["profile_id"])
        return 200, output

    def inspect_audio(self, payload: object) -> tuple[int, dict[str, Any]]:
        required = {"expected_script", "provider_transcript", "audio_base64"}
        if not isinstance(payload, dict) or set(payload) != required:
            return 422, {"error": "unsupported_audio_inspection_request"}
        if not all(isinstance(payload[key], str) for key in required):
            return 422, {"error": "unsupported_audio_inspection_request"}
        if (
            len(payload["expected_script"]) > MAX_SCRIPT_CHARS
            or len(payload["provider_transcript"]) > MAX_PROVIDER_TRANSCRIPT_CHARS
            or len(payload["audio_base64"]) > ((MAX_AUDIO_BYTES + 2) // 3) * 4
        ):
            return 422, {"error": "audio_inspection_fields_out_of_bounds"}
        try:
            audio = base64.b64decode(payload["audio_base64"], validate=True)
        except (ValueError, binascii.Error):
            return 422, {"error": "invalid_audio_base64"}
        if not audio or len(audio) > MAX_AUDIO_BYTES:
            return 422, {"error": "audio_size_out_of_bounds"}

        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
                handle.write(audio)
                temporary_path = Path(handle.name)
            timing = analyze_audio_timing(temporary_path, intended_syllables=None)
            prosody = analyze_prosody_fingerprint(temporary_path)
        except (OSError, ValueError, EOFError, wave.Error):
            return 422, {"error": "invalid_wav"}
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

        transcript = check_transcript(
            payload["expected_script"], payload["provider_transcript"]
        )
        reasons: list[str] = []
        if not transcript.exact_token_match:
            reasons.append("provider_transcript_mismatch")
        if timing.sample_rate_hz != REQUIRED_SAMPLE_RATE_HZ:
            reasons.append("unexpected_sample_rate")
        if not MIN_AUDIO_DURATION_S <= timing.duration_s <= MAX_AUDIO_DURATION_S:
            reasons.append("duration_out_of_bounds")
        if timing.utterance_duration_s <= 0:
            reasons.append("no_detectable_utterance")
        if timing.clipped_fraction > MAX_CLIPPED_FRACTION:
            reasons.append("excessive_clipping")

        return 200, {
            "accepted": not reasons,
            "reasons": reasons,
            "transcript": asdict(transcript),
            "timing": {
                **asdict(timing),
                "interior_pauses": [asdict(pause) for pause in timing.interior_pauses],
            },
            "prosody": asdict(prosody),
            "audio_sha256": hashlib.sha256(audio).hexdigest(),
            "inspection_contract": {
                "service_version": SERVICE_VERSION,
                "required_sample_rate_hz": REQUIRED_SAMPLE_RATE_HZ,
                "duration_bounds_s": [MIN_AUDIO_DURATION_S, MAX_AUDIO_DURATION_S],
                "max_clipped_fraction": MAX_CLIPPED_FRACTION,
            },
        }

    def _candidate_runtime(self) -> Any:
        if self._kokoro_candidate is None:
            with self._kokoro_candidate_lock:
                if self._kokoro_candidate is None:
                    from .kokoro_candidate_service import KokoroCandidateRuntime

                    self._kokoro_candidate = KokoroCandidateRuntime.load()
        return self._kokoro_candidate

    def kokoro_listener_lens(self, payload: object) -> tuple[int, dict[str, Any]]:
        if self.kokoro_candidate_enabled and self.bilingual_candidate_enabled:
            return 503, {
                "status": "unavailable",
                "error": "candidate_configuration_invalid",
                "api_calls_made": 0,
            }
        if not self.kokoro_candidate_enabled:
            return 503, {
                "status": "unavailable",
                "error": "kokoro_candidate_disabled",
                "api_calls_made": 0,
            }
        if not isinstance(payload, dict) or set(payload) != {
            "text",
            "profile_id",
            "voice_id",
        }:
            return 422, {"error": "unsupported_kokoro_request"}
        if (
            not isinstance(payload["text"], str)
            or not isinstance(payload["voice_id"], str)
            or payload["profile_id"] != KOKORO_PROFILE_ID
        ):
            return 422, {"error": "unsupported_kokoro_request"}
        try:
            voice = self.product_voices.resolve("en-US", payload["voice_id"])
        except ProductVoiceError:
            return 422, {"error": "unsupported_kokoro_request"}
        if voice.voice_id != KOKORO_CURRENT_EVIDENCE_VOICE_ID:
            return 409, {
                "status": "unavailable",
                "error": "voice_evidence_unavailable",
                "voice_id": voice.voice_id,
                "api_calls_made": 0,
            }
        try:
            return 200, self._candidate_runtime().render(
                payload["text"], voice_id=voice.voice_id
            )
        except ListenerLensError:
            return 422, {"error": "transform_rejected"}
        except Exception as exc:
            from .kokoro_candidate_service import KokoroCandidateGateError
            from .kokoro_typed_engine import KokoroTypedEngineError

            if isinstance(exc, KokoroTypedEngineError):
                return 422, {
                    "status": "unavailable",
                    "error": exc.code,
                    "api_calls_made": 0,
                }
            if isinstance(exc, KokoroCandidateGateError):
                return 422, {
                    "status": "unavailable",
                    "error": "automatic_gate_rejected",
                    "api_calls_made": 0,
                }
            raise

    def _bilingual_runtime(self, profile_id: str, voice_id: str) -> Any:
        if self._bilingual_candidate is not None:
            return self._bilingual_candidate
        key = (profile_id, voice_id)
        if key not in self._bilingual_candidate_runtimes:
            with self._bilingual_candidate_lock:
                if key not in self._bilingual_candidate_runtimes:
                    from .bilingual_composed_candidate_runtime_v3 import (
                        BilingualComposedCandidateRuntimeV3,
                    )

                    self._bilingual_candidate_runtimes[key] = (
                        BilingualComposedCandidateRuntimeV3.load(profile_id, voice_id)
                    )
        return self._bilingual_candidate_runtimes[key]

    def bilingual_kokoro_listener_lens(
        self, payload: object
    ) -> tuple[int, dict[str, Any]]:
        if self.kokoro_candidate_enabled and self.bilingual_candidate_enabled:
            return 503, {
                "status": "unavailable",
                "error": "candidate_configuration_invalid",
                "api_calls_made": 0,
            }
        if not self.bilingual_candidate_enabled:
            return 503, {
                "status": "unavailable",
                "error": "bilingual_kokoro_candidate_disabled",
                "api_calls_made": 0,
            }
        if not isinstance(payload, dict) or set(payload) != {
            "text",
            "profile_id",
            "voice_id",
        }:
            return 422, {"error": "unsupported_bilingual_kokoro_request"}
        if not all(
            isinstance(payload[key], str) for key in ("text", "profile_id", "voice_id")
        ):
            return 422, {"error": "unsupported_bilingual_kokoro_request"}
        source_language = BILINGUAL_PROFILE_SOURCE_LANGUAGES.get(payload["profile_id"])
        if source_language is None:
            return 422, {"error": "unsupported_bilingual_kokoro_request"}
        try:
            voice = self.product_voices.resolve(source_language, payload["voice_id"])
        except ProductVoiceError:
            return 422, {"error": "unsupported_bilingual_kokoro_request"}
        try:
            result = self._bilingual_runtime(
                payload["profile_id"], voice.voice_id
            ).render(payload["text"])
        except Exception as exc:
            from .bilingual_candidate_runtime import (
                BilingualCandidateAcousticGateError,
                BilingualCandidateRuntimeError,
                BilingualCandidateScopeError,
            )
            from .bilingual_vowel_engine import BilingualVowelEngineError

            if isinstance(exc, BilingualCandidateScopeError):
                return 409, {
                    "status": "unavailable",
                    "error": exc.code,
                    "api_calls_made": 0,
                }
            if isinstance(exc, BilingualCandidateAcousticGateError):
                return 422, {
                    "status": "unavailable",
                    "error": "runtime_acoustic_gate_rejected",
                    "api_calls_made": 0,
                }
            if isinstance(exc, (BilingualCandidateRuntimeError, BilingualVowelEngineError)):
                return 422, {
                    "status": "unavailable",
                    "error": "bilingual_candidate_rejected",
                    "api_calls_made": 0,
                }
            raise
        return 200, result

    def azure_lens(self, payload: object) -> tuple[int, dict[str, Any]]:
        """Deterministic Azure SSML neutral/lens pair for typed text.

        The Azure key never reaches the browser: rendering happens here and
        audio returns base64-encoded. Identical requests are served from the
        in-process cache (deterministic upstream renders make that sound),
        so repeated demo plays cost zero additional calls.
        """

        import base64
        import hashlib as _hashlib

        from .azure_lens_builder import (
            AzureLensBuilderError,
            build_pair,
            load_local_env,
            render_ssml_bytes,
            supported_profile_ids,
        )

        if not self.azure_lens_enabled:
            return 503, {
                "status": "unavailable",
                "error": "azure_lens_disabled",
                "api_calls_made": 0,
            }
        if (
            not isinstance(payload, dict)
            or set(payload) != {"text", "profile_id"}
            or not all(isinstance(payload[key], str) for key in ("text", "profile_id"))
        ):
            return 422, {"error": "unsupported_azure_lens_request"}
        text = payload["text"].strip()
        if not text or len(text) > 200:
            return 422, {"error": "unsupported_azure_lens_request"}
        if payload["profile_id"] not in supported_profile_ids():
            return 422, {"error": "unsupported_azure_lens_request"}
        environment = {**load_local_env(), **os.environ}
        key = environment.get("AZURE_SPEECH_KEY", "")
        region = environment.get("AZURE_SPEECH_REGION", "")
        if not key or not region:
            return 503, {
                "status": "unavailable",
                "error": "azure_lens_key_missing",
                "api_calls_made": 0,
            }
        try:
            pair = build_pair(text, payload["profile_id"])
        except AzureLensBuilderError as exc:
            return 422, {
                "status": "unavailable",
                "error": "azure_lens_rejected",
                "detail": str(exc),
                "api_calls_made": 0,
            }
        cache_key = _hashlib.sha256(
            (pair["ssml_neutral"] + "|" + pair["ssml_lens"] + "|"
             + pair["ssml_speaker"]).encode("utf-8")
        ).hexdigest()
        with self._azure_pair_lock:
            cached = self._azure_pair_cache.get(cache_key)
        if cached is not None:
            return 200, {**cached, "cache_hit": True, "api_calls_made": 0}
        audio: dict[str, Any] = {}
        for index, side in enumerate(("neutral", "lens", "speaker")):
            rendered = render_ssml_bytes(
                pair[f"ssml_{side}"], key=key, region=region
            )
            if not rendered["rendered"]:
                return 502, {
                    "status": "unavailable",
                    "error": "azure_render_failed",
                    "side": side,
                    "upstream_status": rendered["http_status"],
                    "api_calls_made": index,
                }
            wav_bytes = rendered["wav_bytes"]
            audio[side] = {
                "wav_base64": base64.b64encode(wav_bytes).decode("ascii"),
                "wav_sha256": _hashlib.sha256(wav_bytes).hexdigest(),
                "byte_count": len(wav_bytes),
            }
        result = {
            "schema_version": 1,
            "status": "ready_azure_lane",
            "lane_version": pair["lane_version"],
            "profile_id": pair["profile_id"],
            "locale": pair["locale"],
            "voice": pair["voice"],
            "listener_locale": pair["listener_locale"],
            "speaker_voice": pair["speaker_voice"],
            "normalized_text": pair["normalized_text"],
            "words": [
                {
                    "written": row["written"],
                    "source_phone": row["source_phone"],
                    "lens_phone": row["lens_phone"],
                    "applied_rule_ids": row["applied_rule_ids"],
                }
                for row in pair["words"]
            ],
            "applied_rule_ids": pair["applied_rule_ids"],
            "map_neutralized_rule_ids": pair["map_neutralized_rule_ids"],
            "context_absent_rule_ids": pair["context_absent_rule_ids"],
            "renderer_inaudible_rule_ids": pair["renderer_inaudible_rule_ids"],
            "omitted_rule_ids": pair["omitted_rule_ids"],
            "prosody": pair["prosody"],
            "affected_word_count": pair["affected_word_count"],
            "audio": audio,
            "api_calls_made": 3,
            "cache_hit": False,
        }
        with self._azure_pair_lock:
            self._azure_pair_cache[cache_key] = result
        return 200, result

    def gibberish(self, payload: object) -> tuple[int, dict[str, Any]]:
        """Deterministic source-only SSML pair for one language's nonsense.

        Sound Minus Meaning deliberately has no listener-language dimension:
        side A is the typed sentence and side B rebuilds its word/syllable
        skeleton from the source language's frozen syllable bank. Listener
        recategorization remains the separate Listener Lens product mode.
        """

        import base64
        import hashlib as _hashlib

        from .azure_lens_builder import (
            load_local_env,
            render_ssml_bytes,
        )
        from .gibberish_generator import (
            GibberishError,
            build_gibberish,
            supported_locales,
        )

        if not self.gibberish_enabled:
            return 503, {
                "status": "unavailable",
                "error": "gibberish_disabled",
                "api_calls_made": 0,
            }
        if (
            not isinstance(payload, dict)
            or set(payload) != {"text", "source_locale"}
            or not all(isinstance(value, str) for value in payload.values())
        ):
            return 422, {"error": "unsupported_gibberish_request"}
        text = payload["text"].strip()
        source_locale = payload["source_locale"]
        if not text or len(text) > 200:
            return 422, {"error": "unsupported_gibberish_request"}
        if source_locale not in supported_locales():
            return 422, {"error": "unsupported_gibberish_request"}
        environment = {**load_local_env(), **os.environ}
        key = environment.get("AZURE_SPEECH_KEY", "")
        region = environment.get("AZURE_SPEECH_REGION", "")
        if not key or not region:
            return 503, {
                "status": "unavailable",
                "error": "gibberish_key_missing",
                "api_calls_made": 0,
            }
        try:
            pair = build_gibberish(text, source_locale)
        except GibberishError as exc:
            return 422, {
                "status": "unavailable",
                "error": "gibberish_rejected",
                "detail": str(exc),
                "api_calls_made": 0,
            }
        ssml_a, ssml_b = pair["ssml_neutral"], pair["ssml_gibberish"]
        cache_key = _hashlib.sha256(
            (ssml_a + "|" + ssml_b).encode("utf-8")
        ).hexdigest()
        with self._gibberish_lock:
            cached = self._gibberish_cache.get(cache_key)
        if cached is not None:
            return 200, {**cached, "cache_hit": True, "api_calls_made": 0}
        audio: dict[str, Any] = {}
        # The side names stay "neutral" and "gibberish" in both shapes: A is
        # always the untransformed reference and B is always what the mode is
        # about. What each one *contains* moves when a listener is chosen, and
        # that is the page's job to say, not the contract's.
        for side, ssml in (("neutral", ssml_a), ("gibberish", ssml_b)):
            rendered = render_ssml_bytes(ssml, key=key, region=region)
            if not rendered["rendered"]:
                return 502, {
                    "status": "unavailable",
                    "error": "gibberish_render_failed",
                    "side": side,
                    "upstream_status": rendered["http_status"],
                    "api_calls_made": 1 if side == "gibberish" else 0,
                }
            wav_bytes = rendered["wav_bytes"]
            audio[side] = {
                "wav_base64": base64.b64encode(wav_bytes).decode("ascii"),
                "wav_sha256": _hashlib.sha256(wav_bytes).hexdigest(),
                "byte_count": len(wav_bytes),
            }
        result = {
            "schema_version": 1,
            "status": "ready_gibberish_lane",
            "lane_version": pair["lane_version"],
            "locale": pair["locale"],
            "voice": pair["voice"],
            "listener_locale": None,
            "profile_id": None,
            "normalized_text": pair["normalized_text"],
            "words": [
                {
                    "written": row["written"],
                    "gibberish_phone": row["gibberish_phone"],
                    "heard_phone": None,
                    "syllable_count": row["syllable_count"],
                }
                for row in pair["words"]
            ],
            "core_size": pair["core_size"],
            "syllable_shape": pair["syllable_shape"],
            "vowel_reduction": pair["vowel_reduction"],
            "audio": audio,
            "api_calls_made": 2,
            "cache_hit": False,
        }
        with self._gibberish_lock:
            self._gibberish_cache[cache_key] = result
        return 200, result


class DeployHTTPServer(ThreadingHTTPServer):
    """Threaded health endpoint with bounded admission to inference work."""

    daemon_threads = True
    request_queue_size = HTTP_REQUEST_QUEUE_SIZE

    def __init__(
        self,
        server_address: tuple[str, int],
        service: DeployService,
        *,
        max_concurrent_inference: int = MAX_CONCURRENT_INFERENCE_REQUESTS,
        max_admitted_inference: int = MAX_ADMITTED_INFERENCE_REQUESTS,
        max_handler_threads: int = MAX_HTTP_HANDLER_THREADS,
        inference_queue_wait_s: float = INFERENCE_QUEUE_WAIT_S,
        socket_read_timeout_s: float = SOCKET_READ_TIMEOUT_S,
    ) -> None:
        if (
            max_concurrent_inference < 1
            or max_admitted_inference < max_concurrent_inference
            or max_handler_threads <= max_admitted_inference
            or inference_queue_wait_s <= 0
            or socket_read_timeout_s <= 0
        ):
            raise ValueError("invalid_deploy_server_limits")
        self.service = service
        self.inference_slots = threading.BoundedSemaphore(max_concurrent_inference)
        self.admission_slots = threading.BoundedSemaphore(max_admitted_inference)
        self.handler_slots = threading.BoundedSemaphore(max_handler_threads)
        self.inference_queue_wait_s = inference_queue_wait_s
        self.socket_read_timeout_s = socket_read_timeout_s
        super().__init__(server_address, DeployRequestHandler)

    def process_request(self, request: socket.socket, client_address: Any) -> None:
        if not self.handler_slots.acquire(blocking=False):
            body = b'{"error":"http_handler_admission_full"}'
            response = (
                b"HTTP/1.1 503 Service Unavailable\r\n"
                b"content-type: application/json; charset=utf-8\r\n"
                b"cache-control: no-store\r\n"
                b"x-content-type-options: nosniff\r\n"
                + f"retry-after: {OVERLOAD_RETRY_AFTER_S}\r\n".encode()
                + f"content-length: {len(body)}\r\n".encode()
                + b"connection: close\r\n\r\n"
                + body
            )
            try:
                request.sendall(response)
            except OSError:
                pass
            finally:
                self.shutdown_request(request)
            print(
                json.dumps(
                    {
                        "event": "transform_service_http",
                        "route": "other",
                        "status": 503,
                    },
                    separators=(",", ":"),
                )
            )
            return
        try:
            super().process_request(request, client_address)
        except Exception:
            self.handler_slots.release()
            raise

    def process_request_thread(
        self, request: socket.socket, client_address: Any
    ) -> None:
        try:
            super().process_request_thread(request, client_address)
        finally:
            self.handler_slots.release()


class DeployRequestHandler(BaseHTTPRequestHandler):
    server: DeployHTTPServer

    def setup(self) -> None:
        self.request.settimeout(self.server.socket_read_timeout_s)
        super().setup()

    def _write_json(
        self,
        status: int,
        payload: dict[str, Any],
        headers: dict[str, str] | None = None,
    ) -> None:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
        self._response_status = status
        self.send_response(status)
        self.send_header("content-type", "application/json; charset=utf-8")
        self.send_header("content-length", str(len(body)))
        self.send_header("cache-control", "no-store")
        self.send_header("x-content-type-options", "nosniff")
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _route(self) -> str:
        return urlsplit(getattr(self, "path", "")).path

    def _content_length(self) -> tuple[int | None, tuple[int, dict[str, Any]] | None]:
        if self.headers.get_all("transfer-encoding"):
            return None, (400, {"error": "unsupported_transfer_encoding"})
        values = self.headers.get_all("content-length") or []
        if len(values) != 1 or not re.fullmatch(r"[1-9][0-9]*", values[0]):
            return None, (400, {"error": "invalid_content_length"})
        length = int(values[0])
        if length > MAX_REQUEST_BYTES:
            return None, (413, {"error": "request_size_out_of_bounds"})
        return length, None

    def _read_payload(
        self, length: int
    ) -> tuple[object | None, tuple[int, dict[str, Any]] | None]:
        try:
            self.connection.settimeout(self.server.socket_read_timeout_s)
            encoded = self.rfile.read(length)
        except (TimeoutError, socket.timeout):
            return None, (408, {"error": "request_read_timeout"})
        finally:
            self.connection.settimeout(None)
        if len(encoded) != length:
            return None, (400, {"error": "request_body_truncated"})
        try:
            return json.loads(encoded.decode("utf-8")), None
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None, (400, {"error": "invalid_json"})

    def _log_service_failure(self, route: str, exc: Exception) -> None:
        print(
            json.dumps(
                {
                    "event": "transform_service_failure",
                    "route": route,
                    "error_name": type(exc).__name__,
                },
                separators=(",", ":"),
            )
        )

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler contract
        route = self._route()
        if route not in {"/health", "/capabilities"}:
            self._write_json(404, {"error": "not_found"})
            return
        try:
            payload = (
                self.server.service.health()
                if route == "/health"
                else self.server.service.bilingual_capabilities()
            )
        except Exception as exc:  # pragma: no cover - defensive process boundary
            self._log_service_failure(route, exc)
            self._write_json(500, {"error": "internal_service_error"})
            return
        self._write_json(200, payload)

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler contract
        route = self._route()
        if route not in {
            "/transform",
            "/inspect-audio",
            "/kokoro-listener-lens",
            "/bilingual-kokoro-listener-lens",
            "/azure-lens",
            "/gibberish",
        }:
            self._write_json(404, {"error": "not_found"})
            return
        content_type = self.headers.get("content-type", "")
        if not _JSON_CONTENT_TYPE.fullmatch(content_type.strip()):
            self._write_json(415, {"error": "content_type_required"})
            return
        length, header_error = self._content_length()
        if header_error is not None:
            self._write_json(*header_error)
            return
        assert length is not None

        if not self.server.admission_slots.acquire(blocking=False):
            self._write_json(
                503,
                {"error": "inference_admission_full"},
                {"retry-after": str(OVERLOAD_RETRY_AFTER_S)},
            )
            return
        try:
            payload, body_error = self._read_payload(length)
            if body_error is not None:
                self._write_json(*body_error)
                return
            if not self.server.inference_slots.acquire(
                timeout=self.server.inference_queue_wait_s
            ):
                self._write_json(
                    503,
                    {"error": "inference_queue_timeout"},
                    {"retry-after": str(OVERLOAD_RETRY_AFTER_S)},
                )
                return
            try:
                if route == "/transform":
                    status, result = self.server.service.transform(payload)
                elif route == "/kokoro-listener-lens":
                    status, result = self.server.service.kokoro_listener_lens(payload)
                elif route == "/bilingual-kokoro-listener-lens":
                    status, result = (
                        self.server.service.bilingual_kokoro_listener_lens(payload)
                    )
                elif route == "/azure-lens":
                    status, result = self.server.service.azure_lens(payload)
                elif route == "/gibberish":
                    status, result = self.server.service.gibberish(payload)
                else:
                    status, result = self.server.service.inspect_audio(payload)
            except Exception as exc:  # pragma: no cover - defensive process boundary
                self._log_service_failure(route, exc)
                status, result = 500, {"error": "internal_service_error"}
            finally:
                self.server.inference_slots.release()
            self._write_json(status, result)
        finally:
            self.server.admission_slots.release()

    def log_message(self, format: str, *args: object) -> None:
        route = self._route()
        print(
            json.dumps(
                {
                    "event": "transform_service_http",
                    "method": self.command,
                    "route": route
                    if route
                    in {
                        "/health",
                        "/capabilities",
                        "/transform",
                        "/inspect-audio",
                        "/kokoro-listener-lens",
                        "/bilingual-kokoro-listener-lens",
                        "/azure-lens",
                        "/gibberish",
                    }
                    else "other",
                    "status": getattr(self, "_response_status", None),
                },
                separators=(",", ":"),
            )
        )


def run() -> None:
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8080"))
    server = DeployHTTPServer((host, port), DeployService())
    server.serve_forever()


if __name__ == "__main__":
    run()
