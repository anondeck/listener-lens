from __future__ import annotations

import base64
import contextlib
import http.client
import io
import json
import math
import socket
import struct
import threading
import time
import wave
from collections.abc import Iterator

from earshift_bakeoff.deploy_service import DeployHTTPServer, DeployService
from earshift_bakeoff.listener_lens import ListenerLensEngine


class RecordingAnalyzer:
    IPA = {"the": "ðə", "cat": "kæt", "is": "ɪz", "good": "ɡʊd"}

    def phonemize_words(self, words, voice: str) -> list[str]:
        assert voice == "en-us"
        return [self.IPA[word.casefold()] for word in words]


class RecordingNonceChecker:
    @property
    def enabled(self) -> bool:
        return True

    def accepts(self, surface: str, language: str, previous_surface: str | None):
        return True, f"/{surface}/"


def service() -> DeployService:
    return DeployService(
        ListenerLensEngine(
            analyzer=RecordingAnalyzer(), nonce_checker=RecordingNonceChecker()
        )
    )


class StubService:
    def __init__(self) -> None:
        self.transform_calls = 0
        self.inspect_calls = 0
        self.kokoro_calls = 0
        self.bilingual_kokoro_calls = 0
        self.gibberish_calls = 0

    def health(self):
        return {"status": "ok", "service_version": "stub"}

    def bilingual_capabilities(self):
        return {
            "matrix_version": "bilingual-product-matrix-v1",
            "production_enabled": False,
        }

    def transform(self, payload):
        self.transform_calls += 1
        return 200, {"accepted": True, "payload": payload}

    def inspect_audio(self, payload):
        self.inspect_calls += 1
        return 200, {"accepted": True, "payload": payload}

    def kokoro_listener_lens(self, payload):
        self.kokoro_calls += 1
        return 200, {"status": "ready", "payload": payload}

    def bilingual_kokoro_listener_lens(self, payload):
        self.bilingual_kokoro_calls += 1
        return 200, {"status": "ready", "payload": payload}

    def gibberish(self, payload):
        self.gibberish_calls += 1
        return 200, {"status": "ready_gibberish_lane", "payload": payload}


class RecordingKokoroCandidate:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def render(self, text: str, *, voice_id: str):
        self.calls.append((text, voice_id))
        return {"status": "ready", "api_calls_made": 0}


class RecordingBilingualCandidate:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def render(self, text: str):
        self.calls.append(text)
        return {"status": "ready_pending_human_qc", "api_calls_made": 0}


class BlockingStubService(StubService):
    def __init__(self) -> None:
        super().__init__()
        self.entered = threading.Event()
        self.release = threading.Event()

    def transform(self, payload):
        self.transform_calls += 1
        self.entered.set()
        assert self.release.wait(timeout=2)
        return 200, {"accepted": True, "payload": payload}


@contextlib.contextmanager
def running_server(
    service_instance: StubService | None = None, **limits: object
) -> Iterator[tuple[DeployHTTPServer, StubService]]:
    stub = service_instance or StubService()
    server = DeployHTTPServer(("127.0.0.1", 0), stub, **limits)  # type: ignore[arg-type]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server, stub
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def request_json(
    server: DeployHTTPServer,
    method: str,
    path: str,
    payload: object | None = None,
    *,
    headers: dict[str, str] | None = None,
) -> tuple[int, dict[str, object]]:
    status, parsed, _ = request_json_with_headers(
        server, method, path, payload, headers=headers
    )
    return status, parsed


def request_json_with_headers(
    server: DeployHTTPServer,
    method: str,
    path: str,
    payload: object | None = None,
    *,
    headers: dict[str, str] | None = None,
) -> tuple[int, dict[str, object], dict[str, str]]:
    connection = http.client.HTTPConnection(*server.server_address, timeout=2)
    body = None if payload is None else json.dumps(payload)
    request_headers = dict(headers or {})
    if body is not None:
        request_headers.setdefault("content-type", "application/json")
    connection.request(method, path, body=body, headers=request_headers)
    response = connection.getresponse()
    parsed = json.loads(response.read())
    response_headers = {key.casefold(): value for key, value in response.getheaders()}
    connection.close()
    return response.status, parsed, response_headers


def wav_base64(*, sample_rate: int = 24_000, duration_s: float = 0.5) -> str:
    stream = io.BytesIO()
    with wave.open(stream, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        frames = []
        for index in range(round(sample_rate * duration_s)):
            value = round(7_000 * math.sin(2 * math.pi * 220 * index / sample_rate))
            frames.append(struct.pack("<h", value))
        output.writeframes(b"".join(frames))
    return base64.b64encode(stream.getvalue()).decode()


def test_transform_reuses_the_listener_lens_engine() -> None:
    status, payload = service().transform(
        {"text": "The cat is good.", "profile_id": "en-to-pt-BR-vowel-lens"}
    )
    assert status == 200
    assert payload["comparison_available"] is True
    assert payload["nonce_gate_enabled"] is True
    assert payload["deploy_contract"] == {
        "service_version": "typed-transform-service-v1",
        "transform_algorithm_version": 5,
        "rules_sha256": payload["deploy_contract"]["rules_sha256"],
        "gate_database_sha256": None,
        "schema_version": 4,
        "profile_id": "en-to-pt-BR-vowel-lens",
        "enabled_rule_ids": ["ptbr.vowel.ae_to_eh"],
    }


def test_bilingual_capabilities_bind_all_four_voices_without_promotion() -> None:
    instance = service()
    health = instance.health()
    catalog = instance.bilingual_capabilities()

    assert health["bilingual_matrix_version"] == "bilingual-product-matrix-v1"
    assert health["bilingual_structural_classification"] == (
        "all_structural_slots_pass"
    )
    assert health["bilingual_audio_validation_status"] == "pending"
    assert health["bilingual_audio_integrity_classification"] == (
        "all_cells_universal_integrity_pass_family_acoustics_pending"
    )
    assert health["bilingual_isolated_audio_classification"] == (
        "all_isolated_slots_universal_integrity_pass_family_acoustics_pending"
    )
    assert catalog["rule_cell_count"] == 166
    assert catalog["changed_rule_cell_count"] == 98
    assert catalog["product_enabled_cell_count"] == 0
    assert catalog["structural_planner_slot_count"] == 280
    assert catalog["structural_planner_gate_yield"] == 1.0
    assert catalog["audio_validation_status"] == "pending"
    assert catalog["audio_integrity_gate_yield"] == 1.0
    assert catalog["audio_integrity_slot_count"] == 98
    assert catalog["isolated_audio_integrity_gate_yield"] == 1.0
    assert catalog["isolated_audio_slot_count"] == 280
    assert {
        voice["voice_id"]
        for direction in catalog["directions"]
        for voice in direction["voices"]
    } == {"af_heart", "am_michael", "pm_alex", "pf_dora"}


def test_internal_capabilities_route_is_read_only() -> None:
    with running_server() as (server, _):
        status, payload = request_json(server, "GET", "/capabilities")
        assert status == 200
        assert payload == {
            "matrix_version": "bilingual-product-matrix-v1",
            "production_enabled": False,
        }


def test_transform_rejects_extra_fields() -> None:
    status, payload = service().transform(
        {"text": "The cat.", "profile_id": "en-to-pt-BR-vowel-lens", "prompt": "x"}
    )
    assert status == 422
    assert payload["error"] == "unsupported_transform_request"


def test_kokoro_candidate_is_lazy_and_fail_closed_behind_its_flag() -> None:
    candidate = RecordingKokoroCandidate()
    disabled = DeployService(
        ListenerLensEngine(
            analyzer=RecordingAnalyzer(), nonce_checker=RecordingNonceChecker()
        ),
        kokoro_candidate=candidate,
        kokoro_candidate_enabled=False,
    )
    status, payload = disabled.kokoro_listener_lens(
        {
            "text": "Quiet voices map distant roads.",
            "profile_id": "en-to-pt-BR-vowel-lens",
            "voice_id": "af_heart",
        }
    )
    assert status == 503
    assert payload == {
        "status": "unavailable",
        "error": "kokoro_candidate_disabled",
        "api_calls_made": 0,
    }
    assert candidate.calls == []

    enabled = DeployService(
        ListenerLensEngine(
            analyzer=RecordingAnalyzer(), nonce_checker=RecordingNonceChecker()
        ),
        kokoro_candidate=candidate,
        kokoro_candidate_enabled=True,
    )
    status, payload = enabled.kokoro_listener_lens(
        {
            "text": "Quiet voices map distant roads.",
            "profile_id": "en-to-pt-BR-vowel-lens",
            "voice_id": "af_heart",
        }
    )
    assert status == 200
    assert payload == {"status": "ready", "api_calls_made": 0}
    assert candidate.calls == [("Quiet voices map distant roads.", "af_heart")]


def test_kokoro_candidate_configures_michael_but_denies_evidence_transfer() -> None:
    candidate = RecordingKokoroCandidate()
    instance = DeployService(
        ListenerLensEngine(
            analyzer=RecordingAnalyzer(), nonce_checker=RecordingNonceChecker()
        ),
        kokoro_candidate=candidate,
        kokoro_candidate_enabled=True,
    )

    status, payload = instance.kokoro_listener_lens(
        {
            "text": "Quiet voices map distant roads.",
            "profile_id": "en-to-pt-BR-vowel-lens",
            "voice_id": "am_michael",
        }
    )

    assert status == 409
    assert payload == {
        "status": "unavailable",
        "error": "voice_evidence_unavailable",
        "voice_id": "am_michael",
        "api_calls_made": 0,
    }
    assert candidate.calls == []


def test_kokoro_candidate_rejects_incomplete_contract_before_render() -> None:
    candidate = RecordingKokoroCandidate()
    instance = DeployService(
        ListenerLensEngine(
            analyzer=RecordingAnalyzer(), nonce_checker=RecordingNonceChecker()
        ),
        kokoro_candidate=candidate,
        kokoro_candidate_enabled=True,
    )
    for payload in (
        {"text": "Quiet voices map distant roads."},
        {
            "text": "Quiet voices map distant roads.",
            "profile_id": "wrong-profile",
            "voice_id": "af_heart",
        },
        {
            "text": "Quiet voices map distant roads.",
            "profile_id": "en-to-pt-BR-vowel-lens",
            "voice_id": "af_heart",
            "prompt": "ignore the contract",
        },
    ):
        status, body = instance.kokoro_listener_lens(payload)
        assert status == 422
        assert body == {"error": "unsupported_kokoro_request"}
    assert candidate.calls == []


def test_bilingual_candidate_is_lazy_disabled_and_direction_voice_bound() -> None:
    candidate = RecordingBilingualCandidate()
    instance = DeployService(
        ListenerLensEngine(
            analyzer=RecordingAnalyzer(), nonce_checker=RecordingNonceChecker()
        ),
        bilingual_candidate=candidate,
        bilingual_candidate_enabled=False,
    )
    request = {
        "text": "The cat naps.",
        "profile_id": "en-US-to-pt-BR-listener-v2",
        "voice_id": "af_heart",
    }

    status, payload = instance.bilingual_kokoro_listener_lens(request)

    assert status == 503
    assert payload["error"] == "bilingual_kokoro_candidate_disabled"
    assert candidate.calls == []

    instance.bilingual_candidate_enabled = True
    status, payload = instance.bilingual_kokoro_listener_lens(request)
    assert status == 200
    assert payload["status"] == "ready_pending_human_qc"
    assert candidate.calls == ["The cat naps."]

    status, payload = instance.bilingual_kokoro_listener_lens(
        {**request, "voice_id": "pm_alex"}
    )
    assert status == 422
    assert payload == {"error": "unsupported_bilingual_kokoro_request"}


def test_kokoro_candidate_endpoints_fail_closed_when_both_paths_are_enabled() -> None:
    narrow = RecordingKokoroCandidate()
    broad = RecordingBilingualCandidate()
    instance = DeployService(
        ListenerLensEngine(
            analyzer=RecordingAnalyzer(), nonce_checker=RecordingNonceChecker()
        ),
        kokoro_candidate=narrow,
        kokoro_candidate_enabled=True,
        bilingual_candidate=broad,
        bilingual_candidate_enabled=True,
    )

    narrow_status, narrow_payload = instance.kokoro_listener_lens(
        {
            "text": "The cat naps.",
            "profile_id": "en-to-pt-BR-vowel-lens",
            "voice_id": "af_heart",
        }
    )
    broad_status, broad_payload = instance.bilingual_kokoro_listener_lens(
        {
            "text": "The cat naps.",
            "profile_id": "en-US-to-pt-BR-listener-v2",
            "voice_id": "af_heart",
        }
    )

    assert narrow_status == broad_status == 503
    assert narrow_payload["error"] == "candidate_configuration_invalid"
    assert broad_payload["error"] == "candidate_configuration_invalid"
    assert narrow.calls == []
    assert broad.calls == []


def test_bilingual_candidate_http_route_dispatches_exact_payload() -> None:
    with running_server() as (server, stub):
        payload = {
            "text": "O povo corre.",
            "profile_id": "pt-BR-to-en-US-listener-v2",
            "voice_id": "pm_alex",
        }
        status, result = request_json(
            server, "POST", "/bilingual-kokoro-listener-lens", payload
        )

    assert status == 200
    assert result == {"status": "ready", "payload": payload}
    assert stub.bilingual_kokoro_calls == 1


def test_audio_inspection_requires_exact_transcript_and_pcm_contract() -> None:
    status, payload = service().inspect_audio(
        {
            "expected_script": "bavd behvd",
            "provider_transcript": "bavd behvd",
            "audio_base64": wav_base64(),
        }
    )
    assert status == 200
    assert payload["accepted"] is True
    assert payload["reasons"] == []
    assert payload["timing"]["sample_rate_hz"] == 24_000
    assert payload["prosody"]["version"] == "prosody-fingerprint-v1"
    assert payload["prosody"]["bin_count"] == 32
    assert len(payload["prosody"]["energy_contour_db"]) == 32
    assert len(payload["prosody"]["pitch_contour_semitones"]) == 32
    assert 190 <= payload["prosody"]["median_f0_hz"] <= 250

    _, mismatch = service().inspect_audio(
        {
            "expected_script": "bavd behvd",
            "provider_transcript": "I can help with that",
            "audio_base64": wav_base64(),
        }
    )
    assert mismatch["accepted"] is False
    assert "provider_transcript_mismatch" in mismatch["reasons"]


def test_audio_inspection_rejects_wrong_sample_rate() -> None:
    status, payload = service().inspect_audio(
        {
            "expected_script": "bavd",
            "provider_transcript": "bavd",
            "audio_base64": wav_base64(sample_rate=16_000),
        }
    )
    assert status == 200
    assert payload["accepted"] is False
    assert "unexpected_sample_rate" in payload["reasons"]


def test_service_errors_do_not_expose_parser_or_validation_details() -> None:
    status, payload = service().transform(
        {"text": "", "profile_id": "en-to-pt-BR-vowel-lens"}
    )
    assert status == 422
    assert payload == {"error": "transform_rejected"}

    status, payload = service().inspect_audio(
        {
            "expected_script": "bavd",
            "provider_transcript": "bavd",
            "audio_base64": base64.b64encode(b"not a wav").decode(),
        }
    )
    assert status == 422
    assert payload == {"error": "invalid_wav"}


def test_http_boundary_requires_unambiguous_length_type_and_transfer_rules() -> None:
    with running_server() as (server, stub):
        connection = http.client.HTTPConnection(*server.server_address, timeout=2)
        connection.putrequest("POST", "/transform")
        connection.putheader("content-type", "application/json")
        connection.putheader("transfer-encoding", "chunked")
        connection.endheaders()
        connection.send(b"0\r\n\r\n")
        response = connection.getresponse()
        assert response.status == 400
        assert json.loads(response.read()) == {"error": "unsupported_transfer_encoding"}
        connection.close()

        connection = http.client.HTTPConnection(*server.server_address, timeout=2)
        connection.putrequest("POST", "/transform")
        connection.putheader("content-type", "application/json")
        connection.putheader("content-length", "2")
        connection.putheader("content-length", "2")
        connection.endheaders(b"{}")
        response = connection.getresponse()
        assert response.status == 400
        assert json.loads(response.read()) == {"error": "invalid_content_length"}
        connection.close()

        status, payload = request_json(
            server,
            "POST",
            "/transform",
            {"text": "safe"},
            headers={"content-type": "text/plain"},
        )
        assert status == 415
        assert payload == {"error": "content_type_required"}
        assert stub.transform_calls == 0


def test_http_boundary_times_out_a_partial_body() -> None:
    with running_server(socket_read_timeout_s=0.05) as (server, stub):
        with socket.create_connection(server.server_address, timeout=2) as client:
            client.settimeout(2)
            client.sendall(
                b"POST /transform HTTP/1.0\r\n"
                b"content-type: application/json\r\n"
                b"content-length: 50\r\n\r\n{}"
            )
            response = b""
            while chunk := client.recv(4096):
                response += chunk
        assert b" 408 " in response.split(b"\r\n", 1)[0]
        assert json.loads(response.split(b"\r\n\r\n", 1)[1]) == {
            "error": "request_read_timeout"
        }
        assert stub.transform_calls == 0


def test_health_bypasses_bounded_inference_queue() -> None:
    with running_server(
        max_concurrent_inference=1,
        max_admitted_inference=2,
        inference_queue_wait_s=0.05,
    ) as (server, stub):
        assert server.inference_slots.acquire(blocking=False)
        started = time.monotonic()
        status, payload = request_json(server, "GET", "/health")
        assert time.monotonic() - started < 0.5
        assert status == 200
        assert payload["status"] == "ok"

        status, payload, headers = request_json_with_headers(
            server, "POST", "/transform", {"text": "safe"}
        )
        assert status == 503
        assert payload == {"error": "inference_queue_timeout"}
        assert headers["retry-after"] == "2"
        assert stub.transform_calls == 0
        server.inference_slots.release()


def test_admission_limit_rejects_before_inference() -> None:
    with running_server(
        max_concurrent_inference=1,
        max_admitted_inference=1,
    ) as (server, stub):
        assert server.admission_slots.acquire(blocking=False)
        status, payload, headers = request_json_with_headers(
            server, "POST", "/transform", {"text": "safe"}
        )
        assert status == 503
        assert payload == {"error": "inference_admission_full"}
        assert headers["retry-after"] == "2"
        assert stub.transform_calls == 0
        server.admission_slots.release()


def test_handler_threads_are_bounded_with_health_capacity_above_inference() -> None:
    stub = BlockingStubService()
    with running_server(
        stub,
        max_concurrent_inference=1,
        max_admitted_inference=1,
        max_handler_threads=2,
    ) as (server, _):
        result: list[tuple[int, dict[str, object]]] = []
        request_thread = threading.Thread(
            target=lambda: result.append(
                request_json(server, "POST", "/transform", {"text": "safe"})
            )
        )
        request_thread.start()
        assert stub.entered.wait(timeout=1)
        try:
            status, payload = request_json(server, "GET", "/health")
            assert status == 200
            assert payload["status"] == "ok"
        finally:
            stub.release.set()
            request_thread.join(timeout=2)
        assert result[0][0] == 200


def test_handler_thread_cap_returns_retryable_overload_without_spawning() -> None:
    with running_server(
        max_concurrent_inference=1,
        max_admitted_inference=1,
        max_handler_threads=2,
    ) as (server, _):
        assert server.handler_slots.acquire(blocking=False)
        assert server.handler_slots.acquire(blocking=False)
        try:
            status, payload, headers = request_json_with_headers(
                server, "GET", "/health"
            )
            assert status == 503
            assert payload == {"error": "http_handler_admission_full"}
            assert headers["retry-after"] == "2"
        finally:
            server.handler_slots.release()
            server.handler_slots.release()


def test_http_logs_never_include_query_text(capsys) -> None:
    secret = "PRIVATE_TYPED_SOURCE_TEXT"
    with running_server() as (server, _):
        status, _ = request_json(server, "GET", f"/health?text={secret}")
        assert status == 200
    output = capsys.readouterr().out
    assert "transform_service_http" in output
    assert secret not in output


def test_azure_lens_disabled_fails_closed() -> None:
    deploy = service()
    assert deploy.azure_lens_enabled is False
    status, result = deploy.azure_lens({"text": "The cat naps", "profile_id": "x"})
    assert status == 503
    assert result["error"] == "azure_lens_disabled"
    assert result["api_calls_made"] == 0


def test_azure_lens_validates_payload_and_serves_cache(monkeypatch) -> None:
    deploy = service()
    deploy.azure_lens_enabled = True

    status, result = deploy.azure_lens({"text": "hi"})
    assert (status, result["error"]) == (422, "unsupported_azure_lens_request")
    status, result = deploy.azure_lens(
        {"text": "hi", "profile_id": "unknown-profile"}
    )
    assert (status, result["error"]) == (422, "unsupported_azure_lens_request")
    status, result = deploy.azure_lens(
        {"text": "x" * 201, "profile_id": "en-US-to-pt-BR-listener-v2"}
    )
    assert (status, result["error"]) == (422, "unsupported_azure_lens_request")

    from earshift_bakeoff import azure_lens_builder as lane

    monkeypatch.setenv("AZURE_SPEECH_KEY", "test-key")
    monkeypatch.setenv("AZURE_SPEECH_REGION", "test-region")
    pair = {
        "lane_version": "azure-lens-lane-v1",
        "profile_id": "en-US-to-pt-BR-listener-v2",
        "locale": "en-US",
        "voice": "en-US-AvaNeural",
        "normalized_text": "the cat naps",
        "listener_locale": "pt-BR",
        "speaker_voice": "pt-BR-FranciscaNeural",
        "ssml_neutral": "<speak>n</speak>",
        "ssml_lens": "<speak>l</speak>",
        "ssml_speaker": "<speak>s</speak>",
        "words": [
            {
                "written": "cat",
                "source_phone": "kæt",
                "lens_phone": "kɛt",
                "applied_rule_ids": ["enpt.ae_eh"],
            }
        ],
        "applied_rule_ids": ["enpt.ae_eh"],
        "map_neutralized_rule_ids": [],
        "context_absent_rule_ids": ["enpt.lexical_stress_initial_bias"],
        "renderer_inaudible_rule_ids": [],
        "omitted_rule_ids": [],
        "prosody": {"polar_question": False, "contour_applied": False},
        "affected_word_count": 1,
        "api_calls_made": 0,
    }
    renders: list[str] = []

    def fake_build_pair(text: str, profile_id: str, **_: object) -> dict:
        return dict(pair)

    def fake_render(ssml: str, *, key: str, region: str) -> dict:
        assert (key, region) == ("test-key", "test-region")
        renders.append(ssml)
        return {"http_status": 200, "rendered": True, "wav_bytes": b"RIFFdata"}

    monkeypatch.setattr(lane, "build_pair", fake_build_pair)
    monkeypatch.setattr(lane, "render_ssml_bytes", fake_render)

    request = {"text": "The cat naps", "profile_id": "en-US-to-pt-BR-listener-v2"}
    status, result = deploy.azure_lens(request)
    assert status == 200
    assert result["status"] == "ready_azure_lane"
    assert result["api_calls_made"] == 3
    assert result["cache_hit"] is False
    assert len(renders) == 3
    assert set(result["audio"]) == {"neutral", "lens", "speaker"}
    decoded = base64.b64decode(result["audio"]["neutral"]["wav_base64"])
    assert decoded == b"RIFFdata"
    assert result["words"] == [
        {
            "written": "cat",
            "source_phone": "kæt",
            "lens_phone": "kɛt",
            "applied_rule_ids": ["enpt.ae_eh"],
        }
    ]

    status, cached = deploy.azure_lens(request)
    assert status == 200
    assert cached["cache_hit"] is True
    assert cached["api_calls_made"] == 0
    assert len(renders) == 3, "a cache hit must not re-render"


def test_azure_lens_requires_key(monkeypatch) -> None:
    deploy = service()
    deploy.azure_lens_enabled = True
    monkeypatch.delenv("AZURE_SPEECH_KEY", raising=False)
    monkeypatch.delenv("AZURE_SPEECH_REGION", raising=False)
    from earshift_bakeoff import azure_lens_builder as lane

    monkeypatch.setattr(lane, "load_local_env", lambda: {})
    status, result = deploy.azure_lens(
        {"text": "The cat naps", "profile_id": "en-US-to-pt-BR-listener-v2"}
    )
    assert status == 503
    assert result["error"] == "azure_lens_key_missing"


def test_gibberish_disabled_fails_closed() -> None:
    deploy = service()
    assert deploy.gibberish_enabled is False
    status, result = deploy.gibberish({"text": "The cat naps", "source_locale": "en-US"})
    assert status == 503
    assert result["error"] == "gibberish_disabled"
    assert result["api_calls_made"] == 0


def test_gibberish_gates_independently_of_the_lens(monkeypatch) -> None:
    """Two modes, two claims, two switches. Turning the lens on must not turn
    this on with it, or pulling one would silently pull the other."""

    deploy = service()
    deploy.azure_lens_enabled = True
    status, result = deploy.gibberish({"text": "The cat naps", "source_locale": "en-US"})
    assert status == 503
    assert result["error"] == "gibberish_disabled"


def test_gibberish_validates_payload_and_serves_cache(monkeypatch) -> None:
    deploy = service()
    deploy.gibberish_enabled = True

    for payload in (
        {"text": "hi"},
        {"source_locale": "en-US"},
        {"text": "hi", "source_locale": "xx-XX"},
        {"text": "   ", "source_locale": "en-US"},
        {"text": "x" * 201, "source_locale": "en-US"},
        {"text": "hi", "source_locale": "en-US", "core_size": 40},
        # A listener has to be a direction the lens itself serves. Its own
        # source is not one of them, and neither is a locale that does not
        # exist — both are refused before any Azure call.
        {"text": "hi", "source_locale": "en-US", "listener_locale": "xx-XX"},
        {"text": "hi", "source_locale": "en-US", "listener_locale": "en-US"},
    ):
        status, result = deploy.gibberish(payload)
        assert (status, result["error"]) == (422, "unsupported_gibberish_request"), payload

    from earshift_bakeoff import azure_lens_builder as lane
    from earshift_bakeoff import gibberish_generator as gib

    monkeypatch.setenv("AZURE_SPEECH_KEY", "test-key")
    monkeypatch.setenv("AZURE_SPEECH_REGION", "test-region")
    pair = {
        "lane_version": "gibberish-lane-v1",
        "locale": "en-US",
        "voice": "en-US-AvaNeural",
        "normalized_text": "the cat naps",
        "ssml_neutral": "<speak>n</speak>",
        "ssml_gibberish": "<speak>g</speak>",
        "words": [
            {
                "word_index": 0,
                "written": "cat",
                "source_phone": "kæt",
                "gibberish_phone": "noʊ",
                "syllable_count": 1,
                "stressed_syllable": 0,
            }
        ],
        "core_size": 90,
        "syllable_shape": "prefer_open",
        "vowel_reduction": True,
        "stress_mark_emitted": False,
        "api_calls_made": 0,
    }
    renders: list[str] = []

    def fake_render(ssml: str, *, key: str, region: str) -> dict:
        assert (key, region) == ("test-key", "test-region")
        renders.append(ssml)
        return {"http_status": 200, "rendered": True, "wav_bytes": b"RIFF" + ssml.encode()}

    monkeypatch.setattr(gib, "build_gibberish", lambda text, locale, **_: dict(pair))
    monkeypatch.setattr(lane, "render_ssml_bytes", fake_render)

    request = {"text": "The cat naps", "source_locale": "en-US"}
    status, result = deploy.gibberish(request)
    assert status == 200
    assert result["status"] == "ready_gibberish_lane"
    assert result["api_calls_made"] == 2
    assert result["cache_hit"] is False
    assert len(renders) == 2
    assert set(result["audio"]) == {"neutral", "gibberish"}
    # The two sides must be distinct audio; the Worker refuses the pair if not.
    assert (
        result["audio"]["neutral"]["wav_sha256"]
        != result["audio"]["gibberish"]["wav_sha256"]
    )
    assert result["words"] == [
        {
            "written": "cat",
            "gibberish_phone": "noʊ",
            "heard_phone": None,
            "syllable_count": 1,
        }
    ]
    # Nobody heard it, and the answer says so rather than leaving it implied.
    assert result["listener_locale"] is None
    assert result["profile_id"] is None

    status, cached = deploy.gibberish(request)
    assert status == 200
    assert cached["cache_hit"] is True
    assert cached["api_calls_made"] == 0
    assert len(renders) == 2, "a cache hit must not re-render"


def test_gibberish_rejects_listener_dimension_before_render(monkeypatch) -> None:
    deploy = service()
    deploy.gibberish_enabled = True
    status, result = deploy.gibberish(
        {
            "text": "The quick brown fox jumps over the lazy dog",
            "source_locale": "en-US",
            "listener_locale": "es-ES",
        }
    )
    assert status == 422
    assert result["error"] == "unsupported_gibberish_request"


def test_gibberish_listener_re_hears_the_same_nonsense(monkeypatch) -> None:
    """Two ears, one draw — the comparison is worthless if the nonsense moves."""

    from earshift_bakeoff import gibberish_generator as gib
    from earshift_bakeoff.azure_lens_builder import build_pair

    text = "The quick brown fox jumps over the lazy dog"
    first = gib.gibberish_analysis(text, "en-US")
    second = gib.gibberish_analysis(text, "en-US")
    drawn = [row["gibberish_phone"] for row in first["words"]]
    assert drawn == [row["gibberish_phone"] for row in second["words"]]

    spanish = build_pair(text, "en-US-to-es-ES-listener-v1", source_analysis=first["analysis"])
    swedish = build_pair(text, "en-US-to-sv-SE-listener-v1", source_analysis=first["analysis"])
    # Same input to both ears...
    assert [row["source_phone"] for row in spanish["words"]] == drawn
    assert [row["source_phone"] for row in swedish["words"]] == drawn
    # ...and the two ears disagree about it, which is the whole demonstration.
    assert [row["lens_phone"] for row in spanish["words"]] != [
        row["lens_phone"] for row in swedish["words"]
    ]


def test_gibberish_requires_key(monkeypatch) -> None:
    deploy = service()
    deploy.gibberish_enabled = True
    monkeypatch.delenv("AZURE_SPEECH_KEY", raising=False)
    monkeypatch.delenv("AZURE_SPEECH_REGION", raising=False)
    from earshift_bakeoff import azure_lens_builder as lane

    monkeypatch.setattr(lane, "load_local_env", lambda: {})
    status, result = deploy.gibberish({"text": "The cat naps", "source_locale": "en-US"})
    assert status == 503
    assert result["error"] == "gibberish_key_missing"


def test_gibberish_route_reaches_the_service(capsys) -> None:
    request = {"text": "The cat naps", "source_locale": "en-US"}
    with running_server() as (server, stub):
        status, result = request_json(server, "POST", "/gibberish", request)
        assert stub.gibberish_calls == 1
    # A 404 here would mean the path never reached the handler at all.
    assert status == 200
    assert result["payload"] == request
    # The route is named in the access log rather than folded into "other",
    # so a gibberish request is attributable after the fact.
    assert '"route":"/gibberish"' in capsys.readouterr().out
