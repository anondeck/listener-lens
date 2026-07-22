from __future__ import annotations

import base64
import wave
from pathlib import Path
from types import SimpleNamespace

from earshift_bakeoff.sentence_pair_v2 import build_manifest
from earshift_bakeoff.sentence_pair_v2_run import render_slot


def _wav_base64(tmp_path: Path) -> str:
    path = tmp_path / "fixture.wav"
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(24000)
        wav.writeframes(b"\0\0" * 24000)
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _completion(data: str | None, transcript: str) -> SimpleNamespace:
    audio = None if data is None else SimpleNamespace(data=data, transcript=transcript)
    usage = SimpleNamespace(
        prompt_tokens=100,
        completion_tokens=20,
        prompt_tokens_details=SimpleNamespace(audio_tokens=0, cached_tokens=0),
        completion_tokens_details=SimpleNamespace(audio_tokens=10, reasoning_tokens=0),
    )
    return SimpleNamespace(
        _request_id="req_test",
        model="gpt-audio-1.5",
        usage=usage,
        choices=[SimpleNamespace(message=SimpleNamespace(audio=audio))],
    )


class _StatusError(RuntimeError):
    def __init__(self, status_code: int) -> None:
        super().__init__(f"status {status_code}")
        self.status_code = status_code


class _Client:
    def __init__(self, outcomes: list[object]) -> None:
        self.outcomes = outcomes
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **_: object) -> object:
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def test_external_no_audio_failure_gets_one_same_slot_retry(tmp_path: Path) -> None:
    slot = build_manifest()[0]
    client = _Client([_StatusError(429), _completion(_wav_base64(tmp_path), slot.script)])

    record = render_slot(
        client=client,
        slot=slot,
        audio_path=tmp_path / "out.wav",
        sleep=lambda _: None,
    )

    assert record["status"] == "audio_returned"
    assert record["attempt_count"] == 2
    assert record["attempts"][0]["retryable_external_failure"] is True
    assert record["attempts"][1]["audio_returned"] is True


def test_successful_audio_is_final_even_when_transcript_fails(tmp_path: Path) -> None:
    slot = build_manifest()[0]
    client = _Client(
        [
            _completion(_wav_base64(tmp_path), "wrong transcript"),
            AssertionError("must not retry returned audio"),
        ]
    )

    record = render_slot(
        client=client,
        slot=slot,
        audio_path=tmp_path / "out.wav",
        sleep=lambda _: None,
    )

    assert record["status"] == "audio_returned"
    assert record["attempt_count"] == 1
    assert record["transcript_check"]["exact_token_match"] is False
    assert len(client.outcomes) == 1


def test_non_external_no_audio_response_is_not_retried(tmp_path: Path) -> None:
    slot = build_manifest()[0]
    client = _Client([_completion(None, ""), AssertionError("must not retry")])

    record = render_slot(
        client=client,
        slot=slot,
        audio_path=tmp_path / "out.wav",
        sleep=lambda _: None,
    )

    assert record["status"] == "failed_no_audio"
    assert record["attempt_count"] == 1
    assert len(client.outcomes) == 1
