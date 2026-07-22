from __future__ import annotations

import base64
import json
import os
import time
from pathlib import Path
from typing import Any, Protocol, Sequence

from openai import OpenAI

from .audio_conformance import ConformanceSample, build_messages, check_transcript
from .config import load_config, load_rules, stable_json
from .models import GenerationBatch, RenderResult
from .util import sha256_file


SMOKE_TEXT = "narevik soluma, drevin paloreth."


class ApiConfigurationError(RuntimeError):
    pass


def require_api_key() -> None:
    value = os.environ.get("OPENAI_API_KEY", "")
    if not value.strip():
        raise ApiConfigurationError(
            "OPENAI_API_KEY is not set. Phase A local setup is complete; "
            "a funded key is required for smoke and the timed run."
        )


class Generator(Protocol):
    def generate(self, profile_ids: Sequence[str], count: int, refill_index: int = 0) -> GenerationBatch: ...


class Renderer(Protocol):
    slug: str
    model: str

    def render(self, script: str, instruction: str, voice: str, output: Path) -> RenderResult: ...


class OpenAIGenerator:
    def __init__(self, client: OpenAI | None = None, model: str | None = None) -> None:
        config = load_config()
        self.client = client or OpenAI(max_retries=0)
        self.model = model or config["generator"]["model"]
        self.rules = load_rules()
        self.last_response_id: str | None = None
        self.last_resolved_model: str | None = None

    def _prompt(self, profile_ids: Sequence[str], count: int, refill_index: int) -> str:
        profile_set = set(profile_ids)
        profiles = [
            profile for profile in self.rules["profiles"] if profile["id"] in profile_set
        ]
        return (
            "Create pronounceable but semantically opaque scripts for the requested "
            "reference profiles. Return exactly the structured schema. Every surface "
            "token must be invented; do not use examples from the prompt, real words, "
            "names, abbreviations, numbers, or recognizable productive morphology. "
            "Use 18–24 tokens, 30–42 intended syllables, two declarative phrases, "
            "55–70% content-like tokens, and 3–5 recurring invented filler types. "
            "Supply intended IPA and stable rule IDs as generation targets, not as "
            "claims of acoustic verification. Include an internal comma and a final "
            "period via punctuation_after_token. Produce "
            f"{count} candidates for each profile. This is refill batch {refill_index}.\n\n"
            "RULE TABLE:\n"
            + json.dumps(
                {
                    "shared": self.rules["shared_engineering_constraints"],
                    "profiles": profiles,
                },
                ensure_ascii=False,
                indent=2,
            )
        )

    def generate(
        self, profile_ids: Sequence[str], count: int, refill_index: int = 0
    ) -> GenerationBatch:
        response = self.client.responses.parse(
            model=self.model,
            input=[
                {
                    "role": "developer",
                    "content": (
                        "You design controlled nonce-language stimuli for a phonetics "
                        "experiment. Follow the supplied rule table and never smuggle "
                        "semantic content into the surface script."
                    ),
                },
                {"role": "user", "content": self._prompt(profile_ids, count, refill_index)},
            ],
            text_format=GenerationBatch,
        )
        if response.output_parsed is None:
            raise ApiConfigurationError("GPT-5.6 returned no parsed generation batch")
        self.last_response_id = getattr(response, "id", None)
        self.last_resolved_model = getattr(response, "model", None)
        return response.output_parsed


class SpeechRenderer:
    slug = "gpt-4o-mini-tts-2025-12-15"
    model = "gpt-4o-mini-tts-2025-12-15"

    def __init__(self, client: OpenAI | None = None) -> None:
        self.client = client or OpenAI(max_retries=0)

    def render(self, script: str, instruction: str, voice: str, output: Path) -> RenderResult:
        output.parent.mkdir(parents=True, exist_ok=True)
        partial = output.with_suffix(output.suffix + ".partial")
        started = time.monotonic()
        with self.client.audio.speech.with_streaming_response.create(
            model=self.model,
            voice=voice,
            input=script,
            instructions=instruction,
            response_format="wav",
        ) as response:
            response.stream_to_file(partial)
            headers = dict(response.headers)
        partial.replace(output)
        return RenderResult(
            renderer_slug=self.slug,
            renderer_model=self.model,
            status="ok",
            output_path=str(output),
            request_id=headers.get("x-request-id"),
            resolved_model=self.model,
            latency_ms=round((time.monotonic() - started) * 1000),
            response_headers={
                key: value
                for key, value in headers.items()
                if key.lower().startswith("x-ratelimit") or key.lower() == "x-request-id"
            },
        )


class ChatAudioRenderer:
    slug = "gpt-audio-1.5"
    model = "gpt-audio-1.5"

    def __init__(self, client: OpenAI | None = None) -> None:
        self.client = client or OpenAI(max_retries=0)

    def render(self, script: str, instruction: str, voice: str, output: Path) -> RenderResult:
        output.parent.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        sample = ConformanceSample(
            sample_id="runtime",
            language="",
            script=script,
            delivery=instruction,
        )
        completion = self.client.chat.completions.create(
            model=self.model,
            modalities=["text", "audio"],
            audio={"voice": voice, "format": "wav"},
            messages=build_messages(sample, "json-flow-v2"),
            store=False,
        )
        message = completion.choices[0].message
        audio = message.audio
        if audio is None or not audio.data:
            raise ApiConfigurationError("gpt-audio-1.5 returned no audio payload")
        transcript = getattr(audio, "transcript", None) or ""
        transcript_check = check_transcript(script, transcript)
        if not transcript_check.exact_token_match:
            raise ApiConfigurationError(
                "gpt-audio-1.5 violated the verbatim transcript contract: "
                f"similarity={transcript_check.token_similarity}, "
                f"extra_tokens={transcript_check.extra_token_count}, "
                f"missing_tokens={transcript_check.missing_token_count}"
            )
        partial = output.with_suffix(output.suffix + ".partial")
        partial.write_bytes(base64.b64decode(audio.data))
        partial.replace(output)
        return RenderResult(
            renderer_slug=self.slug,
            renderer_model=self.model,
            status="ok",
            output_path=str(output),
            request_id=getattr(completion, "_request_id", None),
            resolved_model=getattr(completion, "model", self.model),
            provider_transcript=transcript,
            latency_ms=round((time.monotonic() - started) * 1000),
            response_headers=None,
        )


def renderer_instruction(profile_id: str) -> str:
    for profile in load_rules()["profiles"]:
        if profile["id"] == profile_id:
            return profile["renderer_instruction"]
    raise ApiConfigurationError(f"Unknown profile: {profile_id}")


def api_contract_fingerprint(voice: str) -> str:
    config = load_config()
    payload = {
        "generator": config["generator"],
        "renderers": config["renderers"],
        "voice": voice,
        "rules": load_rules(),
    }
    import hashlib

    return hashlib.sha256(stable_json(payload).encode("utf-8")).hexdigest()


def smoke_render(voice: str) -> dict[str, Any]:
    require_api_key()
    instruction = (
        "Read this invented phrase exactly as written in natural mainstream U.S. "
        "English. Do not spell, translate, correct, replace, explain, or add anything."
    )
    output_dir = Path("artifacts/smoke") / voice
    results = []
    for renderer in (SpeechRenderer(), ChatAudioRenderer()):
        output = output_dir / f"{renderer.slug}.wav"
        result = renderer.render(SMOKE_TEXT, instruction, voice, output)
        result_dict = result.to_dict()
        result_dict["audio_sha256"] = sha256_file(output)
        results.append(result_dict)
    return {"voice": voice, "text": SMOKE_TEXT, "results": results}
