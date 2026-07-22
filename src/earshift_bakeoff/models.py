from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


Language = Literal["en", "es", "pt"]


class TokenSpec(BaseModel):
    surface: str = Field(min_length=1, max_length=24)
    role: Literal["content", "filler"]
    intended_ipa: str = Field(min_length=1, max_length=64)
    syllables: int = Field(ge=1, le=5)
    primary_stress_index: int | None = Field(default=None, ge=0, le=4)
    rule_ids: list[str] = Field(default_factory=list, max_length=12)

    @field_validator("surface")
    @classmethod
    def surface_has_no_whitespace(cls, value: str) -> str:
        if any(ch.isspace() for ch in value):
            raise ValueError("token surfaces cannot contain whitespace")
        return value


class ScriptCandidate(BaseModel):
    candidate_id: str = Field(min_length=1, max_length=80)
    profile_id: Literal["en-US-mae", "es-MX-cdmx", "pt-BR-sp"]
    tokens: list[TokenSpec] = Field(min_length=18, max_length=24)
    punctuation_after_token: dict[int, Literal[",", "."]] = Field(default_factory=dict)

    @property
    def language(self) -> Language:
        return {"en-US-mae": "en", "es-MX-cdmx": "es", "pt-BR-sp": "pt"}[self.profile_id]  # type: ignore[return-value]

    @property
    def text(self) -> str:
        chunks: list[str] = []
        for index, token in enumerate(self.tokens):
            suffix = self.punctuation_after_token.get(index, "")
            chunks.append(token.surface + suffix)
        text = " ".join(chunks).strip()
        return text if text.endswith(".") else text + "."

    @property
    def syllable_count(self) -> int:
        return sum(token.syllables for token in self.tokens)


class LanguageCandidates(BaseModel):
    profile_id: Literal["en-US-mae", "es-MX-cdmx", "pt-BR-sp"]
    candidates: list[ScriptCandidate] = Field(min_length=1, max_length=80)


class GenerationBatch(BaseModel):
    languages: list[LanguageCandidates] = Field(min_length=1, max_length=3)


class CorpusProvenance(BaseModel):
    source: Literal["codex"]
    model_label: str = Field(min_length=1, max_length=80)
    created_at_utc: str = Field(min_length=1, max_length=64)
    prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    rules_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    notes: str = Field(default="", max_length=500)


class CorpusBundle(BaseModel):
    schema_version: Literal[1]
    provenance: CorpusProvenance
    rounds: list[GenerationBatch] = Field(min_length=1, max_length=3)


@dataclass
class RenderResult:
    renderer_slug: str
    renderer_model: str
    status: str
    output_path: str | None = None
    request_id: str | None = None
    resolved_model: str | None = None
    provider_transcript: str | None = None
    latency_ms: int | None = None
    retry_count: int = 0
    response_headers: dict[str, str] | None = None
    error_code: str | None = None
    error_detail: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class VerificationResult:
    audio_ok: bool
    duration_s: float | None
    sample_rate_hz: int | None
    clipped_fraction: float | None
    top_language: str | None
    target_score: float | None
    runner_up_language: str | None
    margin: float | None
    language_scores: dict[str, float]
    transcript: str | None
    transcript_nonword_rate: float | None
    no_speech_probability: float | None
    avg_logprob: float | None
    compression_ratio: float | None
    sister_language_split: bool
    machine_pass: bool
    error_detail: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
