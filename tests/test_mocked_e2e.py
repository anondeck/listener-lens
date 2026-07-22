from __future__ import annotations

import csv
import json
import math
import shutil
import struct
import subprocess
import wave
from pathlib import Path

import pytest

from earshift_bakeoff.models import (
    GenerationBatch,
    LanguageCandidates,
    RenderResult,
    ScriptCandidate,
    TokenSpec,
    VerificationResult,
)
from earshift_bakeoff.config import RULES_PATH
from earshift_bakeoff.corpus import CodexCorpusGenerator
from earshift_bakeoff.pipeline import Services, execute_run
from earshift_bakeoff.review import build_review
from earshift_bakeoff.util import sha256_file


def write_tone(path: Path, duration: float = 6.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rate = 16_000
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        frames = (
            struct.pack("<h", int(3000 * math.sin(2 * math.pi * 220 * i / rate)))
            for i in range(int(duration * rate))
        )
        handle.writeframes(b"".join(frames))


def make_candidate(profile: str, index: int) -> ScriptCandidate:
    stems = [f"vemor{index}{letter}" for letter in "abcdefghij"]
    fillers = ["zavi", "melo", "luno", "vesa", "zavi", "melo", "luno", "vesa"]
    tokens = [
        TokenSpec(
            surface=surface,
            role="content" if token_index < 10 else "filler",
            intended_ipa=f"/{surface}/",
            syllables=2,
            primary_stress_index=0,
            rule_ids=["fixture", "sensitive"] if token_index in {6, 7} else ["fixture"],
        )
        for token_index, surface in enumerate(stems + fillers)
    ]
    return ScriptCandidate(
        candidate_id=f"model-{profile}-{index}",
        profile_id=profile,
        tokens=tokens,
        punctuation_after_token={8: ",", 17: "."},
    )


class FakeGenerator:
    last_response_id = "resp_mock"
    last_resolved_model = "gpt-5.6-mock"

    def generate(self, profile_ids, count, refill_index=0):
        return GenerationBatch(
            languages=[
                LanguageCandidates(
                    profile_id=profile,
                    candidates=[make_candidate(profile, index) for index in range(10)],
                )
                for profile in profile_ids
            ]
        )


class FakeGate:
    class Result:
        passed = True
        reasons = []
        token_count = 18
        syllable_count = 36

    def gate(self, candidate):
        return self.Result()


class FakeRenderer:
    def __init__(self, slug):
        self.slug = slug
        self.model = slug

    def render(self, script, instruction, voice, output):
        write_tone(output)
        return RenderResult(
            renderer_slug=self.slug,
            renderer_model=self.model,
            status="ok",
            output_path=str(output),
            request_id=f"req_{self.slug}",
            resolved_model=self.model,
            latency_ms=5,
        )


class FakeVerifier:
    def verify(self, source, target_language, normalized):
        normalized.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, normalized)
        return VerificationResult(
            audio_ok=True,
            duration_s=6.0,
            sample_rate_hz=16000,
            clipped_fraction=0.0,
            top_language=target_language,
            target_score=0.9,
            runner_up_language="fr",
            margin=0.8,
            language_scores={target_language: 0.9, "fr": 0.1},
            transcript="vemor dalek",
            transcript_nonword_rate=1.0,
            no_speech_probability=0.01,
            avg_logprob=-0.2,
            compression_ratio=1.0,
            sister_language_split=False,
            machine_pass=True,
        )


class FakePhonemizer:
    def phonemize(self, tokens, voice):
        return [f"_{token}" for token in tokens]

    def reference_wav(self, token, voice, output):
        write_tone(output, duration=1.0)


class FakePaths:
    def __init__(self, root: Path):
        self.root = root

    def run_dir(self, run_id: str) -> Path:
        return self.root / run_id


def test_mocked_end_to_end_produces_sixty_rows(tmp_path, monkeypatch) -> None:
    import earshift_bakeoff.pipeline as pipeline

    monkeypatch.setattr(pipeline, "Paths", lambda: FakePaths(tmp_path))
    services = Services(
        generator=FakeGenerator(),
        renderers=[
            FakeRenderer("gpt-4o-mini-tts-2025-12-15"),
            FakeRenderer("gpt-audio-1.5"),
        ],
        gate=FakeGate(),
        verifier=FakeVerifier(),
        phonemizer=FakePhonemizer(),
        whisper_variant="large-v3-mock",
    )
    results = execute_run(
        "mock-run",
        services=services,
        require_live_prerequisites=False,
        voice="marin",
    )
    with results.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 60
    assert len({row["audio_filename"] for row in rows}) == 60
    assert sum(row["g2p_sampled"] == "True" for row in rows) == 30
    assert all(row["machine_pass"] == "True" for row in rows)


def test_codex_corpus_import_produces_paired_sixty_row_run(
    tmp_path, monkeypatch
) -> None:
    import earshift_bakeoff.pipeline as pipeline

    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    prompt = corpus_dir / "PROMPT.md"
    prompt.write_text("frozen Codex prompt\n", encoding="utf-8")
    corpus_path = corpus_dir / "candidates.json"
    payload = {
        "schema_version": 1,
        "provenance": {
            "source": "codex",
            "model_label": "gpt-5.6 via Codex",
            "created_at_utc": "2026-07-14T00:00:00Z",
            "prompt_sha256": sha256_file(prompt),
            "rules_sha256": sha256_file(RULES_PATH),
            "notes": "end-to-end fixture",
        },
        "rounds": [
            {
                "languages": [
                    {
                        "profile_id": profile,
                        "candidates": [
                            make_candidate(profile, index) for index in range(10)
                        ],
                    }
                    for profile in ("en-US-mae", "es-MX-cdmx", "pt-BR-sp")
                ]
            }
        ],
    }
    corpus_path.write_text(
        json.dumps(
            payload,
            default=lambda value: value.model_dump(mode="json"),
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(pipeline, "Paths", lambda: FakePaths(tmp_path))
    services = Services(
        generator=CodexCorpusGenerator(corpus_path),
        renderers=[FakeRenderer("renderer-a"), FakeRenderer("renderer-b")],
        gate=FakeGate(),
        verifier=FakeVerifier(),
        phonemizer=FakePhonemizer(),
        whisper_variant="large-v3-mock",
    )

    results = execute_run(
        "codex-corpus-run",
        services=services,
        require_live_prerequisites=False,
        voice="marin",
    )
    with results.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 60
    assert {row["generator_source"] for row in rows} == {"codex"}


def test_live_prerequisite_failure_does_not_create_run_directory(
    tmp_path, monkeypatch
) -> None:
    import earshift_bakeoff.pipeline as pipeline

    monkeypatch.setattr(pipeline, "Paths", lambda: FakePaths(tmp_path))
    monkeypatch.setattr(
        pipeline,
        "require_api_key",
        lambda: (_ for _ in ()).throw(RuntimeError("missing key")),
    )

    with pytest.raises(RuntimeError, match="missing key"):
        execute_run("must-not-exist")

    assert not (tmp_path / "must-not-exist").exists()


def test_generated_review_contains_cards_and_valid_javascript(
    tmp_path, monkeypatch
) -> None:
    import earshift_bakeoff.review as review_module

    class ReviewPaths:
        def run_dir(self, run_id: str) -> Path:
            return tmp_path / run_id

    run_dir = tmp_path / "review-run"
    run_dir.mkdir()
    rows = [
        {
            "render_status": "ok",
            "language": "en",
            "profile_id": "en-US-mae",
            "blind_id": "blind-1",
            "script_text": "navor pelum.",
            "audio_filename": "test.wav",
            "g2p_sampled": "False",
        }
    ]
    with (run_dir / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    monkeypatch.setattr(review_module, "Paths", lambda: ReviewPaths())

    review_path = build_review("review-run")
    rendered = review_path.read_text(encoding="utf-8")
    script = rendered.split("<script>", 1)[1].split("</script>", 1)[0]
    script_path = tmp_path / "review.js"
    script_path.write_text(script, encoding="utf-8")

    assert "const ROWS = [{" in rendered
    subprocess.run(["node", "--check", script_path], check=True, capture_output=True)
