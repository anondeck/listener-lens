from __future__ import annotations

import json
from pathlib import Path

import pytest

from earshift_bakeoff.corpus import CodexCorpusGenerator, CorpusError
from earshift_bakeoff.models import GenerationBatch, LanguageCandidates, ScriptCandidate, TokenSpec
from earshift_bakeoff.util import sha256_file


def candidate(candidate_id: str) -> ScriptCandidate:
    tokens = [
        TokenSpec(
            surface=f"navor{index}x",
            role="content" if index < 12 else "filler",
            intended_ipa=f"/navor{index}x/",
            syllables=2,
            primary_stress_index=0,
            rule_ids=["fixture"],
        )
        for index in range(18)
    ]
    return ScriptCandidate(
        candidate_id=candidate_id,
        profile_id="en-US-mae",
        tokens=tokens,
        punctuation_after_token={8: ",", 17: "."},
    )


def write_bundle(tmp_path: Path, rules_path: Path, ids: list[str]) -> Path:
    prompt = tmp_path / "PROMPT.md"
    prompt.write_text("frozen prompt\n", encoding="utf-8")
    corpus_path = tmp_path / "candidates.json"
    batch = GenerationBatch(
        languages=[
            LanguageCandidates(
                profile_id="en-US-mae",
                candidates=[candidate(candidate_id) for candidate_id in ids],
            )
        ]
    )
    payload = {
        "schema_version": 1,
        "provenance": {
            "source": "codex",
            "model_label": "gpt-5.6-codex",
            "created_at_utc": "2026-07-14T00:00:00Z",
            "prompt_sha256": sha256_file(prompt),
            "rules_sha256": sha256_file(rules_path),
            "notes": "fixture",
        },
        "rounds": [batch.model_dump(mode="json")],
    }
    corpus_path.write_text(json.dumps(payload), encoding="utf-8")
    return corpus_path


def test_codex_corpus_loads_with_bound_provenance(tmp_path, monkeypatch) -> None:
    import earshift_bakeoff.corpus as corpus_module

    rules = tmp_path / "rules.json"
    rules.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(corpus_module, "RULES_PATH", rules)
    path = write_bundle(tmp_path, rules, ["candidate-1"])

    generator = CodexCorpusGenerator(path)
    batch = generator.generate(["en-US-mae"], 20)

    assert generator.receipt()["candidates"] == 1
    assert batch.languages[0].candidates[0].candidate_id == "candidate-1"


def test_codex_corpus_rejects_duplicate_candidate_ids(tmp_path, monkeypatch) -> None:
    import earshift_bakeoff.corpus as corpus_module

    rules = tmp_path / "rules.json"
    rules.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(corpus_module, "RULES_PATH", rules)
    path = write_bundle(tmp_path, rules, ["duplicate", "duplicate"])

    with pytest.raises(CorpusError, match="Duplicate candidate id"):
        CodexCorpusGenerator(path)
