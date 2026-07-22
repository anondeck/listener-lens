from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from wordfreq import iter_wordlist

from earshift_bakeoff.config import load_config
from earshift_bakeoff.gates import (
    CandidateGate,
    EspeakPhonemizer,
    GateBuildError,
    _wordfreq_resource_path,
    canonical_token,
    domain_hash,
)
from earshift_bakeoff.models import ScriptCandidate, TokenSpec
from earshift_bakeoff.util import sha256_file


class FakePhonemizer:
    mapping = {"nite": "_n_ˈaɪ_t", "night": "_n_ˈaɪ_t"}

    def phonemize(self, tokens, voice):
        return [self.mapping.get(token, f"_{token}") for token in tokens]


def make_db(path: Path) -> None:
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE written_hash(sha256 BLOB PRIMARY KEY)")
        conn.execute(
            "CREATE TABLE phone_hash(lang TEXT, sha256 BLOB, PRIMARY KEY(lang, sha256))"
        )
        conn.execute("CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT)")
        for word in ("night", "hola", "cem", "foo"):
            conn.execute(
                "INSERT INTO written_hash(sha256) VALUES (?)", (domain_hash("text", word),)
            )
        conn.execute(
            "INSERT INTO phone_hash(lang, sha256) VALUES (?, ?)",
            ("en", domain_hash("phone", "en", "_n_ˈaɪ_t")),
        )


def candidate(first: str = "vemor", second: str = "dalek") -> ScriptCandidate:
    content = [first, second, "pelum", "sovan", "tramel", "gavor", "luden", "bravic", "noral", "kesum"]
    fillers = ["za", "mi", "lo", "ve", "za", "mi", "lo", "ve"]
    tokens = [
        TokenSpec(
            surface=surface,
            role="content" if index < 10 else "filler",
            intended_ipa=f"/{surface}/",
            syllables=2,
            primary_stress_index=0,
            rule_ids=["fixture"],
        )
        for index, surface in enumerate(content + fillers)
    ]
    return ScriptCandidate(
        candidate_id="fixture",
        profile_id="en-US-mae",
        tokens=tokens,
        punctuation_after_token={8: ",", 17: "."},
    )


def test_written_homophone_and_adjacent_fixtures(tmp_path: Path) -> None:
    database = tmp_path / "gate.sqlite3"
    make_db(database)
    gate = CandidateGate(database, phonemizer=FakePhonemizer())

    assert "predicted_homophone" in gate.gate(candidate("nite")).reasons
    assert "written_word_match" in gate.gate(candidate("hola")).reasons
    assert "adjacent_written_word_match" in gate.gate(candidate("fo", "o")).reasons
    assert gate.gate(candidate()).passed


def test_pinned_wordlist_checksums_and_counts() -> None:
    config = load_config()["word_gate"]
    for language in config["languages"]:
        assert sha256_file(_wordfreq_resource_path(language)) == config["resource_sha256"][language]
        canonical = {
            token
            for raw in iter_wordlist(language, wordlist="large")
            if (token := canonical_token(raw)) is not None
        }
        assert len(canonical) == config["expected_counts"][language]


def test_reference_wav_enables_audio_output(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("earshift_bakeoff.gates.shutil.which", lambda _: "/fake/espeak-ng")

    def fake_run(args, *, check):
        assert check is True
        assert "-q" not in args
        output = Path(args[args.index("-w") + 1])
        output.write_bytes(b"reference audio")

    monkeypatch.setattr("earshift_bakeoff.gates.subprocess.run", fake_run)
    output = tmp_path / "reference.wav"

    EspeakPhonemizer().reference_wav("vushvot", "en-us", output)

    assert output.read_bytes() == b"reference audio"


def test_reference_wav_rejects_silent_success(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("earshift_bakeoff.gates.shutil.which", lambda _: "/fake/espeak-ng")
    monkeypatch.setattr(
        "earshift_bakeoff.gates.subprocess.run", lambda args, *, check: None
    )

    with pytest.raises(GateBuildError, match="did not create reference audio"):
        EspeakPhonemizer().reference_wav("vushvot", "en-us", tmp_path / "missing.wav")
