from __future__ import annotations

import hashlib
import importlib.resources
import json
import os
import re
import shutil
import sqlite3
import subprocess
import tempfile
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

from wordfreq import iter_wordlist

from .config import Paths, load_config
from .models import ScriptCandidate
from .util import atomic_write_json, sha256_file


FOREIGN_SWITCH_RE = re.compile(r"\([^)]*\)")
PRODUCTIVE_ENDINGS = ("ing", "mente", "ção")


def canonical_token(value: str) -> str | None:
    value = unicodedata.normalize("NFC", value).casefold().strip()
    if not value:
        return None
    for ch in value:
        category = unicodedata.category(ch)[0]
        if category == "M":
            continue
        if category != "L" or "LATIN" not in unicodedata.name(ch, ""):
            return None
    return value


def canonical_ipa(value: str) -> str:
    return unicodedata.normalize("NFC", value).strip()


def domain_hash(domain: str, *parts: str) -> bytes:
    payload = domain + "\0" + "\0".join(parts)
    return hashlib.sha256(payload.encode("utf-8")).digest()


def chunks(items: Sequence[str], size: int) -> Iterator[Sequence[str]]:
    for index in range(0, len(items), size):
        yield items[index : index + size]


class GateBuildError(RuntimeError):
    pass


class EspeakPhonemizer:
    def __init__(self, binary: str = "espeak-ng") -> None:
        resolved = shutil.which(binary)
        if not resolved:
            raise GateBuildError(f"{binary} is not installed")
        self.binary = resolved

    def version(self) -> str:
        proc = subprocess.run(
            [self.binary, "--version"], capture_output=True, text=True, check=True
        )
        return (proc.stdout or proc.stderr).splitlines()[0].strip()

    def phonemize(self, tokens: Sequence[str], voice: str) -> list[str]:
        if not tokens:
            return []
        proc = subprocess.run(
            [self.binary, "-q", "-v", voice, "--ipa=3"],
            input="\n".join(tokens) + "\n",
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode:
            raise GateBuildError(
                f"eSpeak failed for {voice} with exit {proc.returncode}: {proc.stderr.strip()}"
            )
        lines = [canonical_ipa(line) for line in proc.stdout.splitlines() if line.strip()]
        if len(lines) != len(tokens):
            # A small number of upstream inventory entries can produce a blank line
            # only in large stdin batches. Split deterministically until alignment is
            # restored; a truly unpronounceable singleton becomes a switch-marker
            # sentinel and is excluded from the real-word phone index.
            if len(tokens) == 1:
                return ["(unpronounceable)"]
            midpoint = len(tokens) // 2
            return self.phonemize(tokens[:midpoint], voice) + self.phonemize(
                tokens[midpoint:], voice
            )
        return lines

    def reference_wav(self, token: str, voice: str, output: Path) -> None:
        output.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [self.binary, "-v", voice, "-w", str(output), token], check=True
        )
        if not output.is_file() or output.stat().st_size == 0:
            raise GateBuildError(f"eSpeak did not create reference audio: {output}")


def _wordfreq_resource_path(language: str) -> Path:
    resource = importlib.resources.files("wordfreq").joinpath(
        "data", f"large_{language}.msgpack.gz"
    )
    path = Path(str(resource))
    if not path.is_file():
        raise GateBuildError(f"wordfreq resource missing: {path}")
    return path


def _insert_metadata(conn: sqlite3.Connection, key: str, value: object) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)",
        (key, json.dumps(value, sort_keys=True)),
    )


def build_gate_database(
    destination: Path | None = None,
    *,
    config: dict | None = None,
    phonemizer: EspeakPhonemizer | None = None,
    chunk_size: int = 10_000,
) -> dict:
    config = config or load_config()
    gate_config = config["word_gate"]
    destination = destination or Paths().gate_db
    phonemizer = phonemizer or EspeakPhonemizer()
    destination.parent.mkdir(parents=True, exist_ok=True)

    source_hashes: dict[str, str] = {}
    canonical_counts: dict[str, int] = {}
    temp_fd, temp_name = tempfile.mkstemp(
        prefix=destination.name + ".", suffix=".partial", dir=destination.parent
    )
    os.close(temp_fd)
    temp_path = Path(temp_name)
    temp_path.unlink(missing_ok=True)

    try:
        conn = sqlite3.connect(temp_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("CREATE TABLE written_hash(sha256 BLOB PRIMARY KEY)")
        conn.execute(
            "CREATE TABLE phone_hash(lang TEXT NOT NULL, sha256 BLOB NOT NULL, "
            "PRIMARY KEY(lang, sha256))"
        )
        conn.execute("CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL)")

        for language in gate_config["languages"]:
            resource_path = _wordfreq_resource_path(language)
            source_hash = sha256_file(resource_path)
            expected_hash = gate_config["resource_sha256"][language]
            if source_hash != expected_hash:
                raise GateBuildError(
                    f"wordfreq {language} checksum mismatch: {source_hash} != {expected_hash}"
                )
            source_hashes[language] = source_hash

            words: list[str] = []
            for raw_word in iter_wordlist(language, wordlist="large"):
                word = canonical_token(raw_word)
                if word is not None:
                    words.append(word)
            # The upstream list can contain canonically equivalent spellings.
            words = list(dict.fromkeys(words))
            canonical_counts[language] = len(words)
            expected_count = gate_config["expected_counts"][language]
            if len(words) != expected_count:
                raise GateBuildError(
                    f"wordfreq {language} canonical count {len(words)} != {expected_count}"
                )

            with conn:
                conn.executemany(
                    "INSERT OR IGNORE INTO written_hash(sha256) VALUES (?)",
                    ((domain_hash("text", word),) for word in words),
                )

            voice = gate_config["voices"][language]
            for word_chunk in chunks(words, chunk_size):
                ipa_lines = phonemizer.phonemize(word_chunk, voice)
                rows = []
                for ipa in ipa_lines:
                    if FOREIGN_SWITCH_RE.search(ipa):
                        continue
                    rows.append((language, domain_hash("phone", language, ipa)))
                with conn:
                    conn.executemany(
                        "INSERT OR IGNORE INTO phone_hash(lang, sha256) VALUES (?, ?)", rows
                    )

        union_count = conn.execute("SELECT COUNT(*) FROM written_hash").fetchone()[0]
        if union_count != gate_config["expected_counts"]["union"]:
            raise GateBuildError(
                f"written union count {union_count} != {gate_config['expected_counts']['union']}"
            )
        phone_counts = {
            language: conn.execute(
                "SELECT COUNT(*) FROM phone_hash WHERE lang = ?", (language,)
            ).fetchone()[0]
            for language in gate_config["languages"]
        }
        metadata = {
            "schema_version": 1,
            "wordfreq_version": gate_config["package_version"],
            "source_hashes": source_hashes,
            "canonical_counts": canonical_counts,
            "written_union_count": union_count,
            "phone_counts": phone_counts,
            "espeak_version": phonemizer.version(),
            "voices": gate_config["voices"],
        }
        with conn:
            for key, value in metadata.items():
                _insert_metadata(conn, key, value)
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        conn.close()
        os.replace(temp_path, destination)
        metadata["database_sha256"] = sha256_file(destination)
        atomic_write_json(destination.with_suffix(".receipt.json"), metadata)
        return metadata
    except Exception:
        temp_path.unlink(missing_ok=True)
        Path(str(temp_path) + "-wal").unlink(missing_ok=True)
        Path(str(temp_path) + "-shm").unlink(missing_ok=True)
        raise


@dataclass
class GateResult:
    passed: bool
    reasons: list[str]
    token_count: int
    syllable_count: int


class CandidateGate:
    def __init__(
        self,
        database: Path | None = None,
        *,
        voices: dict[str, str] | None = None,
        phonemizer: EspeakPhonemizer | None = None,
    ) -> None:
        self.database = database or Paths().gate_db
        if not self.database.is_file():
            raise GateBuildError(f"Gate database is missing: {self.database}")
        config = load_config()
        self.voices = voices or config["word_gate"]["voices"]
        self.phonemizer = phonemizer or EspeakPhonemizer()

    def _contains(self, query: str, params: tuple) -> bool:
        with sqlite3.connect(self.database) as conn:
            return conn.execute(query, params).fetchone() is not None

    def text_match(self, token: str) -> bool:
        canonical = canonical_token(token)
        if canonical is None:
            return True
        return self._contains(
            "SELECT 1 FROM written_hash WHERE sha256 = ?", (domain_hash("text", canonical),)
        )

    def phone_match(self, language: str, ipa: str) -> bool:
        return self._contains(
            "SELECT 1 FROM phone_hash WHERE lang = ? AND sha256 = ?",
            (language, domain_hash("phone", language, canonical_ipa(ipa))),
        )

    def gate(self, candidate: ScriptCandidate) -> GateResult:
        reasons: list[str] = []
        tokens = [canonical_token(token.surface) for token in candidate.tokens]
        if any(token is None for token in tokens):
            reasons.append("invalid_surface_token")
            return GateResult(False, reasons, len(candidate.tokens), candidate.syllable_count)
        clean_tokens = [token for token in tokens if token is not None]

        if not 18 <= len(clean_tokens) <= 24:
            reasons.append("token_count")
        if not 30 <= candidate.syllable_count <= 42:
            reasons.append("syllable_count")
        internal_punctuation = [
            (index, value)
            for index, value in candidate.punctuation_after_token.items()
            if index < len(clean_tokens) - 1
        ]
        if not any(value == "," for _, value in internal_punctuation):
            reasons.append("missing_internal_phrase_boundary")

        content_count = sum(token.role == "content" for token in candidate.tokens)
        content_ratio = content_count / len(candidate.tokens)
        if not 0.55 <= content_ratio <= 0.70:
            reasons.append("content_ratio")
        filler_surfaces = {
            canonical_token(token.surface)
            for token in candidate.tokens
            if token.role == "filler"
        }
        filler_surfaces.discard(None)
        if not 3 <= len(filler_surfaces) <= 5:
            reasons.append("filler_type_count")

        if any(token.endswith(PRODUCTIVE_ENDINGS) for token in clean_tokens):
            reasons.append("productive_morphology")
        if any(self.text_match(token) for token in clean_tokens):
            reasons.append("written_word_match")
        if any(self.text_match(a + b) for a, b in zip(clean_tokens, clean_tokens[1:])):
            reasons.append("adjacent_written_word_match")

        language = candidate.language
        ipa_lines = self.phonemizer.phonemize(clean_tokens, self.voices[language])
        if any(FOREIGN_SWITCH_RE.search(ipa) for ipa in ipa_lines):
            reasons.append("foreign_language_switch")
        if any(self.phone_match(language, ipa) for ipa in ipa_lines):
            reasons.append("predicted_homophone")
        if any(
            self.phone_match(language, left + right)
            for left, right in zip(ipa_lines, ipa_lines[1:])
        ):
            reasons.append("adjacent_predicted_homophone")

        return GateResult(not reasons, reasons, len(clean_tokens), candidate.syllable_count)


def transcript_nonword_rate(text: str, database: Path | None = None) -> float | None:
    tokens: list[str] = []
    for raw in re.findall(r"[^\W\d_]+", text, flags=re.UNICODE):
        canonical = canonical_token(raw)
        if canonical:
            tokens.append(canonical)
    if not tokens:
        return None
    database = database or Paths().gate_db
    with sqlite3.connect(database) as conn:
        real_count = sum(
            conn.execute(
                "SELECT 1 FROM written_hash WHERE sha256 = ?",
                (domain_hash("text", token),),
            ).fetchone()
            is not None
            for token in tokens
        )
    return (len(tokens) - real_count) / len(tokens)
