from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import json
import os
import sqlite3
import tempfile
import threading
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from wordfreq import iter_wordlist

from .config import Paths, stable_json
from .gates import _wordfreq_resource_path, canonical_token, domain_hash
from .kokoro_specs import resolve_pinned_file
from .kokoro_synthesis import CONFIG_FILE, MODEL_REPO, MODEL_REVISION
from .util import atomic_write_json, sha256_file


RUN_ID = "20260717-pt-kokoro-homophone-index-v1"
SCHEMA_VERSION = 1
NORMALIZATION_VERSION = "ptbr-kokoro-isolated-word-v1"
LANGUAGE_ID = "pt-BR"
WORD_LANGUAGE = "pt"
KOKORO_LANG_CODE = "p"
ESPEAK_VOICE = "pt-br"
WORD_COUNT = 262_151
WORDFREQ_VERSION = "3.1.1"
WORDFREQ_RESOURCE_SHA256 = (
    "7d764586bca6262f554d5fa77ad8e6841ef42534776e70558065140853660ce2"
)
KOKORO_VERSION = "0.9.4"
MISAKI_VERSION = "0.9.4"
KOKORO_PIPELINE_SHA256 = (
    "09da32aab781f7a163cbf9f6e379d53db40957cbbab50bc26a21c8440c0eabd6"
)
MISAKI_ESPEAK_SHA256 = (
    "ebc39e807eeae7b99322109d8477aa16b605427f9cc852a4b1f6f72c45806a09"
)
SAMPLE_SIZE = 4_096

CHALLENGE_WORDS = {
    "pão": "pˈɐ̃ʊ̃",
    "carro": "kˈaxʊ",
    "caro": "kˈaɾʊ",
    "filho": "fˈiljʊ",
    "ninho": "nˈiɲʊ",
    "tia": "ʧˈiæ",
    "dia": "ʤˈiæ",
    "avó": "avˈɔ",
    "avô": "avˈo",
}


class PortugueseKokoroGateError(RuntimeError):
    pass


def _phone_hash(phone: str) -> bytes:
    return domain_hash("kokoro-phone-v2", LANGUAGE_ID, NORMALIZATION_VERSION, phone)


def _word_hash(word: str) -> bytes:
    return domain_hash("kokoro-word-v2", LANGUAGE_ID, NORMALIZATION_VERSION, word)


def normalize_portuguese_phone(value: str) -> str:
    return unicodedata.normalize("NFC", value).strip()


def portuguese_inventory() -> tuple[str, ...]:
    resource = _wordfreq_resource_path(WORD_LANGUAGE)
    if sha256_file(resource) != WORDFREQ_RESOURCE_SHA256:
        raise PortugueseKokoroGateError("Portuguese wordfreq resource hash mismatch")
    words = tuple(
        dict.fromkeys(
            word
            for raw in iter_wordlist(WORD_LANGUAGE, wordlist="large")
            if (word := canonical_token(raw)) is not None
        )
    )
    if len(words) != WORD_COUNT:
        raise PortugueseKokoroGateError(
            f"Portuguese inventory count {len(words)} != {WORD_COUNT}"
        )
    return words


def _model_vocab() -> frozenset[str]:
    config = json.loads(resolve_pinned_file(CONFIG_FILE).read_text(encoding="utf-8"))
    return frozenset(config["vocab"])


def _package_assets() -> dict[str, Any]:
    import kokoro.pipeline
    import misaki.espeak

    pipeline_path = Path(inspect.getfile(kokoro.pipeline))
    espeak_path = Path(inspect.getfile(misaki.espeak))
    actual = {
        "kokoro_pipeline_sha256": sha256_file(pipeline_path),
        "misaki_espeak_sha256": sha256_file(espeak_path),
    }
    expected = {
        "kokoro_pipeline_sha256": KOKORO_PIPELINE_SHA256,
        "misaki_espeak_sha256": MISAKI_ESPEAK_SHA256,
    }
    if actual != expected:
        raise PortugueseKokoroGateError("Portuguese G2P source hashes mismatch")
    if importlib.metadata.version("kokoro") != KOKORO_VERSION:
        raise PortugueseKokoroGateError("Kokoro version mismatch")
    if importlib.metadata.version("misaki") != MISAKI_VERSION:
        raise PortugueseKokoroGateError("Misaki version mismatch")
    return {
        "kokoro_version": KOKORO_VERSION,
        "misaki_version": MISAKI_VERSION,
        **actual,
    }


@dataclass(frozen=True)
class ExtractedPhone:
    word: str
    phone: str | None
    rejection_reason: str | None


class PortugueseKokoroExtractor:
    """Pinned batch equivalent of KPipeline(lang_code='p').g2p for isolated words."""

    def __init__(self) -> None:
        from kokoro import KPipeline

        self.pipeline = KPipeline(
            lang_code=KOKORO_LANG_CODE, repo_id=MODEL_REPO, model=False
        )
        self.g2p = self.pipeline.g2p
        self.vocab = _model_vocab()

    def _convert_backend_phone(self, raw: str) -> str:
        phone = raw.strip()
        for old, new in self.g2p.e2m:
            phone = phone.replace(old, new)
        phone = phone.replace("^", "").replace("-", "")
        return normalize_portuguese_phone(phone)

    def extract(self, words: Sequence[str]) -> list[ExtractedPhone]:
        if not words:
            return []
        raw_phones = self.g2p.backend.phonemize(list(words))
        if len(raw_phones) != len(words):
            raise PortugueseKokoroGateError("Portuguese batch G2P lost row alignment")
        records: list[ExtractedPhone] = []
        for word, raw in zip(words, raw_phones, strict=True):
            phone = self._convert_backend_phone(raw)
            if not phone:
                records.append(ExtractedPhone(word, None, "empty_phone"))
                continue
            unsupported = sorted(set(phone) - self.vocab)
            if unsupported:
                records.append(
                    ExtractedPhone(
                        word,
                        None,
                        "unsupported_symbols:" + "".join(unsupported),
                    )
                )
                continue
            records.append(ExtractedPhone(word, phone, None))
        return records


def _sample(inventory: Sequence[str]) -> tuple[str, ...]:
    mandatory = tuple(CHALLENGE_WORDS)
    missing = sorted(set(mandatory) - set(inventory))
    if missing:
        raise PortugueseKokoroGateError(f"challenge words missing: {missing}")
    selected = list(mandatory)
    selected_set = set(selected)
    candidates = [word for word in inventory if word not in selected_set]
    candidates.sort(key=lambda word: domain_hash("ptbr-kokoro-index-sample-v1", word))
    selected.extend(candidates[: SAMPLE_SIZE - len(selected)])
    return tuple(selected)


def _records_sha256(records: Sequence[ExtractedPhone]) -> str:
    payload = [
        {
            "word_sha256": _word_hash(record.word).hex(),
            "phone_sha256": (
                _phone_hash(record.phone).hex() if record.phone is not None else None
            ),
            "rejection_reason": record.rejection_reason,
        }
        for record in records
    ]
    return hashlib.sha256(stable_json(payload).encode("utf-8")).hexdigest()


def protocol_record() -> dict[str, Any]:
    inventory = portuguese_inventory()
    sample = _sample(inventory)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "status": "frozen_before_full_index_build",
        "language_id": LANGUAGE_ID,
        "scope": {
            "wordfreq_version": WORDFREQ_VERSION,
            "wordfreq_resource_sha256": WORDFREQ_RESOURCE_SHA256,
            "canonical_word_count": WORD_COUNT,
            "sample_size": SAMPLE_SIZE,
            "sample_words_sha256": hashlib.sha256(
                stable_json(list(sample)).encode("utf-8")
            ).hexdigest(),
        },
        "assets": {
            **_package_assets(),
            "model_repo": MODEL_REPO,
            "model_revision": MODEL_REVISION,
            "config_sha256": sha256_file(resolve_pinned_file(CONFIG_FILE)),
            "kokoro_lang_code": KOKORO_LANG_CODE,
            "espeak_voice": ESPEAK_VOICE,
        },
        "extraction": {
            "normalization_version": NORMALIZATION_VERSION,
            "mode": "isolated default KPipeline EspeakG2P prediction",
            "challenge_words": CHALLENGE_WORDS,
            "repeatability": "same frozen sample extracted twice before full build",
            "storage": "language-scoped hashes only; no plaintext words or phones",
        },
        "automatic_branches": {
            "complete_supplemental_index": (
                "sample extraction is byte-repeatable, all challenge predictions "
                "match, and full isolated-word coverage is at least 0.99"
            ),
            "partial_positive_only_index": (
                "repeatability and challenges pass, and coverage is at least 0.95 "
                "but below 0.99; positive collisions may reject candidates but a "
                "negative lookup cannot support clearance"
            ),
            "index_not_eligible_as_required_gate": (
                "repeatability or a challenge fails, or coverage is below 0.95"
            ),
        },
        "negative_lookup_scope": (
            "One isolated default pt-br eSpeak/Kokoro-mapped prediction per pinned "
            "wordfreq Portuguese word. It does not enumerate contextual variants, "
            "regional variants, phrase-level sandhi, alternate stress, names, or "
            "out-of-inventory words; even the complete branch is supplemental "
            "predicted-homophone evidence, not proof that no homophone exists."
        ),
        "api_calls_made": 0,
        "audio_renders_made": 0,
    }
    payload["protocol_sha256"] = hashlib.sha256(
        stable_json(payload).encode("utf-8")
    ).hexdigest()
    return payload


def verify_frozen_protocol(path: Path | None = None) -> dict[str, Any]:
    path = path or Paths().artifacts / "portuguese" / RUN_ID / "protocol.json"
    if not path.is_file():
        raise PortugueseKokoroGateError("frozen Portuguese index protocol is missing")
    stored = json.loads(path.read_text(encoding="utf-8"))
    expected = protocol_record()
    if stored != expected:
        raise PortugueseKokoroGateError("frozen Portuguese index protocol drifted")
    return stored


def _insert_metadata(conn: sqlite3.Connection, key: str, value: object) -> None:
    conn.execute(
        "INSERT INTO metadata(key, value) VALUES (?, ?)", (key, stable_json(value))
    )


def build_full_index(
    destination: Path | None = None,
    *,
    receipt_destination: Path | None = None,
    chunk_size: int = 4_096,
) -> dict[str, Any]:
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    protocol = verify_frozen_protocol()
    inventory = portuguese_inventory()
    sample = _sample(inventory)
    extractor = PortugueseKokoroExtractor()

    sample_first = extractor.extract(sample)
    sample_second = extractor.extract(sample)
    repeatable = _records_sha256(sample_first) == _records_sha256(sample_second)
    challenge = {
        word: next(record.phone for record in sample_first if record.word == word)
        for word in CHALLENGE_WORDS
    }
    challenge_pass = challenge == CHALLENGE_WORDS

    destination = destination or Paths().portuguese_kokoro_gate_db
    receipt_destination = receipt_destination or (
        Paths().artifacts / "portuguese" / RUN_ID / "full-index-receipt.json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f"{destination.name}.", suffix=".partial", dir=destination.parent
    )
    os.close(fd)
    temporary = Path(temporary_name)
    temporary.unlink(missing_ok=True)
    started = time.perf_counter()
    counts: Counter[str] = Counter()
    rejection_reasons: Counter[str] = Counter()
    record_digest = hashlib.sha256()

    try:
        conn = sqlite3.connect(temporary)
        conn.execute("PRAGMA page_size=4096")
        conn.execute("PRAGMA journal_mode=DELETE")
        conn.execute("PRAGMA synchronous=FULL")
        conn.execute("PRAGMA locking_mode=EXCLUSIVE")
        conn.execute(
            "CREATE TABLE word_phone("
            "word_sha256 BLOB NOT NULL, phone_sha256 BLOB NOT NULL, "
            "PRIMARY KEY(word_sha256, phone_sha256)) WITHOUT ROWID"
        )
        conn.execute("CREATE INDEX word_phone_by_phone ON word_phone(phone_sha256)")
        conn.execute("CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL)")

        for start in range(0, len(inventory), chunk_size):
            records = extractor.extract(inventory[start : start + chunk_size])
            rows: list[tuple[bytes, bytes]] = []
            for record in records:
                counts["input_words"] += 1
                if record.phone is None:
                    counts["uncovered_words"] += 1
                    rejection_reasons[record.rejection_reason or "unknown"] += 1
                    continue
                counts["covered_words"] += 1
                word_hash = _word_hash(record.word)
                phone_hash = _phone_hash(record.phone)
                rows.append((word_hash, phone_hash))
                record_digest.update(word_hash)
                record_digest.update(phone_hash)
            with conn:
                conn.executemany(
                    "INSERT INTO word_phone(word_sha256, phone_sha256) VALUES (?, ?)",
                    rows,
                )

        counts["database_rows"] = conn.execute(
            "SELECT COUNT(*) FROM word_phone"
        ).fetchone()[0]
        counts["unique_phone_hashes"] = conn.execute(
            "SELECT COUNT(DISTINCT phone_sha256) FROM word_phone"
        ).fetchone()[0]
        coverage = counts["covered_words"] / len(inventory)
        if repeatable and challenge_pass and coverage >= 0.99:
            status = "complete_supplemental_index"
        elif repeatable and challenge_pass and coverage >= 0.95:
            status = "partial_positive_only_index"
        else:
            status = "index_not_eligible_as_required_gate"
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "language_id": LANGUAGE_ID,
            "normalization_version": NORMALIZATION_VERSION,
            "protocol_sha256": protocol["protocol_sha256"],
            "record_stream_sha256": record_digest.hexdigest(),
            "status": status,
        }
        with conn:
            for key, value in metadata.items():
                _insert_metadata(conn, key, value)
        conn.execute("VACUUM")
        conn.close()
        os.replace(temporary, destination)

        receipt = {
            "schema_version": 1,
            "run_id": RUN_ID,
            "status": status,
            "protocol_sha256": protocol["protocol_sha256"],
            "sample_records_sha256": _records_sha256(sample_first),
            "sample_repeatable": repeatable,
            "challenge_predictions": challenge,
            "challenge_pass": challenge_pass,
            "counts": dict(sorted(counts.items())),
            "coverage_rate": coverage,
            "rejection_reasons": dict(sorted(rejection_reasons.items())),
            "record_stream_sha256": record_digest.hexdigest(),
            "database_sha256": sha256_file(destination),
            "database_bytes": destination.stat().st_size,
            "build_seconds": time.perf_counter() - started,
            "contains_plaintext_words_or_phones": False,
            "negative_lookup_scope": protocol["negative_lookup_scope"],
            "api_calls_made": 0,
            "audio_renders_made": 0,
        }
        atomic_write_json(receipt_destination, receipt)
        return receipt
    except Exception:
        try:
            conn.close()
        except (NameError, sqlite3.Error):
            pass
        temporary.unlink(missing_ok=True)
        raise


class PortugueseKokoroGateIndex:
    def __init__(self, database: Path | None = None) -> None:
        self.database = database or Paths().portuguese_kokoro_gate_db
        if not self.database.is_file():
            raise PortugueseKokoroGateError(
                f"Portuguese Kokoro gate database is missing: {self.database}"
            )
        self._thread_local = threading.local()

    def _connection(self) -> sqlite3.Connection:
        connection = getattr(self._thread_local, "connection", None)
        if connection is None:
            connection = sqlite3.connect(
                f"file:{self.database}?mode=ro&immutable=1", uri=True
            )
            self._thread_local.connection = connection
        return connection

    def phone_match(self, phone: str) -> bool:
        normalized = normalize_portuguese_phone(phone)
        if not normalized:
            raise PortugueseKokoroGateError("empty Portuguese Kokoro phone plan")
        return (
            self._connection()
            .execute(
                "SELECT 1 FROM word_phone WHERE phone_sha256 = ? LIMIT 1",
                (_phone_hash(normalized),),
            )
            .fetchone()
            is not None
        )
