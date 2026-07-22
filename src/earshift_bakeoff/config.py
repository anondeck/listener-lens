from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "bakeoff.yaml"
RULES_PATH = ROOT / "rules" / "phonotactics.yaml"
DEVLOG_PATH = ROOT / "DEVLOG.md"
CACHE_DIR = ROOT / ".cache"
ARTIFACTS_DIR = ROOT / "artifacts"

CRITERIA_START = "<!-- criteria:start -->"
CRITERIA_END = "<!-- criteria:end -->"


class ConfigurationError(RuntimeError):
    pass


def load_json_yaml(path: Path) -> dict[str, Any]:
    """Load our JSON-compatible YAML without adding a YAML dependency."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigurationError(f"Could not load {path}: {exc}") from exc


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    return load_json_yaml(path)


def load_rules(path: Path = RULES_PATH) -> dict[str, Any]:
    return load_json_yaml(path)


def criteria_text(path: Path = DEVLOG_PATH) -> str:
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    try:
        start = text.index(CRITERIA_START) + len(CRITERIA_START)
        end = text.index(CRITERIA_END, start)
    except ValueError as exc:
        raise ConfigurationError(
            "DEVLOG criteria markers are missing or malformed"
        ) from exc
    return text[start:end].strip() + "\n"


def criteria_sha256(path: Path = DEVLOG_PATH) -> str:
    return hashlib.sha256(criteria_text(path).encode("utf-8")).hexdigest()


def verify_criteria_hash(config: dict[str, Any] | None = None) -> str:
    config = config or load_config()
    actual = criteria_sha256()
    expected = config.get("criteria_sha256")
    if actual != expected:
        raise ConfigurationError(
            "DEVLOG pass criteria hash mismatch: "
            f"expected {expected!r}, computed {actual}. Do not run live stages."
        )
    return actual


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_json(value: Any) -> str:
    return hashlib.sha256(stable_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Paths:
    root: Path = ROOT
    cache: Path = CACHE_DIR
    artifacts: Path = ARTIFACTS_DIR

    @property
    def gate_db(self) -> Path:
        return self.cache / "gates" / "word-g2p.sqlite3"

    @property
    def kokoro_gate_db(self) -> Path:
        return self.cache / "gates" / "kokoro-word-phone.sqlite3"

    @property
    def portuguese_kokoro_gate_db(self) -> Path:
        return self.cache / "gates" / "kokoro-ptbr-word-phone-v1.sqlite3"

    @property
    def whisper_cache(self) -> Path:
        return self.cache / "whisper"

    @property
    def prepare_receipt(self) -> Path:
        return self.cache / "prepare-receipt.json"

    @property
    def smoke_receipt(self) -> Path:
        return self.artifacts / "smoke" / "receipt.json"

    def run_dir(self, run_id: str) -> Path:
        if not run_id or any(
            c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
            for c in run_id
        ):
            raise ConfigurationError(
                "run-id may contain only letters, digits, hyphens, and underscores"
            )
        return self.artifacts / "runs" / run_id
