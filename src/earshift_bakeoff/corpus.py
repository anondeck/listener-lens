from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from .config import RULES_PATH
from .models import CorpusBundle, GenerationBatch, LanguageCandidates
from .util import sha256_file


class CorpusError(RuntimeError):
    pass


class CodexCorpusGenerator:
    """Expose a frozen Codex-authored corpus through the generator protocol."""

    def __init__(self, path: Path) -> None:
        self.path = path
        try:
            self.bundle = CorpusBundle.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise CorpusError(f"Could not load Codex corpus {path}: {exc}") from exc
        self.corpus_sha256 = sha256_file(path)
        self.last_response_id = f"corpus:{self.corpus_sha256}"
        self.last_resolved_model = self.bundle.provenance.model_label
        self._validate_provenance()

    def _validate_provenance(self) -> None:
        prompt_path = self.path.with_name("PROMPT.md")
        if not prompt_path.is_file():
            raise CorpusError(f"Codex corpus prompt is missing: {prompt_path}")
        if sha256_file(prompt_path) != self.bundle.provenance.prompt_sha256:
            raise CorpusError("Codex corpus prompt checksum does not match provenance")
        if sha256_file(RULES_PATH) != self.bundle.provenance.rules_sha256:
            raise CorpusError("Codex corpus rule-table checksum does not match provenance")
        seen: set[str] = set()
        for round_index, batch in enumerate(self.bundle.rounds):
            for language in batch.languages:
                for candidate in language.candidates:
                    if candidate.profile_id != language.profile_id:
                        raise CorpusError(
                            f"Candidate {candidate.candidate_id} profile does not match "
                            f"its round-{round_index} language group"
                        )
                    if candidate.candidate_id in seen:
                        raise CorpusError(
                            f"Duplicate candidate id: {candidate.candidate_id}"
                        )
                    seen.add(candidate.candidate_id)

    def receipt(self) -> dict[str, object]:
        return {
            "schema_version": self.bundle.schema_version,
            "corpus_path": str(self.path),
            "corpus_sha256": self.corpus_sha256,
            "source": self.bundle.provenance.source,
            "model_label": self.bundle.provenance.model_label,
            "rounds": len(self.bundle.rounds),
            "candidates": sum(
                len(language.candidates)
                for batch in self.bundle.rounds
                for language in batch.languages
            ),
        }

    def generate(
        self, profile_ids: Sequence[str], count: int, refill_index: int = 0
    ) -> GenerationBatch:
        if refill_index >= len(self.bundle.rounds):
            raise CorpusError(
                f"Codex corpus has no round {refill_index}; "
                f"available rounds: {len(self.bundle.rounds)}"
            )
        wanted = set(profile_ids)
        languages: list[LanguageCandidates] = []
        for language in self.bundle.rounds[refill_index].languages:
            if language.profile_id not in wanted:
                continue
            languages.append(
                LanguageCandidates(
                    profile_id=language.profile_id,
                    candidates=language.candidates[:count],
                )
            )
        if not languages:
            raise CorpusError(
                f"Codex corpus round {refill_index} has none of: {sorted(wanted)}"
            )
        return GenerationBatch(languages=languages)
