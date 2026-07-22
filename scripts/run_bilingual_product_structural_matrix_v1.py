#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Sequence
import unicodedata

from earshift_bakeoff.bilingual_listener_engine import (
    BilingualListenerPlanner,
    load_listener_profiles,
)
from earshift_bakeoff.bilingual_product_matrix import (
    BilingualProductMatrixError,
    load_bilingual_product_matrix,
)
from earshift_bakeoff.bilingual_vowel_engine import (
    BilingualVowelEngineError,
    SourceAnalysis,
    SourceWord,
)
from earshift_bakeoff.config import Paths, stable_json
from earshift_bakeoff.kokoro_gate_bridge import KokoroGateIndex
from earshift_bakeoff.kokoro_synthesis import CONFIG_FILE, verify_model_files
from earshift_bakeoff.listener_lens import DatabaseNonceChecker
from earshift_bakeoff.portuguese_carrier_planner_v1 import (
    PortuguesePositiveOnlyIndexV1,
)
from earshift_bakeoff.util import atomic_write_json, sha256_file


RUN_ID = "20260717-bilingual-product-structural-matrix-v1"
RUN_DIR = Paths().artifacts / "product-matrix" / RUN_ID


@dataclass(frozen=True)
class FixtureAdapter:
    language_id: str
    source_words: tuple[str, ...]
    source_phones: tuple[str, ...]
    punctuation: str

    def analyze(self, normalized_text: str) -> SourceAnalysis:
        if len(self.source_words) != len(self.source_phones):
            raise BilingualVowelEngineError(
                "fixture_alignment_drift", "Fixture word and phone counts differ."
            )
        expected_text = " ".join(self.source_words) + self.punctuation
        if normalized_text != expected_text:
            raise BilingualVowelEngineError(
                "fixture_text_drift", "Fixture text changed before analysis."
            )
        words = tuple(
            SourceWord(
                word_index=index,
                source=source,
                phone=unicodedata.normalize("NFD", phone),
            )
            for index, (source, phone) in enumerate(
                zip(self.source_words, self.source_phones, strict=True)
            )
        )
        separators = ("", *(" " for _ in words[1:]), self.punctuation)
        chunks = [separators[0]]
        for index, word in enumerate(words):
            chunks.extend((word.phone, separators[index + 1]))
        return SourceAnalysis(
            language_id=self.language_id,
            normalized_text=normalized_text,
            source_phonemes="".join(chunks),
            words=words,
            phone_separators=tuple(separators),
        )


def _rule_occurrence_count(plan: Any, rule_id: str) -> int:
    count = 0
    for word in plan.words:
        for attribute in (
            "vowel_occurrences",
            "consonant_occurrences",
            "insertion_occurrences",
            "prosody_occurrences",
        ):
            count += sum(
                occurrence.changed and occurrence.rule_id == rule_id
                for occurrence in getattr(word, attribute, ())
            )
    if rule_id in plan.active_prosody_rule_ids and count == 0:
        count = 1
    return count


def _record_hash(payload: dict[str, Any]) -> str:
    semantic = dict(payload)
    semantic.pop("record_sha256", None)
    return hashlib.sha256(stable_json(semantic).encode("utf-8")).hexdigest()


def _tuple(value: Sequence[str]) -> tuple[str, ...]:
    return tuple(str(item) for item in value)


def main() -> None:
    matrix = load_bilingual_product_matrix()
    manifest = matrix.validation_manifest()
    profiles = load_listener_profiles()
    files = verify_model_files(download=False)
    model_vocab = set(
        json.loads(files[CONFIG_FILE].read_text(encoding="utf-8"))["vocab"]
    )
    nonce_checker = DatabaseNonceChecker()
    phone_indexes = (KokoroGateIndex(), PortuguesePositiveOnlyIndexV1())
    outcomes: list[dict[str, Any]] = []
    for slot in manifest["slots"]:
        fixture = slot["fixture_spec"]
        try:
            profile = {
                **profiles[slot["profile_id"]],
                "voice_id": slot["voice_id"],
            }
            planner = BilingualListenerPlanner(
                profile=profile,
                adapter=FixtureAdapter(
                    language_id=profile["source_language"],
                    source_words=_tuple(fixture["source_words"]),
                    source_phones=_tuple(fixture["source_phones"]),
                    punctuation=fixture["punctuation"],
                ),
                model_vocab=model_vocab,
                nonce_checker=nonce_checker,
                phone_indexes=phone_indexes,
            )
            plan = planner.plan(fixture["text"])
            evidence = matrix.evaluate_plan(plan)
            occurrence_count = _rule_occurrence_count(plan, slot["rule_id"])
            expected_indexes = tuple(fixture["expected_target_word_indexes"])
            missing_expected_indexes = sorted(
                set(expected_indexes) - set(plan.target_word_indexes)
            )
            if occurrence_count < 1:
                raise BilingualProductMatrixError(
                    "fixture_rule_missing",
                    f"Fixture did not activate {slot['rule_id']}.",
                )
            if missing_expected_indexes:
                raise BilingualProductMatrixError(
                    "fixture_target_index_missing",
                    f"Fixture lost target indexes: {missing_expected_indexes}.",
                )
            if slot["rule_id"] not in evidence.changed_rule_ids:
                raise BilingualProductMatrixError(
                    "fixture_evidence_missing",
                    "Matrix evidence did not include the fixture rule.",
                )
            outcomes.append(
                {
                    "logical_slot_id": slot["logical_slot_id"],
                    "status": "pass",
                    "profile_id": slot["profile_id"],
                    "voice_id": slot["voice_id"],
                    "family": slot["family"],
                    "rule_id": slot["rule_id"],
                    "context": slot["context"],
                    "plan_sha256": plan.plan_sha256,
                    "target_rule_occurrence_count": occurrence_count,
                    "target_word_indexes": plan.target_word_indexes,
                    "candidate_attempts": plan.gates.candidate_attempts,
                    "candidate_rejection_counts": (
                        plan.gates.candidate_rejection_counts
                    ),
                    "matrix_product_ready": evidence.product_ready,
                }
            )
        except (BilingualVowelEngineError, BilingualProductMatrixError) as exc:
            outcomes.append(
                {
                    "logical_slot_id": slot["logical_slot_id"],
                    "status": "fail",
                    "profile_id": slot["profile_id"],
                    "voice_id": slot["voice_id"],
                    "family": slot["family"],
                    "rule_id": slot["rule_id"],
                    "context": slot["context"],
                    "error_code": getattr(exc, "code", type(exc).__name__),
                    "error": str(exc),
                }
            )
    passed = sum(row["status"] == "pass" for row in outcomes)
    failed = len(outcomes) - passed
    record = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "classification": (
            "all_structural_slots_pass"
            if failed == 0
            else "structural_gate_yield_incomplete"
        ),
        "matrix_version": matrix.matrix_version,
        "matrix_sha256": matrix.matrix_sha256,
        "source_manifest_sha256": hashlib.sha256(
            stable_json(manifest).encode("utf-8")
        ).hexdigest(),
        "planner_slot_count": len(outcomes),
        "planner_pass_count": passed,
        "planner_fail_count": failed,
        "planner_gate_yield": passed / len(outcomes),
        "api_calls_made": 0,
        "audio_renders_made": 0,
        "production_enabled": False,
        "model_config_sha256": sha256_file(files[CONFIG_FILE]),
        "outcomes": outcomes,
    }
    record["record_sha256"] = _record_hash(record)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    output = RUN_DIR / "results.json"
    atomic_write_json(output, record)
    print(
        json.dumps(
            {
                "output": str(output),
                "classification": record["classification"],
                "planner_slot_count": len(outcomes),
                "planner_pass_count": passed,
                "planner_fail_count": failed,
                "planner_gate_yield": record["planner_gate_yield"],
                "api_calls_made": 0,
                "audio_renders_made": 0,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
