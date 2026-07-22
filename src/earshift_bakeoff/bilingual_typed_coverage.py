from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time
from typing import Any

from wordfreq import iter_wordlist, word_frequency

from .bilingual_candidate_registry import (
    BilingualCandidateRegistry,
    load_bilingual_candidate_registry,
)
from .bilingual_listener_engine_v8 import BilingualListenerPlannerV8
from .bilingual_product_isolation import active_changed_rule_ids
from .config import ROOT, Paths, sha256_json, stable_json
from .gates import _wordfreq_resource_path, canonical_token
from .util import atomic_write_json, sha256_file


VERSION = "bilingual-typed-coverage-v1"
RUN_ID = "20260718-bilingual-typed-coverage-v1"
PROTOCOL_FILE = "protocol.json"
RESULT_FILE = "results.json"
LEXICAL_LIMIT = 5_000
EXAMPLE_LIMIT = 12

CASES = (
    ("en-US-to-pt-BR-listener-v2", "af_heart", "en"),
    ("en-US-to-pt-BR-listener-v2", "am_michael", "en"),
    ("pt-BR-to-en-US-listener-v2", "pm_alex", "pt"),
    ("pt-BR-to-en-US-listener-v2", "pf_dora", "pt"),
)


def run_dir() -> Path:
    return Paths().artifacts / "typed-engine" / RUN_ID


def _top_words(language: str, limit: int) -> list[str]:
    words: list[str] = []
    seen: set[str] = set()
    for raw in iter_wordlist(language, wordlist="large"):
        word = canonical_token(raw)
        if word is None or word in seen:
            continue
        seen.add(word)
        words.append(word)
        if len(words) == limit:
            break
    if len(words) != limit:
        raise RuntimeError(f"{language} word list ended before {limit} entries")
    return words


def protocol_record() -> dict[str, Any]:
    bindings = {
        path: sha256_file(ROOT / path)
        for path in (
            "rules/bilingual-vowel-lenses.json",
            "rules/bilingual-listener-lenses-v2.json",
            "rules/bilingual-kokoro-candidate-state-v1.json",
            "rules/bilingual-kokoro-composition-candidate-v3.json",
            "rules/kokoro-product-voices.json",
        )
    }
    payload: dict[str, Any] = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "version": VERSION,
        "status": "frozen_before_characterization",
        "purpose": (
            "Measure real planner and automatic-candidate coverage over a fixed "
            "high-frequency lexical denominator in both directions and all four voices."
        ),
        "cases": [
            {"profile_id": profile, "voice_id": voice, "wordfreq_language": language}
            for profile, voice, language in CASES
        ],
        "inventory": {
            "wordlist": "wordfreq-large",
            "lexical_limit_per_case": LEXICAL_LIMIT,
            "wordfreq_version": importlib.metadata.version("wordfreq"),
            "resource_sha256": {
                language: sha256_file(_wordfreq_resource_path(language))
                for language in sorted({case[2] for case in CASES})
            },
            "interpretation": (
                "A high-frequency lexical stress test, not sentence traffic or a "
                "population-weighted usage estimate."
            ),
        },
        "classification": {
            "automatic_candidate_full": (
                "Every changed rule has a passing single-rule or adaptive-v8 cell."
            ),
            "automatic_candidate_partial": (
                "At least one passing cell exists but one or more detected changes "
                "would be omitted."
            ),
            "automatic_evidence_failed": "A detected candidate cell has a frozen failure.",
            "unsupported_rule_or_composition": (
                "No passing cell, too many passing rules, or a non-v8 composition."
            ),
            "no_listener_change": "The profile maps every detected segment identically.",
            "planner_failure": "The typed planner fails closed before candidate selection.",
        },
        "bindings": bindings,
        "scope": {
            "audio_renders": 0,
            "api_calls": 0,
            "production_enabled": False,
        },
    }
    return {**payload, "protocol_sha256": sha256_json(payload)}


def prepare() -> dict[str, Any]:
    protocol = protocol_record()
    destination = run_dir() / PROTOCOL_FILE
    if destination.is_file():
        existing = json.loads(destination.read_text(encoding="utf-8"))
        if stable_json(existing) != stable_json(protocol):
            raise RuntimeError("typed coverage protocol differs from the frozen record")
        return existing
    if (run_dir() / RESULT_FILE).exists():
        raise RuntimeError("typed coverage result exists before its protocol")
    destination.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(destination, protocol)
    return protocol


def _classification(
    plan: Any, registry: BilingualCandidateRegistry
) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    changed = active_changed_rule_ids(plan)
    if not changed:
        return "no_listener_change", (), ()
    passing = tuple(
        cell
        for rule_id in changed
        if (cell := registry.cell(plan.profile_id, plan.voice_id, rule_id)) is not None
        and cell.automatic_pass
    )
    selected = tuple(sorted({cell.rule_id for cell in passing}))
    omitted = tuple(rule_id for rule_id in changed if rule_id not in selected)
    composition_ok = bool(
        len(passing) == 1
        or (
            2 <= len(passing) <= 3
            and all(cell.candidate_rung == "v8" for cell in passing)
        )
    )
    if passing and composition_ok:
        return (
            "automatic_candidate_partial" if omitted else "automatic_candidate_full",
            selected,
            omitted,
        )
    matching = tuple(
        cell
        for rule_id in changed
        if (cell := registry.cell(plan.profile_id, plan.voice_id, rule_id)) is not None
    )
    if any(not cell.automatic_pass for cell in matching):
        return "automatic_evidence_failed", selected, changed
    return "unsupported_rule_or_composition", selected, changed


def _changed_occurrences(plan: Any) -> Counter[str]:
    counter: Counter[str] = Counter()
    for word in plan.words:
        for family in (
            word.vowel_occurrences,
            word.consonant_occurrences,
            word.prosody_occurrences,
            word.insertion_occurrences,
        ):
            counter.update(row.rule_id for row in family if row.changed)
    return counter


def _run_case(
    profile_id: str,
    voice_id: str,
    language: str,
    registry: BilingualCandidateRegistry,
) -> dict[str, Any]:
    planner = BilingualListenerPlannerV8.load(profile_id, voice_id=voice_id)
    words = _top_words(language, LEXICAL_LIMIT)
    statuses: Counter[str] = Counter()
    status_weight: Counter[str] = Counter()
    status_examples: dict[str, list[str]] = defaultdict(list)
    errors: Counter[str] = Counter()
    rule_words: Counter[str] = Counter()
    rule_occurrences: Counter[str] = Counter()
    selected_rule_words: Counter[str] = Counter()
    omitted_rule_words: Counter[str] = Counter()
    total_weight = 0.0
    started = time.perf_counter()
    for word in words:
        weight = word_frequency(word, language)
        total_weight += weight
        try:
            plan = planner.plan(word)
            occurrence_counts = _changed_occurrences(plan)
            rule_words.update(occurrence_counts.keys())
            rule_occurrences.update(occurrence_counts)
            status, selected, omitted = _classification(plan, registry)
            selected_rule_words.update(selected)
            omitted_rule_words.update(omitted)
        except Exception as exc:
            status = "planner_failure"
            code = f"{type(exc).__name__}:{getattr(exc, 'code', 'unknown')}"
            errors[code] += 1
        statuses[status] += 1
        status_weight[status] += weight
        if len(status_examples[status]) < EXAMPLE_LIMIT:
            status_examples[status].append(word)
    eligible = (
        statuses["automatic_candidate_full"] + statuses["automatic_candidate_partial"]
    )
    eligible_weight = (
        status_weight["automatic_candidate_full"]
        + status_weight["automatic_candidate_partial"]
    )
    return {
        "profile_id": profile_id,
        "voice_id": voice_id,
        "wordfreq_language": language,
        "lexical_denominator": len(words),
        "frequency_weight_denominator": total_weight,
        "status_counts": dict(sorted(statuses.items())),
        "status_frequency_weights": dict(sorted(status_weight.items())),
        "status_examples": dict(sorted(status_examples.items())),
        "automatic_candidate_word_yield": eligible / len(words),
        "automatic_candidate_frequency_yield": (
            eligible_weight / total_weight if total_weight else 0.0
        ),
        "full_candidate_word_yield": statuses["automatic_candidate_full"] / len(words),
        "partial_candidate_word_yield": (
            statuses["automatic_candidate_partial"] / len(words)
        ),
        "planner_failure_count": statuses["planner_failure"],
        "planner_errors": dict(sorted(errors.items())),
        "changed_rule_word_counts": dict(rule_words.most_common()),
        "changed_rule_occurrence_counts": dict(rule_occurrences.most_common()),
        "selected_rule_word_counts": dict(selected_rule_words.most_common()),
        "omitted_rule_word_counts": dict(omitted_rule_words.most_common()),
        "elapsed_s": time.perf_counter() - started,
    }


def run() -> dict[str, Any]:
    destination = run_dir() / RESULT_FILE
    if destination.is_file():
        return json.loads(destination.read_text(encoding="utf-8"))
    frozen = json.loads((run_dir() / PROTOCOL_FILE).read_text(encoding="utf-8"))
    if stable_json(frozen) != stable_json(protocol_record()):
        raise RuntimeError("typed coverage protocol or its inputs drifted")
    registry = load_bilingual_candidate_registry()
    cases = [
        _run_case(profile, voice, language, registry)
        for profile, voice, language in CASES
    ]
    payload = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "protocol_sha256": frozen["protocol_sha256"],
        "classification": "typed_coverage_baseline_complete_no_promotion",
        "cases": cases,
        "lexical_denominator_total": sum(row["lexical_denominator"] for row in cases),
        "audio_renders": 0,
        "api_calls": 0,
        "production_enabled": False,
    }
    result = {
        **payload,
        "record_sha256": hashlib.sha256(
            stable_json(payload).encode("utf-8")
        ).hexdigest(),
    }
    atomic_write_json(destination, result)
    return result
