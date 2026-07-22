from __future__ import annotations

from collections import Counter
import hashlib
import importlib.metadata
import time
import unicodedata
from typing import Any, Sequence

from wordfreq import iter_wordlist

from .bilingual_vowel_engine import BilingualVowelPlanner, VowelRule
from .config import Paths, load_config, stable_json
from .gates import _wordfreq_resource_path, canonical_token
from .util import atomic_write_json, sha256_file


BILINGUAL_G2P_REACHABILITY_VERSION = "bilingual-g2p-reachability-v1"
RUN_ID = "20260717-bilingual-g2p-reachability-v1"
RUN_DIR = Paths().artifacts / "product-matrix" / RUN_ID


def scan_vowel_rule_ids(
    phone: str,
    *,
    rule_sources: Sequence[str],
    rules: dict[str, VowelRule],
) -> tuple[str, ...]:
    """Apply the planner's frozen longest-source vowel matching without carriers."""

    normalized = unicodedata.normalize("NFD", phone)
    found: list[str] = []
    position = 0
    while position < len(normalized):
        rule = next(
            (
                rules[source]
                for source in rule_sources
                if normalized.startswith(source, position)
            ),
            None,
        )
        if rule is None:
            position += 1
            continue
        found.append(rule.rule_id)
        position += len(rule.source)
    return tuple(found)


def _profile_characterization(
    *,
    language: str,
    profile_id: str,
    voice_id: str,
    limit: int | None,
) -> dict[str, Any]:
    planner = BilingualVowelPlanner.load(profile_id, voice_id=voice_id)
    rules_by_id = {rule.rule_id: rule for rule in planner.rules.values()}
    changed_ids = tuple(
        sorted(
            rule.rule_id
            for rule in planner.rules.values()
            if rule.source != rule.target
        )
    )
    all_ids = tuple(sorted(rules_by_id))
    word_counts: Counter[str] = Counter()
    occurrence_counts: Counter[str] = Counter()
    symbol_counts: Counter[str] = Counter()
    errors: Counter[str] = Counter()
    examples: dict[str, list[dict[str, str]]] = {}
    canonical_seen: set[str] = set()
    analyzed = 0
    started = time.perf_counter()
    for raw_word in iter_wordlist(language, wordlist="large"):
        word = canonical_token(raw_word)
        if word is None or word in canonical_seen:
            continue
        if limit is not None and len(canonical_seen) >= limit:
            break
        canonical_seen.add(word)
        try:
            analysis = planner.adapter.analyze(word)
        except Exception as exc:  # characterization preserves every adapter rejection
            errors[f"{type(exc).__name__}:{getattr(exc, 'code', 'unknown')}"] += 1
            continue
        analyzed += 1
        for symbol in analysis.source_phonemes:
            symbol_counts[symbol] += 1
        matched: list[str] = []
        for source_word in analysis.words:
            matched.extend(
                scan_vowel_rule_ids(
                    source_word.phone,
                    rule_sources=planner.rule_sources,
                    rules=planner.rules,
                )
            )
        per_word = set(matched)
        word_counts.update(per_word)
        occurrence_counts.update(matched)
        for rule_id in per_word:
            rows = examples.setdefault(rule_id, [])
            if len(rows) < 3:
                rows.append({"word": word, "source_phonemes": analysis.source_phonemes})
    rules = []
    for rule_id in all_ids:
        rule = rules_by_id[rule_id]
        rules.append(
            {
                "rule_id": rule_id,
                "source": rule.source,
                "target": rule.target,
                "changed": rule.source != rule.target,
                "word_count": word_counts[rule_id],
                "occurrence_count": occurrence_counts[rule_id],
                "observed_in_inventory": word_counts[rule_id] > 0,
                "examples": examples.get(rule_id, []),
            }
        )
    return {
        "language": language,
        "profile_id": profile_id,
        "voice_id_used_for_adapter": voice_id,
        "canonical_word_count": len(canonical_seen),
        "analyzed_word_count": analyzed,
        "analysis_error_count": sum(errors.values()),
        "analysis_errors": dict(sorted(errors.items())),
        "changed_rule_count": len(changed_ids),
        "observed_changed_rule_count": sum(
            row["changed"] and row["observed_in_inventory"] for row in rules
        ),
        "observed_changed_rule_ids": [
            row["rule_id"]
            for row in rules
            if row["changed"] and row["observed_in_inventory"]
        ],
        "not_observed_changed_rule_ids": [
            row["rule_id"]
            for row in rules
            if row["changed"] and not row["observed_in_inventory"]
        ],
        "source_symbol_counts": dict(sorted(symbol_counts.items())),
        "rules": rules,
        "elapsed_s": time.perf_counter() - started,
    }


def characterize(*, limit: int | None = None) -> dict[str, Any]:
    config = load_config()["word_gate"]
    profiles = (
        ("en", "en-US-to-pt-BR-vowels-v1", "af_heart"),
        ("pt", "pt-BR-to-en-US-vowels-v1", "pm_alex"),
    )
    characterizations = [
        _profile_characterization(
            language=language,
            profile_id=profile_id,
            voice_id=voice_id,
            limit=limit,
        )
        for language, profile_id, voice_id in profiles
    ]
    if limit is None:
        for row in characterizations:
            expected = int(config["expected_counts"][row["language"]])
            if row["canonical_word_count"] != expected:
                raise RuntimeError(
                    f"{row['language']} word inventory drifted: "
                    f"{row['canonical_word_count']} != {expected}"
                )
    result = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "version": BILINGUAL_G2P_REACHABILITY_VERSION,
        "status": "descriptive_source_g2p_inventory_no_product_promotion",
        "inventory": "wordfreq-large canonical Latin-letter words",
        "limit": limit,
        "wordfreq_version": importlib.metadata.version("wordfreq"),
        "wordfreq_resource_sha256": {
            language: sha256_file(_wordfreq_resource_path(language))
            for language, _, _ in profiles
        },
        "profiles": characterizations,
        "interpretation": (
            "Observed rules are lexical-priority cells for the pinned source adapters. "
            "Not observed does not prove absolute impossibility for names, code-switches, "
            "future G2P versions, or out-of-inventory text."
        ),
        "api_calls_made": 0,
        "production_enabled": False,
    }
    result["record_sha256"] = hashlib.sha256(
        stable_json(result).encode("utf-8")
    ).hexdigest()
    return result


def write_characterization(*, limit: int | None = None) -> dict[str, Any]:
    if RUN_DIR.exists():
        raise RuntimeError(f"refusing to overwrite G2P reachability run: {RUN_DIR}")
    result = characterize(limit=limit)
    RUN_DIR.mkdir(parents=True, exist_ok=False)
    atomic_write_json(RUN_DIR / "results.json", result)
    return result
