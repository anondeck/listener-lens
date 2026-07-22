from __future__ import annotations

import hashlib
import importlib.metadata
import time
from collections import Counter
from typing import Any

from wordfreq import iter_wordlist, word_frequency

from .bilingual_vowel_engine import BilingualVowelPlanner
from .config import Paths, load_config, stable_json
from .gates import _wordfreq_resource_path, canonical_token
from .kokoro_strict_shell import STRICT_SHELL_VERSION
from .kokoro_typed_engine import _CONSONANT_SYMBOLS, _VOWEL_SYMBOLS
from .util import atomic_write_json, sha256_file


EN_AE_SHAPE_COVERAGE_VERSION = "en-ae-shape-coverage-v1"
RUN_ID = "20260718-en-ae-shape-coverage-v1"
RUN_DIR = Paths().artifacts / "typed-engine" / RUN_ID

_STRESS_MARKS = frozenset("ˈˌ")
_TARGET = "æ"

# Blocker identifiers name the exact structural reason a word's /ae/ occurrence
# falls outside the shipped strict C-stress-vowel-C shape.
BLOCKER_MULTISYLLABIC = "multisyllabic"
BLOCKER_MULTIPLE_TARGETS = "multiple_targets"
BLOCKER_UNSTRESSED_TARGET = "unstressed_target"
BLOCKER_NO_ONSET = "no_onset"
BLOCKER_ONSET_CLUSTER = "onset_cluster"
BLOCKER_NO_CODA = "no_coda"
BLOCKER_CODA_CLUSTER = "coda_cluster"
BLOCKER_NONCONFORMING = "nonconforming_symbols"


def classify_ae_word(phone: str) -> dict[str, Any]:
    """Classify one source word's /ae/ shape against the strict-shell contract.

    The strict shell (STRICT_SHELL_VERSION 1) accepts only the exact
    four-symbol shape consonant + stress mark + /ae/ + consonant. Every other
    /ae/-bearing word carries one or more named blockers.
    """

    ae_offsets = tuple(i for i, ch in enumerate(phone) if ch == _TARGET)
    vowel_count = sum(1 for ch in phone if ch in _VOWEL_SYMBOLS)
    blockers: set[str] = set()
    if not ae_offsets:
        return {"ae_count": 0, "blockers": (), "strict_supported": False}
    if len(ae_offsets) > 1:
        blockers.add(BLOCKER_MULTIPLE_TARGETS)
    if vowel_count > 1:
        blockers.add(BLOCKER_MULTISYLLABIC)

    index = ae_offsets[0]
    stressed = index > 0 and phone[index - 1] in _STRESS_MARKS
    if not stressed:
        blockers.add(BLOCKER_UNSTRESSED_TARGET)
    onset = phone[: index - 1] if stressed else phone[:index]
    coda = phone[index + 1 :]

    if vowel_count == 1 and len(ae_offsets) == 1:
        segments_conform = all(ch in _CONSONANT_SYMBOLS for ch in onset) and all(
            ch in _CONSONANT_SYMBOLS for ch in coda
        )
        if not segments_conform:
            blockers.add(BLOCKER_NONCONFORMING)
        else:
            if len(onset) == 0:
                blockers.add(BLOCKER_NO_ONSET)
            elif len(onset) >= 2:
                blockers.add(BLOCKER_ONSET_CLUSTER)
            if len(coda) == 0:
                blockers.add(BLOCKER_NO_CODA)
            elif len(coda) >= 2:
                blockers.add(BLOCKER_CODA_CLUSTER)

    return {
        "ae_count": len(ae_offsets),
        "blockers": tuple(sorted(blockers)),
        "strict_supported": not blockers,
    }


def characterize(*, limit: int | None = None) -> dict[str, Any]:
    planner = BilingualVowelPlanner.load(
        "en-US-to-pt-BR-vowels-v1", voice_id="af_heart"
    )
    config = load_config()["word_gate"]
    canonical_seen: set[str] = set()
    analyzed = 0
    errors: Counter[str] = Counter()
    ae_word_count = 0
    ae_freq_weight = 0.0
    bucket_words: Counter[str] = Counter()
    bucket_weight: Counter[str] = Counter()
    bucket_examples: dict[str, list[str]] = {}
    started = time.perf_counter()
    for raw_word in iter_wordlist("en", wordlist="large"):
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
        if _TARGET not in analysis.source_phonemes:
            continue
        shapes = [
            classify_ae_word(source_word.phone)
            for source_word in analysis.words
            if _TARGET in source_word.phone
        ]
        if not shapes:
            continue
        blockers = tuple(
            sorted({blocker for shape in shapes for blocker in shape["blockers"]})
        )
        bucket = "strict" if not blockers else "+".join(blockers)
        weight = word_frequency(word, "en")
        ae_word_count += 1
        ae_freq_weight += weight
        bucket_words[bucket] += 1
        bucket_weight[bucket] += weight
        rows = bucket_examples.setdefault(bucket, [])
        if len(rows) < 8:
            rows.append(word)
    if limit is None:
        expected = int(config["expected_counts"]["en"])
        if len(canonical_seen) != expected:
            raise RuntimeError(
                f"en word inventory drifted: {len(canonical_seen)} != {expected}"
            )
    buckets = [
        {
            "bucket": bucket,
            "word_count": bucket_words[bucket],
            "freq_weight": bucket_weight[bucket],
            "freq_share_of_ae": (
                bucket_weight[bucket] / ae_freq_weight if ae_freq_weight else 0.0
            ),
            "examples": bucket_examples.get(bucket, []),
        }
        for bucket in sorted(
            bucket_words, key=lambda name: bucket_weight[name], reverse=True
        )
    ]
    single_blocker_marginals = {
        row["bucket"]: {
            "word_count": row["word_count"],
            "freq_share_of_ae": row["freq_share_of_ae"],
        }
        for row in buckets
        if row["bucket"] != "strict" and "+" not in row["bucket"]
    }
    result = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "version": EN_AE_SHAPE_COVERAGE_VERSION,
        "status": "descriptive_shape_coverage_no_product_promotion",
        "strict_shell_version": STRICT_SHELL_VERSION,
        "profile_id": "en-US-to-pt-BR-vowels-v1",
        "voice_id_used_for_adapter": "af_heart",
        "inventory": "wordfreq-large canonical Latin-letter words",
        "limit": limit,
        "wordfreq_version": importlib.metadata.version("wordfreq"),
        "wordfreq_resource_sha256": {"en": sha256_file(_wordfreq_resource_path("en"))},
        "canonical_word_count": len(canonical_seen),
        "analyzed_word_count": analyzed,
        "analysis_error_count": sum(errors.values()),
        "analysis_errors": dict(sorted(errors.items())),
        "ae_word_count": ae_word_count,
        "ae_freq_weight": ae_freq_weight,
        "strict_supported_word_count": bucket_words["strict"],
        "strict_supported_freq_share_of_ae": (
            bucket_weight["strict"] / ae_freq_weight if ae_freq_weight else 0.0
        ),
        "buckets": buckets,
        "single_blocker_marginals": single_blocker_marginals,
        "interpretation": (
            "Frequency shares are wordfreq lexical priors over /ae/-bearing words, "
            "not user traffic. A blocker bucket names the structural gap only; "
            "supporting a new shape still requires its own frozen structural, "
            "acoustic, and listening evidence before any product enablement."
        ),
        "api_calls_made": 0,
        "production_enabled": False,
        "elapsed_s": time.perf_counter() - started,
    }
    result["record_sha256"] = hashlib.sha256(
        stable_json(result).encode("utf-8")
    ).hexdigest()
    return result


def write_characterization(*, limit: int | None = None) -> dict[str, Any]:
    if RUN_DIR.exists():
        raise RuntimeError(f"refusing to overwrite shape coverage run: {RUN_DIR}")
    result = characterize(limit=limit)
    RUN_DIR.mkdir(parents=True, exist_ok=False)
    atomic_write_json(RUN_DIR / "results.json", result)
    return result
