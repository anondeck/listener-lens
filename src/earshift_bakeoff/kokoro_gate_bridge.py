from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.resources
import inspect
import json
import math
import os
import re
import sqlite3
import tempfile
import threading
import time
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from wordfreq import iter_wordlist

from .config import Paths, load_config, sha256_json, stable_json
from .gates import _wordfreq_resource_path, canonical_token, domain_hash
from .kokoro_synthesis import (
    CONFIG_FILE,
    KOKORO_VERSION,
    MODEL_FILE,
    MODEL_REPO,
    MODEL_REVISION,
    VOICE_FILE,
    verify_model_files,
)
from .util import atomic_write_json, sha256_file


RUN_ID = "20260716-kokoro-gate-bridge-feasibility-v1"
NORMALIZATION_VERSION = "kokoro-phone-normalization-v1"
SAMPLE_SIZE = 4_096
RANK_STRATA = 8
FULL_INVENTORY_COUNT = 292_477
FULL_INDEX_SCHEMA_VERSION = 1
MISAKI_VERSION = "0.9.4"
ESPEAK_LOADER_VERSION = "0.2.4"
PHONEMIZER_FORK_VERSION = "3.3.2"

SOURCE_BITS = {
    "gold": 1,
    "silver": 2,
    "special": 4,
    "morphology": 8,
    "espeak_fallback": 16,
}

MISAKI_ASSET_HASHES = {
    "en.py": "ddf67ad3bc4dd98143dcc9b6fbcde259b9d879d8b91684201cf0deeb07aa9910",
    "us_gold.json": "dc414872a49a28ae6c141463d502fd945f3b2fde040484fdc47d00cc4612686f",
    "us_silver.json": "de8f67be911bb6c659187b4a65fd966b6a30e56350e0f790d763210b053ac475",
    "RECORD": "43d97b49297cf4932a244a4e198adc447cb3685213de087c9a386ce14c549fc7",
}
KOKORO_PACKAGE_HASHES = {
    "pipeline.py": "09da32aab781f7a163cbf9f6e379d53db40957cbbab50bc26a21c8440c0eabd6",
    "RECORD": "895cadbd43b17e7cda5dddf231296de4603719c80bd70ea513f4bb12d9c4e8de",
}
ESPEAK_ASSET_HASHES = {
    "libespeak-ng.dylib": "bb635eee1ee9c456f4a5cf06fb6cb352ecdd4d61e1951743b423ef22bb57f470",
    "data_tree": "af3c2cec93f3e9a813b43f3419dbfdce44f66bebf89b1b26eee485aa606aa336",
    "data_file_count": 364,
    "loader_RECORD": "7967d6cc80ee79d79661a962aa4b414def4b7edcf47c2f194495583ba58ba9a7",
    "phonemizer_RECORD": "0ae7663cab9c6e1d98a93c0d9d1811ebfd8370ef67be3d25c460fbe61604f87e",
}

FUNCTION_WORDS = ("a", "am", "an", "by", "i", "in", "the", "to", "used")
FUNCTION_MINIMUM_VARIANTS = {
    "a": 2,
    "am": 2,
    "an": 1,
    "by": 1,
    "i": 1,
    "in": 2,
    "the": 2,
    "to": 3,
    "used": 2,
}
HETERONYM_WORDS = (
    "absent",
    "abstract",
    "abuse",
    "addict",
    "alternate",
    "conduct",
    "content",
    "contract",
    "desert",
    "invalid",
    "object",
    "permit",
    "present",
    "produce",
    "project",
    "record",
    "refuse",
    "subject",
    "use",
)
RHOTIC_CASES = ("bird", "car", "near", "nurse", "start")
DIPHTHONG_CASES = {
    "face": "A",
    "goat": "O",
    "price": "I",
    "mouth": "W",
    "choice": "Y",
}
BOUNDARY_CASES = (("an", "ice"), ("a", "nice"), ("some", "sun"))
STRESS_CASES = ("abstract", "permit", "present", "produce", "record")
MANDATORY_WORDS = tuple(
    dict.fromkeys(
        (
            *FUNCTION_WORDS,
            *HETERONYM_WORDS,
            *RHOTIC_CASES,
            *DIPHTHONG_CASES,
            *(word for pair in BOUNDARY_CASES for word in pair),
            *STRESS_CASES,
        )
    )
)

TAG_CASES = (
    "ADJ",
    "ADV",
    "DT",
    "IN",
    "JJ",
    "NN",
    "NNP",
    "NOUN",
    "PRP",
    "RB",
    "TO",
    "VBD",
    "VERB",
    "VB",
    "XX",
)
CONTEXT_CASES = tuple(
    (future_vowel, future_to)
    for future_vowel in (None, False, True)
    for future_to in (False, True)
)


class KokoroGateBridgeError(RuntimeError):
    """The preregistered Kokoro-native gate bridge could not be constructed."""


def _tree_hash(directory: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    files = sorted(path for path in directory.rglob("*") if path.is_file())
    for path in files:
        relative = str(path.relative_to(directory)).encode("utf-8")
        content = path.read_bytes()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest(), len(files)


def _distribution_record(name: str) -> Path:
    return Path(importlib.metadata.distribution(name)._path) / "RECORD"


def asset_manifest() -> dict[str, Any]:
    if importlib.metadata.version("kokoro") != KOKORO_VERSION:
        raise KokoroGateBridgeError("Kokoro package version differs from the freeze")
    if importlib.metadata.version("misaki") != MISAKI_VERSION:
        raise KokoroGateBridgeError("Misaki package version differs from the freeze")
    if importlib.metadata.version("espeakng-loader") != ESPEAK_LOADER_VERSION:
        raise KokoroGateBridgeError("espeakng-loader version differs from the freeze")
    if importlib.metadata.version("phonemizer-fork") != PHONEMIZER_FORK_VERSION:
        raise KokoroGateBridgeError("phonemizer-fork version differs from the freeze")

    import espeakng_loader
    import kokoro.pipeline
    import misaki.en
    from misaki import data

    files = verify_model_files(download=False)
    misaki_files = {
        "en.py": Path(inspect.getfile(misaki.en)),
        "us_gold.json": Path(
            str(importlib.resources.files(data).joinpath("us_gold.json"))
        ),
        "us_silver.json": Path(
            str(importlib.resources.files(data).joinpath("us_silver.json"))
        ),
        "RECORD": _distribution_record("misaki"),
    }
    kokoro_files = {
        "pipeline.py": Path(inspect.getfile(kokoro.pipeline)),
        "RECORD": _distribution_record("kokoro"),
    }
    espeak_data = Path(espeakng_loader.get_data_path())
    espeak_tree_hash, espeak_file_count = _tree_hash(espeak_data)
    espeak_values = {
        "libespeak-ng.dylib": sha256_file(Path(espeakng_loader.get_library_path())),
        "data_tree": espeak_tree_hash,
        "data_file_count": espeak_file_count,
        "loader_RECORD": sha256_file(_distribution_record("espeakng-loader")),
        "phonemizer_RECORD": sha256_file(_distribution_record("phonemizer-fork")),
    }
    misaki_values = {name: sha256_file(path) for name, path in misaki_files.items()}
    kokoro_values = {name: sha256_file(path) for name, path in kokoro_files.items()}
    if misaki_values != MISAKI_ASSET_HASHES:
        raise KokoroGateBridgeError("Misaki asset hashes differ from the freeze")
    if kokoro_values != KOKORO_PACKAGE_HASHES:
        raise KokoroGateBridgeError("Kokoro package hashes differ from the freeze")
    if espeak_values != ESPEAK_ASSET_HASHES:
        raise KokoroGateBridgeError(
            "Misaki eSpeak fallback assets differ from the freeze"
        )
    return {
        "kokoro": {
            "version": KOKORO_VERSION,
            "package_hashes": kokoro_values,
            "model_repo": MODEL_REPO,
            "model_revision": MODEL_REVISION,
            "model_hashes": {
                name: sha256_file(files[name])
                for name in (CONFIG_FILE, MODEL_FILE, VOICE_FILE)
            },
        },
        "misaki": {"version": MISAKI_VERSION, "asset_hashes": misaki_values},
        "fallback": {
            "espeakng_loader_version": ESPEAK_LOADER_VERSION,
            "phonemizer_fork_version": PHONEMIZER_FORK_VERSION,
            "asset_hashes": espeak_values,
        },
    }


def english_inventory() -> tuple[str, ...]:
    words: list[str] = []
    for raw_word in iter_wordlist("en", wordlist="large"):
        word = canonical_token(raw_word)
        if word is not None:
            words.append(word)
    inventory = tuple(dict.fromkeys(words))
    if len(inventory) != FULL_INVENTORY_COUNT:
        raise KokoroGateBridgeError(
            f"English inventory count {len(inventory)} != {FULL_INVENTORY_COUNT}"
        )
    return inventory


def select_sample(inventory: Sequence[str]) -> tuple[str, ...]:
    positions = {word: index for index, word in enumerate(inventory)}
    missing = [word for word in MANDATORY_WORDS if word not in positions]
    if missing:
        raise KokoroGateBridgeError(f"mandatory sample words are absent: {missing}")
    selected = list(MANDATORY_WORDS)
    selected_set = set(selected)
    remaining = SAMPLE_SIZE - len(selected)
    allocations = [remaining // RANK_STRATA] * RANK_STRATA
    for index in range(remaining % RANK_STRATA):
        allocations[index] += 1
    for stratum in range(RANK_STRATA):
        start = math.floor(len(inventory) * stratum / RANK_STRATA)
        end = math.floor(len(inventory) * (stratum + 1) / RANK_STRATA)
        candidates = [word for word in inventory[start:end] if word not in selected_set]
        candidates.sort(
            key=lambda word: domain_hash(
                "kokoro-gate-feasibility-sample-v1", str(stratum), word
            )
        )
        chosen = candidates[: allocations[stratum]]
        selected.extend(chosen)
        selected_set.update(chosen)
    if len(selected) != SAMPLE_SIZE or len(selected_set) != SAMPLE_SIZE:
        raise KokoroGateBridgeError(
            "sample selection did not produce unique bounded rows"
        )
    return tuple(selected)


def protocol_record() -> dict[str, Any]:
    config = load_config()
    configured_resource_hash = config["word_gate"]["resource_sha256"]["en"]
    resource_hash = sha256_file(_wordfreq_resource_path("en"))
    if resource_hash != configured_resource_hash:
        raise KokoroGateBridgeError(
            "wordfreq English resource hash differs from the gate freeze"
        )
    inventory = english_inventory()
    sample = select_sample(inventory)
    payload = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "status": "frozen_before_pronunciation_extraction",
        "purpose": (
            "Determine whether a variant-preserving Misaki/Kokoro-native predicted-homophone "
            "index is deterministic, representable, fast enough for the complete pinned English inventory, "
            "and suitable as a second screen beside the existing eSpeak gate."
        ),
        "scope": {
            "inventory_count": len(inventory),
            "wordfreq_version": config["word_gate"]["package_version"],
            "wordfreq_resource_sha256": resource_hash,
            "normalization_version": NORMALIZATION_VERSION,
            "sample_size": len(sample),
            "rank_strata": RANK_STRATA,
            "sample_selection": (
                "mandatory challenge union, then fixed allocation from each equal rank stratum by ascending "
                "SHA-256 domain kokoro-gate-feasibility-sample-v1"
            ),
            "sample_words_sha256": sha256_json(list(sample)),
            "mandatory_words": list(MANDATORY_WORDS),
        },
        "assets": asset_manifest(),
        "variant_contract": {
            "native": [
                "all reachable US gold variants, including every non-null selector in dictionary-valued entries",
                "US silver only when no gold entry is reachable for the casefolded word",
                "all function/special forms exposed over the frozen POS and future-vowel/future-to context grid",
                "reachable suffix-derived forms for words not directly covered",
            ],
            "casefolding": (
                "all Misaki surface keys sharing the canonical word are unioned; this is deliberately conservative"
            ),
            "fallback": (
                "only words with no native variant receive the pinned Misaki EspeakFallback batch path; "
                "fallback negatives are supplemental predicted-homophone evidence, not exhaustive proof"
            ),
            "provenance": (
                "every retained normalized word-phone pair carries its sorted source/selector provenance"
            ),
            "representability": (
                "post-G2P default-version T/t normalization; every remaining character must exist in the "
                "pinned Kokoro model vocabulary; otherwise reject the variant without character deletion"
            ),
        },
        "challenge_cases": {
            "function_minimum_variants": FUNCTION_MINIMUM_VARIANTS,
            "heteronyms": list(HETERONYM_WORDS),
            "rhotic": list(RHOTIC_CASES),
            "diphthongs": DIPHTHONG_CASES,
            "stress": list(STRESS_CASES),
            "boundary_pairs": [list(pair) for pair in BOUNDARY_CASES],
            "frozen_tags": list(TAG_CASES),
            "frozen_contexts": [list(context) for context in CONTEXT_CASES],
        },
        "measurements": [
            "two independent extraction-record hashes for exact repeatability",
            "cold asset/index initialization time and extraction throughput",
            "linear and 1.25x-conservative full-inventory build projection",
            "native, fallback, rejected, and unrepresentable word/variant counts",
            "unique exact Kokoro phone hashes and provenance-source counts",
            "stress, rhotic, diphthong, boundary, heteronym, and function behavior",
        ],
        "viability_gate": {
            "repeatability": "both normalized sample-record SHA-256 values must match",
            "coverage": "at least 99 percent of sampled words retain one representable native or fallback variant",
            "variant_rejection": "at most 1 percent of emitted raw variants may be unrepresentable",
            "challenge_behavior": "every frozen challenge family must pass its predetermined structural check",
            "projected_full_build": "1.25x-conservative projection must be at most 600 seconds",
            "decision": (
                "all gates pass -> build the full hash-only index; otherwise use an audited lossless "
                "Kokoro-token to gate-IPA mapping. No threshold changes after extraction."
            ),
        },
        "interpretation_limit": (
            "Even a pass establishes a pinned predicted-pronunciation screen, not exhaustive proof that no "
            "English homophone exists. Runtime must still pass the independent eSpeak spelling/phone gates "
            "and fail closed on unrepresentable plans or eligible-target disagreement."
        ),
        "api_calls": 0,
        "audio_renders": 0,
    }
    return {**payload, "protocol_sha256": sha256_json(payload)}


def prepare() -> dict[str, Any]:
    protocol = protocol_record()
    destination = Paths().artifacts / "typed-engine" / RUN_ID / "protocol.json"
    if destination.is_file():
        existing = json.loads(destination.read_text(encoding="utf-8"))
        if stable_json(existing) != stable_json(protocol):
            raise KokoroGateBridgeError(
                "existing gate-bridge protocol differs from freeze"
            )
    else:
        atomic_write_json(destination, protocol)
    return protocol


@dataclass
class WordVariants:
    word: str
    phones: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    rejections: list[str] = field(default_factory=list)
    raw_variant_count: int = 0

    def add(self, raw_phone: str | None, provenance: str, vocab: set[str]) -> None:
        self.raw_variant_count += 1
        if raw_phone is None:
            self.rejections.append(f"null:{provenance}")
            return
        phone = normalize_kokoro_phone(raw_phone)
        unsupported = sorted(set(phone) - vocab)
        if not phone:
            self.rejections.append(f"empty:{provenance}")
        elif unsupported:
            self.rejections.append(
                f"unrepresentable:{''.join(unsupported)}:{provenance}"
            )
        else:
            self.phones[phone].add(provenance)

    def stable_record(self) -> dict[str, Any]:
        return {
            "word": self.word,
            "variants": [
                {"phone": phone, "provenance": sorted(provenance)}
                for phone, provenance in sorted(self.phones.items())
            ],
            "rejections": sorted(self.rejections),
            "raw_variant_count": self.raw_variant_count,
        }


def normalize_kokoro_phone(value: str) -> str:
    return (
        unicodedata.normalize("NFC", value).strip().replace("ɾ", "T").replace("ʔ", "t")
    )


def _entry_values(
    value: str | dict[str, str | None],
) -> Iterable[tuple[str, str | None]]:
    if isinstance(value, str):
        yield "DEFAULT", value
    else:
        yield from sorted(value.items())


class MisakiVariantExtractor:
    def __init__(self) -> None:
        from misaki.en import Lexicon

        files = verify_model_files(download=False)
        config = json.loads(files[CONFIG_FILE].read_text(encoding="utf-8"))
        self.vocab = set(config["vocab"])
        self.lexicon = Lexicon(british=False)
        self.gold = self._canonical_entries(self.lexicon.golds)
        self.silver = self._canonical_entries(self.lexicon.silvers)

    @staticmethod
    def _canonical_entries(
        entries: dict[str, str | dict[str, str | None]],
    ) -> dict[str, list[tuple[str, str | dict[str, str | None]]]]:
        result: dict[str, list[tuple[str, str | dict[str, str | None]]]] = defaultdict(
            list
        )
        for surface, value in entries.items():
            canonical = canonical_token(surface)
            if canonical is not None:
                result[canonical].append((surface, value))
        return result

    def _direct(self, word: str, bundle: WordVariants) -> bool:
        source = (
            "gold" if word in self.gold else ("silver" if word in self.silver else None)
        )
        if source is None:
            return False
        entries = self.gold[word] if source == "gold" else self.silver[word]
        for surface, value in entries:
            for selector, phone in _entry_values(value):
                # Null selectors are explicit non-pronunciations in Misaki, not
                # emitted pronunciation variants. Preserve the selector contract
                # by skipping them rather than counting them as a rejected phone.
                if phone is None:
                    continue
                bundle.add(
                    phone,
                    f"{source}:{selector}:case={surface}",
                    self.vocab,
                )
        return bool(bundle.phones)

    def _special(self, word: str, bundle: WordVariants) -> None:
        if word not in FUNCTION_WORDS and word != "i":
            return
        from misaki.en import TokenContext

        surfaces = tuple(dict.fromkeys((word, word.capitalize(), word.upper())))
        for surface in surfaces:
            for tag in TAG_CASES:
                for future_vowel, future_to in CONTEXT_CASES:
                    context = TokenContext(
                        future_vowel=future_vowel, future_to=future_to
                    )
                    phone, _ = self.lexicon.get_special_case(
                        surface, tag, None, context
                    )
                    if phone is not None:
                        bundle.add(
                            phone,
                            (
                                f"special:surface={surface}:tag={tag}:"
                                f"future_vowel={future_vowel}:future_to={future_to}"
                            ),
                            self.vocab,
                        )

    def _morphology(self, word: str, bundle: WordVariants) -> None:
        if bundle.phones or not (
            word.endswith("s") or word.endswith("d") or word.endswith("ing")
        ):
            return
        from misaki.en import TokenContext

        for tag in TAG_CASES:
            for future_vowel, future_to in CONTEXT_CASES:
                context = TokenContext(future_vowel=future_vowel, future_to=future_to)
                phone, _ = self.lexicon.get_word(word, tag, None, context)
                if phone is not None:
                    bundle.add(
                        phone,
                        (
                            f"morphology:tag={tag}:future_vowel={future_vowel}:"
                            f"future_to={future_to}"
                        ),
                        self.vocab,
                    )

    def native(self, word: str) -> WordVariants:
        bundle = WordVariants(word)
        self._direct(word, bundle)
        self._special(word, bundle)
        self._morphology(word, bundle)
        return bundle

    def extract(self, words: Sequence[str]) -> list[WordVariants]:
        bundles = [self.native(word) for word in words]
        missing = [bundle for bundle in bundles if not bundle.phones]
        if missing:
            raw = batch_espeak_fallback([bundle.word for bundle in missing])
            if len(raw) != len(missing):
                raise KokoroGateBridgeError(
                    "Misaki fallback output lost word alignment"
                )
            for bundle, phone in zip(missing, raw, strict=True):
                bundle.add(phone, "espeak_fallback:isolated", self.vocab)
        return bundles


def _fallback_phone(raw: str, *, replacements: Sequence[tuple[str, str]]) -> str:
    phone = raw.strip()
    for old, new in replacements:
        phone = phone.replace(old, new)
    phone = re.sub(r"(\S)\u0329", r"ᵊ\1", phone).replace(chr(809), "")
    phone = phone.replace("o^ʊ", "O")
    phone = phone.replace("ɜːɹ", "ɜɹ").replace("ɜː", "ɜɹ")
    phone = phone.replace("ɪə", "iə").replace("ː", "")
    phone = phone.replace("o", "ɔ")
    return phone.replace("ɾ", "T").replace("ʔ", "t").replace("^", "")


def batch_espeak_fallback(words: Sequence[str]) -> list[str | None]:
    if not words:
        return []
    from misaki.espeak import EspeakFallback

    fallback = EspeakFallback(british=False)
    raw = fallback.backend.phonemize(list(words))
    if len(raw) != len(words):
        raise KokoroGateBridgeError("EspeakFallback batch output count mismatch")
    return [
        _fallback_phone(phone, replacements=fallback.E2M) if phone.strip() else None
        for phone in raw
    ]


def _records_hash(bundles: Sequence[WordVariants]) -> str:
    return sha256_json([bundle.stable_record() for bundle in bundles])


def _source_family(provenance: str) -> str:
    return provenance.split(":", 1)[0]


def _challenge_report(
    extractor: MisakiVariantExtractor, bundles: Sequence[WordVariants]
) -> dict[str, Any]:
    by_word = {bundle.word: bundle for bundle in bundles}
    functions = {
        word: {
            "variants": sorted(by_word[word].phones),
            "minimum": minimum,
            "pass": len(by_word[word].phones) >= minimum,
        }
        for word, minimum in FUNCTION_MINIMUM_VARIANTS.items()
    }
    heteronyms: dict[str, Any] = {}
    for word in HETERONYM_WORDS:
        expected = extractor.native(word)
        direct_expected = {
            phone
            for phone, provenance in expected.phones.items()
            if any(source.startswith("gold:") for source in provenance)
        }
        retained = set(by_word[word].phones)
        heteronyms[word] = {
            "variants": sorted(retained),
            "direct_gold_variant_count": len(direct_expected),
            "all_direct_gold_variants_retained": direct_expected <= retained,
            "multiple_variants": len(direct_expected) >= 2,
        }
    rhotic = {
        word: {
            "variants": sorted(by_word[word].phones),
            "pass": any("ɹ" in phone for phone in by_word[word].phones),
        }
        for word in RHOTIC_CASES
    }
    diphthongs = {
        word: {
            "symbol": symbol,
            "variants": sorted(by_word[word].phones),
            "pass": any(symbol in phone for phone in by_word[word].phones),
        }
        for word, symbol in DIPHTHONG_CASES.items()
    }
    stress = {
        word: {
            "variants": sorted(by_word[word].phones),
            "has_stress": any(
                "ˈ" in phone or "ˌ" in phone for phone in by_word[word].phones
            ),
        }
        for word in STRESS_CASES
    }
    boundary: dict[str, Any] = {}
    for left, right in BOUNDARY_CASES:
        combinations = sorted(
            {
                left_phone + right_phone
                for left_phone in by_word[left].phones
                for right_phone in by_word[right].phones
            }
        )
        boundary[f"{left}|{right}"] = {
            "combination_count": len(combinations),
            "phone_hashes": [
                domain_hash("kokoro-phone", phone).hex() for phone in combinations
            ],
            "pass": bool(combinations)
            and all(" " not in phone for phone in combinations),
        }
    family_pass = {
        "functions": all(item["pass"] for item in functions.values()),
        "heteronyms": all(
            item["multiple_variants"] and item["all_direct_gold_variants_retained"]
            for item in heteronyms.values()
        ),
        "rhotic": all(item["pass"] for item in rhotic.values()),
        "diphthongs": all(item["pass"] for item in diphthongs.values()),
        "stress": all(item["has_stress"] for item in stress.values()),
        "boundary": all(item["pass"] for item in boundary.values()),
    }
    return {
        "functions": functions,
        "heteronyms": heteronyms,
        "rhotic": rhotic,
        "diphthongs": diphthongs,
        "stress": stress,
        "boundary": boundary,
        "family_pass": family_pass,
        "pass": all(family_pass.values()),
    }


def measure() -> dict[str, Any]:
    protocol = prepare()
    inventory = english_inventory()
    sample = select_sample(inventory)
    if sha256_json(list(sample)) != protocol["scope"]["sample_words_sha256"]:
        raise KokoroGateBridgeError("sample no longer matches the frozen protocol")

    runs: list[dict[str, Any]] = []
    run_bundles: list[list[WordVariants]] = []
    for run_number in (1, 2):
        initialization_start = time.perf_counter()
        extractor = MisakiVariantExtractor()
        initialization_seconds = time.perf_counter() - initialization_start
        extraction_start = time.perf_counter()
        bundles = extractor.extract(sample)
        extraction_seconds = time.perf_counter() - extraction_start
        run_bundles.append(bundles)
        runs.append(
            {
                "run": run_number,
                "initialization_seconds": initialization_seconds,
                "extraction_seconds": extraction_seconds,
                "words_per_second": len(sample) / extraction_seconds,
                "records_sha256": _records_hash(bundles),
            }
        )
    first = run_bundles[0]
    repeatable = runs[0]["records_sha256"] == runs[1]["records_sha256"]
    total_raw = sum(bundle.raw_variant_count for bundle in first)
    total_rejected = sum(len(bundle.rejections) for bundle in first)
    covered = [bundle for bundle in first if bundle.phones]
    native = [
        bundle
        for bundle in first
        if any(
            _source_family(item) != "espeak_fallback"
            for provenance_set in bundle.phones.values()
            for item in provenance_set
        )
    ]
    fallback = [
        bundle
        for bundle in first
        if bundle.phones
        and all(
            _source_family(item) == "espeak_fallback"
            for provenance_set in bundle.phones.values()
            for item in provenance_set
        )
    ]
    phone_hashes = {
        domain_hash("kokoro-phone", phone).hex()
        for bundle in first
        for phone in bundle.phones
    }
    provenance_counts: Counter[str] = Counter()
    for bundle in first:
        for provenance_set in bundle.phones.values():
            provenance_counts.update(_source_family(item) for item in provenance_set)
    challenge = _challenge_report(MisakiVariantExtractor(), first)
    mean_extraction = sum(item["extraction_seconds"] for item in runs) / len(runs)
    linear_projection = mean_extraction * len(inventory) / len(sample)
    conservative_projection = linear_projection * 1.25
    coverage_rate = len(covered) / len(sample)
    rejection_rate = total_rejected / total_raw if total_raw else 1.0
    gates = {
        "repeatability": repeatable,
        "coverage": coverage_rate >= 0.99,
        "variant_rejection": rejection_rate <= 0.01,
        "challenge_behavior": challenge["pass"],
        "projected_full_build": conservative_projection <= 600,
    }
    viable = all(gates.values())
    result = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "status": "viable_build_full_index" if viable else "fallback_mapping_required",
        "protocol_sha256": protocol["protocol_sha256"],
        "api_calls_made": 0,
        "audio_renders_made": 0,
        "runs": runs,
        "aggregate": {
            "sample_words": len(sample),
            "covered_words": len(covered),
            "coverage_rate": coverage_rate,
            "native_words": len(native),
            "fallback_words": len(fallback),
            "rejected_words": len(sample) - len(covered),
            "raw_variants": total_raw,
            "retained_word_phone_variants": sum(len(bundle.phones) for bundle in first),
            "rejected_variants": total_rejected,
            "variant_rejection_rate": rejection_rate,
            "unique_phone_hashes": len(phone_hashes),
            "provenance_counts": dict(sorted(provenance_counts.items())),
        },
        "throughput": {
            "mean_sample_extraction_seconds": mean_extraction,
            "linear_full_build_seconds": linear_projection,
            "conservative_full_build_seconds": conservative_projection,
        },
        "challenge": challenge,
        "gates": gates,
        "viable": viable,
        "negative_lookup_scope": (
            "variant-complete for retained direct Misaki dictionary selectors and the frozen special/morphology "
            "enumeration; fallback-only words receive one pinned isolated eSpeak prediction, so all negative "
            "lookups remain supplemental predicted-homophone evidence rather than exhaustive proof"
        ),
    }
    run_dir = Paths().artifacts / "typed-engine" / RUN_ID
    public_records = [
        {
            "word_sha256": domain_hash("word", bundle.word).hex(),
            "phone_hashes": sorted(
                domain_hash("kokoro-phone", phone).hex() for phone in bundle.phones
            ),
            "variant_count": len(bundle.phones),
            "provenance_families": sorted(
                {
                    _source_family(item)
                    for provenance in bundle.phones.values()
                    for item in provenance
                }
            ),
            "rejection_count": len(bundle.rejections),
        }
        for bundle in first
    ]
    atomic_write_json(run_dir / "sample-records.json", public_records)
    result["sample_records_sha256"] = sha256_file(run_dir / "sample-records.json")
    atomic_write_json(run_dir / "sample-result.json", result)
    return result


def _insert_metadata(conn: sqlite3.Connection, key: str, value: object) -> None:
    conn.execute(
        "INSERT INTO metadata(key, value) VALUES (?, ?)",
        (key, stable_json(value)),
    )


def _source_mask(provenance: Iterable[str]) -> int:
    mask = 0
    for item in provenance:
        family = _source_family(item)
        try:
            mask |= SOURCE_BITS[family]
        except KeyError as exc:
            raise KokoroGateBridgeError(
                f"unknown pronunciation provenance family: {family}"
            ) from exc
    return mask


def _verified_feasibility_result() -> tuple[dict[str, Any], dict[str, Any]]:
    protocol = prepare()
    result_path = Paths().artifacts / "typed-engine" / RUN_ID / "sample-result.json"
    if not result_path.is_file():
        raise KokoroGateBridgeError("frozen feasibility result is missing")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("protocol_sha256") != protocol["protocol_sha256"]:
        raise KokoroGateBridgeError("feasibility result protocol hash mismatch")
    if not result.get("viable") or not all(result.get("gates", {}).values()):
        raise KokoroGateBridgeError("feasibility result does not authorize an index")
    return protocol, result


def build_full_index(
    destination: Path | None = None,
    *,
    inventory: Sequence[str] | None = None,
    extractor: MisakiVariantExtractor | None = None,
    chunk_size: int = 4_096,
    require_frozen_feasibility: bool = True,
    receipt_destination: Path | None = None,
) -> dict[str, Any]:
    """Build the deterministic, plaintext-free Kokoro pronunciation index."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    protocol: dict[str, Any] | None = None
    feasibility: dict[str, Any] | None = None
    if require_frozen_feasibility:
        protocol, feasibility = _verified_feasibility_result()

    paths = Paths()
    destination = destination or paths.kokoro_gate_db
    receipt_destination = receipt_destination or (
        paths.artifacts / "typed-engine" / RUN_ID / "full-index-receipt.json"
    )
    words = tuple(inventory) if inventory is not None else english_inventory()
    if require_frozen_feasibility and len(words) != FULL_INVENTORY_COUNT:
        raise KokoroGateBridgeError("full index input count differs from the freeze")
    if len(set(words)) != len(words):
        raise KokoroGateBridgeError("full index input contains duplicate words")

    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f"{destination.name}.", suffix=".partial", dir=destination.parent
    )
    os.close(fd)
    temporary = Path(temporary_name)
    temporary.unlink(missing_ok=True)
    started = time.perf_counter()
    extractor = extractor or MisakiVariantExtractor()
    counts: Counter[str] = Counter()
    rejection_reasons: Counter[str] = Counter()
    provenance_counts: Counter[str] = Counter()
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
            "source_mask INTEGER NOT NULL, PRIMARY KEY(word_sha256, phone_sha256)) "
            "WITHOUT ROWID"
        )
        conn.execute("CREATE INDEX word_phone_by_phone ON word_phone(phone_sha256)")
        conn.execute("CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL)")

        for start in range(0, len(words), chunk_size):
            chunk = words[start : start + chunk_size]
            bundles = extractor.extract(chunk)
            if [bundle.word for bundle in bundles] != list(chunk):
                raise KokoroGateBridgeError("extractor changed full-index word order")
            rows: list[tuple[bytes, bytes, int]] = []
            for bundle in bundles:
                counts["input_words"] += 1
                counts["raw_variants"] += bundle.raw_variant_count
                counts["rejected_variants"] += len(bundle.rejections)
                if bundle.phones:
                    counts["covered_words"] += 1
                else:
                    counts["uncovered_words"] += 1
                for rejection in bundle.rejections:
                    rejection_reasons[rejection.split(":", 1)[0]] += 1
                word_hash = domain_hash("word", bundle.word)
                for phone, provenance in sorted(bundle.phones.items()):
                    phone_hash = domain_hash("kokoro-phone", phone)
                    source_mask = _source_mask(provenance)
                    rows.append((word_hash, phone_hash, source_mask))
                    counts["word_phone_variants"] += 1
                    record_digest.update(word_hash)
                    record_digest.update(phone_hash)
                    record_digest.update(source_mask.to_bytes(4, "big"))
                    families = {_source_family(item) for item in provenance}
                    provenance_counts.update(families)
            with conn:
                conn.executemany(
                    "INSERT INTO word_phone(word_sha256, phone_sha256, source_mask) "
                    "VALUES (?, ?, ?)",
                    rows,
                )

        counts["unique_phone_hashes"] = conn.execute(
            "SELECT COUNT(DISTINCT phone_sha256) FROM word_phone"
        ).fetchone()[0]
        counts["database_rows"] = conn.execute(
            "SELECT COUNT(*) FROM word_phone"
        ).fetchone()[0]
        if counts["database_rows"] != counts["word_phone_variants"]:
            raise KokoroGateBridgeError("full index silently collapsed a variant row")
        if counts["covered_words"] + counts["uncovered_words"] != len(words):
            raise KokoroGateBridgeError("full index coverage accounting mismatch")

        metadata = {
            "schema_version": FULL_INDEX_SCHEMA_VERSION,
            "normalization_version": NORMALIZATION_VERSION,
            "input_words": len(words),
            "record_stream_sha256": record_digest.hexdigest(),
            "source_bits": SOURCE_BITS,
            "wordfreq_resource_sha256": (
                protocol["scope"]["wordfreq_resource_sha256"] if protocol else None
            ),
            "kokoro_version": KOKORO_VERSION,
            "misaki_version": MISAKI_VERSION,
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
            "status": "complete",
            "protocol_sha256": protocol["protocol_sha256"] if protocol else None,
            "feasibility_result_sha256": (
                sha256_file(
                    paths.artifacts / "typed-engine" / RUN_ID / "sample-result.json"
                )
                if feasibility
                else None
            ),
            "inventory": (
                {
                    "wordfreq_version": protocol["scope"]["wordfreq_version"],
                    "wordfreq_resource_sha256": protocol["scope"][
                        "wordfreq_resource_sha256"
                    ],
                    "input_words": len(words),
                }
                if protocol
                else {"input_words": len(words)}
            ),
            "assets": asset_manifest() if require_frozen_feasibility else None,
            "normalization_version": NORMALIZATION_VERSION,
            "source_bits": SOURCE_BITS,
            "counts": dict(sorted(counts.items())),
            "rejection_reasons": dict(sorted(rejection_reasons.items())),
            "provenance_variant_counts": dict(sorted(provenance_counts.items())),
            "record_stream_sha256": record_digest.hexdigest(),
            "database_sha256": sha256_file(destination),
            "database_bytes": destination.stat().st_size,
            "build_seconds": time.perf_counter() - started,
            "negative_lookup_scope": (
                "Variant-complete for pinned direct Misaki selectors and the frozen "
                "special/morphology enumeration. Fallback-only negatives remain a "
                "supplemental screen and do not prove that no homophone exists."
            ),
            "contains_plaintext_words_or_phones": False,
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


class KokoroGateIndex:
    def __init__(self, database: Path | None = None) -> None:
        self.database = database or Paths().kokoro_gate_db
        if not self.database.is_file():
            raise KokoroGateBridgeError(
                f"Kokoro gate database is missing: {self.database}"
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
        normalized = normalize_kokoro_phone(phone)
        if not normalized:
            raise KokoroGateBridgeError("empty Kokoro phone plan")
        phone_hash = domain_hash("kokoro-phone", normalized)
        return (
            self._connection()
            .execute(
                "SELECT 1 FROM word_phone WHERE phone_sha256 = ? LIMIT 1",
                (phone_hash,),
            )
            .fetchone()
            is not None
        )

    def source_mask(self, word: str, phone: str) -> int | None:
        canonical = canonical_token(word)
        if canonical is None:
            raise KokoroGateBridgeError("invalid word lookup")
        row = (
            self._connection()
            .execute(
                "SELECT source_mask FROM word_phone "
                "WHERE word_sha256 = ? AND phone_sha256 = ?",
                (
                    domain_hash("word", canonical),
                    domain_hash("kokoro-phone", normalize_kokoro_phone(phone)),
                ),
            )
            .fetchone()
        )
        return int(row[0]) if row is not None else None
