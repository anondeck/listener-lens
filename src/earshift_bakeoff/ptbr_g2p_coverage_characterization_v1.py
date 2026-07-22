from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import json
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import Paths, sha256_json, stable_json
from .kokoro_gate_bridge import (
    ESPEAK_ASSET_HASHES,
    ESPEAK_LOADER_VERSION,
    PHONEMIZER_FORK_VERSION,
    KokoroGateIndex,
    _distribution_record,
    _tree_hash,
)
from .kokoro_specs import resolve_pinned_file
from .kokoro_synthesis import CONFIG_FILE, MODEL_REPO, MODEL_REVISION
from .portuguese_kokoro_gate import (
    CHALLENGE_WORDS,
    ESPEAK_VOICE,
    KOKORO_LANG_CODE,
    LANGUAGE_ID,
    NORMALIZATION_VERSION,
    RUN_ID as PORTUGUESE_INDEX_RUN_ID,
    PortugueseKokoroExtractor,
    _package_assets,
    verify_frozen_protocol,
)
from .util import atomic_write_json, atomic_write_text, sha256_file


RUN_ID = "20260717-ptbr-g2p-coverage-characterization-v1"
ENGLISH_INDEX_RUN_ID = "20260716-kokoro-gate-bridge-feasibility-v1"
CHARACTERIZATION_FILE = "characterization.json"
REPORT_FILE = "REPORT.md"
SCHEMA_FILE = "schema.json"
PUNCTUATION = ",;.!?"


@dataclass(frozen=True)
class IsolatedProbe:
    probe_id: str
    phenomenon: str
    word: str
    expected_output: str
    focus_symbols: tuple[str, ...]
    observation: str


@dataclass(frozen=True)
class PhraseProbe:
    probe_id: str
    phenomenon: str
    text: str
    expected_output: str
    observation: str


ISOLATED_PROBES = (
    IsolatedProbe(
        "nasal-diphthong-pao",
        "nasal_diphthong",
        "pão",
        "pˈɐ̃ʊ̃",
        ("ɐ̃", "ʊ̃"),
        "The emitted plan retains two explicitly nasalized vowel components.",
    ),
    IsolatedProbe(
        "rhotics-caro",
        "rhotics_contrast",
        "caro",
        "kˈaɾʊ",
        ("ɾ",),
        "The single written r is emitted as a tap symbol.",
    ),
    IsolatedProbe(
        "rhotics-carro",
        "rhotics_contrast",
        "carro",
        "kˈaxʊ",
        ("x",),
        "The written rr is emitted as the distinct back-fricative symbol x.",
    ),
    IsolatedProbe(
        "palatals-filho",
        "lh_nh_palatal_consonants",
        "filho",
        "fˈiljʊ",
        ("lj",),
        "The lh spelling is emitted as the two-symbol renderer sequence lj.",
    ),
    IsolatedProbe(
        "palatals-ninho",
        "lh_nh_palatal_consonants",
        "ninho",
        "nˈiɲʊ",
        ("ɲ",),
        "The nh spelling is emitted with the palatal nasal symbol ɲ.",
    ),
    IsolatedProbe(
        "affrication-dia",
        "pre_i_affrication",
        "dia",
        "ʤˈiæ",
        ("ʤ",),
        "The initial d before stressed i is emitted as the voiced affricate symbol ʤ.",
    ),
    IsolatedProbe(
        "affrication-tia",
        "pre_i_affrication",
        "tia",
        "ʧˈiæ",
        ("ʧ",),
        "The initial t before stressed i is emitted as the voiceless affricate symbol ʧ.",
    ),
    IsolatedProbe(
        "open-mid-avo-acute",
        "stress_open_mid_contrast",
        "avó",
        "avˈɔ",
        ("ˈɔ",),
        "The acute form carries stress before the open-mid back vowel symbol ɔ.",
    ),
    IsolatedProbe(
        "close-mid-avo-circumflex",
        "stress_open_mid_contrast",
        "avô",
        "avˈo",
        ("ˈo",),
        "The circumflex form carries stress before the distinct close-mid symbol o.",
    ),
    IsolatedProbe(
        "final-unstressed-a",
        "final_unstressed_vowels",
        "casa",
        "kˈazæ",
        ("æ",),
        "The isolated final written a maps to the renderer symbol æ.",
    ),
    IsolatedProbe(
        "final-unstressed-e",
        "final_unstressed_vowels",
        "gente",
        "ʒˈAŋʧy",
        ("y",),
        "The isolated final written e maps to y in this renderer plan.",
    ),
    IsolatedProbe(
        "final-unstressed-o",
        "final_unstressed_vowels",
        "livro",
        "lˈivrʊ",
        ("ʊ",),
        "The isolated final written o maps to the renderer symbol ʊ.",
    ),
)


PHRASE_PROBES = (
    PhraseProbe(
        "punctuation-stress-pair",
        "punctuation_and_boundaries",
        "Avó, avô!",
        "avˈɔ, avˈo!",
        "Comma, space, and final exclamation boundary markers remain explicit.",
    ),
    PhraseProbe(
        "punctuation-nasal-rhotics",
        "punctuation_and_boundaries",
        "Pão; caro, carro.",
        "pˈɐ̃ʊ̃; kˈaɾʊ, kˈaxʊ.",
        "Semicolon, comma, word spaces, and terminal period remain explicit.",
    ),
    PhraseProbe(
        "affrication-boundary-plain",
        "punctuation_and_boundaries",
        "Dia tia",
        "ʤˈiæ ʧˈiæ",
        "The unpunctuated two-word boundary is represented by one space.",
    ),
    PhraseProbe(
        "affrication-boundary-punctuated",
        "punctuation_and_boundaries",
        "Dia, tia!",
        "ʤˈiæ, ʧˈiæ!",
        "Adding comma and final punctuation leaves the two segment plans visible.",
    ),
    PhraseProbe(
        "final-vowels-connected-context",
        "final_unstressed_vowels_in_phrase_context",
        "A casa, a gente e o livro.",
        "a kˈazæ, a ʒˈAŋʧj i ʊ lˈivrʊ.",
        "The phrase output is context-sensitive: gente ends in j here, not isolated y.",
    ),
)


COLLISION_DESK_ASSESSMENTS = {
    "pão": (
        "plausible",
        "pow",
        "An AmE-oriented reader may collapse the nasalized diphthong toward ‘pow’; "
        "the nasal components distinguish the written plan but have not been listened to.",
    ),
    "caro": (
        "plausible",
        "Caro/Karo name or brand",
        "A name or brand-like parse is possible even though the tap and final ʊ are not "
        "an exact ordinary AmE lexical plan.",
    ),
    "carro": (
        "none_obvious",
        None,
        "No stable AmE parse was identified; x is itself an unusual AmE segment.",
    ),
    "filho": (
        "salient",
        "feel you",
        "The sequence f-il-j-ʊ presents a strong cross-word ‘feel you’ cue on paper.",
    ),
    "ninho": (
        "salient",
        "Nino",
        "The plan is close enough to the familiar proper-name pattern ‘Nino’ to require "
        "an audio collision check.",
    ),
    "dia": (
        "salient",
        "Gia",
        "The initial voiced affricate plus high vowel invites the proper-name parse ‘Gia’.",
    ),
    "tia": (
        "salient",
        "chia",
        "The initial voiceless affricate plus high vowel invites the English word ‘chia’.",
    ),
    "avó": (
        "none_obvious",
        None,
        "No stable AmE word or short phrase was identified from the plan alone.",
    ),
    "avô": (
        "none_obvious",
        None,
        "No stable AmE word or short phrase was identified from the plan alone.",
    ),
    "casa": (
        "salient",
        "casa",
        "The source is already a recognizable loan/name element for many AmE listeners, "
        "regardless of the final-vowel difference.",
    ),
    "gente": (
        "plausible",
        "Genji/name-like sequence",
        "The affricated ending can invite a name-like parse, but no exact stable English "
        "word was identified.",
    ),
    "livro": (
        "none_obvious",
        None,
        "No stable AmE word or short phrase was identified from the plan alone.",
    ),
}


def run_dir() -> Path:
    return Paths().artifacts / "portuguese" / RUN_ID


def _load_json(path: Path) -> Any:
    if not path.is_file():
        raise RuntimeError(f"required frozen artifact is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _write_once_json(path: Path, payload: Any) -> None:
    if path.exists():
        if _load_json(path) != payload:
            raise RuntimeError(f"existing versioned JSON differs: {path}")
        return
    atomic_write_json(path, payload)


def _write_once_text(path: Path, payload: str) -> None:
    if path.exists():
        if path.read_text(encoding="utf-8") != payload:
            raise RuntimeError(f"existing versioned report differs: {path}")
        return
    atomic_write_text(path, payload)


def _fallback_assets() -> dict[str, Any]:
    import espeakng_loader
    import phonemizer.backend.espeak.base
    import phonemizer.backend.espeak.words_mismatch
    import phonemizer.backend.espeak.wrapper

    if importlib.metadata.version("espeakng-loader") != ESPEAK_LOADER_VERSION:
        raise RuntimeError("espeakng-loader version drifted")
    if importlib.metadata.version("phonemizer-fork") != PHONEMIZER_FORK_VERSION:
        raise RuntimeError("phonemizer-fork version drifted")
    data_hash, data_file_count = _tree_hash(Path(espeakng_loader.get_data_path()))
    actual_material = {
        "libespeak-ng.dylib": sha256_file(
            Path(espeakng_loader.get_library_path())
        ),
        "data_tree": data_hash,
        "data_file_count": data_file_count,
        "loader_RECORD": sha256_file(_distribution_record("espeakng-loader")),
    }
    expected_material = {
        key: value
        for key, value in ESPEAK_ASSET_HASHES.items()
        if key != "phonemizer_RECORD"
    }
    if actual_material != expected_material:
        raise RuntimeError("pinned eSpeak fallback assets drifted")
    phonemizer_record = sha256_file(_distribution_record("phonemizer-fork"))
    return {
        "espeakng_loader_version": ESPEAK_LOADER_VERSION,
        "phonemizer_fork_version": PHONEMIZER_FORK_VERSION,
        "material_asset_hashes": actual_material,
        "phonemizer_record_sha256": phonemizer_record,
        "phonemizer_record_matches_shared_english_freeze": phonemizer_record
        == ESPEAK_ASSET_HASHES["phonemizer_RECORD"],
        "shared_english_freeze_phonemizer_record_sha256": ESPEAK_ASSET_HASHES[
            "phonemizer_RECORD"
        ],
        "phonemizer_espeak_source_sha256": {
            name: sha256_file(Path(inspect.getfile(module)))
            for name, module in {
                "base.py": phonemizer.backend.espeak.base,
                "words_mismatch.py": phonemizer.backend.espeak.words_mismatch,
                "wrapper.py": phonemizer.backend.espeak.wrapper,
            }.items()
        },
        "record_note": (
            "The current phonemizer distribution RECORD differs from the older shared "
            "English-index freeze. This artifact therefore binds the current RECORD and "
            "the exact active eSpeak source files instead of claiming RECORD equality; "
            "the eSpeak library, data tree, loader RECORD, versions, Kokoro pipeline, "
            "and Misaki eSpeak source still match their material pins."
        ),
    }


def _verified_parent_evidence() -> dict[str, Any]:
    root = Paths().root
    portuguese_dir = (
        Paths().artifacts / "portuguese" / PORTUGUESE_INDEX_RUN_ID
    )
    portuguese_protocol_path = portuguese_dir / "protocol.json"
    portuguese_receipt_path = portuguese_dir / "full-index-receipt.json"
    portuguese_protocol = verify_frozen_protocol(portuguese_protocol_path)
    portuguese_receipt = _load_json(portuguese_receipt_path)
    if portuguese_receipt.get("protocol_sha256") != portuguese_protocol.get(
        "protocol_sha256"
    ):
        raise RuntimeError("Portuguese index protocol/receipt binding drifted")
    if portuguese_receipt.get("status") != "partial_positive_only_index":
        raise RuntimeError("Portuguese index is no longer the frozen positive-only result")
    counts = portuguese_receipt["counts"]
    if counts["covered_words"] + counts["uncovered_words"] != counts["input_words"]:
        raise RuntimeError("Portuguese index coverage accounting drifted")
    if portuguese_receipt["coverage_rate"] != (
        counts["covered_words"] / counts["input_words"]
    ):
        raise RuntimeError("Portuguese index coverage rate drifted")
    if portuguese_receipt.get("challenge_predictions") != CHALLENGE_WORDS:
        raise RuntimeError("Portuguese challenge predictions drifted")
    if not portuguese_receipt.get("sample_repeatable") or not portuguese_receipt.get(
        "challenge_pass"
    ):
        raise RuntimeError("Portuguese parent repeatability/challenge gate drifted")
    if portuguese_receipt.get("api_calls_made") != 0 or portuguese_receipt.get(
        "audio_renders_made"
    ) != 0:
        raise RuntimeError("Portuguese parent is no longer zero-call/no-audio evidence")
    if sha256_file(Paths().portuguese_kokoro_gate_db) != portuguese_receipt.get(
        "database_sha256"
    ):
        raise RuntimeError("Portuguese positive-only database drifted")

    english_dir = Paths().artifacts / "typed-engine" / ENGLISH_INDEX_RUN_ID
    english_protocol_path = english_dir / "protocol.json"
    english_sample_path = english_dir / "sample-result.json"
    english_receipt_path = english_dir / "full-index-receipt.json"
    english_protocol = _load_json(english_protocol_path)
    english_sample = _load_json(english_sample_path)
    english_receipt = _load_json(english_receipt_path)
    if english_sample.get("protocol_sha256") != english_protocol.get(
        "protocol_sha256"
    ):
        raise RuntimeError("English index protocol/sample binding drifted")
    if english_receipt.get("protocol_sha256") != english_protocol.get(
        "protocol_sha256"
    ):
        raise RuntimeError("English index protocol/receipt binding drifted")
    if english_receipt.get("feasibility_result_sha256") != sha256_file(
        english_sample_path
    ):
        raise RuntimeError("English index sample/receipt binding drifted")
    if english_receipt.get("status") != "complete":
        raise RuntimeError("English collision index is incomplete")
    if english_receipt.get("contains_plaintext_words_or_phones") is not False:
        raise RuntimeError("English collision index plaintext contract drifted")
    if sha256_file(Paths().kokoro_gate_db) != english_receipt.get("database_sha256"):
        raise RuntimeError("English collision database drifted")

    return {
        "portuguese_positive_only_index": {
            "run_id": PORTUGUESE_INDEX_RUN_ID,
            "protocol_sha256": portuguese_protocol["protocol_sha256"],
            "protocol_file_sha256": sha256_file(portuguese_protocol_path),
            "receipt_file_sha256": sha256_file(portuguese_receipt_path),
            "database_sha256": portuguese_receipt["database_sha256"],
            "status": portuguese_receipt["status"],
        },
        "american_english_exact_phone_index": {
            "run_id": ENGLISH_INDEX_RUN_ID,
            "protocol_sha256": english_protocol["protocol_sha256"],
            "protocol_file_sha256": sha256_file(english_protocol_path),
            "sample_result_file_sha256": sha256_file(english_sample_path),
            "receipt_file_sha256": sha256_file(english_receipt_path),
            "database_sha256": english_receipt["database_sha256"],
            "status": english_receipt["status"],
            "negative_lookup_scope": english_receipt["negative_lookup_scope"],
        },
        "bound_paths": [
            str(path.relative_to(root))
            for path in (
                portuguese_protocol_path,
                portuguese_receipt_path,
                english_protocol_path,
                english_sample_path,
                english_receipt_path,
            )
        ],
    }


def _stable_hash(value: Any) -> str:
    return hashlib.sha256(stable_json(value).encode("utf-8")).hexdigest()


def _json_file_sha256(value: Any) -> str:
    rendered = (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    return hashlib.sha256(rendered).hexdigest()


def _extract_isolated(
    extractor: PortugueseKokoroExtractor,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    words = tuple(probe.word for probe in ISOLATED_PROBES)
    first = extractor.extract(words)
    second = extractor.extract(words)

    def compact(rows: Any) -> list[dict[str, Any]]:
        return [
            {
                "word": row.word,
                "output": row.phone,
                "rejection_reason": row.rejection_reason,
            }
            for row in rows
        ]

    first_compact = compact(first)
    second_compact = compact(second)
    first_hash = _stable_hash(first_compact)
    second_hash = _stable_hash(second_compact)
    if first_hash != second_hash:
        raise RuntimeError("isolated Portuguese probes are not byte-repeatable")
    records: list[dict[str, Any]] = []
    english_index = KokoroGateIndex()
    for probe, actual in zip(ISOLATED_PROBES, first, strict=True):
        if actual.word != probe.word or actual.phone != probe.expected_output:
            raise RuntimeError(f"frozen isolated probe drifted: {probe.probe_id}")
        if actual.rejection_reason is not None:
            raise RuntimeError(f"isolated probe became unrepresentable: {probe.probe_id}")
        risk, possible_parse, rationale = COLLISION_DESK_ASSESSMENTS[probe.word]
        records.append(
            {
                "probe_id": probe.probe_id,
                "phenomenon": probe.phenomenon,
                "input": probe.word,
                "actual_output": actual.phone,
                "focus_symbols": list(probe.focus_symbols),
                "repeatable": True,
                "model_vocab_representable": True,
                "observation": probe.observation,
                "american_english_exact_phone_index_collision": english_index.phone_match(
                    actual.phone
                ),
                "american_english_desk_collision_risk": risk,
                "possible_american_english_parse": possible_parse,
                "desk_assessment_rationale": rationale,
            }
        )
    return records, {
        "passes": 2,
        "first_pass_sha256": first_hash,
        "second_pass_sha256": second_hash,
        "byte_repeatable": True,
    }


def _phrase_output(extractor: PortugueseKokoroExtractor, text: str) -> str:
    phone, tokens = extractor.pipeline.g2p(text)
    if tokens is not None:
        raise RuntimeError("Portuguese phrase G2P unexpectedly returned token objects")
    output = unicodedata.normalize("NFC", phone).strip()
    if not output:
        raise RuntimeError("Portuguese phrase G2P returned an empty plan")
    unsupported = sorted(set(output) - extractor.vocab)
    if unsupported:
        raise RuntimeError(
            "Portuguese phrase output has unsupported symbols: " + "".join(unsupported)
        )
    return output


def _punctuation(value: str) -> str:
    return "".join(character for character in value if character in PUNCTUATION)


def _extract_phrases(
    extractor: PortugueseKokoroExtractor,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    first = [_phrase_output(extractor, probe.text) for probe in PHRASE_PROBES]
    second = [_phrase_output(extractor, probe.text) for probe in PHRASE_PROBES]
    first_payload = [
        {"input": probe.text, "output": output}
        for probe, output in zip(PHRASE_PROBES, first, strict=True)
    ]
    second_payload = [
        {"input": probe.text, "output": output}
        for probe, output in zip(PHRASE_PROBES, second, strict=True)
    ]
    first_hash = _stable_hash(first_payload)
    second_hash = _stable_hash(second_payload)
    if first_hash != second_hash:
        raise RuntimeError("phrase-level Portuguese probes are not byte-repeatable")
    records: list[dict[str, Any]] = []
    for probe, output in zip(PHRASE_PROBES, first, strict=True):
        if output != probe.expected_output:
            raise RuntimeError(f"frozen phrase probe drifted: {probe.probe_id}")
        records.append(
            {
                "probe_id": probe.probe_id,
                "phenomenon": probe.phenomenon,
                "input": probe.text,
                "actual_output": output,
                "repeatable": True,
                "model_vocab_representable": True,
                "input_punctuation": _punctuation(probe.text),
                "output_punctuation": _punctuation(output),
                "punctuation_sequence_preserved": _punctuation(probe.text)
                == _punctuation(output),
                "observation": probe.observation,
            }
        )
    return records, {
        "passes": 2,
        "first_pass_sha256": first_hash,
        "second_pass_sha256": second_hash,
        "byte_repeatable": True,
    }


def schema_record() -> dict[str, Any]:
    sha = {"type": "string", "pattern": "^[0-9a-f]{64}$"}
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": f"https://earshift.local/schemas/{RUN_ID}.json",
        "title": "Portuguese G2P coverage characterization v1",
        "type": "object",
        "additionalProperties": False,
        "required": [
            "$schema",
            "schema_version",
            "run_id",
            "status",
            "language_id",
            "provenance",
            "parent_evidence",
            "coverage_characterization",
            "probe_protocol",
            "isolated_probes",
            "phrase_probes",
            "boundary_characterization",
            "american_english_collision_screen",
            "findings",
            "limits_and_nonclaims",
            "zero_cost_receipt",
            "schema_binding",
            "characterization_sha256",
        ],
        "properties": {
            "$schema": {"const": SCHEMA_FILE},
            "schema_version": {"const": 1},
            "run_id": {"const": RUN_ID},
            "status": {"const": "characterization_complete_nonpromotional"},
            "language_id": {"const": "pt-BR"},
            "provenance": {"type": "object"},
            "parent_evidence": {"type": "object"},
            "coverage_characterization": {
                "type": "object",
                "required": [
                    "status",
                    "counts",
                    "coverage_rate",
                    "rejection_reasons",
                    "interpretation",
                ],
            },
            "probe_protocol": {"type": "object"},
            "isolated_probes": {
                "type": "array",
                "minItems": len(ISOLATED_PROBES),
                "maxItems": len(ISOLATED_PROBES),
                "items": {
                    "type": "object",
                    "required": [
                        "probe_id",
                        "phenomenon",
                        "input",
                        "actual_output",
                        "repeatable",
                        "model_vocab_representable",
                        "american_english_exact_phone_index_collision",
                        "american_english_desk_collision_risk",
                    ],
                },
            },
            "phrase_probes": {
                "type": "array",
                "minItems": len(PHRASE_PROBES),
                "maxItems": len(PHRASE_PROBES),
                "items": {
                    "type": "object",
                    "required": [
                        "probe_id",
                        "phenomenon",
                        "input",
                        "actual_output",
                        "repeatable",
                        "punctuation_sequence_preserved",
                    ],
                },
            },
            "boundary_characterization": {"type": "object"},
            "american_english_collision_screen": {"type": "object"},
            "findings": {"type": "array", "minItems": 1},
            "limits_and_nonclaims": {"type": "array", "minItems": 1},
            "zero_cost_receipt": {
                "type": "object",
                "properties": {
                    "api_calls_made": {"const": 0},
                    "paid_calls_made": {"const": 0},
                    "audio_renders_made": {"const": 0},
                    "feature_flags_changed": {"const": False},
                },
                "required": [
                    "api_calls_made",
                    "paid_calls_made",
                    "audio_renders_made",
                    "feature_flags_changed",
                ],
            },
            "schema_binding": {
                "type": "object",
                "required": ["relative_path", "sha256"],
                "properties": {
                    "relative_path": {"const": SCHEMA_FILE},
                    "sha256": sha,
                },
            },
            "characterization_sha256": sha,
        },
    }


def characterization_record() -> dict[str, Any]:
    root = Paths().root
    parent_evidence = _verified_parent_evidence()
    package_assets = _package_assets()
    fallback_assets = _fallback_assets()
    extractor = PortugueseKokoroExtractor()
    isolated, isolated_repeatability = _extract_isolated(extractor)
    phrases, phrase_repeatability = _extract_phrases(extractor)

    portuguese_receipt_path = (
        Paths().artifacts
        / "portuguese"
        / PORTUGUESE_INDEX_RUN_ID
        / "full-index-receipt.json"
    )
    receipt = _load_json(portuguese_receipt_path)
    schema = schema_record()
    source_paths = {
        "generator": root
        / "src"
        / "earshift_bakeoff"
        / "ptbr_g2p_coverage_characterization_v1.py",
        "portuguese_extractor": root
        / "src"
        / "earshift_bakeoff"
        / "portuguese_kokoro_gate.py",
        "english_collision_index": root
        / "src"
        / "earshift_bakeoff"
        / "kokoro_gate_bridge.py",
        "runner": root
        / "scripts"
        / "prepare_ptbr_g2p_coverage_characterization_v1.py",
        "dependency_lock": root / "uv.lock",
    }
    risk_counts = {
        risk: sum(
            row["american_english_desk_collision_risk"] == risk for row in isolated
        )
        for risk in ("salient", "plausible", "none_obvious")
    }
    exact_collision_count = sum(
        row["american_english_exact_phone_index_collision"] for row in isolated
    )
    plain = next(
        row for row in phrases if row["probe_id"] == "affrication-boundary-plain"
    )
    punctuated = next(
        row
        for row in phrases
        if row["probe_id"] == "affrication-boundary-punctuated"
    )
    strip_punctuation = str.maketrans("", "", PUNCTUATION)
    boundary_segments_equal = plain["actual_output"] == punctuated[
        "actual_output"
    ].translate(strip_punctuation)
    payload: dict[str, Any] = {
        "$schema": SCHEMA_FILE,
        "schema_version": 1,
        "run_id": RUN_ID,
        "status": "characterization_complete_nonpromotional",
        "language_id": LANGUAGE_ID,
        "provenance": {
            "mode": "actual local pinned Kokoro/Misaki/eSpeak G2P; no synthesis",
            "normalization_version": NORMALIZATION_VERSION,
            "kokoro_lang_code": KOKORO_LANG_CODE,
            "espeak_voice": ESPEAK_VOICE,
            "model_repo": MODEL_REPO,
            "model_revision": MODEL_REVISION,
            "model_config_sha256": sha256_file(resolve_pinned_file(CONFIG_FILE)),
            "kokoro_version": package_assets["kokoro_version"],
            "misaki_version": package_assets["misaki_version"],
            "kokoro_pipeline_sha256": package_assets["kokoro_pipeline_sha256"],
            "misaki_espeak_sha256": package_assets["misaki_espeak_sha256"],
            "fallback": fallback_assets,
            "source_file_sha256": {
                name: sha256_file(path) for name, path in sorted(source_paths.items())
            },
        },
        "parent_evidence": parent_evidence,
        "coverage_characterization": {
            "status": receipt["status"],
            "counts": receipt["counts"],
            "coverage_rate": receipt["coverage_rate"],
            "coverage_percent": receipt["coverage_rate"] * 100.0,
            "rejection_reasons": receipt["rejection_reasons"],
            "sample_repeatable": receipt["sample_repeatable"],
            "challenge_pass": receipt["challenge_pass"],
            "challenge_predictions": receipt["challenge_predictions"],
            "interpretation": (
                "97.608% isolated-word renderer-vocabulary coverage is a partial "
                "positive-only index. A matching phone may reject a carrier; a missing "
                "match cannot clear it because 6,270 inventory words are uncovered and "
                "contextual, regional, alternate-stress, name, and out-of-inventory "
                "variants are outside the index."
            ),
        },
        "probe_protocol": {
            "isolated_mode": "PortugueseKokoroExtractor.extract on one fixed ordered list",
            "phrase_mode": "KPipeline(lang_code='p', model=False).g2p(text)",
            "passes_per_probe_set": 2,
            "output_notation": (
                "Kokoro renderer-symbol plan after pinned Misaki eSpeak mapping and NFC "
                "normalization; it is not asserted to be a broad IPA transcription, a "
                "narrow phonetic transcription, or proof of acoustic realization"
            ),
            "isolated_repeatability": isolated_repeatability,
            "phrase_repeatability": phrase_repeatability,
            "required_phenomena": [
                "nasal/diphthong pão",
                "caro/carro rhotics",
                "filho/ninho lh/nh",
                "dia/tia affrication",
                "avó/avô stress and open/close-mid contrast",
                "final unstressed vowels",
                "punctuation and boundaries",
            ],
        },
        "isolated_probes": isolated,
        "phrase_probes": phrases,
        "boundary_characterization": {
            "plain_probe_id": plain["probe_id"],
            "punctuated_probe_id": punctuated["probe_id"],
            "segment_plan_equal_after_removing_added_punctuation": boundary_segments_equal,
            "all_phrase_punctuation_sequences_preserved": all(
                row["punctuation_sequence_preserved"] for row in phrases
            ),
            "context_sensitivity_observed": True,
            "context_sensitivity_example": {
                "word": "gente",
                "isolated_output": "ʒˈAŋʧy",
                "connected_phrase_subplan": "ʒˈAŋʧj",
            },
            "interpretation": (
                "The tested punctuation marks and visible word separators survive in "
                "these exact phrase plans. The gente difference also proves that an "
                "isolated-word plan cannot be assumed to enumerate phrase-context output."
            ),
        },
        "american_english_collision_screen": {
            "automated_exact_phone_index": {
                "collision_count": exact_collision_count,
                "probe_count": len(isolated),
                "all_exact_lookups_negative": exact_collision_count == 0,
                "scope": parent_evidence["american_english_exact_phone_index"][
                    "negative_lookup_scope"
                ],
            },
            "desk_assessment": {
                "method": (
                    "American-English-oriented phone-plan desk screen; model-assisted, "
                    "with no recruited listener and no audio audition"
                ),
                "risk_scale": {
                    "salient": "clear lexical, proper-name, loan, or short-phrase cue",
                    "plausible": "possible cue worth an audio screen",
                    "none_obvious": "no cue identified, not evidence that none exists",
                },
                "counts": risk_counts,
                "assessment_location": "per isolated probe",
            },
            "honest_assessment": (
                "Every exact American-English index lookup is negative, yet the desk "
                "screen identifies salient or plausible parses for several plans. Exact "
                "hash-index negatives therefore must not be treated as listener-level "
                "collision clearance; blinded audio screening remains necessary."
            ),
        },
        "findings": [
            "All 12 isolated outputs and all 5 phrase outputs repeated byte-for-byte.",
            "The fixed probes expose the requested nasal, rhotic, palatal, affricate, stress/open-mid, final-vowel, punctuation, and boundary distinctions in the emitted renderer plans.",
            "The full Portuguese inventory result is partial positive-only at 97.60824868110364% coverage, not a complete dictionary or negative-clearance gate.",
            "Connected phrase context can change an emitted subplan: isolated gente ends in y while the tested connected phrase uses j.",
            "Mechanical exact-phone collision lookup and likely listener collision risk are different questions and disagree on this probe set.",
        ],
        "limits_and_nonclaims": [
            "No audio was synthesized or listened to, so no acoustic phone realization, naturalness, intelligibility, or listener perception is established.",
            "The output strings are Kokoro renderer symbols after a pinned mapping, not a claim of linguistically canonical or dialect-complete IPA.",
            "Twelve isolated tokens and five phrases characterize selected behaviors; they do not establish arbitrary-word, arbitrary-context, regional, sociolectal, or phrase-level coverage.",
            "The positive-only Portuguese index may reject a known predicted collision, but a negative result cannot clear a candidate.",
            "The American-English desk screen is not a recruited-listener study, population estimate, or validated collision gate.",
            "No result enables a renderer, changes a feature flag, promotes a Portuguese listener lens, or supports production deployment.",
        ],
        "zero_cost_receipt": {
            "api_calls_made": 0,
            "paid_calls_made": 0,
            "audio_renders_made": 0,
            "feature_flags_changed": False,
        },
        "schema_binding": {
            "relative_path": SCHEMA_FILE,
            "sha256": _json_file_sha256(schema),
        },
    }
    return {**payload, "characterization_sha256": sha256_json(payload)}


def report_text(record: dict[str, Any], characterization_file_sha256: str) -> str:
    coverage = record["coverage_characterization"]
    counts = coverage["counts"]
    lines = [
        "# Brazilian Portuguese G2P and coverage characterization v1",
        "",
        f"Run: `{RUN_ID}`",
        "",
        f"Characterization SHA-256: `{record['characterization_sha256']}`",
        "",
        f"Characterization file SHA-256: `{characterization_file_sha256}`",
        f"Schema SHA-256: `{record['schema_binding']['sha256']}`",
        "",
        "## Outcome",
        "",
        "The pinned local Portuguese Kokoro/Misaki/eSpeak path is repeatable on this "
        "fixed probe set and emits distinct renderer plans for every requested behavior. "
        "The broader index remains **partial positive-only**, not a complete coverage or "
        "negative-clearance result.",
        "",
        "No synthesis, API call, paid call, or feature-flag change was made.",
        "",
        "## Pinned provenance",
        "",
        f"- Kokoro `{record['provenance']['kokoro_version']}`, Misaki "
        f"`{record['provenance']['misaki_version']}`, language code "
        f"`{record['provenance']['kokoro_lang_code']}`, eSpeak voice "
        f"`{record['provenance']['espeak_voice']}`.",
        f"- Model repository `{record['provenance']['model_repo']}` at revision "
        f"`{record['provenance']['model_revision']}`; only the pinned config/vocabulary "
        "is used here, not model inference.",
        "- The current phonemizer distribution RECORD does not match the older shared "
        "English-index freeze. The JSON binds the current RECORD plus the exact active "
        "phonemizer eSpeak source files; the eSpeak library/data, loader, package "
        "versions, Kokoro pipeline, and Misaki eSpeak source retain their material pins.",
        f"- Portuguese index parent `{record['parent_evidence']['portuguese_positive_only_index']['run_id']}` "
        f"and American-English exact-phone index parent "
        f"`{record['parent_evidence']['american_english_exact_phone_index']['run_id']}` "
        "are file- and database-hash bound in the JSON artifact.",
        "",
        "## Inventory coverage",
        "",
        f"- Covered: **{counts['covered_words']:,} / {counts['input_words']:,}** "
        f"(**{coverage['coverage_percent']:.6f}%**).",
        f"- Uncovered: **{counts['uncovered_words']:,}**.",
        f"- Unique retained phone hashes: **{counts['unique_phone_hashes']:,}**.",
        f"- Rejections: `{json.dumps(coverage['rejection_reasons'], ensure_ascii=False, sort_keys=True)}`.",
        "- Interpretation: a positive predicted-phone match may reject a candidate. A "
        "negative lookup cannot clear it because uncovered inventory items and contextual, "
        "regional, alternate-stress, name, and out-of-inventory variants remain.",
        "",
        "## Actual isolated outputs",
        "",
        "| Phenomenon | Input | Actual pinned output | Observation |",
        "|---|---:|---:|---|",
    ]
    for row in record["isolated_probes"]:
        lines.append(
            f"| `{row['phenomenon']}` | **{row['input']}** | `{row['actual_output']}` | "
            f"{row['observation']} |"
        )
    lines.extend(
        [
            "",
            "These are renderer-symbol plans, not claims of canonical IPA or acoustic "
            "realization. Both complete isolated passes had SHA-256 "
            f"`{record['probe_protocol']['isolated_repeatability']['first_pass_sha256']}`.",
            "",
            "## Phrase punctuation and boundaries",
            "",
            "| Input | Actual pinned output | Observation |",
            "|---|---|---|",
        ]
    )
    for row in record["phrase_probes"]:
        lines.append(
            f"| {row['input']} | `{row['actual_output']}` | {row['observation']} |"
        )
    lines.extend(
        [
            "",
            "All tested punctuation sequences are preserved. Removing the added comma "
            "and exclamation point from the `Dia, tia!` output yields the same segment "
            "plan as `Dia tia`. However, isolated `gente` ends in `y`, while the connected "
            "phrase probe contains `ʒˈAŋʧj`; this is direct evidence that isolated-word "
            "coverage does not enumerate contextual output.",
            "",
            "## American-English collision screen",
            "",
            "All 12 exact phone-plan lookups in the pinned American-English Kokoro index "
            "were negative. That is **not listener clearance**.",
            "",
            "| Input | Exact index collision | AmE-oriented desk risk | Possible parse | Rationale |",
            "|---|---:|---|---|---|",
        ]
    )
    for row in record["isolated_probes"]:
        possible = row["possible_american_english_parse"] or "—"
        lines.append(
            f"| **{row['input']}** | "
            f"{'yes' if row['american_english_exact_phone_index_collision'] else 'no'} | "
            f"`{row['american_english_desk_collision_risk']}` | {possible} | "
            f"{row['desk_assessment_rationale']} |"
        )
    lines.extend(
        [
            "",
            "This is a model-assisted, American-English-oriented phone-plan desk screen. "
            "No recruited listener heard audio. Its value is to show why exact hash-index "
            "negatives and listener collision risk are not interchangeable; any candidate "
            "still needs blinded audio screening.",
            "",
            "## Limits and nonclaims",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in record["limits_and_nonclaims"])
    lines.extend(
        [
            "",
            "## Machine-readable files",
            "",
            f"- `{CHARACTERIZATION_FILE}`: complete evidence and assessments.",
            f"- `{SCHEMA_FILE}`: JSON Schema 2020-12 contract.",
            f"- `{REPORT_FILE}`: this human-readable rendering.",
            "",
        ]
    )
    return "\n".join(lines)


def prepare() -> dict[str, Any]:
    schema = schema_record()
    record = characterization_record()
    destination = run_dir()
    schema_path = destination / SCHEMA_FILE
    record_path = destination / CHARACTERIZATION_FILE
    _write_once_json(schema_path, schema)
    _write_once_json(record_path, record)
    report = report_text(record, sha256_file(record_path))
    _write_once_text(destination / REPORT_FILE, report)
    return record
