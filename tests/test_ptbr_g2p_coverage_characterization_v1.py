from __future__ import annotations

import json

import pytest
from jsonschema import Draft202012Validator

from earshift_bakeoff.config import sha256_json
from earshift_bakeoff.ptbr_g2p_coverage_characterization_v1 import (
    CHARACTERIZATION_FILE,
    REPORT_FILE,
    RUN_ID,
    SCHEMA_FILE,
    characterization_record,
    run_dir,
    schema_record,
)
from earshift_bakeoff.util import sha256_file


@pytest.fixture(scope="module")
def record() -> dict:
    return characterization_record()


def test_record_validates_against_versioned_schema_and_internal_hash(record) -> None:
    schema = schema_record()
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(record)
    payload = dict(record)
    digest = payload.pop("characterization_sha256")
    assert digest == sha256_json(payload)
    assert record["run_id"] == RUN_ID
    assert record["status"] == "characterization_complete_nonpromotional"


def test_global_coverage_is_bound_and_honestly_positive_only(record) -> None:
    coverage = record["coverage_characterization"]
    assert coverage["status"] == "partial_positive_only_index"
    assert coverage["counts"] == {
        "covered_words": 255_881,
        "database_rows": 255_881,
        "input_words": 262_151,
        "uncovered_words": 6_270,
        "unique_phone_hashes": 238_702,
    }
    assert coverage["coverage_rate"] == 0.9760824868110364
    assert coverage["rejection_reasons"] == {
        "empty_phone": 1,
        "unsupported_symbols:õ": 1_757,
        "unsupported_symbols:õũ": 26,
        "unsupported_symbols:ũ": 4_486,
    }
    assert "missing match cannot clear" in coverage["interpretation"]
    parents = record["parent_evidence"]
    assert parents["portuguese_positive_only_index"]["database_sha256"] == (
        "ee63da3ebd3bd73eaa50beffc083a0d098e9150576b48ea0a4e9c3778805f89e"
    )
    assert parents["american_english_exact_phone_index"]["database_sha256"] == (
        "91aba4e91b993860956959180f48028ff4262626b7fb1c9fa2b76f2b8a41f9a3"
    )


def test_actual_outputs_cover_every_required_phenomenon_and_repeat(record) -> None:
    rows = {row["input"]: row for row in record["isolated_probes"]}
    assert {word: rows[word]["actual_output"] for word in rows} == {
        "pão": "pˈɐ̃ʊ̃",
        "caro": "kˈaɾʊ",
        "carro": "kˈaxʊ",
        "filho": "fˈiljʊ",
        "ninho": "nˈiɲʊ",
        "dia": "ʤˈiæ",
        "tia": "ʧˈiæ",
        "avó": "avˈɔ",
        "avô": "avˈo",
        "casa": "kˈazæ",
        "gente": "ʒˈAŋʧy",
        "livro": "lˈivrʊ",
    }
    assert {row["phenomenon"] for row in rows.values()} == {
        "nasal_diphthong",
        "rhotics_contrast",
        "lh_nh_palatal_consonants",
        "pre_i_affrication",
        "stress_open_mid_contrast",
        "final_unstressed_vowels",
    }
    assert all(row["repeatable"] for row in rows.values())
    assert all(row["model_vocab_representable"] for row in rows.values())
    repeatability = record["probe_protocol"]["isolated_repeatability"]
    assert repeatability["byte_repeatable"] is True
    assert repeatability["first_pass_sha256"] == repeatability["second_pass_sha256"]


def test_phrase_outputs_preserve_tested_boundaries_and_expose_context(record) -> None:
    rows = {row["input"]: row for row in record["phrase_probes"]}
    assert rows["Avó, avô!"]["actual_output"] == "avˈɔ, avˈo!"
    assert rows["Pão; caro, carro."]["actual_output"] == ("pˈɐ̃ʊ̃; kˈaɾʊ, kˈaxʊ.")
    assert rows["Dia tia"]["actual_output"] == "ʤˈiæ ʧˈiæ"
    assert rows["Dia, tia!"]["actual_output"] == "ʤˈiæ, ʧˈiæ!"
    assert all(row["punctuation_sequence_preserved"] for row in rows.values())
    boundary = record["boundary_characterization"]
    assert boundary["segment_plan_equal_after_removing_added_punctuation"] is True
    assert boundary["context_sensitivity_example"] == {
        "word": "gente",
        "isolated_output": "ʒˈAŋʧy",
        "connected_phrase_subplan": "ʒˈAŋʧj",
    }


def test_collision_screen_distinguishes_exact_hashes_from_listener_risk(record) -> None:
    screen = record["american_english_collision_screen"]
    assert screen["automated_exact_phone_index"] == {
        "collision_count": 0,
        "probe_count": 12,
        "all_exact_lookups_negative": True,
        "scope": screen["automated_exact_phone_index"]["scope"],
    }
    assert screen["desk_assessment"]["counts"] == {
        "salient": 5,
        "plausible": 3,
        "none_obvious": 4,
    }
    rows = {row["input"]: row for row in record["isolated_probes"]}
    assert rows["filho"]["possible_american_english_parse"] == "feel you"
    assert rows["ninho"]["possible_american_english_parse"] == "Nino"
    assert rows["dia"]["possible_american_english_parse"] == "Gia"
    assert rows["tia"]["possible_american_english_parse"] == "chia"
    assert "no recruited listener" in screen["desk_assessment"]["method"]
    assert "must not be treated" in screen["honest_assessment"]


def test_frozen_artifacts_are_intact_and_hash_bound() -> None:
    # The 2026-07-17 freeze predates later uv.lock changes, so prepare() can no
    # longer reproduce these files byte-for-byte; the durable invariants are
    # each frozen artifact's internal hash bindings and report content.
    stored = json.loads((run_dir() / CHARACTERIZATION_FILE).read_text(encoding="utf-8"))
    payload = dict(stored)
    digest = payload.pop("characterization_sha256")
    assert digest == sha256_json(payload)
    assert stored["schema_binding"]["sha256"] == sha256_file(run_dir() / SCHEMA_FILE)
    report = (run_dir() / REPORT_FILE).read_text(encoding="utf-8")
    assert digest in report
    assert "255,881 / 262,151" in report
    assert "feel you" in report
    assert "No recruited listener heard audio" in report
    assert "No synthesis, API call, paid call, or feature-flag change" in report
