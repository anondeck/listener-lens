from __future__ import annotations

import hashlib
import json

from earshift_bakeoff.config import ROOT, stable_json


RESULT = (
    ROOT
    / "artifacts"
    / "product-matrix"
    / "20260717-bilingual-g2p-reachability-v1"
    / "results.json"
)


def _load() -> dict:
    return json.loads(RESULT.read_text(encoding="utf-8"))


def _semantic_hash(payload: dict) -> str:
    semantic = dict(payload)
    semantic.pop("record_sha256", None)
    return hashlib.sha256(stable_json(semantic).encode("utf-8")).hexdigest()


def test_reachability_result_is_complete_zero_api_and_nonpromotional() -> None:
    result = _load()

    assert result["record_sha256"] == _semantic_hash(result)
    assert result["status"] == ("descriptive_source_g2p_inventory_no_product_promotion")
    assert result["api_calls_made"] == 0
    assert result["production_enabled"] is False
    assert result["limit"] is None
    assert result["wordfreq_resource_sha256"] == {
        "en": "dffae8066b78dce0a6667cf5f58e567054f902674667090a7ac8a8a44628b05c",
        "pt": "7d764586bca6262f554d5fa77ad8e6841ef42534776e70558065140853660ce2",
    }


def test_reachability_result_records_full_pinned_inventories() -> None:
    by_language = {row["language"]: row for row in _load()["profiles"]}

    assert by_language["en"]["canonical_word_count"] == 292477
    assert by_language["en"]["analyzed_word_count"] == 289029
    assert by_language["en"]["analysis_error_count"] == 3448
    assert by_language["pt"]["canonical_word_count"] == 262151
    assert by_language["pt"]["analyzed_word_count"] == 262136
    assert by_language["pt"]["analysis_error_count"] == 15


def test_reachability_result_separates_core_rules_from_unobserved_symbols() -> None:
    by_language = {row["language"]: row for row in _load()["profiles"]}

    assert by_language["en"]["observed_changed_rule_count"] == 10
    assert set(by_language["en"]["observed_changed_rule_ids"]) == {
        "enpt.aa_a",
        "enpt.ae_eh",
        "enpt.ah_a",
        "enpt.goat_o",
        "enpt.ih_i",
        "enpt.nurse_eh",
        "enpt.reduced_i_i",
        "enpt.reduced_schwa_a",
        "enpt.schwa_reduced_a",
        "enpt.uh_u",
    }
    assert by_language["pt"]["observed_changed_rule_count"] == 11
    portuguese = {
        row["rule_id"]: row
        for row in by_language["pt"]["rules"]
        if row["changed"] and row["observed_in_inventory"]
    }
    assert {
        rule_id for rule_id, row in portuguese.items() if row["word_count"] >= 100
    } == {
        "pten.a_ae",
        "pten.ao_aa",
        "pten.e_ih",
        "pten.final_a_schwa",
        "pten.final_e_i",
        "pten.nasal_a_schwa",
        "pten.nasal_o_goat",
        "pten.o_goat",
    }
    assert {
        rule_id for rule_id, row in portuguese.items() if row["word_count"] < 100
    } == {
        "pten.front_rounded_face",
        "pten.open_back_low_back",
        "pten.open_back_symbol_low_back",
    }
