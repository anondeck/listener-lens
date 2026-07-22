from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from earshift_bakeoff.util import sha256_file


ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = (
    ROOT
    / "artifacts"
    / "runtime-readiness"
    / "20260717-kokoro-js-onnx-structural-audit-v1"
)


def _load(name: str) -> dict[str, Any]:
    return json.loads((RUN_DIR / name).read_text(encoding="utf-8"))


def test_audit_validates_against_frozen_schema_and_bound_files() -> None:
    schema = _load("audit.schema.json")
    audit = _load("audit.json")

    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(audit)

    for binding in audit["artifact_bindings"].values():
        assert sha256_file(ROOT / binding["path"]) == binding["sha256"]


def test_exact_package_model_and_runtime_identities_are_pinned() -> None:
    audit = _load("audit.json")
    components = [row["component"] for row in audit["identities"]]
    identities = {row["component"]: row for row in audit["identities"]}

    assert len(components) == len(set(components))
    assert identities["kokoro_js_npm"]["revision"] == "1.2.1"
    assert identities["kokoro_js_source"]["revision"] == (
        "664c76a704021239ba59c84dcbaa4d3dece01fe9"
    )
    assert identities["onnx_model_repo"]["revision"] == (
        "1939ad2a8e416c0acfeecc08a694d14ef25f2231"
    )
    assert identities["transformers_js"]["revision"] == "3.5.1"
    assert identities["phonemizer"]["revision"] == "1.2.1"
    assert identities["onnxruntime_web"]["revision"] == (
        "1.22.0-dev.20250409-89f8206ba4"
    )

    assert all(row["license"] for row in identities.values())
    assert audit["retrieved_on"] == "2026-07-17"


def test_all_model_variants_have_exact_size_hash_and_dtype_mapping() -> None:
    audit = _load("audit.json")
    rows = {row["path"]: row for row in audit["model_files"]}

    assert set(rows) == {
        "onnx/model.onnx",
        "onnx/model_fp16.onnx",
        "onnx/model_quantized.onnx",
        "onnx/model_q8f16.onnx",
        "onnx/model_uint8.onnx",
        "onnx/model_uint8f16.onnx",
        "onnx/model_q4.onnx",
        "onnx/model_q4f16.onnx",
    }
    assert rows["onnx/model.onnx"]["bytes"] == 325_532_232
    assert rows["onnx/model.onnx"]["sha256"] == (
        "8fbea51ea711f2af382e88c833d9e288c6dc82ce5e98421ea61c058ce21a34cb"
    )
    assert rows["onnx/model_quantized.onnx"]["kokoro_js_declared_dtype"] == "q8"
    assert rows["onnx/model_q8f16.onnx"]["kokoro_js_declared_dtype"] is None
    assert rows["onnx/model_uint8f16.onnx"]["kokoro_js_declared_dtype"] is None
    assert all(re.fullmatch(r"[0-9a-f]{64}", row["sha256"]) for row in rows.values())


def test_graph_boundary_does_not_overclaim_uninspected_model_metadata() -> None:
    audit = _load("audit.json")
    contract = audit["effective_graph_contract"]

    assert [row["name"] for row in contract["inputs"]] == [
        "input_ids",
        "style",
        "speed",
    ]
    assert contract["consumed_output"]["name"] == "waveform"
    assert contract["raw_secondary_outputs"]["status"] == (
        "not_documented_not_parsed"
    )
    assert contract["sample_rate_hz"] == 24_000
    assert {
        "duration_override",
        "alignment_input",
        "text_encoder_state",
        "target_state_replacement",
        "f0_input",
        "noise_input",
        "decoder_only_entrypoint",
    }.issubset(set(contract["controls_not_exposed_by_effective_api"]))


def test_disposition_is_ordinary_synthesis_yes_controlled_replacement_no() -> None:
    audit = _load("audit.json")
    decision = audit["decision"]
    feasibility = audit["controlled_synthesis_feasibility"]

    assert decision == {
        "ordinary_browser_synthesis": "structurally_available",
        "token_id_injection": "available_via_generate_from_ids",
        "direct_controlled_target_state_replacement": (
            "not_exposed_by_documented_kokoro_js_effective_api"
        ),
        "substantial_custom_graph_work_required": True,
        "identified_implementation_path": (
            "staged_exports_or_equivalent_graph_restructuring_plus_custom_"
            "orchestration"
        ),
        "build_week_disposition": "post_build_week",
        "summary": decision["summary"],
    }
    assert feasibility["ordinary_synthesis"] is True
    assert feasibility["token_ids"] is True
    assert feasibility["documented_effective_api_target_state_replacement"] is False
    assert feasibility["raw_graph_inventory_verified"] is False
    assert len(feasibility["identified_post_build_week_work"]) >= 5


def test_browser_and_portuguese_claims_remain_bounded() -> None:
    audit = _load("audit.json")
    runtime = audit["browser_runtime"]
    pt = audit["tokenizer_and_g2p"]["portuguese_public_api"]

    assert runtime["wasm"] == "declared_supported"
    assert runtime["webgpu"] == "declared_supported_when_available"
    assert runtime["runtime_tested"] is False
    assert runtime["performance_claimed"] is False
    assert pt["supported"] is False
    assert pt["voice_asset_listed_in_immutable_metadata"] is True
    assert pt["public_voice_registry_entry"] is False
    assert audit["claim_limits"]["portuguese_claim"] == (
        "none_from_this_audit"
    )


def test_sources_are_primary_dated_and_hash_bound_when_stable() -> None:
    sources = _load("audit.json")["sources"]
    assert len({row["source_id"] for row in sources}) == len(sources)
    assert all(row["retrieved_on"] == "2026-07-17" for row in sources)
    assert all(row["url"].startswith("https://") for row in sources)

    stable_source_ids = {
        "kokoro_js_runtime_source",
        "kokoro_js_phonemize_source",
        "kokoro_js_voices_source",
        "kokoro_js_readme_source",
        "transformers_js_models_source",
        "transformers_js_onnx_backend_source",
        "transformers_js_dtype_source",
        "phonemizer_source",
        "espeak_ng_license",
        "onnx_tokenizer",
        "onnx_tokenizer_config",
        "onnx_transformers_config",
        "onnx_model_card",
    }
    by_id = {row["source_id"]: row for row in sources}
    assert stable_source_ids.issubset(by_id)
    for source_id in stable_source_ids:
        assert re.fullmatch(r"[0-9a-f]{64}", by_id[source_id]["sha256"])

    provenance_source_ids = {
        "kokoro_repository_head",
        "upstream_model_metadata",
        "transformers_js_npm_registry",
        "phonemizer_npm_registry",
        "onnxruntime_web_npm_registry",
        "onnx_model_card_stale_base_config_link",
    }
    assert provenance_source_ids.issubset(by_id)
    assert by_id["kokoro_repository_head"]["revision"] == (
        "dfb907a02bba8152ca444717ca5d78747ccb4bec"
    )
    assert by_id["upstream_model_metadata"]["revision"] == (
        "f3ff3571791e39611d31c381e3a41a3af07b4987"
    )
    assert by_id["onnx_model_card_stale_base_config_link"]["http_status"] == 404
    assert by_id["transformers_js_npm_registry"]["git_head"] == (
        "746c8c25bf27c5e0684a20f76889b4bb8d23e295"
    )
    assert by_id["phonemizer_npm_registry"]["git_head"] == (
        "6835144b7ee9043129222549c1ed2f6a27216278"
    )
    assert by_id["onnxruntime_web_npm_registry"]["git_head"] is None
    for row in sources:
        if "sha256" in row:
            assert (
                "raw.githubusercontent.com" in row["url"]
                or "/resolve/" in row["url"]
            )


def test_license_and_revision_gaps_are_explicit_not_silently_resolved() -> None:
    audit = _load("audit.json")
    risk_ids = [row["risk_id"] for row in audit["risks_and_gaps"]]
    gaps = {row["risk_id"]: row for row in audit["risks_and_gaps"]}

    assert len(risk_ids) == len(set(risk_ids))
    assert gaps["R2"]["severity"] == "high"
    assert "mutable" in gaps["R2"]["finding"]
    assert gaps["R4"]["severity"] == "high"
    assert "eSpeak" in gaps["R4"]["finding"]
    assert audit["licenses"]["legal_conclusion"] == (
        "not_provided_requires_compliance_review"
    )


def test_tokenizer_g2p_and_id_injection_contract_is_exact() -> None:
    sections = _load("audit.json")["tokenizer_and_g2p"]
    tokenizer = sections["tokenizer"]
    g2p = sections["g2p"]
    injection = sections["token_id_injection"]

    assert tokenizer["model_max_length"] == 512
    assert tokenizer["raw_phoneme_token_limit"] == 510
    assert tokenizer["pad_unknown_bos_eos_symbol"] == "$"
    assert tokenizer["pad_unknown_bos_eos_id"] == 0
    assert tokenizer["target_ids"] == {"ɑ": 69, "ɔ": 76}
    assert g2p["package"] == "phonemizer@1.2.1"
    assert g2p["public_language_modes"] == {"a": "en-us", "b": "en"}
    assert g2p["exact_embedded_espeak_revision"] is None
    assert injection["token_id_injection_supported"] is True
    assert injection["bypasses_text_normalization_and_g2p"] is True
    assert injection["raw_phoneme_string_api"] is False
    assert injection["state_replacement_enabled"] is False


def test_independent_review_findings_are_explicitly_resolved() -> None:
    review = _load("audit.json")["independent_review"]

    assert review["disposition_received"] == "revise_before_freezing"
    assert review["resolved"] is True
    assert len(review["resolutions"]) >= 5
    assert any("raw graph inventory" in item for item in review["resolutions"])
