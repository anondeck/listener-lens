from __future__ import annotations

import hashlib
import json
import subprocess
import unicodedata
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from earshift_bakeoff.kokoro_specs import VOICE_SPECS_BY_ID
from earshift_bakeoff.kokoro_synthesis import MODEL_FILE, MODEL_HASHES
from earshift_bakeoff.util import sha256_file


ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = (
    ROOT / "artifacts" / "runtime-readiness" / "20260717-kokoro-container-benchmark-v1"
)


def _load(name: str) -> dict[str, Any]:
    return json.loads((RUN_DIR / name).read_text(encoding="utf-8"))


def _canonical_request(value: dict[str, Any]) -> bytes:
    normalized = {
        key: unicodedata.normalize("NFC", item) if isinstance(item, str) else item
        for key, item in value.items()
    }
    return json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def test_protocol_and_result_schemas_are_valid_draft_2020_12() -> None:
    protocol_schema = _load("protocol.schema.json")
    result_schema = _load("benchmark-result.schema.json")
    protocol = _load("protocol.json")

    Draft202012Validator.check_schema(protocol_schema)
    Draft202012Validator(protocol_schema).validate(protocol)
    Draft202012Validator.check_schema(result_schema)


def test_protocol_binds_exact_identity_fixture_and_host_files() -> None:
    protocol = _load("protocol.json")
    bindings = protocol["artifact_bindings"]

    for key in ("identity_manifest", "fixtures", "current_host_receipt"):
        binding = bindings[key]
        assert sha256_file(RUN_DIR / binding["path"]) == binding["sha256"]

    assert not (RUN_DIR / bindings["future_result_filename"]).exists()
    assert protocol["status"] == "frozen_not_executed"
    assert (
        protocol["result_contract"]["empty_or_placeholder_measurements_allowed"]
        is False
    )


def test_fixture_hashes_are_canonical_exact_and_results_are_hash_only() -> None:
    artifact = _load("fixtures.json")
    rows = artifact["fixtures"]
    assert [row["fixture_id"] for row in rows] == [
        "fx-short-01",
        "fx-repeat-02",
        "fx-punct-03",
        "fx-long-04",
    ]
    assert len({row["request_sha256"] for row in rows}) == len(rows)
    for row in rows:
        actual = hashlib.sha256(_canonical_request(row["request"])).hexdigest()
        assert actual == row["request_sha256"]
        assert row["request"] == {
            "output_format": "wav-pcm16-mono-24000",
            "profile_id": "en-to-pt-BR-vowel-lens",
            "request_version": 1,
            "seed": 20260717,
            "speed": 1.0,
            "text": row["request"]["text"],
            "voice_id": "af_heart",
        }

    result_schema = _load("benchmark-result.schema.json")
    fixture_properties = result_schema["$defs"]["fixture_ref"]["properties"]
    assert set(fixture_properties) == {"fixture_id", "request_sha256"}
    assert result_schema["$defs"]["fixture_ref"]["additionalProperties"] is False
    forbidden_result_keys = {
        "text",
        "request",
        "script",
        "neutral_script",
        "lens_script",
        "phonemes",
        "transcript",
        "audio",
    }

    def property_keys(value: object) -> set[str]:
        if isinstance(value, dict):
            keys = set(value.get("properties", {}))
            for item in value.values():
                keys.update(property_keys(item))
            return keys
        if isinstance(value, list):
            result: set[str] = set()
            for item in value:
                result.update(property_keys(item))
            return result
        return set()

    assert property_keys(result_schema).isdisjoint(forbidden_result_keys)


def test_identity_manifest_matches_its_frozen_repository_snapshot() -> None:
    manifest = _load("identity-manifest.json")
    commit = manifest["repository_snapshot"]["git_commit"]
    for row in [*manifest["runtime_files"], *manifest["lockfiles"]]:
        payload = subprocess.run(
            ["git", "show", f"{commit}:{row['path']}"],
            cwd=ROOT,
            check=True,
            capture_output=True,
        ).stdout
        assert len(payload) == row["bytes"]
        assert hashlib.sha256(payload).hexdigest() == row["sha256"]

    model = manifest["kokoro_model"]
    assert model["bytes"] == 327_212_226
    assert model["sha256"] == MODEL_HASHES[MODEL_FILE]
    assert model["config_sha256"] == MODEL_HASHES["config.json"]

    voice = manifest["voice"]
    assert voice["voice_id"] == "af_heart"
    assert voice["sha256"] == VOICE_SPECS_BY_ID["af_heart"].sha256
    assert voice["selection_status"] == (
        "frozen_evidence_anchor_not_a_product_voice_selection"
    )

    kokoro_gate_receipt = json.loads(
        (
            ROOT
            / "artifacts"
            / "typed-engine"
            / "20260716-kokoro-gate-bridge-feasibility-v1"
            / "full-index-receipt.json"
        ).read_text(encoding="utf-8")
    )
    gates = {row["gate_id"]: row for row in manifest["gate_databases"]}
    assert (
        gates["kokoro_en_phone_gate"]["sha256"]
        == (kokoro_gate_receipt["database_sha256"])
    )
    assert (
        gates["kokoro_en_phone_gate"]["bytes"]
        == (kokoro_gate_receipt["database_bytes"])
    )
    typed_worker = (ROOT / "worker" / "typed-audio.js").read_text(encoding="utf-8")
    assert gates["written_espeak_nonce_gate"]["sha256"] in typed_worker


def test_trial_counts_order_capacity_and_overload_are_predeclared() -> None:
    protocol = _load("protocol.json")
    tiers = {row["instance_type"]: row for row in protocol["capacity_tiers"]}
    assert protocol["execution_order"] == ["standard-1", "basic"]
    assert tiers["standard-1"] == {
        "instance_type": "standard-1",
        "vcpu": 0.5,
        "memory_bytes": 4_294_967_296,
        "disk_bytes": 8_000_000_000,
        "benchmark_status": "conservative_primary_first",
        "interpretation": tiers["standard-1"]["interpretation"],
    }
    assert tiers["basic"]["benchmark_status"] == "lower_bound_experiment_second"
    assert tiers["lite"]["benchmark_status"] == "excluded_structurally_impossible"
    assert (
        _load("identity-manifest.json")["kokoro_model"]["bytes"]
        > tiers["lite"]["memory_bytes"]
    )

    design = protocol["run_design"]
    assert design["cold_start"]["trials_per_tier"] == 5
    assert design["warm_sequential"]["warmup_count"] == 1
    assert design["warm_sequential"]["measured_count"] == 10
    assert len(design["warm_sequential"]["measured_fixture_order"]) == 10
    scenarios = {row["scenario_id"]: row for row in design["concurrency"]["scenarios"]}
    assert design["concurrency"]["measured_batches_per_scenario"] == 5
    assert {key: row["n"] for key, row in scenarios.items()} == {
        "n2_identical": 2,
        "n2_mixed": 2,
        "n4_identical": 4,
        "n4_mixed": 4,
    }
    assert all(len(row["fixture_order"]) == row["n"] for row in scenarios.values())
    admission = protocol["execution_prerequisites"]["candidate_service"][
        "admission_limit"
    ]
    assert admission == {
        "active": 1,
        "waiting": 2,
        "overflow_status": 503,
        "required_header": "Retry-After",
    }
    assert "0.75" in protocol["pass_criteria"]["memory_capacity"]
    assert "0.75" in protocol["pass_criteria"]["disk_capacity"]


def test_current_host_receipt_contains_no_benchmark_measurements_or_claims() -> None:
    receipt = _load("current-host-capability-receipt.json")
    assert receipt["receipt_status"] == "host_ineligible_benchmark_not_executed"
    assert receipt["host"] == {
        "operating_system": "Darwin",
        "kernel_release": "27.0.0",
        "architecture": "arm64",
        "processor": "Apple M2 Pro",
        "required_operating_system": "linux",
        "required_architecture": "amd64",
        "platform_match": False,
    }
    assert {
        row["command"]: row["lookup"] for row in receipt["container_runtime_probes"]
    } == {
        "docker": "absent",
        "podman": "absent",
        "colima": "absent",
        "finch": "absent",
    }
    assert receipt["eligibility_checks"]["eligible"] is False
    assert receipt["execution"] == {
        "benchmark_executed": False,
        "containers_started": 0,
        "requests_sent": 0,
        "audio_renders_made": 0,
        "api_calls_made": 0,
        "network_calls_made": 0,
        "measurements_recorded": False,
        "benchmark_result_created": False,
    }
    assert set(receipt["claim_limits"].values()) == {False}


def test_protocol_keeps_candidate_disabled_and_makes_no_sla_claim() -> None:
    protocol = _load("protocol.json")
    assert set(
        value
        for key, value in protocol["claim_limits"].items()
        if key.endswith("_claimed")
    ) == {False}
    assert protocol["claim_limits"]["candidate_enabled"] is False
    assert protocol["claim_limits"]["deployment_authorized"] is False
    assert protocol["claim_limits"]["api_calls_authorized"] == 0
    environment = protocol["run_design"]["common_setup"]["environment"]
    assert all(
        value == "false"
        for key, value in environment.items()
        if key.endswith("_ENABLED")
    )
    assert protocol["execution_prerequisites"]["network_policy"].startswith(
        "Start the measured container with no network interface"
    )
