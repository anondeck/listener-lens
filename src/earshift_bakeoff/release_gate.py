from __future__ import annotations

import hashlib
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .bilingual_product_matrix import (
    load_bilingual_product_matrix,
    load_bilingual_structural_state,
)
from .bilingual_product_audio_state import (
    load_bilingual_audio_integrity_state,
    load_bilingual_isolated_audio_state,
)
from .config import ROOT, load_json_yaml


@dataclass(frozen=True)
class ReleaseCheck:
    name: str
    status: str
    detail: str


REQUIRED_RECORDS = (
    "LICENSE",
    "DESIGN.md",
    "deployment-state.json",
)
VISIBLE_RECORDS = (
    "README.md",
    "DESIGN.md",
    "prototype/listener-lens/index.html",
    "site/index.html",
)
SHIPPING_FILES = (
    ".dockerignore",
    "site/index.html",
    "site/styles.css",
    "site/app.js",
    "site/audio/neutral-carrier.wav",
    "site/audio/altered-carrier.wav",
    "worker/index.js",
    "worker/app.js",
    "worker/typed-audio.js",
    "worker/request-utils.js",
    "worker/transform-container.js",
    "rules/kokoro-candidate-state.json",
    "rules/bilingual-kokoro-candidate-state-v1.json",
    "rules/bilingual-kokoro-composition-candidate-v1.json",
    "rules/bilingual-kokoro-composition-candidate-v2.json",
    "rules/bilingual-kokoro-composition-candidate-v3.json",
    "rules/bilingual-rule-display-v1.json",
    "rules/bilingual-product-matrix-v1.json",
    "rules/bilingual-product-structural-state-v1.json",
    "rules/bilingual-product-audio-integrity-state-v1.json",
    "rules/bilingual-product-isolated-audio-state-v1.json",
    "src/earshift_bakeoff/bilingual_product_matrix.py",
    "src/earshift_bakeoff/bilingual_product_engine.py",
    "src/earshift_bakeoff/bilingual_candidate_runtime.py",
    "src/earshift_bakeoff/bilingual_composed_candidate_runtime.py",
    "src/earshift_bakeoff/bilingual_composed_candidate_runtime_v2.py",
    "src/earshift_bakeoff/bilingual_composed_candidate_runtime_v3.py",
    "src/earshift_bakeoff/bilingual_v8_adaptive_carrier.py",
    "src/earshift_bakeoff/bilingual_v8_carrier_retry.py",
    "src/earshift_bakeoff/azure_lens_builder.py",
    "rules/azure-ipa-map-v1.json",
    "worker/azure-lens.js",
    "src/earshift_bakeoff/deploy_service.py",
    "src/earshift_bakeoff/kokoro_candidate_service.py",
    "container/Dockerfile",
    "container/requirements.in",
    "container/requirements.txt",
    "wrangler.jsonc",
    "wrangler.runtime-v2.jsonc",
)
CANDIDATE_FEATURE_FLAGS = (
    "KOKORO_ENGLISH_CANDIDATE_ENABLED",
    "KOKORO_BILINGUAL_CANDIDATE_ENABLED",
    "PORTUGUESE_RENDERER_CANDIDATE_ENABLED",
    "RECIPROCAL_LISTENER_LENS_RESEARCH_ENABLED",
    "AZURE_LENS_CANDIDATE_ENABLED",
    "GIBBERISH_CANDIDATE_ENABLED",
)
LOCAL_TOGGLEABLE_CANDIDATE_FLAGS = (
    "KOKORO_ENGLISH_CANDIDATE_ENABLED",
    "KOKORO_BILINGUAL_CANDIDATE_ENABLED",
)
SHIPPING_AUDIO_HASHES = {
    "site/audio/neutral-carrier.wav": "1052138ac9e9829e28089718bd856a3dd72c63f176241a716ebc743b22613f54",
    "site/audio/altered-carrier.wav": "43b4355211d4702523ab7932cd5bd47ef4cbb2f2c8667d2008f5f188ae2944f8",
}
FORBIDDEN_TRACKED_PARTS = (
    ".cache/",
    "artifacts/",
    "output/",
    "tmp/",
    "/g2p_reference/",
    "/ratings.csv",
    "/controls/",
)


def _tracked_files(root: Path) -> list[str]:
    proc = subprocess.run(
        ["git", "ls-files"], cwd=root, capture_output=True, text=True, check=True
    )
    return [line for line in proc.stdout.splitlines() if line]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dotenv_assignments(path: Path) -> dict[str, list[str]]:
    assignments: dict[str, list[str]] = {}
    if not path.is_file():
        return assignments
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        assignments.setdefault(key.strip(), []).append(value.strip())
    return assignments


def candidate_flag_configuration_errors(root: Path = ROOT) -> list[str]:
    """Validate disabled deploy defaults and the one allowed local candidate."""

    errors: list[str] = []
    try:
        project_flags = load_json_yaml(root / "bakeoff.yaml").get(
            "candidate_features", {}
        )
    except Exception as exc:  # pragma: no cover - reported as a release failure
        project_flags = {}
        errors.append(f"bakeoff.yaml unreadable: {type(exc).__name__}")
    try:
        wrangler_flags = load_json_yaml(root / "wrangler.runtime-v2.jsonc").get(
            "vars", {}
        )
    except Exception as exc:  # pragma: no cover - reported as a release failure
        wrangler_flags = {}
        errors.append(f"wrangler.runtime-v2.jsonc unreadable: {type(exc).__name__}")
    dotenv_flags = _dotenv_assignments(root / ".env.example")

    for flag in CANDIDATE_FEATURE_FLAGS:
        if project_flags.get(flag) is not False:
            errors.append(f"bakeoff.yaml {flag} is not boolean false")
        if wrangler_flags.get(flag) != "false":
            errors.append(
                f"wrangler.runtime-v2.jsonc {flag} is not exact string false"
            )
        if dotenv_flags.get(flag) != ["false"]:
            errors.append(f".env.example {flag} is not one exact false assignment")

    worker_path = root / "worker" / "typed-audio.js"
    local_path = root / "scripts" / "dev-local.mjs"
    worker = worker_path.read_text(encoding="utf-8") if worker_path.is_file() else ""
    local = local_path.read_text(encoding="utf-8") if local_path.is_file() else ""
    if (
        "candidateFlagsAreExactlyFalse" not in worker
        or "env?.[flag] === 'false'" not in worker
    ):
        errors.append("Worker exact-false preflight is missing")
    if (
        any(
            f"['false', 'true'].includes(candidateFlagEnv.{flag})" not in local
            for flag in LOCAL_TOGGLEABLE_CANDIDATE_FLAGS
        )
        or "kokoroCandidateEnabled && bilingualCandidateEnabled" not in local
        or "candidateFlagEnv[name] !== 'false'" not in local
    ):
        errors.append("local single-candidate startup preflight is missing")
    for flag in CANDIDATE_FEATURE_FLAGS:
        if flag not in worker:
            errors.append(f"Worker does not name {flag}")
        if flag not in local:
            errors.append(f"local runtime does not name {flag}")
    return errors


def collect_release_checks(
    *, root: Path = ROOT, run_id: str = "20260714-renderer-bakeoff"
) -> list[ReleaseCheck]:
    checks: list[ReleaseCheck] = []
    missing = [name for name in REQUIRED_RECORDS if not (root / name).is_file()]
    checks.append(
        ReleaseCheck(
            "release-records",
            "PASS" if not missing else "FAIL",
            "present" if not missing else "missing: " + ", ".join(missing),
        )
    )

    notices_path = root / "THIRD_PARTY_NOTICES.md"
    notices = notices_path.read_text(encoding="utf-8") if notices_path.is_file() else ""
    required_components = (
        "huggingface-hub",
        "mlx-whisper",
        "numpy",
        "openai",
        "pydantic",
        "praat-parselmouth",
        "wordfreq",
        "eSpeak-NG",
        "FFmpeg",
        "whisper-large-v3-mlx",
        "Wrangler",
        "@cloudflare/containers",
        "Kokoro inference package",
        "Misaki",
        "spaCy",
        "PyTorch",
        "Python",
    )
    absent_components = [name for name in required_components if name not in notices]
    checks.append(
        ReleaseCheck(
            "third-party-inventory",
            "PASS" if not absent_components else "FAIL",
            "direct tools inventoried"
            if not absent_components
            else "missing: " + ", ".join(absent_components),
        )
    )

    visible_hits: list[str] = []
    working_name = "ear" + "shift"
    for relative in VISIBLE_RECORDS:
        path = root / relative
        if (
            path.is_file()
            and working_name in path.read_text(encoding="utf-8").casefold()
        ):
            visible_hits.append(relative)
    checks.append(
        ReleaseCheck(
            "working-name-boundary",
            "PASS" if not visible_hits else "FAIL",
            "no placeholder name in user-facing records"
            if not visible_hits
            else "visible in: " + ", ".join(visible_hits),
        )
    )

    tracked = _tracked_files(root)
    forbidden = []
    for path in tracked:
        secret_env = path == ".env" or (
            path.startswith(".env.") and path != ".env.example"
        )
        local_artifact = any(
            part in path or path.startswith(part) for part in FORBIDDEN_TRACKED_PARTS
        )
        if secret_env or local_artifact:
            forbidden.append(path)
    checks.append(
        ReleaseCheck(
            "sensitive-and-local-files",
            "PASS" if not forbidden else "FAIL",
            "no secrets, caches, ratings, controls, or report-only references tracked"
            if not forbidden
            else "tracked: " + ", ".join(forbidden),
        )
    )

    lens_rules = load_json_yaml(root / "rules" / "listener_lenses.yaml")
    rule_errors: list[str] = []
    for profile in lens_rules["profiles"]:
        source_ids = {
            source["id"] for source in profile["sources"] if source.get("url")
        }
        for rule in profile["transformations"]:
            if rule.get("kind") != "derived_engineering_rule":
                rule_errors.append(f"{rule.get('id')}: missing derived label")
            if not rule.get("confidence"):
                rule_errors.append(f"{rule.get('id')}: missing confidence")
            if not set(rule.get("source_ids", [])).issubset(source_ids):
                rule_errors.append(f"{rule.get('id')}: unresolved source")
    checks.append(
        ReleaseCheck(
            "listener-rule-provenance",
            "PASS" if not rule_errors else "FAIL",
            "all experimental rules labeled, confidence-scored, and sourced"
            if not rule_errors
            else "; ".join(rule_errors),
        )
    )

    site_index_path = root / "site" / "index.html"
    site_index = (
        site_index_path.read_text(encoding="utf-8") if site_index_path.is_file() else ""
    )
    current_product_surface = all(
        phrase in site_index
        for phrase in (
            "Listener lens",
            "Sound minus meaning",
            "Production comparison · not part of A/B evidence",
            "Open evidence receipt",
        )
    )
    checks.append(
        ReleaseCheck(
            "shipping-audio-mode",
            "PASS" if current_product_surface else "FAIL",
            "listener lens, sound-minus-meaning, production comparison, and per-request receipt remain distinct"
            if current_product_surface
            else "one or more current listening surfaces are missing",
        )
    )

    missing_shipping = [name for name in SHIPPING_FILES if not (root / name).is_file()]
    mismatched_audio = [
        name
        for name, expected in SHIPPING_AUDIO_HASHES.items()
        if (root / name).is_file() and _sha256(root / name) != expected
    ]
    shipping_errors = [
        *(f"missing {name}" for name in missing_shipping),
        *(f"hash {name}" for name in mismatched_audio),
    ]
    checks.append(
        ReleaseCheck(
            "shipping-bundle",
            "PASS" if not shipping_errors else "FAIL",
            "site, Worker, and frozen audio hashes present"
            if not shipping_errors
            else "; ".join(shipping_errors),
        )
    )

    matrix_errors: list[str] = []
    try:
        matrix = load_bilingual_product_matrix()
        structural = load_bilingual_structural_state(matrix)
        audio_integrity = load_bilingual_audio_integrity_state(
            matrix_version=matrix.matrix_version,
            matrix_sha256=matrix.matrix_sha256,
        )
        isolated_audio = load_bilingual_isolated_audio_state(
            matrix_version=matrix.matrix_version,
            matrix_sha256=matrix.matrix_sha256,
        )
        catalog = matrix.safe_catalog()
        if (
            catalog["rule_cell_count"] != 166
            or catalog["changed_rule_cell_count"] != 98
            or catalog["product_enabled_cell_count"] != 0
            or structural["planner_slot_count"] != 280
            or structural["planner_pass_count"] != 280
            or structural["audio_validation_status"] != "pending"
            or structural["production_enabled"] is not False
            or audio_integrity["slot_count"] != 98
            or audio_integrity["universal_integrity_pass_count"] != 98
            or audio_integrity["universal_integrity_fail_count"] != 0
            or audio_integrity["family_acoustic_validation_status"] != "pending"
            or audio_integrity["production_enabled"] is not False
            or isolated_audio["slot_count"] != 280
            or isolated_audio["isolated_universal_integrity_pass_count"] != 280
            or isolated_audio["isolated_universal_integrity_fail_count"] != 0
            or isolated_audio["family_acoustic_validation_status"] != "pending"
            or isolated_audio["production_enabled"] is not False
        ):
            matrix_errors.append("matrix counts or disabled evidence state drifted")
    except Exception as exc:  # pragma: no cover - reported as a release failure
        matrix_errors.append(f"{type(exc).__name__}: {exc}")
    checks.append(
        ReleaseCheck(
            "bilingual-product-matrix",
            "PASS" if not matrix_errors else "FAIL",
            "four voices, both directions, 280/280 structural slots, and 280/280 atomically isolated audio-integrity slots bound; family acoustics and product cells remain pending"
            if not matrix_errors
            else "; ".join(matrix_errors),
        )
    )

    candidate_flag_errors = candidate_flag_configuration_errors(root)
    checks.append(
        ReleaseCheck(
            "disabled-candidate-flags",
            "PASS" if not candidate_flag_errors else "FAIL",
            "deploy defaults are exact-false; local startup permits only the bounded Kokoro candidate"
            if not candidate_flag_errors
            else "; ".join(candidate_flag_errors),
        )
    )

    # Durable claim-boundary anchors: the runtime approximation, the separately
    # frozen automatic evidence, the private-experience disclaimer, and the
    # disabled-by-default posture must all stay labelled. These survive the
    # Azure renderer pivot; renderer-specific mechanism strings do not belong
    # in a claim-honesty gate.
    concept_boundary = all(
        phrase in site_index
        for phrase in (
            "Research-informed approximation—not private perception.",
            "C uses the listener-language voice and is separate from the A/B perception comparison.",
            "Only rules that match this sentence and remain audible in the selected voice appear in B.",
        )
    )
    checks.append(
        ReleaseCheck(
            "concept-evidence-boundary",
            "PASS" if concept_boundary else "FAIL",
            "the concise approximation boundary, production boundary, and per-sentence rule boundary are present"
            if concept_boundary
            else "required claim-boundary labels are missing",
        )
    )

    worker_paths = (
        root / "worker" / "index.js",
        root / "worker" / "app.js",
        root / "worker" / "typed-audio.js",
    )
    worker = "\n".join(
        path.read_text(encoding="utf-8") for path in worker_paths if path.is_file()
    )
    worker_ready = all(
        phrase in worker
        for phrase in (
            "/openai/v1/responses",
            "AZURE_FOUNDRY_ENDPOINT",
            "AZURE_FOUNDRY_API_KEY",
            "AZURE_FOUNDRY_DEPLOYMENT",
            "luna",
            "json_schema",
            "cached_fallback",
            "store: false",
        )
    ) and not re.search(r"sk-[A-Za-z0-9_-]{12,}", worker)
    checks.append(
        ReleaseCheck(
            "protected-activity-worker",
            "PASS" if worker_ready else "FAIL",
            "bounded Azure Foundry Responses call, secret binding, Luna deployment, and fallback present"
            if worker_ready
            else "Foundry Worker contract, secret boundary, or fallback is incomplete",
        )
    )

    deploy_service_path = root / "src" / "earshift_bakeoff" / "deploy_service.py"
    candidate_service_path = (
        root / "src" / "earshift_bakeoff" / "kokoro_candidate_service.py"
    )
    composed_candidate_path = (
        root
        / "src"
        / "earshift_bakeoff"
        / "bilingual_composed_candidate_runtime_v3.py"
    )
    candidate_state_path = root / "rules" / "kokoro-candidate-state.json"
    composition_state_path = (
        root / "rules" / "bilingual-kokoro-composition-candidate-v3.json"
    )
    product_voice_registry_path = root / "rules" / "kokoro-product-voices.json"
    typed_runtime = (
        worker
        + "\n"
        + (
            deploy_service_path.read_text(encoding="utf-8")
            if deploy_service_path.is_file()
            else ""
        )
        + "\n"
        + (
            candidate_service_path.read_text(encoding="utf-8")
            if candidate_service_path.is_file()
            else ""
        )
        + "\n"
        + (
            candidate_state_path.read_text(encoding="utf-8")
            if candidate_state_path.is_file()
            else ""
        )
        + "\n"
        + (
            composed_candidate_path.read_text(encoding="utf-8")
            if composed_candidate_path.is_file()
            else ""
        )
        + "\n"
        + (
            composition_state_path.read_text(encoding="utf-8")
            if composition_state_path.is_file()
            else ""
        )
        + "\n"
        + (
            product_voice_registry_path.read_text(encoding="utf-8")
            if product_voice_registry_path.is_file()
            else ""
        )
    )
    typed_runtime_ready = all(
        phrase in typed_runtime
        for phrase in (
            "/api/listener-lens",
            "/kokoro-listener-lens",
            "/bilingual-kokoro-listener-lens",
            "/api/voices",
            "kokoro-controlled-pair-service-v2",
            "bilingual-kokoro-controlled-pair-service-v4",
            "kokoro-product-voices-v1",
            "am_michael",
            "pm_alex",
            "pf_dora",
            "output-domain-splice-v1",
            "automatic_gates_passed",
            "primary_50_acoustic_gate",
            "localization_fail_closed",
            "KOKORO_ENGLISH_CANDIDATE_ENABLED",
            "KOKORO_BILINGUAL_CANDIDATE_ENABLED",
            "multi_rule_v8",
            "ready_automatic_only",
            "adaptive_algorithm_automatic_pass_3_of_3_two_rescues",
            "v8-adaptive-carrier-v1",
            "maximum_carrier_retry_rounds",
            "failed_or_exhausted_contexts_fail_closed",
            "TYPED_AUDIO_SERVE_ENABLED",
            "TYPED_AUDIO_RENDER_ENABLED",
            "handleLegacyTypedAudio",
        )
    ) and all(
        (root / path).is_file()
        for path in (
            "src/earshift_bakeoff/deploy_service.py",
            "src/earshift_bakeoff/kokoro_candidate_service.py",
            "rules/kokoro-candidate-state.json",
            "rules/kokoro-product-voices.json",
            "rules/bilingual-rule-display-v1.json",
            "container/Dockerfile",
            "container/requirements.txt",
        )
    )
    checks.append(
        ReleaseCheck(
            "typed-runtime-contract",
            "PASS" if typed_runtime_ready else "FAIL",
            "controlled Kokoro candidate is locally integrated and fail-closed behind a disabled production flag"
            if typed_runtime_ready
            else "typed production runtime contract is incomplete",
        )
    )

    site_app_path = root / "site" / "app.js"
    site_app = (
        site_app_path.read_text(encoding="utf-8") if site_app_path.is_file() else ""
    )
    dockerignore_path = root / ".dockerignore"
    dockerignore = (
        dockerignore_path.read_text(encoding="utf-8")
        if dockerignore_path.is_file()
        else ""
    )
    deploy_service = (
        deploy_service_path.read_text(encoding="utf-8")
        if deploy_service_path.is_file()
        else ""
    )
    runtime_security_ready = (
        ".innerHTML" not in site_app
        and "MAX_ADMITTED_INFERENCE_REQUESTS" in deploy_service
        and "SOCKET_READ_TIMEOUT_S" in deploy_service
        and "unsupported_transfer_encoding" in deploy_service
        and '"message": format % args' not in deploy_service
        and "detail: result.body.detail" not in worker
        and dockerignore.startswith("**\n")
        and all(
            item in dockerignore
            for item in (
                "!bakeoff.yaml",
                "!container/Dockerfile",
                "!container/requirements.txt",
                "!rules/**",
                "!src/**",
            )
        )
    )
    checks.append(
        ReleaseCheck(
            "runtime-security-and-privacy",
            "PASS" if runtime_security_ready else "FAIL",
            "bounded admission/read limits, sanitized logs/errors, safe DOM rendering, and allowlisted container context present"
            if runtime_security_ready
            else "runtime hardening or privacy assertions are incomplete",
        )
    )

    ratings = root / "artifacts" / "runs" / run_id / "ratings.csv"
    checks.append(
        ReleaseCheck(
            "blind-ratings",
            "PASS" if ratings.is_file() else "WAIT",
            str(ratings) if ratings.is_file() else "ratings.csv not yet exported",
        )
    )

    site = root / "site"
    disclosure_found = False
    if site.is_dir():
        for path in site.rglob("*"):
            if path.suffix.lower() not in {".html", ".js", ".jsx", ".ts", ".tsx"}:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore").casefold()
            if "ai-generated" in text and (
                "not a human voice" in text or "not human recordings" in text
            ):
                disclosure_found = True
                break
    checks.append(
        ReleaseCheck(
            "shipping-ai-voice-disclosure",
            "PASS" if disclosure_found else "WAIT",
            "found in deployable site"
            if disclosure_found
            else "final site not present or disclosure not yet implemented",
        )
    )

    deployment_state_path = root / "deployment-state.json"
    deployment_status = "FAIL"
    deployment_detail = "machine-readable deployment state missing or invalid"
    try:
        deployment_state = json.loads(deployment_state_path.read_text(encoding="utf-8"))
        if set(deployment_state) != {
            "schema_version",
            "active",
            "url",
            "last_changed",
            "state",
        }:
            raise ValueError("unexpected deployment-state fields")
        if deployment_state["schema_version"] != 1 or not isinstance(
            deployment_state["active"], bool
        ):
            raise ValueError("invalid deployment-state schema")
        if deployment_state["active"] is False:
            deployment_status = "WAIT"
            deployment_detail = f"inactive ({deployment_state['state']}); historical URL is not release evidence"
        elif re.fullmatch(r"https://[^\s/]+\.workers\.dev/?", deployment_state["url"]):
            deployment_status = "PASS"
            deployment_detail = deployment_state["url"]
        else:
            deployment_detail = "active deployment state lacks a valid Workers URL"
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError, TypeError):
        pass
    checks.append(
        ReleaseCheck(
            "deployment-record",
            deployment_status,
            deployment_detail,
        )
    )
    return checks


def release_check(*, strict: bool = False) -> int:
    checks = collect_release_checks()
    for check in checks:
        print(f"{check.status:<4}  {check.name}: {check.detail}")
    hard_failure = any(check.status == "FAIL" for check in checks)
    incomplete = any(check.status == "WAIT" for check in checks)
    if hard_failure or (strict and incomplete):
        return 4
    return 0
