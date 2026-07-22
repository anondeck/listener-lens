from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any

from earshift_bakeoff.bilingual_candidate_runtime import (
    BilingualCandidateRuntime,
    _pcm_hash,
    _wav_bytes,
)
from earshift_bakeoff.bilingual_v8_separated_decode_composition import (
    SEPARATED_DECODE_COMPOSITION_VERSION,
    render_separated_decode_composition_candidate,
)
from earshift_bakeoff.config import ROOT, stable_json
from earshift_bakeoff.util import atomic_write_json, sha256_file


PROTOCOL_PATH = (
    ROOT / "rules" / "bilingual-v8-separated-decode-composition-v3.json"
)
BASELINE_DIR = (
    ROOT
    / "artifacts"
    / "product-matrix"
    / "20260718-bilingual-v8-factorized-composition-v2"
)
RUN_DIR = (
    ROOT
    / "artifacts"
    / "product-matrix"
    / "20260718-bilingual-v8-separated-decode-composition-v3"
)


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object in {path}")
    return value


def _semantic_hash(value: dict[str, Any]) -> str:
    payload = dict(value)
    payload.pop("record_sha256", None)
    return hashlib.sha256(stable_json(payload).encode("utf-8")).hexdigest()


def _write_bytes_once(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to replace frozen artifact: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".partial", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _validate_protocol(protocol: dict[str, Any]) -> None:
    if (
        protocol.get("schema_version") != 1
        or protocol.get("protocol_version")
        != "bilingual-v8-separated-decode-composition-v3"
        or protocol.get("production_enabled") is not False
        or protocol.get("api_calls_allowed") != 0
        or len(protocol.get("fixtures", ())) != 3
    ):
        raise ValueError("separated-decode protocol contract drifted")
    for binding in protocol["bindings"].values():
        path = ROOT / binding["path"]
        if sha256_file(path) != binding["sha256"]:
            raise ValueError(f"separated-decode binding drifted: {binding['path']}")


def _audio_receipt(path: Path, pcm: Any) -> dict[str, Any]:
    wav = _wav_bytes(pcm)
    _write_bytes_once(path, wav)
    return {
        "relative_path": str(path.relative_to(RUN_DIR)),
        "wav_sha256": hashlib.sha256(wav).hexdigest(),
        "pcm_sha256": _pcm_hash(pcm),
        "sample_count": int(pcm.size),
        "duration_s": int(pcm.size) / 24_000,
    }


def main() -> int:
    if (RUN_DIR / "results.json").exists():
        raise FileExistsError("separated-decode run already has a frozen result")
    protocol = _load_object(PROTOCOL_PATH)
    _validate_protocol(protocol)
    baseline = _load_object(BASELINE_DIR / "results.json")
    baseline_by_id = {row["fixture_id"]: row for row in baseline["fixtures"]}
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    for fixture in protocol["fixtures"]:
        baseline_row = baseline_by_id[fixture["fixture_id"]]
        runtime = BilingualCandidateRuntime.load(
            fixture["profile_id"], fixture["voice_id"]
        )
        candidate = render_separated_decode_composition_candidate(
            runtime, fixture["text"]
        )
        actual_counts = {
            cell.rule_id: sum(
                occurrence.changed and occurrence.rule_id == cell.rule_id
                for word in candidate.isolated_plan.words
                for occurrence in word.vowel_occurrences
            )
            for cell in candidate.cells
        }
        neutral_hash = _pcm_hash(candidate.render.neutral_pcm)
        baseline_neutral_hash = baseline_row["audio"]["neutral"]["pcm_sha256"]
        separated = candidate.render.prosody
        per_rule = separated["per_rule"]
        separated_contract_pass = bool(
            separated["version"] == SEPARATED_DECODE_COMPOSITION_VERSION
            and set(per_rule) == set(actual_counts)
            and separated["candidate_decoder_render_count"] == 2 + len(actual_counts)
            and all(row["windows"] for row in per_rule.values())
            and all(row["boundary_metrics_pass"] for row in per_rule.values())
        )
        contract_pass = bool(
            actual_counts == fixture["selected_rule_occurrences"]
            and list(candidate.omitted_rule_ids)
            == fixture["expected_omitted_rule_ids"]
            and candidate.isolated_plan.plan_sha256 == fixture["plan_sha256"]
            and neutral_hash == baseline_neutral_hash
            and separated_contract_pass
        )
        audio_dir = RUN_DIR / "audio"
        neutral_path = audio_dir / f"{fixture['fixture_id']}__neutral.wav"
        lens_path = audio_dir / f"{fixture['fixture_id']}__lens.wav"
        row = {
            "fixture_id": fixture["fixture_id"],
            "profile_id": fixture["profile_id"],
            "voice_id": fixture["voice_id"],
            "text": fixture["text"],
            "selected_rule_occurrences": actual_counts,
            "omitted_rule_ids": list(candidate.omitted_rule_ids),
            "plan_sha256": candidate.isolated_plan.plan_sha256,
            "baseline_v2": {
                "automatic_pass": baseline_row["automatic_pass"],
                "neutral_pcm_sha256": baseline_neutral_hash,
                "lens_pcm_sha256": baseline_row["audio"]["lens"]["pcm_sha256"],
            },
            "neutral_baseline_bit_exact": neutral_hash == baseline_neutral_hash,
            "separated_decode": separated,
            "contract_pass": contract_pass,
            "render_integrity": asdict(candidate.render.verification),
            "acoustic": candidate.acoustic,
            "audio": {
                "neutral": _audio_receipt(
                    neutral_path, candidate.render.neutral_pcm
                ),
                "lens": _audio_receipt(lens_path, candidate.render.lens_pcm),
            },
        }
        row["lens_changed_from_v2"] = bool(
            row["audio"]["lens"]["pcm_sha256"]
            != row["baseline_v2"]["lens_pcm_sha256"]
        )
        row["automatic_pass"] = bool(
            contract_pass
            and candidate.render.verification.integrity_pass
            and candidate.acoustic["pass"]
        )
        rows.append(row)
    all_pass = all(row["automatic_pass"] for row in rows)
    result = {
        "schema_version": 1,
        "run_id": RUN_DIR.name,
        "protocol_version": protocol["protocol_version"],
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "baseline_result_sha256": sha256_file(BASELINE_DIR / "results.json"),
        "classification": (
            "separated_decode_composition_pass_eligible_for_unseen_confirmation"
            if all_pass
            else "separated_decode_composition_incomplete_preserve_exact_failure"
        ),
        "production_enabled": False,
        "api_calls_made": 0,
        "fixture_count": len(rows),
        "automatic_pass_count": sum(row["automatic_pass"] for row in rows),
        "render_set_count": len(rows),
        "candidate_decoder_render_count": sum(
            row["separated_decode"]["candidate_decoder_render_count"]
            for row in rows
        ),
        "shared_natural_decoder_render_count": sum(
            row["acoustic"]["shared_natural_decoder_render_count"] for row in rows
        ),
        "elapsed_s": time.perf_counter() - started,
        "fixtures": rows,
        "interpretation_limit": protocol["outcomes"]["interpretation_limit"],
    }
    result["record_sha256"] = _semantic_hash(result)
    atomic_write_json(RUN_DIR / "results.json", result)
    print(
        json.dumps(
            {
                "classification": result["classification"],
                "automatic_pass_count": result["automatic_pass_count"],
                "fixture_count": result["fixture_count"],
                "api_calls_made": 0,
                "result_sha256": sha256_file(RUN_DIR / "results.json"),
                "record_sha256": result["record_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0 if all_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
