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
    _count_rule_occurrences,
    _pcm_hash,
    _wav_bytes,
)
from earshift_bakeoff.bilingual_product_isolation import active_changed_rule_ids
from earshift_bakeoff.bilingual_v8_adaptive_carrier import (
    ADAPTIVE_CARRIER_CANDIDATE_VERSION,
    BilingualAdaptiveCarrierRuntime,
)
from earshift_bakeoff.config import ROOT, stable_json
from earshift_bakeoff.util import atomic_write_json, sha256_file


PROTOCOL_VERSION = "bilingual-v8-carrier-retry-unseen-confirmation-v1"
PROTOCOL_PATH = ROOT / "rules" / f"{PROTOCOL_VERSION}.json"
RUN_DIR = (
    ROOT
    / "artifacts"
    / "product-matrix"
    / "20260718-bilingual-v8-carrier-retry-unseen-confirmation-v1"
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


def _validate_protocol(protocol: dict[str, Any]) -> None:
    fixtures = protocol.get("fixture_groups", [])
    if (
        protocol.get("schema_version") != 1
        or protocol.get("protocol_version") != PROTOCOL_VERSION
        or protocol.get("status") != "frozen_before_first_unseen_audio"
        or protocol.get("candidate_version")
        != ADAPTIVE_CARRIER_CANDIDATE_VERSION
        or protocol.get("api_calls_allowed") != 0
        or protocol.get("production_enabled") is not False
        or protocol.get("maximum_retry_rounds_per_fixture") != 5
        or len(fixtures) != 3
        or [row["fixture_id"] for row in fixtures]
        != [
            "heart_adaptive_unseen",
            "michael_adaptive_unseen",
            "dora_adaptive_unseen",
        ]
    ):
        raise ValueError("adaptive-carrier unseen protocol contract drifted")
    for binding in protocol["bindings"]:
        if sha256_file(ROOT / binding["path"]) != binding["sha256"]:
            raise ValueError(
                f"adaptive-carrier unseen binding drifted: {binding['path']}"
            )


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


def _plan_receipt(plan: Any) -> dict[str, Any]:
    words = [
        {
            "word_index": word.word_index,
            "source_casefold": word.source.casefold(),
            "source_phone": word.source_phone,
            "neutral_phone": word.neutral_phone,
            "lens_phone": word.lens_phone,
            "candidate_attempt": word.candidate_attempt,
        }
        for word in plan.words
    ]
    return {
        "plan_sha256": plan.plan_sha256,
        "neutral_script": plan.neutral_script,
        "lens_script": plan.lens_script,
        "gates": asdict(plan.gates),
        "word_mapping_sha256": hashlib.sha256(
            stable_json(words).encode("utf-8")
        ).hexdigest(),
        "word_candidate_attempts": [row["candidate_attempt"] for row in words],
    }


def _fixture_setup(
    runtime: BilingualCandidateRuntime, fixture: dict[str, Any]
) -> tuple[Any, tuple[Any, ...], tuple[str, ...]]:
    text = fixture["selected_fixture"]["text"]
    source_plan = runtime.base_planner.plan(text)
    changed_rule_ids = active_changed_rule_ids(source_plan)
    passing_cells = tuple(
        cell
        for rule_id in changed_rule_ids
        if (
            (cell := runtime.registry.cell(
                fixture["profile_id"], fixture["voice_id"], rule_id
            ))
            is not None
            and cell.automatic_pass
            and cell.candidate_rung == "v8"
        )
    )
    selected_rule_ids = tuple(sorted(cell.rule_id for cell in passing_cells))
    if selected_rule_ids != tuple(fixture["selected_fixture"]["selected_rule_ids"]):
        raise ValueError(f"selected rule set drifted for {fixture['fixture_id']}")
    planner = runtime._composition_planner(selected_rule_ids)
    initial_plan = planner.plan(text)
    receipt = _plan_receipt(initial_plan)
    expected = fixture["selected_fixture"]
    actual = {
        "text": text,
        "selected_rule_ids": list(selected_rule_ids),
        "selected_rule_occurrences": {
            rule_id: _count_rule_occurrences(initial_plan, rule_id)
            for rule_id in selected_rule_ids
        },
        "omitted_rule_ids": [
            rule_id for rule_id in changed_rule_ids if rule_id not in selected_rule_ids
        ],
        "initial_plan_receipt": receipt,
    }
    if actual != expected:
        raise ValueError(f"frozen first unseen plan drifted for {fixture['fixture_id']}")
    cells_by_id = {cell.rule_id: cell for cell in passing_cells}
    cells = tuple(cells_by_id[rule_id] for rule_id in selected_rule_ids)
    return planner, cells, tuple(actual["omitted_rule_ids"])


def main() -> int:
    if RUN_DIR.exists():
        raise FileExistsError(f"refusing to overwrite frozen run: {RUN_DIR}")
    protocol = _load_object(PROTOCOL_PATH)
    _validate_protocol(protocol)
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    selected_audio: list[tuple[str, Any]] = []
    for fixture in protocol["fixture_groups"]:
        runtime = BilingualCandidateRuntime.load(
            fixture["profile_id"], fixture["voice_id"]
        )
        planner, cells, omitted_rule_ids = _fixture_setup(runtime, fixture)
        row: dict[str, Any] = {
            "fixture_id": fixture["fixture_id"],
            "profile_id": fixture["profile_id"],
            "voice_id": fixture["voice_id"],
            "selected_text": fixture["selected_fixture"]["text"],
            "selected_rule_ids": fixture["selected_fixture"]["selected_rule_ids"],
            "selected_rule_occurrences": fixture["selected_fixture"][
                "selected_rule_occurrences"
            ],
            "omitted_rule_ids": list(omitted_rule_ids),
        }
        try:
            adaptive = BilingualAdaptiveCarrierRuntime(
                base_planner=planner,
                synthesis=runtime.synthesis,
                cells=cells,
                scaler=runtime.scaler,
                maximum_retry_rounds=protocol[
                    "maximum_retry_rounds_per_fixture"
                ],
            ).render(row["selected_text"])
            attempts = []
            for attempt in adaptive.attempts:
                attempts.append(
                    {
                        "round_index": attempt.round_index,
                        "retry_specs": [
                            asdict(spec) for spec in attempt.retry_specs
                        ],
                        "plan": _plan_receipt(attempt.plan),
                        "failed_mapping_keys": [
                            asdict(key) for key in attempt.failed_mapping_keys
                        ],
                        "render_integrity": asdict(attempt.render.verification),
                        "neutral_pcm_sha256": _pcm_hash(
                            attempt.render.neutral_pcm
                        ),
                        "lens_pcm_sha256": _pcm_hash(attempt.render.lens_pcm),
                        "acoustic": attempt.acoustic,
                        "automatic_pass": attempt.automatic_pass,
                    }
                )
            row.update(
                {
                    "automatic_pass": adaptive.automatic_pass,
                    "rescued_after_retry": adaptive.rescued_after_retry,
                    "selected_round_index": adaptive.selected_round_index,
                    "attempt_count": len(adaptive.attempts),
                    "failure_reason": adaptive.failure_reason,
                    "attempts": attempts,
                    "execution_error": None,
                    "selected_audio": None,
                }
            )
            if adaptive.selected_attempt is not None:
                selected_audio.append((fixture["fixture_id"], adaptive.selected_attempt))
        except Exception as exc:
            row.update(
                {
                    "automatic_pass": False,
                    "rescued_after_retry": False,
                    "selected_round_index": None,
                    "attempt_count": 0,
                    "failure_reason": "execution_error",
                    "attempts": [],
                    "execution_error": {
                        "type": type(exc).__name__,
                        "code": getattr(exc, "code", None),
                        "message": str(exc),
                    },
                    "selected_audio": None,
                }
            )
        rows.append(row)
    all_pass = all(row["automatic_pass"] for row in rows)
    rescued_count = sum(row["rescued_after_retry"] for row in rows)
    if all_pass and rescued_count:
        classification = (
            "unseen_carrier_retry_algorithm_automatic_pass_pending_human_qc"
        )
    elif all_pass:
        classification = (
            "unseen_composition_pass_carrier_retry_mechanism_unexercised"
        )
    else:
        classification = "unseen_carrier_retry_algorithm_failed_preserve_exact_result"
    RUN_DIR.mkdir(parents=True, exist_ok=False)
    for fixture_id, attempt in selected_audio:
        row = next(value for value in rows if value["fixture_id"] == fixture_id)
        row["selected_audio"] = {
            "neutral": _audio_receipt(
                RUN_DIR / "audio" / f"{fixture_id}__neutral.wav",
                attempt.render.neutral_pcm,
            ),
            "lens": _audio_receipt(
                RUN_DIR / "audio" / f"{fixture_id}__lens.wav",
                attempt.render.lens_pcm,
            ),
        }
    result = {
        "schema_version": 1,
        "run_id": RUN_DIR.name,
        "protocol_version": PROTOCOL_VERSION,
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "candidate_version": ADAPTIVE_CARRIER_CANDIDATE_VERSION,
        "classification": classification,
        "production_enabled": False,
        "api_calls_made": 0,
        "fixture_count": len(rows),
        "automatic_pass_count": sum(row["automatic_pass"] for row in rows),
        "rescued_fixture_count": rescued_count,
        "total_attempt_count": sum(row["attempt_count"] for row in rows),
        "human_review_generated": False,
        "fixtures": rows,
        "elapsed_s": time.perf_counter() - started,
        "interpretation_limit": protocol["outcomes"]["interpretation_limit"],
    }
    result["record_sha256"] = _semantic_hash(result)
    atomic_write_json(RUN_DIR / "results.json", result)
    print(
        json.dumps(
            {
                "classification": classification,
                "automatic_pass_count": result["automatic_pass_count"],
                "fixture_count": result["fixture_count"],
                "rescued_fixture_count": rescued_count,
                "total_attempt_count": result["total_attempt_count"],
                "api_calls_made": 0,
                "result_sha256": sha256_file(RUN_DIR / "results.json"),
                "record_sha256": result["record_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0 if all_pass and rescued_count else 2


if __name__ == "__main__":
    raise SystemExit(main())
