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
    evaluate_current_context_composition_acoustics,
)
from earshift_bakeoff.bilingual_listener_engine_v8 import BilingualListenerRuntimeV8
from earshift_bakeoff.bilingual_v8_carrier_retry import (
    CARRIER_RETRY_CANDIDATE_VERSION,
    BilingualListenerPlannerV8CarrierRetry,
    CarrierRetrySpec,
)
from earshift_bakeoff.bilingual_vowel_engine import BilingualVowelRender
from earshift_bakeoff.config import ROOT, stable_json
from earshift_bakeoff.util import atomic_write_json, sha256_file


PROTOCOL_VERSION = "bilingual-v8-carrier-retry-correction-v1"
PROTOCOL_PATH = ROOT / "rules" / f"{PROTOCOL_VERSION}.json"
RUN_DIR = (
    ROOT
    / "artifacts"
    / "product-matrix"
    / "20260718-bilingual-v8-carrier-retry-correction-v1"
)
PARENT_RESULT_PATH = (
    ROOT
    / "artifacts"
    / "product-matrix"
    / "20260718-bilingual-v8-composition-unseen-confirmation-v2"
    / "results.json"
)
STRENGTH_RESULT_PATH = (
    ROOT
    / "artifacts"
    / "product-matrix"
    / "20260718-bilingual-v8-occurrence-strength-correction-v1"
    / "results.json"
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
    candidates = protocol.get("candidate_plans", [])
    if (
        protocol.get("schema_version") != 1
        or protocol.get("protocol_version") != PROTOCOL_VERSION
        or protocol.get("status") != "frozen_before_first_candidate_audio"
        or protocol.get("api_calls_allowed") != 0
        or protocol.get("production_enabled") is not False
        or protocol.get("candidate_version") != CARRIER_RETRY_CANDIDATE_VERSION
        or len(candidates) != 5
        or [row["round"] for row in candidates] != [1, 2, 3, 4, 5]
        or [row["minimum_attempt"] for row in candidates] != [4, 5, 6, 7, 8]
        or [row["selected_attempt"] for row in candidates] != [4, 5, 6, 7, 9]
        or protocol["selection_rule"]
        != "first_complete_composition_gate_pass_under_frozen_plan_order"
    ):
        raise ValueError("carrier-retry protocol contract drifted")
    for binding in protocol["bindings"]:
        if sha256_file(ROOT / binding["path"]) != binding["sha256"]:
            raise ValueError(f"carrier-retry binding drifted: {binding['path']}")


def _heart_parent(parent: dict[str, Any]) -> dict[str, Any]:
    if (
        sha256_file(PARENT_RESULT_PATH)
        != "6afcd959d5fc95c2668d2232ce3b461db185c83b0f4d238b3969367ba84cf2a3"
        or parent["record_sha256"]
        != "4ee292c546702f4cb016eb8248f57c97b08fbd997561ebb47c89a45543c72a3b"
    ):
        raise ValueError("frozen v2 parent drifted")
    return next(
        row
        for row in parent["fixtures"]
        if row["fixture_id"] == "heart_unseen_continuous"
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
    retried = words[3]
    return {
        "plan_sha256": plan.plan_sha256,
        "neutral_script": plan.neutral_script,
        "lens_script": plan.lens_script,
        "gates": asdict(plan.gates),
        "word_mapping_sha256": hashlib.sha256(
            stable_json(words).encode("utf-8")
        ).hexdigest(),
        "word_candidate_attempts": [row["candidate_attempt"] for row in words],
        "retried_word": retried,
    }


def main() -> int:
    if RUN_DIR.exists():
        raise FileExistsError(f"refusing to overwrite frozen run: {RUN_DIR}")
    protocol = _load_object(PROTOCOL_PATH)
    _validate_protocol(protocol)
    parent = _load_object(PARENT_RESULT_PATH)
    heart = _heart_parent(parent)
    strength = _load_object(STRENGTH_RESULT_PATH)
    if (
        strength["classification"]
        != "known_failure_occurrence_correction_failed_preserve_parent_failure"
        or strength["selected_strength"] is not None
    ):
        raise ValueError("scalar-strength parent no longer records a closed failure")
    runtime = BilingualCandidateRuntime.load(
        heart["profile_id"], heart["voice_id"]
    )
    rule_ids = tuple(sorted(heart["selected_rule_occurrences"]))
    base_planner = runtime._composition_planner(rule_ids)
    cells = tuple(
        runtime.registry.cell(heart["profile_id"], heart["voice_id"], rule_id)
        for rule_id in rule_ids
    )
    if any(cell is None for cell in cells):
        raise ValueError("a frozen Heart cell disappeared")
    typed_cells = tuple(cell for cell in cells if cell is not None)
    attempts: list[dict[str, Any]] = []
    selected_render: BilingualVowelRender | None = None
    selected_round: int | None = None
    started = time.perf_counter()
    for candidate in protocol["candidate_plans"]:
        attempt_started = time.perf_counter()
        retry_spec = CarrierRetrySpec(
            source_casefold=protocol["retry_mapping"]["source_casefold"],
            source_phone=protocol["retry_mapping"]["source_phone"],
            carrier_role=protocol["retry_mapping"]["carrier_role"],
            minimum_attempt=candidate["minimum_attempt"],
        )
        planner = BilingualListenerPlannerV8CarrierRetry.from_planner(
            base_planner, retry_specs=(retry_spec,)
        )
        plan = planner.plan(protocol["fixture"]["text"])
        receipt = _plan_receipt(plan)
        if receipt != candidate["receipt"]:
            raise ValueError(
                f"frozen retry plan drifted in round {candidate['round']}"
            )
        render = BilingualListenerRuntimeV8(
            planner=planner, synthesis=runtime.synthesis
        ).render(protocol["fixture"]["text"])
        if not isinstance(render, BilingualVowelRender):
            raise ValueError("carrier retry produced no controlled pair")
        acoustic = evaluate_current_context_composition_acoustics(
            cells=typed_cells,
            render=render,
            synthesis=runtime.synthesis,
            scaler=runtime.scaler,
        )
        automatic_pass = bool(
            render.verification.integrity_pass and acoustic["pass"]
        )
        attempts.append(
            {
                "round": candidate["round"],
                "minimum_attempt": candidate["minimum_attempt"],
                "selected_attempt": candidate["selected_attempt"],
                "plan": receipt,
                "render_integrity": asdict(render.verification),
                "neutral_pcm_sha256": _pcm_hash(render.neutral_pcm),
                "lens_pcm_sha256": _pcm_hash(render.lens_pcm),
                "acoustic": acoustic,
                "automatic_pass": automatic_pass,
                "elapsed_s": time.perf_counter() - attempt_started,
            }
        )
        if automatic_pass:
            selected_render = render
            selected_round = candidate["round"]
            break
    correction_pass = selected_render is not None
    RUN_DIR.mkdir(parents=True, exist_ok=False)
    audio = None
    if selected_render is not None:
        audio = {
            "neutral": _audio_receipt(
                RUN_DIR / "audio" / "heart_carrier_retry__neutral.wav",
                selected_render.neutral_pcm,
            ),
            "lens": _audio_receipt(
                RUN_DIR / "audio" / "heart_carrier_retry__lens.wav",
                selected_render.lens_pcm,
            ),
        }
    result = {
        "schema_version": 1,
        "run_id": RUN_DIR.name,
        "protocol_version": PROTOCOL_VERSION,
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "candidate_version": CARRIER_RETRY_CANDIDATE_VERSION,
        "classification": (
            "known_failure_carrier_retry_pass_eligible_fresh_unseen_confirmation"
            if correction_pass
            else "known_failure_carrier_retry_failed_preserve_parent_failure"
        ),
        "production_enabled": False,
        "api_calls_made": 0,
        "parent_result_sha256": sha256_file(PARENT_RESULT_PATH),
        "parent_result_record_sha256": parent["record_sha256"],
        "strength_result_sha256": sha256_file(STRENGTH_RESULT_PATH),
        "strength_result_record_sha256": strength["record_sha256"],
        "attempt_count": len(attempts),
        "attempted_rounds": [row["round"] for row in attempts],
        "selected_round": selected_round,
        "selected_audio": audio,
        "attempts": attempts,
        "human_review_generated": False,
        "fresh_unseen_confirmation_required": correction_pass,
        "elapsed_s": time.perf_counter() - started,
        "interpretation_limit": protocol["outcomes"]["interpretation_limit"],
    }
    result["record_sha256"] = _semantic_hash(result)
    atomic_write_json(RUN_DIR / "results.json", result)
    print(
        json.dumps(
            {
                "classification": result["classification"],
                "attempted_rounds": result["attempted_rounds"],
                "selected_round": result["selected_round"],
                "api_calls_made": 0,
                "result_sha256": sha256_file(RUN_DIR / "results.json"),
                "record_sha256": result["record_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0 if correction_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
