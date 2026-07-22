#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Any
import wave

import numpy as np

from earshift_bakeoff.bilingual_listener_engine import (
    BilingualListenerPlanner,
    BilingualListenerRuntime,
    load_listener_profiles,
)
from earshift_bakeoff.bilingual_product_isolation import (
    ISOLATED_VALIDATION_PROFILE_VERSION,
    active_changed_rule_ids,
    isolate_listener_profile,
)
from earshift_bakeoff.bilingual_product_matrix import (
    load_bilingual_product_matrix,
    load_bilingual_structural_state,
)
from earshift_bakeoff.bilingual_vowel_engine import (
    BilingualVowelRender,
    _load_pinned_synthesis_voice,
)
from earshift_bakeoff.config import Paths, stable_json
from earshift_bakeoff.kokoro_gate_bridge import KokoroGateIndex
from earshift_bakeoff.kokoro_synthesis import (
    CONFIG_FILE,
    SAMPLE_RATE_HZ,
    verify_model_files,
)
from earshift_bakeoff.listener_lens import DatabaseNonceChecker
from earshift_bakeoff.portuguese_carrier_planner_v1 import (
    PortuguesePositiveOnlyIndexV1,
)
from earshift_bakeoff.util import atomic_write_json, sha256_file

from run_bilingual_product_audio_integrity_screen_v1 import (
    FixtureAdapter,
    _safe_name,
    _target_rows,
    _tuple,
    _universal_pass,
)


PROTOCOL_VERSION = "bilingual-product-isolated-audio-screen-v1"
PROTOCOL_PATH = Paths().root / "rules" / f"{PROTOCOL_VERSION}.json"
RUN_ID = "20260717-bilingual-product-isolated-audio-screen-v1"
RUN_DIR = Paths().artifacts / "product-matrix" / RUN_ID
MANIFEST_PATH = (
    Paths().artifacts
    / "product-matrix"
    / "20260717-bilingual-product-isolated-acoustic-manifest-v1"
    / "manifest.json"
)


def _semantic_hash(payload: Any) -> str:
    return hashlib.sha256(stable_json(payload).encode("utf-8")).hexdigest()


def _write_wav(path: Path, pcm: np.ndarray) -> dict[str, Any]:
    values = np.asarray(pcm, dtype="<i2").reshape(-1)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with wave.open(str(temporary), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE_HZ)
        handle.writeframes(values.tobytes())
    temporary.replace(path)
    return {
        "relative_path": str(path.relative_to(RUN_DIR)),
        "wav_sha256": sha256_file(path),
        "pcm_sha256": hashlib.sha256(values.tobytes()).hexdigest(),
        "sample_count": int(values.size),
        "duration_s": values.size / SAMPLE_RATE_HZ,
    }


def _load_protocol(matrix_sha256: str, manifest: dict[str, Any]) -> dict[str, Any]:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    expected_keys = {
        "schema_version",
        "protocol_version",
        "status",
        "production_enabled",
        "matrix_binding",
        "isolated_manifest_binding",
        "renderer_policy",
        "universal_gates",
        "classification_policy",
        "source_bindings",
    }
    if set(protocol) != expected_keys:
        raise RuntimeError("isolated audio protocol schema drifted")
    matrix_binding = protocol["matrix_binding"]
    manifest_binding = protocol["isolated_manifest_binding"]
    if (
        protocol["schema_version"] != 1
        or protocol["protocol_version"] != PROTOCOL_VERSION
        or protocol["status"] != "frozen_before_first_isolated_audio_render"
        or protocol["production_enabled"] is not False
        or matrix_binding["matrix_sha256"] != matrix_sha256
        or manifest_binding["path"]
        != str(MANIFEST_PATH.relative_to(Paths().root))
        or manifest_binding["sha256"] != sha256_file(MANIFEST_PATH)
        or manifest_binding["record_sha256"] != manifest["record_sha256"]
        or manifest_binding["logical_slot_count"] != 280
        or protocol["renderer_policy"]["api_calls_allowed"] != 0
        or protocol["renderer_policy"]["render_every_slot_once"] is not True
        or protocol["renderer_policy"]["replacement_slots_allowed"] is not False
        or protocol["classification_policy"]["product_promotion_allowed"]
        is not False
    ):
        raise RuntimeError("isolated audio protocol binding drifted")
    for binding in protocol["source_bindings"]:
        if sha256_file(Paths().root / binding["path"]) != binding["sha256"]:
            raise RuntimeError(f"isolated audio source drifted: {binding['path']}")
    return protocol


def _planner(
    *,
    slot: dict[str, Any],
    profiles: dict[str, dict[str, Any]],
    model_vocab: set[str],
    nonce_checker: DatabaseNonceChecker,
    phone_indexes: tuple[Any, ...],
) -> BilingualListenerPlanner:
    fixture = slot["fixture_spec"]
    base_profile = profiles[slot["profile_id"]]
    profile = isolate_listener_profile(base_profile, slot["rule_id"])
    return BilingualListenerPlanner(
        profile={**profile, "voice_id": slot["voice_id"]},
        adapter=FixtureAdapter(
            language_id=base_profile["source_language"],
            source_words=_tuple(fixture["source_words"]),
            source_phones=_tuple(fixture["source_phones"]),
            punctuation=fixture["punctuation"],
        ),
        model_vocab=model_vocab,
        nonce_checker=nonce_checker,
        phone_indexes=phone_indexes,
    )


def _render_slot(
    *,
    slot: dict[str, Any],
    profiles: dict[str, dict[str, Any]],
    synthesis: Any,
    model_vocab: set[str],
    nonce_checker: DatabaseNonceChecker,
    phone_indexes: tuple[Any, ...],
) -> dict[str, Any]:
    planner = _planner(
        slot=slot,
        profiles=profiles,
        model_vocab=model_vocab,
        nonce_checker=nonce_checker,
        phone_indexes=phone_indexes,
    )
    runtime = BilingualListenerRuntime(planner=planner, synthesis=synthesis)
    rendered = runtime.render(slot["fixture_spec"]["text"])
    if not isinstance(rendered, BilingualVowelRender):
        raise RuntimeError("isolated fixture produced no comparison")
    active = active_changed_rule_ids(rendered.plan)
    if active != (slot["rule_id"],):
        raise RuntimeError(f"isolated plan activated unexpected rules: {active}")
    if rendered.plan.plan_sha256 != slot["isolated_plan_sha256"]:
        raise RuntimeError("isolated plan changed after manifest freeze")
    rows = _target_rows(rendered, slot["rule_id"])
    stem = _safe_name(slot["logical_slot_id"])
    audio = {
        "neutral": _write_wav(
            RUN_DIR / "audio" / f"{stem}__neutral.wav", rendered.neutral_pcm
        ),
        "lens": _write_wav(
            RUN_DIR / "audio" / f"{stem}__lens.wav", rendered.lens_pcm
        ),
        "identity_pcm_sha256": hashlib.sha256(
            np.asarray(rendered.identity_pcm, dtype="<i2").tobytes()
        ).hexdigest(),
        "full_lens_pcm_sha256": hashlib.sha256(
            np.asarray(rendered.full_lens_pcm, dtype="<i2").tobytes()
        ).hexdigest(),
    }
    passed = bool(_universal_pass(rendered, rows))
    return {
        "logical_slot_id": slot["logical_slot_id"],
        "cell_id": slot["cell_id"],
        "profile_id": slot["profile_id"],
        "voice_id": slot["voice_id"],
        "family": slot["family"],
        "rule_id": slot["rule_id"],
        "context": slot["context"],
        "status": "isolated_universal_integrity_pass"
        if passed
        else "isolated_universal_integrity_fail",
        "isolated_validation_profile_version": (
            ISOLATED_VALIDATION_PROFILE_VERSION
        ),
        "isolated_active_changed_rule_ids": active,
        "plan_sha256": rendered.plan.plan_sha256,
        "target_occurrences": rows,
        "splice_windows": rendered.splice_windows,
        "verification": asdict(rendered.verification),
        "audio": audio,
        "family_acoustic_status": "pending_separate_family_analysis",
        "product_enabled": False,
        "api_calls_made": 0,
    }


def _error_outcome(
    slot: dict[str, Any], *, status: str, exc: Exception
) -> dict[str, Any]:
    return {
        "logical_slot_id": slot["logical_slot_id"],
        "cell_id": slot["cell_id"],
        "profile_id": slot["profile_id"],
        "voice_id": slot["voice_id"],
        "family": slot["family"],
        "rule_id": slot["rule_id"],
        "context": slot["context"],
        "status": status,
        "error_code": getattr(exc, "code", type(exc).__name__),
        "error": str(exc),
        "product_enabled": False,
        "api_calls_made": 0,
    }


def main() -> None:
    if RUN_DIR.exists():
        raise RuntimeError(f"refusing to overwrite isolated audio run: {RUN_DIR}")
    matrix = load_bilingual_product_matrix()
    load_bilingual_structural_state(matrix)
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if (
        manifest["classification"] != "all_acoustic_slots_atomically_isolated"
        or manifest["isolated_plan_pass_count"] != 280
    ):
        raise RuntimeError("isolated acoustic manifest is not complete")
    protocol = _load_protocol(matrix.matrix_sha256, manifest)
    slots = manifest["slots"]
    profiles = load_listener_profiles()
    files = verify_model_files(download=False)
    model_vocab = set(
        json.loads(files[CONFIG_FILE].read_text(encoding="utf-8"))["vocab"]
    )
    nonce_checker = DatabaseNonceChecker()
    phone_indexes = (KokoroGateIndex(), PortuguesePositiveOnlyIndexV1())
    outcomes: list[dict[str, Any]] = []
    RUN_DIR.mkdir(parents=True, exist_ok=False)
    for voice_id in ("af_heart", "am_michael", "pm_alex", "pf_dora"):
        voice_slots = [slot for slot in slots if slot["voice_id"] == voice_id]
        try:
            synthesis = _load_pinned_synthesis_voice(voice_id)
        except Exception as exc:
            outcomes.extend(
                _error_outcome(slot, status="renderer_load_error", exc=exc)
                for slot in voice_slots
            )
            continue
        for slot in voice_slots:
            try:
                outcomes.append(
                    _render_slot(
                        slot=slot,
                        profiles=profiles,
                        synthesis=synthesis,
                        model_vocab=model_vocab,
                        nonce_checker=nonce_checker,
                        phone_indexes=phone_indexes,
                    )
                )
            except Exception as exc:
                outcomes.append(
                    _error_outcome(
                        slot, status="isolated_render_or_integrity_error", exc=exc
                    )
                )
    passed = sum(
        row["status"] == "isolated_universal_integrity_pass"
        for row in outcomes
    )
    failed = len(outcomes) - passed
    result = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "protocol_version": PROTOCOL_VERSION,
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "matrix_version": matrix.matrix_version,
        "matrix_sha256": matrix.matrix_sha256,
        "isolated_manifest_sha256": sha256_file(MANIFEST_PATH),
        "isolated_manifest_record_sha256": manifest["record_sha256"],
        "classification": (
            "all_isolated_slots_universal_integrity_pass_family_acoustics_pending"
            if failed == 0
            else "isolated_universal_integrity_yield_incomplete"
        ),
        "scope": "all_atomically_isolated_rule_voice_context_slots",
        "slot_count": len(outcomes),
        "isolated_universal_integrity_pass_count": passed,
        "isolated_universal_integrity_fail_count": failed,
        "isolated_universal_integrity_yield": passed / len(outcomes),
        "family_acoustic_classification_status": "pending",
        "api_calls_made": 0,
        "audio_render_sets_made": sum("audio" in row for row in outcomes),
        "replacement_slots_used": 0,
        "production_enabled": False,
        "protocol": protocol,
        "outcomes": outcomes,
    }
    result["record_sha256"] = _semantic_hash(result)
    atomic_write_json(RUN_DIR / "results.json", result)
    print(
        json.dumps(
            {
                "output": str(RUN_DIR / "results.json"),
                "classification": result["classification"],
                "slot_count": len(outcomes),
                "pass_count": passed,
                "fail_count": failed,
                "audio_render_sets_made": result["audio_render_sets_made"],
                "api_calls_made": 0,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
