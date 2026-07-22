#!/usr/bin/env python3
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Sequence
import wave

import numpy as np

from earshift_bakeoff.bilingual_listener_engine import load_listener_profiles
from earshift_bakeoff.bilingual_product_isolation import active_changed_rule_ids
from earshift_bakeoff.bilingual_product_matrix import load_bilingual_product_matrix
from earshift_bakeoff.bilingual_vowel_acoustics import run_praat_formant_frames
from earshift_bakeoff.bilingual_vowel_acoustics_v2 import (
    VOWEL_ACOUSTIC_VERSION_V2,
    measurement_mode,
    stress_core_measurement,
)
from earshift_bakeoff.bilingual_vowel_engine import (
    BilingualVowelRender,
    _load_pinned_synthesis_voice,
)
from earshift_bakeoff.bilingual_vowel_word_context import (
    BilingualVowelWordContextRuntime,
    VOWEL_WORD_CONTEXT_CANDIDATE_VERSION,
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

from run_bilingual_product_v8_vowel_acoustic_screen import (
    FORMANT_CEILINGS_HZ,
    PRAAT_PATH,
    PRAAT_SCRIPT_PATH,
    RHOTIC_RULE_IDS,
    V8_MANIFEST_PATH,
    _aggregate,
    _analysis_classification,
    _pcm_hash,
    _planner_v8,
    _read_wav,
    _safe_name,
)


PROTOCOL_VERSION = "bilingual-vowel-word-context-screen-v1"
PROTOCOL_PATH = Paths().root / "rules" / f"{PROTOCOL_VERSION}.json"
RUN_ID = "20260717-bilingual-vowel-word-context-screen-v1"
RUN_DIR = Paths().artifacts / "product-matrix" / RUN_ID
V8_RESULT_PATH = (
    Paths().artifacts
    / "product-matrix"
    / "20260717-bilingual-product-v8-vowel-acoustic-screen"
    / "results.json"
)
V8_RESULT_DIR = V8_RESULT_PATH.parent
VOICE_ORDER = ("af_heart", "am_michael", "pm_alex", "pf_dora")


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
        "pcm_sha256": _pcm_hash(values),
        "sample_count": int(values.size),
        "duration_s": values.size / SAMPLE_RATE_HZ,
    }


def _candidate_cell_ids(v8_result: dict[str, Any]) -> tuple[str, ...]:
    analyses: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for outcome in v8_result["outcomes"]:
        for ceiling in outcome["analysis_by_formant_ceiling"]:
            analyses[outcome["cell_id"]].extend(ceiling["occurrence_classifications"])
    selected = []
    for cell in v8_result["cell_summaries"]:
        rows = analyses[cell["cell_id"]]
        if (
            cell["classification"] == "fail"
            and rows
            and all(row["classification"] != "measurement_exclusion" for row in rows)
            and all(row["base_vowel"]["anchor_gate_pass"] for row in rows)
            and all(
                row["rhoticity"] is None or row["rhoticity"]["anchor_gate_pass"]
                for row in rows
            )
        ):
            selected.append(cell["cell_id"])
    return tuple(sorted(selected))


def _load_protocol(
    *,
    matrix_sha256: str,
    manifest: dict[str, Any],
    v8_result: dict[str, Any],
    candidate_cell_ids: tuple[str, ...],
) -> dict[str, Any]:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    expected_keys = {
        "schema_version",
        "protocol_version",
        "status",
        "production_enabled",
        "parent_bindings",
        "selection_basis",
        "scope",
        "candidate_intervention",
        "instrument_and_gates",
        "aggregation_policy",
        "claim_limits",
        "stopping_rule",
        "source_bindings",
    }
    if set(protocol) != expected_keys:
        raise RuntimeError("word-context protocol schema drifted")
    parents = protocol["parent_bindings"]
    if (
        protocol["schema_version"] != 1
        or protocol["protocol_version"] != PROTOCOL_VERSION
        or protocol["status"] != "frozen_before_first_word_context_matrix_render"
        or protocol["production_enabled"] is not False
        or parents["matrix_sha256"] != matrix_sha256
        or parents["v8_manifest_sha256"] != sha256_file(V8_MANIFEST_PATH)
        or parents["v8_manifest_record_sha256"] != manifest["record_sha256"]
        or parents["v8_result_sha256"] != sha256_file(V8_RESULT_PATH)
        or parents["v8_result_record_sha256"] != v8_result["record_sha256"]
        or tuple(protocol["scope"]["cell_ids_in_order"]) != candidate_cell_ids
        or protocol["scope"]["voice_rule_cell_count"] != 16
        or protocol["scope"]["logical_slot_count"] != 48
        or protocol["scope"]["candidate_render_set_count"] != 48
        or protocol["candidate_intervention"]["api_calls_allowed"] != 0
        or protocol["candidate_intervention"]["replacement_slots_allowed"] is not False
        or protocol["aggregation_policy"]["product_promotion_allowed"] is not False
    ):
        raise RuntimeError("word-context protocol binding drifted")
    if sha256_file(PRAAT_PATH) != protocol["instrument_and_gates"]["praat_sha256"]:
        raise RuntimeError("Praat binary changed after word-context freeze")
    if (
        sha256_file(PRAAT_SCRIPT_PATH)
        != protocol["instrument_and_gates"]["praat_script_sha256"]
    ):
        raise RuntimeError("Praat script changed after word-context freeze")
    for binding in protocol["source_bindings"]:
        if sha256_file(Paths().root / binding["path"]) != binding["sha256"]:
            raise RuntimeError(f"word-context source drifted: {binding['path']}")
    return protocol


def _measure_lens(
    *,
    path: Path,
    stem: str,
    intervals: Sequence[dict[str, Any]],
    ceiling: int,
    mode: str,
) -> tuple[dict[str, Any], ...]:
    frame_path = RUN_DIR / "praat-frames" / f"{stem}__ceiling-{ceiling}.tsv"
    frames = run_praat_formant_frames(
        path,
        frame_path,
        maximum_formant_hz=ceiling,
        praat_path=PRAAT_PATH,
        script_path=PRAAT_SCRIPT_PATH,
    )
    return tuple(
        stress_core_measurement(
            frames,
            start_s=float(interval["start_s"]),
            end_s=float(interval["end_s"]),
            mode=mode,
        )
        for interval in intervals
    )


def _slot_outcome(
    *,
    slot: dict[str, Any],
    baseline: dict[str, Any],
    planner: Any,
    synthesis: Any,
) -> dict[str, Any]:
    started = time.perf_counter()
    rendered = BilingualVowelWordContextRuntime(
        planner=planner, synthesis=synthesis
    ).render(slot["fixture_spec"]["text"])
    if not isinstance(rendered, BilingualVowelRender):
        raise RuntimeError("word-context slot produced no comparison")
    if rendered.plan.plan_sha256 != slot["v8_plan_sha256"]:
        raise RuntimeError("word-context candidate changed the frozen v8 plan")
    if active_changed_rule_ids(rendered.plan) != (slot["rule_id"],):
        raise RuntimeError("word-context candidate lost atomic rule isolation")
    if not rendered.verification.integrity_pass:
        raise RuntimeError("word-context candidate failed universal integrity")
    baseline_neutral_path = (
        V8_RESULT_DIR / baseline["audio"]["neutral"]["relative_path"]
    )
    baseline_neutral = _read_wav(baseline_neutral_path)
    neutral_reuse_pass = bool(
        _pcm_hash(baseline_neutral) == baseline["audio"]["neutral"]["pcm_sha256"]
        and np.array_equal(rendered.neutral_pcm, baseline_neutral)
    )
    rows = [
        row
        for row in rendered.alignment["target_occurrences"]
        if row["segment_type"] == "vowel" and row["rule_id"] == slot["rule_id"]
    ]
    if len(rows) != len(baseline["occurrence_outcomes"]):
        raise RuntimeError("word-context occurrence count changed")
    mode = measurement_mode(slot["source"], slot["target"])
    intervals = tuple(row["measurement_interval"] for row in rows)
    stem = _safe_name(slot["logical_slot_id"])
    lens_path = RUN_DIR / "audio" / f"{stem}__word-context-lens.wav"
    audio = {
        "reused_v8_neutral": baseline["audio"]["neutral"],
        "word_context_lens": _write_wav(lens_path, rendered.lens_pcm),
    }
    analysis = []
    for baseline_ceiling in baseline["analysis_by_formant_ceiling"]:
        ceiling = int(baseline_ceiling["maximum_formant_hz"])
        if ceiling not in FORMANT_CEILINGS_HZ:
            raise RuntimeError("word-context formant family drifted")
        lens_measurements = _measure_lens(
            path=lens_path,
            stem=f"{stem}__word-context-lens",
            intervals=intervals,
            ceiling=ceiling,
            mode=mode,
        )
        measurements = baseline_ceiling["measurements"]
        classifications = tuple(
            _analysis_classification(
                source=measurements["source_anchor"][index],
                target=measurements["target_anchor"][index],
                neutral=measurements["neutral"][index],
                lens=lens_measurements[index],
                rhotic=slot["rule_id"] in RHOTIC_RULE_IDS,
            )
            for index in range(len(rows))
        )
        analysis.append(
            {
                "maximum_formant_hz": ceiling,
                "reused_v8_source_anchor": measurements["source_anchor"],
                "reused_v8_target_anchor": measurements["target_anchor"],
                "reused_v8_neutral": measurements["neutral"],
                "word_context_lens": lens_measurements,
                "occurrence_classifications": classifications,
            }
        )
    occurrences = []
    for index, row in enumerate(rows):
        classifications = tuple(
            ceiling["occurrence_classifications"][index] for ceiling in analysis
        )
        classification = _aggregate(classifications)
        occurrences.append(
            {
                "occurrence_index": index,
                "word_index": row["word_index"],
                "source": row["source"],
                "target": row["target"],
                "measurement_interval": row["measurement_interval"],
                "classification": classification,
                "exact_category_pass": classification == "exact_category_pass",
                "directional_pass": classification
                in {"exact_category_pass", "directional_only_pass"},
            }
        )
    classification = _aggregate(occurrences) if neutral_reuse_pass else "fail"
    return {
        "logical_slot_id": slot["logical_slot_id"],
        "cell_id": slot["cell_id"],
        "profile_id": slot["profile_id"],
        "voice_id": slot["voice_id"],
        "rule_id": slot["rule_id"],
        "context": slot["context"],
        "source": slot["source"],
        "target": slot["target"],
        "status": "measured",
        "baseline_v8_classification": baseline["classification"],
        "candidate_classification": classification,
        "exact_category_pass": classification == "exact_category_pass",
        "directional_pass": classification
        in {"exact_category_pass", "directional_only_pass"},
        "measurement_mode": mode,
        "neutral_pcm_reused_from_v8": neutral_reuse_pass,
        "verification": asdict(rendered.verification),
        "candidate_metadata": rendered.prosody,
        "splice_windows": rendered.splice_windows,
        "audio": audio,
        "occurrence_outcomes": occurrences,
        "analysis_by_formant_ceiling": analysis,
        "elapsed_s": time.perf_counter() - started,
        "api_calls_made": 0,
        "product_enabled": False,
    }


def _cell_summaries(outcomes: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for outcome in outcomes:
        groups[outcome["cell_id"]].append(outcome)
    summaries = []
    for cell_id, rows in groups.items():
        measured = [row for row in rows if row["status"] == "measured"]
        occurrence_count = sum(len(row["occurrence_outcomes"]) for row in measured)
        complete = bool(len(rows) == 3 and len(measured) == 3 and occurrence_count == 4)
        classification = _aggregate(measured) if complete else "fail"
        nasal = any("̃" in str(row[key]) for row in rows for key in ("source", "target"))
        eligible = bool(
            complete
            and classification in {"exact_category_pass", "directional_only_pass"}
            and not nasal
        )
        summaries.append(
            {
                "cell_id": cell_id,
                "profile_id": rows[0]["profile_id"],
                "voice_id": rows[0]["voice_id"],
                "rule_id": rows[0]["rule_id"],
                "source": rows[0]["source"],
                "target": rows[0]["target"],
                "slot_count": len(rows),
                "occurrence_count": occurrence_count,
                "complete_three_context_four_occurrence_yield": complete,
                "baseline_v8_classification": "fail",
                "candidate_classification": classification,
                "first_passing_context_mode": (
                    "target_word_state_plus_excitation"
                    if classification
                    in {"exact_category_pass", "directional_only_pass"}
                    else None
                ),
                "automatic_human_qc_eligible": eligible,
                "human_qc_status": "pending" if eligible else "not_eligible",
                "product_enabled": False,
            }
        )
    return sorted(summaries, key=lambda row: row["cell_id"])


def main() -> None:
    if RUN_DIR.exists():
        raise RuntimeError(f"refusing to overwrite frozen word-context run: {RUN_DIR}")
    matrix = load_bilingual_product_matrix()
    manifest = json.loads(V8_MANIFEST_PATH.read_text(encoding="utf-8"))
    v8_result = json.loads(V8_RESULT_PATH.read_text(encoding="utf-8"))
    candidate_cell_ids = _candidate_cell_ids(v8_result)
    protocol = _load_protocol(
        matrix_sha256=matrix.matrix_sha256,
        manifest=manifest,
        v8_result=v8_result,
        candidate_cell_ids=candidate_cell_ids,
    )
    slots = [
        slot for slot in manifest["slots"] if slot["cell_id"] in candidate_cell_ids
    ]
    if len(slots) != 48:
        raise RuntimeError("word-context candidate manifest is not exactly 48 slots")
    baseline_by_id = {
        outcome["logical_slot_id"]: outcome for outcome in v8_result["outcomes"]
    }
    profiles = load_listener_profiles()
    files = verify_model_files(download=False)
    model_vocab = set(
        json.loads(files[CONFIG_FILE].read_text(encoding="utf-8"))["vocab"]
    )
    nonce_checker = DatabaseNonceChecker()
    phone_indexes = (KokoroGateIndex(), PortuguesePositiveOnlyIndexV1())
    outcomes = []
    RUN_DIR.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    for voice_id in VOICE_ORDER:
        synthesis = _load_pinned_synthesis_voice(voice_id)
        for slot in (row for row in slots if row["voice_id"] == voice_id):
            try:
                planner = _planner_v8(
                    slot=slot,
                    profiles=profiles,
                    model_vocab=model_vocab,
                    nonce_checker=nonce_checker,
                    phone_indexes=phone_indexes,
                )
                outcomes.append(
                    _slot_outcome(
                        slot=slot,
                        baseline=baseline_by_id[slot["logical_slot_id"]],
                        planner=planner,
                        synthesis=synthesis,
                    )
                )
            except Exception as exc:
                outcomes.append(
                    {
                        "logical_slot_id": slot["logical_slot_id"],
                        "cell_id": slot["cell_id"],
                        "profile_id": slot["profile_id"],
                        "voice_id": slot["voice_id"],
                        "rule_id": slot["rule_id"],
                        "context": slot["context"],
                        "source": slot["source"],
                        "target": slot["target"],
                        "status": "render_or_measurement_error",
                        "baseline_v8_classification": "fail",
                        "candidate_classification": "fail",
                        "error_code": getattr(exc, "code", type(exc).__name__),
                        "error": str(exc),
                        "api_calls_made": 0,
                        "product_enabled": False,
                    }
                )
    cells = _cell_summaries(outcomes)
    counts = {
        label: sum(row["candidate_classification"] == label for row in cells)
        for label in ("exact_category_pass", "directional_only_pass", "fail")
    }
    result = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "protocol_version": PROTOCOL_VERSION,
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "candidate_version": VOWEL_WORD_CONTEXT_CANDIDATE_VERSION,
        "acoustic_version": VOWEL_ACOUSTIC_VERSION_V2,
        "matrix_sha256": matrix.matrix_sha256,
        "v8_result_sha256": sha256_file(V8_RESULT_PATH),
        "v8_result_record_sha256": v8_result["record_sha256"],
        "classification": "word_context_candidate_screen_complete_no_product_promotion",
        "candidate_cell_count": len(cells),
        "logical_slot_count": len(outcomes),
        "measured_slot_count": sum(row["status"] == "measured" for row in outcomes),
        "error_slot_count": sum(row["status"] != "measured" for row in outcomes),
        "cell_classification_counts": counts,
        "rescued_cell_count": counts["exact_category_pass"]
        + counts["directional_only_pass"],
        "automatic_human_qc_eligible_cell_count": sum(
            row["automatic_human_qc_eligible"] for row in cells
        ),
        "api_calls_made": 0,
        "replacement_slots_used": 0,
        "elapsed_s": time.perf_counter() - started,
        "production_enabled": False,
        "protocol": protocol,
        "cell_summaries": cells,
        "outcomes": outcomes,
    }
    result["record_sha256"] = _semantic_hash(result)
    atomic_write_json(RUN_DIR / "results.json", result)
    print(
        json.dumps(
            {
                "output": str(RUN_DIR / "results.json"),
                "classification": result["classification"],
                "logical_slot_count": result["logical_slot_count"],
                "measured_slot_count": result["measured_slot_count"],
                "error_slot_count": result["error_slot_count"],
                "cell_classification_counts": counts,
                "rescued_cell_count": result["rescued_cell_count"],
                "automatic_human_qc_eligible_cell_count": result[
                    "automatic_human_qc_eligible_cell_count"
                ],
                "api_calls_made": 0,
                "elapsed_s": result["elapsed_s"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
