#!/usr/bin/env python3
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Sequence
import wave

import numpy as np

from earshift_bakeoff.bilingual_listener_engine import (
    SEGMENT_SPLICE_CONTEXT_SAMPLES,
    load_listener_profiles,
)
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
from earshift_bakeoff.bilingual_vowel_state_strength import (
    BilingualVowelStateStrengthRuntime,
    VOWEL_STATE_STRENGTH_CANDIDATE_VERSION,
)
from earshift_bakeoff.config import Paths, stable_json
from earshift_bakeoff.controlled_vowel_state_strength import (
    CONTROLLED_VOWEL_STATE_STRENGTH_VERSION,
)
from earshift_bakeoff.kokoro_gate_bridge import KokoroGateIndex
from earshift_bakeoff.kokoro_output_domain_splice import (
    boundary_artifact_report,
    output_domain_splice,
)
from earshift_bakeoff.kokoro_synthesis import (
    CONFIG_FILE,
    SAMPLE_RATE_HZ,
    verify_model_files,
)
from earshift_bakeoff.kokoro_typed_diagnostic import localization_report
from earshift_bakeoff.listener_lens import DatabaseNonceChecker
from earshift_bakeoff.portuguese_carrier_planner_v1 import (
    PortuguesePositiveOnlyIndexV1,
)
from earshift_bakeoff.util import atomic_write_json, sha256_file

import run_bilingual_vowel_word_context_screen_v1 as word_screen
from run_bilingual_product_v8_vowel_acoustic_screen import (
    FORMANT_CEILINGS_HZ,
    PRAAT_PATH,
    PRAAT_SCRIPT_PATH,
    V8_MANIFEST_PATH,
    _aggregate,
    _analysis_classification,
    _pcm_hash,
    _read_wav,
    _safe_name,
)


PROTOCOL_VERSION = "bilingual-vowel-adaptive-strength-screen-v1"
PROTOCOL_PATH = Paths().root / "rules" / f"{PROTOCOL_VERSION}.json"
RUN_ID = "20260717-bilingual-vowel-adaptive-strength-screen-v1"
RUN_DIR = Paths().artifacts / "product-matrix" / RUN_ID
V8_RESULT_PATH = (
    Paths().artifacts
    / "product-matrix"
    / "20260717-bilingual-product-v8-vowel-acoustic-screen"
    / "results.json"
)
V8_RESULT_DIR = V8_RESULT_PATH.parent
FULL_RESULT_PATH = (
    Paths().artifacts
    / "product-matrix"
    / "20260717-bilingual-vowel-full-context-screen-v1"
    / "results.json"
)
STRENGTHS = (1.0, 0.75, 1.25, 0.5, 1.5, 2.0)
VOICE_ORDER = ("af_heart", "am_michael", "pm_alex", "pf_dora")


def _semantic_hash(payload: Any) -> str:
    return hashlib.sha256(stable_json(payload).encode("utf-8")).hexdigest()


def _label(strength: float) -> str:
    return f"strength-{int(round(strength * 100)):03d}"


def _candidate_cell_ids(full_result: dict[str, Any]) -> tuple[str, ...]:
    return tuple(
        sorted(
            row["cell_id"]
            for row in full_result["cell_summaries"]
            if row["candidate_classification"] == "fail"
        )
    )


def _load_protocol(
    *,
    matrix_sha256: str,
    manifest: dict[str, Any],
    v8_result: dict[str, Any],
    full_result: dict[str, Any],
    candidate_cell_ids: tuple[str, ...],
) -> dict[str, Any]:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if (
        protocol["schema_version"] != 1
        or protocol["protocol_version"] != PROTOCOL_VERSION
        or protocol["status"] != "frozen_before_first_adaptive_strength_render"
        or protocol["production_enabled"] is not False
        or protocol["parent_bindings"]["matrix_sha256"] != matrix_sha256
        or protocol["parent_bindings"]["v8_manifest_sha256"]
        != sha256_file(V8_MANIFEST_PATH)
        or protocol["parent_bindings"]["v8_manifest_record_sha256"]
        != manifest["record_sha256"]
        or protocol["parent_bindings"]["v8_result_sha256"]
        != sha256_file(V8_RESULT_PATH)
        or protocol["parent_bindings"]["v8_result_record_sha256"]
        != v8_result["record_sha256"]
        or protocol["parent_bindings"]["full_result_sha256"]
        != sha256_file(FULL_RESULT_PATH)
        or protocol["parent_bindings"]["full_result_record_sha256"]
        != full_result["record_sha256"]
        or tuple(protocol["scope"]["cell_ids_in_order"]) != candidate_cell_ids
        or protocol["scope"]["voice_rule_cell_count"] != 12
        or protocol["scope"]["logical_slot_count"] != 36
        or protocol["scope"]["candidate_render_set_count"] != 216
        or tuple(protocol["candidate_intervention"]["strength_order"]) != STRENGTHS
        or protocol["candidate_intervention"]["api_calls_allowed"] != 0
        or protocol["aggregation_policy"]["product_promotion_allowed"] is not False
    ):
        raise RuntimeError("adaptive-strength protocol binding drifted")
    if sha256_file(PRAAT_PATH) != protocol["instrument_and_gates"]["praat_sha256"]:
        raise RuntimeError("Praat binary changed after adaptive-strength freeze")
    if (
        sha256_file(PRAAT_SCRIPT_PATH)
        != protocol["instrument_and_gates"]["praat_script_sha256"]
    ):
        raise RuntimeError("Praat script changed after adaptive-strength freeze")
    for binding in protocol["source_bindings"]:
        if sha256_file(Paths().root / binding["path"]) != binding["sha256"]:
            raise RuntimeError(f"adaptive-strength source drifted: {binding['path']}")
    return protocol


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


def _measure(
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


def _occurrence_windows(
    intervals: Sequence[dict[str, Any]], sample_count: int
) -> tuple[dict[str, int], ...]:
    windows = tuple(
        {
            "start_sample": max(
                0,
                int(interval["start_sample"]) - SEGMENT_SPLICE_CONTEXT_SAMPLES,
            ),
            "end_sample_exclusive": min(
                sample_count,
                int(interval["end_sample_exclusive"]) + SEGMENT_SPLICE_CONTEXT_SAMPLES,
            ),
        }
        for interval in intervals
    )
    if any(
        left["end_sample_exclusive"] > right["start_sample"]
        for left, right in zip(windows, windows[1:])
    ):
        raise RuntimeError("adaptive-strength occurrence windows overlap")
    return windows


def _candidate_outcome(
    *,
    slot: dict[str, Any],
    baseline: dict[str, Any],
    planner: Any,
    synthesis: Any,
    strength: float,
) -> tuple[dict[str, Any], BilingualVowelRender]:
    started = time.perf_counter()
    rendered = BilingualVowelStateStrengthRuntime(
        planner=planner,
        synthesis=synthesis,
        state_strength=strength,
    ).render(slot["fixture_spec"]["text"])
    if not isinstance(rendered, BilingualVowelRender):
        raise RuntimeError("adaptive-strength slot produced no comparison")
    if rendered.plan.plan_sha256 != slot["v8_plan_sha256"]:
        raise RuntimeError("adaptive-strength candidate changed the frozen v8 plan")
    if active_changed_rule_ids(rendered.plan) != (slot["rule_id"],):
        raise RuntimeError("adaptive-strength candidate lost atomic rule isolation")
    if not rendered.verification.integrity_pass:
        raise RuntimeError("adaptive-strength candidate failed universal integrity")
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
        raise RuntimeError("adaptive-strength occurrence count changed")
    mode = measurement_mode(slot["source"], slot["target"])
    intervals = tuple(row["measurement_interval"] for row in rows)
    stem = _safe_name(slot["logical_slot_id"])
    label = _label(strength)
    lens_path = RUN_DIR / "audio" / f"{stem}__{label}.wav"
    audio = _write_wav(lens_path, rendered.lens_pcm)
    analysis = []
    for baseline_ceiling in baseline["analysis_by_formant_ceiling"]:
        ceiling = int(baseline_ceiling["maximum_formant_hz"])
        if ceiling not in FORMANT_CEILINGS_HZ:
            raise RuntimeError("adaptive-strength formant family drifted")
        lens_measurements = _measure(
            path=lens_path,
            stem=f"{stem}__{label}",
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
                rhotic=False,
            )
            for index in range(len(rows))
        )
        analysis.append(
            {
                "maximum_formant_hz": ceiling,
                "lens": lens_measurements,
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
                "classification": classification,
                "exact_category_pass": classification == "exact_category_pass",
                "directional_pass": classification
                in {"exact_category_pass", "directional_only_pass"},
            }
        )
    classification = _aggregate(occurrences) if neutral_reuse_pass else "fail"
    return (
        {
            "state_strength": strength,
            "label": label,
            "status": "measured",
            "classification": classification,
            "neutral_pcm_reused_from_v8": neutral_reuse_pass,
            "verification": asdict(rendered.verification),
            "candidate_metadata": rendered.prosody,
            "audio": audio,
            "occurrence_outcomes": occurrences,
            "analysis_by_formant_ceiling": analysis,
            "elapsed_s": time.perf_counter() - started,
        },
        rendered,
    )


def _choose_occurrence_strength(
    candidates: Sequence[dict[str, Any]], occurrence_index: int
) -> dict[str, Any] | None:
    for desired in ("exact_category_pass", "directional_only_pass"):
        for candidate in candidates:
            classification = candidate["occurrence_outcomes"][occurrence_index][
                "classification"
            ]
            if classification == desired:
                return {
                    "state_strength": candidate["state_strength"],
                    "label": candidate["label"],
                    "classification": classification,
                }
    return None


def _adaptive_slot(
    *,
    slot: dict[str, Any],
    baseline: dict[str, Any],
    planner: Any,
    synthesis: Any,
) -> dict[str, Any]:
    candidates = []
    renders = {}
    for strength in STRENGTHS:
        candidate, rendered = _candidate_outcome(
            slot=slot,
            baseline=baseline,
            planner=planner,
            synthesis=synthesis,
            strength=strength,
        )
        candidates.append(candidate)
        renders[strength] = rendered
    sample = renders[STRENGTHS[0]]
    rows = [
        row
        for row in sample.alignment["target_occurrences"]
        if row["segment_type"] == "vowel" and row["rule_id"] == slot["rule_id"]
    ]
    intervals = tuple(row["measurement_interval"] for row in rows)
    selections = tuple(
        _choose_occurrence_strength(candidates, index) for index in range(len(rows))
    )
    unresolved = tuple(
        index for index, selection in enumerate(selections) if not selection
    )
    adaptive_audio = None
    adaptive_analysis = []
    adaptive_verification = None
    occurrence_outcomes = []
    classification = "fail"
    if not unresolved:
        neutral = sample.neutral_pcm
        windows = _occurrence_windows(intervals, neutral.size)
        composite_full_lens = neutral.copy()
        for window, selection in zip(windows, selections, strict=True):
            assert selection is not None
            source = renders[selection["state_strength"]].full_lens_pcm
            start = window["start_sample"]
            end = window["end_sample_exclusive"]
            composite_full_lens[start:end] = source[start:end]
        adaptive_lens, weights = output_domain_splice(
            neutral,
            composite_full_lens,
            windows,
        )
        boundary = boundary_artifact_report(
            neutral,
            composite_full_lens,
            adaptive_lens,
            windows,
        )
        localization = localization_report(neutral, adaptive_lens, intervals)
        adaptive_verification = {
            "equal_nonempty_samples": bool(
                neutral.size
                and neutral.size == composite_full_lens.size == adaptive_lens.size
            ),
            "finite": bool(
                np.isfinite(composite_full_lens.astype(np.float64)).all()
                and np.isfinite(adaptive_lens.astype(np.float64)).all()
            ),
            "unclipped": bool(
                np.mean(np.abs(composite_full_lens.astype(np.int64)) >= 32767) < 0.001
                and np.mean(np.abs(adaptive_lens.astype(np.int64)) >= 32767) < 0.001
            ),
            "outside_splice_exact_neutral": bool(
                np.array_equal(adaptive_lens[weights == 0.0], neutral[weights == 0.0])
            ),
            "full_weight_interior_exact_lens": bool(
                np.any(weights == 1.0)
                and np.array_equal(
                    adaptive_lens[weights == 1.0],
                    composite_full_lens[weights == 1.0],
                )
            ),
            "boundary_metrics_pass": bool(boundary.get("pass")),
            "localization_pass": bool(localization.get("pass")),
            "localization_fraction": float(
                localization.get("inside_difference_energy_fraction", 0.0)
            ),
        }
        adaptive_verification["integrity_pass"] = all(
            adaptive_verification[key]
            for key in (
                "equal_nonempty_samples",
                "finite",
                "unclipped",
                "outside_splice_exact_neutral",
                "full_weight_interior_exact_lens",
                "boundary_metrics_pass",
                "localization_pass",
            )
        )
        stem = _safe_name(slot["logical_slot_id"])
        adaptive_path = RUN_DIR / "audio" / f"{stem}__adaptive-composite.wav"
        adaptive_audio = _write_wav(adaptive_path, adaptive_lens)
        mode = measurement_mode(slot["source"], slot["target"])
        for baseline_ceiling in baseline["analysis_by_formant_ceiling"]:
            ceiling = int(baseline_ceiling["maximum_formant_hz"])
            lens_measurements = _measure(
                path=adaptive_path,
                stem=f"{stem}__adaptive-composite",
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
                    rhotic=False,
                )
                for index in range(len(rows))
            )
            adaptive_analysis.append(
                {
                    "maximum_formant_hz": ceiling,
                    "lens": lens_measurements,
                    "occurrence_classifications": classifications,
                }
            )
        for index, row in enumerate(rows):
            fine = tuple(
                ceiling["occurrence_classifications"][index]
                for ceiling in adaptive_analysis
            )
            outcome = _aggregate(fine)
            occurrence_outcomes.append(
                {
                    "occurrence_index": index,
                    "word_index": row["word_index"],
                    "selection": selections[index],
                    "classification": outcome,
                    "exact_category_pass": outcome == "exact_category_pass",
                    "directional_pass": outcome
                    in {"exact_category_pass", "directional_only_pass"},
                }
            )
        classification = (
            _aggregate(occurrence_outcomes)
            if adaptive_verification["integrity_pass"]
            else "fail"
        )
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
        "candidate_classification": classification,
        "exact_category_pass": classification == "exact_category_pass",
        "directional_pass": classification
        in {"exact_category_pass", "directional_only_pass"},
        "selection_complete": not unresolved,
        "unresolved_occurrence_indexes": unresolved,
        "occurrence_selections": selections,
        "adaptive_verification": adaptive_verification,
        "adaptive_audio": adaptive_audio,
        "occurrence_outcomes": occurrence_outcomes,
        "adaptive_analysis_by_formant_ceiling": adaptive_analysis,
        "strength_candidates": candidates,
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
        complete = bool(
            len(rows) == 3
            and len(measured) == 3
            and all(row["selection_complete"] for row in measured)
            and sum(len(row["occurrence_outcomes"]) for row in measured) == 4
        )
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
                "complete_three_context_four_occurrence_yield": complete,
                "candidate_classification": classification,
                "selected_strength_counts": dict(
                    Counter(
                        selection["label"]
                        for row in measured
                        for selection in row["occurrence_selections"]
                        if selection is not None
                    )
                ),
                "automatic_human_qc_eligible": eligible,
                "human_qc_status": "pending" if eligible else "not_eligible",
                "product_enabled": False,
            }
        )
    return sorted(summaries, key=lambda row: row["cell_id"])


def main() -> None:
    if RUN_DIR.exists():
        raise RuntimeError(f"refusing to overwrite frozen adaptive run: {RUN_DIR}")
    matrix = load_bilingual_product_matrix()
    manifest = json.loads(V8_MANIFEST_PATH.read_text(encoding="utf-8"))
    v8_result = json.loads(V8_RESULT_PATH.read_text(encoding="utf-8"))
    full_result = json.loads(FULL_RESULT_PATH.read_text(encoding="utf-8"))
    candidate_cell_ids = _candidate_cell_ids(full_result)
    protocol = _load_protocol(
        matrix_sha256=matrix.matrix_sha256,
        manifest=manifest,
        v8_result=v8_result,
        full_result=full_result,
        candidate_cell_ids=candidate_cell_ids,
    )
    slots = [
        slot for slot in manifest["slots"] if slot["cell_id"] in candidate_cell_ids
    ]
    if len(slots) != 36:
        raise RuntimeError("adaptive-strength manifest is not exactly 36 slots")
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
                planner = word_screen._planner_v8(
                    slot=slot,
                    profiles=profiles,
                    model_vocab=model_vocab,
                    nonce_checker=nonce_checker,
                    phone_indexes=phone_indexes,
                )
                outcomes.append(
                    _adaptive_slot(
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
        "candidate_version": VOWEL_STATE_STRENGTH_CANDIDATE_VERSION,
        "controlled_version": CONTROLLED_VOWEL_STATE_STRENGTH_VERSION,
        "acoustic_version": VOWEL_ACOUSTIC_VERSION_V2,
        "matrix_sha256": matrix.matrix_sha256,
        "v8_result_sha256": sha256_file(V8_RESULT_PATH),
        "full_result_sha256": sha256_file(FULL_RESULT_PATH),
        "classification": "adaptive_strength_screen_complete_no_product_promotion",
        "candidate_cell_count": len(cells),
        "logical_slot_count": len(outcomes),
        "candidate_render_set_count": len(outcomes) * len(STRENGTHS),
        "measured_slot_count": sum(row["status"] == "measured" for row in outcomes),
        "error_slot_count": sum(row["status"] != "measured" for row in outcomes),
        "cell_classification_counts": counts,
        "rescued_cell_count": counts["exact_category_pass"]
        + counts["directional_only_pass"],
        "automatic_human_qc_eligible_cell_count": sum(
            row["automatic_human_qc_eligible"] for row in cells
        ),
        "selection_strength_counts": dict(
            Counter(
                selection["label"]
                for outcome in outcomes
                if outcome["status"] == "measured"
                for selection in outcome["occurrence_selections"]
                if selection is not None
            )
        ),
        "unresolved_occurrence_count": sum(
            len(outcome.get("unresolved_occurrence_indexes", ()))
            for outcome in outcomes
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
                "logical_slot_count": result["logical_slot_count"],
                "candidate_render_set_count": result["candidate_render_set_count"],
                "measured_slot_count": result["measured_slot_count"],
                "error_slot_count": result["error_slot_count"],
                "cell_classification_counts": counts,
                "rescued_cell_count": result["rescued_cell_count"],
                "unresolved_occurrence_count": result["unresolved_occurrence_count"],
                "api_calls_made": 0,
                "elapsed_s": result["elapsed_s"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
