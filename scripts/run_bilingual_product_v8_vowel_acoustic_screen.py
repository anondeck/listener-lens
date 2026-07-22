#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Sequence
import wave

import numpy as np

from earshift_bakeoff.bilingual_listener_engine import load_listener_profiles
from earshift_bakeoff.bilingual_listener_engine_v8 import (
    BILINGUAL_LISTENER_CANDIDATE_VERSION_V8,
    BilingualListenerPlannerV8,
    BilingualListenerRuntimeV8,
    VOWEL_MEASUREMENT_ALIGNMENT_VERSION,
    bilingual_alignment_record_v8,
)
from earshift_bakeoff.bilingual_product_isolation import (
    active_changed_rule_ids,
    isolate_listener_profile,
)
from earshift_bakeoff.bilingual_product_matrix import (
    load_bilingual_product_matrix,
    load_bilingual_structural_state,
)
from earshift_bakeoff.bilingual_vowel_acoustics import run_praat_formant_frames
from earshift_bakeoff.bilingual_vowel_acoustics_v2 import (
    MINIMUM_RHOTIC_ANCHOR_SEPARATION_BARK_RMS,
    MINIMUM_RHOTIC_CONTROLLED_MOVEMENT_BARK_RMS,
    VOWEL_ACOUSTIC_VERSION_V2,
    classify_stress_core_endpoint,
    measurement_mode,
    stress_core_measurement,
)
from earshift_bakeoff.bilingual_vowel_engine import (
    BilingualVowelRender,
    _load_pinned_synthesis_voice,
)
from earshift_bakeoff.config import Paths, stable_json
from earshift_bakeoff.controlled_listener_synthesis import render_natural_condition
from earshift_bakeoff.controlled_vowel_synthesis_v2 import (
    CONTROLLED_VOWEL_SYNTHESIS_VERSION,
)
from earshift_bakeoff.kokoro_gate_bridge import KokoroGateIndex
from earshift_bakeoff.kokoro_synthesis import (
    CONFIG_FILE,
    SAMPLE_RATE_HZ,
    pcm16_bytes,
    verify_model_files,
)
from earshift_bakeoff.listener_lens import DatabaseNonceChecker
from earshift_bakeoff.portuguese_carrier_planner_v1 import (
    PortuguesePositiveOnlyIndexV1,
)
from earshift_bakeoff.util import atomic_write_json, sha256_file

from run_bilingual_product_audio_integrity_screen_v1 import (
    FixtureAdapter,
    _tuple,
    _universal_pass,
)


PROTOCOL_VERSION = "bilingual-product-v8-vowel-acoustic-screen"
PROTOCOL_PATH = Paths().root / "rules" / f"{PROTOCOL_VERSION}.json"
RUN_ID = "20260717-bilingual-product-v8-vowel-acoustic-screen"
RUN_DIR = Paths().artifacts / "product-matrix" / RUN_ID
V8_MANIFEST_PATH = (
    Paths().artifacts
    / "product-matrix"
    / "20260717-bilingual-product-v8-vowel-manifest"
    / "manifest.json"
)
V7_AUDIO_RESULT_PATH = (
    Paths().artifacts
    / "product-matrix"
    / "20260717-bilingual-product-isolated-audio-screen-v1"
    / "results.json"
)
V7_AUDIO_DIR = V7_AUDIO_RESULT_PATH.parent
V1_ACOUSTIC_RESULT_PATH = (
    Paths().artifacts
    / "product-matrix"
    / "20260717-bilingual-product-vowel-acoustic-screen-v1"
    / "results.json"
)
V1_ACOUSTIC_DIR = V1_ACOUSTIC_RESULT_PATH.parent
PRAAT_PATH = Path("/Applications/Praat.app/Contents/MacOS/Praat")
PRAAT_SCRIPT_PATH = (
    Paths().root / "scripts" / "praat_bilingual_vowel_trajectory_v1.praat"
)
FORMANT_CEILINGS_HZ = (5500, 5750, 6000)
VOICE_ORDER = ("af_heart", "am_michael", "pm_alex", "pf_dora")
RHOTIC_RULE_IDS = frozenset(("enpt.rhotic_schwa_reduced_a",))


def _semantic_hash(payload: Any) -> str:
    return hashlib.sha256(stable_json(payload).encode("utf-8")).hexdigest()


def _safe_name(value: str) -> str:
    return "".join(character if character.isalnum() else "-" for character in value)


def _pcm_hash(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype="<i2").tobytes()).hexdigest()


def _read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as handle:
        if (
            handle.getnchannels() != 1
            or handle.getsampwidth() != 2
            or handle.getframerate() != SAMPLE_RATE_HZ
        ):
            raise RuntimeError(f"unexpected WAV format: {path}")
        return np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2").copy()


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


def _natural_pcm(audio: np.ndarray) -> np.ndarray:
    return np.frombuffer(pcm16_bytes(audio), dtype="<i2").copy()


def _load_protocol(
    *,
    matrix_sha256: str,
    manifest: dict[str, Any],
    v7_audio: dict[str, Any],
    v1_acoustic: dict[str, Any],
) -> dict[str, Any]:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    expected_keys = {
        "schema_version",
        "protocol_version",
        "status",
        "production_enabled",
        "defect_and_intervention",
        "source_data_bindings",
        "scope",
        "render_policy",
        "instrument",
        "measurement_protocol",
        "analysis_gates",
        "aggregation_policy",
        "claim_limits",
        "stopping_rule",
        "source_bindings",
    }
    if set(protocol) != expected_keys:
        raise RuntimeError("v8 vowel protocol schema drifted")
    source = protocol["source_data_bindings"]
    if (
        protocol["schema_version"] != 1
        or protocol["protocol_version"] != PROTOCOL_VERSION
        or protocol["status"] != "frozen_before_first_broad_v8_vowel_audio_render"
        or protocol["production_enabled"] is not False
        or source["matrix_sha256"] != matrix_sha256
        or source["v8_manifest_sha256"] != sha256_file(V8_MANIFEST_PATH)
        or source["v8_manifest_record_sha256"] != manifest["record_sha256"]
        or source["v7_audio_result_sha256"] != sha256_file(V7_AUDIO_RESULT_PATH)
        or source["v7_audio_record_sha256"] != v7_audio["record_sha256"]
        or source["v1_acoustic_result_sha256"] != sha256_file(V1_ACOUSTIC_RESULT_PATH)
        or source["v1_acoustic_record_sha256"] != v1_acoustic["record_sha256"]
        or protocol["scope"]["logical_slot_count"] != 240
        or protocol["scope"]["voice_rule_cell_count"] != 80
        or tuple(protocol["scope"]["voices_in_order"]) != VOICE_ORDER
        or tuple(protocol["instrument"]["formant_ceilings_hz"]) != FORMANT_CEILINGS_HZ
        or protocol["render_policy"]["api_calls_allowed"] != 0
        or protocol["render_policy"]["replacement_slots_allowed"] is not False
        or protocol["aggregation_policy"]["product_promotion_allowed"] is not False
    ):
        raise RuntimeError("v8 vowel protocol binding drifted")
    if sha256_file(PRAAT_PATH) != protocol["instrument"]["praat_binary_sha256"]:
        raise RuntimeError("Praat binary changed after v8 vowel freeze")
    if sha256_file(PRAAT_SCRIPT_PATH) != protocol["instrument"]["praat_script_sha256"]:
        raise RuntimeError("Praat script changed after v8 vowel freeze")
    for binding in protocol["source_bindings"]:
        if sha256_file(Paths().root / binding["path"]) != binding["sha256"]:
            raise RuntimeError(f"v8 vowel source drifted: {binding['path']}")
    return protocol


def _planner_v8(
    *,
    slot: dict[str, Any],
    profiles: dict[str, dict[str, Any]],
    model_vocab: set[str],
    nonce_checker: DatabaseNonceChecker,
    phone_indexes: tuple[Any, ...],
) -> BilingualListenerPlannerV8:
    fixture = slot["fixture_spec"]
    base_profile = profiles[slot["profile_id"]]
    profile = isolate_listener_profile(base_profile, slot["rule_id"])
    return BilingualListenerPlannerV8(
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


def _measure_condition(
    *,
    wav_path: Path,
    stem: str,
    intervals: Sequence[dict[str, Any]],
    ceiling: int,
    mode: str,
) -> tuple[dict[str, Any], ...]:
    frame_path = RUN_DIR / "praat-frames" / f"{stem}__ceiling-{ceiling}.tsv"
    frames = run_praat_formant_frames(
        wav_path,
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


def _analysis_classification(
    *,
    source: dict[str, Any],
    target: dict[str, Any],
    neutral: dict[str, Any],
    lens: dict[str, Any],
    rhotic: bool,
) -> dict[str, Any]:
    measurements = (source, target, neutral, lens)
    if not all(row["measurable"] for row in measurements):
        return {
            "classification": "measurement_exclusion",
            "directional_pass": False,
            "exact_category_pass": False,
            "base_vowel": None,
            "rhoticity": None,
        }
    base = classify_stress_core_endpoint(
        source_anchor=source["feature_bark"],
        target_anchor=target["feature_bark"],
        neutral=neutral["feature_bark"],
        lens=lens["feature_bark"],
    )
    rhoticity: dict[str, Any] | None = None
    if rhotic:
        if not all(row["rhoticity_gap_bark"] is not None for row in measurements):
            return {
                "classification": "measurement_exclusion",
                "directional_pass": False,
                "exact_category_pass": False,
                "base_vowel": base,
                "rhoticity": None,
            }
        rhoticity = classify_stress_core_endpoint(
            source_anchor=source["rhoticity_gap_bark"],
            target_anchor=target["rhoticity_gap_bark"],
            neutral=neutral["rhoticity_gap_bark"],
            lens=lens["rhoticity_gap_bark"],
            minimum_anchor_separation=(MINIMUM_RHOTIC_ANCHOR_SEPARATION_BARK_RMS),
            minimum_controlled_movement=(MINIMUM_RHOTIC_CONTROLLED_MOVEMENT_BARK_RMS),
        )
    components = (base,) if rhoticity is None else (base, rhoticity)
    exact = all(row["exact_category_pass"] for row in components)
    directional = all(row["directional_pass"] for row in components)
    return {
        "classification": (
            "exact_category_pass"
            if exact
            else "directional_only_pass"
            if directional
            else "fail"
        ),
        "directional_pass": directional,
        "exact_category_pass": exact,
        "base_vowel": base,
        "rhoticity": rhoticity,
    }


def _aggregate(rows: Sequence[dict[str, Any]]) -> str:
    if rows and all(row["exact_category_pass"] for row in rows):
        return "exact_category_pass"
    if rows and all(row["directional_pass"] for row in rows):
        return "directional_only_pass"
    return "fail"


def _slot_outcome(
    *,
    slot: dict[str, Any],
    v7_audio: dict[str, Any],
    v1_acoustic: dict[str, Any],
    planner: BilingualListenerPlannerV8,
    synthesis: Any,
) -> dict[str, Any]:
    started = time.perf_counter()
    runtime = BilingualListenerRuntimeV8(planner=planner, synthesis=synthesis)
    rendered = runtime.render(slot["fixture_spec"]["text"])
    if not isinstance(rendered, BilingualVowelRender):
        raise RuntimeError("v8 vowel slot produced no comparison")
    if rendered.plan.plan_sha256 != slot["v8_plan_sha256"]:
        raise RuntimeError("v8 vowel plan changed after manifest freeze")
    if active_changed_rule_ids(rendered.plan) != (slot["rule_id"],):
        raise RuntimeError("v8 vowel plan no longer isolates one rule")
    rows = [
        row
        for row in rendered.alignment["target_occurrences"]
        if row["segment_type"] == "vowel" and row["rule_id"] == slot["rule_id"]
    ]
    if not _universal_pass(rendered, rows):
        raise RuntimeError("v8 vowel render failed universal integrity")
    v7_neutral_path = V7_AUDIO_DIR / v7_audio["audio"]["neutral"]["relative_path"]
    v7_neutral_pcm = _read_wav(v7_neutral_path)
    if _pcm_hash(v7_neutral_pcm) != v7_audio["audio"]["neutral"]["pcm_sha256"]:
        raise RuntimeError("v7 neutral source PCM changed")
    neutral_unchanged = bool(np.array_equal(rendered.neutral_pcm, v7_neutral_pcm))
    source_anchor = render_natural_condition(
        synthesis,
        phonemes=rendered.plan.neutral_phonemes,
        reference_phonemes=rendered.plan.render_reference_phonemes,
    )
    target_anchor = render_natural_condition(
        synthesis,
        phonemes=rendered.plan.lens_phonemes,
        reference_phonemes=rendered.plan.render_reference_phonemes,
    )
    source_pcm = _natural_pcm(source_anchor.audio)
    target_pcm = _natural_pcm(target_anchor.audio)
    source_anchor_path = (
        V1_ACOUSTIC_DIR / v1_acoustic["anchor_audio"]["source"]["relative_path"]
    )
    target_anchor_path = (
        V1_ACOUSTIC_DIR / v1_acoustic["anchor_audio"]["target"]["relative_path"]
    )
    anchor_reuse_pass = bool(
        _pcm_hash(source_pcm) == v1_acoustic["anchor_audio"]["source"]["pcm_sha256"]
        and _pcm_hash(target_pcm) == v1_acoustic["anchor_audio"]["target"]["pcm_sha256"]
        and sha256_file(source_anchor_path)
        == v1_acoustic["anchor_audio"]["source"]["wav_sha256"]
        and sha256_file(target_anchor_path)
        == v1_acoustic["anchor_audio"]["target"]["wav_sha256"]
        and np.array_equal(source_pcm, rendered.neutral_pcm)
    )
    source_alignment = bilingual_alignment_record_v8(
        model=synthesis.model,
        plan=rendered.plan,
        durations=source_anchor.predicted_durations,
        sample_count=source_pcm.size,
    )
    target_alignment = bilingual_alignment_record_v8(
        model=synthesis.model,
        plan=rendered.plan,
        durations=target_anchor.predicted_durations,
        sample_count=target_pcm.size,
    )
    source_rows = [
        row
        for row in source_alignment["target_occurrences"]
        if row["segment_type"] == "vowel" and row["rule_id"] == slot["rule_id"]
    ]
    target_rows = [
        row
        for row in target_alignment["target_occurrences"]
        if row["segment_type"] == "vowel" and row["rule_id"] == slot["rule_id"]
    ]
    if not len(rows) == len(source_rows) == len(target_rows):
        raise RuntimeError("v8 vowel occurrence counts differ across conditions")
    stem = _safe_name(slot["logical_slot_id"])
    neutral_path = RUN_DIR / "audio" / f"{stem}__neutral.wav"
    lens_path = RUN_DIR / "audio" / f"{stem}__lens.wav"
    audio = {
        "neutral": _write_wav(neutral_path, rendered.neutral_pcm),
        "lens": _write_wav(lens_path, rendered.lens_pcm),
        "reused_source_anchor": v1_acoustic["anchor_audio"]["source"],
        "reused_target_anchor": v1_acoustic["anchor_audio"]["target"],
    }
    mode = measurement_mode(slot["source"], slot["target"])
    controlled_intervals = tuple(row["measurement_interval"] for row in rows)
    source_intervals = tuple(row["measurement_interval"] for row in source_rows)
    target_intervals = tuple(row["measurement_interval"] for row in target_rows)
    by_ceiling = []
    for ceiling in FORMANT_CEILINGS_HZ:
        measurements = {
            "source_anchor": _measure_condition(
                wav_path=source_anchor_path,
                stem=f"{stem}__natural-source",
                intervals=source_intervals,
                ceiling=ceiling,
                mode=mode,
            ),
            "target_anchor": _measure_condition(
                wav_path=target_anchor_path,
                stem=f"{stem}__natural-target",
                intervals=target_intervals,
                ceiling=ceiling,
                mode=mode,
            ),
            "neutral": _measure_condition(
                wav_path=neutral_path,
                stem=f"{stem}__controlled-neutral",
                intervals=controlled_intervals,
                ceiling=ceiling,
                mode=mode,
            ),
            "lens": _measure_condition(
                wav_path=lens_path,
                stem=f"{stem}__controlled-lens",
                intervals=controlled_intervals,
                ceiling=ceiling,
                mode=mode,
            ),
        }
        classifications = tuple(
            _analysis_classification(
                source=measurements["source_anchor"][index],
                target=measurements["target_anchor"][index],
                neutral=measurements["neutral"][index],
                lens=measurements["lens"][index],
                rhotic=slot["rule_id"] in RHOTIC_RULE_IDS,
            )
            for index in range(len(rows))
        )
        by_ceiling.append(
            {
                "maximum_formant_hz": ceiling,
                "measurements": measurements,
                "occurrence_classifications": classifications,
            }
        )
    occurrence_outcomes = []
    for index, row in enumerate(rows):
        analyses = tuple(
            ceiling["occurrence_classifications"][index] for ceiling in by_ceiling
        )
        classification = _aggregate(analyses)
        occurrence_outcomes.append(
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
    slot_classification = (
        _aggregate(occurrence_outcomes)
        if neutral_unchanged and anchor_reuse_pass
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
        "measurement_mode": mode,
        "classification": slot_classification,
        "exact_category_pass": slot_classification == "exact_category_pass",
        "directional_pass": slot_classification
        in {"exact_category_pass", "directional_only_pass"},
        "v8_plan_sha256": rendered.plan.plan_sha256,
        "vowel_unit_columns": rendered.prosody["vowel_columns"],
        "vowel_state_columns": rendered.prosody["vowel_state_columns"],
        "neutral_pcm_unchanged_from_v7": neutral_unchanged,
        "v1_natural_anchor_reuse_pass": anchor_reuse_pass,
        "verification": asdict(rendered.verification),
        "splice_windows": rendered.splice_windows,
        "audio": audio,
        "source_anchor_intervals": source_intervals,
        "target_anchor_intervals": target_intervals,
        "occurrence_outcomes": occurrence_outcomes,
        "analysis_by_formant_ceiling": by_ceiling,
        "elapsed_s": time.perf_counter() - started,
        "api_calls_made": 0,
        "product_enabled": False,
    }


def _cell_summaries(outcomes: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in outcomes:
        groups.setdefault(row["cell_id"], []).append(row)
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
                "classification": classification,
                "claim_limit": (
                    "oral_trajectory_only_nasality_unvalidated_not_eligible"
                    if nasal
                    else "stress_core_f1_f2_and_rhoticity"
                    if rows[0]["rule_id"] in RHOTIC_RULE_IDS
                    else "stress_core_f1_f2"
                ),
                "automatic_human_qc_eligible": eligible,
                "human_qc_status": "pending" if eligible else "not_eligible",
                "product_enabled": False,
            }
        )
    return sorted(summaries, key=lambda row: row["cell_id"])


def main() -> None:
    if RUN_DIR.exists():
        raise RuntimeError(f"refusing to overwrite frozen v8 vowel run: {RUN_DIR}")
    matrix = load_bilingual_product_matrix()
    load_bilingual_structural_state(matrix)
    manifest = json.loads(V8_MANIFEST_PATH.read_text(encoding="utf-8"))
    v7_audio = json.loads(V7_AUDIO_RESULT_PATH.read_text(encoding="utf-8"))
    v1_acoustic = json.loads(V1_ACOUSTIC_RESULT_PATH.read_text(encoding="utf-8"))
    protocol = _load_protocol(
        matrix_sha256=matrix.matrix_sha256,
        manifest=manifest,
        v7_audio=v7_audio,
        v1_acoustic=v1_acoustic,
    )
    if (
        manifest["classification"] != "all_v8_vowel_slots_frozen_with_stress_context"
        or manifest["pass_count"] != 240
    ):
        raise RuntimeError("v8 vowel manifest is incomplete")
    slots = manifest["slots"]
    v7_by_id = {row["logical_slot_id"]: row for row in v7_audio["outcomes"]}
    v1_by_id = {row["logical_slot_id"]: row for row in v1_acoustic["outcomes"]}
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
                        v7_audio=v7_by_id[slot["logical_slot_id"]],
                        v1_acoustic=v1_by_id[slot["logical_slot_id"]],
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
                        "classification": "fail",
                        "error_code": getattr(exc, "code", type(exc).__name__),
                        "error": str(exc),
                        "api_calls_made": 0,
                        "product_enabled": False,
                    }
                )
    cells = _cell_summaries(outcomes)
    counts = {
        label: sum(row["classification"] == label for row in cells)
        for label in ("exact_category_pass", "directional_only_pass", "fail")
    }
    result = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "protocol_version": PROTOCOL_VERSION,
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "v8_candidate_version": BILINGUAL_LISTENER_CANDIDATE_VERSION_V8,
        "controlled_vowel_synthesis_version": CONTROLLED_VOWEL_SYNTHESIS_VERSION,
        "measurement_alignment_version": VOWEL_MEASUREMENT_ALIGNMENT_VERSION,
        "acoustic_version": VOWEL_ACOUSTIC_VERSION_V2,
        "matrix_sha256": matrix.matrix_sha256,
        "v8_manifest_sha256": sha256_file(V8_MANIFEST_PATH),
        "v8_manifest_record_sha256": manifest["record_sha256"],
        "classification": "v8_vowel_acoustic_screen_complete_no_product_promotion",
        "logical_slot_count": len(outcomes),
        "measured_slot_count": sum(row["status"] == "measured" for row in outcomes),
        "error_slot_count": sum(row["status"] != "measured" for row in outcomes),
        "voice_rule_cell_count": len(cells),
        "cell_classification_counts": counts,
        "automatic_human_qc_eligible_cell_count": sum(
            row["automatic_human_qc_eligible"] for row in cells
        ),
        "v8_render_set_count": sum(row["status"] == "measured" for row in outcomes),
        "reused_anchor_pair_count": sum(
            row["status"] == "measured" for row in outcomes
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
                "logical_slot_count": len(outcomes),
                "measured_slot_count": result["measured_slot_count"],
                "error_slot_count": result["error_slot_count"],
                "cell_classification_counts": counts,
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
