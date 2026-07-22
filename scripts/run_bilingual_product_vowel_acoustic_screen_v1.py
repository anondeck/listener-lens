#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
from typing import Any, Sequence
import wave

import numpy as np

from earshift_bakeoff.bilingual_listener_engine import (
    BilingualListenerRuntime,
    load_listener_profiles,
)
from earshift_bakeoff.bilingual_product_isolation import active_changed_rule_ids
from earshift_bakeoff.bilingual_product_matrix import (
    load_bilingual_product_matrix,
    load_bilingual_structural_state,
)
from earshift_bakeoff.bilingual_vowel_acoustics import (
    MINIMUM_RHOTIC_ANCHOR_SEPARATION_BARK_RMS,
    MINIMUM_RHOTIC_CONTROLLED_MOVEMENT_BARK_RMS,
    VOWEL_TRAJECTORY_ACOUSTIC_VERSION,
    classify_vowel_endpoint,
    run_praat_formant_frames,
    trajectory_measurement,
)
from earshift_bakeoff.bilingual_vowel_engine import (
    BilingualVowelPlan,
    _load_pinned_synthesis_voice,
)
from earshift_bakeoff.config import Paths, stable_json
from earshift_bakeoff.consonant_acoustics import decoder_column_interval
from earshift_bakeoff.controlled_listener_synthesis import render_natural_condition
from earshift_bakeoff.kokoro_gate_bridge import KokoroGateIndex
from earshift_bakeoff.kokoro_synthesis import (
    CONFIG_FILE,
    SAMPLE_RATE_HZ,
    _word_column_spans,
    pcm16_bytes,
    verify_model_files,
)
from earshift_bakeoff.listener_lens import DatabaseNonceChecker
from earshift_bakeoff.portuguese_carrier_planner_v1 import (
    PortuguesePositiveOnlyIndexV1,
)
from earshift_bakeoff.util import atomic_write_json, sha256_file

from run_bilingual_product_isolated_audio_screen_v1 import _planner


PROTOCOL_VERSION = "bilingual-product-vowel-acoustic-screen-v1"
PROTOCOL_PATH = Paths().root / "rules" / f"{PROTOCOL_VERSION}.json"
RUN_ID = "20260717-bilingual-product-vowel-acoustic-screen-v1"
RUN_DIR = Paths().artifacts / "product-matrix" / RUN_ID
MANIFEST_PATH = (
    Paths().artifacts
    / "product-matrix"
    / "20260717-bilingual-product-isolated-acoustic-manifest-v1"
    / "manifest.json"
)
ISOLATED_AUDIO_RESULT_PATH = (
    Paths().artifacts
    / "product-matrix"
    / "20260717-bilingual-product-isolated-audio-screen-v1"
    / "results.json"
)
ISOLATED_AUDIO_DIR = ISOLATED_AUDIO_RESULT_PATH.parent
PRAAT_PATH = Path("/Applications/Praat.app/Contents/MacOS/Praat")
PRAAT_SCRIPT_PATH = (
    Paths().root / "scripts" / "praat_bilingual_vowel_trajectory_v1.praat"
)
FORMANT_CEILINGS_HZ = (5000, 5500, 6000)
VOICE_ORDER = ("af_heart", "am_michael", "pm_alex", "pf_dora")
RHOTIC_RULE_IDS = frozenset(("enpt.rhotic_schwa_reduced_a",))


def _semantic_hash(payload: Any) -> str:
    return hashlib.sha256(stable_json(payload).encode("utf-8")).hexdigest()


def _safe_name(value: str) -> str:
    return "".join(character if character.isalnum() else "-" for character in value)


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
        "pcm_sha256": hashlib.sha256(values.tobytes()).hexdigest(),
        "sample_count": int(values.size),
        "duration_s": values.size / SAMPLE_RATE_HZ,
    }


def _natural_pcm(audio: np.ndarray) -> np.ndarray:
    return np.frombuffer(pcm16_bytes(audio), dtype="<i2").copy()


def _load_protocol(
    *, matrix_sha256: str, manifest: dict[str, Any], isolated_audio: dict[str, Any]
) -> dict[str, Any]:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    expected_keys = {
        "schema_version",
        "protocol_version",
        "status",
        "production_enabled",
        "source_data_bindings",
        "scope",
        "natural_anchor_policy",
        "instrument",
        "measurement_protocol",
        "analysis_gates",
        "aggregation_policy",
        "claim_limits",
        "stopping_rule",
        "source_bindings",
    }
    if set(protocol) != expected_keys:
        raise RuntimeError("vowel acoustic protocol schema drifted")
    source_data = protocol["source_data_bindings"]
    if (
        protocol["schema_version"] != 1
        or protocol["protocol_version"] != PROTOCOL_VERSION
        or protocol["status"]
        != "frozen_before_first_vowel_anchor_render_or_measurement"
        or protocol["production_enabled"] is not False
        or source_data["matrix_sha256"] != matrix_sha256
        or source_data["isolated_manifest_sha256"] != sha256_file(MANIFEST_PATH)
        or source_data["isolated_manifest_record_sha256"] != manifest["record_sha256"]
        or source_data["isolated_audio_result_sha256"]
        != sha256_file(ISOLATED_AUDIO_RESULT_PATH)
        or source_data["isolated_audio_record_sha256"]
        != isolated_audio["record_sha256"]
        or protocol["scope"]["logical_slot_count"] != 240
        or protocol["scope"]["voice_rule_cell_count"] != 80
        or tuple(protocol["scope"]["voices_in_order"]) != VOICE_ORDER
        or tuple(protocol["instrument"]["formant_ceilings_hz"]) != FORMANT_CEILINGS_HZ
        or protocol["natural_anchor_policy"]["api_calls_allowed"] != 0
        or protocol["natural_anchor_policy"]["replacement_anchor_renders_allowed"]
        is not False
        or protocol["aggregation_policy"]["product_promotion_allowed"] is not False
    ):
        raise RuntimeError("vowel acoustic protocol binding drifted")
    if sha256_file(PRAAT_PATH) != protocol["instrument"]["praat_binary_sha256"]:
        raise RuntimeError("Praat binary changed after vowel protocol freeze")
    if sha256_file(PRAAT_SCRIPT_PATH) != protocol["instrument"]["praat_script_sha256"]:
        raise RuntimeError("Praat script changed after vowel protocol freeze")
    for binding in protocol["source_bindings"]:
        if sha256_file(Paths().root / binding["path"]) != binding["sha256"]:
            raise RuntimeError(f"vowel acoustic source drifted: {binding['path']}")
    return protocol


def _condition_columns(
    *,
    model: Any,
    plan: BilingualVowelPlan,
    rule_id: str,
    side: str,
) -> tuple[tuple[int, ...], ...]:
    if side not in {"source", "target"}:
        raise ValueError("condition side must be source or target")
    phonemes = plan.neutral_phonemes if side == "source" else plan.lens_phonemes
    spans = _word_column_spans(model, phonemes)
    if len(spans) != len(plan.words):
        raise RuntimeError("natural anchor word alignment drifted")
    selected: list[tuple[int, ...]] = []
    for word_index in plan.target_word_indexes:
        word = plan.words[word_index]
        columns = spans[word_index]
        phone = word.neutral_phone if side == "source" else word.lens_phone
        if len(columns) != len(phone):
            raise RuntimeError("natural anchor phone columns drifted")
        for occurrence in word.vowel_occurrences:
            if not occurrence.changed or occurrence.rule_id != rule_id:
                continue
            token = occurrence.source if side == "source" else occurrence.target
            start = occurrence.phone_offset
            stop = start + len(token)
            if len(token) != occurrence.phone_length or phone[start:stop] != token:
                raise RuntimeError("natural anchor occurrence alignment drifted")
            occurrence_columns = tuple(columns[start:stop])
            if len(occurrence_columns) != len(token):
                raise RuntimeError("natural anchor target lost decoder columns")
            selected.append(occurrence_columns)
    if not selected:
        raise RuntimeError("natural anchor has no named changed vowel occurrence")
    return tuple(selected)


def _anchor_intervals(
    *,
    model: Any,
    plan: BilingualVowelPlan,
    rule_id: str,
    side: str,
    durations: Sequence[int],
    sample_count: int,
) -> tuple[dict[str, Any], ...]:
    return tuple(
        decoder_column_interval(
            durations,
            columns,
            sample_count=sample_count,
            sample_rate_hz=SAMPLE_RATE_HZ,
        ).as_record()
        for columns in _condition_columns(
            model=model,
            plan=plan,
            rule_id=rule_id,
            side=side,
        )
    )


def _measure_condition(
    *,
    wav_path: Path,
    stem: str,
    intervals: Sequence[dict[str, Any]],
    maximum_formant_hz: int,
) -> tuple[dict[str, Any], ...]:
    frame_path = RUN_DIR / "praat-frames" / f"{stem}__ceiling-{maximum_formant_hz}.tsv"
    frames = run_praat_formant_frames(
        wav_path,
        frame_path,
        maximum_formant_hz=maximum_formant_hz,
        praat_path=PRAAT_PATH,
        script_path=PRAAT_SCRIPT_PATH,
    )
    return tuple(
        trajectory_measurement(
            frames,
            start_s=float(interval["start_s"]),
            end_s=float(interval["end_s"]),
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
    base = classify_vowel_endpoint(
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
        rhoticity = classify_vowel_endpoint(
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
    audio_outcome: dict[str, Any],
    planner: Any,
    synthesis: Any,
) -> dict[str, Any]:
    started = time.perf_counter()
    plan = planner.plan(slot["fixture_spec"]["text"])
    if plan.plan_sha256 != slot["isolated_plan_sha256"]:
        raise RuntimeError("isolated vowel plan changed after manifest freeze")
    if active_changed_rule_ids(plan) != (slot["rule_id"],):
        raise RuntimeError("isolated vowel plan no longer has one changed rule")
    target_rows = [
        row
        for row in audio_outcome["target_occurrences"]
        if row["segment_type"] == "vowel" and row["rule_id"] == slot["rule_id"]
    ]
    neutral_path = (
        ISOLATED_AUDIO_DIR / audio_outcome["audio"]["neutral"]["relative_path"]
    )
    lens_path = ISOLATED_AUDIO_DIR / audio_outcome["audio"]["lens"]["relative_path"]
    if (
        sha256_file(neutral_path) != audio_outcome["audio"]["neutral"]["wav_sha256"]
        or sha256_file(lens_path) != audio_outcome["audio"]["lens"]["wav_sha256"]
    ):
        raise RuntimeError("isolated vowel WAV changed after result freeze")
    neutral_pcm = _read_wav(neutral_path)
    lens_pcm = _read_wav(lens_path)
    if lens_pcm.size != neutral_pcm.size:
        raise RuntimeError("controlled vowel pair sample count changed")
    source_anchor = render_natural_condition(
        synthesis,
        phonemes=plan.neutral_phonemes,
        reference_phonemes=plan.render_reference_phonemes,
    )
    target_anchor = render_natural_condition(
        synthesis,
        phonemes=plan.lens_phonemes,
        reference_phonemes=plan.render_reference_phonemes,
    )
    source_pcm = _natural_pcm(source_anchor.audio)
    target_pcm = _natural_pcm(target_anchor.audio)
    stem = _safe_name(slot["logical_slot_id"])
    source_path = RUN_DIR / "anchors" / f"{stem}__natural-source.wav"
    target_path = RUN_DIR / "anchors" / f"{stem}__natural-target.wav"
    source_record = _write_wav(source_path, source_pcm)
    target_record = _write_wav(target_path, target_pcm)
    source_intervals = _anchor_intervals(
        model=synthesis.model,
        plan=plan,
        rule_id=slot["rule_id"],
        side="source",
        durations=source_anchor.predicted_durations,
        sample_count=source_pcm.size,
    )
    target_intervals = _anchor_intervals(
        model=synthesis.model,
        plan=plan,
        rule_id=slot["rule_id"],
        side="target",
        durations=target_anchor.predicted_durations,
        sample_count=target_pcm.size,
    )
    controlled_intervals = tuple(row["measurement_interval"] for row in target_rows)
    occurrence_count = len(target_rows)
    if not (
        occurrence_count
        == len(source_intervals)
        == len(target_intervals)
        == len(controlled_intervals)
    ):
        raise RuntimeError("vowel occurrence count differs across conditions")
    source_identity_pass = bool(np.array_equal(source_pcm, neutral_pcm))
    anchor_integrity_pass = bool(
        source_pcm.size > 0
        and target_pcm.size > 0
        and np.isfinite(source_pcm.astype(np.float64)).all()
        and np.isfinite(target_pcm.astype(np.float64)).all()
        and float(np.mean(np.abs(source_pcm.astype(np.int64)) >= 32767)) < 0.001
        and float(np.mean(np.abs(target_pcm.astype(np.int64)) >= 32767)) < 0.001
    )
    rhotic = slot["rule_id"] in RHOTIC_RULE_IDS
    by_ceiling: list[dict[str, Any]] = []
    for ceiling in FORMANT_CEILINGS_HZ:
        measurements = {
            "source_anchor": _measure_condition(
                wav_path=source_path,
                stem=f"{stem}__natural-source",
                intervals=source_intervals,
                maximum_formant_hz=ceiling,
            ),
            "target_anchor": _measure_condition(
                wav_path=target_path,
                stem=f"{stem}__natural-target",
                intervals=target_intervals,
                maximum_formant_hz=ceiling,
            ),
            "neutral": _measure_condition(
                wav_path=neutral_path,
                stem=f"{stem}__controlled-neutral",
                intervals=controlled_intervals,
                maximum_formant_hz=ceiling,
            ),
            "lens": _measure_condition(
                wav_path=lens_path,
                stem=f"{stem}__controlled-lens",
                intervals=controlled_intervals,
                maximum_formant_hz=ceiling,
            ),
        }
        classifications = tuple(
            _analysis_classification(
                source=measurements["source_anchor"][index],
                target=measurements["target_anchor"][index],
                neutral=measurements["neutral"][index],
                lens=measurements["lens"][index],
                rhotic=rhotic,
            )
            for index in range(occurrence_count)
        )
        by_ceiling.append(
            {
                "maximum_formant_hz": ceiling,
                "measurements": measurements,
                "occurrence_classifications": classifications,
            }
        )
    occurrence_outcomes = []
    for index in range(occurrence_count):
        analysis_rows = tuple(
            ceiling["occurrence_classifications"][index] for ceiling in by_ceiling
        )
        classification = _aggregate(analysis_rows)
        occurrence_outcomes.append(
            {
                "occurrence_index": index,
                "word_index": target_rows[index]["word_index"],
                "source": target_rows[index]["source"],
                "target": target_rows[index]["target"],
                "classification": classification,
                "exact_category_pass": classification == "exact_category_pass",
                "directional_pass": classification
                in {"exact_category_pass", "directional_only_pass"},
                "all_formant_ceilings_required": True,
            }
        )
    slot_classification = (
        _aggregate(occurrence_outcomes)
        if source_identity_pass and anchor_integrity_pass
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
        "classification": slot_classification,
        "exact_category_pass": slot_classification == "exact_category_pass",
        "directional_pass": slot_classification
        in {"exact_category_pass", "directional_only_pass"},
        "source_anchor_pcm_identical_to_controlled_neutral": source_identity_pass,
        "anchor_integrity_pass": anchor_integrity_pass,
        "anchor_audio": {"source": source_record, "target": target_record},
        "anchor_intervals": {
            "source": source_intervals,
            "target": target_intervals,
        },
        "controlled_intervals": controlled_intervals,
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
        occurrence_rows = [
            occurrence for row in measured for occurrence in row["occurrence_outcomes"]
        ]
        complete = bool(
            len(rows) == 3 and len(measured) == 3 and len(occurrence_rows) == 4
        )
        classification = _aggregate(measured) if complete else "fail"
        nasal = any("̃" in str(row[key]) for row in rows for key in ("source", "target"))
        rhotic = rows[0]["rule_id"] in RHOTIC_RULE_IDS
        automatic_eligibility = bool(
            complete
            and classification in {"exact_category_pass", "directional_only_pass"}
            and not nasal
        )
        claim_limit = (
            "oral_trajectory_only_nasality_unvalidated_not_eligible"
            if nasal
            else "rhotic_and_base_vowel_trajectory"
            if rhotic
            else "full_f1_f2_trajectory"
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
                "occurrence_count": len(occurrence_rows),
                "complete_three_context_four_occurrence_yield": complete,
                "classification": classification,
                "claim_limit": claim_limit,
                "automatic_human_qc_eligible": automatic_eligibility,
                "human_qc_status": "pending"
                if automatic_eligibility
                else "not_eligible",
                "product_enabled": False,
            }
        )
    return sorted(summaries, key=lambda row: row["cell_id"])


def main() -> None:
    if RUN_DIR.exists():
        raise RuntimeError(f"refusing to overwrite frozen vowel run: {RUN_DIR}")
    matrix = load_bilingual_product_matrix()
    load_bilingual_structural_state(matrix)
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    isolated_audio = json.loads(ISOLATED_AUDIO_RESULT_PATH.read_text(encoding="utf-8"))
    protocol = _load_protocol(
        matrix_sha256=matrix.matrix_sha256,
        manifest=manifest,
        isolated_audio=isolated_audio,
    )
    if (
        manifest["classification"] != "all_acoustic_slots_atomically_isolated"
        or isolated_audio["classification"]
        != "all_isolated_slots_universal_integrity_pass_family_acoustics_pending"
    ):
        raise RuntimeError("isolated vowel source data are not complete")
    slots = [slot for slot in manifest["slots"] if slot["family"] == "vowel"]
    if len(slots) != 240:
        raise RuntimeError("frozen vowel slot count changed")
    audio_by_id = {row["logical_slot_id"]: row for row in isolated_audio["outcomes"]}
    profiles = load_listener_profiles()
    files = verify_model_files(download=False)
    model_vocab = set(
        json.loads(files[CONFIG_FILE].read_text(encoding="utf-8"))["vocab"]
    )
    nonce_checker = DatabaseNonceChecker()
    phone_indexes = (KokoroGateIndex(), PortuguesePositiveOnlyIndexV1())
    outcomes: list[dict[str, Any]] = []
    RUN_DIR.mkdir(parents=True, exist_ok=False)
    run_started = time.perf_counter()
    for voice_id in VOICE_ORDER:
        synthesis = _load_pinned_synthesis_voice(voice_id)
        for slot in (row for row in slots if row["voice_id"] == voice_id):
            try:
                planner = _planner(
                    slot=slot,
                    profiles=profiles,
                    model_vocab=model_vocab,
                    nonce_checker=nonce_checker,
                    phone_indexes=phone_indexes,
                )
                runtime = BilingualListenerRuntime(planner=planner, synthesis=synthesis)
                if runtime.synthesis is not synthesis:
                    raise RuntimeError(
                        "vowel runtime changed its frozen synthesis voice"
                    )
                outcomes.append(
                    _slot_outcome(
                        slot=slot,
                        audio_outcome=audio_by_id[slot["logical_slot_id"]],
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
                        "status": "measurement_error",
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
        "acoustic_version": VOWEL_TRAJECTORY_ACOUSTIC_VERSION,
        "matrix_sha256": matrix.matrix_sha256,
        "isolated_manifest_sha256": sha256_file(MANIFEST_PATH),
        "isolated_manifest_record_sha256": manifest["record_sha256"],
        "isolated_audio_result_sha256": sha256_file(ISOLATED_AUDIO_RESULT_PATH),
        "isolated_audio_record_sha256": isolated_audio["record_sha256"],
        "classification": "vowel_acoustic_screen_complete_no_product_promotion",
        "logical_slot_count": len(outcomes),
        "measured_slot_count": sum(row["status"] == "measured" for row in outcomes),
        "measurement_error_slot_count": sum(
            row["status"] != "measured" for row in outcomes
        ),
        "voice_rule_cell_count": len(cells),
        "cell_classification_counts": counts,
        "automatic_human_qc_eligible_cell_count": sum(
            row["automatic_human_qc_eligible"] for row in cells
        ),
        "natural_anchor_render_count": sum(
            2 for row in outcomes if row["status"] == "measured"
        ),
        "api_calls_made": 0,
        "replacement_renders_used": 0,
        "elapsed_s": time.perf_counter() - run_started,
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
                "cell_classification_counts": counts,
                "automatic_human_qc_eligible_cell_count": result[
                    "automatic_human_qc_eligible_cell_count"
                ],
                "natural_anchor_render_count": result["natural_anchor_render_count"],
                "api_calls_made": 0,
                "elapsed_s": result["elapsed_s"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
