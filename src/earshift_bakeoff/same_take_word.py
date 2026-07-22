from __future__ import annotations

import hashlib
import json
import math
import subprocess
import tempfile
import wave
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .config import DEVLOG_PATH, Paths, stable_json
from .pcm import DecodedPcm16Wav, decode_pcm16_mono
from .same_take import (
    ANCHORS,
    MAX_FORMANTS_FAMILY,
    PRAAT,
    PRAAT_SHA256,
    RULE_ENDPOINTS,
    _cosine,
    _eligible_frame,
    _probe_frames,
    _rms_to_centroid,
    align_vowel_core,
    measure_formantpath,
)
from .util import atomic_write_json, sha256_file


RUN_ID = "20260715-same-take-word-v1"
PREREGISTRATION_HEADING = "## Same-take-word-v1 preregistration — July 15, 2026"
PREREGISTRATION_COMMIT = "7e078ab"
SOURCE_FREEZE_PATH = (
    Paths().artifacts / "same-take" / "20260715-same-take-v1" / "source-freeze.json"
)
SOURCE_FREEZE_SHA256 = "86db82049048cbdae978efa5604041fd78a6dd6d3ece9cfe42adfb4bfab9f9ce"
ACTIVE_WINDOW_S = 0.025
ACTIVE_STEP_S = 0.005
ACTIVE_RELATIVE_DB = -35.0
ANCHOR_FAMILY_COSINE_MIN = 0.75
SHIFT_FAMILY_COSINE_MIN = 0.75
STRENGTHS = (0.50, 0.75, 1.00)
EDITOR_SCRIPT = Paths().root / "scripts" / "praat_same_take_formantgrid.praat"
SHIFT_TAPER_S = 0.020
SPLICE_TAPER_S = 0.010
ANCHOR_DIRECTION_COSINE_MIN = 0.50
PLAUSIBILITY = {
    "f1_hz": (180.0, 1200.0),
    "f2_hz": (600.0, 3500.0),
    "minimum_f2_minus_f1_hz": 250.0,
}


@dataclass(frozen=True)
class WordSource:
    rule_id: str
    token: str
    shell: str
    take: int
    path: Path
    sha256: str


SOURCE_ROOT = (
    Paths().artifacts
    / "acoustic-calibration"
    / "20260715-calibration-v3-confirmatory"
    / "audio"
)
WORD_SOURCES = (
    WordSource(
        "ptbr.vowel.ae_to_eh",
        "vap",
        "v_V_p",
        2,
        SOURCE_ROOT / "007__contrast__ae_to_eh__v_V_p__neutral__take-2.wav",
        "1dda9767a9da6b3de5b3cc0feb29832df379cbf163f887b378b1e1c6fc7819a3",
    ),
    WordSource(
        "ptbr.vowel.ih_to_i",
        "vihp",
        "v_V_p",
        1,
        SOURCE_ROOT / "017__contrast__ih_to_i__v_V_p__neutral__take-1.wav",
        "29fc9446871907620c66e459665ed764720757d4273fc7c507c21fe07809e9c5",
    ),
)


def decoded_active_bounds(audio: DecodedPcm16Wav) -> dict[str, int | float]:
    window = max(1, round(ACTIVE_WINDOW_S * audio.sample_rate_hz))
    step = max(1, round(ACTIVE_STEP_S * audio.sample_rate_hz))
    frames: list[tuple[int, int, float]] = []
    widened = audio.samples.astype(np.float64, copy=False)
    for start in range(0, max(1, audio.decoded_sample_count - window + 1), step):
        end = min(audio.decoded_sample_count, start + window)
        values = widened[start:end]
        if values.size:
            rms = math.sqrt(float(np.mean(values * values)))
            frames.append((start, end, rms))
    if not frames:
        raise RuntimeError("decoded source has no complete RMS frame")
    peak = max(item[2] for item in frames)
    if peak <= 0:
        raise RuntimeError("decoded source is silent")
    threshold = peak * (10 ** (ACTIVE_RELATIVE_DB / 20))
    retained = [item for item in frames if item[2] >= threshold]
    if not retained:
        raise RuntimeError("decoded source has no active RMS frame")
    start = retained[0][0]
    end = retained[-1][1]
    return {
        "start_sample": start,
        "end_sample_exclusive": end,
        "start_s": start / audio.sample_rate_hz,
        "end_s": end / audio.sample_rate_hz,
        "sample_count": end - start,
        "window_samples": window,
        "step_samples": step,
        "relative_threshold_db": ACTIVE_RELATIVE_DB,
        "peak_rms": peak,
        "threshold_rms": threshold,
    }


def _point(record: dict[str, Any]) -> np.ndarray:
    return np.array([record["f1_bark"], record["f2_bark"]], dtype=float)


def _plausible(f1_hz: float, f2_hz: float) -> bool:
    return bool(
        PLAUSIBILITY["f1_hz"][0] <= f1_hz <= PLAUSIBILITY["f1_hz"][1]
        and PLAUSIBILITY["f2_hz"][0] <= f2_hz <= PLAUSIBILITY["f2_hz"][1]
        and f2_hz - f1_hz >= PLAUSIBILITY["minimum_f2_minus_f1_hz"]
    )


def _anchor_gates(anchor_records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by_category: dict[str, list[dict[str, Any]]] = {}
    for record in anchor_records:
        by_category.setdefault(record["category"], []).append(record)
    results: dict[str, Any] = {}
    for rule_id, (source_category, target_category) in RULE_ENDPOINTS.items():
        families: dict[str, Any] = {}
        vectors: list[np.ndarray] = []
        for maximum_formants in MAX_FORMANTS_FAMILY:
            key = f"{maximum_formants:.1f}"
            source_measurements = [item["measurements"][key] for item in by_category[source_category]]
            target_measurements = [item["measurements"][key] for item in by_category[target_category]]
            source_points = [_point(item) for item in source_measurements]
            target_points = [_point(item) for item in target_measurements]
            source_centroid = np.mean(np.stack(source_points), axis=0)
            target_centroid = np.mean(np.stack(target_points), axis=0)
            source_hz = np.mean(
                np.array([[item["f1_hz"], item["f2_hz"]] for item in source_measurements]),
                axis=0,
            )
            target_hz = np.mean(
                np.array([[item["f1_hz"], item["f2_hz"]] for item in target_measurements]),
                axis=0,
            )
            vector = target_centroid - source_centroid
            vectors.append(vector)
            variance = max(
                _rms_to_centroid(source_points, source_centroid),
                _rms_to_centroid(target_points, target_centroid),
            )
            magnitude = float(np.linalg.norm(vector))
            cross_take_cosines = [
                _cosine(target - source, vector)
                for source in source_points
                for target in target_points
            ]
            within_marin_pass = bool(
                magnitude > max(0.25, 2 * variance)
                and min(cross_take_cosines) >= ANCHOR_DIRECTION_COSINE_MIN
            )
            plausibility_pass = _plausible(*source_hz) and _plausible(*target_hz)
            sign_pass = bool(vector[0] < 0 and vector[1] > 0)
            families[key] = {
                "source_centroid_bark": source_centroid.tolist(),
                "target_centroid_bark": target_centroid.tolist(),
                "source_centroid_hz": source_hz.tolist(),
                "target_centroid_hz": target_hz.tolist(),
                "anchor_vector_bark": vector.tolist(),
                "endpoint_take_variance_bark": variance,
                "magnitude_bark": magnitude,
                "magnitude_threshold_bark": max(0.25, 2 * variance),
                "cross_take_cosines": cross_take_cosines,
                "within_marin_pass": within_marin_pass,
                "plausibility_pass": plausibility_pass,
                "front_vowel_sign_pass": sign_pass,
                "passed": within_marin_pass and plausibility_pass and sign_pass,
            }
        pairwise = [
            {
                "left": f"{MAX_FORMANTS_FAMILY[left]:.1f}",
                "right": f"{MAX_FORMANTS_FAMILY[right]:.1f}",
                "cosine": _cosine(vectors[left], vectors[right]),
            }
            for left, right in combinations(range(len(vectors)), 2)
        ]
        vector_consistency_pass = bool(
            pairwise and min(item["cosine"] for item in pairwise) >= ANCHOR_FAMILY_COSINE_MIN
        )
        results[rule_id] = {
            "families": families,
            "pairwise_anchor_vector_cosines": pairwise,
            "minimum_pairwise_cosine": ANCHOR_FAMILY_COSINE_MIN,
            "vector_consistency_pass": vector_consistency_pass,
            "passed": all(item["passed"] for item in families.values())
            and vector_consistency_pass,
        }
    return results


def _verify_inputs() -> dict[str, Any]:
    if PREREGISTRATION_HEADING not in DEVLOG_PATH.read_text(encoding="utf-8"):
        raise RuntimeError("same-take-word-v1 preregistration is missing")
    if not PRAAT.is_file() or sha256_file(PRAAT) != PRAAT_SHA256:
        raise RuntimeError("standalone Praat executable does not match the frozen hash")
    if not SOURCE_FREEZE_PATH.is_file() or sha256_file(SOURCE_FREEZE_PATH) != SOURCE_FREEZE_SHA256:
        raise RuntimeError("same-take-v1 anchor source freeze changed")
    for item in (*ANCHORS, *WORD_SOURCES):
        if not item.path.is_file() or sha256_file(item.path) != item.sha256:
            raise RuntimeError(f"frozen input mismatch: {item.path}")
    return json.loads(SOURCE_FREEZE_PATH.read_text(encoding="utf-8"))


def build_word_source_freeze(run_id: str = RUN_ID) -> dict[str, Any]:
    if run_id != RUN_ID:
        raise RuntimeError("same-take-word-v1 run ID is fixed")
    old_freeze = _verify_inputs()
    output_dir = Paths().artifacts / "same-take" / run_id
    output_path = output_dir / "source-freeze.json"
    if output_path.exists():
        raise RuntimeError("same-take-word-v1 source freeze already exists and is immutable")

    old_anchors = {
        (item["category"], int(item["take"])): item
        for item in old_freeze["anchors"]
    }
    anchor_records = []
    for anchor in ANCHORS:
        audio = decode_pcm16_mono(anchor.path)
        old = old_anchors[(anchor.category, anchor.take)]
        interval = old["interval"]
        measurements = {
            f"{maximum_formants:.1f}": measure_formantpath(
                anchor.path, interval, maximum_formants
            )
            for maximum_formants in MAX_FORMANTS_FAMILY
        }
        anchor_records.append(
            {
                "category": anchor.category,
                "ipa": anchor.ipa,
                "take": anchor.take,
                "decoded_wav": audio.metadata(),
                "interval": interval,
                "measurements": measurements,
            }
        )
    anchor_gates = _anchor_gates(anchor_records)

    rules: dict[str, Any] = {}
    for source in WORD_SOURCES:
        audio = decode_pcm16_mono(source.path)
        active = decoded_active_bounds(audio)
        frames = _probe_frames(source.path)
        interval = align_vowel_core(
            frames,
            word_start_s=float(active["start_s"]),
            word_end_s=float(active["end_s"]),
            search_fraction=(0.10, 0.90),
            sample_rate_hz=audio.sample_rate_hz,
        )
        measurements = {
            f"{maximum_formants:.1f}": measure_formantpath(
                source.path, interval, maximum_formants
            )
            for maximum_formants in MAX_FORMANTS_FAMILY
        }
        family_results: dict[str, Any] = {}
        for key, measurement in measurements.items():
            gate = anchor_gates[source.rule_id]["families"][key]
            point = _point(measurement)
            source_distance = float(
                np.linalg.norm(point - np.array(gate["source_centroid_bark"]))
            )
            target_distance = float(
                np.linalg.norm(point - np.array(gate["target_centroid_bark"]))
            )
            plausible = _plausible(measurement["f1_hz"], measurement["f2_hz"])
            family_results[key] = {
                "source_distance_bark": source_distance,
                "target_distance_bark": target_distance,
                "plausibility_pass": plausible,
                "source_category_pass": source_distance < target_distance,
                "anchor_family_pass": gate["passed"],
                "passed": plausible
                and source_distance < target_distance
                and gate["passed"],
            }
        pre_edit_pass = anchor_gates[source.rule_id]["passed"] and all(
            item["passed"] for item in family_results.values()
        )
        rules[source.rule_id] = {
            "token": source.token,
            "shell": source.shell,
            "take": source.take,
            "decoded_wav": audio.metadata(),
            "active_bounds": active,
            "singleton_edit_interval": interval,
            "measurements": measurements,
            "source_gate": family_results,
            "pre_edit_pass": pre_edit_pass,
            "status": "eligible_for_one_pass_edit"
            if pre_edit_pass
            else "stopped_pre_edit_gate",
        }

    protocol = {
        "protocol": "same-take-word-v1-source-freeze",
        "preregistration_commit": PREREGISTRATION_COMMIT,
        "prior_source_freeze_sha256": SOURCE_FREEZE_SHA256,
        "praat_sha256": PRAAT_SHA256,
        "maximum_formants_family": list(MAX_FORMANTS_FAMILY),
        "active_window_s": ACTIVE_WINDOW_S,
        "active_step_s": ACTIVE_STEP_S,
        "active_relative_db": ACTIVE_RELATIVE_DB,
        "anchor_family_cosine_minimum": ANCHOR_FAMILY_COSINE_MIN,
        "shift_family_cosine_minimum": SHIFT_FAMILY_COSINE_MIN,
        "api_calls": 0,
        "api_cost_usd": 0.0,
    }
    protocol["protocol_sha256"] = hashlib.sha256(
        stable_json(protocol).encode("utf-8")
    ).hexdigest()
    receipt = {
        "schema_version": 1,
        "status": "decoded_sources_singleton_boundaries_and_pre_edit_gates_frozen",
        "run_id": run_id,
        "protocol": protocol,
        "anchor_records": anchor_records,
        "anchor_gates": anchor_gates,
        "rules": rules,
        "no_audio_was_edited": True,
    }
    receipt["receipt_sha256"] = hashlib.sha256(
        stable_json(receipt).encode("utf-8")
    ).hexdigest()
    atomic_write_json(output_path, receipt)
    return receipt


def _raised_cosine_splice(length: int, taper: int) -> np.ndarray:
    weights = np.ones(length, dtype=np.float64)
    taper = min(taper, length // 4)
    if taper <= 0:
        return weights
    phase = np.linspace(0.0, math.pi, taper, endpoint=True)
    edge = 0.5 - 0.5 * np.cos(phase)
    weights[:taper] = edge
    weights[-taper:] = edge[::-1]
    return weights


def _write_pcm16(path: Path, samples: np.ndarray, sample_rate_hz: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate_hz)
        handle.writeframes(samples.astype("<i2", copy=False).tobytes())


def _rms(samples: np.ndarray) -> float:
    values = samples.astype(np.float64, copy=False)
    return math.sqrt(float(np.mean(values * values))) if values.size else 0.0


def _pitch_median(path: Path, interval: dict[str, Any]) -> float | None:
    values = [
        float(frame["pitch_hz"])
        for frame in _probe_frames(path)
        if frame.get("time_s") is not None
        and interval["start_s"] <= float(frame["time_s"]) <= interval["end_s"]
        and frame.get("pitch_hz") is not None
    ]
    return float(np.median(values)) if values else None


def _relative_change(original: float | None, edited: float | None) -> float | None:
    if original is None or edited is None or original == 0:
        return None
    return abs(edited - original) / abs(original)


def _boundary_checks(
    original: np.ndarray,
    edited: np.ndarray,
    start: int,
    end: int,
    sample_rate: int,
) -> dict[str, Any]:
    radius = max(2, round(0.010 * sample_rate))
    results = {}
    for label, boundary in (("start", start), ("end", end)):
        local_start = max(0, boundary - radius)
        local_end = min(original.size, boundary + radius)
        local_diffs = np.abs(np.diff(original[local_start:local_end].astype(np.int32)))
        p95 = float(np.percentile(local_diffs, 95)) if local_diffs.size else 0.0
        threshold = max(2 * p95, 0.01 * 32768)
        index = min(max(1, boundary), edited.size - 1)
        edited_difference = abs(int(edited[index]) - int(edited[index - 1]))
        results[label] = {
            "original_local_p95_first_difference": p95,
            "threshold": threshold,
            "edited_first_difference": edited_difference,
            "passed": edited_difference <= threshold,
        }
    return results


def _engineering_record(
    original_path: Path,
    edited_path: Path,
    interval: dict[str, Any],
    original: DecodedPcm16Wav,
    edited: DecodedPcm16Wav,
) -> dict[str, Any]:
    start = int(interval["start_sample"])
    end = int(interval["end_sample_exclusive"])
    same_container = (
        original.channels == edited.channels
        and original.sample_width_bytes == edited.sample_width_bytes
        and original.sample_rate_hz == edited.sample_rate_hz
        and original.decoded_sample_count == edited.decoded_sample_count
    )
    outside_identical = bool(
        same_container
        and np.array_equal(original.samples[:start], edited.samples[:start])
        and np.array_equal(original.samples[end:], edited.samples[end:])
    )
    original_measurement = measure_formantpath(original_path, interval, 5.0)
    edited_measurement = measure_formantpath(edited_path, interval, 5.0)
    original_f0 = _pitch_median(original_path, interval)
    edited_f0 = _pitch_median(edited_path, interval)
    f0_change = _relative_change(original_f0, edited_f0)
    f3_change = _relative_change(
        original_measurement.get("f3_hz"), edited_measurement.get("f3_hz")
    )
    f4_change = _relative_change(
        original_measurement.get("f4_hz"), edited_measurement.get("f4_hz")
    )
    original_rms = _rms(original.samples[start:end])
    edited_rms = _rms(edited.samples[start:end])
    rms_db_change = (
        abs(20 * math.log10(edited_rms / original_rms))
        if original_rms > 0 and edited_rms > 0
        else math.inf
    )
    boundaries = _boundary_checks(
        original.samples, edited.samples, start, end, original.sample_rate_hz
    )
    passed = bool(
        same_container
        and outside_identical
        and edited.clipped_sample_count == 0
        and f0_change is not None
        and f0_change <= 0.02
        and f3_change is not None
        and f3_change <= 0.05
        and f4_change is not None
        and f4_change <= 0.05
        and rms_db_change <= 1.0
        and all(item["passed"] for item in boundaries.values())
    )
    return {
        "same_container": same_container,
        "outside_interval_bit_identical": outside_identical,
        "clipped_sample_count": edited.clipped_sample_count,
        "original_f0_hz": original_f0,
        "edited_f0_hz": edited_f0,
        "f0_relative_change": f0_change,
        "f3_relative_change": f3_change,
        "f4_relative_change": f4_change,
        "rms_db_change": rms_db_change,
        "boundary_checks": boundaries,
        "passed": passed,
    }


def _classify_strength(
    *,
    identity_measurements: dict[str, Any],
    shifted_measurements: dict[str, Any],
    anchor_gate: dict[str, Any],
) -> dict[str, Any]:
    families: dict[str, Any] = {}
    vectors = []
    for key, identity in identity_measurements.items():
        shifted = shifted_measurements[key]
        gate = anchor_gate["families"][key]
        identity_point = _point(identity)
        shifted_point = _point(shifted)
        vector = shifted_point - identity_point
        vectors.append(vector)
        magnitude = float(np.linalg.norm(vector))
        threshold = max(0.15, 1.5 * float(gate["endpoint_take_variance_bark"]))
        cosine = _cosine(vector, np.array(gate["anchor_vector_bark"]))
        identity_source = float(
            np.linalg.norm(identity_point - np.array(gate["source_centroid_bark"]))
        )
        identity_target = float(
            np.linalg.norm(identity_point - np.array(gate["target_centroid_bark"]))
        )
        shifted_source = float(
            np.linalg.norm(shifted_point - np.array(gate["source_centroid_bark"]))
        )
        shifted_target = float(
            np.linalg.norm(shifted_point - np.array(gate["target_centroid_bark"]))
        )
        plausible = _plausible(identity["f1_hz"], identity["f2_hz"]) and _plausible(
            shifted["f1_hz"], shifted["f2_hz"]
        )
        directional = bool(
            gate["passed"]
            and plausible
            and magnitude > threshold
            and cosine >= ANCHOR_DIRECTION_COSINE_MIN
        )
        proximity = identity_source < identity_target and shifted_target < shifted_source
        families[key] = {
            "vector_bark": vector.tolist(),
            "magnitude_bark": magnitude,
            "magnitude_threshold_bark": threshold,
            "anchor_direction_cosine": cosine,
            "identity_source_distance_bark": identity_source,
            "identity_target_distance_bark": identity_target,
            "shifted_source_distance_bark": shifted_source,
            "shifted_target_distance_bark": shifted_target,
            "plausibility_pass": plausible,
            "directional_pass": directional,
            "endpoint_proximity_pass": proximity,
            "classification": "exact-category"
            if directional and proximity
            else "directional-only"
            if directional
            else "fail",
        }
    pairwise = [_cosine(vectors[left], vectors[right]) for left, right in combinations(range(len(vectors)), 2)]
    cross_family = bool(
        pairwise and min(pairwise) >= SHIFT_FAMILY_COSINE_MIN
    )
    if cross_family and all(item["classification"] == "exact-category" for item in families.values()):
        classification = "exact-category"
    elif cross_family and all(item["classification"] in {"exact-category", "directional-only"} for item in families.values()):
        classification = "directional-only"
    else:
        classification = "fail"
    return {
        "families": families,
        "pairwise_vector_cosines": pairwise,
        "cross_family_vector_consistency_pass": cross_family,
        "classification": classification,
    }


def run_word_editor(run_id: str = RUN_ID) -> dict[str, Any]:
    if run_id != RUN_ID:
        raise RuntimeError("same-take-word-v1 run ID is fixed")
    output_dir = Paths().artifacts / "same-take" / run_id
    freeze_path = output_dir / "source-freeze.json"
    result_path = output_dir / "praat-pass.json"
    if result_path.exists():
        raise RuntimeError("same-take-word-v1 Praat pass is already final")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze.get("no_audio_was_edited") is not True:
        raise RuntimeError("source freeze does not precede editing")
    if not EDITOR_SCRIPT.is_file():
        raise RuntimeError("project-authored FormantGrid editor is missing")

    outcomes: dict[str, Any] = {}
    for source in WORD_SOURCES:
        frozen_rule = freeze["rules"][source.rule_id]
        if not frozen_rule["pre_edit_pass"]:
            outcomes[source.rule_id] = {
                "status": "stopped_pre_edit_gate",
                "outputs": 0,
            }
            continue
        original = decode_pcm16_mono(source.path)
        interval = frozen_rule["singleton_edit_interval"]
        key = "5.0"
        source_point = _point(frozen_rule["measurements"][key])
        target_point = np.array(
            freeze["anchor_gates"][source.rule_id]["families"][key][
                "target_centroid_bark"
            ]
        )
        delta = target_point - source_point
        raw: dict[float, DecodedPcm16Wav] = {}
        with tempfile.TemporaryDirectory(prefix="same-take-word-praat-") as temp:
            for alpha in (0.0, *STRENGTHS):
                raw_path = Path(temp) / f"raw-{alpha:.2f}.wav"
                subprocess.run(
                    [
                        str(PRAAT),
                        "--run",
                        str(EDITOR_SCRIPT),
                        str(source.path),
                        str(raw_path),
                        f"{interval['start_s']:.9f}",
                        f"{interval['end_s']:.9f}",
                        str(original.sample_rate_hz),
                        f"{alpha:.2f}",
                        f"{delta[0]:.12f}",
                        f"{delta[1]:.12f}",
                        f"{SHIFT_TAPER_S:.3f}",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                raw[alpha] = decode_pcm16_mono(raw_path)
            start = int(interval["start_sample"])
            end = int(interval["end_sample_exclusive"])
            if any(item.decoded_sample_count != original.decoded_sample_count for item in raw.values()):
                raise RuntimeError("Praat round-trip changed decoded sample count")
            original_rms = _rms(original.samples[start:end])
            identity_rms = _rms(raw[0.0].samples[start:end])
            if identity_rms <= 0:
                raise RuntimeError("Praat identity resynthesis has zero interval RMS")
            gain = original_rms / identity_rms
            taper = _raised_cosine_splice(
                end - start, round(SPLICE_TAPER_S * original.sample_rate_hz)
            )
            output_records: dict[str, Any] = {}
            identity_measurements: dict[str, Any] | None = None
            strength_classifications: dict[str, Any] = {}
            for alpha in (0.0, *STRENGTHS):
                processed = raw[alpha].samples[start:end].astype(np.float64) * gain
                original_core = original.samples[start:end].astype(np.float64)
                blended = original_core + (processed - original_core) * taper
                unclipped = bool(np.max(blended) <= 32767 and np.min(blended) >= -32768)
                final = original.samples.copy()
                final[start:end] = np.rint(np.clip(blended, -32768, 32767)).astype(np.int16)
                label = "identity" if alpha == 0 else f"shift-{int(alpha * 100):03d}"
                path = output_dir / "audio" / f"ih-to-i__{label}.wav"
                _write_pcm16(path, final, original.sample_rate_hz)
                decoded = decode_pcm16_mono(path)
                measurements = {
                    f"{maximum_formants:.1f}": measure_formantpath(
                        path, interval, maximum_formants
                    )
                    for maximum_formants in MAX_FORMANTS_FAMILY
                }
                engineering = _engineering_record(
                    source.path, path, interval, original, decoded
                )
                engineering["unclipped_before_encoding"] = unclipped
                engineering["passed"] = engineering["passed"] and unclipped
                output_records[label] = {
                    "alpha": alpha,
                    "decoded_wav": decoded.metadata(),
                    "measurements": measurements,
                    "engineering": engineering,
                }
                if alpha == 0:
                    identity_measurements = measurements
                else:
                    assert identity_measurements is not None
                    strength_classifications[label] = _classify_strength(
                        identity_measurements=identity_measurements,
                        shifted_measurements=measurements,
                        anchor_gate=freeze["anchor_gates"][source.rule_id],
                    )

        exact = [
            (alpha, f"shift-{int(alpha * 100):03d}")
            for alpha in STRENGTHS
            if strength_classifications[f"shift-{int(alpha * 100):03d}"]["classification"] == "exact-category"
            and output_records[f"shift-{int(alpha * 100):03d}"]["engineering"]["passed"]
        ]
        directional = [
            (alpha, f"shift-{int(alpha * 100):03d}")
            for alpha in STRENGTHS
            if strength_classifications[f"shift-{int(alpha * 100):03d}"]["classification"] in {"exact-category", "directional-only"}
            and output_records[f"shift-{int(alpha * 100):03d}"]["engineering"]["passed"]
        ]
        identity_pass = output_records["identity"]["engineering"]["passed"]
        selected = min(exact)[1] if exact else max(directional)[1] if directional else None
        outcomes[source.rule_id] = {
            "status": "eligible_for_headphone_qc"
            if identity_pass and selected
            else "praat_pass_failed",
            "canonical_editor_delta_bark": delta.tolist(),
            "gain": gain,
            "identity_engineering_pass": identity_pass,
            "selected_shift": selected if identity_pass else None,
            "outputs": output_records,
            "strength_classifications": strength_classifications,
        }

    result = {
        "schema_version": 1,
        "status": "praat_one_pass_complete",
        "run_id": run_id,
        "source_freeze_file_sha256": sha256_file(freeze_path),
        "editor_script_sha256": sha256_file(EDITOR_SCRIPT),
        "api_calls": 0,
        "api_cost_usd": 0.0,
        "rules": outcomes,
    }
    result["receipt_sha256"] = hashlib.sha256(
        stable_json(result).encode("utf-8")
    ).hexdigest()
    atomic_write_json(result_path, result)
    return result
