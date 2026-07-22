#!/usr/bin/env python3
from __future__ import annotations

from collections import Counter, defaultdict
import argparse
import hashlib
import json
import time
from typing import Any, Sequence

import numpy as np

from earshift_bakeoff.config import Paths, sha256_json, stable_json
from earshift_bakeoff.kokoro_output_domain_splice import boundary_artifact_report
from earshift_bakeoff.kokoro_synthesis import SAMPLE_RATE_HZ
from earshift_bakeoff.kokoro_typed_diagnostic import localization_report
from earshift_bakeoff.spectral_envelope_warp import (
    FormantWarpSpec,
    SpectralEnvelopeWarpResult,
    spectral_envelope_warp,
)
from earshift_bakeoff.util import atomic_write_json, sha256_file

import run_bilingual_product_v8_vowel_acoustic_screen as v8_screen
import run_bilingual_vowel_adaptive_strength_screen_v1 as adaptive


VERSION = "english-central-vowel-spectral-correction-v1"
RUN_ID = "20260718-english-central-vowel-spectral-correction-v1"
PROTOCOL_PATH = Paths().root / "rules" / f"{VERSION}.json"
RUN_DIR = Paths().artifacts / "product-matrix" / RUN_ID
RESULT_PATH = RUN_DIR / "results.json"
V8_MANIFEST_PATH = v8_screen.V8_MANIFEST_PATH
V8_RESULT_PATH = adaptive.V8_RESULT_PATH
V8_RESULT_DIR = V8_RESULT_PATH.parent
RULE_IDS = ("enpt.reduced_schwa_a", "enpt.schwa_reduced_a")
VOICE_ORDER = ("af_heart", "am_michael")
STRENGTHS = (1.0, 0.75, 1.25, 0.5, 1.5, 2.0)
HIGH_BAND_START_HZ = 4_500.0
MAXIMUM_HIGH_BAND_DELTA_DB = 1.5
MAXIMUM_RMS_DELTA_DB = 1.0


def _semantic_hash(payload: Any) -> str:
    return hashlib.sha256(stable_json(payload).encode("utf-8")).hexdigest()


def _slots(manifest: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    return tuple(
        slot
        for slot in manifest["slots"]
        if slot["rule_id"] in RULE_IDS and slot["voice_id"] in VOICE_ORDER
    )


def _source_bindings() -> dict[str, str]:
    paths = (
        "scripts/run_english_central_vowel_spectral_correction_v1.py",
        "src/earshift_bakeoff/spectral_envelope_warp.py",
        "scripts/run_bilingual_product_v8_vowel_acoustic_screen.py",
        "scripts/run_bilingual_vowel_adaptive_strength_screen_v1.py",
    )
    return {path: sha256_file(Paths().root / path) for path in paths}


def protocol_record() -> dict[str, Any]:
    manifest = json.loads(V8_MANIFEST_PATH.read_text(encoding="utf-8"))
    result = json.loads(V8_RESULT_PATH.read_text(encoding="utf-8"))
    slots = _slots(manifest)
    payload = {
        "schema_version": 1,
        "version": VERSION,
        "run_id": RUN_ID,
        "status": "frozen_before_signal_processing",
        "purpose": (
            "Test a localized smooth-spectral-envelope warp after the bounded "
            "latent-strength mechanism failed to generalize for English central vowels."
        ),
        "parents": {
            "v8_manifest_sha256": sha256_file(V8_MANIFEST_PATH),
            "v8_manifest_record_sha256": manifest["record_sha256"],
            "v8_result_sha256": sha256_file(V8_RESULT_PATH),
            "v8_result_record_sha256": result["record_sha256"],
            "central_latent_result_sha256": sha256_file(
                Paths().artifacts
                / "product-matrix"
                / "20260718-bilingual-central-vowel-correction-v1"
                / "results.json"
            ),
        },
        "scope": {
            "rule_ids": list(RULE_IDS),
            "voice_order": list(VOICE_ORDER),
            "cell_ids": sorted({slot["cell_id"] for slot in slots}),
            "cell_count": 4,
            "logical_slot_ids": [slot["logical_slot_id"] for slot in slots],
            "logical_slot_count": 12,
            "candidate_count": 12 * len(STRENGTHS),
        },
        "intervention": {
            "version": "spectral-envelope-warp-v1",
            "strength_order": list(STRENGTHS),
            "formant_endpoints": (
                "Median source and target F1/F2 in Hz across the three frozen v8 "
                "analysis ceilings, separately for every occurrence."
            ),
            "processing": (
                "Warp the smoothed log-magnitude envelope while retaining harmonic "
                "bin positions and phase; preserve the transfer function above 4500 "
                "Hz and taper to bit-exact neutral outside each edit window."
            ),
            "selection": (
                "Choose the first exact-category strength per occurrence in frozen "
                "order, otherwise first directional-only; no listening selection."
            ),
        },
        "engineering_gates": {
            "outside_windows_bit_exact": True,
            "finite": True,
            "clipped_sample_count_maximum": 0,
            "absolute_rms_delta_db_maximum": MAXIMUM_RMS_DELTA_DB,
            "absolute_high_band_delta_db_maximum": MAXIMUM_HIGH_BAND_DELTA_DB,
            "boundary_metrics": "existing frozen output-domain thresholds",
            "localization": "existing frozen runtime localization gate",
        },
        "acoustic_gates": (
            "Reuse every frozen v8 source/target anchor, measurement interval, "
            "three-ceiling analysis family, threshold, and aggregate rule unchanged."
        ),
        "stopping_rule": (
            "Process all 72 candidates once without replacement. A known-fixture "
            "pass permits unseen confirmation only; no production promotion."
        ),
        "scope_controls": {
            "kokoro_renders": 0,
            "api_calls": 0,
            "paid_calls": 0,
            "production_enabled": False,
            "deployment": False,
        },
        "source_bindings": _source_bindings(),
    }
    return {**payload, "protocol_sha256": sha256_json(payload)}


def prepare() -> dict[str, Any]:
    protocol = protocol_record()
    if PROTOCOL_PATH.exists():
        existing = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        if stable_json(existing) != stable_json(protocol):
            raise RuntimeError("spectral-correction protocol differs from frozen record")
        return existing
    if RUN_DIR.exists():
        raise RuntimeError("spectral-correction run exists before its protocol")
    atomic_write_json(PROTOCOL_PATH, protocol)
    return protocol


def _median_formants(
    baseline: dict[str, Any], occurrence_index: int, label: str
) -> tuple[float, float]:
    values = []
    for analysis in baseline["analysis_by_formant_ceiling"]:
        measurement = analysis["measurements"][label][occurrence_index]
        bins = measurement["bins"]
        if not measurement["measurable"] or len(bins) != 1:
            raise RuntimeError("frozen central-vowel endpoint is not measurable")
        values.append((float(bins[0]["f1_hz"]), float(bins[0]["f2_hz"])))
    result = tuple(float(value) for value in np.median(values, axis=0))
    if len(result) != 2 or not 60.0 < result[0] < result[1] < 4_500.0:
        raise RuntimeError("frozen central-vowel formant endpoint is implausible")
    return result


def _warp_specs(baseline: dict[str, Any]) -> tuple[FormantWarpSpec, ...]:
    specs = []
    for index, occurrence in enumerate(baseline["occurrence_outcomes"]):
        interval = occurrence["measurement_interval"]
        source = _median_formants(baseline, index, "source_anchor")
        target = _median_formants(baseline, index, "target_anchor")
        specs.append(
            FormantWarpSpec(
                start_sample=int(interval["start_sample"]),
                end_sample_exclusive=int(interval["end_sample_exclusive"]),
                source_f1_hz=source[0],
                source_f2_hz=source[1],
                target_f1_hz=target[0],
                target_f2_hz=target[1],
            )
        )
    return tuple(specs)


def _band_energy_db(values: np.ndarray, minimum_hz: float) -> float:
    signal = np.asarray(values, dtype=np.float64)
    if signal.size < 16:
        return -120.0
    windowed = signal * np.hanning(signal.size)
    spectrum = np.fft.rfft(windowed)
    frequencies = np.fft.rfftfreq(signal.size, d=1.0 / SAMPLE_RATE_HZ)
    power = np.square(np.abs(spectrum[frequencies >= minimum_hz]))
    return 10.0 * np.log10(max(float(np.mean(power)), 1e-12))


def _engineering(
    neutral: np.ndarray,
    warped: SpectralEnvelopeWarpResult,
    intervals: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    high_band = []
    for window in warped.edit_windows:
        start = window["start_sample"]
        end = window["end_sample_exclusive"]
        high_band.append(
            _band_energy_db(warped.pcm[start:end], HIGH_BAND_START_HZ)
            - _band_energy_db(neutral[start:end], HIGH_BAND_START_HZ)
        )
    boundary = boundary_artifact_report(
        neutral,
        warped.pcm,
        warped.pcm,
        list(warped.edit_windows),
    )
    localization = localization_report(neutral, warped.pcm, list(intervals))
    metrics = {
        **warped.metrics,
        "high_band_delta_db_by_window": high_band,
        "boundary_metrics_pass": bool(boundary.get("pass")),
        "localization_pass": bool(localization.get("pass")),
        "localization_fraction": float(
            localization.get("inside_difference_energy_fraction", 0.0)
        ),
    }
    metrics["pass"] = bool(
        metrics["outside_windows_bit_exact"]
        and metrics["finite"]
        and metrics["clipped_sample_count"] == 0
        and max(abs(value) for value in metrics["rms_db_change_by_window"])
        <= MAXIMUM_RMS_DELTA_DB
        and max(abs(value) for value in high_band)
        <= MAXIMUM_HIGH_BAND_DELTA_DB
        and metrics["boundary_metrics_pass"]
        and metrics["localization_pass"]
    )
    return metrics


def _candidate(
    *,
    slot: dict[str, Any],
    baseline: dict[str, Any],
    neutral: np.ndarray,
    specs: tuple[FormantWarpSpec, ...],
    strength: float,
) -> tuple[dict[str, Any], SpectralEnvelopeWarpResult]:
    warped = spectral_envelope_warp(
        neutral,
        specs,
        sample_rate_hz=SAMPLE_RATE_HZ,
        strength=strength,
    )
    stem = adaptive._safe_name(slot["logical_slot_id"])
    label = adaptive._label(strength)
    path = RUN_DIR / "audio" / f"{stem}__spectral-{label}.wav"
    audio = adaptive._write_wav(path, warped.pcm)
    intervals = tuple(
        row["measurement_interval"] for row in baseline["occurrence_outcomes"]
    )
    engineering = _engineering(neutral, warped, intervals)
    analysis = []
    for baseline_ceiling in baseline["analysis_by_formant_ceiling"]:
        ceiling = int(baseline_ceiling["maximum_formant_hz"])
        measurements = adaptive._measure(
            path=path,
            stem=f"{stem}__spectral-{label}",
            intervals=intervals,
            ceiling=ceiling,
            mode=baseline["measurement_mode"],
        )
        frozen = baseline_ceiling["measurements"]
        classifications = tuple(
            adaptive._analysis_classification(
                source=frozen["source_anchor"][index],
                target=frozen["target_anchor"][index],
                neutral=frozen["neutral"][index],
                lens=measurements[index],
                rhotic=False,
            )
            for index in range(len(intervals))
        )
        analysis.append(
            {
                "maximum_formant_hz": ceiling,
                "lens": measurements,
                "occurrence_classifications": classifications,
            }
        )
    occurrences = []
    for index in range(len(intervals)):
        classification = adaptive._aggregate(
            tuple(row["occurrence_classifications"][index] for row in analysis)
        )
        occurrences.append(
            {
                "occurrence_index": index,
                "classification": classification,
                "exact_category_pass": classification == "exact_category_pass",
                "directional_pass": classification
                in {"exact_category_pass", "directional_only_pass"},
            }
        )
    classification = adaptive._aggregate(occurrences) if engineering["pass"] else "fail"
    return (
        {
            "strength": strength,
            "label": label,
            "classification": classification,
            "engineering": engineering,
            "audio": audio,
            "occurrence_outcomes": occurrences,
            "analysis_by_formant_ceiling": analysis,
        },
        warped,
    )


def _selection(candidates: Sequence[dict[str, Any]], index: int) -> dict[str, Any] | None:
    for desired in ("exact_category_pass", "directional_only_pass"):
        for candidate in candidates:
            occurrence = candidate["occurrence_outcomes"][index]
            if candidate["engineering"]["pass"] and occurrence["classification"] == desired:
                return {
                    "strength": candidate["strength"],
                    "label": candidate["label"],
                    "classification": desired,
                }
    return None


def _slot_outcome(slot: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    neutral_path = V8_RESULT_DIR / baseline["audio"]["neutral"]["relative_path"]
    neutral = adaptive._read_wav(neutral_path)
    specs = _warp_specs(baseline)
    candidates = []
    renders: dict[float, SpectralEnvelopeWarpResult] = {}
    for strength in STRENGTHS:
        candidate, rendered = _candidate(
            slot=slot,
            baseline=baseline,
            neutral=neutral,
            specs=specs,
            strength=strength,
        )
        candidates.append(candidate)
        renders[strength] = rendered
    selections = tuple(
        _selection(candidates, index) for index in range(len(specs))
    )
    unresolved = tuple(index for index, selection in enumerate(selections) if not selection)
    classification = "fail"
    occurrence_outcomes: list[dict[str, Any]] = []
    composite_audio = None
    composite_engineering = None
    if not unresolved:
        composite = neutral.copy()
        windows = []
        for selection, spec in zip(selections, specs, strict=True):
            assert selection is not None
            rendered = renders[selection["strength"]]
            window = rendered.edit_windows[len(windows)]
            start = window["start_sample"]
            end = window["end_sample_exclusive"]
            composite[start:end] = rendered.pcm[start:end]
            windows.append(window)
        weights = np.zeros(neutral.size, dtype=np.float64)
        selected_renders = []
        for index, (window, selection) in enumerate(
            zip(windows, selections, strict=True)
        ):
            assert selection is not None
            rendered = renders[selection["strength"]]
            start = window["start_sample"]
            end = window["end_sample_exclusive"]
            weights[start:end] = rendered.weights[start:end]
            selected_renders.append((index, rendered))
        composite_result = SpectralEnvelopeWarpResult(
            pcm=composite,
            weights=weights,
            edit_windows=tuple(windows),
            metrics={
                "identity": False,
                "outside_windows_bit_exact": bool(
                    np.array_equal(composite[weights == 0.0], neutral[weights == 0.0])
                ),
                "finite": bool(np.isfinite(composite.astype(np.float64)).all()),
                "clipped_sample_count": max(
                    rendered.metrics["clipped_sample_count"]
                    for _, rendered in selected_renders
                ),
                "rms_db_change_by_window": [
                    rendered.metrics["rms_db_change_by_window"][index]
                    for index, rendered in selected_renders
                ],
            },
        )
        intervals = tuple(
            row["measurement_interval"] for row in baseline["occurrence_outcomes"]
        )
        composite_engineering = _engineering(neutral, composite_result, intervals)
        stem = adaptive._safe_name(slot["logical_slot_id"])
        path = RUN_DIR / "audio" / f"{stem}__spectral-composite.wav"
        composite_audio = adaptive._write_wav(path, composite)
        analyses = []
        for baseline_ceiling in baseline["analysis_by_formant_ceiling"]:
            ceiling = int(baseline_ceiling["maximum_formant_hz"])
            measurements = adaptive._measure(
                path=path,
                stem=f"{stem}__spectral-composite",
                intervals=intervals,
                ceiling=ceiling,
                mode=baseline["measurement_mode"],
            )
            frozen = baseline_ceiling["measurements"]
            analyses.append(
                tuple(
                    adaptive._analysis_classification(
                        source=frozen["source_anchor"][index],
                        target=frozen["target_anchor"][index],
                        neutral=frozen["neutral"][index],
                        lens=measurements[index],
                        rhotic=False,
                    )
                    for index in range(len(intervals))
                )
            )
        for index, selection in enumerate(selections):
            result = adaptive._aggregate(tuple(row[index] for row in analyses))
            occurrence_outcomes.append(
                {
                    "occurrence_index": index,
                    "selection": selection,
                    "classification": result,
                    "exact_category_pass": result == "exact_category_pass",
                    "directional_pass": result
                    in {"exact_category_pass", "directional_only_pass"},
                }
            )
        if composite_engineering["pass"]:
            classification = adaptive._aggregate(occurrence_outcomes)
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
        "classification": classification,
        "selection_complete": not unresolved,
        "unresolved_occurrence_indexes": list(unresolved),
        "occurrence_selections": selections,
        "occurrence_outcomes": occurrence_outcomes,
        "composite_audio": composite_audio,
        "composite_engineering": composite_engineering,
        "candidates": candidates,
        "api_calls_made": 0,
        "production_enabled": False,
    }


def _summaries(outcomes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for outcome in outcomes:
        groups[outcome["cell_id"]].append(outcome)
    rows = []
    for cell_id, slots in sorted(groups.items()):
        complete = bool(
            len(slots) == 3
            and all(slot["selection_complete"] for slot in slots)
            and sum(len(slot["occurrence_outcomes"]) for slot in slots) == 4
        )
        classification = adaptive._aggregate(slots) if complete else "fail"
        rows.append(
            {
                "cell_id": cell_id,
                "voice_id": slots[0]["voice_id"],
                "rule_id": slots[0]["rule_id"],
                "classification": classification,
                "complete": complete,
                "selected_strength_counts": dict(
                    Counter(
                        selection["label"]
                        for slot in slots
                        for selection in slot["occurrence_selections"]
                        if selection is not None
                    )
                ),
                "production_enabled": False,
            }
        )
    return rows


def run() -> dict[str, Any]:
    if RESULT_PATH.exists():
        return json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if stable_json(protocol) != stable_json(protocol_record()):
        raise RuntimeError("spectral-correction protocol or sources drifted")
    manifest = json.loads(V8_MANIFEST_PATH.read_text(encoding="utf-8"))
    v8_result = json.loads(V8_RESULT_PATH.read_text(encoding="utf-8"))
    slots = _slots(manifest)
    baseline_by_id = {row["logical_slot_id"]: row for row in v8_result["outcomes"]}
    if len(slots) != 12:
        raise RuntimeError("spectral-correction denominator drifted")
    RUN_DIR.mkdir(parents=True, exist_ok=False)
    original_run_dir = adaptive.RUN_DIR
    adaptive.RUN_DIR = RUN_DIR
    started = time.perf_counter()
    try:
        outcomes = [
            _slot_outcome(slot, baseline_by_id[slot["logical_slot_id"]])
            for slot in slots
        ]
    finally:
        adaptive.RUN_DIR = original_run_dir
    cells = _summaries(outcomes)
    counts = Counter(row["classification"] for row in cells)
    classification = (
        "english_central_spectral_mechanism_pass_pending_unseen_confirmation"
        if cells and all(row["classification"] == "exact_category_pass" for row in cells)
        else "english_central_spectral_mechanism_mixed_no_promotion"
    )
    payload = {
        "schema_version": 1,
        "version": VERSION,
        "run_id": RUN_ID,
        "protocol_sha256": protocol["protocol_sha256"],
        "classification": classification,
        "cell_classification_counts": dict(sorted(counts.items())),
        "cell_count": len(cells),
        "logical_slot_count": len(outcomes),
        "candidate_count": len(outcomes) * len(STRENGTHS),
        "kokoro_renders": 0,
        "api_calls_made": 0,
        "production_enabled": False,
        "elapsed_s": time.perf_counter() - started,
        "cell_summaries": cells,
        "outcomes": outcomes,
    }
    result = {**payload, "record_sha256": _semantic_hash(payload)}
    atomic_write_json(RESULT_PATH, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "run"))
    args = parser.parse_args()
    result = prepare() if args.command == "prepare" else run()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
