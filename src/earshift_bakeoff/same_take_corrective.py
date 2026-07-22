from __future__ import annotations

import hashlib
import json
import math
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import DEVLOG_PATH, Paths, stable_json
from .pcm import DecodedPcm16Wav, decode_pcm16_mono
from .same_take import MAX_FORMANTS_FAMILY, PRAAT, _cosine, measure_formantpath
from .same_take_followup import (
    BAND_HIGH_HZ,
    BAND_LOW_HZ,
    band_power_db,
    signed_rms_db,
)
from .same_take_word import (
    ANCHOR_DIRECTION_COSINE_MIN,
    EDITOR_SCRIPT,
    SHIFT_FAMILY_COSINE_MIN,
    SHIFT_TAPER_S,
    SPLICE_TAPER_S,
    _boundary_checks,
    _classify_strength,
    _pitch_median,
    _plausible,
    _raised_cosine_splice,
    _relative_change,
    _rms,
    _write_pcm16,
)
from .util import atomic_write_json, sha256_file


RUN_ID = "20260715-same-take-word-v2"
PREREGISTRATION_HEADING = (
    "## Same-take-word-v2 corrective grid preregistration — July 15, 2026"
)
PREREGISTRATION_COMMIT = "7419da1"
PRIOR_ROOT = Paths().artifacts / "same-take" / "20260715-same-take-word-v1"
FREEZE_PATH = PRIOR_ROOT / "source-freeze.json"
PRIOR_PASS_PATH = PRIOR_ROOT / "praat-pass.json"
DIAGNOSIS_PATH = Paths().artifacts / "same-take" / RUN_ID / "processing-path-audit.json"
FROZEN_HASHES = {
    "source_freeze": "d27e2e1142bbb0cce7ef0950bb68f7e38f3a5f5063b8a26213bda3b2d3fbb282",
    "prior_pass": "5e1692daeb81d7040ffe3cd072b2ca1e2a9f77a215664736540459f2458fa20e",
    "diagnosis": "6284b1b7f81dce4cf2d5af78792b8ecc05d12d7624bf6e5122e0a5090666d2de",
}
RULE_ID = "ptbr.vowel.ih_to_i"
SOURCE_SHA256 = "29fc9446871907620c66e459665ed764720757d4273fc7c507c21fe07809e9c5"
STRENGTHS = (0.0, 0.50, 0.75, 1.00)
CANDIDATE_ORDER = (
    "baseline",
    "per-condition-loudness",
    "high-frequency-restoration",
    "combined",
)
FILTER_TAPS = 255
FILTER_CUTOFF_HZ = 5_500.0
FILTER_PASSBAND_END_HZ = 5_000.0
FILTER_STOPBAND_START_HZ = 6_000.0
FILTER_PASSBAND_LOSS_DB_MAX = 0.5
FILTER_STOPBAND_REJECTION_DB_MIN = 40.0
SHARED_GAIN = 1.0692645440887207
PEAK_HEADROOM_DB_MIN = 1.0
RMS_DELTA_DB_MAX = 1.0
HIGH_BAND_DELTA_DB_MAX = 3.0
IDENTITY_LOW_BAND_LSD_DB_MAX = 6.0
BOUNDARY_HIGH_BAND_DELTA_DB_MAX = 6.0
LOW_BAND_HZ = (200.0, 5_000.0)
LOW_BAND_REFERENCE_FLOOR_DB = -60.0


@dataclass(frozen=True)
class CandidateConfig:
    per_condition_gain: bool
    restore_high_band: bool


CANDIDATES = {
    "baseline": CandidateConfig(False, False),
    "per-condition-loudness": CandidateConfig(True, False),
    "high-frequency-restoration": CandidateConfig(False, True),
    "combined": CandidateConfig(True, True),
}


def design_complementary_lowpass(sample_rate_hz: int) -> tuple[np.ndarray, dict[str, Any]]:
    if sample_rate_hz != 24_000:
        raise RuntimeError("corrective grid is frozen to 24 kHz")
    center = (FILTER_TAPS - 1) / 2
    positions = np.arange(FILTER_TAPS, dtype=np.float64) - center
    normalized = FILTER_CUTOFF_HZ / sample_rate_hz
    coefficients = 2 * normalized * np.sinc(2 * normalized * positions)
    coefficients *= np.blackman(FILTER_TAPS)
    coefficients /= float(np.sum(coefficients))

    fft_size = 262_144
    response = np.fft.rfft(coefficients, n=fft_size)
    frequencies = np.fft.rfftfreq(fft_size, d=1.0 / sample_rate_hz)
    magnitudes_db = 20 * np.log10(np.maximum(np.abs(response), 1e-15))
    passband = magnitudes_db[frequencies <= FILTER_PASSBAND_END_HZ]
    stopband = magnitudes_db[frequencies >= FILTER_STOPBAND_START_HZ]
    passband_loss = float(max(0.0, -np.min(passband)))
    stopband_rejection = float(-np.max(stopband))
    coefficients_sha = hashlib.sha256(
        coefficients.astype("<f8", copy=False).tobytes()
    ).hexdigest()
    record = {
        "sample_rate_hz": sample_rate_hz,
        "taps": FILTER_TAPS,
        "window": "numpy.blackman",
        "cutoff_hz": FILTER_CUTOFF_HZ,
        "declared_transition_hz": [
            FILTER_PASSBAND_END_HZ,
            FILTER_STOPBAND_START_HZ,
        ],
        "passband_loss_db": passband_loss,
        "passband_loss_db_max": FILTER_PASSBAND_LOSS_DB_MAX,
        "stopband_rejection_db": stopband_rejection,
        "stopband_rejection_db_min": FILTER_STOPBAND_REJECTION_DB_MIN,
        "coefficients_sha256": coefficients_sha,
        "passed": passband_loss <= FILTER_PASSBAND_LOSS_DB_MAX
        and stopband_rejection >= FILTER_STOPBAND_REJECTION_DB_MIN,
    }
    return coefficients, record


def centered_convolution(samples: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    return np.convolve(samples.astype(np.float64), coefficients, mode="same")


def solve_rms_gain(
    original: np.ndarray,
    component: np.ndarray,
    residual: np.ndarray,
    taper: np.ndarray,
) -> float | None:
    o = original.astype(np.float64, copy=False)
    p = component.astype(np.float64, copy=False)
    h = residual.astype(np.float64, copy=False)
    w = taper.astype(np.float64, copy=False)
    a = w * p
    b = o + w * (h - o)
    target = float(np.mean(o * o))
    qa = float(np.mean(a * a))
    qb = 2.0 * float(np.mean(a * b))
    qc = float(np.mean(b * b)) - target
    if qa <= 0:
        return None
    discriminant = qb * qb - 4 * qa * qc
    if discriminant < 0:
        return None
    root = math.sqrt(max(0.0, discriminant))
    candidates = [(-qb + root) / (2 * qa), (-qb - root) / (2 * qa)]
    eligible = [item for item in candidates if math.isfinite(item) and item >= 0]
    return min(eligible, key=lambda item: abs(item - SHARED_GAIN)) if eligible else None


def low_band_log_spectral_distance_db(
    reference: np.ndarray,
    candidate: np.ndarray,
    sample_rate_hz: int,
) -> float:
    ref = reference.astype(np.float64, copy=False)
    test = candidate.astype(np.float64, copy=False)
    if ref.size != test.size or ref.size < 2:
        return math.inf
    window = np.hanning(ref.size)
    ref_magnitude = np.abs(np.fft.rfft((ref - np.mean(ref)) * window))
    test_magnitude = np.abs(np.fft.rfft((test - np.mean(test)) * window))
    frequencies = np.fft.rfftfreq(ref.size, d=1.0 / sample_rate_hz)
    ref_db = 20 * np.log10(np.maximum(ref_magnitude, 1e-12))
    test_db = 20 * np.log10(np.maximum(test_magnitude, 1e-12))
    selected = (
        (frequencies >= LOW_BAND_HZ[0])
        & (frequencies <= LOW_BAND_HZ[1])
        & (ref_db >= float(np.max(ref_db)) + LOW_BAND_REFERENCE_FLOOR_DB)
    )
    return (
        math.sqrt(float(np.mean((test_db[selected] - ref_db[selected]) ** 2)))
        if np.any(selected)
        else math.inf
    )


def _peak_dbfs(samples: np.ndarray) -> float:
    peak = int(np.max(np.abs(samples.astype(np.int32)))) if samples.size else 0
    return 20 * math.log10(peak / 32768.0) if peak else -math.inf


def _high_band_delta(reference: np.ndarray, candidate: np.ndarray, rate: int) -> float:
    return band_power_db(candidate, rate) - band_power_db(reference, rate)


def _boundary_spectral_records(
    original: np.ndarray,
    candidate: np.ndarray,
    start: int,
    end: int,
    rate: int,
) -> dict[str, Any]:
    count = round(0.010 * rate)
    records = {}
    for label, left, right in (
        ("start", start, start + count),
        ("end", end - count, end),
    ):
        delta = _high_band_delta(original[left:right], candidate[left:right], rate)
        records[label] = {
            "high_band_delta_db": delta,
            "absolute_limit_db": BOUNDARY_HIGH_BAND_DELTA_DB_MAX,
            "passed": abs(delta) <= BOUNDARY_HIGH_BAND_DELTA_DB_MAX,
        }
    return records


def _measure_output(
    *,
    original_path: Path,
    output_path: Path,
    original: DecodedPcm16Wav,
    interval: dict[str, Any],
    is_identity: bool,
    unclipped_before_encoding: bool,
) -> dict[str, Any]:
    edited = decode_pcm16_mono(output_path)
    start = int(interval["start_sample"])
    end = int(interval["end_sample_exclusive"])
    original_core = original.samples[start:end]
    edited_core = edited.samples[start:end]
    measurements = {
        f"{member:.1f}": measure_formantpath(output_path, interval, member)
        for member in MAX_FORMANTS_FAMILY
    }
    original_measurement = measure_formantpath(original_path, interval, 5.0)
    original_f0 = _pitch_median(original_path, interval)
    edited_f0 = _pitch_median(output_path, interval)
    boundaries = _boundary_checks(
        original.samples, edited.samples, start, end, original.sample_rate_hz
    )
    boundary_spectral = _boundary_spectral_records(
        original.samples, edited.samples, start, end, original.sample_rate_hz
    )
    signed_rms = signed_rms_db(original_core, edited_core)
    high_band_delta = _high_band_delta(
        original_core, edited_core, original.sample_rate_hz
    )
    peak_dbfs = _peak_dbfs(edited.samples)
    f0_change = _relative_change(original_f0, edited_f0)
    f3_change = _relative_change(
        original_measurement.get("f3_hz"), measurements["5.0"].get("f3_hz")
    )
    f4_change = _relative_change(
        original_measurement.get("f4_hz"), measurements["5.0"].get("f4_hz")
    )
    same_container = bool(
        edited.sample_rate_hz == original.sample_rate_hz
        and edited.channels == original.channels
        and edited.sample_width_bytes == original.sample_width_bytes
        and edited.decoded_sample_count == original.decoded_sample_count
    )
    outside_identical = bool(
        same_container
        and np.array_equal(original.samples[:start], edited.samples[:start])
        and np.array_equal(original.samples[end:], edited.samples[end:])
    )
    exact_interval_edges = bool(
        edited.samples[start] == original.samples[start]
        and edited.samples[end - 1] == original.samples[end - 1]
    )
    identity_lsd = (
        low_band_log_spectral_distance_db(
            original_core, edited_core, original.sample_rate_hz
        )
        if is_identity
        else None
    )
    common_pass = bool(
        same_container
        and outside_identical
        and edited.clipped_sample_count == 0
        and unclipped_before_encoding
        and peak_dbfs <= -PEAK_HEADROOM_DB_MIN
        and abs(signed_rms) <= RMS_DELTA_DB_MAX
        and abs(high_band_delta) <= HIGH_BAND_DELTA_DB_MAX
        and f0_change is not None
        and f0_change <= 0.02
        and f3_change is not None
        and f3_change <= 0.05
        and f4_change is not None
        and f4_change <= 0.05
        and all(item["passed"] for item in boundaries.values())
        and exact_interval_edges
        and all(item["passed"] for item in boundary_spectral.values())
    )
    passed = common_pass and (
        not is_identity
        or (identity_lsd is not None and identity_lsd <= IDENTITY_LOW_BAND_LSD_DB_MAX)
    )
    return {
        "decoded_wav": edited.metadata(),
        "measurements": measurements,
        "engineering": {
            "same_container": same_container,
            "outside_interval_bit_identical": outside_identical,
            "clipped_sample_count": edited.clipped_sample_count,
            "unclipped_before_encoding": unclipped_before_encoding,
            "whole_file_peak_dbfs": peak_dbfs,
            "minimum_headroom_db": PEAK_HEADROOM_DB_MIN,
            "signed_rms_db": signed_rms,
            "absolute_rms_limit_db": RMS_DELTA_DB_MAX,
            "high_band_delta_db": high_band_delta,
            "absolute_high_band_limit_db": HIGH_BAND_DELTA_DB_MAX,
            "identity_low_band_lsd_db": identity_lsd,
            "identity_low_band_lsd_limit_db": IDENTITY_LOW_BAND_LSD_DB_MAX,
            "original_f0_hz": original_f0,
            "edited_f0_hz": edited_f0,
            "f0_relative_change": f0_change,
            "f3_relative_change": f3_change,
            "f4_relative_change": f4_change,
            "boundary_checks": boundaries,
            "exact_interval_edge_samples": exact_interval_edges,
            "boundary_spectral_checks": boundary_spectral,
            "passed_without_identity_source_gate": passed,
        },
    }


def _identity_source_gate(
    measurements: dict[str, Any], anchor_gate: dict[str, Any]
) -> dict[str, Any]:
    families = {}
    for key, measurement in measurements.items():
        gate = anchor_gate["families"][key]
        point = np.array([measurement["f1_bark"], measurement["f2_bark"]])
        source_distance = float(
            np.linalg.norm(point - np.array(gate["source_centroid_bark"]))
        )
        target_distance = float(
            np.linalg.norm(point - np.array(gate["target_centroid_bark"]))
        )
        plausible = _plausible(measurement["f1_hz"], measurement["f2_hz"])
        families[key] = {
            "source_distance_bark": source_distance,
            "target_distance_bark": target_distance,
            "source_proximity_pass": source_distance < target_distance,
            "plausibility_pass": plausible,
            "passed": plausible and source_distance < target_distance,
        }
    return {"families": families, "passed": all(x["passed"] for x in families.values())}


def _verify_frozen_inputs() -> tuple[dict[str, Any], dict[str, Any]]:
    if PREREGISTRATION_HEADING not in DEVLOG_PATH.read_text(encoding="utf-8"):
        raise RuntimeError("v2 corrective grid preregistration is missing")
    for label, path in (
        ("source_freeze", FREEZE_PATH),
        ("prior_pass", PRIOR_PASS_PATH),
        ("diagnosis", DIAGNOSIS_PATH),
    ):
        if sha256_file(path) != FROZEN_HASHES[label]:
            raise RuntimeError(f"frozen {label} hash changed")
    freeze = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    prior = json.loads(PRIOR_PASS_PATH.read_text(encoding="utf-8"))
    source = Path(freeze["rules"][RULE_ID]["decoded_wav"]["path"])
    if sha256_file(source) != SOURCE_SHA256:
        raise RuntimeError("frozen source WAV changed")
    return freeze, prior


def run_corrective_grid() -> dict[str, Any]:
    freeze, prior = _verify_frozen_inputs()
    output_root = Paths().artifacts / "same-take" / RUN_ID
    result_path = output_root / "corrective-grid.json"
    if result_path.exists():
        raise RuntimeError("same-take-word-v2 corrective grid is already final")

    frozen_rule = freeze["rules"][RULE_ID]
    interval = frozen_rule["singleton_edit_interval"]
    source_path = Path(frozen_rule["decoded_wav"]["path"])
    original = decode_pcm16_mono(source_path)
    coefficients, filter_record = design_complementary_lowpass(original.sample_rate_hz)
    if not filter_record["passed"]:
        raise RuntimeError("preregistered crossover failed its pre-generation response gate")

    start = int(interval["start_sample"])
    end = int(interval["end_sample_exclusive"])
    original_core = original.samples[start:end].astype(np.float64)
    source_low = centered_convolution(original.samples, coefficients)
    source_high = original.samples.astype(np.float64) - source_low
    taper = _raised_cosine_splice(end - start, round(SPLICE_TAPER_S * original.sample_rate_hz))
    delta = np.array(prior["rules"][RULE_ID]["canonical_editor_delta_bark"])
    prior_outputs = prior["rules"][RULE_ID]["outputs"]

    with tempfile.TemporaryDirectory(prefix="same-take-word-v2-") as temp_name:
        temp = Path(temp_name)
        raw: dict[float, DecodedPcm16Wav] = {}
        baseline_hash_check = {}
        for alpha in STRENGTHS:
            raw_path = temp / f"raw-{alpha:.2f}.wav"
            subprocess.run(
                [
                    str(PRAAT),
                    "--run",
                    str(EDITOR_SCRIPT),
                    str(source_path),
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
            baseline_core = raw[alpha].samples[start:end].astype(np.float64) * SHARED_GAIN
            blended = original_core + (baseline_core - original_core) * taper
            baseline_final = original.samples.copy()
            baseline_final[start:end] = np.rint(
                np.clip(blended, -32768, 32767)
            ).astype(np.int16)
            check_path = temp / f"baseline-{alpha:.2f}.wav"
            _write_pcm16(check_path, baseline_final, original.sample_rate_hz)
            label = "identity" if alpha == 0 else f"shift-{int(alpha * 100):03d}"
            expected = prior_outputs[label]["decoded_wav"]["sha256"]
            observed = sha256_file(check_path)
            baseline_hash_check[label] = {
                "expected_sha256": expected,
                "observed_sha256": observed,
                "passed": expected == observed,
            }
        if not all(item["passed"] for item in baseline_hash_check.values()):
            raise RuntimeError("Praat baseline is not deterministic against frozen v1 outputs")

        candidate_records: dict[str, Any] = {}
        for candidate_name in CANDIDATE_ORDER:
            config = CANDIDATES[candidate_name]
            output_records = {}
            identity_measurements = None
            strength_classifications = {}
            for alpha in STRENGTHS:
                label = "identity" if alpha == 0 else f"shift-{int(alpha * 100):03d}"
                if candidate_name == "baseline":
                    path = Path(prior_outputs[label]["decoded_wav"]["path"])
                    gain = SHARED_GAIN
                    unclipped = prior_outputs[label]["engineering"][
                        "unclipped_before_encoding"
                    ]
                else:
                    raw_full = raw[alpha].samples.astype(np.float64)
                    if config.restore_high_band:
                        component = centered_convolution(raw_full, coefficients)[start:end]
                        residual = source_high[start:end]
                    else:
                        component = raw_full[start:end]
                        residual = np.zeros(end - start, dtype=np.float64)
                    gain = (
                        solve_rms_gain(original_core, component, residual, taper)
                        if config.per_condition_gain
                        else SHARED_GAIN
                    )
                    if gain is None:
                        raise RuntimeError(f"no nonnegative RMS gain root: {candidate_name}/{label}")
                    processed = gain * component + residual
                    blended = original_core + (processed - original_core) * taper
                    unclipped = bool(
                        np.max(blended) <= 32767 and np.min(blended) >= -32768
                    )
                    final = original.samples.copy()
                    final[start:end] = np.rint(
                        np.clip(blended, -32768, 32767)
                    ).astype(np.int16)
                    path = output_root / "audio" / f"{candidate_name}__{label}.wav"
                    _write_pcm16(path, final, original.sample_rate_hz)

                record = _measure_output(
                    original_path=source_path,
                    output_path=path,
                    original=original,
                    interval=interval,
                    is_identity=alpha == 0,
                    unclipped_before_encoding=unclipped,
                )
                record["alpha"] = alpha
                record["gain"] = gain
                output_records[label] = record
                if alpha == 0:
                    identity_measurements = record["measurements"]
                else:
                    assert identity_measurements is not None
                    strength_classifications[label] = _classify_strength(
                        identity_measurements=identity_measurements,
                        shifted_measurements=record["measurements"],
                        anchor_gate=freeze["anchor_gates"][RULE_ID],
                    )

            identity_source_gate = _identity_source_gate(
                output_records["identity"]["measurements"],
                freeze["anchor_gates"][RULE_ID],
            )
            identity_technical = output_records["identity"]["engineering"][
                "passed_without_identity_source_gate"
            ]
            identity_pass = identity_technical and identity_source_gate["passed"]
            exact = [
                (alpha, f"shift-{int(alpha * 100):03d}")
                for alpha in STRENGTHS[1:]
                if strength_classifications[f"shift-{int(alpha * 100):03d}"][
                    "classification"
                ]
                == "exact-category"
                and output_records[f"shift-{int(alpha * 100):03d}"]["engineering"][
                    "passed_without_identity_source_gate"
                ]
            ]
            directional = [
                (alpha, f"shift-{int(alpha * 100):03d}")
                for alpha in STRENGTHS[1:]
                if strength_classifications[f"shift-{int(alpha * 100):03d}"][
                    "classification"
                ]
                in {"exact-category", "directional-only"}
                and output_records[f"shift-{int(alpha * 100):03d}"]["engineering"][
                    "passed_without_identity_source_gate"
                ]
            ]
            selected = min(exact)[1] if exact else max(directional)[1] if directional else None
            candidate_records[candidate_name] = {
                "configuration": {
                    "per_condition_gain": config.per_condition_gain,
                    "restore_high_band": config.restore_high_band,
                },
                "identity_source_gate": identity_source_gate,
                "identity_pass": identity_pass,
                "selected_shift": selected if identity_pass else None,
                "passed": bool(identity_pass and selected),
                "outputs": output_records,
                "strength_classifications": strength_classifications,
            }

    selected_candidate = next(
        (name for name in CANDIDATE_ORDER if candidate_records[name]["passed"]), None
    )
    result = {
        "schema_version": 1,
        "status": "eligible_for_blinded_headphone_qc"
        if selected_candidate
        else "corrective_grid_failed",
        "run_id": RUN_ID,
        "preregistration_commit": PREREGISTRATION_COMMIT,
        "frozen_hashes": FROZEN_HASHES,
        "source_sha256": SOURCE_SHA256,
        "editor_script_sha256": sha256_file(EDITOR_SCRIPT),
        "filter": filter_record,
        "baseline_determinism": baseline_hash_check,
        "candidate_order": list(CANDIDATE_ORDER),
        "candidates": candidate_records,
        "selected_candidate": selected_candidate,
        "selected_shift": candidate_records[selected_candidate]["selected_shift"]
        if selected_candidate
        else None,
        "api_calls": 0,
        "api_cost_usd": 0.0,
    }
    result["receipt_sha256"] = hashlib.sha256(
        stable_json(result).encode("utf-8")
    ).hexdigest()
    atomic_write_json(result_path, result)
    return result

