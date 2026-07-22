from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pyworld

from .config import DEVLOG_PATH, Paths, stable_json
from .pcm import decode_pcm16_mono
from .same_take import MAX_FORMANTS_FAMILY
from .same_take_corrective import (
    DIAGNOSIS_PATH,
    FREEZE_PATH,
    PRIOR_PASS_PATH,
    RULE_ID,
    SOURCE_SHA256,
    STRENGTHS,
    _identity_source_gate,
    _measure_output,
    solve_rms_gain,
)
from .same_take_word import (
    SHIFT_TAPER_S,
    SPLICE_TAPER_S,
    _classify_strength,
    _raised_cosine_splice,
    _write_pcm16,
)
from .util import atomic_write_json, sha256_file


RUN_ID = "20260715-same-take-word-v3"
PREREGISTRATION_HEADING = "## Same-take-word-v3 WORLD preregistration — July 15, 2026"
PREREGISTRATION_COMMIT = "114e053"
V2_RESULT_PATH = (
    Paths().artifacts
    / "same-take"
    / "20260715-same-take-word-v2"
    / "corrective-grid.json"
)
V2_RESULT_SHA256 = "2da76ebc1ce232a3c4f4bd785fc03a8a53f30d6ae3d28d8436428c123e698393"
PYWORLD_VERSION = "0.3.5"
PYWORLD_EXTENSION_SHA256 = "56864d0c027e3b3893d63ced99bf19eb48e75b63f95e0101d392372151b57c30"
FRAME_PERIOD_MS = 5.0
FFT_SIZE = 1024
SOURCE_FORMANTS_HZ = (558.760239, 2082.752696, 2955.859882, 4334.203846)
FROZEN_DELTA_BARK = (-2.280007554222678, 2.3070023450657615)


def _hz_to_bark(value: float) -> float:
    return 26.81 / (1 + 1960 / value) - 0.53


def _bark_to_hz(value: float) -> float:
    return 1960 / (26.81 / (value + 0.53) - 1)


def shift_envelope(time_s: float, start_s: float, end_s: float) -> float:
    if time_s < start_s or time_s > end_s:
        return 0.0
    return max(
        0.0,
        min(
            1.0,
            (time_s - start_s) / SHIFT_TAPER_S,
            (end_s - time_s) / SHIFT_TAPER_S,
        ),
    )


def warp_spectral_envelope(
    spectral_envelope: np.ndarray,
    time_axis: np.ndarray,
    sample_rate_hz: int,
    alpha: float,
    start_s: float,
    end_s: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    if alpha == 0:
        return spectral_envelope.copy(), {
            "affected_frame_count": 0,
            "minimum_destination_knot_spacing_hz": None,
        }
    output = spectral_envelope.copy()
    frequencies = np.linspace(0.0, sample_rate_hz / 2, spectral_envelope.shape[1])
    f1, f2, f3, f4 = SOURCE_FORMANTS_HZ
    source_knots = np.array([0.0, f1, f2, f3, f4, sample_rate_hz / 2])
    minimum_spacing = math.inf
    affected = 0
    for index, time_s in enumerate(time_axis):
        envelope = shift_envelope(float(time_s), start_s, end_s)
        if envelope <= 0:
            continue
        effective = alpha * envelope
        f1_target = _bark_to_hz(
            _hz_to_bark(f1) + effective * FROZEN_DELTA_BARK[0]
        )
        f2_target = _bark_to_hz(
            _hz_to_bark(f2) + effective * FROZEN_DELTA_BARK[1]
        )
        f3_target = max(f3, f2_target + 1.0)
        destination_knots = np.array(
            [0.0, f1_target, f2_target, f3_target, f4, sample_rate_hz / 2]
        )
        spacings = np.diff(destination_knots)
        if np.any(spacings <= 0):
            raise RuntimeError(f"non-monotonic WORLD warp knots at frame {index}")
        minimum_spacing = min(minimum_spacing, float(np.min(spacings)))
        source_coordinates = np.interp(
            frequencies, destination_knots, source_knots
        )
        log_power = np.log(np.maximum(spectral_envelope[index], 1e-30))
        output[index] = np.exp(
            np.interp(source_coordinates, frequencies, log_power)
        )
        affected += 1
    return output, {
        "affected_frame_count": affected,
        "minimum_destination_knot_spacing_hz": minimum_spacing
        if affected
        else None,
    }


def _extension_path() -> Path:
    module = Path(pyworld.__file__)
    matches = sorted(module.parent.glob("pyworld*.so"))
    if len(matches) != 1:
        raise RuntimeError("expected one compiled PyWORLD extension")
    return matches[0]


def _verify_inputs() -> tuple[dict[str, Any], dict[str, Any]]:
    if PREREGISTRATION_HEADING not in DEVLOG_PATH.read_text(encoding="utf-8"):
        raise RuntimeError("WORLD preregistration is missing")
    if sha256_file(V2_RESULT_PATH) != V2_RESULT_SHA256:
        raise RuntimeError("frozen v2 result changed")
    if sha256_file(_extension_path()) != PYWORLD_EXTENSION_SHA256:
        raise RuntimeError("compiled PyWORLD extension changed")
    freeze = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    prior = json.loads(PRIOR_PASS_PATH.read_text(encoding="utf-8"))
    source_path = Path(freeze["rules"][RULE_ID]["decoded_wav"]["path"])
    if sha256_file(source_path) != SOURCE_SHA256:
        raise RuntimeError("frozen source changed")
    return freeze, prior


def run_world_pass() -> dict[str, Any]:
    freeze, _ = _verify_inputs()
    output_root = Paths().artifacts / "same-take" / RUN_ID
    result_path = output_root / "world-pass.json"
    if result_path.exists():
        raise RuntimeError("WORLD pass is already final")

    frozen_rule = freeze["rules"][RULE_ID]
    interval = frozen_rule["singleton_edit_interval"]
    source_path = Path(frozen_rule["decoded_wav"]["path"])
    original = decode_pcm16_mono(source_path)
    source_float = original.samples.astype(np.float64) / 32768.0
    f0_raw, time_axis = pyworld.dio(
        source_float,
        original.sample_rate_hz,
        frame_period=FRAME_PERIOD_MS,
    )
    f0 = pyworld.stonemask(
        source_float, f0_raw, time_axis, original.sample_rate_hz
    )
    spectral = pyworld.cheaptrick(
        source_float,
        f0,
        time_axis,
        original.sample_rate_hz,
        fft_size=FFT_SIZE,
    )
    aperiodicity = pyworld.d4c(
        source_float, f0, time_axis, original.sample_rate_hz
    )
    if spectral.shape != aperiodicity.shape or spectral.shape[1] != FFT_SIZE // 2 + 1:
        raise RuntimeError("WORLD analysis dimensions violate the frozen protocol")

    start = int(interval["start_sample"])
    end = int(interval["end_sample_exclusive"])
    original_core = original.samples[start:end].astype(np.float64)
    taper = _raised_cosine_splice(
        end - start, round(SPLICE_TAPER_S * original.sample_rate_hz)
    )
    output_records: dict[str, Any] = {}
    warp_records: dict[str, Any] = {}
    identity_measurements = None
    strength_classifications: dict[str, Any] = {}

    for alpha in STRENGTHS:
        label = "identity" if alpha == 0 else f"shift-{int(alpha * 100):03d}"
        warped, warp_record = warp_spectral_envelope(
            spectral,
            time_axis,
            original.sample_rate_hz,
            alpha,
            float(interval["start_s"]),
            float(interval["end_s"]),
        )
        synthesized = pyworld.synthesize(
            f0,
            warped,
            aperiodicity,
            original.sample_rate_hz,
            frame_period=FRAME_PERIOD_MS,
        )
        if synthesized.size < original.decoded_sample_count:
            raise RuntimeError(f"WORLD output shorter than source: {label}")
        synthesized_pcm = synthesized[: original.decoded_sample_count] * 32768.0
        component = synthesized_pcm[start:end]
        residual = np.zeros(end - start, dtype=np.float64)
        gain = solve_rms_gain(original_core, component, residual, taper)
        if gain is None:
            raise RuntimeError(f"no nonnegative WORLD RMS gain: {label}")
        processed = gain * component
        blended = original_core + (processed - original_core) * taper
        unclipped = bool(np.max(blended) <= 32767 and np.min(blended) >= -32768)
        final = original.samples.copy()
        final[start:end] = np.rint(np.clip(blended, -32768, 32767)).astype(np.int16)
        path = output_root / "audio" / f"world__{label}.wav"
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
        record["world_synthesized_sample_count"] = int(synthesized.size)
        output_records[label] = record
        warp_records[label] = warp_record
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
    identity_pass = bool(
        output_records["identity"]["engineering"][
            "passed_without_identity_source_gate"
        ]
        and identity_source_gate["passed"]
    )
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
    result = {
        "schema_version": 1,
        "status": "eligible_for_blinded_headphone_qc"
        if identity_pass and selected
        else "world_pass_failed",
        "run_id": RUN_ID,
        "preregistration_commit": PREREGISTRATION_COMMIT,
        "v2_result_sha256": V2_RESULT_SHA256,
        "source_sha256": SOURCE_SHA256,
        "pyworld_version": PYWORLD_VERSION,
        "pyworld_extension_sha256": PYWORLD_EXTENSION_SHA256,
        "analysis": {
            "frame_period_ms": FRAME_PERIOD_MS,
            "fft_size": FFT_SIZE,
            "frame_count": int(time_axis.size),
            "voiced_frame_count": int(np.count_nonzero(f0 > 0)),
            "spectral_shape": list(spectral.shape),
            "aperiodicity_shape": list(aperiodicity.shape),
        },
        "identity_source_gate": identity_source_gate,
        "identity_pass": identity_pass,
        "selected_shift": selected if identity_pass else None,
        "outputs": output_records,
        "warp_records": warp_records,
        "strength_classifications": strength_classifications,
        "api_calls": 0,
        "api_cost_usd": 0.0,
    }
    result["receipt_sha256"] = hashlib.sha256(
        stable_json(result).encode("utf-8")
    ).hexdigest()
    atomic_write_json(result_path, result)
    return result

