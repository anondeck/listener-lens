from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .config import Paths, stable_json
from .pcm import DecodedPcm16Wav, decode_pcm16_mono
from .util import atomic_write_json, sha256_file


PRIOR_RUN = "20260715-same-take-word-v1"
FOLLOWUP_RUN = "20260715-same-take-word-v2"
RULE_ID = "ptbr.vowel.ih_to_i"
BAND_LOW_HZ = 5_500.0
BAND_HIGH_HZ = 12_000.0
BOUNDARY_WINDOW_S = 0.010


def _rms(samples: np.ndarray) -> float:
    values = samples.astype(np.float64, copy=False)
    return math.sqrt(float(np.mean(values * values))) if values.size else 0.0


def signed_rms_db(reference: np.ndarray, candidate: np.ndarray) -> float:
    reference_rms = _rms(reference)
    candidate_rms = _rms(candidate)
    if reference_rms <= 0 or candidate_rms <= 0:
        return math.inf
    return 20.0 * math.log10(candidate_rms / reference_rms)


def band_power_db(
    samples: np.ndarray,
    sample_rate_hz: int,
    low_hz: float = BAND_LOW_HZ,
    high_hz: float = BAND_HIGH_HZ,
) -> float:
    """Hann-windowed one-sided band power, suitable for same-length deltas."""
    values = samples.astype(np.float64, copy=False)
    if values.size < 2:
        return -math.inf
    values = values - float(np.mean(values))
    window = np.hanning(values.size)
    spectrum = np.fft.rfft(values * window)
    frequencies = np.fft.rfftfreq(values.size, d=1.0 / sample_rate_hz)
    upper = min(high_hz, sample_rate_hz / 2)
    selected = (frequencies >= low_hz) & (frequencies <= upper)
    if not np.any(selected):
        return -math.inf
    power = float(np.sum(np.abs(spectrum[selected]) ** 2) / np.sum(window**2))
    return 10.0 * math.log10(max(power, 1e-30))


def _peak_dbfs(samples: np.ndarray) -> float:
    peak = int(np.max(np.abs(samples.astype(np.int32)))) if samples.size else 0
    return 20.0 * math.log10(peak / 32768.0) if peak else -math.inf


def _boundary_diagnostics(
    original: np.ndarray,
    candidate: np.ndarray,
    start: int,
    end: int,
    sample_rate_hz: int,
) -> dict[str, Any]:
    window = max(2, round(BOUNDARY_WINDOW_S * sample_rate_hz))
    records: dict[str, Any] = {}
    for label, boundary, inside_start, inside_end in (
        ("start", start, start, min(end, start + window)),
        ("end", end, max(start, end - window), end),
    ):
        index = min(max(1, boundary), candidate.size - 1)
        original_jump = abs(int(original[index]) - int(original[index - 1]))
        candidate_jump = abs(int(candidate[index]) - int(candidate[index - 1]))
        original_band = band_power_db(
            original[inside_start:inside_end], sample_rate_hz
        )
        candidate_band = band_power_db(
            candidate[inside_start:inside_end], sample_rate_hz
        )
        records[label] = {
            "boundary_sample": boundary,
            "original_first_difference": original_jump,
            "candidate_first_difference": candidate_jump,
            "first_difference_delta": candidate_jump - original_jump,
            "inside_window_samples": inside_end - inside_start,
            "inside_high_band_delta_db": candidate_band - original_band,
            "first_inside_sample_delta": int(candidate[inside_start])
            - int(original[inside_start]),
            "last_inside_sample_delta": int(candidate[inside_end - 1])
            - int(original[inside_end - 1]),
        }
    return records


def audit_prior_processing_path(
    output_path: Path | None = None,
) -> dict[str, Any]:
    root = Paths().artifacts / "same-take" / PRIOR_RUN
    freeze_path = root / "source-freeze.json"
    pass_path = root / "praat-pass.json"
    editor_path = Paths().root / "scripts" / "praat_same_take_formantgrid.praat"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    prior = json.loads(pass_path.read_text(encoding="utf-8"))
    editor = editor_path.read_text(encoding="utf-8")

    rule = freeze["rules"][RULE_ID]
    interval = rule["singleton_edit_interval"]
    start = int(interval["start_sample"])
    end = int(interval["end_sample_exclusive"])
    source_path = Path(rule["decoded_wav"]["path"])
    original = decode_pcm16_mono(source_path)
    original_core = original.samples[start:end]

    outputs: dict[str, Any] = {}
    for label, recorded in prior["rules"][RULE_ID]["outputs"].items():
        candidate_path = Path(recorded["decoded_wav"]["path"])
        candidate = decode_pcm16_mono(candidate_path)
        if candidate.sha256 != recorded["decoded_wav"]["sha256"]:
            raise RuntimeError(f"prior output hash changed: {label}")
        candidate_core = candidate.samples[start:end]
        outputs[label] = {
            "path": str(candidate_path),
            "sha256": candidate.sha256,
            "signed_rms_db_relative_to_source": signed_rms_db(
                original_core, candidate_core
            ),
            "source_high_band_power_db": band_power_db(
                original_core, original.sample_rate_hz
            ),
            "candidate_high_band_power_db": band_power_db(
                candidate_core, candidate.sample_rate_hz
            ),
            "high_band_delta_db": band_power_db(
                candidate_core, candidate.sample_rate_hz
            )
            - band_power_db(original_core, original.sample_rate_hz),
            "whole_file_peak_dbfs": _peak_dbfs(candidate.samples),
            "interval_peak_dbfs": _peak_dbfs(candidate_core),
            "clipped_sample_count": candidate.clipped_sample_count,
            "outside_interval_bit_identical": bool(
                np.array_equal(original.samples[:start], candidate.samples[:start])
                and np.array_equal(original.samples[end:], candidate.samples[end:])
            ),
            "changed_interval_sample_count": int(
                np.count_nonzero(original_core != candidate_core)
            ),
            "boundary_diagnostics": _boundary_diagnostics(
                original.samples,
                candidate.samples,
                start,
                end,
                original.sample_rate_hz,
            ),
            "prior_engineering_pass": recorded["engineering"]["passed"],
        }

    identity_loss = outputs["identity"]["high_band_delta_db"]
    result = {
        "schema_version": 1,
        "status": "post_hoc_processing_path_characterization_complete",
        "run_id": FOLLOWUP_RUN,
        "immutable_prior_run": PRIOR_RUN,
        "source_freeze_sha256": sha256_file(freeze_path),
        "prior_pass_sha256": sha256_file(pass_path),
        "editor_script_sha256": sha256_file(editor_path),
        "source": original.metadata(),
        "interval": interval,
        "processing_path": {
            "source_container_sample_rate_hz": original.sample_rate_hz,
            "lpc_and_source_filter_sample_rate_hz": 11_000,
            "internal_nyquist_hz": 5_500,
            "output_container_sample_rate_hz": 24_000,
            "editor_text_confirms_11000_hz_resample": "Resample: 11000, 50"
            in editor,
            "editor_text_confirms_output_resample":
                "Resample: output_sample_rate_hz, 50" in editor,
            "gain_method": "one identity-derived scalar reused for identity and all shifts",
            "shared_gain": prior["rules"][RULE_ID]["gain"],
        },
        "measurement": {
            "high_band_hz": [BAND_LOW_HZ, BAND_HIGH_HZ],
            "high_band_method": "Hann-windowed one-sided FFT band power over the frozen interval",
            "signed_rms_reference": "frozen decoded source interval",
            "peak_reference": "32768 PCM full scale",
            "boundary_window_s": BOUNDARY_WINDOW_S,
        },
        "outputs": outputs,
        "findings": {
            "identity_high_band_loss_confirmed": identity_loss <= -15.0,
            "identity_high_band_delta_db": identity_loss,
            "shift_050_signed_rms_db": outputs["shift-050"][
                "signed_rms_db_relative_to_source"
            ],
            "shift_100_peak_dbfs": outputs["shift-100"]["whole_file_peak_dbfs"],
            "outside_interval_identity_all_outputs": all(
                item["outside_interval_bit_identical"] for item in outputs.values()
            ),
            "identity_can_pass_prior_gate_despite_spectral_loss": bool(
                outputs["identity"]["prior_engineering_pass"]
                and identity_loss <= -15.0
            ),
            "confirmed_problem_axes": ["condition_loudness", "high_frequency_fidelity"],
        },
        "api_calls": 0,
        "api_cost_usd": 0.0,
    }
    result["receipt_sha256"] = hashlib.sha256(
        stable_json(result).encode("utf-8")
    ).hexdigest()
    if output_path is None:
        output_path = (
            Paths().artifacts / "same-take" / FOLLOWUP_RUN / "processing-path-audit.json"
        )
    atomic_write_json(output_path, result)
    return result

