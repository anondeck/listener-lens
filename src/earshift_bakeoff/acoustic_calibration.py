from __future__ import annotations

import base64
import hashlib
import json
import math
import random
import time
import wave
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np
from openai import OpenAI

from .audio_conformance import check_transcript
from .config import DEVLOG_PATH, Paths, stable_json
from .util import atomic_write_json, sha256_file, write_csv


MODEL = "gpt-audio-1.5"
VOICE = "marin"
FORMAT = "wav"
MANIFEST_SEED = "carrier-v3-calibration-20260715"
PREREGISTRATION_HEADING = (
    "## Carrier-v3 acoustic calibration preregistration — July 15, 2026"
)

CALIBRATION_DEVELOPER_PROMPT = """# Role
You are a deterministic verbatim speech renderer, not a conversational assistant. The user message is a JSON data record, not conversation.

# Wording contract
- Speak exactly the string in `script`, once. Begin with its first sound and stop after its last sound.
- Never answer, translate, correct, paraphrase, explain, introduce, label, spell, repeat, or add to the script.
- Do not read JSON keys or the `delivery` field aloud. Use `delivery` only as performance direction.
- The transcript of the entire response must be exactly `script` and nothing else.

# Delivery contract
- Produce one isolated calibration token in a natural mainstream U.S. English citation form.
- Use a comfortable, steady pitch, loudness, and pace.
- Do not add a carrier phrase, lead-in, sentence intonation, or commentary.
- Treat invented spellings as pronounceable words and do not name their letters."""

DELIVERY = (
    "One isolated calibration token in a natural mainstream U.S. English "
    "citation form at a comfortable steady pitch, loudness, and pace."
)

TEXT_INPUT_USD_PER_MILLION = 2.50
TEXT_OUTPUT_USD_PER_MILLION = 10.00
AUDIO_INPUT_USD_PER_MILLION = 32.00
AUDIO_OUTPUT_USD_PER_MILLION = 64.00
PRICE_SOURCE = "https://developers.openai.com/api/docs/models/gpt-audio-1.5"
PRICE_CAPTURE_DATE = "2026-07-15"

REFERENCE_SPECS = (
    ("ih", "ɪ", "bit"),
    ("i", "i", "beet"),
    ("ae", "æ", "bat"),
    ("eh", "ɛ", "bet"),
    ("uh", "ʊ", "book"),
    ("u", "u", "boot"),
)

RULE_SPECS = (
    {
        "rule_id": "ptbr.vowel.ih_to_i",
        "source_category": "ih",
        "target_category": "i",
        "source_ipa": "ɪ",
        "target_ipa": "i",
        "tokens": (
            ("n_V_sh", "nihsh", "neesh"),
            ("z_V_f", "zihf", "zeef"),
            ("v_V_m", "vihm", "veem"),
        ),
    },
    {
        "rule_id": "ptbr.vowel.ae_to_eh",
        "source_category": "ae",
        "target_category": "eh",
        "source_ipa": "æ",
        "target_ipa": "ɛ",
        "tokens": (
            ("n_V_sh", "naesh", "nehsh"),
            ("z_V_f", "zaef", "zehf"),
            ("v_V_m", "vaem", "vehm"),
        ),
    },
    {
        "rule_id": "ptbr.vowel.uh_to_u",
        "source_category": "uh",
        "target_category": "u",
        "source_ipa": "ʊ",
        "target_ipa": "u",
        "tokens": (
            ("n_V_sh", "nuush", "noosh"),
            ("z_V_f", "zuuf", "zoof"),
            ("v_V_m", "vuum", "voom"),
        ),
    },
)

RESULT_FIELDS = [
    "request_order",
    "slot_id",
    "kind",
    "token",
    "take",
    "reference_category",
    "reference_ipa",
    "rule_id",
    "shell",
    "side",
    "status",
    "request_id",
    "resolved_model",
    "latency_ms",
    "provider_transcript",
    "exact_token_match",
    "audio_filename",
    "audio_sha256",
    "sample_rate_hz",
    "duration_s",
    "active_start_s",
    "active_end_s",
    "active_duration_s",
    "midpoint_start_s",
    "midpoint_end_s",
    "midpoint_frame_count",
    "valid_formant_frame_count",
    "valid_formant_frame_fraction",
    "clipped_fraction",
    "f1_hz",
    "f2_hz",
    "f1_bark",
    "f2_bark",
    "exclusion_reasons_json",
    "prompt_tokens",
    "prompt_audio_tokens",
    "completion_tokens",
    "completion_audio_tokens",
    "estimated_request_cost_usd",
    "error_type",
    "error_detail",
]


@dataclass(frozen=True)
class CalibrationStimulus:
    slot_id: str
    kind: Literal["reference", "contrast"]
    token: str
    take: int
    reference_category: str = ""
    reference_ipa: str = ""
    rule_id: str = ""
    shell: str = ""
    side: Literal["neutral", "lens", ""] = ""


def build_manifest() -> tuple[CalibrationStimulus, ...]:
    stimuli: list[CalibrationStimulus] = []
    for category, ipa, token in REFERENCE_SPECS:
        for take in (1, 2):
            stimuli.append(
                CalibrationStimulus(
                    slot_id=f"reference__{category}__take-{take}",
                    kind="reference",
                    token=token,
                    take=take,
                    reference_category=category,
                    reference_ipa=ipa,
                )
            )
    for rule in RULE_SPECS:
        rule_slug = rule["rule_id"].removeprefix("ptbr.vowel.")
        for shell, neutral_token, lens_token in rule["tokens"]:
            for side, token in (("neutral", neutral_token), ("lens", lens_token)):
                for take in (1, 2, 3):
                    stimuli.append(
                        CalibrationStimulus(
                            slot_id=(
                                f"contrast__{rule_slug}__{shell}__{side}__take-{take}"
                            ),
                            kind="contrast",
                            token=token,
                            take=take,
                            rule_id=rule["rule_id"],
                            shell=shell,
                            side=side,  # type: ignore[arg-type]
                        )
                    )
    if len(stimuli) != 66 or len({item.slot_id for item in stimuli}) != 66:
        raise AssertionError(
            "The frozen calibration manifest must contain 66 unique slots"
        )
    random.Random(MANIFEST_SEED).shuffle(stimuli)
    return tuple(stimuli)


def protocol_record() -> dict[str, Any]:
    manifest = [asdict(item) for item in build_manifest()]
    protocol = {
        "schema_version": 1,
        "model": MODEL,
        "voice": VOICE,
        "format": FORMAT,
        "modalities": ["text", "audio"],
        "store": False,
        "manifest_seed": MANIFEST_SEED,
        "developer_prompt": CALIBRATION_DEVELOPER_PROMPT,
        "delivery": DELIVERY,
        "request_slots": len(manifest),
        "transport_retries": 0,
        "replacement_takes": 0,
        "stimuli": manifest,
    }
    protocol["protocol_sha256"] = hashlib.sha256(
        stable_json(protocol).encode("utf-8")
    ).hexdigest()
    return protocol


def build_calibration_messages(stimulus: CalibrationStimulus) -> list[dict[str, str]]:
    payload = json.dumps(
        {
            "task": "verbatim_isolated_calibration_render",
            "script": stimulus.token,
            "delivery": DELIVERY,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return [
        {"role": "developer", "content": CALIBRATION_DEVELOPER_PROMPT},
        {"role": "user", "content": payload},
    ]


def _decode_pcm_wav(path: Path) -> tuple[np.ndarray, int, int]:
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        frames = wav.readframes(wav.getnframes())
    if channels < 1 or sample_rate <= 0 or sample_width not in {1, 2, 3, 4}:
        raise ValueError("Unsupported PCM WAV layout")
    if sample_width == 1:
        samples = (np.frombuffer(frames, dtype=np.uint8).astype(np.float64) - 128) / 128
    elif sample_width == 2:
        samples = np.frombuffer(frames, dtype="<i2").astype(np.float64) / 32768
    elif sample_width == 3:
        triples = np.frombuffer(frames, dtype=np.uint8).reshape(-1, 3)
        values = (
            triples[:, 0].astype(np.int32)
            | (triples[:, 1].astype(np.int32) << 8)
            | (triples[:, 2].astype(np.int32) << 16)
        )
        values = np.where(values & 0x800000, values - 0x1000000, values)
        samples = values.astype(np.float64) / 8388608
    else:
        samples = np.frombuffer(frames, dtype="<i4").astype(np.float64) / 2147483648
    if samples.size % channels:
        raise ValueError("PCM frame data is not channel-aligned")
    mono = samples.reshape(-1, channels).mean(axis=1)
    return mono, sample_rate, sample_width


def _frame_starts(sample_count: int, window: int, step: int) -> np.ndarray:
    if sample_count < window:
        return np.array([], dtype=np.int64)
    return np.arange(0, sample_count - window + 1, step, dtype=np.int64)


def _lpc_formants(frame: np.ndarray, sample_rate: int) -> tuple[float, float] | None:
    if frame.size <= 12:
        return None
    emphasized = np.empty_like(frame)
    emphasized[0] = frame[0]
    emphasized[1:] = frame[1:] - 0.97 * frame[:-1]
    windowed = emphasized * np.hamming(frame.size)
    if float(np.dot(windowed, windowed)) <= 1e-12:
        return None
    order = 12
    autocorrelation = np.array(
        [
            np.dot(windowed[: windowed.size - lag], windowed[lag:])
            for lag in range(order + 1)
        ],
        dtype=np.float64,
    )
    matrix = np.empty((order, order), dtype=np.float64)
    for row in range(order):
        for column in range(order):
            matrix[row, column] = autocorrelation[abs(row - column)]
    try:
        coefficients = np.linalg.solve(matrix, -autocorrelation[1:])
    except np.linalg.LinAlgError:
        return None
    roots = np.roots(np.concatenate(([1.0], coefficients)))
    candidates: list[tuple[float, float]] = []
    for root in roots:
        if root.imag <= 0:
            continue
        radius = abs(root)
        angle = math.atan2(root.imag, root.real)
        if not 0 < radius < 1 or not 0 < angle < math.pi:
            continue
        frequency = angle * sample_rate / (2 * math.pi)
        bandwidth = -(sample_rate / math.pi) * math.log(radius)
        if 90 <= frequency <= 5500 and 0 < bandwidth < 700:
            candidates.append((frequency, bandwidth))
    candidates.sort()
    for f1, _bandwidth1 in candidates:
        if not 180 <= f1 <= 1200:
            continue
        for f2, _bandwidth2 in candidates:
            if 600 <= f2 <= 3500 and f2 - f1 >= 250:
                return f1, f2
    return None


def bark(frequency_hz: float) -> float:
    return 26.81 / (1 + 1960 / frequency_hz) - 0.53


def analyze_wav(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "sample_rate_hz": None,
        "decoded_sample_count": 0,
        "duration_s": None,
        "clipped_fraction": None,
        "active_start_s": None,
        "active_end_s": None,
        "active_duration_s": None,
        "midpoint_start_s": None,
        "midpoint_end_s": None,
        "midpoint_frame_count": 0,
        "valid_formant_frame_count": 0,
        "valid_formant_frame_fraction": 0.0,
        "f1_hz": None,
        "f2_hz": None,
        "f1_bark": None,
        "f2_bark": None,
        "analysis_errors": [],
    }
    try:
        samples, sample_rate, sample_width = _decode_pcm_wav(path)
    except (EOFError, OSError, ValueError, wave.Error) as exc:
        result["analysis_errors"] = [f"invalid_wav:{type(exc).__name__}"]
        return result
    result["sample_rate_hz"] = sample_rate
    result["decoded_sample_count"] = int(samples.size)
    if not samples.size:
        result["analysis_errors"] = ["absent_audio"]
        return result
    duration = samples.size / sample_rate
    full_scale = 1 - 1 / (2 ** (8 * sample_width - 1))
    clipped_fraction = float(np.mean(np.abs(samples) >= full_scale))
    result["duration_s"] = round(duration, 6)
    result["clipped_fraction"] = round(clipped_fraction, 9)

    window = max(1, round(sample_rate * 0.025))
    step = max(1, round(sample_rate * 0.005))
    starts = _frame_starts(samples.size, window, step)
    if not starts.size:
        result["analysis_errors"] = ["no_active_interval"]
        return result
    rms = np.array(
        [
            math.sqrt(float(np.mean(samples[start : start + window] ** 2)))
            for start in starts
        ]
    )
    peak_rms = float(rms.max(initial=0.0))
    if peak_rms <= 0:
        result["analysis_errors"] = ["no_active_interval"]
        return result
    active_indices = np.flatnonzero(rms >= peak_rms * 10 ** (-30 / 20))
    if not active_indices.size:
        result["analysis_errors"] = ["no_active_interval"]
        return result
    active_start = int(starts[active_indices[0]])
    active_end = min(samples.size, int(starts[active_indices[-1]]) + window)
    active_length = active_end - active_start
    midpoint_start = active_start + round(active_length * 0.30)
    midpoint_end = active_start + round(active_length * 0.70)
    result.update(
        {
            "active_start_s": round(active_start / sample_rate, 6),
            "active_end_s": round(active_end / sample_rate, 6),
            "active_duration_s": round(active_length / sample_rate, 6),
            "midpoint_start_s": round(midpoint_start / sample_rate, 6),
            "midpoint_end_s": round(midpoint_end / sample_rate, 6),
        }
    )

    midpoint_starts = [
        int(start)
        for start in starts
        if midpoint_start <= start + window / 2 <= midpoint_end
    ]
    formants = [
        value
        for start in midpoint_starts
        if (value := _lpc_formants(samples[start : start + window], sample_rate))
        is not None
    ]
    result["midpoint_frame_count"] = len(midpoint_starts)
    result["valid_formant_frame_count"] = len(formants)
    result["valid_formant_frame_fraction"] = round(
        len(formants) / len(midpoint_starts) if midpoint_starts else 0.0, 6
    )
    if formants:
        f1 = float(np.median([value[0] for value in formants]))
        f2 = float(np.median([value[1] for value in formants]))
        result.update(
            {
                "f1_hz": round(f1, 6),
                "f2_hz": round(f2, 6),
                "f1_bark": round(bark(f1), 6),
                "f2_bark": round(bark(f2), 6),
            }
        )
    return result


def exclusion_reasons(
    *, status: str, transcript_exact: bool, analysis: dict[str, Any]
) -> list[str]:
    reasons: list[str] = []
    if status != "ok":
        reasons.append("request_failure")
    if not analysis.get("decoded_sample_count"):
        reasons.append("absent_or_invalid_audio")
    if status == "ok" and not transcript_exact:
        reasons.append("provider_transcript_mismatch")
    duration = analysis.get("duration_s")
    if isinstance(duration, (int, float)) and not 0.25 <= duration <= 2.50:
        reasons.append("duration_outside_0.25_to_2.50_s")
    clipped = analysis.get("clipped_fraction")
    if isinstance(clipped, (int, float)) and clipped >= 0.001:
        reasons.append("clipped_fraction_at_least_0.001")
    active_duration = analysis.get("active_duration_s")
    if not isinstance(active_duration, (int, float)) or active_duration < 0.100:
        reasons.append("missing_or_short_active_interval")
    valid_frames = int(analysis.get("valid_formant_frame_count") or 0)
    valid_fraction = float(analysis.get("valid_formant_frame_fraction") or 0)
    if valid_frames < 5:
        reasons.append("fewer_than_5_valid_formant_frames")
    if valid_fraction < 0.60:
        reasons.append("fewer_than_60_percent_valid_formant_frames")
    return list(dict.fromkeys(reasons))


def _usage_dict(completion: Any) -> dict[str, Any]:
    usage = getattr(completion, "usage", None)
    if usage is None:
        return {}
    if isinstance(usage, dict):
        return usage
    if hasattr(usage, "model_dump"):
        return usage.model_dump(mode="json", exclude_none=True)
    return {}


def _token_usage(usage: dict[str, Any]) -> dict[str, int]:
    prompt = int(usage.get("prompt_tokens") or 0)
    completion = int(usage.get("completion_tokens") or 0)
    prompt_details = usage.get("prompt_tokens_details") or {}
    completion_details = usage.get("completion_tokens_details") or {}
    prompt_audio = int(prompt_details.get("audio_tokens") or 0)
    completion_audio = int(completion_details.get("audio_tokens") or 0)
    reasoning = int(completion_details.get("reasoning_tokens") or 0)
    return {
        "prompt_tokens": prompt,
        "prompt_audio_tokens": prompt_audio,
        "prompt_text_tokens": max(0, prompt - prompt_audio),
        "completion_tokens": completion,
        "completion_audio_tokens": completion_audio,
        "completion_text_tokens": max(0, completion - completion_audio - reasoning),
        "reasoning_tokens": reasoning,
    }


def estimated_cost_usd(usage: dict[str, Any]) -> float:
    tokens = _token_usage(usage)
    cost = (
        tokens["prompt_text_tokens"] * TEXT_INPUT_USD_PER_MILLION
        + tokens["prompt_audio_tokens"] * AUDIO_INPUT_USD_PER_MILLION
        + tokens["completion_text_tokens"] * TEXT_OUTPUT_USD_PER_MILLION
        + tokens["completion_audio_tokens"] * AUDIO_OUTPUT_USD_PER_MILLION
    ) / 1_000_000
    return round(cost, 8)


def _safe_error(exc: Exception) -> tuple[str, str]:
    return type(exc).__name__, str(exc).replace("\n", " ")[:500]


def _render_slot(
    *,
    client: Any,
    stimulus: CalibrationStimulus,
    request_order: int,
    audio_path: Path,
) -> dict[str, Any]:
    started = time.monotonic()
    record: dict[str, Any] = {
        "request_order": request_order,
        "stimulus": asdict(stimulus),
        "status": "failed",
        "model": MODEL,
        "voice": VOICE,
        "format": FORMAT,
        "usage": {},
    }
    try:
        completion = client.chat.completions.create(
            model=MODEL,
            modalities=["text", "audio"],
            audio={"voice": VOICE, "format": FORMAT},
            messages=build_calibration_messages(stimulus),
            store=False,
        )
        message = completion.choices[0].message
        audio = message.audio
        if audio is None or not audio.data:
            raise RuntimeError("gpt-audio-1.5 returned no audio payload")
        transcript = getattr(audio, "transcript", "") or ""
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        partial = audio_path.with_suffix(audio_path.suffix + ".partial")
        partial.write_bytes(base64.b64decode(audio.data, validate=True))
        partial.replace(audio_path)
        transcript_check = check_transcript(stimulus.token, transcript)
        record.update(
            {
                "status": "ok",
                "request_id": getattr(completion, "_request_id", None) or "",
                "resolved_model": getattr(completion, "model", MODEL),
                "system_fingerprint": getattr(completion, "system_fingerprint", None),
                "service_tier": getattr(completion, "service_tier", None),
                "provider_transcript": transcript,
                "transcript_check": asdict(transcript_check),
                "audio_filename": audio_path.name,
                "audio_sha256": sha256_file(audio_path),
                "usage": _usage_dict(completion),
            }
        )
    except Exception as exc:
        error_type, error_detail = _safe_error(exc)
        record.update({"error_type": error_type, "error_detail": error_detail})
    record["latency_ms"] = round((time.monotonic() - started) * 1000)
    analysis = analyze_wav(audio_path)
    transcript_exact = bool(
        (record.get("transcript_check") or {}).get("exact_token_match")
    )
    record["analysis"] = analysis
    record["exclusion_reasons"] = exclusion_reasons(
        status=record["status"],
        transcript_exact=transcript_exact,
        analysis=analysis,
    )
    record["estimated_cost_usd"] = estimated_cost_usd(record["usage"])
    return record


def _point(record: dict[str, Any]) -> np.ndarray | None:
    if record.get("exclusion_reasons"):
        return None
    analysis = record.get("analysis") or {}
    f1 = analysis.get("f1_bark")
    f2 = analysis.get("f2_bark")
    if not isinstance(f1, (int, float)) or not isinstance(f2, (int, float)):
        return None
    return np.array([f1, f2], dtype=np.float64)


def _centroid(points: Sequence[np.ndarray]) -> np.ndarray:
    return np.mean(np.stack(points), axis=0)


def _rms_to_centroid(points: Sequence[np.ndarray], centroid: np.ndarray) -> float:
    distances = [np.dot(point - centroid, point - centroid) for point in points]
    return math.sqrt(float(np.mean(distances)))


def _magnitude(vector: np.ndarray) -> float:
    return float(np.linalg.norm(vector))


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = _magnitude(left) * _magnitude(right)
    return float(np.dot(left, right) / denominator) if denominator > 0 else -1.0


def _serialized_point(point: np.ndarray) -> list[float]:
    return [round(float(point[0]), 6), round(float(point[1]), 6)]


def classify_calibration(
    records: Sequence[dict[str, Any]],
    *,
    rule_specs: Sequence[dict[str, Any]] = RULE_SPECS,
) -> dict[str, Any]:
    references: dict[str, list[np.ndarray]] = {}
    contrasts: dict[tuple[str, str, str], list[np.ndarray]] = {}
    for record in records:
        point = _point(record)
        if point is None:
            continue
        stimulus = record["stimulus"]
        if stimulus["kind"] == "reference":
            references.setdefault(stimulus["reference_category"], []).append(point)
        else:
            key = (stimulus["rule_id"], stimulus["shell"], stimulus["side"])
            contrasts.setdefault(key, []).append(point)

    rule_results: dict[str, Any] = {}
    for rule in rule_specs:
        rule_id = rule["rule_id"]
        source_points = references.get(rule["source_category"], [])
        target_points = references.get(rule["target_category"], [])
        anchor: dict[str, Any] = {
            "source_take_count": len(source_points),
            "target_take_count": len(target_points),
            "passed": False,
        }
        anchor_vector: np.ndarray | None = None
        source_anchor: np.ndarray | None = None
        target_anchor: np.ndarray | None = None
        if len(source_points) == 2 and len(target_points) == 2:
            source_anchor = _centroid(source_points)
            target_anchor = _centroid(target_points)
            anchor_vector = target_anchor - source_anchor
            reference_noise = max(
                _rms_to_centroid(source_points, source_anchor),
                _rms_to_centroid(target_points, target_anchor),
            )
            magnitude = _magnitude(anchor_vector)
            threshold = max(0.25, 2 * reference_noise)
            cross_cosines = [
                _cosine(target - source, anchor_vector)
                for source in source_points
                for target in target_points
            ]
            anchor.update(
                {
                    "source_centroid_bark": _serialized_point(source_anchor),
                    "target_centroid_bark": _serialized_point(target_anchor),
                    "vector_bark": _serialized_point(anchor_vector),
                    "magnitude_bark": round(magnitude, 6),
                    "reference_noise_bark": round(reference_noise, 6),
                    "magnitude_threshold_bark": round(threshold, 6),
                    "cross_take_cosines": [round(value, 6) for value in cross_cosines],
                    "passed": magnitude > threshold and min(cross_cosines) >= 0.50,
                }
            )
        else:
            anchor["failure_reason"] = "requires_two_non_excluded_takes_per_anchor"

        shell_results: list[dict[str, Any]] = []
        shell_vectors: list[np.ndarray] = []
        shell_neutral_centroids: list[np.ndarray] = []
        shell_lens_centroids: list[np.ndarray] = []
        shell_variances: list[float] = []
        for shell, _neutral_token, _lens_token in rule["tokens"]:
            neutral = contrasts.get((rule_id, shell, "neutral"), [])
            lens = contrasts.get((rule_id, shell, "lens"), [])
            shell_result: dict[str, Any] = {
                "shell": shell,
                "neutral_take_count": len(neutral),
                "lens_take_count": len(lens),
                "directional_pass": False,
                "exact_proximity_pass": False,
            }
            if len(neutral) >= 2 and len(lens) >= 2:
                neutral_centroid = _centroid(neutral)
                lens_centroid = _centroid(lens)
                vector = lens_centroid - neutral_centroid
                take_variance = max(
                    _rms_to_centroid(neutral, neutral_centroid),
                    _rms_to_centroid(lens, lens_centroid),
                )
                threshold = max(0.15, 1.5 * take_variance)
                magnitude = _magnitude(vector)
                direction_cosine = (
                    _cosine(vector, anchor_vector)
                    if anchor_vector is not None
                    else -1.0
                )
                directional_pass = bool(
                    anchor["passed"]
                    and magnitude > threshold
                    and direction_cosine >= 0.50
                )
                exact_proximity = bool(
                    source_anchor is not None
                    and target_anchor is not None
                    and _magnitude(neutral_centroid - source_anchor)
                    < _magnitude(neutral_centroid - target_anchor)
                    and _magnitude(lens_centroid - target_anchor)
                    < _magnitude(lens_centroid - source_anchor)
                )
                shell_result.update(
                    {
                        "neutral_centroid_bark": _serialized_point(neutral_centroid),
                        "lens_centroid_bark": _serialized_point(lens_centroid),
                        "vector_bark": _serialized_point(vector),
                        "magnitude_bark": round(magnitude, 6),
                        "take_variance_bark": round(take_variance, 6),
                        "magnitude_threshold_bark": round(threshold, 6),
                        "anchor_direction_cosine": round(direction_cosine, 6),
                        "directional_pass": directional_pass,
                        "exact_proximity_pass": exact_proximity,
                    }
                )
                shell_vectors.append(vector)
                shell_neutral_centroids.append(neutral_centroid)
                shell_lens_centroids.append(lens_centroid)
                shell_variances.append(take_variance)
            else:
                shell_result["failure_reason"] = (
                    "requires_two_non_excluded_takes_per_side"
                )
            shell_results.append(shell_result)

        aggregate: dict[str, Any] = {"passed": False, "exact_proximity_pass": False}
        if len(shell_vectors) == 3:
            aggregate_vector = _centroid(shell_vectors)
            aggregate_neutral = _centroid(shell_neutral_centroids)
            aggregate_lens = _centroid(shell_lens_centroids)
            aggregate_variance = max(shell_variances)
            threshold = max(0.15, 1.5 * aggregate_variance)
            magnitude = _magnitude(aggregate_vector)
            direction_cosine = (
                _cosine(aggregate_vector, anchor_vector)
                if anchor_vector is not None
                else -1.0
            )
            exact_proximity = bool(
                source_anchor is not None
                and target_anchor is not None
                and _magnitude(aggregate_neutral - source_anchor)
                < _magnitude(aggregate_neutral - target_anchor)
                and _magnitude(aggregate_lens - target_anchor)
                < _magnitude(aggregate_lens - source_anchor)
            )
            aggregate.update(
                {
                    "vector_bark": _serialized_point(aggregate_vector),
                    "neutral_centroid_bark": _serialized_point(aggregate_neutral),
                    "lens_centroid_bark": _serialized_point(aggregate_lens),
                    "magnitude_bark": round(magnitude, 6),
                    "take_variance_bark": round(aggregate_variance, 6),
                    "magnitude_threshold_bark": round(threshold, 6),
                    "anchor_direction_cosine": round(direction_cosine, 6),
                    "passed": bool(
                        anchor["passed"]
                        and magnitude > threshold
                        and direction_cosine >= 0.50
                    ),
                    "exact_proximity_pass": exact_proximity,
                }
            )
        else:
            aggregate["failure_reason"] = "requires_all_three_measurable_shell_vectors"

        directional_shells = sum(item["directional_pass"] for item in shell_results)
        exact_shells = sum(item["exact_proximity_pass"] for item in shell_results)
        realization_pass = bool(
            anchor["passed"] and directional_shells >= 2 and aggregate["passed"]
        )
        if realization_pass and exact_shells >= 2 and aggregate["exact_proximity_pass"]:
            outcome = "exact-category pass"
        elif realization_pass:
            outcome = "directional-only pass"
        else:
            outcome = "fail"
        rule_results[rule_id] = {
            "source_ipa": rule["source_ipa"],
            "target_ipa": rule["target_ipa"],
            "anchor_sanity": anchor,
            "shells": shell_results,
            "aggregate": aggregate,
            "directional_shell_pass_count": directional_shells,
            "exact_proximity_shell_count": exact_shells,
            "contrast_realization_pass": realization_pass,
            "outcome": outcome,
        }
    return {
        "schema_version": 1,
        "classification_protocol": "carrier-v3-preregistered-formant-gates-v1",
        "rules": rule_results,
        "all_rules_pass_directionally": all(
            result["outcome"] in {"exact-category pass", "directional-only pass"}
            for result in rule_results.values()
        ),
    }


def summarize_usage(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    totals = {
        "prompt_tokens": 0,
        "prompt_audio_tokens": 0,
        "prompt_text_tokens": 0,
        "completion_tokens": 0,
        "completion_audio_tokens": 0,
        "completion_text_tokens": 0,
        "reasoning_tokens": 0,
    }
    for record in records:
        tokens = _token_usage(record.get("usage") or {})
        for key in totals:
            totals[key] += tokens[key]
    estimated = (
        totals["prompt_text_tokens"] * TEXT_INPUT_USD_PER_MILLION
        + totals["prompt_audio_tokens"] * AUDIO_INPUT_USD_PER_MILLION
        + totals["completion_text_tokens"] * TEXT_OUTPUT_USD_PER_MILLION
        + totals["completion_audio_tokens"] * AUDIO_OUTPUT_USD_PER_MILLION
    ) / 1_000_000
    return {
        **totals,
        "usage_records": sum(bool(record.get("usage")) for record in records),
        "estimated_cost_usd": round(estimated, 6),
        "price_source": PRICE_SOURCE,
        "price_capture_date": PRICE_CAPTURE_DATE,
        "prices_usd_per_million_tokens": {
            "text_input": TEXT_INPUT_USD_PER_MILLION,
            "text_output": TEXT_OUTPUT_USD_PER_MILLION,
            "audio_input": AUDIO_INPUT_USD_PER_MILLION,
            "audio_output": AUDIO_OUTPUT_USD_PER_MILLION,
        },
    }


def _flatten_record(record: dict[str, Any]) -> dict[str, Any]:
    stimulus = record["stimulus"]
    analysis = record.get("analysis") or {}
    transcript = record.get("transcript_check") or {}
    usage = _token_usage(record.get("usage") or {})
    return {
        "request_order": record["request_order"],
        **stimulus,
        "status": record.get("status", "failed"),
        "request_id": record.get("request_id", ""),
        "resolved_model": record.get("resolved_model", ""),
        "latency_ms": record.get("latency_ms", ""),
        "provider_transcript": record.get("provider_transcript", ""),
        "exact_token_match": transcript.get("exact_token_match", False),
        "audio_filename": record.get("audio_filename", ""),
        "audio_sha256": record.get("audio_sha256", ""),
        **analysis,
        "exclusion_reasons_json": json.dumps(
            record.get("exclusion_reasons", []), separators=(",", ":")
        ),
        **usage,
        "estimated_request_cost_usd": record.get("estimated_cost_usd", 0),
        "error_type": record.get("error_type", ""),
        "error_detail": record.get("error_detail", ""),
    }


def _record_path(
    run_dir: Path, request_order: int, stimulus: CalibrationStimulus
) -> Path:
    return run_dir / "slots" / f"{request_order:03d}__{stimulus.slot_id}.json"


def _recover_interrupted(
    path: Path, *, request_order: int, stimulus: CalibrationStimulus, audio_path: Path
) -> dict[str, Any]:
    analysis = analyze_wav(audio_path)
    record = {
        "request_order": request_order,
        "stimulus": asdict(stimulus),
        "status": "failed",
        "model": MODEL,
        "voice": VOICE,
        "format": FORMAT,
        "usage": {},
        "error_type": "InterruptedRequestState",
        "error_detail": (
            "A started receipt existed without a completed response receipt; the slot "
            "was not retried because replacement or duplicate requests are forbidden."
        ),
        "analysis": analysis,
    }
    record["exclusion_reasons"] = exclusion_reasons(
        status="failed", transcript_exact=False, analysis=analysis
    )
    record["estimated_cost_usd"] = 0.0
    atomic_write_json(path, record)
    return record


def run_acoustic_calibration(
    run_id: str, *, client: Any | None = None
) -> dict[str, Any]:
    from .api import require_api_key

    if PREREGISTRATION_HEADING not in DEVLOG_PATH.read_text(encoding="utf-8"):
        raise RuntimeError("The frozen acoustic-calibration preregistration is missing")
    if client is None:
        require_api_key()
        client = OpenAI(max_retries=0)

    paths = Paths()
    paths.run_dir(run_id)  # Reuse the repository's strict run-id validation.
    run_dir = paths.artifacts / "acoustic-calibration" / run_id
    protocol = protocol_record()
    manifest_path = run_dir / "manifest.json"
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing != protocol:
            raise RuntimeError(
                "Existing calibration manifest does not match the frozen protocol"
            )
    else:
        if run_dir.exists() and any(run_dir.iterdir()):
            raise RuntimeError(
                "Calibration directory exists without its frozen manifest"
            )
        atomic_write_json(manifest_path, protocol)

    records: list[dict[str, Any]] = []
    manifest = build_manifest()
    for request_order, stimulus in enumerate(manifest, start=1):
        record_path = _record_path(run_dir, request_order, stimulus)
        audio_path = run_dir / "audio" / f"{request_order:03d}__{stimulus.slot_id}.wav"
        if record_path.is_file():
            record = json.loads(record_path.read_text(encoding="utf-8"))
            if record.get("status") == "started":
                record = _recover_interrupted(
                    record_path,
                    request_order=request_order,
                    stimulus=stimulus,
                    audio_path=audio_path,
                )
            records.append(record)
            continue
        atomic_write_json(
            record_path,
            {
                "request_order": request_order,
                "stimulus": asdict(stimulus),
                "status": "started",
            },
        )
        record = _render_slot(
            client=client,
            stimulus=stimulus,
            request_order=request_order,
            audio_path=audio_path,
        )
        atomic_write_json(record_path, record)
        records.append(record)
        print(
            f"calibration {request_order:02d}/66 {stimulus.slot_id}: "
            f"{record['status']} ({len(record['exclusion_reasons'])} exclusions)",
            flush=True,
        )

    if len(records) != 66:
        raise AssertionError("Calibration must retain one record for every frozen slot")
    rows = [_flatten_record(record) for record in records]
    write_csv(run_dir / "results.csv", rows, RESULT_FIELDS)
    classification = classify_calibration(records)
    atomic_write_json(run_dir / "analysis.json", classification)
    usage = summarize_usage(records)
    summary = {
        "schema_version": 1,
        "run_id": run_id,
        "protocol_sha256": protocol["protocol_sha256"],
        "logical_request_slots": 66,
        "completed_records": len(records),
        "successful_requests": sum(record.get("status") == "ok" for record in records),
        "exact_transcripts": sum(
            bool((record.get("transcript_check") or {}).get("exact_token_match"))
            for record in records
        ),
        "non_excluded_takes": sum(
            not record.get("exclusion_reasons") for record in records
        ),
        "excluded_takes": sum(
            bool(record.get("exclusion_reasons")) for record in records
        ),
        "outcomes": {
            rule_id: result["outcome"]
            for rule_id, result in classification["rules"].items()
        },
        "all_rules_pass_directionally": classification["all_rules_pass_directionally"],
        "usage": usage,
        "cost_estimate_scope": (
            "API usage returned in completed response receipts; an interrupted "
            "request without a response receipt would remain cost-unknown."
        ),
        "manifest": str(manifest_path),
        "results_csv": str(run_dir / "results.csv"),
        "analysis_json": str(run_dir / "analysis.json"),
    }
    atomic_write_json(run_dir / "summary.json", summary)
    return summary
