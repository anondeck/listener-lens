from __future__ import annotations

import csv
import gc
import hashlib
import json
import math
import re
import subprocess
import tempfile
import wave
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import mlx_whisper
import mlx.core as mx
import numpy as np

from .config import DEVLOG_PATH, Paths, stable_json
from .util import atomic_write_json, sha256_file


RUN_ID = "20260715-same-take-v1"
PREREGISTRATION_HEADING = "## Same-take-v1 audio-domain preregistration — July 15, 2026"
PRAAT = Path("/Applications/Praat.app/Contents/MacOS/Praat")
PRAAT_SHA256 = "b0311abb9ae606a5204715b0ab861d4ab863b932cbe3f0faecb7cce80b609d8d"
PROBE_SCRIPT = Paths().root / "scripts" / "praat_same_take_probe.praat"
FORMANTPATH_SCRIPT = Paths().root / "scripts" / "praat_same_take_formantpath.praat"
WHISPER_MODEL = Paths().cache / "whisper" / "large-v3-full"
MAX_FORMANTS_FAMILY = (5.0, 5.5, 6.0)
MIN_CORE_S = 0.060
MAX_CORE_S = 0.100


@dataclass(frozen=True)
class AnchorSpec:
    category: str
    ipa: str
    word: str
    take: int
    path: Path
    sha256: str


@dataclass(frozen=True)
class SourceCandidate:
    rule_id: str
    take_index: int
    path: Path
    sha256: str
    script: str
    target_word: str
    target_indices: tuple[int, ...]
    search_fraction: tuple[float, float]


ANCHOR_ROOT = Paths().artifacts / "acoustic-calibration" / "20260715-carrier-v3-calibration" / "audio"
AE_ROOT = Paths().artifacts / "matched-pairs" / "20260715-curated-ptbr-ae-matched-pair" / "audio"
IH_ROOT = Paths().artifacts / "audio-prosody" / "20260714-flow-v2" / "audio"

ANCHORS = (
    AnchorSpec("ae", "æ", "bat", 1, ANCHOR_ROOT / "004__reference__ae__take-1.wav", "91d802aa2d7423d68ea21fa2e6f1a09ffd8167945abacb12d26ebffd4fe9063f"),
    AnchorSpec("ae", "æ", "bat", 2, ANCHOR_ROOT / "008__reference__ae__take-2.wav", "523e3e6169c0842470e08ab1dd272dd1f609224caaa3dc16fbdc550449a2d31f"),
    AnchorSpec("eh", "ɛ", "bet", 1, ANCHOR_ROOT / "057__reference__eh__take-1.wav", "37c5191922870e47141f4eb0509127928c9fb0b44c14ef743be4f0d1e2fa64c2"),
    AnchorSpec("eh", "ɛ", "bet", 2, ANCHOR_ROOT / "062__reference__eh__take-2.wav", "5759e14c40521e55d9f17edaa5cc2c858587daa33112753e5df2ecdfb3f0bf56"),
    AnchorSpec("ih", "ɪ", "bit", 1, ANCHOR_ROOT / "032__reference__ih__take-1.wav", "1a2c0f656fc95ed6ed4d6cbb4d426b5a55f809221bdb2141c80d1722631ad9b8"),
    AnchorSpec("ih", "ɪ", "bit", 2, ANCHOR_ROOT / "040__reference__ih__take-2.wav", "09541bf7263390205427e0794e4e02336e514ad31a0aa998c54430ce9ca532a1"),
    AnchorSpec("i", "i", "beat", 1, ANCHOR_ROOT / "059__reference__i__take-1.wav", "28add35a7ac4cd52789350422ccb4a6be1f88cdb85464a60d09c89e7150fcdd7"),
    AnchorSpec("i", "i", "beat", 2, ANCHOR_ROOT / "035__reference__i__take-2.wav", "5222e836ddb74c0f6d2866c1e2355c44c9f07aa4f682216c3762fd78c872ffad"),
)

AE_SCRIPT = "frohr bavd bavd bavd lohm frohr tadril prohk."
AE_CANDIDATES = (
    SourceCandidate("ptbr.vowel.ae_to_eh", 4, AE_ROOT / "008__neutral__take-4.wav", "1052138ac9e9829e28089718bd856a3dd72c63f176241a716ebc743b22613f54", AE_SCRIPT, "bavd", (1, 2, 3), (0.15, 0.60)),
    SourceCandidate("ptbr.vowel.ae_to_eh", 1, AE_ROOT / "003__neutral__take-1.wav", "c0266afe7cb519ab060b98150f1bb3cdad5f5f0aa87c47fa12b41251e0f3ce14", AE_SCRIPT, "bavd", (1, 2, 3), (0.15, 0.60)),
    SourceCandidate("ptbr.vowel.ae_to_eh", 2, AE_ROOT / "006__neutral__take-2.wav", "5e0570ee75169955285913b783d9aacaf3c5eaba9837b54f72db231a0caba870", AE_SCRIPT, "bavd", (1, 2, 3), (0.15, 0.60)),
    SourceCandidate("ptbr.vowel.ae_to_eh", 3, AE_ROOT / "002__neutral__take-3.wav", "38c7927426a53f860968b4f4960e47bd5742f5ba0742b526ea02bee800e5e65c", AE_SCRIPT, "bavd", (1, 2, 3), (0.15, 0.60)),
)

IH_SCRIPT = "nushvot kaezmor plimzang dovkrish faempud glornik wuftesh traezbin skootvash, nempool zhaevrik plimzang draskoop moltven glornik peftash voongrik skootvash."
IH_CANDIDATES = (
    SourceCandidate("ptbr.vowel.ih_to_i", 1, IH_ROOT / "4d5949bf82.wav", "0c61c7b1605eabc089f7e77cc9f35128fa3b16a8f1fd3eec2f246cae623642b4", IH_SCRIPT, "plimzang", (2, 11), (0.15, 0.45)),
    SourceCandidate("ptbr.vowel.ih_to_i", 2, IH_ROOT / "07c1f64e07.wav", "fab8e62db2a314fe4982933608e71911b81bcc0fee092cc5300645aa0e8e2179", IH_SCRIPT, "plimzang", (2, 11), (0.15, 0.45)),
    SourceCandidate("ptbr.vowel.ih_to_i", 3, IH_ROOT / "28742fe8ab.wav", "5a13f8bdd0a54cb2c9633f133359dfa271da7e345ae25e52b39c20f8e549dab6", IH_SCRIPT, "plimzang", (2, 11), (0.15, 0.45)),
)

RULE_ENDPOINTS = {
    "ptbr.vowel.ae_to_eh": ("ae", "eh"),
    "ptbr.vowel.ih_to_i": ("ih", "i"),
}


def bark(hz: float) -> float:
    return 26.81 / (1 + 1960 / hz) - 0.53


def _normalize_token(value: str) -> str:
    return re.sub(r"[^a-z']+", "", value.casefold())


def _wav_info(path: Path) -> dict[str, int | float]:
    with wave.open(str(path), "rb") as handle:
        return {
            "channels": handle.getnchannels(),
            "sample_width": handle.getsampwidth(),
            "sample_rate_hz": handle.getframerate(),
            "sample_count": handle.getnframes(),
            "duration_s": handle.getnframes() / handle.getframerate(),
        }


def _verify_inputs() -> None:
    if PREREGISTRATION_HEADING not in DEVLOG_PATH.read_text(encoding="utf-8"):
        raise RuntimeError("same-take-v1 preregistration is missing")
    if not PRAAT.is_file() or sha256_file(PRAAT) != PRAAT_SHA256:
        raise RuntimeError("standalone Praat executable does not match the preregistered hash")
    if not WHISPER_MODEL.is_dir():
        raise RuntimeError("pinned local Whisper model is unavailable")
    for item in (*ANCHORS, *AE_CANDIDATES, *IH_CANDIDATES):
        if not item.path.is_file() or sha256_file(item.path) != item.sha256:
            raise RuntimeError(f"frozen input mismatch: {item.path}")


def _word_timestamps(
    path: Path, script: str, *, allow_singleton_anchor_label: bool = False
) -> list[dict[str, Any]]:
    result = mlx_whisper.transcribe(
        str(path),
        path_or_hf_repo=str(WHISPER_MODEL),
        language="en",
        temperature=0,
        condition_on_previous_text=False,
        word_timestamps=True,
        initial_prompt=script,
        verbose=False,
    )
    words = [word for segment in result.get("segments", []) for word in segment.get("words", [])]
    expected = [_normalize_token(token) for token in script.split()]
    actual = [_normalize_token(str(word.get("word", ""))) for word in words]
    singleton_anchor = (
        allow_singleton_anchor_label
        and len(expected) == 1
        and len(actual) == 1
    )
    if actual != expected and not singleton_anchor:
        raise RuntimeError(f"word timestamp token mismatch for {path.name}: {actual!r} != {expected!r}")
    timestamps = [
        {
            "token": actual[index],
            "start_s": float(word["start"]),
            "end_s": float(word["end"]),
            "probability": float(word.get("probability") or 0),
        }
        for index, word in enumerate(words)
    ]
    del result, words
    mx.clear_cache()
    gc.collect()
    return timestamps


def _parse_number(value: str) -> float | None:
    try:
        number = float(value)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def _probe_frames(path: Path) -> list[dict[str, float | None]]:
    with tempfile.TemporaryDirectory(prefix="same-take-probe-") as temp:
        output = Path(temp) / "probe.tsv"
        subprocess.run(
            [str(PRAAT), "--run", str(PROBE_SCRIPT), str(path), str(output)],
            check=True,
            capture_output=True,
            text=True,
        )
        with output.open(encoding="utf-8", newline="") as handle:
            return [
                {key: _parse_number(value) for key, value in row.items()}
                for row in csv.DictReader(handle, delimiter="\t")
            ]


def _eligible_frame(frame: dict[str, float | None]) -> bool:
    f1, f2, f3, f4, pitch = (frame.get(name) for name in ("f1_hz", "f2_hz", "f3_hz", "f4_hz", "pitch_hz"))
    return bool(
        f1 is not None and 150 <= f1 <= 1300
        and f2 is not None and 500 <= f2 <= 4000
        and f3 is not None and 1000 <= f3 <= 5500
        and f4 is not None and 1500 <= f4 <= 5500
        and pitch is not None and 75 <= pitch <= 500
    )


def align_vowel_core(
    frames: Sequence[dict[str, float | None]],
    *,
    word_start_s: float,
    word_end_s: float,
    search_fraction: tuple[float, float],
    sample_rate_hz: int,
) -> dict[str, Any]:
    duration = word_end_s - word_start_s
    search_start = word_start_s + duration * search_fraction[0]
    search_end = word_start_s + duration * search_fraction[1]
    indices = [
        index for index, frame in enumerate(frames)
        if frame["time_s"] is not None
        and search_start <= float(frame["time_s"]) <= search_end
        and _eligible_frame(frame)
    ]
    if not indices:
        raise RuntimeError("no formant-valid voiced frame in preregistered search band")
    peak = max(indices, key=lambda index: float(frames[index].get("rms") or 0))
    eligible = set(indices)
    left = peak
    right = peak
    while left - 1 in eligible:
        left -= 1
    while right + 1 in eligible:
        right += 1
    available_start = float(frames[left]["time_s"]) - 0.0025
    available_end = float(frames[right]["time_s"]) + 0.0025
    if available_end - available_start < MIN_CORE_S:
        raise RuntimeError("voiced/formant-valid run is shorter than 60 ms")
    peak_time = float(frames[peak]["time_s"])
    core_duration = min(MAX_CORE_S, available_end - available_start)
    start = max(available_start, min(peak_time - core_duration / 2, available_end - core_duration))
    end = start + core_duration
    start_sample = round(start * sample_rate_hz)
    end_sample = round(end * sample_rate_hz)
    if end_sample - start_sample < round(MIN_CORE_S * sample_rate_hz):
        raise RuntimeError("sample-snapped core is shorter than 60 ms")
    return {
        "start_s": start_sample / sample_rate_hz,
        "end_s": end_sample / sample_rate_hz,
        "start_sample": start_sample,
        "end_sample_exclusive": end_sample,
        "sample_count": end_sample - start_sample,
        "search_start_s": round(search_start, 6),
        "search_end_s": round(search_end, 6),
        "peak_frame_s": round(peak_time, 6),
        "available_voiced_start_s": round(available_start, 6),
        "available_voiced_end_s": round(available_end, 6),
    }


def measure_formantpath(path: Path, interval: dict[str, Any], maximum_formants: float) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="same-take-formantpath-") as temp:
        output = Path(temp) / "path.tsv"
        subprocess.run(
            [
                str(PRAAT), "--run", str(FORMANTPATH_SCRIPT), str(path), str(output),
                f"{interval['start_s']:.9f}", f"{interval['end_s']:.9f}", f"{maximum_formants:.1f}",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        with output.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
    middle_start = interval["start_s"] + (interval["end_s"] - interval["start_s"]) * 0.25
    middle_end = interval["start_s"] + (interval["end_s"] - interval["start_s"]) * 0.75
    values: dict[str, list[float]] = {name: [] for name in ("F1(Hz)", "F2(Hz)", "F3(Hz)", "F4(Hz)", "Ceiling(Hz)", "Stress")}
    frame_count = 0
    for row in rows:
        time = _parse_number(row.get("time(s)", ""))
        if time is None or not middle_start <= time <= middle_end:
            continue
        frame_count += 1
        for name in values:
            value = _parse_number(row.get(name, ""))
            if value is not None:
                values[name].append(value)
    valid = min(len(values["F1(Hz)"]), len(values["F2(Hz)"]))
    if frame_count < 5 or valid / frame_count < 0.60:
        raise RuntimeError("FormantPath interval failed the frozen frame-retention gate")
    result = {
        "maximum_formants": maximum_formants,
        "middle_frame_count": frame_count,
        "valid_f1_f2_frame_count": valid,
        "valid_f1_f2_fraction": valid / frame_count,
    }
    for source, target in (("F1(Hz)", "f1_hz"), ("F2(Hz)", "f2_hz"), ("F3(Hz)", "f3_hz"), ("F4(Hz)", "f4_hz"), ("Ceiling(Hz)", "optimal_ceiling_hz"), ("Stress", "stress")):
        result[target] = float(np.median(values[source])) if values[source] else None
    result["f1_bark"] = bark(float(result["f1_hz"]))
    result["f2_bark"] = bark(float(result["f2_hz"]))
    return result


def _centroid(points: Sequence[np.ndarray]) -> np.ndarray:
    return np.mean(np.stack(points), axis=0)


def _rms_to_centroid(points: Sequence[np.ndarray], centroid: np.ndarray) -> float:
    return math.sqrt(float(np.mean([np.dot(point - centroid, point - centroid) for point in points])))


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator > 0 else -1.0


def _anchor_gates(anchor_records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by_category: dict[str, list[dict[str, Any]]] = {}
    for record in anchor_records:
        by_category.setdefault(record["category"], []).append(record)
    results: dict[str, Any] = {}
    for rule_id, (source_category, target_category) in RULE_ENDPOINTS.items():
        families: dict[str, Any] = {}
        for maximum_formants in MAX_FORMANTS_FAMILY:
            key = f"{maximum_formants:.1f}"
            source = [np.array([item["measurements"][key]["f1_bark"], item["measurements"][key]["f2_bark"]]) for item in by_category[source_category]]
            target = [np.array([item["measurements"][key]["f1_bark"], item["measurements"][key]["f2_bark"]]) for item in by_category[target_category]]
            if len(source) != 2 or len(target) != 2:
                raise RuntimeError("same-take-v1 requires exactly two designated takes per endpoint")
            source_centroid = _centroid(source)
            target_centroid = _centroid(target)
            vector = target_centroid - source_centroid
            variance = max(_rms_to_centroid(source, source_centroid), _rms_to_centroid(target, target_centroid))
            magnitude = float(np.linalg.norm(vector))
            threshold = max(0.25, 2 * variance)
            cross = [_cosine(t - s, vector) for s in source for t in target]
            families[key] = {
                "source_centroid_bark": source_centroid.tolist(),
                "target_centroid_bark": target_centroid.tolist(),
                "anchor_vector_bark": vector.tolist(),
                "endpoint_take_variance_bark": variance,
                "magnitude_bark": magnitude,
                "magnitude_threshold_bark": threshold,
                "cross_take_cosines": cross,
                "passed": magnitude > threshold and min(cross) >= 0.50,
            }
        results[rule_id] = {"families": families, "passed_all_families": all(item["passed"] for item in families.values())}
    return results


def _align_and_measure(
    path: Path,
    script: str,
    target_indices: Iterable[int],
    search_fraction: tuple[float, float],
    *,
    allow_singleton_anchor_label: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    info = _wav_info(path)
    words = _word_timestamps(
        path, script, allow_singleton_anchor_label=allow_singleton_anchor_label
    )
    frames = _probe_frames(path)
    slots = []
    for occurrence, index in enumerate(target_indices, start=1):
        word = words[index]
        core = align_vowel_core(
            frames,
            word_start_s=word["start_s"],
            word_end_s=word["end_s"],
            search_fraction=search_fraction,
            sample_rate_hz=int(info["sample_rate_hz"]),
        )
        measurements = {f"{maximum_formants:.1f}": measure_formantpath(path, core, maximum_formants) for maximum_formants in MAX_FORMANTS_FAMILY}
        slots.append({"occurrence": occurrence, "word_index": index, "word": word, "interval": core, "measurements": measurements})
    return words, slots


def build_source_freeze(run_id: str = RUN_ID) -> dict[str, Any]:
    if run_id != RUN_ID:
        raise RuntimeError("same-take-v1 source-freeze run ID is fixed")
    _verify_inputs()
    output_dir = Paths().artifacts / "same-take" / run_id
    output_path = output_dir / "source-freeze.json"
    if output_path.exists():
        raise RuntimeError("source-freeze already exists and is immutable")

    anchor_records = []
    for anchor in ANCHORS:
        words, slots = _align_and_measure(
            anchor.path,
            anchor.word,
            (0,),
            (0.10, 0.90),
            allow_singleton_anchor_label=True,
        )
        anchor_records.append({
            "category": anchor.category,
            "ipa": anchor.ipa,
            "take": anchor.take,
            "path": str(anchor.path),
            "sha256": anchor.sha256,
            "wav": _wav_info(anchor.path),
            "word_timestamps": words,
            "interval": slots[0]["interval"],
            "measurements": slots[0]["measurements"],
        })
    gates = _anchor_gates(anchor_records)

    rule_records: dict[str, Any] = {}
    for rule_id, candidates in (("ptbr.vowel.ae_to_eh", AE_CANDIDATES), ("ptbr.vowel.ih_to_i", IH_CANDIDATES)):
        source_gate = gates[rule_id]
        candidate_records = []
        selected = None
        for candidate in candidates:
            record: dict[str, Any] = {
                "take_index": candidate.take_index,
                "path": str(candidate.path),
                "sha256": candidate.sha256,
                "wav": _wav_info(candidate.path),
                "status": "screened",
                "all_slots_source_category": False,
            }
            try:
                words, slots = _align_and_measure(candidate.path, candidate.script, candidate.target_indices, candidate.search_fraction)
                record["word_timestamps"] = words
                record["slots"] = slots
                slot_passes = []
                for slot in slots:
                    family_results = {}
                    for key, gate in source_gate["families"].items():
                        point = np.array([slot["measurements"][key]["f1_bark"], slot["measurements"][key]["f2_bark"]])
                        source_distance = float(np.linalg.norm(point - np.array(gate["source_centroid_bark"])))
                        target_distance = float(np.linalg.norm(point - np.array(gate["target_centroid_bark"])))
                        family_results[key] = {"source_distance_bark": source_distance, "target_distance_bark": target_distance, "passed": gate["passed"] and source_distance < target_distance}
                    slot["source_category_gate"] = family_results
                    slot["source_category_pass"] = all(item["passed"] for item in family_results.values())
                    slot_passes.append(slot["source_category_pass"])
                record["all_slots_source_category"] = all(slot_passes)
            except Exception as exc:
                record.update({"status": "excluded", "error_type": type(exc).__name__, "error_detail": str(exc)})
            candidate_records.append(record)
            if record["all_slots_source_category"]:
                selected = record
                break
        if selected is not None:
            different = min((item for item in candidates if item.take_index != selected["take_index"]), key=lambda item: item.take_index)
            selected_summary = {
                "take_index": selected["take_index"],
                "path": selected["path"],
                "sha256": selected["sha256"],
                "slots": selected["slots"],
                "different_neutral_reference": {"take_index": different.take_index, "path": str(different.path), "sha256": different.sha256},
            }
        else:
            selected_summary = None
        rule_records[rule_id] = {"candidates": candidate_records, "selected_source": selected_summary, "running_sentence_spike_status": "eligible" if selected_summary else "stopped_no_gate_passing_source"}

    protocol = {
        "protocol": "same-take-v1-source-freeze",
        "preregistration_commit": "89164bf",
        "praat_sha256": PRAAT_SHA256,
        "formantpath_maximum_formants_family": list(MAX_FORMANTS_FAMILY),
        "api_calls": 0,
        "api_cost_usd": 0.0,
    }
    protocol["protocol_sha256"] = hashlib.sha256(stable_json(protocol).encode()).hexdigest()
    receipt = {
        "schema_version": 1,
        "status": "source_hashes_alignments_and_gates_frozen_before_editing",
        "run_id": run_id,
        "protocol": protocol,
        "anchors": anchor_records,
        "anchor_gates": gates,
        "rules": rule_records,
        "no_audio_was_edited": True,
    }
    receipt["receipt_sha256"] = hashlib.sha256(stable_json(receipt).encode()).hexdigest()
    atomic_write_json(output_path, receipt)
    return receipt
