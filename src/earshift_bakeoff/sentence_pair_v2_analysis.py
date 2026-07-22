from __future__ import annotations

import csv
import gc
import json
import math
import subprocess
import tempfile
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Sequence

import mlx.core as mx
import mlx_whisper
import numpy as np

from .audio_conformance import AudioTiming, PauseInterval
from .config import Paths
from .matched_pairs import PairingThresholds, TakeCandidate, evaluate_pair
from .pcm import decode_pcm16_mono
from .same_take import WHISPER_MODEL, _probe_frames, align_vowel_core, bark
from .sentence_pair_v2 import (
    ANCHOR_GATE,
    CARRIERS,
    MODEL,
    MEASUREMENT_SCRIPT_SHA256,
    PRAAT_SHA256,
    RUN_ID,
    VOICE,
    prompt_contract_fingerprint,
    protocol_record,
)
from .sentence_pair_v2_run import EXPECTED_PROTOCOL_SHA256
from .util import atomic_write_json, sha256_file


PRAAT = Path("/Applications/Praat.app/Contents/MacOS/Praat")
MEASUREMENT_SCRIPT = Paths().root / "scripts" / "praat_sentence_pair_v2_burg.praat"
CEILINGS = (5500, 5750, 6000)


def _number(value: str) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _word_intervals(path: Path, script: str) -> list[dict[str, Any]]:
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
    words = [
        word
        for segment in result.get("segments", [])
        for word in segment.get("words", [])
        if str(word.get("word", "")).strip()
    ]
    intervals = [
        {
            "whisper_label": str(word.get("word", "")).strip(),
            "start_s": float(word["start"]),
            "end_s": float(word["end"]),
            "probability": float(word.get("probability") or 0),
        }
        for word in words
    ]
    del result, words
    mx.clear_cache()
    gc.collect()
    if len(intervals) != 5:
        raise RuntimeError(f"requires exactly five Whisper word intervals; got {len(intervals)}")
    if any(
        item["start_s"] < 0
        or item["end_s"] <= item["start_s"]
        or index and item["start_s"] < intervals[index - 1]["end_s"] - 1e-6
        for index, item in enumerate(intervals)
    ):
        raise RuntimeError("Whisper word intervals are not monotonic and non-overlapping")
    return intervals


def _measure(path: Path, interval: dict[str, Any], ceiling: int) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="sentence-pair-v2-burg-") as temp:
        output = Path(temp) / "frames.tsv"
        subprocess.run(
            [
                str(PRAAT),
                "--run",
                str(MEASUREMENT_SCRIPT),
                str(path),
                str(output),
                f"{interval['start_s']:.9f}",
                f"{interval['end_s']:.9f}",
                str(ceiling),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        with output.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
    middle_start = interval["start_s"] + (interval["end_s"] - interval["start_s"]) * 0.25
    middle_end = interval["start_s"] + (interval["end_s"] - interval["start_s"]) * 0.75
    frame_count = 0
    pairs: list[tuple[float, float]] = []
    for row in rows:
        time_s = _number(row.get("time_s", ""))
        if time_s is None or not middle_start <= time_s <= middle_end:
            continue
        frame_count += 1
        f1 = _number(row.get("f1_hz", ""))
        f2 = _number(row.get("f2_hz", ""))
        if f1 is not None and f2 is not None:
            pairs.append((f1, f2))
    fraction = len(pairs) / frame_count if frame_count else 0.0
    if frame_count < 5 or len(pairs) < 5 or fraction < 0.60:
        raise RuntimeError(
            f"frame retention failed at {ceiling} Hz: {len(pairs)}/{frame_count}"
        )
    f1_hz, f2_hz = np.median(np.asarray(pairs), axis=0)
    plausible = bool(180 <= f1_hz <= 1200 and 600 <= f2_hz <= 3500 and f2_hz - f1_hz >= 250)
    return {
        "ceiling_hz": ceiling,
        "middle_frame_count": frame_count,
        "valid_f1_f2_frame_count": len(pairs),
        "valid_f1_f2_fraction": fraction,
        "f1_hz": float(f1_hz),
        "f2_hz": float(f2_hz),
        "f1_bark": bark(float(f1_hz)),
        "f2_bark": bark(float(f2_hz)),
        "plausibility_pass": plausible,
    }


def _analyze_take(record: dict[str, Any]) -> dict[str, Any]:
    analysis: dict[str, Any] = {"status": "excluded", "exclusion_reasons": []}
    if record.get("status") != "audio_returned":
        analysis["exclusion_reasons"].append("no_returned_audio")
        return analysis
    if not record.get("integrity_ok"):
        analysis["exclusion_reasons"].append("wav_integrity")
    if not (record.get("transcript_check") or {}).get("exact_token_match"):
        analysis["exclusion_reasons"].append("provider_transcript")
    if analysis["exclusion_reasons"]:
        return analysis
    try:
        path = Path(record["audio_path"])
        decoded = decode_pcm16_mono(path)
        words = _word_intervals(path, record["slot"]["script"])
        target_word = words[2]
        frames = _probe_frames(path)
        core = align_vowel_core(
            frames,
            word_start_s=target_word["start_s"],
            word_end_s=target_word["end_s"],
            search_fraction=(0.10, 0.75),
            sample_rate_hz=decoded.sample_rate_hz,
        )
        measurements = {str(ceiling): _measure(path, core, ceiling) for ceiling in CEILINGS}
        analysis.update(
            {
                "status": "measurable",
                "word_intervals": words,
                "target_word_interval": target_word,
                "vowel_core": core,
                "measurements": measurements,
            }
        )
    except Exception as exc:
        analysis["exclusion_reasons"].append(type(exc).__name__ + ": " + str(exc)[:300])
    return analysis


def _centroid(points: Sequence[np.ndarray]) -> np.ndarray:
    return np.mean(np.stack(points), axis=0)


def _variance(points: Sequence[np.ndarray], centroid: np.ndarray) -> float:
    return math.sqrt(float(np.mean([np.dot(point - centroid, point - centroid) for point in points])))


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator else -1.0


def _classify(records: list[dict[str, Any]]) -> dict[str, Any]:
    carriers: dict[str, Any] = {}
    for carrier in CARRIERS:
        selected = [item for item in records if item["slot"]["carrier_id"] == carrier.carrier_id]
        measurable = [item for item in selected if item["analysis"]["status"] == "measurable"]
        family_results: dict[str, Any] = {}
        for ceiling in CEILINGS:
            key = str(ceiling)
            neutral = [
                np.array([item["analysis"]["measurements"][key]["f1_bark"], item["analysis"]["measurements"][key]["f2_bark"]])
                for item in measurable if item["slot"]["side"] == "neutral"
            ]
            lens = [
                np.array([item["analysis"]["measurements"][key]["f1_bark"], item["analysis"]["measurements"][key]["f2_bark"]])
                for item in measurable if item["slot"]["side"] == "lens"
            ]
            anchor = ANCHOR_GATE["families"][key]
            source = np.array(anchor["source_centroid_bark"])
            target = np.array(anchor["target_centroid_bark"])
            result: dict[str, Any] = {"passed": False, "neutral_take_count": len(neutral), "lens_take_count": len(lens)}
            if len(neutral) >= 2 and len(lens) >= 2:
                nc = _centroid(neutral); lc = _centroid(lens); vector = lc - nc
                variance = max(_variance(neutral, nc), _variance(lens, lc))
                threshold = max(0.15, 1.5 * variance)
                magnitude = float(np.linalg.norm(vector)); cosine = _cosine(vector, np.array(anchor["anchor_vector_bark"]))
                source_ok = float(np.linalg.norm(nc-source)) < float(np.linalg.norm(nc-target))
                target_ok = float(np.linalg.norm(lc-target)) < float(np.linalg.norm(lc-source))
                result.update({
                    "neutral_centroid_bark": nc.tolist(), "lens_centroid_bark": lc.tolist(),
                    "vector_bark": vector.tolist(), "take_variance_bark": variance,
                    "magnitude_bark": magnitude, "magnitude_threshold_bark": threshold,
                    "anchor_direction_cosine": cosine, "neutral_source_category_pass": source_ok,
                    "lens_target_category_pass": target_ok,
                    "passed": bool(source_ok and target_ok and magnitude > threshold and cosine >= 0.50),
                })
            family_results[key] = result
        carrier_pass = all(item["passed"] for item in family_results.values())
        carriers[carrier.carrier_id] = {"shell": carrier.shell, "families": family_results, "passed": carrier_pass}

        for item in selected:
            category: dict[str, Any] = {}
            if item["analysis"]["status"] == "measurable":
                for ceiling in CEILINGS:
                    key = str(ceiling); measurement = item["analysis"]["measurements"][key]
                    point = np.array([measurement["f1_bark"], measurement["f2_bark"]])
                    anchor = ANCHOR_GATE["families"][key]
                    source = np.array(anchor["source_centroid_bark"]); target = np.array(anchor["target_centroid_bark"])
                    source_distance = float(np.linalg.norm(point-source)); target_distance = float(np.linalg.norm(point-target))
                    side_pass = source_distance < target_distance if item["slot"]["side"] == "neutral" else target_distance < source_distance
                    category[key] = {"source_distance_bark": source_distance, "target_distance_bark": target_distance, "passed": bool(measurement["plausibility_pass"] and side_pass)}
            item["analysis"]["individual_category"] = category
            item["analysis"]["individual_category_pass"] = bool(category and all(x["passed"] for x in category.values()))
            item["analysis"]["carrier_contrast_pass"] = carrier_pass
            item["analysis"]["eligible_for_pairing"] = bool(item["analysis"]["individual_category_pass"] and carrier_pass)
    return carriers


def _timing(record: dict[str, Any]) -> AudioTiming:
    payload = record["timing"]
    return AudioTiming(
        duration_s=float(payload["duration_s"]), sample_rate_hz=int(payload["sample_rate_hz"]),
        decoded_sample_count=int(payload["decoded_sample_count"]), clipped_fraction=float(payload["clipped_fraction"]),
        utterance_duration_s=float(payload["utterance_duration_s"]), estimated_syllables_per_second=payload.get("estimated_syllables_per_second"),
        interior_pause_count=int(payload["interior_pause_count"]), interior_pause_s=float(payload["interior_pause_s"]),
        interior_pauses=tuple(PauseInterval(**x) for x in payload.get("interior_pauses", [])),
    )


def _candidate(record: dict[str, Any]) -> TakeCandidate:
    return TakeCandidate(
        side=record["slot"]["side"], take_index=int(record["slot"]["take_index"]), audio_sha256=record["audio_sha256"],
        renderer_model=record.get("resolved_model", MODEL), voice=VOICE, prompt_contract_fingerprint=prompt_contract_fingerprint(),
        transcript_exact=bool(record["transcript_check"]["exact_token_match"]), integrity_ok=bool(record["integrity_ok"]),
        timing=_timing(record), audio_path=record["audio_path"], request_id=record.get("request_id", ""), resolved_model=record.get("resolved_model", MODEL),
    )


def _select_blocks(records: list[dict[str, Any]], carriers: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    thresholds = PairingThresholds()
    for carrier in CARRIERS:
        selected = [item for item in records if item["slot"]["carrier_id"] == carrier.carrier_id and item["analysis"]["eligible_for_pairing"]]
        neutral = sorted((_candidate(x) for x in selected if x["slot"]["side"] == "neutral"), key=lambda x: x.take_index)
        lens = sorted((_candidate(x) for x in selected if x["slot"]["side"] == "lens"), key=lambda x: x.take_index)
        blocks=[]
        for baseline in neutral:
            for control in neutral:
                if control.take_index == baseline.take_index: continue
                nn = evaluate_pair(baseline, replace(control, side="lens"), thresholds)
                for lens_take in lens:
                    nl = evaluate_pair(baseline, lens_take, thresholds)
                    if nn.qualified and nl.qualified:
                        blocks.append({"baseline_neutral_take":baseline.take_index,"control_neutral_take":control.take_index,"lens_take":lens_take.take_index,"combined_score":nn.score+nl.score,"neutral_neutral":asdict(nn),"neutral_lens":asdict(nl)})
        blocks.sort(key=lambda x:(x["combined_score"],x["baseline_neutral_take"],x["lens_take"],x["control_neutral_take"]))
        output[carrier.carrier_id]={"carrier_contrast_pass":carriers[carrier.carrier_id]["passed"],"eligible_neutral_takes":[x.take_index for x in neutral],"eligible_lens_takes":[x.take_index for x in lens],"qualified_joint_block_count":len(blocks),"selected_block":blocks[0] if blocks else None,"all_qualified_blocks":blocks}
    return output


def analyze_sentence_pair_v2() -> dict[str, Any]:
    if protocol_record()["protocol_sha256"] != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError("sentence-pair-v2 protocol hash mismatch")
    if sha256_file(PRAAT) != PRAAT_SHA256:
        raise RuntimeError("Praat executable changed")
    if sha256_file(MEASUREMENT_SCRIPT) != MEASUREMENT_SCRIPT_SHA256:
        raise RuntimeError("sentence-pair-v2 measurement script changed")
    run_dir=Paths().artifacts/"sentence-pair-v2"/RUN_ID
    records=json.loads((run_dir/"render-records.json").read_text(encoding="utf-8"))
    for index,record in enumerate(records,start=1):
        record["analysis"]=_analyze_take(record)
        print(f"acoustic {index:02d}/24 {record['slot']['slot_id']}: {record['analysis']['status']}",flush=True)
    carriers=_classify(records)
    blocks=_select_blocks(records,carriers)
    unresolved_external=any(x.get("external_failure_unresolved") for x in records)
    complete_blocks=sum(x["selected_block"] is not None for x in blocks.values())
    classification="ready_for_blinded_listener_pilot" if complete_blocks>=2 else "inconclusive_external_failure" if unresolved_external else "architectural_failed_or_insufficient"
    result={"schema_version":1,"status":"analysis_complete","run_id":RUN_ID,"protocol_sha256":EXPECTED_PROTOCOL_SHA256,"classification":classification,"complete_carrier_blocks":complete_blocks,"measurable_take_count":sum(x["analysis"]["status"]=="measurable" for x in records),"individual_pairing_eligible_take_count":sum(x["analysis"].get("eligible_for_pairing",False) for x in records),"carriers":carriers,"pair_blocks":blocks,"records":records}
    atomic_write_json(run_dir/"analysis.json",result)
    return result
