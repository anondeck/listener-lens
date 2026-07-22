from __future__ import annotations

import base64
import hashlib
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

from openai import OpenAI

from .acoustic_calibration import _usage_dict, estimated_cost_usd, summarize_usage
from .audio_conformance import (
    FLOW_DEVELOPER_PROMPT,
    AudioTiming,
    ConformanceSample,
    PauseInterval,
    analyze_audio_timing,
    build_messages,
    check_transcript,
)
from .config import DEVLOG_PATH, Paths, stable_json
from .listener_lens import ListenerLensEngine, TRANSFORM_ALGORITHM_VERSION
from .matched_pairs import (
    PairSelection,
    PairingThresholds,
    TakeCandidate,
    select_best_pair,
)
from .util import atomic_write_json, sha256_file, write_csv


PREREGISTRATION_HEADING = (
    "## Curated 4+4 matched-pair preregistration — July 15, 2026"
)
RUN_ID = "20260715-curated-ptbr-ae-matched-pair"
MANIFEST_SEED = "curated-ptbr-ae-pair-20260715"
MODEL = "gpt-audio-1.5"
VOICE = "marin"
FORMAT = "wav"
PROFILE_ID = "en-to-pt-BR-vowel-lens"
SOURCE_SENTENCE = "The black cat sat on the wooden bench."
NEUTRAL_SCRIPT = "frohr bavd bavd bavd lohm frohr tadril prohk."
LENS_SCRIPT = "frohr behvd behvd behvd lohm frohr tadril prohk."
DELIVERY = (
    "Fluent natural mainstream U.S. English. Perform this invented sentence at "
    "a normal conversational pace with connected speech, natural reductions, "
    "and one continuous intonation contour. Do not read it as a list."
)
RULES_SHA256 = "fd532cf18eec50595a6bdf9420619fa9f5993927ea2ef78f94d54b022823e76b"
TRANSFORM_CACHE_KEY = "83a7f81070fa2d4eceef0ddec92b187907f170d8c56db728e36fed83d47c77e7"
TRANSFORM_RESULT_SHA256 = (
    "e58db2f4ada4217da45753885be6f17dbfcdc353ec7c0cb48d7c32eca2ffae29"
)
EXPECTED_PROTOCOL_SHA256 = (
    "a0d1edeec453b6393c54d88402acd232697c62cfe68fc0681a3b9d2dd92ab8b3"
)
FROZEN_MANIFEST_PATH = (
    Paths().root
    / "artifacts"
    / "matched-pairs"
    / RUN_ID
    / "manifest.json"
)

RESULT_FIELDS = [
    "request_order",
    "slot_id",
    "side",
    "take_index",
    "script",
    "status",
    "request_id",
    "resolved_model",
    "latency_ms",
    "provider_transcript",
    "exact_token_match",
    "integrity_ok",
    "audio_filename",
    "audio_sha256",
    "duration_s",
    "sample_rate_hz",
    "decoded_sample_count",
    "clipped_fraction",
    "utterance_duration_s",
    "interior_pause_count",
    "interior_pause_s",
    "interior_pause_positions_json",
    "prompt_tokens",
    "prompt_audio_tokens",
    "completion_tokens",
    "completion_audio_tokens",
    "estimated_request_cost_usd",
    "error_type",
    "error_detail",
]


@dataclass(frozen=True)
class CuratedSlot:
    slot_id: str
    side: Literal["neutral", "lens"]
    take_index: int
    script: str


def build_manifest() -> tuple[CuratedSlot, ...]:
    slots = [
        CuratedSlot(
            slot_id=f"{side}__take-{take_index}",
            side=side,  # type: ignore[arg-type]
            take_index=take_index,
            script=NEUTRAL_SCRIPT if side == "neutral" else LENS_SCRIPT,
        )
        for side in ("neutral", "lens")
        for take_index in range(1, 5)
    ]
    random.Random(MANIFEST_SEED).shuffle(slots)
    if len(slots) != 8 or len({slot.slot_id for slot in slots}) != 8:
        raise AssertionError("Curated matched-pair manifest requires eight slots")
    return tuple(slots)


def prompt_contract_fingerprint() -> str:
    contract = {
        "model": MODEL,
        "voice": VOICE,
        "delivery": DELIVERY,
        "developer_prompt": FLOW_DEVELOPER_PROMPT,
        "protocol": "json-flow-v2",
    }
    return hashlib.sha256(stable_json(contract).encode("utf-8")).hexdigest()


def frozen_transform_record() -> dict[str, Any]:
    # The current profile may acquire claim-label or product-state metadata after a
    # run. Read the immutable run record rather than pretending those later edits
    # were part of the frozen stimulus.
    if FROZEN_MANIFEST_PATH.is_file():
        manifest = json.loads(FROZEN_MANIFEST_PATH.read_text(encoding="utf-8"))
        if manifest.get("protocol_sha256") != EXPECTED_PROTOCOL_SHA256:
            raise RuntimeError("Frozen curated manifest protocol hash changed")
        stimulus = manifest.get("stimulus")
        if not isinstance(stimulus, dict):
            raise RuntimeError("Frozen curated manifest has no stimulus record")
        if stimulus.get("rules_sha256") != RULES_SHA256:
            raise RuntimeError("Frozen curated manifest rule hash changed")
        if stimulus.get("transform_cache_key") != TRANSFORM_CACHE_KEY:
            raise RuntimeError("Frozen curated manifest transform cache key changed")
        if stimulus.get("transform_result_sha256") != TRANSFORM_RESULT_SHA256:
            raise RuntimeError("Frozen curated manifest transform result changed")
        return stimulus

    rules_path = Paths().root / "rules" / "listener_lenses.yaml"
    if sha256_file(rules_path) != RULES_SHA256:
        raise RuntimeError("Listener-lens rule table changed after stimulus freeze")
    result = ListenerLensEngine().transform(SOURCE_SENTENCE, PROFILE_ID)
    result_sha256 = hashlib.sha256(
        stable_json(result.to_dict()).encode("utf-8")
    ).hexdigest()
    if result.cache_key != TRANSFORM_CACHE_KEY:
        raise RuntimeError("Frozen transform cache key changed")
    if result_sha256 != TRANSFORM_RESULT_SHA256:
        raise RuntimeError("Frozen transform output changed")
    if result.neutral_script != NEUTRAL_SCRIPT or result.lens_script != LENS_SCRIPT:
        raise RuntimeError("Frozen matched scripts changed")
    if [rule.rule_id for rule in result.applied_rules] != [
        "ptbr.vowel.ae_to_eh"
    ]:
        raise RuntimeError("Flagship stimulus must apply only the enabled ae rule")
    if len(result.slots) != 3:
        raise RuntimeError("Flagship stimulus must contain exactly three ae slots")
    return {
        "profile_id": PROFILE_ID,
        "source_sentence": SOURCE_SENTENCE,
        "algorithm_version": TRANSFORM_ALGORITHM_VERSION,
        "rules_sha256": RULES_SHA256,
        "transform_cache_key": result.cache_key,
        "transform_result_sha256": result_sha256,
        "neutral_script": result.neutral_script,
        "lens_script": result.lens_script,
        "comparison_available": result.comparison_available,
        "applied_rules": [asdict(rule) for rule in result.applied_rules],
        "slots": [asdict(slot) for slot in result.slots],
    }


def protocol_record() -> dict[str, Any]:
    thresholds = PairingThresholds()
    protocol: dict[str, Any] = {
        "schema_version": 1,
        "status": "curated_matched_pair_preregistered",
        "run_id": RUN_ID,
        "stimulus": frozen_transform_record(),
        "model": MODEL,
        "voice": VOICE,
        "format": FORMAT,
        "modalities": ["text", "audio"],
        "store": False,
        "protocol": "json-flow-v2",
        "delivery": DELIVERY,
        "prompt_contract_fingerprint": prompt_contract_fingerprint(),
        "manifest_seed": MANIFEST_SEED,
        "logical_slots": 8,
        "stimuli": [asdict(slot) for slot in build_manifest()],
        "request_policy": {
            "sdk_automatic_retries": 0,
            "manual_retries": 0,
            "replacement_takes": 0,
            "maximum_api_attempts": 8,
            "valid_or_invalid_response_makes_slot_final": True,
            "interrupted_started_slot_makes_slot_final_failed": True,
        },
        "qualification": {
            "exact_provider_transcript_required": True,
            "valid_decodable_wav_required": True,
            "maximum_duration_s": 45.0,
            "maximum_clipped_fraction_exclusive": 0.001,
        },
        "pair_selection": {
            "candidate_pairs": 16,
            "thresholds": asdict(thresholds),
            "selection_inputs": [
                "transcript exactness",
                "WAV integrity",
                "renderer/voice/prompt equality",
                "utterance duration",
                "pause count",
                "normalized pause position",
                "pause duration",
            ],
            "formant_analysis_used_for_selection": False,
            "human_listening_used_for_selection": False,
            "thresholds_remain_provisional": True,
            "degraded_result_if_none_qualify": True,
        },
    }
    protocol["protocol_sha256"] = hashlib.sha256(
        stable_json(protocol).encode("utf-8")
    ).hexdigest()
    return protocol


def _empty_timing() -> AudioTiming:
    return AudioTiming(
        duration_s=0.0,
        sample_rate_hz=0,
        decoded_sample_count=0,
        clipped_fraction=1.0,
        utterance_duration_s=0.0,
        estimated_syllables_per_second=None,
        interior_pause_count=0,
        interior_pause_s=0.0,
        interior_pauses=(),
    )


def _render_slot(
    *,
    client: Any,
    slot: CuratedSlot,
    request_order: int,
    audio_path: Path,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "request_order": request_order,
        "slot": asdict(slot),
        "status": "failed",
        "model": MODEL,
        "voice": VOICE,
        "format": FORMAT,
        "usage": {},
        "estimated_cost_usd": 0.0,
        "integrity_ok": False,
    }
    started = time.monotonic()
    partial = audio_path.with_suffix(".partial.wav")
    try:
        sample = ConformanceSample(
            sample_id=slot.slot_id,
            language="en",
            script=slot.script,
            delivery=DELIVERY,
        )
        completion = client.chat.completions.create(
            model=MODEL,
            modalities=["text", "audio"],
            audio={"voice": VOICE, "format": FORMAT},
            messages=build_messages(sample, "json-flow-v2"),
            store=False,
        )
        record["request_id"] = getattr(completion, "_request_id", None) or ""
        record["resolved_model"] = getattr(completion, "model", MODEL)
        record["usage"] = _usage_dict(completion)
        record["estimated_cost_usd"] = estimated_cost_usd(record["usage"])
        message = completion.choices[0].message
        audio = message.audio
        if audio is None or not audio.data:
            raise ValueError("gpt-audio-1.5 returned no audio payload")
        transcript = getattr(audio, "transcript", "") or ""
        record["provider_transcript"] = transcript
        record["transcript_check"] = asdict(check_transcript(slot.script, transcript))
        partial.parent.mkdir(parents=True, exist_ok=True)
        partial.write_bytes(base64.b64decode(audio.data, validate=True))
        timing = analyze_audio_timing(partial, intended_syllables=None)
        if timing.decoded_sample_count <= 0:
            raise ValueError("response audio was not a valid decodable PCM WAV")
        partial.replace(audio_path)
        integrity_ok = (
            0 < timing.duration_s <= 45.0 and timing.clipped_fraction < 0.001
        )
        record.update(
            {
                "status": "ok",
                "audio_filename": audio_path.name,
                "audio_path": str(audio_path),
                "audio_sha256": sha256_file(audio_path),
                "timing": asdict(timing),
                "integrity_ok": integrity_ok,
            }
        )
    except Exception as exc:
        partial.unlink(missing_ok=True)
        record.update(
            {
                "error_type": type(exc).__name__,
                "error_detail": str(exc).replace("\n", " ")[:500],
            }
        )
    record["latency_ms"] = round((time.monotonic() - started) * 1000)
    return record


def _interrupted_record(
    *, request_order: int, slot: CuratedSlot
) -> dict[str, Any]:
    return {
        "request_order": request_order,
        "slot": asdict(slot),
        "status": "failed",
        "model": MODEL,
        "voice": VOICE,
        "format": FORMAT,
        "usage": {},
        "estimated_cost_usd": 0.0,
        "integrity_ok": False,
        "latency_ms": 0,
        "error_type": "InterruptedRequestState",
        "error_detail": (
            "The slot had a started receipt without a completed response and is "
            "final-failed without retry or replacement."
        ),
    }


def _timing_from_record(record: dict[str, Any]) -> AudioTiming:
    payload = record.get("timing")
    if not payload:
        return _empty_timing()
    return AudioTiming(
        duration_s=float(payload["duration_s"]),
        sample_rate_hz=int(payload["sample_rate_hz"]),
        decoded_sample_count=int(payload["decoded_sample_count"]),
        clipped_fraction=float(payload["clipped_fraction"]),
        utterance_duration_s=float(payload["utterance_duration_s"]),
        estimated_syllables_per_second=payload.get(
            "estimated_syllables_per_second"
        ),
        interior_pause_count=int(payload["interior_pause_count"]),
        interior_pause_s=float(payload["interior_pause_s"]),
        interior_pauses=tuple(
            PauseInterval(**pause) for pause in payload.get("interior_pauses", [])
        ),
    )


def _candidate(record: dict[str, Any]) -> TakeCandidate:
    slot = record["slot"]
    transcript_exact = bool(
        (record.get("transcript_check") or {}).get("exact_token_match")
    )
    return TakeCandidate(
        side=slot["side"],
        take_index=int(slot["take_index"]),
        audio_sha256=record.get("audio_sha256", ""),
        renderer_model=record.get("resolved_model", MODEL),
        voice=VOICE,
        prompt_contract_fingerprint=prompt_contract_fingerprint(),
        transcript_exact=transcript_exact,
        integrity_ok=bool(record.get("integrity_ok")),
        timing=_timing_from_record(record),
        failure_detail=record.get("error_detail", ""),
        audio_path=record.get("audio_path", ""),
        request_id=record.get("request_id", ""),
        resolved_model=record.get("resolved_model", MODEL),
    )


def _flatten_record(record: dict[str, Any]) -> dict[str, Any]:
    slot = record["slot"]
    timing = _timing_from_record(record).to_result_fields()
    transcript = record.get("transcript_check") or {}
    usage = record.get("usage") or {}
    prompt_details = usage.get("prompt_tokens_details") or {}
    completion_details = usage.get("completion_tokens_details") or {}
    return {
        "request_order": record["request_order"],
        **slot,
        "status": record.get("status", "failed"),
        "request_id": record.get("request_id", ""),
        "resolved_model": record.get("resolved_model", ""),
        "latency_ms": record.get("latency_ms", ""),
        "provider_transcript": record.get("provider_transcript", ""),
        "exact_token_match": transcript.get("exact_token_match", False),
        "integrity_ok": record.get("integrity_ok", False),
        "audio_filename": record.get("audio_filename", ""),
        "audio_sha256": record.get("audio_sha256", ""),
        **timing,
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "prompt_audio_tokens": int(prompt_details.get("audio_tokens") or 0),
        "completion_tokens": int(usage.get("completion_tokens") or 0),
        "completion_audio_tokens": int(completion_details.get("audio_tokens") or 0),
        "estimated_request_cost_usd": record.get("estimated_cost_usd", 0),
        "error_type": record.get("error_type", ""),
        "error_detail": record.get("error_detail", ""),
    }


def _selection_record(records: Sequence[dict[str, Any]]) -> PairSelection:
    candidates = [_candidate(record) for record in records]
    neutral = tuple(
        sorted(
            (candidate for candidate in candidates if candidate.side == "neutral"),
            key=lambda candidate: candidate.take_index,
        )
    )
    lens = tuple(
        sorted(
            (candidate for candidate in candidates if candidate.side == "lens"),
            key=lambda candidate: candidate.take_index,
        )
    )
    if len(neutral) != 4 or len(lens) != 4:
        raise RuntimeError("Curated selection requires four records per side")
    thresholds = PairingThresholds()
    selected, scores = select_best_pair(neutral, lens, thresholds)
    return PairSelection(
        mode="curated",
        selected=selected,
        neutral_takes=neutral,
        lens_takes=lens,
        all_pair_scores=scores,
        thresholds=thresholds,
        retry_side=None,
        total_renders=8,
    )


def run_curated_matched_pair(
    *, client: Any | None = None, run_id: str = RUN_ID
) -> dict[str, Any]:
    from .api import require_api_key

    if PREREGISTRATION_HEADING not in DEVLOG_PATH.read_text(encoding="utf-8"):
        raise RuntimeError("The curated matched-pair preregistration is missing")
    protocol = protocol_record()
    if protocol["protocol_sha256"] != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError("Curated matched-pair protocol does not match its freeze")
    if run_id != RUN_ID:
        raise RuntimeError(f"The frozen curated run id is {RUN_ID}")
    if client is None:
        require_api_key()
        client = OpenAI(max_retries=0)

    paths = Paths()
    paths.run_dir(run_id)
    run_dir = paths.artifacts / "matched-pairs" / run_id
    manifest_path = run_dir / "manifest.json"
    if manifest_path.is_file():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != protocol:
            raise RuntimeError("Existing matched-pair manifest does not match freeze")
    else:
        if run_dir.exists() and any(run_dir.iterdir()):
            raise RuntimeError("Matched-pair directory exists without its manifest")
        atomic_write_json(manifest_path, protocol)

    records: list[dict[str, Any]] = []
    for request_order, slot in enumerate(build_manifest(), start=1):
        receipt_path = run_dir / "slots" / f"{request_order:03d}__{slot.slot_id}.json"
        audio_path = run_dir / "audio" / f"{request_order:03d}__{slot.slot_id}.wav"
        if receipt_path.is_file():
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            if receipt.get("status") == "complete":
                records.append(receipt["record"])
                continue
            record = _interrupted_record(request_order=request_order, slot=slot)
        else:
            atomic_write_json(
                receipt_path,
                {"status": "started", "request_order": request_order, "slot": asdict(slot)},
            )
            record = _render_slot(
                client=client,
                slot=slot,
                request_order=request_order,
                audio_path=audio_path,
            )
        atomic_write_json(receipt_path, {"status": "complete", "record": record})
        records.append(record)
        print(
            f"curated {request_order:02d}/8 {slot.slot_id}: {record['status']} "
            f"(exact={bool((record.get('transcript_check') or {}).get('exact_token_match'))}, "
            f"integrity={record.get('integrity_ok', False)})",
            flush=True,
        )

    if len(records) != 8:
        raise AssertionError("Curated run must retain exactly eight logical slots")
    write_csv(
        run_dir / "results.csv",
        [_flatten_record(record) for record in records],
        RESULT_FIELDS,
    )
    selection = _selection_record(records)
    selection_payload = selection.to_record()
    selection_payload.update(
        {
            "selection_basis": "conformance_and_provisional_timing_only",
            "formant_analysis_used": False,
            "human_listening_used": False,
        }
    )
    atomic_write_json(run_dir / "pair-selection.json", selection_payload)

    selected_neutral = next(
        candidate
        for candidate in selection.neutral_takes
        if candidate.take_index == selection.selected.neutral_take_index
    )
    selected_lens = next(
        candidate
        for candidate in selection.lens_takes
        if candidate.take_index == selection.selected.lens_take_index
    )
    usage = summarize_usage(records)
    summary = {
        "schema_version": 1,
        "status": "complete",
        "run_id": run_id,
        "protocol_sha256": protocol["protocol_sha256"],
        "logical_slots": 8,
        "api_attempts": 8,
        "successful_audio_responses": sum(record["status"] == "ok" for record in records),
        "exact_transcripts": sum(
            bool((record.get("transcript_check") or {}).get("exact_token_match"))
            for record in records
        ),
        "integrity_passes": sum(bool(record.get("integrity_ok")) for record in records),
        "qualified_pair_count": sum(score.qualified for score in selection.all_pair_scores),
        "selected_pair_qualified": selection.selected.qualified,
        "selected_pair_degraded": selection.degraded,
        "selected_pair": asdict(selection.selected),
        "selected_neutral_audio_path": selected_neutral.audio_path,
        "selected_lens_audio_path": selected_lens.audio_path,
        "thresholds": asdict(selection.thresholds),
        "thresholds_remain_provisional": True,
        "usage": usage,
        "manifest": str(manifest_path),
        "results_csv": str(run_dir / "results.csv"),
        "pair_selection_json": str(run_dir / "pair-selection.json"),
    }
    atomic_write_json(run_dir / "summary.json", summary)
    return summary
