from __future__ import annotations

import base64
import json
import os
from dataclasses import asdict
from typing import Any

from .api import require_api_key
from .config import Paths, sha256_json, stable_json
from .native_realtime_probe import (
    GROUP_SYLLABLE_PATTERN,
    GROUPED_CONTROL_SHA256,
    GROUPED_TRANSFER_PROMPT,
    PROFILE_ID,
    REALTIME_MODEL,
    REFERENCE_GATE,
    SAMPLE_RATE_HZ,
    SOURCE_ANCHOR_SHA256,
    SOURCE_SYLLABLES,
    SOURCE_TEXT,
    STRUCTURED_DELEXICALIZATION_PROMPT,
    TARGET_GLOBAL_SYLLABLE,
    TARGET_GROUP_INDEX,
    VOICE,
    ProbeSlot,
    _build_review,
    _combined_cost,
    _pcm16_from_wav,
    _reference_record,
    _render_slot,
    _slot_prompt,
    _verify_source_audio,
    _whisper_and_acoustic_audit,
)
from .sentence_pair_v2_analysis import CEILINGS
from .util import atomic_write_json


RUN_ID = "20260716-realtime-transport-correction-v1"
MAX_COST_USD = 0.06


def build_manifest() -> tuple[ProbeSlot, ...]:
    return (
        ProbeSlot(1, "realtime21-structured-delex-corrected-1", REALTIME_MODEL, "realtime_websocket", "structured_delex"),
        ProbeSlot(2, "realtime21-exact-grouped-corrected-1", REALTIME_MODEL, "realtime_websocket", "exact_grouped"),
    )


def protocol_record() -> dict[str, Any]:
    anchor, control = _verify_source_audio()
    payload = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "status": "transport_correction_frozen_before_paid_calls_and_listening",
        "relationship_to_v2": (
            "The two native Realtime v2 slots returned no audio because PCM output rate was omitted. "
            "This separate run changes only the required output-format rate and slot identities."
        ),
        "source_text": SOURCE_TEXT,
        "profile_id": PROFILE_ID,
        "source_syllables": SOURCE_SYLLABLES,
        "voice": VOICE,
        "source_bindings": {
            "anchor_path": str(anchor),
            "anchor_sha256": SOURCE_ANCHOR_SHA256,
            "audio15_exact_grouped_control_path": str(control),
            "audio15_exact_grouped_control_sha256": GROUPED_CONTROL_SHA256,
        },
        "manifest": [asdict(slot) for slot in build_manifest()],
        "prompts": {
            "structured_delex": STRUCTURED_DELEXICALIZATION_PROMPT,
            "exact_grouped": GROUPED_TRANSFER_PROMPT,
        },
        "request_payloads": {
            "structured_delex": _slot_prompt(build_manifest()[0])[1],
            "exact_grouped": _slot_prompt(build_manifest()[1])[1],
        },
        "only_amendment": {
            "session_audio_output_format": {"type": "audio/pcm", "rate": SAMPLE_RATE_HZ},
            "preflight": "session.created then corrected session.updated succeeded without response.create",
        },
        "unchanged_gates": {
            "group_syllables": list(GROUP_SYLLABLE_PATTERN),
            "target_group_index_zero_based": TARGET_GROUP_INDEX,
            "target_global_syllable": TARGET_GLOBAL_SYLLABLE,
            "ceilings_hz": list(CEILINGS),
            "reference_gate": REFERENCE_GATE,
            "provider_content_and_local_whisper_leakage": "same as native-Realtime v2",
            "target_acoustic": "same as native-Realtime v2",
        },
        "selection": "none; every valid returned slot is final and analyzed",
        "retry_policy": (
            "one same-slot retry only for 429, 5xx, timeout, connection failure, "
            "or completed transport returning no audio; quality or gate failure is never retried"
        ),
        "limits": {
            "logical_slots": 2,
            "maximum_successful_audio_returns": 2,
            "maximum_estimated_cost_usd": MAX_COST_USD,
        },
        "stopping_rule": (
            "No further transport rerun follows this correction. A schema failure is recorded as a "
            "harness result; valid audio proceeds to the unchanged evidence decomposition."
        ),
        "official_api_reference": (
            "https://platform.openai.com/docs/api-reference/realtime-client-events/session"
        ),
    }
    return {**payload, "protocol_sha256": sha256_json(payload)}


def prepare_probe() -> dict[str, Any]:
    protocol = protocol_record()
    path = Paths().artifacts / "native-realtime" / RUN_ID / "protocol.json"
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if stable_json(existing) != stable_json(protocol):
            raise RuntimeError("Existing Realtime transport-correction protocol differs from the freeze")
    else:
        atomic_write_json(path, protocol)
    return protocol


def run_probe() -> dict[str, Any]:
    protocol = prepare_probe()
    require_api_key()
    api_key = os.environ["OPENAI_API_KEY"].strip()
    anchor, control = _verify_source_audio()
    anchor_pcm16 = _pcm16_from_wav(anchor)
    anchor_wav_base64 = base64.b64encode(anchor.read_bytes()).decode("ascii")
    anchor_record = _reference_record(anchor)
    run_dir = Paths().artifacts / "native-realtime" / RUN_ID
    records: list[dict[str, Any]] = []
    for slot in build_manifest():
        receipt = run_dir / "slots" / f"{slot.request_order:02d}__{slot.slot_id}.json"
        output = run_dir / "audio" / f"{slot.request_order:02d}__{slot.slot_id}.wav"
        if receipt.is_file():
            saved = json.loads(receipt.read_text(encoding="utf-8"))
            if saved.get("receipt_status") != "complete":
                raise RuntimeError(f"Ambiguous interrupted Realtime correction slot: {slot.slot_id}")
            record = saved["record"]
        else:
            atomic_write_json(receipt, {"receipt_status": "started", "slot": asdict(slot)})
            record = _render_slot(
                client=None,
                slot=slot,
                anchor_pcm16=anchor_pcm16,
                anchor_wav_base64=anchor_wav_base64,
                anchor_record=anchor_record,
                api_key=api_key,
                output=output,
                realtime_output_rate_hz=SAMPLE_RATE_HZ,
            )
            atomic_write_json(receipt, {"receipt_status": "complete", "record": record})
        records.append(record)
        usage = _combined_cost(records)
        if usage["estimated_cost_usd"] > MAX_COST_USD:
            raise RuntimeError("Realtime transport-correction cost cap exceeded")
        print(
            f"realtime correction {slot.request_order}/2 {slot.slot_id}: {record['status']} "
            f"cost=${usage['estimated_cost_usd']:.4f}",
            flush=True,
        )

    for record in records:
        _whisper_and_acoustic_audit(record, run_dir)
    usage = _combined_cost(records)
    summary = {
        "schema_version": 1,
        "status": "probe_complete",
        "run_id": RUN_ID,
        "protocol_sha256": protocol["protocol_sha256"],
        "logical_slots": len(build_manifest()),
        "audio_returned": sum(bool(record.get("audio_sha256")) for record in records),
        "structured_content_passes": sum(
            bool(record.get("content_control_pass"))
            for record in records if record["slot"]["mode"] == "structured_delex"
        ),
        "audio_leak_screen_passes": sum(
            bool((record.get("local_whisper") or {}).get("audio_leak_screen_pass"))
            for record in records if record["slot"]["mode"] == "structured_delex"
        ),
        "neutral_source_category_passes": sum(
            bool((record.get("target_acoustic") or {}).get("neutral_source_category_pass"))
            for record in records if record["slot"]["mode"] == "structured_delex"
        ),
        "usage": usage,
    }
    atomic_write_json(run_dir / "records.json", records)
    atomic_write_json(run_dir / "summary.json", summary)
    _build_review(records, run_dir, control)
    return summary
