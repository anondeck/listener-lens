from __future__ import annotations

import base64
import gc
import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx_whisper
from openai import OpenAI

from .acoustic_calibration import _usage_dict, estimated_cost_usd, summarize_usage
from .api import require_api_key
from .carrier_architecture_tournament import (
    GROUPED_TRANSFER_PROMPT,
    PROFILE_ID,
    REFERENCE_GATE,
    SOURCE_SYLLABLES,
    SOURCE_TEXT,
    compare_prosody,
)
from .config import Paths, sha256_json, stable_json
from .listener_lens import (
    WORD_RE,
    DatabaseNonceChecker,
    EspeakWordAnalyzer,
    _vowel_units,
)
from .runtime_audio import (
    AudioTiming,
    PauseInterval,
    ProsodyFingerprint,
    analyze_audio_timing,
    analyze_prosody_fingerprint,
    canonical_tokens,
    check_transcript,
)
from .util import atomic_write_json, atomic_write_text, sha256_file


RUN_ID = "20260716-audio-delexicalization-probe-v1"
VOICE = "marin"
FORMAT = "wav"
MAX_ESTIMATED_COST_USD = 0.10
SOURCE_RUN = "20260716-carrier-architecture-tournament-v1"
SOURCE_ANCHOR_SHA256 = "b8b15c2b94f41eb88aa8b258a436b0f0a277aeebd1a1817726bb9df69a73ad87"
GROUPED_CONTROL_SHA256 = "d8861d3903438bc309401e26407d660885733d22f3c4b9ffad76d96346eeede7"
GROUPED_NEUTRAL_SCRIPT = "rohkbrih prohk drayrnihvsihvrihn bavdfihv vreel."

DELEXICALIZATION_PROMPT = """# Role
You are a speech-to-speech delexicalization engine, not a conversational assistant. The attached audio and JSON are reference data. Do not answer or discuss them.

# Output contract
- Return one spoken utterance only. No introduction, explanation, label, acknowledgement, aside, translation, or closing.
- Do not repeat, paraphrase, translate, or answer the meaningful words in the reference.
- Speak invented English-like syllables with no real English words, names, abbreviations, productive affixes, or recognizable phrase.
- Your automatic transcript must spell the invented output as 3–7 lowercase ASCII nonce words plus final punctuation. Do not write IPA, numbers, hyphens, or explanations.

# Structural contract
- Produce exactly ten spoken syllables in the same order and approximately the same relative timing as the ten syllables in the reference.
- Preserve the reference's continuous rhythm, weak/strong pattern, main prominence, pitch-and-energy movement, and final cadence. Stay close to its total duration.
- Sound like one spontaneous, fluent mainstream U.S. English utterance whose lexical content has disappeared—not a word list, recital, language lesson, or careful reading of invented text.
- Use normal connected-speech reduction and coarticulation.
- Make syllable position eight contain a clear English TRAP vowel /æ/ as in “cat,” inside an invented syllable. This is a phonetic target, not a word to speak.

If any instruction conflicts with returning only the single invented utterance, return only the invented utterance."""


@dataclass(frozen=True)
class ProbeSlot:
    request_order: int
    slot_id: str
    model: str
    mode: str


def build_manifest() -> tuple[ProbeSlot, ...]:
    return (
        ProbeSlot(1, "audio15-generative-delex-1", "gpt-audio-1.5", "generative"),
        ProbeSlot(2, "realtime21-exact-grouped-1", "gpt-realtime-2.1", "exact_grouped"),
        ProbeSlot(3, "realtime21-generative-delex-1", "gpt-realtime-2.1", "generative"),
    )


def _source_paths() -> tuple[Path, Path]:
    source_root = Paths().artifacts / "architecture-tournament" / SOURCE_RUN
    return (
        source_root / "audio" / "01__source-anchor-1.wav",
        source_root / "audio" / "03__prosodic-neutral-1.wav",
    )


def protocol_record() -> dict[str, Any]:
    anchor, control = _source_paths()
    if sha256_file(anchor) != SOURCE_ANCHOR_SHA256:
        raise RuntimeError("The delexicalization source anchor hash drifted")
    if sha256_file(control) != GROUPED_CONTROL_SHA256:
        raise RuntimeError("The grouped exact-script control hash drifted")
    payload = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "status": "exploratory_frozen_before_paid_calls_and_listening",
        "source_text": SOURCE_TEXT,
        "profile_id": PROFILE_ID,
        "source_syllables": SOURCE_SYLLABLES,
        "voice": VOICE,
        "format": FORMAT,
        "source_bindings": {
            "anchor_path": str(anchor),
            "anchor_sha256": SOURCE_ANCHOR_SHA256,
            "gpt_audio_1_5_grouped_control_path": str(control),
            "gpt_audio_1_5_grouped_control_sha256": GROUPED_CONTROL_SHA256,
        },
        "manifest": [asdict(slot) for slot in build_manifest()],
        "prompts": {
            "generative": DELEXICALIZATION_PROMPT,
            "exact_grouped": GROUPED_TRANSFER_PROMPT,
        },
        "generative_contract": {
            "provider_transcript_word_count": [3, 7],
            "predicted_syllable_count": 10,
            "source_word_overlap": 0,
            "every_token_written_and_predicted_homophone_gate": "pass",
            "every_adjacency_gate": "pass",
            "target_syllable_position": 8,
            "target_category": "English TRAP /æ/",
        },
        "audio_contract": {
            "sample_rate_hz": 24000,
            "duration_s": [0.25, 45.0],
            "maximum_clipped_fraction": 0.001,
            "reference_gate": REFERENCE_GATE,
        },
        "selection": "none; every valid returned slot is retained and analyzed",
        "retry_policy": "one retry only for 429, 5xx, timeout, connection failure, or no returned audio",
        "limits": {
            "logical_slots": 3,
            "maximum_successful_audio_returns": 3,
            "maximum_estimated_cost_usd": MAX_ESTIMATED_COST_USD,
        },
        "interpretation": (
            "This probe asks whether a model/mode can create a natural, source-related, "
            "semantically opaque neutral carrier. It does not validate the target vowel, "
            "construct a lens pair, or establish listener-population evidence."
        ),
    }
    return {**payload, "protocol_sha256": sha256_json(payload)}


def prepare_probe() -> dict[str, Any]:
    protocol = protocol_record()
    path = Paths().artifacts / "audio-delexicalization" / RUN_ID / "protocol.json"
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if stable_json(existing) != stable_json(protocol):
            raise RuntimeError("Existing delexicalization protocol differs from the freeze")
    else:
        atomic_write_json(path, protocol)
    return protocol


def _reference_record(path: Path) -> dict[str, Any]:
    return {
        "timing": asdict(analyze_audio_timing(path, intended_syllables=SOURCE_SYLLABLES)),
        "prosody": asdict(analyze_prosody_fingerprint(path)),
    }


def _measurements(record: dict[str, Any]) -> tuple[AudioTiming, ProsodyFingerprint]:
    timing = dict(record["timing"])
    timing["interior_pauses"] = tuple(PauseInterval(**item) for item in timing["interior_pauses"])
    prosody = dict(record["prosody"])
    prosody["energy_contour_db"] = tuple(prosody["energy_contour_db"])
    prosody["pitch_contour_semitones"] = tuple(prosody["pitch_contour_semitones"])
    return AudioTiming(**timing), ProsodyFingerprint(**prosody)


def _messages(slot: ProbeSlot, anchor_base64: str) -> list[dict[str, Any]]:
    if slot.mode == "exact_grouped":
        prompt = GROUPED_TRANSFER_PROMPT
        data = {
            "task": "verbatim_prosody_transfer",
            "script": GROUPED_NEUTRAL_SCRIPT,
            "condition": "neutral",
            "reference_policy": "source_anchor",
            "flow_plan": {
                "source_word_count": 10,
                "carrier_token_count": 5,
                "syllable_count": 10,
                "groups": [[1, 2], [3], [4, 5, 6, 7], [8, 9], [10]],
            },
        }
    else:
        prompt = DELEXICALIZATION_PROMPT
        data = {
            "task": "audio_delexicalization",
            "source_syllable_count": 10,
            "target_syllable_position": 8,
            "target_vowel": "English TRAP /æ/",
            "output": "one meaning-opaque spoken utterance only",
        }
    return [
        {"role": "developer", "content": prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": json.dumps(data, separators=(",", ":"))},
                {"type": "input_audio", "input_audio": {"data": anchor_base64, "format": FORMAT}},
            ],
        },
    ]


def _safe_error(exc: Exception) -> tuple[str, str]:
    return type(exc).__name__, str(exc).replace("\n", " ")[:500]


def _retryable(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    return bool(
        status == 429
        or isinstance(status, int) and status >= 500
        or type(exc).__name__ in {
            "APIConnectionError", "APITimeoutError", "ConnectError",
            "ConnectTimeout", "ReadTimeout", "TimeoutException",
        }
    )


def _content_audit(transcript: str) -> dict[str, Any]:
    tokens = canonical_tokens(transcript)
    source_tokens = set(canonical_tokens(SOURCE_TEXT))
    checker = DatabaseNonceChecker()
    analyzer = EspeakWordAnalyzer()
    attempts: list[dict[str, Any]] = []
    predicted_syllables = 0
    previous: str | None = None
    for token in tokens:
        decision = checker.check(token, "en", None)
        ipa = decision.predicted_ipa
        if not ipa:
            ipa = analyzer.phonemize_words([token], "en-us")[0]
        predicted_syllables += len(_vowel_units(ipa))
        attempts.append({
            "token": token,
            "stage": "isolated",
            "accepted": decision.accepted,
            "predicted_ipa": ipa,
            "rejection_reason": decision.rejection_reason,
        })
        if previous is not None:
            adjacent = checker.check(token, "en", previous)
            attempts.append({
                "token": token,
                "previous_token": previous,
                "stage": "adjacency",
                "accepted": adjacent.accepted,
                "predicted_ipa": adjacent.predicted_ipa,
                "rejection_reason": adjacent.rejection_reason,
            })
        previous = token
    source_overlap = sorted(set(tokens) & source_tokens)
    ascii_nonce_shape = bool(re.fullmatch(r"[a-z'.,!?\s]+", transcript.casefold()))
    gate_pass = bool(attempts) and all(item["accepted"] for item in attempts)
    return {
        "tokens": tokens,
        "token_count": len(tokens),
        "predicted_syllable_count": predicted_syllables,
        "source_word_overlap": source_overlap,
        "ascii_nonce_shape": ascii_nonce_shape,
        "gate_attempts": attempts,
        "word_count_pass": 3 <= len(tokens) <= 7,
        "syllable_count_pass": predicted_syllables == SOURCE_SYLLABLES,
        "source_overlap_pass": not source_overlap,
        "dictionary_homophone_adjacency_pass": gate_pass,
        "contract_pass": bool(
            3 <= len(tokens) <= 7
            and predicted_syllables == SOURCE_SYLLABLES
            and not source_overlap
            and ascii_nonce_shape
            and gate_pass
        ),
    }


def _attempt(
    client: Any,
    slot: ProbeSlot,
    anchor_base64: str,
    anchor_record: dict[str, Any],
    output: Path,
    attempt_number: int,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    started = time.monotonic()
    attempt: dict[str, Any] = {
        "attempt_number": attempt_number,
        "status": "failed_no_audio",
        "request_id": "",
        "usage": {},
        "estimated_cost_usd": 0.0,
        "retryable_external_failure": False,
    }
    partial = output.with_suffix(".partial.wav")
    try:
        completion = client.chat.completions.create(
            model=slot.model,
            modalities=["text", "audio"],
            audio={"voice": VOICE, "format": FORMAT},
            messages=_messages(slot, anchor_base64),
            store=False,
        )
        attempt["request_id"] = getattr(completion, "_request_id", None) or ""
        attempt["resolved_model"] = getattr(completion, "model", slot.model)
        attempt["usage"] = _usage_dict(completion)
        attempt["estimated_cost_usd"] = estimated_cost_usd(attempt["usage"])
        audio = completion.choices[0].message.audio
        if audio is None or not audio.data:
            raise ValueError(f"{slot.model} returned no audio payload")
        transcript = getattr(audio, "transcript", "") or ""
        output.parent.mkdir(parents=True, exist_ok=True)
        partial.write_bytes(base64.b64decode(audio.data, validate=True))
        partial.replace(output)
        timing = analyze_audio_timing(output, intended_syllables=SOURCE_SYLLABLES)
        prosody = analyze_prosody_fingerprint(output)
        integrity_reasons: list[str] = []
        if timing.sample_rate_hz != 24_000:
            integrity_reasons.append("unexpected_sample_rate")
        if not 0.25 <= timing.duration_s <= 45.0:
            integrity_reasons.append("duration_out_of_bounds")
        if timing.utterance_duration_s <= 0:
            integrity_reasons.append("no_detectable_utterance")
        if timing.clipped_fraction > 0.001:
            integrity_reasons.append("excessive_clipping")
        reference_timing, reference_prosody = _measurements(anchor_record)
        reference_match = compare_prosody(reference_timing, reference_prosody, timing, prosody)
        if slot.mode == "exact_grouped":
            transcript_audit: dict[str, Any] = asdict(check_transcript(GROUPED_NEUTRAL_SCRIPT, transcript))
            content_pass = bool(transcript_audit["exact_token_match"])
        else:
            transcript_audit = _content_audit(transcript)
            content_pass = bool(transcript_audit["contract_pass"])
        attempt["status"] = "audio_returned"
        return attempt, {
            "slot": asdict(slot),
            "status": "analyzed",
            "provider_transcript": transcript,
            "transcript_audit": transcript_audit,
            "content_control_pass": content_pass,
            "integrity_pass": not integrity_reasons,
            "integrity_reasons": integrity_reasons,
            "reference_match": reference_match,
            "audio_relative_path": str(output.relative_to(output.parents[1])),
            "audio_sha256": sha256_file(output),
            "timing": asdict(timing),
            "prosody": asdict(prosody),
            "request_id": attempt["request_id"],
            "resolved_model": attempt["resolved_model"],
            "usage": attempt["usage"],
            "estimated_cost_usd": attempt["estimated_cost_usd"],
        }
    except Exception as exc:
        partial.unlink(missing_ok=True)
        error_type, error_detail = _safe_error(exc)
        attempt.update({
            "error_type": error_type,
            "error_detail": error_detail,
            "retryable_external_failure": _retryable(exc),
        })
        return attempt, None
    finally:
        attempt["latency_ms"] = round((time.monotonic() - started) * 1000)


def _render_slot(
    client: Any,
    slot: ProbeSlot,
    anchor_base64: str,
    anchor_record: dict[str, Any],
    output: Path,
) -> dict[str, Any]:
    attempts: list[dict[str, Any]] = []
    final: dict[str, Any] | None = None
    for attempt_number in (1, 2):
        attempt, final = _attempt(
            client, slot, anchor_base64, anchor_record, output, attempt_number
        )
        attempts.append(attempt)
        if final is not None or not attempt["retryable_external_failure"]:
            break
        time.sleep(1)
    if final is None:
        final = {
            "slot": asdict(slot),
            "status": "external_failure_unresolved" if attempts[-1]["retryable_external_failure"] else "failed_no_audio",
            "reasons": [attempts[-1].get("error_type", "unknown_error")],
            "usage": {},
            "estimated_cost_usd": 0.0,
        }
    final["attempts"] = attempts
    return final


def _whisper_audit(record: dict[str, Any], run_dir: Path) -> None:
    if not record.get("audio_relative_path"):
        return
    model = Paths().whisper_cache / "large-v3-full"
    try:
        result = mlx_whisper.transcribe(
            str(run_dir / record["audio_relative_path"]),
            path_or_hf_repo=str(model),
            temperature=0,
            condition_on_previous_text=False,
            verbose=False,
        )
        transcript = str(result.get("text") or "").strip()
        record["local_whisper"] = {
            "detected_language": result.get("language"),
            "transcript": transcript,
            "tokens": canonical_tokens(transcript),
        }
    except Exception as exc:
        error_type, error_detail = _safe_error(exc)
        record["local_whisper"] = {"error_type": error_type, "error_detail": error_detail}
    finally:
        mx.clear_cache()
        gc.collect()


def run_probe(client: Any | None = None) -> dict[str, Any]:
    protocol = prepare_probe()
    if client is None:
        require_api_key()
        client = OpenAI(max_retries=0, timeout=60.0)
    anchor_path, control_path = _source_paths()
    anchor_base64 = base64.b64encode(anchor_path.read_bytes()).decode("ascii")
    anchor_record = _reference_record(anchor_path)
    run_dir = Paths().artifacts / "audio-delexicalization" / RUN_ID
    records: list[dict[str, Any]] = []
    for slot in build_manifest():
        receipt = run_dir / "slots" / f"{slot.request_order:02d}__{slot.slot_id}.json"
        output = run_dir / "audio" / f"{slot.request_order:02d}__{slot.slot_id}.wav"
        if receipt.is_file():
            saved = json.loads(receipt.read_text(encoding="utf-8"))
            if saved.get("receipt_status") != "complete":
                raise RuntimeError(f"Ambiguous interrupted probe slot: {slot.slot_id}")
            record = saved["record"]
        else:
            atomic_write_json(receipt, {"receipt_status": "started", "slot": asdict(slot)})
            record = _render_slot(client, slot, anchor_base64, anchor_record, output)
            atomic_write_json(receipt, {"receipt_status": "complete", "record": record})
        records.append(record)
        usage = summarize_usage([attempt for item in records for attempt in item.get("attempts", [])])
        if usage["estimated_cost_usd"] > MAX_ESTIMATED_COST_USD:
            raise RuntimeError("Audio-delexicalization probe cost cap exceeded")
        print(f"delex probe {slot.request_order}/3 {slot.slot_id}: {record['status']} cost=${usage['estimated_cost_usd']:.4f}", flush=True)

    for record in records:
        _whisper_audit(record, run_dir)
    attempts = [attempt for record in records for attempt in record.get("attempts", [])]
    usage = summarize_usage(attempts)
    summary = {
        "schema_version": 1,
        "status": "probe_complete",
        "run_id": RUN_ID,
        "protocol_sha256": protocol["protocol_sha256"],
        "logical_slots": 3,
        "audio_returned": sum(bool(record.get("audio_sha256")) for record in records),
        "content_control_passes": sum(bool(record.get("content_control_pass")) for record in records),
        "reference_match_passes": sum(bool((record.get("reference_match") or {}).get("eligible")) for record in records),
        "usage": usage,
    }
    atomic_write_json(run_dir / "records.json", records)
    atomic_write_json(run_dir / "summary.json", summary)
    _build_review(records, run_dir, control_path)
    return summary


def _build_review(records: list[dict[str, Any]], run_dir: Path, control_path: Path) -> None:
    rows = [{
        "blind_id": "clip-01",
        "audio": str(Path("../..") / control_path.relative_to(Paths().artifacts)),
        "key": "gpt-audio-1.5 exact grouped control",
    }]
    rows.extend({
        "blind_id": f"clip-{index + 2:02d}",
        "audio": record["audio_relative_path"],
        "key": record["slot"]["slot_id"],
    } for index, record in enumerate(records) if record.get("audio_relative_path"))
    import random
    random.Random(f"{RUN_ID}-blind-v1").shuffle(rows)
    atomic_write_json(run_dir / "blind-key.json", {row["blind_id"]: row["key"] for row in rows})
    public = [{"blind_id": row["blind_id"], "audio": row["audio"]} for row in rows]
    html = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Audio delexicalization blind review</title><style>body{font:16px/1.5 system-ui;max-width:800px;margin:auto;padding:24px;background:#f5f2e9;color:#17221c}.card{background:white;padding:18px;border:1px solid #d6d3c9;border-radius:16px;margin:16px 0}audio,textarea{width:100%}label{display:block;margin:8px 0}button{padding:10px 16px;border:0;border-radius:99px;background:#154f3e;color:white;font-weight:700}</style></head><body><h1>Blind neutral-carrier probe</h1><p>Judge whether each clip sounds like one natural but meaningless English-like utterance. Model and generation mode are hidden.</p><div id="cards"></div><button id="download">Download ratings.csv</button><script>const R=__ROWS__;const S=JSON.parse(localStorage.getItem('delex-ratings')||'{}');const save=(i,k,v)=>{S[i]??={};S[i][k]=v;localStorage.setItem('delex-ratings',JSON.stringify(S))};const sel=(i,k,v)=>`<select onchange="save('${i}','${k}',this.value)"><option value="">—</option>${v.map(x=>`<option ${S[i]?.[k]==x?'selected':''}>${x}</option>`).join('')}</select>`;document.getElementById('cards').innerHTML=R.map(r=>`<section class="card"><h2>${r.blind_id}</h2><audio controls src="${r.audio}"></audio><label>Naturalness ${sel(r.blind_id,'naturalness',['1','2','3','4','5'])}</label><label>List-like ${sel(r.blind_id,'list_like',['none','slight','dominant'])}</label><label>Meaning leakage ${sel(r.blind_id,'meaning_leak',['none','possible','clear'])}</label><label>Source-rhythm resemblance ${sel(r.blind_id,'source_rhythm',['yes','partly','no','uncertain'])}</label><label>Content failure/commentary ${sel(r.blind_id,'content_failure',['no','yes','uncertain'])}</label><textarea oninput="save('${r.blind_id}','notes',this.value)" placeholder="Notes">${S[r.blind_id]?.notes??''}</textarea></section>`).join('');document.getElementById('download').onclick=()=>{const F=['blind_id','naturalness','list_like','meaning_leak','source_rhythm','content_failure','notes'];const q=v=>`"${String(v??'').replaceAll('"','""')}"`;const b=new Blob([[F.join(','),...R.map(r=>F.map(k=>q(k==='blind_id'?r.blind_id:S[r.blind_id]?.[k])).join(','))].join('\n')+'\n'],{type:'text/csv'});const a=document.createElement('a');a.href=URL.createObjectURL(b);a.download='delex-ratings.csv';a.click()};</script></body></html>""".replace("__ROWS__", json.dumps(public))
    atomic_write_text(run_dir / "review.html", html)
