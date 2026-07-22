from __future__ import annotations

import gc
import importlib.metadata
import json
import os
import random
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx_whisper
import numpy as np

from .config import Paths, sha256_json, stable_json
from .kokoro_phoneme_spike import (
    CONFIG_FILE,
    KOKORO_VERSION,
    MODEL_FILE,
    MODEL_REPO,
    MODEL_REVISION,
    SAMPLE_RATE_HZ,
    VOICE_FILE,
    _audio_metrics,
    _content_audit,
    _verify_model_files,
    _write_pcm16,
)
from .kokoro_source_aligned import (
    CARRIER_LENS,
    CARRIER_NEUTRAL,
    RUN_ID as PARENT_RUN_ID,
    SOURCE_PHONEMES,
    SOURCE_SYLLABLES,
    SPEED,
    _classify_pair,
    _f0n,
    _input_ids,
    _predicted_alignment,
    _target_token_index,
    _text_features,
    protocol_record as parent_protocol_record,
    stress_span_alignment,
)
from .runtime_audio import canonical_tokens
from .same_take import WHISPER_MODEL
from .sentence_pair_v2_analysis import _measure
from .util import atomic_write_json, atomic_write_text, sha256_file


RUN_ID = "20260716-kokoro-latent-span-sweep-v3"
VARIANT_ORDER = (
    "target-only",
    "stress-plus-target",
    "target-word",
    "target-word-plus-boundaries",
    "full-contextual-state",
)


@dataclass(frozen=True)
class BatchSlot:
    request_order: int
    slot_id: str
    condition: str


def manifest() -> tuple[BatchSlot, ...]:
    return (
        BatchSlot(1, "batch-neutral", "neutral_carrier_state"),
        BatchSlot(2, "batch-neutral-identity", "exact_duplicate_neutral_state"),
        *(BatchSlot(index + 3, f"lens-{name}", name) for index, name in enumerate(VARIANT_ORDER)),
    )


def _parent_paths() -> tuple[Path, Path, Path]:
    parent = Paths().artifacts / "phoneme-renderer" / PARENT_RUN_ID
    return parent, parent / "stress-span-records.json", parent / "stress-span-summary.json"


def _fixed_anchor_geometry() -> dict[str, Any]:
    _, _, summary_path = _parent_paths()
    return json.loads(summary_path.read_text(encoding="utf-8"))["anchor_geometry"]


def protocol_record() -> dict[str, Any]:
    _verify_model_files(download=False)
    parent_dir, records_path, summary_path = _parent_paths()
    if not records_path.is_file() or not summary_path.is_file():
        raise RuntimeError("source-aligned v2 stress-span artifacts are missing")
    parent_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    geometry = parent_summary["anchor_geometry"]
    if not all(item["anchor_direction_consistency_pass"] for item in geometry["families"].values()):
        raise RuntimeError("target-specific Kokoro anchor direction is not consistent")
    payload = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "status": "zero_api_latent_span_sweep_frozen_before_rendering_and_listening",
        "question": (
            "What is the smallest neutral-to-lens contextual content span that realizes the same-Kokoro-voice "
            "/ae/->/eh/ gate while preserving one shared source alignment and neutral-carrier F0/noise state?"
        ),
        "parent": {
            "protocol_sha256": parent_protocol_record()["protocol_sha256"],
            "stress_span_records_sha256": sha256_file(records_path),
            "stress_span_summary_sha256": sha256_file(summary_path),
            "audio_hashes": {
                path.name: sha256_file(path)
                for path in sorted((parent_dir / "audio").glob("*.wav"))
            },
        },
        "renderer": {
            "package": "kokoro",
            "package_version": KOKORO_VERSION,
            "model_repo": MODEL_REPO,
            "model_revision": MODEL_REVISION,
            "device": "cpu_single_thread_mkldnn_disabled",
            "voice": "af_heart",
            "sample_rate_hz": SAMPLE_RATE_HZ,
            "api_calls": 0,
        },
        "fixed_inputs": {
            "source_phonemes": SOURCE_PHONEMES,
            "neutral_phonemes": CARRIER_NEUTRAL,
            "lens_phonemes": CARRIER_LENS,
            "source_alignment": "newly recomputed from the unchanged source input and frozen model",
            "prosody_state": "neutral-carrier F0/noise predicted over the source alignment",
            "input_delta": "one raw phoneme: /ae/ becomes /eh/",
        },
        "manifest": [slot.__dict__ for slot in manifest()],
        "latent_span_order": {
            "target-only": "replace only the target vowel text-encoder column",
            "stress-plus-target": "replace the immediately preceding stress and target vowel columns",
            "target-word": "replace onset, stress, vowel, and coda columns inside target word eight",
            "target-word-plus-boundaries": "replace target word eight plus its immediately surrounding space columns",
            "full-contextual-state": "use every column returned by the lens text encoder; raw input still differs by one phoneme",
        },
        "batch_contract": {
            "single_decoder_call": "neutral, duplicate neutral, and all five lens states are decoded as one batch",
            "shared": "alignment, duration, F0, noise, voice style, model state, and output length",
            "identity": "the two duplicate neutral batch rows must be bit-identical",
            "locality_report": "report pair-difference energy within the stress+vowel interval plus/minus 150 ms and outside it",
        },
        "acoustic_gate": {
            "geometry": geometry,
            "measurement": "frozen stress-plus-vowel span; middle 50 percent; standalone Praat at 5500/5750/6000 Hz",
            "per_family": (
                "neutral closer to /ae/ centroid, lens closer to /eh/ centroid, cosine >= 0.5, and magnitude >= "
                "the frozen target-specific threshold"
            ),
            "selection": "select the first passing span in VARIANT_ORDER; never select the largest effect",
            "scope": (
                "target-specific /ae/ and /eh/ anchors are consistent across three shells; the failed six-vowel "
                "global topology remains disclosed and no broader Kokoro vowel-system claim is made"
            ),
        },
        "semantic_opacity": (
            "all planned phone and adjacency gates remain inherited; local Whisper source overlap is hard, while "
            "coherent-looking ASR text is a manual stable-meaning warning rather than an automatic rejection"
        ),
        "manual_gate": (
            "only an acoustically passing selected pair advances; blind review requires both sides >=4/5 naturalness, "
            "sentence-like delivery, no stable recoverable meaning, clear correctly directed target change, and "
            "manageable unrelated interference"
        ),
        "stopping_rule": (
            "Run exactly one seven-row decoder batch. If no span passes, close this localization method. No post-listening "
            "span, threshold, carrier, speed, or state change belongs to this run."
        ),
    }
    return {**payload, "protocol_sha256": sha256_json(payload)}


def prepare() -> dict[str, Any]:
    protocol = protocol_record()
    path = Paths().artifacts / "phoneme-renderer" / RUN_ID / "protocol.json"
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if stable_json(existing) != stable_json(protocol):
            raise RuntimeError("existing latent-span protocol differs from freeze")
    else:
        atomic_write_json(path, protocol)
    return protocol


def _span_indices(model: Any, neutral: str, target_with_boundary: int) -> dict[str, tuple[int, ...]]:
    filtered = [symbol for symbol in neutral if model.vocab.get(symbol) is not None]
    target_without_boundary = target_with_boundary - 1
    if filtered[target_without_boundary] != "æ" or filtered[target_without_boundary - 1] not in {"ˈ", "ˌ"}:
        raise RuntimeError("unexpected target neighborhood")
    left_space = max(index for index in range(target_without_boundary) if filtered[index] == " ")
    right_space = min(index for index in range(target_without_boundary + 1, len(filtered)) if filtered[index] == " ")
    word = tuple(index + 1 for index in range(left_space + 1, right_space))
    word_and_boundaries = tuple(index + 1 for index in range(left_space, right_space + 1))
    return {
        "target-only": (target_with_boundary,),
        "stress-plus-target": (target_with_boundary - 1, target_with_boundary),
        "target-word": word,
        "target-word-plus-boundaries": word_and_boundaries,
        "full-contextual-state": tuple(range(0, len(filtered) + 2)),
    }


def _variant_states(neutral_state: Any, lens_state: Any, spans: dict[str, tuple[int, ...]]) -> list[Any]:
    states = [neutral_state, neutral_state.clone()]
    for name in VARIANT_ORDER:
        state = neutral_state.clone()
        columns = spans[name]
        state[:, :, list(columns)] = lens_state[:, :, list(columns)]
        states.append(state)
    return states


def _decode_batch(model: Any, states: list[Any], alignment: Any, f0: Any, noise: Any, ref_s: Any, torch: Any) -> Any:
    count = len(states)
    text = torch.cat(states, dim=0)
    aligned = alignment.expand(count, -1, -1)
    asr = torch.bmm(text, aligned)
    f0_batch = f0.expand(count, -1)
    noise_batch = noise.expand(count, -1)
    if ref_s.ndim != 2 or ref_s.shape[-1] < 128:
        raise RuntimeError("Kokoro reference style does not contain the decoder's 128-value acoustic half")
    style_batch = ref_s[:, :128].expand(count, -1)
    if style_batch.shape != (count, 128):
        raise RuntimeError("Kokoro decoder style batch has the wrong shape")
    audio = model.decoder(asr, f0_batch, noise_batch, style_batch).detach().cpu()
    return audio.reshape(count, -1)


def _measure_stress_target(path: Path, model: Any, phonemes: str, durations: list[int], symbol: str) -> dict[str, Any]:
    with wave.open(str(path), "rb") as handle:
        sample_count = handle.getnframes()
    alignment = stress_span_alignment(
        vocab=model.vocab,
        phonemes=phonemes,
        pred_dur=durations,
        sample_count=sample_count,
        target_symbol=symbol,
    )
    interval = {"start_s": alignment["start_s"], "end_s": alignment["end_s"]}
    return {
        "alignment": alignment,
        "measurements": {str(ceiling): _measure(path, interval, ceiling) for ceiling in (5500, 5750, 6000)},
    }


def _whisper(record: dict[str, Any], run_dir: Path) -> None:
    try:
        result = mlx_whisper.transcribe(
            str(run_dir / record["audio_relative_path"]),
            path_or_hf_repo=str(WHISPER_MODEL),
            language="en",
            temperature=0,
            condition_on_previous_text=False,
            verbose=False,
        )
        transcript = str(result.get("text") or "").strip()
        audit = _content_audit(transcript)
        record["local_whisper"] = {
            "transcript": transcript,
            "tokens": canonical_tokens(transcript),
            "source_overlap_pass": audit["source_overlap_pass"],
            "audit": audit,
        }
    except Exception as exc:
        record["local_whisper"] = {"error": f"{type(exc).__name__}: {str(exc)[:500]}"}
    finally:
        mx.clear_cache()
        gc.collect()


def _pcm(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as handle:
        return np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2").astype(np.float64)


def _difference_report(neutral: Path, other: Path, alignment: dict[str, Any]) -> dict[str, Any]:
    left = _pcm(neutral)
    right = _pcm(other)
    if left.shape != right.shape:
        return {"sample_count_equal": False}
    delta = right - left
    pad = round(0.150 * SAMPLE_RATE_HZ)
    start = max(0, int(alignment["start_sample"]) - pad)
    end = min(len(delta), int(alignment["end_sample_exclusive"]) + pad)
    inside = delta[start:end]
    outside = np.concatenate((delta[:start], delta[end:]))
    total_energy = float(np.dot(delta, delta))
    inside_energy = float(np.dot(inside, inside))
    return {
        "sample_count_equal": True,
        "maximum_absolute_pcm_delta": float(np.max(np.abs(delta), initial=0)),
        "mean_absolute_pcm_delta": float(np.mean(np.abs(delta))),
        "inside_window_start_s": start / SAMPLE_RATE_HZ,
        "inside_window_end_s": end / SAMPLE_RATE_HZ,
        "inside_difference_energy_fraction": inside_energy / total_energy if total_energy else 1.0,
        "outside_rms_pcm": float(np.sqrt(np.mean(outside**2))) if outside.size else 0.0,
    }


def _review(records: list[dict[str, Any]], selected: str | None, run_dir: Path) -> None:
    include = {"batch-neutral", "batch-neutral-identity"}
    if selected:
        include.add(f"lens-{selected}")
    else:
        include.update(f"lens-{name}" for name in VARIANT_ORDER)
    rows = [
        {"blind_id": f"clip-{index + 1:02d}", "audio": record["audio_relative_path"], "key": record["slot_id"]}
        for index, record in enumerate(records)
        if record["slot_id"] in include
    ]
    random.Random(f"{RUN_ID}-blind").shuffle(rows)
    atomic_write_json(run_dir / "blind-key.json", {row["blind_id"]: row["key"] for row in rows})
    public = [{"blind_id": row["blind_id"], "audio": row["audio"]} for row in rows]
    html = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Latent-span blind review</title><style>body{font:17px/1.5 system-ui;max-width:800px;margin:auto;padding:24px;background:#f5f2e9;color:#17221c}.card{background:white;padding:18px;border:1px solid #d6d3c9;border-radius:16px;margin:16px 0}audio,textarea{width:100%}label{display:block;margin:9px 0}button{padding:11px 17px;border:0;border-radius:99px;background:#154f3e;color:white;font-weight:700}</style></head><body><h1>Blind shared-prosody review</h1><p>The files use the same timing and prosody state. Judge whether each sounds like one natural sentence-like utterance and whether any stable English meaning is actually recoverable.</p><div id="cards"></div><button id="download">Download ratings.csv</button><script>const R=__ROWS__;const K='kokoro-latent-span-v3';const S=JSON.parse(localStorage.getItem(K)||'{}');const save=(i,k,v)=>{S[i]??={};S[i][k]=v;localStorage.setItem(K,JSON.stringify(S))};const sel=(i,k,v)=>`<select onchange="save('${i}','${k}',this.value)"><option value="">—</option>${v.map(x=>`<option ${S[i]?.[k]==x?'selected':''}>${x}</option>`).join('')}</select>`;document.getElementById('cards').innerHTML=R.map(r=>`<section class="card"><h2>${r.blind_id}</h2><audio controls src="${r.audio}"></audio><label>Naturalness ${sel(r.blind_id,'naturalness',['1','2','3','4','5'])}</label><label>Delivery ${sel(r.blind_id,'delivery',['sentence-like','slightly list-like','dominantly list-like','other'])}</label><label>Stable recoverable meaning ${sel(r.blind_id,'meaning',['none','isolated possible word','coherent phrase','clear source sentence'])}</label><label>Artifact ${sel(r.blind_id,'artifact',['none','minor','major','uncertain'])}</label><textarea oninput="save('${r.blind_id}','notes',this.value)" placeholder="Notes">${S[r.blind_id]?.notes??''}</textarea></section>`).join('');document.getElementById('download').onclick=()=>{const F=['blind_id','naturalness','delivery','meaning','artifact','notes'];const q=v=>`"${String(v??'').replaceAll('"','""')}"`;const b=new Blob([[F.join(','),...R.map(r=>F.map(k=>q(k==='blind_id'?r.blind_id:S[r.blind_id]?.[k])).join(','))].join('\n')+'\n'],{type:'text/csv'});const a=document.createElement('a');a.href=URL.createObjectURL(b);a.download='latent-span-v3-ratings.csv';a.click()};</script></body></html>""".replace("__ROWS__", json.dumps(public))
    atomic_write_text(run_dir / "review.html", html)


def run() -> dict[str, Any]:
    protocol = prepare()
    if importlib.metadata.version("kokoro") != KOKORO_VERSION:
        raise RuntimeError("Kokoro package version differs from freeze")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    files = _verify_model_files(download=False)
    import torch
    from kokoro import KModel

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.backends.mkldnn.enabled = False
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    model = KModel(repo_id=MODEL_REPO, config=str(files[CONFIG_FILE]), model=str(files[MODEL_FILE])).to("cpu").eval()
    voice_pack = torch.load(files[VOICE_FILE], map_location="cpu", weights_only=True)
    ref_s = voice_pack[len(SOURCE_PHONEMES) - 1]
    if ref_s.ndim == 1:
        ref_s = ref_s.unsqueeze(0)
    with torch.no_grad():
        source_ids = _input_ids(model, SOURCE_PHONEMES, torch)
        source_features = _text_features(model, source_ids, ref_s, torch)
        durations, alignment = _predicted_alignment(model, source_features, SPEED, torch)
        neutral_ids = _input_ids(model, CARRIER_NEUTRAL, torch)
        lens_ids = _input_ids(model, CARRIER_LENS, torch)
        neutral_features = _text_features(model, neutral_ids, ref_s, torch)
        lens_features = _text_features(model, lens_ids, ref_s, torch)
        carrier_f0, carrier_noise = _f0n(model, neutral_features, alignment)
        target_index = _target_token_index(model, CARRIER_NEUTRAL, "æ")
        if target_index != _target_token_index(model, CARRIER_LENS, "ɛ"):
            raise RuntimeError("neutral/lens target indices differ")
        spans = _span_indices(model, CARRIER_NEUTRAL, target_index)
        states = _variant_states(neutral_features["t_en"], lens_features["t_en"], spans)
        audio_batch = _decode_batch(model, states, alignment, carrier_f0, carrier_noise, ref_s, torch)
    durations_list = [int(value) for value in durations.detach().cpu().tolist()]
    run_dir = Paths().artifacts / "phoneme-renderer" / RUN_ID
    records: list[dict[str, Any]] = []
    for slot, audio in zip(manifest(), audio_batch, strict=True):
        path = run_dir / "audio" / f"{slot.request_order:02d}__{slot.slot_id}.wav"
        _write_pcm16(path, audio.numpy())
        symbol = "æ" if slot.request_order <= 2 else "ɛ"
        phonemes = CARRIER_NEUTRAL if slot.request_order <= 2 else CARRIER_LENS
        record = {
            "request_order": slot.request_order,
            "slot_id": slot.slot_id,
            "condition": slot.condition,
            "phonemes": phonemes,
            "replaced_columns_with_boundary": [] if slot.request_order <= 2 else list(spans[slot.condition]),
            "replaced_full_lens_delta_energy_fraction": (
                0.0
                if slot.request_order <= 2
                else float(
                    torch.sum((lens_features["t_en"][:, :, list(spans[slot.condition])] - neutral_features["t_en"][:, :, list(spans[slot.condition])]) ** 2).item()
                    / torch.sum((lens_features["t_en"] - neutral_features["t_en"]) ** 2).item()
                )
            ),
            "audio_relative_path": str(path.relative_to(run_dir)),
            "audio_sha256": sha256_file(path),
            "predicted_durations": durations_list,
            **_audio_metrics(path, SOURCE_SYLLABLES),
        }
        record["target"] = _measure_stress_target(path, model, phonemes, durations_list, symbol)
        records.append(record)
        print(f"latent sweep {slot.request_order}/7 {slot.slot_id}: {record['timing']['utterance_duration_s']:.3f}s", flush=True)
    for record in records:
        _whisper(record, run_dir)
    geometry = _fixed_anchor_geometry()
    neutral = records[0]
    pair_results: dict[str, Any] = {}
    for record in records[2:]:
        result = _classify_pair(neutral, record, geometry)
        result["difference_localization"] = _difference_report(
            run_dir / neutral["audio_relative_path"],
            run_dir / record["audio_relative_path"],
            neutral["target"]["alignment"],
        )
        pair_results[record["condition"]] = result
    identity = _difference_report(
        run_dir / records[0]["audio_relative_path"],
        run_dir / records[1]["audio_relative_path"],
        neutral["target"]["alignment"],
    )
    identity["bit_identical"] = records[0]["audio_sha256"] == records[1]["audio_sha256"]
    selected = next((name for name in VARIANT_ORDER if pair_results[name]["pass"]), None)
    summary = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "status": "automatic_analysis_complete_manual_review_pending" if selected else "no_latent_span_passed",
        "protocol_sha256": protocol["protocol_sha256"],
        "api_calls_made": 0,
        "decoder_batch_rows": len(records),
        "batch_identity": identity,
        "pair_results": pair_results,
        "selected_smallest_passing_span": selected,
        "automatic_candidate": bool(identity["bit_identical"] and selected),
    }
    atomic_write_json(run_dir / "records.json", records)
    atomic_write_json(run_dir / "summary.json", summary)
    _review(records, selected, run_dir)
    return summary
