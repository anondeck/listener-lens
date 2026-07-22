from __future__ import annotations

import gc
import importlib.metadata
import json
import random
import wave
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx_whisper
import numpy as np
from huggingface_hub import hf_hub_download

from .audio_delexicalization_probe import _content_audit
from .carrier_architecture_tournament import REFERENCE_GATE, SOURCE_SYLLABLES, SOURCE_TEXT, compare_prosody
from .config import Paths, sha256_json, stable_json
from .gates import CandidateGate
from .runtime_audio import (
    AudioTiming,
    PauseInterval,
    ProsodyFingerprint,
    analyze_audio_timing,
    analyze_prosody_fingerprint,
    canonical_tokens,
)
from .same_take import WHISPER_MODEL
from .sentence_pair_v2 import ANCHOR_GATE
from .sentence_pair_v2_analysis import CEILINGS, _measure
from .util import atomic_write_json, atomic_write_text, sha256_file


RUN_ID = "20260716-kokoro-phoneme-spike-v1"
MODEL_REPO = "hexgrad/Kokoro-82M"
MODEL_REVISION = "f3ff3571791e39611d31c381e3a41a3af07b4987"
MODEL_FILE = "kokoro-v1_0.pth"
CONFIG_FILE = "config.json"
VOICE_FILE = "voices/af_heart.pt"
VOICE = "af_heart"
SAMPLE_RATE_HZ = 24_000
SPEED = 1.0
KOKORO_VERSION = "0.9.4"
MODEL_HASHES = {
    CONFIG_FILE: "5abb01e2403b072bf03d04fde160443e209d7a0dad49a423be15196b9b43c17f",
    MODEL_FILE: "496dba118d1a58f5f3db2efc88dbdc216e0483fc89fe6e47ee1f2c53f18ad1e4",
    VOICE_FILE: "0ab5709b8ffab19bfd849cd11d98f75b60af7733253ad0d67b12382a102cb4ff",
}

SOURCE_PHONEMES = "wˌʌt ɐ ɡɹˈAt dˈA ɪt ɪz tə kˈɛʧ sˌʌm sˈʌn."
ALL_STRESSED_NEUTRAL = "ɹˈOk bɹˈɪ pɹˈOk dɹˈAɹ nˈɪv sˈɪv ɹˈɪn bˈævd fˈɪv vɹˈil."
UNSTRESSED_WEAK_NEUTRAL = "ɹˈOk bɹɪ pɹˈOk dɹˈAɹ nɪv sɪv ɹɪn bˈævd fɪv vɹˈil."
SCHWA_WEAK_NEUTRAL = "ɹˈOk bɹə pɹˈOk dɹˈAɹ nəv səv ɹən bˈævd fəv vɹˈil."
SCHWA_WEAK_LENS = "ɹˈOk bɹə pɹˈOk dɹˈAɹ nəv səv ɹən bˈɛvd fəv vɹˈil."

SURFACE_WORDS_NEUTRAL = (
    "rohk", "brih", "prohk", "drayr", "nihv", "sihv", "rihn", "bavd", "fihv", "vreel",
)
SURFACE_WORDS_LENS = (*SURFACE_WORDS_NEUTRAL[:7], "behvd", *SURFACE_WORDS_NEUTRAL[8:])
GATE_IPA_NEUTRAL = (
    "ɹoʊk", "bɹə", "pɹoʊk", "dɹeɪɹ", "nəv", "səv", "ɹən", "bævd", "fəv", "vɹiːl",
)
GATE_IPA_LENS = (*GATE_IPA_NEUTRAL[:7], "bɛvd", *GATE_IPA_NEUTRAL[8:])


@dataclass(frozen=True)
class SpikeSlot:
    request_order: int
    slot_id: str
    condition: str
    phonemes: str
    target_symbol: str | None


def build_manifest() -> tuple[SpikeSlot, ...]:
    return (
        SpikeSlot(1, "kokoro-source-anchor", "meaningful_source_anchor", SOURCE_PHONEMES, None),
        SpikeSlot(2, "kokoro-all-stressed-neutral", "all_stressed_neutral_control", ALL_STRESSED_NEUTRAL, "æ"),
        SpikeSlot(3, "kokoro-unstressed-weak-neutral", "unstressed_weak_neutral", UNSTRESSED_WEAK_NEUTRAL, "æ"),
        SpikeSlot(4, "kokoro-schwa-weak-neutral", "schwa_weak_neutral", SCHWA_WEAK_NEUTRAL, "æ"),
        SpikeSlot(5, "kokoro-schwa-weak-neutral-repeat", "schwa_weak_neutral_identity_repeat", SCHWA_WEAK_NEUTRAL, "æ"),
        SpikeSlot(6, "kokoro-schwa-weak-lens", "schwa_weak_lens", SCHWA_WEAK_LENS, "ɛ"),
    )


def _verify_model_files(download: bool = True) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for filename, expected in MODEL_HASHES.items():
        path = Path(hf_hub_download(repo_id=MODEL_REPO, revision=MODEL_REVISION, filename=filename, local_files_only=not download))
        if sha256_file(path) != expected:
            raise RuntimeError(f"Kokoro artifact hash mismatch: {filename}")
        paths[filename] = path
    return paths


def _phone_gate_report() -> dict[str, Any]:
    gate = CandidateGate()
    conditions: dict[str, Any] = {}
    for name, words, ipas in (
        ("neutral", SURFACE_WORDS_NEUTRAL, GATE_IPA_NEUTRAL),
        ("lens", SURFACE_WORDS_LENS, GATE_IPA_LENS),
    ):
        isolated = [
            {
                "surface": surface,
                "ipa": ipa,
                "written_match": gate.text_match(surface),
                "phone_match": gate.phone_match("en", ipa),
            }
            for surface, ipa in zip(words, ipas, strict=True)
        ]
        adjacency = [
            {
                "left": words[index],
                "right": words[index + 1],
                "written_match": gate.text_match(words[index] + words[index + 1]),
                "phone_match": gate.phone_match("en", ipas[index] + ipas[index + 1]),
            }
            for index in range(len(words) - 1)
        ]
        conditions[name] = {
            "isolated": isolated,
            "adjacency": adjacency,
            "pass": not any(
                item["written_match"] or item["phone_match"]
                for item in (*isolated, *adjacency)
            ),
        }
    return conditions


def _one_symbol_difference(left: str, right: str) -> dict[str, Any]:
    if len(left) != len(right):
        return {"pass": False, "reason": "length_mismatch"}
    differences = [
        {"index": index, "neutral": a, "lens": b}
        for index, (a, b) in enumerate(zip(left, right, strict=True))
        if a != b
    ]
    return {
        "pass": differences == [{"index": SCHWA_WEAK_NEUTRAL.index("æ"), "neutral": "æ", "lens": "ɛ"}],
        "differences": differences,
    }


def protocol_record() -> dict[str, Any]:
    _verify_model_files(download=False)
    gates = _phone_gate_report()
    if not all(item["pass"] for item in gates.values()):
        raise RuntimeError("Kokoro spike phone plan is not gate-clean")
    delta = _one_symbol_difference(SCHWA_WEAK_NEUTRAL, SCHWA_WEAK_LENS)
    if not delta["pass"]:
        raise RuntimeError("Kokoro neutral/lens plan differs by more than the target vowel")
    payload = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "status": "zero_api_phoneme_renderer_spike_frozen_before_rendering_and_listening",
        "question": (
            "Can a deterministic phoneme-native neural renderer produce a natural, non-list-like, "
            "semantically opaque carrier while realizing a one-symbol /ae/->/eh/ contrast?"
        ),
        "source_text": SOURCE_TEXT,
        "source_syllables": SOURCE_SYLLABLES,
        "renderer": {
            "package": "kokoro",
            "package_version": KOKORO_VERSION,
            "package_license": "Apache-2.0",
            "model_repo": MODEL_REPO,
            "model_revision": MODEL_REVISION,
            "model_license": "Apache-2.0",
            "model_hashes": MODEL_HASHES,
            "voice": VOICE,
            "sample_rate_hz": SAMPLE_RATE_HZ,
            "speed": SPEED,
            "device": "cpu",
            "direct_interface": "KModel.forward(raw_phoneme_string, voice_style, speed)",
            "api_calls": 0,
        },
        "manifest": [asdict(slot) for slot in build_manifest()],
        "carrier_plan": {
            "source_words": SOURCE_TEXT,
            "neutral_surfaces": list(SURFACE_WORDS_NEUTRAL),
            "lens_surfaces": list(SURFACE_WORDS_LENS),
            "neutral_gate_ipa": list(GATE_IPA_NEUTRAL),
            "lens_gate_ipa": list(GATE_IPA_LENS),
            "gate_report": gates,
            "neutral_lens_symbol_delta": delta,
            "weak_source_positions_one_based": [2, 5, 6, 7, 9],
            "rule_source_position_one_based": 8,
        },
        "conditions": {
            "source_anchor": "Kokoro's frozen American-English G2P phoneme plan for the meaningful sentence",
            "all_stressed": "all ten nonce carrier words retain lexical stress",
            "unstressed_weak": "weak positions lose stress but retain their /ih/ segment",
            "schwa_weak": "weak positions lose stress and reduce to /schwa/; content words retain stress",
            "identity_repeat": "the schwa-weak neutral input is rendered a second time without any changed parameter",
            "lens": "the schwa-weak neutral input with only target /ae/ replaced by /eh/",
        },
        "automatic_gates": {
            "audio_integrity": "mono 24-kHz PCM16, no clipping above 0.001, detectable utterance",
            "identity": "neutral and identity-repeat WAV hashes compared; sample difference reported if unequal",
            "semantic_opacity": "pinned local Whisper transcript receives written, homophone, adjacency, and source-overlap audit",
            "target_alignment": (
                "map the unique target phoneme token through KModel's returned per-token duration vector; "
                "require a stable integer samples-per-duration-frame ratio"
            ),
            "target_acoustics": (
                "unchanged standalone-Praat three-ceiling family; neutral closer to Marin /ae/, "
                "lens closer to Marin /eh/, plus frozen direction and magnitude thresholds"
            ),
            "prosody": "duration, pauses, F0/voicing, and pitch/energy contour compared with the Kokoro source anchor",
        },
        "manual_gate": {
            "presentation": "blind randomized clips; condition, phoneme string, and target spelling hidden",
            "product_candidate": (
                "both schwa-weak sides at least 4/5 naturalness, no dominant list delivery, "
                "no clear meaning leakage, clear correctly directed target difference, and manageable unrelated interference"
            ),
            "role": "artifact and product-QC evidence, not Brazilian-Portuguese population evidence",
        },
        "interpretation": (
            "A pass would establish a controlled renderer candidate for the existing one-rule typed transform. "
            "It would not prove Brazilian-Portuguese profile fit or arbitrary-language coverage. A failure "
            "redirects the renderer layer without changing the frozen isolated /ae/->/eh/ evidence."
        ),
        "official_sources": {
            "model_card": "https://huggingface.co/hexgrad/Kokoro-82M/blob/main/README.md",
            "package": "https://pypi.org/project/kokoro/",
            "architecture": "https://github.com/hexgrad/kokoro",
        },
    }
    return {**payload, "protocol_sha256": sha256_json(payload)}


def prepare_spike() -> dict[str, Any]:
    protocol = protocol_record()
    path = Paths().artifacts / "phoneme-renderer" / RUN_ID / "protocol.json"
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if stable_json(existing) != stable_json(protocol):
            raise RuntimeError("Existing Kokoro phoneme-spike protocol differs from the freeze")
    else:
        atomic_write_json(path, protocol)
    return protocol


def _write_pcm16(path: Path, audio: np.ndarray) -> None:
    values = np.asarray(audio, dtype=np.float64).reshape(-1)
    pcm = np.rint(np.clip(values, -1.0, 1.0) * 32767.0).astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".partial.wav")
    with wave.open(str(partial), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE_HZ)
        handle.writeframes(pcm.tobytes())
    partial.replace(path)


def duration_alignment(
    *,
    vocab: dict[str, int],
    phonemes: str,
    pred_dur: list[int],
    sample_count: int,
    target_symbol: str,
) -> dict[str, Any]:
    filtered = [(raw_index, symbol) for raw_index, symbol in enumerate(phonemes) if vocab.get(symbol) is not None]
    if len(pred_dur) != len(filtered) + 2:
        raise RuntimeError(f"duration/token mismatch: {len(pred_dur)} != {len(filtered)} + 2")
    targets = [index for index, (_, symbol) in enumerate(filtered) if symbol == target_symbol]
    if len(targets) != 1:
        raise RuntimeError(f"expected one target symbol {target_symbol!r}, got {len(targets)}")
    duration_index = targets[0] + 1
    total_frames = sum(pred_dur)
    samples_per_frame = sample_count / total_frames
    if abs(samples_per_frame - round(samples_per_frame)) > 1e-9:
        raise RuntimeError("decoder samples do not have an integer duration-frame ratio")
    start_sample = round(sum(pred_dur[:duration_index]) * samples_per_frame)
    end_sample = round(sum(pred_dur[: duration_index + 1]) * samples_per_frame)
    return {
        "raw_character_index": filtered[targets[0]][0],
        "filtered_token_index_zero_based": targets[0],
        "duration_index_with_boundary": duration_index,
        "target_symbol": target_symbol,
        "target_duration_frames": pred_dur[duration_index],
        "total_duration_frames": total_frames,
        "samples_per_duration_frame": samples_per_frame,
        "start_sample": start_sample,
        "end_sample_exclusive": end_sample,
        "start_s": start_sample / SAMPLE_RATE_HZ,
        "end_s": end_sample / SAMPLE_RATE_HZ,
        "sample_count": end_sample - start_sample,
    }


def _audio_metrics(path: Path, intended_syllables: int) -> dict[str, Any]:
    timing = analyze_audio_timing(path, intended_syllables=intended_syllables)
    prosody = analyze_prosody_fingerprint(path)
    reasons: list[str] = []
    if timing.sample_rate_hz != SAMPLE_RATE_HZ:
        reasons.append("unexpected_sample_rate")
    if timing.utterance_duration_s <= 0:
        reasons.append("no_detectable_utterance")
    if timing.clipped_fraction > 0.001:
        reasons.append("excessive_clipping")
    return {
        "integrity_pass": not reasons,
        "integrity_reasons": reasons,
        "timing": asdict(timing),
        "prosody": asdict(prosody),
    }


def _measure_target(path: Path, alignment: dict[str, Any]) -> dict[str, Any]:
    interval = {"start_s": alignment["start_s"], "end_s": alignment["end_s"]}
    measurements = {str(ceiling): _measure(path, interval, ceiling) for ceiling in CEILINGS}
    return {"alignment": alignment, "measurements": measurements}


def _reference_objects(record: dict[str, Any]) -> tuple[AudioTiming, ProsodyFingerprint]:
    timing = dict(record["timing"])
    timing["interior_pauses"] = tuple(PauseInterval(**item) for item in timing["interior_pauses"])
    prosody = dict(record["prosody"])
    prosody["energy_contour_db"] = tuple(prosody["energy_contour_db"])
    prosody["pitch_contour_semitones"] = tuple(prosody["pitch_contour_semitones"])
    return AudioTiming(**timing), ProsodyFingerprint(**prosody)


def _whisper_audit(record: dict[str, Any], run_dir: Path) -> None:
    if record["condition"] == "meaningful_source_anchor":
        return
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
        leak = _content_audit(transcript)
        record["local_whisper"] = {
            "detected_language": result.get("language"),
            "transcript": transcript,
            "tokens": canonical_tokens(transcript),
            "leak_audit": leak,
            "audio_leak_screen_pass": bool(
                leak["dictionary_homophone_adjacency_pass"] and leak["source_overlap_pass"]
            ),
        }
    except Exception as exc:
        record["local_whisper"] = {"error": f"{type(exc).__name__}: {str(exc)[:500]}"}
    finally:
        mx.clear_cache()
        gc.collect()


def _category_measurement(target: dict[str, Any], expected: str) -> dict[str, Any]:
    families: dict[str, Any] = {}
    for ceiling in CEILINGS:
        key = str(ceiling)
        measurement = target["measurements"][key]
        point = np.asarray([measurement["f1_bark"], measurement["f2_bark"]])
        anchor = ANCHOR_GATE["families"][key]
        source = np.asarray(anchor["source_centroid_bark"])
        lens = np.asarray(anchor["target_centroid_bark"])
        source_distance = float(np.linalg.norm(point - source))
        lens_distance = float(np.linalg.norm(point - lens))
        category_pass = source_distance < lens_distance if expected == "ae" else lens_distance < source_distance
        families[key] = {
            "source_distance_bark": source_distance,
            "lens_distance_bark": lens_distance,
            "expected_category": expected,
            "category_pass": bool(measurement["plausibility_pass"] and category_pass),
        }
    return {"families": families, "category_pass": all(item["category_pass"] for item in families.values())}


def _pair_classification(neutral: dict[str, Any], lens: dict[str, Any]) -> dict[str, Any]:
    from .runtime_pair_diagnostic import classify_points

    return classify_points(
        {
            str(ceiling): (
                neutral["measurements"][str(ceiling)]["f1_bark"],
                neutral["measurements"][str(ceiling)]["f2_bark"],
            )
            for ceiling in CEILINGS
        },
        {
            str(ceiling): (
                lens["measurements"][str(ceiling)]["f1_bark"],
                lens["measurements"][str(ceiling)]["f2_bark"],
            )
            for ceiling in CEILINGS
        },
    )


def _identity_comparison(left: Path, right: Path) -> dict[str, Any]:
    with wave.open(str(left), "rb") as a, wave.open(str(right), "rb") as b:
        left_pcm = np.frombuffer(a.readframes(a.getnframes()), dtype="<i2").astype(np.int32)
        right_pcm = np.frombuffer(b.readframes(b.getnframes()), dtype="<i2").astype(np.int32)
    if left_pcm.shape != right_pcm.shape:
        return {"bit_identical": False, "sample_count_equal": False, "maximum_absolute_sample_delta": None}
    delta = np.abs(left_pcm - right_pcm)
    return {
        "bit_identical": bool(np.array_equal(left_pcm, right_pcm)),
        "sample_count_equal": True,
        "maximum_absolute_sample_delta": int(delta.max(initial=0)),
        "mean_absolute_sample_delta": float(delta.mean()) if delta.size else 0.0,
    }


def run_spike() -> dict[str, Any]:
    protocol = prepare_spike()
    if importlib.metadata.version("kokoro") != KOKORO_VERSION:
        raise RuntimeError("Kokoro package version differs from the freeze")
    files = _verify_model_files(download=False)
    import torch
    from kokoro import KModel

    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    model = KModel(
        repo_id=MODEL_REPO,
        config=str(files[CONFIG_FILE]),
        model=str(files[MODEL_FILE]),
    ).to("cpu").eval()
    voice_pack = torch.load(files[VOICE_FILE], map_location="cpu", weights_only=True)
    run_dir = Paths().artifacts / "phoneme-renderer" / RUN_ID
    records: list[dict[str, Any]] = []
    for slot in build_manifest():
        output_path = run_dir / "audio" / f"{slot.request_order:02d}__{slot.slot_id}.wav"
        with torch.no_grad():
            style = voice_pack[len(slot.phonemes) - 1]
            result = model(slot.phonemes, style, SPEED, return_output=True)
        audio = result.audio.detach().cpu().numpy()
        pred_dur = [int(value) for value in result.pred_dur.detach().cpu().tolist()]
        _write_pcm16(output_path, audio)
        record = {
            "slot": asdict(slot),
            "condition": slot.condition,
            "renderer": {
                "package": "kokoro",
                "package_version": KOKORO_VERSION,
                "model_repo": MODEL_REPO,
                "model_revision": MODEL_REVISION,
                "model_sha256": MODEL_HASHES[MODEL_FILE],
                "voice": VOICE,
                "voice_sha256": MODEL_HASHES[VOICE_FILE],
                "speed": SPEED,
                "device": "cpu",
            },
            "predicted_durations": pred_dur,
            "audio_relative_path": str(output_path.relative_to(run_dir)),
            "audio_sha256": sha256_file(output_path),
            **_audio_metrics(output_path, SOURCE_SYLLABLES),
        }
        if slot.target_symbol is not None:
            alignment = duration_alignment(
                vocab=model.vocab,
                phonemes=slot.phonemes,
                pred_dur=pred_dur,
                sample_count=len(audio),
                target_symbol=slot.target_symbol,
            )
            record["target"] = _measure_target(output_path, alignment)
            record["target"]["individual_category"] = _category_measurement(
                record["target"], "eh" if slot.condition == "schwa_weak_lens" else "ae"
            )
        records.append(record)
        print(f"kokoro spike {slot.request_order}/6 {slot.slot_id}: {record['timing']['utterance_duration_s']:.3f}s", flush=True)

    source = records[0]
    source_timing, source_prosody = _reference_objects(source)
    for record in records[1:]:
        timing, prosody = _reference_objects(record)
        record["source_reference_match"] = compare_prosody(source_timing, source_prosody, timing, prosody)
        _whisper_audit(record, run_dir)

    neutral = next(item for item in records if item["condition"] == "schwa_weak_neutral")
    repeat = next(item for item in records if item["condition"] == "schwa_weak_neutral_identity_repeat")
    lens = next(item for item in records if item["condition"] == "schwa_weak_lens")
    identity = _identity_comparison(run_dir / neutral["audio_relative_path"], run_dir / repeat["audio_relative_path"])
    pair = _pair_classification(neutral["target"], lens["target"])
    duration_vectors = {
        "same_length": len(neutral["predicted_durations"]) == len(lens["predicted_durations"]),
        "changed_duration_indices": [
            index for index, (a, b) in enumerate(zip(neutral["predicted_durations"], lens["predicted_durations"], strict=True)) if a != b
        ],
    }
    summary = {
        "schema_version": 1,
        "status": "automatic_analysis_complete_manual_blind_review_pending",
        "run_id": RUN_ID,
        "protocol_sha256": protocol["protocol_sha256"],
        "api_calls_made": 0,
        "render_count": len(records),
        "identity": identity,
        "neutral_lens_duration_vectors": duration_vectors,
        "pair_acoustic_classification": pair,
        "schwa_neutral_audio_leak_screen_pass": bool(neutral["local_whisper"].get("audio_leak_screen_pass")),
        "schwa_lens_audio_leak_screen_pass": bool(lens["local_whisper"].get("audio_leak_screen_pass")),
        "automatic_candidate_pass": bool(
            identity["bit_identical"]
            and neutral["integrity_pass"]
            and lens["integrity_pass"]
            and neutral["target"]["individual_category"]["category_pass"]
            and lens["target"]["individual_category"]["category_pass"]
            and pair["classification"] == "category_and_direction_diagnostic_pass"
            and neutral["local_whisper"].get("audio_leak_screen_pass")
            and lens["local_whisper"].get("audio_leak_screen_pass")
        ),
    }
    atomic_write_json(run_dir / "records.json", records)
    atomic_write_json(run_dir / "summary.json", summary)
    _build_review(records, run_dir)
    return summary


def _build_review(records: list[dict[str, Any]], run_dir: Path) -> None:
    rows = [
        {
            "blind_id": f"clip-{index + 1:02d}",
            "audio": record["audio_relative_path"],
            "key": record["condition"],
        }
        for index, record in enumerate(records)
        if record["condition"] != "schwa_weak_neutral_identity_repeat"
    ]
    random.Random(f"{RUN_ID}-blind-v1").shuffle(rows)
    atomic_write_json(run_dir / "blind-key.json", {row["blind_id"]: row["key"] for row in rows})
    public = [{"blind_id": row["blind_id"], "audio": row["audio"]} for row in rows]
    html = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Phoneme-renderer blind review</title><style>body{font:16px/1.5 system-ui;max-width:800px;margin:auto;padding:24px;background:#f5f2e9;color:#17221c}.card{background:white;padding:18px;border:1px solid #d6d3c9;border-radius:16px;margin:16px 0}audio,textarea{width:100%}label{display:block;margin:8px 0}button{padding:10px 16px;border:0;border-radius:99px;background:#154f3e;color:white;font-weight:700}</style></head><body><h1>Blind phoneme-renderer review</h1><p>Judge each clip as audio. Conditions and phoneme plans are hidden. A strong carrier sounds like one natural sentence-like utterance, not a list, and does not suggest clear real words or names.</p><div id="cards"></div><button id="download">Download ratings.csv</button><script>const R=__ROWS__;const S=JSON.parse(localStorage.getItem('kokoro-spike-ratings')||'{}');const save=(i,k,v)=>{S[i]??={};S[i][k]=v;localStorage.setItem('kokoro-spike-ratings',JSON.stringify(S))};const sel=(i,k,v)=>`<select onchange="save('${i}','${k}',this.value)"><option value="">—</option>${v.map(x=>`<option ${S[i]?.[k]==x?'selected':''}>${x}</option>`).join('')}</select>`;document.getElementById('cards').innerHTML=R.map(r=>`<section class="card"><h2>${r.blind_id}</h2><audio controls src="${r.audio}"></audio><label>Naturalness ${sel(r.blind_id,'naturalness',['1','2','3','4','5'])}</label><label>List-like ${sel(r.blind_id,'list_like',['none','slight','dominant'])}</label><label>Meaning/name leakage ${sel(r.blind_id,'meaning_leak',['none','possible','clear'])}</label><label>Sentence-like rhythm ${sel(r.blind_id,'sentence_rhythm',['yes','partly','no','uncertain'])}</label><label>Artifact or pronunciation failure ${sel(r.blind_id,'artifact',['no','yes','uncertain'])}</label><textarea oninput="save('${r.blind_id}','notes',this.value)" placeholder="Notes">${S[r.blind_id]?.notes??''}</textarea></section>`).join('');document.getElementById('download').onclick=()=>{const F=['blind_id','naturalness','list_like','meaning_leak','sentence_rhythm','artifact','notes'];const q=v=>`"${String(v??'').replaceAll('"','""')}"`;const b=new Blob([[F.join(','),...R.map(r=>F.map(k=>q(k==='blind_id'?r.blind_id:S[r.blind_id]?.[k])).join(','))].join('\n')+'\n'],{type:'text/csv'});const a=document.createElement('a');a.href=URL.createObjectURL(b);a.download='kokoro-spike-ratings.csv';a.click()};</script></body></html>""".replace("__ROWS__", json.dumps(public))
    atomic_write_text(run_dir / "review.html", html)
