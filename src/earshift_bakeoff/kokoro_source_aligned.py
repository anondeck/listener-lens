from __future__ import annotations

import gc
import importlib.metadata
import json
import math
import os
import random
import wave
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx_whisper
import numpy as np

from .config import Paths, sha256_json, stable_json
from .gates import CandidateGate
from .kokoro_phoneme_spike import (
    CONFIG_FILE,
    KOKORO_VERSION,
    MODEL_FILE,
    MODEL_HASHES,
    MODEL_REPO,
    MODEL_REVISION,
    SAMPLE_RATE_HZ,
    VOICE,
    VOICE_FILE,
    _audio_metrics,
    _content_audit,
    _measure_target,
    _reference_objects,
    _verify_model_files,
    _write_pcm16,
    duration_alignment,
)
from .runtime_audio import canonical_tokens
from .same_take import WHISPER_MODEL
from .sentence_pair_v2_analysis import CEILINGS
from .util import atomic_write_json, atomic_write_text, sha256_file


RUN_ID = "20260716-kokoro-source-aligned-v2"
SOURCE_TEXT = "What a great day it is to catch some sun."
SOURCE_WORDS = ("what", "a", "great", "day", "it", "is", "to", "catch", "some", "sun")
SOURCE_PHONEMES = "wˌʌt ɐ ɡɹˈAt dˈA ɪt ɪz tə kˈæʧ sˌʌm sˈʌn."
SOURCE_LENS_CONTROL = SOURCE_PHONEMES.replace("æ", "ɛ")
CARRIER_NEUTRAL = "ɹˌOk ə dɹˈOk ɡˈʊ ək əʒ kə tˈæʧ fˌəm ʃˈʊŋ."
CARRIER_LENS = CARRIER_NEUTRAL.replace("æ", "ɛ")
CARRIER_WORDS_NEUTRAL = ("ɹoʊk", "ə", "dɹoʊk", "ɡʊ", "ək", "əʒ", "kə", "tæʧ", "fəm", "ʃʊŋ")
CARRIER_WORDS_LENS = (*CARRIER_WORDS_NEUTRAL[:7], "tɛʧ", *CARRIER_WORDS_NEUTRAL[8:])
TARGET_RAW_INDEX = CARRIER_NEUTRAL.index("æ")
SOURCE_SYLLABLES = 10
SPEED = 1.0

VOWELS = {"i": "i", "ih": "ɪ", "eh": "ɛ", "ae": "æ", "uu": "ʊ", "u": "u"}
ANCHOR_SHELLS = {
    "h_V_d": ("h", "d"),
    "b_V_d": ("b", "d"),
    "z_V_v": ("z", "v"),
}


@dataclass(frozen=True)
class AnchorSlot:
    request_order: int
    slot_id: str
    shell_id: str
    vowel_label: str
    vowel_symbol: str
    phonemes: str


def anchor_manifest() -> tuple[AnchorSlot, ...]:
    slots: list[AnchorSlot] = []
    order = 9
    for shell_id, (onset, coda) in ANCHOR_SHELLS.items():
        labels = tuple(VOWELS) if shell_id == "h_V_d" else ("eh", "ae")
        for label in labels:
            symbol = VOWELS[label]
            slots.append(
                AnchorSlot(
                    request_order=order,
                    slot_id=f"anchor-{shell_id.lower()}-{label}",
                    shell_id=shell_id,
                    vowel_label=label,
                    vowel_symbol=symbol,
                    phonemes=f"{onset}ˈ{symbol}{coda}",
                )
            )
            order += 1
    return tuple(slots)


def _symbol_class(symbol: str) -> str:
    if symbol in " ˈˌ.,!?;:":
        return symbol
    if symbol in set(VOWELS.values()) | {"ɐ", "ə", "ʌ", "A", "O"}:
        return "V"
    return "C"


def isomorphism_report() -> dict[str, Any]:
    if not (len(SOURCE_PHONEMES) == len(CARRIER_NEUTRAL) == len(CARRIER_LENS)):
        return {"pass": False, "reason": "raw_length_mismatch"}
    neutral_lens = [
        {"index": index, "neutral": left, "lens": right}
        for index, (left, right) in enumerate(zip(CARRIER_NEUTRAL, CARRIER_LENS, strict=True))
        if left != right
    ]
    structural_positions = set(" ˈˌ.,!?;:")
    structure_match = all(
        (source == carrier) if source in structural_positions or carrier in structural_positions else True
        for source, carrier in zip(SOURCE_PHONEMES, CARRIER_NEUTRAL, strict=True)
    )
    class_match = all(
        _symbol_class(source) == _symbol_class(carrier)
        for source, carrier in zip(SOURCE_PHONEMES, CARRIER_NEUTRAL, strict=True)
    )
    return {
        "pass": bool(
            structure_match
            and class_match
            and neutral_lens == [{"index": TARGET_RAW_INDEX, "neutral": "æ", "lens": "ɛ"}]
        ),
        "raw_character_count": len(SOURCE_PHONEMES),
        "structure_match": structure_match,
        "segment_class_match": class_match,
        "neutral_lens_differences": neutral_lens,
        "target_raw_index": TARGET_RAW_INDEX,
    }


def phone_gate_report() -> dict[str, Any]:
    gate = CandidateGate()
    conditions: dict[str, Any] = {}
    for label, words in (("neutral", CARRIER_WORDS_NEUTRAL), ("lens", CARRIER_WORDS_LENS)):
        isolated = [
            {"position": index + 1, "ipa": ipa, "predicted_homophone": gate.phone_match("en", ipa)}
            for index, ipa in enumerate(words)
        ]
        adjacency = [
            {
                "left_position": index + 1,
                "right_position": index + 2,
                "ipa": words[index] + words[index + 1],
                "predicted_homophone": gate.phone_match("en", words[index] + words[index + 1]),
            }
            for index in range(len(words) - 1)
        ]
        conditions[label] = {
            "isolated": isolated,
            "adjacency": adjacency,
            "pass": not any(item["predicted_homophone"] for item in (*isolated, *adjacency)),
        }
    return conditions


def protocol_record() -> dict[str, Any]:
    _verify_model_files(download=False)
    isomorphism = isomorphism_report()
    gates = phone_gate_report()
    if not isomorphism["pass"]:
        raise RuntimeError("source/carrier token isomorphism failed")
    if not all(item["pass"] for item in gates.values()):
        raise RuntimeError("source-aligned carrier phone gate failed")
    main_manifest = [
        {"request_order": 1, "slot_id": "source-reference", "method": "source_state_source_content"},
        {"request_order": 2, "slot_id": "independent-carrier-neutral", "method": "ordinary_independent_forward"},
        {"request_order": 3, "slot_id": "source-state-neutral", "method": "source_alignment_source_f0n_neutral_content"},
        {"request_order": 4, "slot_id": "source-state-neutral-repeat", "method": "exact_repeat_of_slot_3"},
        {"request_order": 5, "slot_id": "source-state-lens", "method": "slot_3_state_target_content_swap_only"},
        {"request_order": 6, "slot_id": "carrier-state-neutral", "method": "source_alignment_carrier_f0n_neutral_content"},
        {"request_order": 7, "slot_id": "carrier-state-lens", "method": "slot_6_state_target_content_swap_only"},
        {"request_order": 8, "slot_id": "meaningful-source-lens-control", "method": "source_state_source_target_content_swap_only"},
    ]
    payload = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "status": "zero_api_source_aligned_latent_resynthesis_frozen_before_rendering_and_listening",
        "question": (
            "Can a phoneme-isomorphic nonce carrier inherit a natural source sentence's timing and prosodic state "
            "while neutral and lens share all decoder controls except one target-phoneme content vector?"
        ),
        "renderer": {
            "package": "kokoro",
            "package_version": KOKORO_VERSION,
            "model_repo": MODEL_REPO,
            "model_revision": MODEL_REVISION,
            "model_hashes": MODEL_HASHES,
            "voice": VOICE,
            "sample_rate_hz": SAMPLE_RATE_HZ,
            "speed": SPEED,
            "device": "cpu_single_thread_mkldnn_disabled",
            "api_calls": 0,
        },
        "source": {"text": SOURCE_TEXT, "words": list(SOURCE_WORDS), "phonemes": SOURCE_PHONEMES},
        "carrier": {
            "neutral_phonemes": CARRIER_NEUTRAL,
            "lens_phonemes": CARRIER_LENS,
            "neutral_word_ipa": list(CARRIER_WORDS_NEUTRAL),
            "lens_word_ipa": list(CARRIER_WORDS_LENS),
            "isomorphism": isomorphism,
            "phone_gates": gates,
            "design": (
                "preserve every boundary, stress marker, segment count, and consonant/vowel class; substitute "
                "within broad segment classes; apply /ae/->/eh/ at the aligned eighth source-word position"
            ),
        },
        "manifest": [*main_manifest, *(asdict(slot) for slot in anchor_manifest())],
        "shared_decoder_contract": {
            "alignment": "all transferred conditions use the source reference predicted-duration vector and alignment matrix",
            "source_state_pair": "neutral and lens use identical source-derived F0 and noise tensors",
            "carrier_state_pair": "neutral and lens use identical neutral-carrier-derived F0 and noise tensors over source alignment",
            "content_locality": (
                "lens text-encoder state starts as the neutral state and replaces only the target token column; "
                "all other content columns are bit-identical before decoder convolution"
            ),
            "identity": "slot 4 repeats slot 3 from the same frozen tensors and reports sample equality",
        },
        "same_voice_anchor_design": {
            "topology": "six h_V_d vowels establish broad within-voice sanity",
            "endpoints": "h_V_d, b_V_d, and z_V_v each render /ae/ and /eh/",
            "instrument": "standalone Praat, middle 50 percent of renderer-aligned vowel, 5500/5750/6000 Hz ceilings",
            "pair_gate": (
                "for every ceiling, neutral is closer to the Kokoro /ae/ centroid, lens is closer to the Kokoro /eh/ "
                "centroid, pair direction cosine is at least 0.5, and magnitude is at least max(0.25 Bark, half "
                "the median same-shell anchor shift); all three same-shell anchor vectors must point within cosine "
                "0.5 of their median direction"
            ),
        },
        "automatic_interpretation": {
            "prosody": "compare duration, pauses, F0, pitch contour, and energy contour with source-reference audio",
            "semantic_opacity": (
                "planned phonemes must pass isolated and adjacency homophone gates; local ASR source overlap is hard, "
                "while incidental ASR dictionary spellings are reported but require listener meaning judgment"
            ),
            "no_acoustic_selection": "both transferred pairs are measured and retained; no take selection occurs",
        },
        "manual_gate": {
            "blind": "renderer method and neutral/lens identity are hidden",
            "pass": (
                "a transferred pair requires both sides at least 4/5 naturalness, sentence-like rather than dominant "
                "list delivery, no recoverable sentence meaning, a clear correctly directed target difference, and "
                "manageable unrelated interference"
            ),
            "scope": "product/artifact QC; not Brazilian-Portuguese population evidence",
        },
        "stopping_rule": (
            "This run compares the two preregistered shared-state strategies. A measurement implementation defect may "
            "be corrected without rewriting returned audio; new carrier content or renderer parameters require a new protocol."
        ),
        "official_implementation_source": "https://github.com/hexgrad/kokoro/blob/main/kokoro/model.py",
    }
    return {**payload, "protocol_sha256": sha256_json(payload)}


def prepare() -> dict[str, Any]:
    protocol = protocol_record()
    path = Paths().artifacts / "phoneme-renderer" / RUN_ID / "protocol.json"
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if stable_json(existing) != stable_json(protocol):
            raise RuntimeError("existing source-aligned protocol differs from freeze")
    else:
        atomic_write_json(path, protocol)
    return protocol


def stress_span_alignment(
    *,
    vocab: dict[str, int],
    phonemes: str,
    pred_dur: list[int],
    sample_count: int,
    target_symbol: str,
) -> dict[str, Any]:
    filtered = [(raw_index, symbol) for raw_index, symbol in enumerate(phonemes) if vocab.get(symbol) is not None]
    if len(pred_dur) != len(filtered) + 2:
        raise RuntimeError("stress-span duration/token mismatch")
    targets = [index for index, (_, symbol) in enumerate(filtered) if symbol == target_symbol]
    if len(targets) != 1:
        raise RuntimeError(f"expected one target symbol {target_symbol!r}")
    target_index = targets[0]
    if target_index == 0 or filtered[target_index - 1][1] not in {"ˈ", "ˌ"}:
        raise RuntimeError("target vowel is not immediately preceded by a stress token")
    stress_index = target_index - 1
    stress_duration_index = stress_index + 1
    target_duration_index = target_index + 1
    total_frames = sum(pred_dur)
    samples_per_frame = sample_count / total_frames
    if abs(samples_per_frame - round(samples_per_frame)) > 1e-9:
        raise RuntimeError("decoder samples do not have an integer duration-frame ratio")
    start_sample = round(sum(pred_dur[:stress_duration_index]) * samples_per_frame)
    end_sample = round(sum(pred_dur[: target_duration_index + 1]) * samples_per_frame)
    return {
        "alignment_version": "kokoro-stress-plus-vowel-v1",
        "raw_stress_character_index": filtered[stress_index][0],
        "raw_target_character_index": filtered[target_index][0],
        "stress_symbol": filtered[stress_index][1],
        "target_symbol": target_symbol,
        "stress_duration_frames": pred_dur[stress_duration_index],
        "target_duration_frames": pred_dur[target_duration_index],
        "combined_duration_frames": pred_dur[stress_duration_index] + pred_dur[target_duration_index],
        "total_duration_frames": total_frames,
        "samples_per_duration_frame": samples_per_frame,
        "start_sample": start_sample,
        "end_sample_exclusive": end_sample,
        "start_s": start_sample / SAMPLE_RATE_HZ,
        "end_s": end_sample / SAMPLE_RATE_HZ,
        "sample_count": end_sample - start_sample,
    }


def measurement_amendment_record() -> dict[str, Any]:
    run_dir = Paths().artifacts / "phoneme-renderer" / RUN_ID
    records_path = run_dir / "records.json"
    summary_path = run_dir / "summary.json"
    if not records_path.is_file() or not summary_path.is_file():
        raise RuntimeError("source-aligned audio must exist before freezing the measurement amendment")
    audio_hashes = {
        path.name: sha256_file(path)
        for path in sorted((run_dir / "audio").glob("*.wav"))
    }
    payload = {
        "schema_version": 1,
        "run_id": f"{RUN_ID}-stress-span-reanalysis-v1",
        "status": "measurement_correction_frozen_before_reanalysis",
        "parent_protocol_sha256": protocol_record()["protocol_sha256"],
        "parent_records_sha256": sha256_file(records_path),
        "parent_summary_sha256": sha256_file(summary_path),
        "audio_hashes": audio_hashes,
        "defect": (
            "Kokoro assigns nonzero predicted duration to the stress token immediately before each stressed vowel. "
            "Treating only the following vowel token as the acoustic interval incorrectly excludes renderer frames "
            "that belong to the stressed vowel realization; the known six-vowel anchor topology failed under that mapping."
        ),
        "correction": (
            "For every stressed target, set the aligned acoustic interval from the beginning of the immediately "
            "preceding primary/secondary stress token through the end of the vowel token, then retain the frozen "
            "middle-50-percent measurement procedure."
        ),
        "selection_independence": (
            "The correction is defined by renderer token semantics and checked on anchors. It does not move, trim, "
            "or select any product audio by observed neutral/lens effect."
        ),
        "unchanged": [
            "all 18 WAV files and hashes",
            "Praat executable and Burg settings",
            "5500/5750/6000 Hz ceilings",
            "middle-50-percent aggregation",
            "anchor geometry formulas and product thresholds",
            "shared decoder states and manual review gate",
        ],
        "stopping_rule": (
            "Write separate reanalyzed records and summary; never overwrite v2 records. If the known anchor topology "
            "still fails, the acoustic layer remains invalid and product pair formants remain uninterpretable."
        ),
    }
    return {**payload, "amendment_sha256": sha256_json(payload)}


def prepare_measurement_amendment() -> dict[str, Any]:
    amendment = measurement_amendment_record()
    path = Paths().artifacts / "phoneme-renderer" / RUN_ID / "stress-span-amendment.json"
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if stable_json(existing) != stable_json(amendment):
            raise RuntimeError("existing stress-span amendment differs from freeze")
    else:
        atomic_write_json(path, amendment)
    return amendment


def reanalyze_stress_spans() -> dict[str, Any]:
    amendment = prepare_measurement_amendment()
    run_dir = Paths().artifacts / "phoneme-renderer" / RUN_ID
    original_records = json.loads((run_dir / "records.json").read_text(encoding="utf-8"))
    files = _verify_model_files(download=False)
    import torch
    from kokoro import KModel

    model = KModel(
        repo_id=MODEL_REPO,
        config=str(files[CONFIG_FILE]),
        model=str(files[MODEL_FILE]),
    ).to("cpu").eval()
    records = json.loads(json.dumps(original_records))
    for record in records:
        if "target" not in record:
            continue
        original = record["target"]
        target_symbol = original["alignment"]["target_symbol"]
        with wave.open(str(run_dir / record["audio_relative_path"]), "rb") as handle:
            sample_count = handle.getnframes()
        alignment = stress_span_alignment(
            vocab=model.vocab,
            phonemes=record["phonemes"],
            pred_dur=record["predicted_durations"],
            sample_count=sample_count,
            target_symbol=target_symbol,
        )
        record["original_target_measurement"] = original
        record["target"] = _measure_target(run_dir / record["audio_relative_path"], alignment)
    anchor_records = [record for record in records if record.get("shell_id")]
    geometry = _anchor_geometry(anchor_records)
    topology = _topology(anchor_records)
    source_pair = _classify_pair(records[2], records[4], geometry)
    carrier_pair = _classify_pair(records[5], records[6], geometry)
    summary = {
        "schema_version": 1,
        "run_id": f"{RUN_ID}-stress-span-reanalysis-v1",
        "status": "stress_span_reanalysis_complete",
        "parent_protocol_sha256": protocol_record()["protocol_sha256"],
        "amendment_sha256": amendment["amendment_sha256"],
        "api_calls_made": 0,
        "audio_renders_made": 0,
        "anchor_topology": topology,
        "anchor_geometry": geometry,
        "source_state_pair_acoustic_gate": source_pair,
        "carrier_state_pair_acoustic_gate": carrier_pair,
        "acoustic_layer_valid": topology["pass"],
    }
    atomic_write_json(run_dir / "stress-span-records.json", records)
    atomic_write_json(run_dir / "stress-span-summary.json", summary)
    return summary


def _input_ids(model: Any, phonemes: str, torch: Any) -> Any:
    values = [model.vocab.get(symbol) for symbol in phonemes]
    values = [value for value in values if value is not None]
    return torch.LongTensor([[0, *values, 0]]).to(model.device)


def _text_features(model: Any, input_ids: Any, ref_s: Any, torch: Any) -> dict[str, Any]:
    input_lengths = torch.full(
        (input_ids.shape[0],), input_ids.shape[-1], device=input_ids.device, dtype=torch.long
    )
    text_mask = torch.arange(input_lengths.max()).unsqueeze(0).expand(input_lengths.shape[0], -1).type_as(input_lengths)
    text_mask = torch.gt(text_mask + 1, input_lengths.unsqueeze(1)).to(model.device)
    bert_dur = model.bert(input_ids, attention_mask=(~text_mask).int())
    d_en = model.bert_encoder(bert_dur).transpose(-1, -2)
    style = ref_s[:, 128:]
    d = model.predictor.text_encoder(d_en, style, input_lengths, text_mask)
    t_en = model.text_encoder(input_ids, input_lengths, text_mask)
    return {
        "input_lengths": input_lengths,
        "text_mask": text_mask,
        "style": style,
        "d": d,
        "t_en": t_en,
    }


def _predicted_alignment(model: Any, features: dict[str, Any], speed: float, torch: Any) -> tuple[Any, Any]:
    x, _ = model.predictor.lstm(features["d"])
    duration = model.predictor.duration_proj(x)
    pred_dur = torch.round(torch.sigmoid(duration).sum(axis=-1) / speed).clamp(min=1).long().squeeze()
    indices = torch.repeat_interleave(torch.arange(features["input_lengths"].item(), device=model.device), pred_dur)
    alignment = torch.zeros((features["input_lengths"].item(), indices.shape[0]), device=model.device)
    alignment[indices, torch.arange(indices.shape[0])] = 1
    return pred_dur, alignment.unsqueeze(0)


def _f0n(model: Any, features: dict[str, Any], alignment: Any) -> tuple[Any, Any]:
    encoded = features["d"].transpose(-1, -2) @ alignment
    return model.predictor.F0Ntrain(encoded, features["style"])


def _decode(model: Any, text_state: Any, alignment: Any, f0: Any, noise: Any, ref_s: Any) -> Any:
    asr = text_state @ alignment
    return model.decoder(asr, f0, noise, ref_s[:, :128]).squeeze().detach().cpu()


def _target_token_index(model: Any, phonemes: str, symbol: str) -> int:
    filtered = [item for item in phonemes if model.vocab.get(item) is not None]
    matches = [index for index, item in enumerate(filtered) if item == symbol]
    if len(matches) != 1:
        raise RuntimeError(f"expected one {symbol!r} in filtered phoneme plan")
    return matches[0] + 1


def _localized_lens_state(neutral_state: Any, lens_state: Any, target_with_boundary: int) -> Any:
    mixed = neutral_state.clone()
    mixed[:, :, target_with_boundary] = lens_state[:, :, target_with_boundary]
    return mixed


def _record_audio(
    *,
    run_dir: Path,
    order: int,
    slot_id: str,
    method: str,
    phonemes: str,
    audio: Any,
    pred_dur: Any,
    model: Any,
    target_symbol: str | None,
) -> dict[str, Any]:
    path = run_dir / "audio" / f"{order:02d}__{slot_id}.wav"
    array = audio.detach().cpu().numpy() if hasattr(audio, "detach") else np.asarray(audio)
    durations = [int(value) for value in pred_dur.detach().cpu().tolist()]
    _write_pcm16(path, array)
    record: dict[str, Any] = {
        "request_order": order,
        "slot_id": slot_id,
        "method": method,
        "phonemes": phonemes,
        "audio_relative_path": str(path.relative_to(run_dir)),
        "audio_sha256": sha256_file(path),
        "predicted_durations": durations,
        **_audio_metrics(path, SOURCE_SYLLABLES if order <= 8 else 1),
    }
    if target_symbol is not None:
        alignment = duration_alignment(
            vocab=model.vocab,
            phonemes=phonemes,
            pred_dur=durations,
            sample_count=len(array),
            target_symbol=target_symbol,
        )
        record["target"] = _measure_target(path, alignment)
    return record


def _identity(left: Path, right: Path) -> dict[str, Any]:
    with wave.open(str(left), "rb") as first, wave.open(str(right), "rb") as second:
        a = np.frombuffer(first.readframes(first.getnframes()), dtype="<i2").astype(np.int32)
        b = np.frombuffer(second.readframes(second.getnframes()), dtype="<i2").astype(np.int32)
    if a.shape != b.shape:
        return {"sample_count_equal": False, "bit_identical": False}
    delta = np.abs(a - b)
    return {
        "sample_count_equal": True,
        "bit_identical": bool(np.array_equal(a, b)),
        "maximum_absolute_sample_delta": int(delta.max(initial=0)),
        "mean_absolute_sample_delta": float(delta.mean()) if delta.size else 0.0,
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
            "audit": audit,
            "source_overlap_pass": audit["source_overlap_pass"],
        }
    except Exception as exc:
        record["local_whisper"] = {"error": f"{type(exc).__name__}: {str(exc)[:500]}"}
    finally:
        mx.clear_cache()
        gc.collect()


def _anchor_geometry(anchor_records: list[dict[str, Any]]) -> dict[str, Any]:
    families: dict[str, Any] = {}
    for ceiling in CEILINGS:
        key = str(ceiling)
        endpoints: dict[str, list[np.ndarray]] = {"ae": [], "eh": []}
        by_shell: dict[str, dict[str, np.ndarray]] = {}
        for record in anchor_records:
            label = record["vowel_label"]
            if label not in endpoints:
                continue
            measurement = record["target"]["measurements"][key]
            point = np.asarray([measurement["f1_bark"], measurement["f2_bark"]], dtype=float)
            endpoints[label].append(point)
            by_shell.setdefault(record["shell_id"], {})[label] = point
        ae = np.mean(np.vstack(endpoints["ae"]), axis=0)
        eh = np.mean(np.vstack(endpoints["eh"]), axis=0)
        endpoint_vector = eh - ae
        endpoint_magnitude = float(np.linalg.norm(endpoint_vector))
        paired_vectors = [values["eh"] - values["ae"] for values in by_shell.values()]
        paired_magnitudes = [float(np.linalg.norm(vector)) for vector in paired_vectors]
        paired_cosines = [
            float(np.dot(vector, endpoint_vector) / (np.linalg.norm(vector) * endpoint_magnitude))
            if np.linalg.norm(vector) > 0 and endpoint_magnitude > 0
            else -1.0
            for vector in paired_vectors
        ]
        families[key] = {
            "ae_centroid_bark": ae.tolist(),
            "eh_centroid_bark": eh.tolist(),
            "endpoint_vector_bark": endpoint_vector.tolist(),
            "endpoint_magnitude_bark": endpoint_magnitude,
            "paired_shell_magnitudes_bark": paired_magnitudes,
            "paired_shell_direction_cosines": paired_cosines,
            "anchor_direction_consistency_pass": all(value >= 0.5 for value in paired_cosines),
            "product_magnitude_threshold_bark": max(0.25, 0.5 * float(np.median(paired_magnitudes))),
        }
    return {"families": families}


def _classify_pair(neutral: dict[str, Any], lens: dict[str, Any], geometry: dict[str, Any]) -> dict[str, Any]:
    families: dict[str, Any] = {}
    for ceiling in CEILINGS:
        key = str(ceiling)
        anchor = geometry["families"][key]
        ae = np.asarray(anchor["ae_centroid_bark"])
        eh = np.asarray(anchor["eh_centroid_bark"])
        expected = np.asarray(anchor["endpoint_vector_bark"])
        n_measurement = neutral["target"]["measurements"][key]
        l_measurement = lens["target"]["measurements"][key]
        n = np.asarray([n_measurement["f1_bark"], n_measurement["f2_bark"]])
        l = np.asarray([l_measurement["f1_bark"], l_measurement["f2_bark"]])
        vector = l - n
        magnitude = float(np.linalg.norm(vector))
        expected_magnitude = float(np.linalg.norm(expected))
        cosine = (
            float(np.dot(vector, expected) / (magnitude * expected_magnitude))
            if magnitude > 0 and expected_magnitude > 0
            else -1.0
        )
        neutral_category = float(np.linalg.norm(n - ae)) < float(np.linalg.norm(n - eh))
        lens_category = float(np.linalg.norm(l - eh)) < float(np.linalg.norm(l - ae))
        passed = bool(
            anchor["anchor_direction_consistency_pass"]
            and n_measurement["plausibility_pass"]
            and l_measurement["plausibility_pass"]
            and neutral_category
            and lens_category
            and cosine >= 0.5
            and magnitude >= anchor["product_magnitude_threshold_bark"]
        )
        families[key] = {
            "neutral_bark": n.tolist(),
            "lens_bark": l.tolist(),
            "vector_bark": vector.tolist(),
            "magnitude_bark": magnitude,
            "magnitude_threshold_bark": anchor["product_magnitude_threshold_bark"],
            "direction_cosine": cosine,
            "neutral_category_pass": neutral_category,
            "lens_category_pass": lens_category,
            "anchor_direction_consistency_pass": anchor["anchor_direction_consistency_pass"],
            "pass": passed,
        }
    return {"families": families, "pass": all(item["pass"] for item in families.values())}


def _topology(anchor_records: list[dict[str, Any]]) -> dict[str, Any]:
    h_records = {record["vowel_label"]: record for record in anchor_records if record["shell_id"] == "h_V_d"}
    families: dict[str, Any] = {}
    for ceiling in CEILINGS:
        key = str(ceiling)
        values = {
            label: {
                "f1_bark": record["target"]["measurements"][key]["f1_bark"],
                "f2_bark": record["target"]["measurements"][key]["f2_bark"],
                "plausibility_pass": record["target"]["measurements"][key]["plausibility_pass"],
            }
            for label, record in h_records.items()
        }
        front_height = values["ae"]["f1_bark"] > values["eh"]["f1_bark"] > values["ih"]["f1_bark"] > values["i"]["f1_bark"]
        back_height = values["ae"]["f1_bark"] > values["uu"]["f1_bark"] > values["u"]["f1_bark"]
        families[key] = {
            "values": values,
            "front_height_order_pass": front_height,
            "back_height_order_pass": back_height,
            "broad_plausibility_pass": all(item["plausibility_pass"] for item in values.values()),
            "pass": bool(front_height and back_height and all(item["plausibility_pass"] for item in values.values())),
        }
    return {"families": families, "pass": all(item["pass"] for item in families.values())}


def _build_review(records: list[dict[str, Any]], run_dir: Path) -> None:
    include = {
        "source-reference",
        "independent-carrier-neutral",
        "source-state-neutral",
        "source-state-lens",
        "carrier-state-neutral",
        "carrier-state-lens",
    }
    rows = [
        {"blind_id": f"clip-{index + 1:02d}", "audio": record["audio_relative_path"], "key": record["slot_id"]}
        for index, record in enumerate(records)
        if record["slot_id"] in include
    ]
    random.Random(f"{RUN_ID}-blind").shuffle(rows)
    atomic_write_json(run_dir / "blind-key.json", {row["blind_id"]: row["key"] for row in rows})
    public = [{"blind_id": row["blind_id"], "audio": row["audio"]} for row in rows]
    html = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Source-aligned renderer review</title><style>body{font:17px/1.5 system-ui;max-width:820px;margin:auto;padding:24px;background:#f5f2e9;color:#17221c}.card{background:#fff;padding:18px;border:1px solid #d6d3c9;border-radius:16px;margin:16px 0}audio,textarea{width:100%}label{display:block;margin:9px 0}button{padding:11px 17px;border:0;border-radius:99px;background:#154f3e;color:white;font-weight:700}</style></head><body><h1>Blind source-aligned carrier review</h1><p>Judge what you hear, not the technology. One clip is the meaningful source benchmark; the others are nonce carriers. Conditions, renderer paths, and neutral/lens identity are hidden.</p><div id="cards"></div><button id="download">Download ratings.csv</button><script>const R=__ROWS__;const K='kokoro-source-aligned-v2';const S=JSON.parse(localStorage.getItem(K)||'{}');const save=(i,k,v)=>{S[i]??={};S[i][k]=v;localStorage.setItem(K,JSON.stringify(S))};const sel=(i,k,v)=>`<select onchange="save('${i}','${k}',this.value)"><option value="">—</option>${v.map(x=>`<option ${S[i]?.[k]==x?'selected':''}>${x}</option>`).join('')}</select>`;document.getElementById('cards').innerHTML=R.map(r=>`<section class="card"><h2>${r.blind_id}</h2><audio controls src="${r.audio}"></audio><label>Naturalness ${sel(r.blind_id,'naturalness',['1','2','3','4','5'])}</label><label>Delivery ${sel(r.blind_id,'delivery',['sentence-like','slightly list-like','dominantly list-like','other'])}</label><label>Recoverable English meaning ${sel(r.blind_id,'meaning',['none','isolated possible word','coherent phrase','clear source sentence'])}</label><label>Artifact ${sel(r.blind_id,'artifact',['none','minor','major','uncertain'])}</label><textarea oninput="save('${r.blind_id}','notes',this.value)" placeholder="What did you hear?">${S[r.blind_id]?.notes??''}</textarea></section>`).join('');document.getElementById('download').onclick=()=>{const F=['blind_id','naturalness','delivery','meaning','artifact','notes'];const q=v=>`"${String(v??'').replaceAll('"','""')}"`;const b=new Blob([[F.join(','),...R.map(r=>F.map(k=>q(k==='blind_id'?r.blind_id:S[r.blind_id]?.[k])).join(','))].join('\n')+'\n'],{type:'text/csv'});const a=document.createElement('a');a.href=URL.createObjectURL(b);a.download='source-aligned-v2-ratings.csv';a.click()};</script></body></html>""".replace("__ROWS__", json.dumps(public))
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
    model = KModel(
        repo_id=MODEL_REPO,
        config=str(files[CONFIG_FILE]),
        model=str(files[MODEL_FILE]),
    ).to("cpu").eval()
    voice_pack = torch.load(files[VOICE_FILE], map_location="cpu", weights_only=True)
    ref_s = voice_pack[len(SOURCE_PHONEMES) - 1].unsqueeze(0) if voice_pack[len(SOURCE_PHONEMES) - 1].ndim == 1 else voice_pack[len(SOURCE_PHONEMES) - 1]
    if ref_s.ndim == 1:
        ref_s = ref_s.unsqueeze(0)
    run_dir = Paths().artifacts / "phoneme-renderer" / RUN_ID

    with torch.no_grad():
        source_ids = _input_ids(model, SOURCE_PHONEMES, torch)
        source_features = _text_features(model, source_ids, ref_s, torch)
        source_durations, source_alignment = _predicted_alignment(model, source_features, SPEED, torch)
        source_f0, source_noise = _f0n(model, source_features, source_alignment)

        carrier_ids = _input_ids(model, CARRIER_NEUTRAL, torch)
        lens_ids = _input_ids(model, CARRIER_LENS, torch)
        carrier_features = _text_features(model, carrier_ids, ref_s, torch)
        lens_features = _text_features(model, lens_ids, ref_s, torch)
        if carrier_ids.shape != source_ids.shape or lens_ids.shape != source_ids.shape:
            raise RuntimeError("runtime token isomorphism failed")
        target_index = _target_token_index(model, CARRIER_NEUTRAL, "æ")
        lens_target_index = _target_token_index(model, CARRIER_LENS, "ɛ")
        if target_index != lens_target_index:
            raise RuntimeError("neutral/lens target token indices differ")
        localized_lens = _localized_lens_state(carrier_features["t_en"], lens_features["t_en"], target_index)

        source_lens_ids = _input_ids(model, SOURCE_LENS_CONTROL, torch)
        source_lens_features = _text_features(model, source_lens_ids, ref_s, torch)
        source_target_index = _target_token_index(model, SOURCE_PHONEMES, "æ")
        localized_source_lens = _localized_lens_state(
            source_features["t_en"], source_lens_features["t_en"], source_target_index
        )

        carrier_f0, carrier_noise = _f0n(model, carrier_features, source_alignment)
        main_audio = {
            "source-reference": _decode(model, source_features["t_en"], source_alignment, source_f0, source_noise, ref_s),
            "source-state-neutral": _decode(model, carrier_features["t_en"], source_alignment, source_f0, source_noise, ref_s),
            "source-state-neutral-repeat": _decode(model, carrier_features["t_en"], source_alignment, source_f0, source_noise, ref_s),
            "source-state-lens": _decode(model, localized_lens, source_alignment, source_f0, source_noise, ref_s),
            "carrier-state-neutral": _decode(model, carrier_features["t_en"], source_alignment, carrier_f0, carrier_noise, ref_s),
            "carrier-state-lens": _decode(model, localized_lens, source_alignment, carrier_f0, carrier_noise, ref_s),
            "meaningful-source-lens-control": _decode(
                model, localized_source_lens, source_alignment, source_f0, source_noise, ref_s
            ),
        }
        independent = model(CARRIER_NEUTRAL, ref_s, SPEED, return_output=True)

    main_specs = [
        (1, "source-reference", "source_state_source_content", SOURCE_PHONEMES, None, source_durations),
        (2, "independent-carrier-neutral", "ordinary_independent_forward", CARRIER_NEUTRAL, "æ", independent.pred_dur),
        (3, "source-state-neutral", "source_alignment_source_f0n_neutral_content", CARRIER_NEUTRAL, "æ", source_durations),
        (4, "source-state-neutral-repeat", "exact_repeat_of_slot_3", CARRIER_NEUTRAL, "æ", source_durations),
        (5, "source-state-lens", "slot_3_state_target_content_swap_only", CARRIER_LENS, "ɛ", source_durations),
        (6, "carrier-state-neutral", "source_alignment_carrier_f0n_neutral_content", CARRIER_NEUTRAL, "æ", source_durations),
        (7, "carrier-state-lens", "slot_6_state_target_content_swap_only", CARRIER_LENS, "ɛ", source_durations),
        (8, "meaningful-source-lens-control", "source_state_source_target_content_swap_only", SOURCE_LENS_CONTROL, "ɛ", source_durations),
    ]
    records: list[dict[str, Any]] = []
    for order, slot_id, method, phonemes, target, durations in main_specs:
        audio = independent.audio if slot_id == "independent-carrier-neutral" else main_audio[slot_id]
        record = _record_audio(
            run_dir=run_dir,
            order=order,
            slot_id=slot_id,
            method=method,
            phonemes=phonemes,
            audio=audio,
            pred_dur=durations,
            model=model,
            target_symbol=target,
        )
        records.append(record)
        print(f"source-aligned {order}/18 {slot_id}: {record['timing']['utterance_duration_s']:.3f}s", flush=True)

    anchor_records: list[dict[str, Any]] = []
    for slot in anchor_manifest():
        style = voice_pack[len(slot.phonemes) - 1]
        with torch.no_grad():
            output = model(slot.phonemes, style, SPEED, return_output=True)
        record = _record_audio(
            run_dir=run_dir,
            order=slot.request_order,
            slot_id=slot.slot_id,
            method="same_voice_direct_phoneme_anchor",
            phonemes=slot.phonemes,
            audio=output.audio,
            pred_dur=output.pred_dur,
            model=model,
            target_symbol=slot.vowel_symbol,
        )
        record.update({"shell_id": slot.shell_id, "vowel_label": slot.vowel_label, "vowel_symbol": slot.vowel_symbol})
        records.append(record)
        anchor_records.append(record)
        print(f"source-aligned {slot.request_order}/18 {slot.slot_id}", flush=True)

    source_record = records[0]
    source_timing, source_prosody = _reference_objects(source_record)
    from .carrier_architecture_tournament import compare_prosody

    for record in records[1:8]:
        timing, prosody = _reference_objects(record)
        record["source_reference_match"] = compare_prosody(source_timing, source_prosody, timing, prosody)
        if record["slot_id"] != "meaningful-source-lens-control":
            _whisper(record, run_dir)

    geometry = _anchor_geometry(anchor_records)
    topology = _topology(anchor_records)
    source_pair = _classify_pair(records[2], records[4], geometry)
    carrier_pair = _classify_pair(records[5], records[6], geometry)
    identity = _identity(
        run_dir / records[2]["audio_relative_path"], run_dir / records[3]["audio_relative_path"]
    )
    localized_difference = {
        "neutral_lens_text_state_difference_count": int(
            torch.count_nonzero(carrier_features["t_en"] != localized_lens).detach().cpu().item()
        ),
        "non_target_columns_equal": bool(
            torch.equal(
                torch.cat((carrier_features["t_en"][:, :, :target_index], carrier_features["t_en"][:, :, target_index + 1 :]), dim=2),
                torch.cat((localized_lens[:, :, :target_index], localized_lens[:, :, target_index + 1 :]), dim=2),
            )
        ),
        "shared_alignment": True,
        "shared_source_state_f0": True,
        "shared_source_state_noise": True,
    }
    summary = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "status": "automatic_analysis_complete_manual_blind_review_pending",
        "protocol_sha256": protocol["protocol_sha256"],
        "api_calls_made": 0,
        "render_count": len(records),
        "identity": identity,
        "localized_difference": localized_difference,
        "anchor_topology": topology,
        "anchor_geometry": geometry,
        "source_state_pair_acoustic_gate": source_pair,
        "carrier_state_pair_acoustic_gate": carrier_pair,
        "automatic_architecture_candidate": bool(
            identity["bit_identical"]
            and localized_difference["non_target_columns_equal"]
            and topology["pass"]
            and (source_pair["pass"] or carrier_pair["pass"])
        ),
    }
    atomic_write_json(run_dir / "records.json", records)
    atomic_write_json(run_dir / "summary.json", summary)
    _build_review(records, run_dir)
    return summary
