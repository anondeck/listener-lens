from __future__ import annotations

import importlib.metadata
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import Paths, sha256_json, stable_json
from .kokoro_latent_span_sweep import (
    RUN_ID as PARENT_RUN_ID,
    VARIANT_ORDER,
    _difference_report,
    _measure_stress_target,
    _span_indices,
    _variant_states,
    _whisper,
)
from .kokoro_phoneme_spike import (
    CONFIG_FILE,
    KOKORO_VERSION,
    MODEL_FILE,
    MODEL_REPO,
    MODEL_REVISION,
    SAMPLE_RATE_HZ,
    VOICE_FILE,
    _audio_metrics,
    _verify_model_files,
    _write_pcm16,
)
from .kokoro_source_aligned import (
    CARRIER_LENS,
    CARRIER_NEUTRAL,
    SOURCE_PHONEMES,
    SOURCE_SYLLABLES,
    SPEED,
    _f0n,
    _input_ids,
    _predicted_alignment,
    _target_token_index,
    _text_features,
)
from .sentence_pair_v2_analysis import CEILINGS
from .util import atomic_write_json, atomic_write_text, sha256_file


RUN_ID = "20260716-kokoro-common-rng-confirmation-v4"
RNG_SEED = 20_260_716
ANCHOR_CONTEXTS = {
    "isolated": ("tˈæʧ", "tˈɛʧ"),
    "local-phrase": ("kə tˈæʧ fˌəm.", "kə tˈɛʧ fˌəm."),
    "full-carrier": (CARRIER_NEUTRAL, CARRIER_LENS),
}


@dataclass(frozen=True)
class Slot:
    request_order: int
    slot_id: str
    kind: str
    condition: str
    phonemes: str
    target_symbol: str


def manifest() -> tuple[Slot, ...]:
    slots: list[Slot] = [
        Slot(1, "common-neutral", "shared_state", "neutral", CARRIER_NEUTRAL, "æ"),
        Slot(2, "common-neutral-identity", "shared_state", "identity", CARRIER_NEUTRAL, "æ"),
    ]
    for index, name in enumerate(VARIANT_ORDER, start=3):
        slots.append(Slot(index, f"common-lens-{name}", "shared_state", name, CARRIER_LENS, "ɛ"))
    order = 8
    for context, (neutral, lens) in ANCHOR_CONTEXTS.items():
        slots.append(Slot(order, f"anchor-{context}-ae", "context_anchor", context, neutral, "æ"))
        slots.append(Slot(order + 1, f"anchor-{context}-eh", "context_anchor", context, lens, "ɛ"))
        order += 2
    return tuple(slots)


def _parent_files() -> tuple[Path, Path, Path]:
    directory = Paths().artifacts / "phoneme-renderer" / PARENT_RUN_ID
    return directory, directory / "records.json", directory / "summary.json"


def protocol_record() -> dict[str, Any]:
    _verify_model_files(download=False)
    parent, records, summary = _parent_files()
    if not records.is_file() or not summary.is_file():
        raise RuntimeError("v3 latent-span artifacts are missing")
    payload = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "status": "zero_api_common_random_context_anchor_confirmation_frozen_before_rendering_and_listening",
        "question": (
            "With identical vocoder random excitation and context-matched t_V_ch endpoints, what is the smallest "
            "contextual lens span that earns the /ae/->/eh/ sentence target gate?"
        ),
        "parent": {
            "protocol_sha256": json.loads((parent / "protocol.json").read_text(encoding="utf-8"))["protocol_sha256"],
            "records_sha256": sha256_file(records),
            "summary_sha256": sha256_file(summary),
            "result": "no_latent_span_passed_under_cross-shell_magnitude_threshold",
        },
        "renderer": {
            "package": "kokoro",
            "version": KOKORO_VERSION,
            "model_repo": MODEL_REPO,
            "model_revision": MODEL_REVISION,
            "sample_rate_hz": SAMPLE_RATE_HZ,
            "api_calls": 0,
        },
        "common_random_contract": {
            "seed": RNG_SEED,
            "mechanism": (
                "reset torch CPU RNG to the same seed immediately before every decoder invocation; this holds "
                "ISTFTNet random initial phase and Gaussian excitation common across neutral, identity, and lens"
            ),
            "execution": "separate decoder calls, not different rows of one stochastic batch",
            "identity": "neutral and duplicate neutral must be bit-identical",
            "official_source": "https://github.com/hexgrad/kokoro/blob/main/kokoro/istftnet.py",
        },
        "fixed_product_state": {
            "source_alignment": SOURCE_PHONEMES,
            "neutral": CARRIER_NEUTRAL,
            "lens": CARRIER_LENS,
            "shared": "source durations/alignment plus neutral-carrier F0/noise, same voice and speed",
            "span_order": list(VARIANT_ORDER),
        },
        "context_matched_anchors": {
            context: {"ae": pair[0], "eh": pair[1]} for context, pair in ANCHOR_CONTEXTS.items()
        },
        "anchor_gate": {
            "measurement": "stress-plus-vowel span, middle 50 percent, standalone Praat at 5500/5750/6000 Hz",
            "direction_sanity": "all three context vectors must have cosine >=0.5 with their median vector at each ceiling",
            "endpoints": "full-carrier ae and eh anchor points at the same ceiling",
            "magnitude_threshold": "max(0.25 Bark, half the full-carrier anchor shift at that ceiling)",
            "product": (
                "neutral nearer full-carrier ae, lens nearer full-carrier eh, cosine >=0.5, and magnitude above "
                "the context-matched threshold at all three ceilings"
            ),
            "selection": "first passing span in frozen order; never largest-effect selection",
        },
        "manifest": [asdict(slot) for slot in manifest()],
        "manual_gate": (
            "only a selected acoustic pass advances to blind QC: both >=4/5 naturalness, sentence-like delivery, "
            "no stable recoverable meaning, clear correctly directed eighth-position difference, manageable interference"
        ),
        "stopping_rule": (
            "Exactly 13 local renders. No replacement or parameter change. If common-random identity fails or no "
            "span passes the context-matched gate, close this decoder-control route and retain all evidence."
        ),
    }
    return {**payload, "protocol_sha256": sha256_json(payload)}


def prepare() -> dict[str, Any]:
    protocol = protocol_record()
    path = Paths().artifacts / "phoneme-renderer" / RUN_ID / "protocol.json"
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if stable_json(existing) != stable_json(protocol):
            raise RuntimeError("existing common-RNG protocol differs from freeze")
    else:
        atomic_write_json(path, protocol)
    return protocol


def _decode_seeded(model: Any, state: Any, alignment: Any, f0: Any, noise: Any, ref_s: Any, torch: Any) -> Any:
    torch.manual_seed(RNG_SEED)
    asr = state @ alignment
    return model.decoder(asr, f0, noise, ref_s[:, :128]).squeeze().detach().cpu()


def _render_anchor(model: Any, voice_pack: Any, phonemes: str, torch: Any) -> Any:
    style = voice_pack[len(phonemes) - 1]
    torch.manual_seed(RNG_SEED)
    return model(phonemes, style, SPEED, return_output=True)


def _anchor_geometry(records: list[dict[str, Any]]) -> dict[str, Any]:
    families: dict[str, Any] = {}
    for ceiling in CEILINGS:
        key = str(ceiling)
        by_context: dict[str, dict[str, np.ndarray]] = {}
        for record in records:
            measurement = record["target"]["measurements"][key]
            point = np.asarray([measurement["f1_bark"], measurement["f2_bark"]], dtype=float)
            label = "ae" if record["target_symbol"] == "æ" else "eh"
            by_context.setdefault(record["condition"], {})[label] = point
        vectors = [values["eh"] - values["ae"] for values in by_context.values()]
        median_vector = np.median(np.vstack(vectors), axis=0)
        median_magnitude = float(np.linalg.norm(median_vector))
        cosines = [
            float(np.dot(vector, median_vector) / (np.linalg.norm(vector) * median_magnitude))
            if np.linalg.norm(vector) > 0 and median_magnitude > 0
            else -1.0
            for vector in vectors
        ]
        full = by_context["full-carrier"]
        full_vector = full["eh"] - full["ae"]
        full_magnitude = float(np.linalg.norm(full_vector))
        families[key] = {
            "contexts": {
                context: {label: point.tolist() for label, point in values.items()}
                for context, values in by_context.items()
            },
            "context_vectors_bark": [vector.tolist() for vector in vectors],
            "context_direction_cosines": cosines,
            "direction_sanity_pass": all(value >= 0.5 for value in cosines),
            "full_ae_bark": full["ae"].tolist(),
            "full_eh_bark": full["eh"].tolist(),
            "full_vector_bark": full_vector.tolist(),
            "full_magnitude_bark": full_magnitude,
            "product_magnitude_threshold_bark": max(0.25, 0.5 * full_magnitude),
        }
    return {"families": families, "pass": all(item["direction_sanity_pass"] for item in families.values())}


def _classify(neutral: dict[str, Any], lens: dict[str, Any], geometry: dict[str, Any]) -> dict[str, Any]:
    families: dict[str, Any] = {}
    for ceiling in CEILINGS:
        key = str(ceiling)
        anchor = geometry["families"][key]
        ae = np.asarray(anchor["full_ae_bark"])
        eh = np.asarray(anchor["full_eh_bark"])
        expected = np.asarray(anchor["full_vector_bark"])
        n_measure = neutral["target"]["measurements"][key]
        l_measure = lens["target"]["measurements"][key]
        n = np.asarray([n_measure["f1_bark"], n_measure["f2_bark"]])
        l = np.asarray([l_measure["f1_bark"], l_measure["f2_bark"]])
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
            anchor["direction_sanity_pass"]
            and n_measure["plausibility_pass"]
            and l_measure["plausibility_pass"]
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
            "threshold_bark": anchor["product_magnitude_threshold_bark"],
            "direction_cosine": cosine,
            "neutral_category_pass": neutral_category,
            "lens_category_pass": lens_category,
            "pass": passed,
        }
    return {"families": families, "pass": all(item["pass"] for item in families.values())}


def _review(records: list[dict[str, Any]], selected: str | None, run_dir: Path) -> None:
    if not selected:
        atomic_write_text(
            run_dir / "review.html",
            "<!doctype html><meta charset='utf-8'><title>No candidate</title><p>No latent span passed the frozen automatic gate; no product listening review was generated.</p>",
        )
        return
    neutral = next(record for record in records if record["slot_id"] == "common-neutral")
    identity = next(record for record in records if record["slot_id"] == "common-neutral-identity")
    lens = next(record for record in records if record["condition"] == selected)
    rows = [
        {"audio": neutral["audio_relative_path"], "key": neutral["slot_id"]},
        {"audio": identity["audio_relative_path"], "key": identity["slot_id"]},
        {"audio": lens["audio_relative_path"], "key": lens["slot_id"]},
    ]
    random.Random(f"{RUN_ID}-blind").shuffle(rows)
    for index, row in enumerate(rows, start=1):
        row["blind_id"] = f"clip-{index:02d}"
    pair = [
        {"pair_id": "A", "audio": neutral["audio_relative_path"], "key": neutral["slot_id"]},
        {"pair_id": "B", "audio": lens["audio_relative_path"], "key": lens["slot_id"]},
    ]
    random.Random(f"{RUN_ID}-pair").shuffle(pair)
    for pair_id, row in zip(("A", "B"), pair, strict=True):
        row["pair_id"] = pair_id
    atomic_write_json(
        run_dir / "blind-key.json",
        {
            "individual": {row["blind_id"]: row["key"] for row in rows},
            "pair": {row["pair_id"]: row["key"] for row in pair},
        },
    )
    public = [{"blind_id": row["blind_id"], "audio": row["audio"]} for row in rows]
    pair_public = [{"pair_id": row["pair_id"], "audio": row["audio"]} for row in pair]
    html = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Common-random blind review</title><style>:root{color-scheme:light}body{font:17px/1.5 system-ui;max-width:840px;margin:auto;padding:24px;background:#f5f2e9;color:#17221c}.dots{display:flex;gap:8px;margin:16px 0 24px}.dots i{width:18px;height:18px;border-radius:50%;background:#ccd2cc}.dots i:nth-child(8){background:#d87b35;outline:3px solid #f1c59e}.card{background:white;padding:20px;border:1px solid #d6d3c9;border-radius:16px;margin:16px 0}.pair{display:grid;grid-template-columns:1fr 1fr;gap:14px}.pair .card{margin:0}audio,textarea,select{width:100%;box-sizing:border-box}textarea{min-height:74px}label{display:block;margin:11px 0}button{padding:11px 18px;border:0;border-radius:999px;background:#154f3e;color:white;font-weight:700}.muted{color:#57645e}@media(max-width:620px){.pair{grid-template-columns:1fr}}</style></head><body><h1>Blind shared-prosody QC</h1><p>First judge each clip on its own. All use the same timing and prosody plan; two are identical-condition controls and one contains the selected vowel change. Conditions and spellings are hidden.</p><div class="dots">__DOTS__</div><div id="cards"></div><section class="card"><h2>Direct A/B comparison</h2><p class="muted">Now compare the eighth spoken position, highlighted above. This section is intentionally last.</p><div class="pair" id="pair"></div><label>How strong is the eighth-position difference?<select data-pair="target_difference"><option value="">—</option><option>none</option><option>subtle</option><option>clear</option></select></label><label>Which sounds closer to the vowel in “bet”?<select data-pair="bet_side"><option value="">—</option><option>A</option><option>B</option><option>uncertain</option><option>neither</option></select></label><label>Do unrelated delivery differences interfere?<select data-pair="interference"><option value="">—</option><option>no</option><option>manageable</option><option>dominant</option><option>uncertain</option></select></label><textarea data-pair="notes" placeholder="A/B notes"></textarea></section><button id="download">Download review.json</button><script>const R=__ROWS__,P=__PAIR__,K='kokoro-common-rng-v4-review',S=JSON.parse(localStorage.getItem(K)||'{"clips":{},"pair":{}}');S.clips??={};S.pair??={};const save=()=>localStorage.setItem(K,JSON.stringify(S));const opts=(values,value)=>'<option value="">—</option>'+values.map(x=>`<option ${x===value?'selected':''}>${x}</option>`).join('');document.getElementById('cards').innerHTML=R.map(r=>{const s=S.clips[r.blind_id]??{};return `<section class="card"><h2>${r.blind_id}</h2><audio controls src="${r.audio}"></audio><label>Naturalness<select data-id="${r.blind_id}" data-field="naturalness">${opts(['1','2','3','4','5'],s.naturalness)}</select></label><label>Delivery<select data-id="${r.blind_id}" data-field="delivery">${opts(['sentence-like','slightly list-like','dominantly list-like','other'],s.delivery)}</select></label><label>Stable recoverable meaning<select data-id="${r.blind_id}" data-field="meaning">${opts(['none','isolated possible word','coherent phrase','clear source sentence'],s.meaning)}</select></label><label>Artifact<select data-id="${r.blind_id}" data-field="artifact">${opts(['none','minor','major','uncertain'],s.artifact)}</select></label><textarea data-id="${r.blind_id}" data-field="notes" placeholder="If you heard words or a phrase, write exactly what you heard.">${s.notes??''}</textarea></section>`}).join('');document.getElementById('pair').innerHTML=P.map(r=>`<section class="card"><h3>Clip ${r.pair_id}</h3><audio controls src="${r.audio}"></audio></section>`).join('');document.querySelectorAll('[data-id]').forEach(el=>{el.oninput=()=>{const id=el.dataset.id;S.clips[id]??={};S.clips[id][el.dataset.field]=el.value;save()}});document.querySelectorAll('[data-pair]').forEach(el=>{el.value=S.pair[el.dataset.pair]??'';el.oninput=()=>{S.pair[el.dataset.pair]=el.value;save()}});document.getElementById('download').onclick=()=>{const payload={schema_version:1,run_id:'__RUN_ID__',saved_at:new Date().toISOString(),ratings:S};const blob=new Blob([JSON.stringify(payload,null,2)+'\\n'],{type:'application/json'});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='common-rng-v4-review.json';a.click()};</script></body></html>""".replace("__ROWS__", json.dumps(public)).replace("__PAIR__", json.dumps(pair_public)).replace("__DOTS__", "<i></i>" * 10).replace("__RUN_ID__", RUN_ID)
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
        neutral_features = _text_features(model, _input_ids(model, CARRIER_NEUTRAL, torch), ref_s, torch)
        lens_features = _text_features(model, _input_ids(model, CARRIER_LENS, torch), ref_s, torch)
        f0, noise = _f0n(model, neutral_features, alignment)
        target_index = _target_token_index(model, CARRIER_NEUTRAL, "æ")
        spans = _span_indices(model, CARRIER_NEUTRAL, target_index)
        states = _variant_states(neutral_features["t_en"], lens_features["t_en"], spans)
        main_audio = [_decode_seeded(model, state, alignment, f0, noise, ref_s, torch) for state in states]
    durations_list = [int(value) for value in durations.detach().cpu().tolist()]
    run_dir = Paths().artifacts / "phoneme-renderer" / RUN_ID
    records: list[dict[str, Any]] = []
    for slot, audio in zip(manifest()[:7], main_audio, strict=True):
        path = run_dir / "audio" / f"{slot.request_order:02d}__{slot.slot_id}.wav"
        _write_pcm16(path, audio.numpy())
        record = {
            **asdict(slot),
            "predicted_durations": durations_list,
            "audio_relative_path": str(path.relative_to(run_dir)),
            "audio_sha256": sha256_file(path),
            **_audio_metrics(path, SOURCE_SYLLABLES),
        }
        record["target"] = _measure_stress_target(path, model, slot.phonemes, durations_list, slot.target_symbol)
        records.append(record)
        print(f"common RNG {slot.request_order}/13 {slot.slot_id}: {record['timing']['utterance_duration_s']:.3f}s", flush=True)
    anchor_records: list[dict[str, Any]] = []
    for slot in manifest()[7:]:
        with torch.no_grad():
            output = _render_anchor(model, voice_pack, slot.phonemes, torch)
        path = run_dir / "audio" / f"{slot.request_order:02d}__{slot.slot_id}.wav"
        _write_pcm16(path, output.audio.numpy())
        anchor_durations = [int(value) for value in output.pred_dur.tolist()]
        record = {
            **asdict(slot),
            "predicted_durations": anchor_durations,
            "audio_relative_path": str(path.relative_to(run_dir)),
            "audio_sha256": sha256_file(path),
            **_audio_metrics(path, 1 if slot.condition == "isolated" else (3 if slot.condition == "local-phrase" else SOURCE_SYLLABLES)),
        }
        record["target"] = _measure_stress_target(path, model, slot.phonemes, anchor_durations, slot.target_symbol)
        records.append(record)
        anchor_records.append(record)
        print(f"common RNG {slot.request_order}/13 {slot.slot_id}", flush=True)
    for record in records[:7]:
        _whisper(record, run_dir)
    geometry = _anchor_geometry(anchor_records)
    neutral = records[0]
    pair_results: dict[str, Any] = {}
    for record in records[2:7]:
        result = _classify(neutral, record, geometry)
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
        "status": "manual_blind_qc_pending" if identity["bit_identical"] and selected else "common_rng_or_acoustic_gate_failed",
        "protocol_sha256": protocol["protocol_sha256"],
        "api_calls_made": 0,
        "render_count": len(records),
        "common_random_identity": identity,
        "context_anchor_geometry": geometry,
        "pair_results": pair_results,
        "selected_smallest_passing_span": selected,
        "automatic_candidate": bool(identity["bit_identical"] and geometry["pass"] and selected),
    }
    atomic_write_json(run_dir / "records.json", records)
    atomic_write_json(run_dir / "summary.json", summary)
    _review(records, selected, run_dir)
    return summary
