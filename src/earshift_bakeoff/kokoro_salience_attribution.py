from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

from .config import Paths, sha256_json, stable_json
from .util import atomic_write_json, atomic_write_text, sha256_file


RUN_ID = "20260716-kokoro-salience-attribution-v1"
PARENT_RUN_ID = "20260716-kokoro-common-rng-confirmation-v4"
BLIND_SEED = 20_260_716_01
SOURCE_SLOT_IDS = (
    "common-neutral",
    "common-neutral-identity",
    "common-lens-stress-plus-target",
    "anchor-full-carrier-ae",
    "anchor-full-carrier-eh",
)
LOGICAL_COMPARISONS = (
    {
        "logical_id": "identity-control",
        "source_slots": ("common-neutral", "common-neutral-identity"),
    },
    {
        "logical_id": "selected-lens",
        "source_slots": ("common-neutral", "common-lens-stress-plus-target"),
    },
    {
        "logical_id": "full-context-anchors",
        "source_slots": ("anchor-full-carrier-ae", "anchor-full-carrier-eh"),
    },
)


def _parent_dir() -> Path:
    return Paths().artifacts / "phoneme-renderer" / PARENT_RUN_ID


def _parent_records() -> dict[str, dict[str, Any]]:
    path = _parent_dir() / "records.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    records = {row["slot_id"]: row for row in rows}
    missing = set(SOURCE_SLOT_IDS) - records.keys()
    if missing:
        raise RuntimeError(f"frozen v4 records are missing source slots: {sorted(missing)}")
    return records


def _source_manifest() -> list[dict[str, Any]]:
    parent = _parent_dir()
    records = _parent_records()
    manifest: list[dict[str, Any]] = []
    for slot_id in SOURCE_SLOT_IDS:
        record = records[slot_id]
        path = parent / record["audio_relative_path"]
        actual_hash = sha256_file(path)
        if actual_hash != record["audio_sha256"]:
            raise RuntimeError(f"frozen source WAV hash mismatch: {slot_id}")
        alignment = record["target"]["alignment"]
        manifest.append(
            {
                "slot_id": slot_id,
                "parent_audio_relative_path": record["audio_relative_path"],
                "audio_sha256": actual_hash,
                "sample_rate_hz": record["timing"]["sample_rate_hz"],
                "decoded_sample_count": record["timing"]["decoded_sample_count"],
                "target_start_s": alignment["start_s"],
                "target_end_s": alignment["end_s"],
            }
        )
    return manifest


def protocol_record() -> dict[str, Any]:
    parent = _parent_dir()
    parent_protocol = json.loads((parent / "protocol.json").read_text(encoding="utf-8"))
    payload = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "status": "frozen_before_one_creator_salience_attribution_session",
        "question": (
            "Does synchronized target guidance attribute the v4 subtle/clear boundary to target localization, "
            "to the selected lens edit, or to the same voice's full-context category contrast?"
        ),
        "scope": {
            "sessions": 1,
            "listener": "informed creator-listener",
            "interpretation": "creator product-tuning evidence only",
            "api_calls": 0,
            "new_renders": 0,
            "new_audio_edits": 0,
            "source_audio": "existing frozen v4 WAVs only",
        },
        "parent": {
            "run_id": PARENT_RUN_ID,
            "protocol_sha256": parent_protocol["protocol_sha256"],
            "records_sha256": sha256_file(parent / "records.json"),
            "summary_sha256": sha256_file(parent / "summary.json"),
            "manual_result_sha256": sha256_file(parent / "manual-result.json"),
            "classification": (
                "controlled, correctly directed perceptual near-pass; not a full manual-gate pass"
            ),
            "classification_is_immutable": True,
        },
        "source_wavs": _source_manifest(),
        "logical_comparisons": [
            {"logical_id": row["logical_id"], "source_slots": list(row["source_slots"])}
            for row in LOGICAL_COMPARISONS
        ],
        "blinding": {
            "seed": BLIND_SEED,
            "trial_order": "deterministically shuffled",
            "side_order": "A/B independently and deterministically shuffled within each trial",
            "presentation": "no condition, phoneme, spelling, script, source slot, or descriptive filename is shown",
        },
        "target_cue": {
            "structure": "the same ten-position display with position eight highlighted for every side",
            "synchronization": (
                "moving playhead plus TARGET NOW state; within each comparison both sides use the union of their "
                "frozen target intervals so cue timing cannot disclose the changed side"
            ),
            "audio": "full frozen WAV; cue does not crop, alter, or replay a separate audio segment",
        },
        "measures": {
            "difference_strength": "integer 1-7; 1=no audible difference, 4=moderate, 7=very strong",
            "direction_category": "A, B, same, uncertain, or neither as closer to the vowel in bet",
            "confidence": "integer 1-5",
            "replay_count": (
                "automatic: play starts beyond the first start for each side; pause/resume therefore counts as "
                "another start"
            ),
            "unrelated_interference": "none, manageable, dominant, or uncertain",
            "artifact": "none, minor, major, or uncertain",
            "notes": "optional free text",
        },
        "decision_mapping": {
            "clear_correct_direction": (
                "difference strength >=5, expected bet-side category judgment, confidence >=3, interference not "
                "dominant, and artifact not major"
            ),
            "subtle_correct_direction": (
                "difference strength 2-4, expected bet-side category judgment, confidence >=2, interference not "
                "dominant, and artifact not major"
            ),
            "no_audible_difference": "difference strength 1 with same, uncertain, or neither category judgment",
            "identity_difference_flag": (
                "identity strength >=2 or identity category judgment A or B; investigate the response only"
            ),
            "anchors_unclear": (
                "the full-context anchor comparison does not satisfy clear_correct_direction; the complete session "
                "is nondiagnostic regardless of the lens response"
            ),
            "lens_not_supported_or_indeterminate": (
                "anchors are clear but the lens satisfies neither clear_correct_direction nor subtle_correct_direction"
            ),
            "replays": "descriptive product-tuning telemetry only; no pass threshold",
        },
        "interpretation_rules": {
            "anchors_clear_lens_clear_correct_direction": (
                "target guidance is the product intervention; v4 classification remains unchanged"
            ),
            "anchors_clear_lens_subtle": (
                "preserve v4 near-pass; any stronger or larger edit is a newly versioned candidate with fresh QC"
            ),
            "identity_reported_different": (
                "flag the single response for investigation; do not call listener unreliability and do not "
                "invalidate the bit-identical control"
            ),
            "anchors_unclear": "classify this session as nondiagnostic",
        },
        "stopping_rule": (
            "After the single creator session, preserve the raw response, decode, report, and stop before typed-engine "
            "or deployment work. Never reclassify common-RNG v4 from this diagnostic."
        ),
        "terminology": (
            "shared-state/common-RNG controlled synthesis pair; never literally same-take. Neutral and identity are "
            "bit-identical, while 87.5% of neutral/lens difference energy was localized near the target neighborhood."
        ),
        "subsequent_roadmap_constraints": {
            "claim": (
                "promising controlled architecture candidate on one carrier, one rule, and one informed "
                "creator-listener; not yet a validated general architecture"
            ),
            "typed_engine": (
                "build a narrow vertical slice first; it owns phoneme/stress planning, deterministic target-span "
                "construction and merging, repeated and multiple slots, punctuation, unsupported inputs, and common "
                "RNG for the complete neutral/lens pair; use that implementation for 2-3 carrier fixtures"
            ),
            "runtime": (
                "prefer server-side Kokoro initially; benchmark memory, cold start, and latency on basic or standard-1; "
                "do not assume the Cloudflare lite container is sufficient"
            ),
            "concurrency": (
                "render each pair atomically behind an inference lock or queue because torch.manual_seed is global; "
                "add a concurrent-request determinism regression"
            ),
            "browser_onnx": (
                "defer until excitation/noise is explicit and deterministic equivalence is revalidated"
            ),
            "native_bp": "future pilot informs claim strength and is not a shipping gate",
            "release": "pin Kokoro package, model, and voice hashes; update licensing and provenance",
        },
        "privacy_copy": (
            "Your typed sentence is processed by our Cloudflare-hosted service and is not sent to OpenAI for audio "
            "generation. OpenAI is used only for the optional activity generator, which receives bounded result "
            "metadata—not your sentence."
        ),
    }
    return {**payload, "protocol_sha256": sha256_json(payload)}


def blinded_layout(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    sources = {row["slot_id"]: row for row in protocol["source_wavs"]}
    rng = random.Random(BLIND_SEED)
    pending: list[dict[str, Any]] = []
    for comparison in protocol["logical_comparisons"]:
        slots = list(comparison["source_slots"])
        rng.shuffle(slots)
        pending.append({"logical_id": comparison["logical_id"], "source_slots": slots})
    rng.shuffle(pending)

    layout: list[dict[str, Any]] = []
    for index, row in enumerate(pending, start=1):
        source_rows = [sources[slot_id] for slot_id in row["source_slots"]]
        cue_start = min(source["target_start_s"] for source in source_rows)
        cue_end = max(source["target_end_s"] for source in source_rows)
        trial_id = f"comparison-{index:02d}"
        sides = []
        for side, source in zip(("A", "B"), source_rows, strict=True):
            sides.append(
                {
                    "side": side,
                    "source_slot_id": source["slot_id"],
                    "audio_sha256": source["audio_sha256"],
                    "parent_audio_relative_path": source["parent_audio_relative_path"],
                    "opaque_audio_relative_path": f"audio/{trial_id}-{side.lower()}.wav",
                }
            )
        layout.append(
            {
                "trial_id": trial_id,
                "logical_id": row["logical_id"],
                "cue_start_s": cue_start,
                "cue_end_s": cue_end,
                "sides": sides,
            }
        )
    return layout


def public_review_manifest(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "trial_id": trial["trial_id"],
            "cue_start_s": trial["cue_start_s"],
            "cue_end_s": trial["cue_end_s"],
            "sides": [
                {"side": side["side"], "audio": side["opaque_audio_relative_path"]}
                for side in trial["sides"]
            ],
        }
        for trial in blinded_layout(protocol)
    ]


def _review_html(public: list[dict[str, Any]], protocol_sha256: str) -> str:
    html = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Target-guided salience review</title>
<style>
:root{color-scheme:light;--ink:#17221c;--muted:#59665f;--paper:#f5f2e9;--card:#fff;--line:#d7d4ca;--target:#d87b35;--now:#16704f}
*{box-sizing:border-box}body{font:17px/1.5 system-ui,-apple-system,sans-serif;max-width:900px;margin:auto;padding:26px;background:var(--paper);color:var(--ink)}
h1{font-size:2rem;margin-bottom:.3rem}.intro{max-width:760px}.trial{background:var(--card);border:1px solid var(--line);border-radius:18px;padding:22px;margin:24px 0;box-shadow:0 2px 9px #17221c0a}
.pair{display:grid;grid-template-columns:1fr 1fr;gap:16px}.player{border:1px solid var(--line);border-radius:14px;padding:15px;background:#fbfbf8}.player h3{margin:.1rem 0 .6rem}audio{width:100%}
.positions{display:flex;gap:6px;margin:13px 0 8px}.positions i{width:15px;height:15px;border-radius:50%;background:#d6dcd7}.positions i:nth-child(8){background:var(--target);outline:2px solid #f0c49e}.target-now .positions i:nth-child(8){background:var(--now);outline:4px solid #9ed3bf}
.timeline{height:13px;border-radius:999px;background:#e4e8e4;position:relative;overflow:hidden}.target-band{position:absolute;top:0;bottom:0;background:#e8a36eaa}.playhead{position:absolute;top:0;bottom:0;width:3px;background:#17221c;left:0}.target-status{font-size:.85rem;color:var(--muted);font-weight:700;margin-top:5px}.target-now .target-status{color:var(--now)}
.fields{display:grid;grid-template-columns:1fr 1fr;gap:12px 18px;margin-top:18px}label{display:block;font-weight:650}select,textarea{width:100%;font:inherit;margin-top:5px;padding:9px;border:1px solid #b9c0ba;border-radius:9px;background:white}textarea{min-height:80px}.wide{grid-column:1/-1}.replays{font-size:.92rem;color:var(--muted)}
.scale{font-size:.88rem;color:var(--muted);font-weight:400}.download{padding:12px 20px;border:0;border-radius:999px;background:#154f3e;color:white;font-weight:750;font-size:1rem}.download:disabled{opacity:.42}.completion{margin-left:12px;color:var(--muted)}
@media(max-width:700px){.pair,.fields{grid-template-columns:1fr}.wide{grid-column:auto}}
</style>
</head>
<body>
<h1>Target-guided salience review</h1>
<p class="intro">Complete each comparison independently. Conditions, spellings, filenames, and comparison types are hidden. The orange eighth position is the target in every clip. While audio plays, the moving marker and <strong>TARGET NOW</strong> state show its frozen interval.</p>
<p class="intro"><strong>Replay counting:</strong> the first playback start for A and B is not a replay. Every later playback start—including resuming after pausing—is counted automatically.</p>
<main id="trials"></main>
<button class="download" id="download" disabled>Download response.json</button><span class="completion" id="completion"></span>
<script>
const R=__TRIALS__,PROTOCOL='__PROTOCOL__',KEY='kokoro-salience-attribution-v1';
const fresh=()=>({session_id:(globalThis.crypto&&crypto.randomUUID)?crypto.randomUUID():'session-'+Date.now()+'-'+Math.random().toString(16).slice(2),protocol_sha256:PROTOCOL,trials:{}});
let S;try{S=JSON.parse(localStorage.getItem(KEY))}catch(_){S=null}if(!S||S.protocol_sha256!==PROTOCOL)S=fresh();
const save=()=>localStorage.setItem(KEY,JSON.stringify(S));
const getTrial=id=>S.trials[id]??=( {responses:{},play_starts:{A:0,B:0}} );
const option=(value,label)=>`<option value="${value}">${label}</option>`;
const positions='<div class="positions" aria-label="Ten spoken positions; position eight is the target">'+Array.from({length:10},()=>'<i></i>').join('')+'</div>';
const player=(trial,side)=>`<section class="player" data-player="${trial.trial_id}-${side.side}"><h3>Clip ${side.side}</h3><audio controls preload="auto" data-trial="${trial.trial_id}" data-side="${side.side}" data-cue-start="${trial.cue_start_s}" data-cue-end="${trial.cue_end_s}" src="${side.audio}"></audio>${positions}<div class="timeline" aria-hidden="true"><span class="target-band"></span><i class="playhead"></i></div><div class="target-status" aria-live="polite">Target position</div></section>`;
document.getElementById('trials').innerHTML=R.map((trial,index)=>`<section class="trial" data-card="${trial.trial_id}"><h2>Comparison ${index+1} of ${R.length}</h2><div class="pair">${trial.sides.map(side=>player(trial,side)).join('')}</div><p class="replays">Recorded replays: <strong data-replays="${trial.trial_id}">0</strong></p><div class="fields"><label>Difference strength <span class="scale">1 none · 4 moderate · 7 very strong</span><select data-trial-id="${trial.trial_id}" data-field="difference_strength"><option value="">—</option>${[1,2,3,4,5,6,7].map(n=>option(String(n),String(n))).join('')}</select></label><label>Which side, if either, sounds closer to the vowel in “bet”?<select data-trial-id="${trial.trial_id}" data-field="category_judgment"><option value="">—</option>${option('A','A')}${option('B','B')}${option('same','Same / no category difference')}${option('uncertain','Uncertain')}${option('neither','Neither')}</select></label><label>Confidence <span class="scale">1 guessing · 5 highly confident</span><select data-trial-id="${trial.trial_id}" data-field="confidence"><option value="">—</option>${[1,2,3,4,5].map(n=>option(String(n),String(n))).join('')}</select></label><label>Unrelated delivery interference<select data-trial-id="${trial.trial_id}" data-field="interference"><option value="">—</option>${option('none','None')}${option('manageable','Manageable')}${option('dominant','Dominant')}${option('uncertain','Uncertain')}</select></label><label>Artifact or audio defect<select data-trial-id="${trial.trial_id}" data-field="artifact"><option value="">—</option>${option('none','None')}${option('minor','Minor')}${option('major','Major')}${option('uncertain','Uncertain')}</select></label><label class="wide">Notes<textarea data-trial-id="${trial.trial_id}" data-field="notes" placeholder="What did you hear? Note localization difficulty or anything unusual."></textarea></label></div></section>`).join('');
const replayCount=id=>{const p=getTrial(id).play_starts;return Math.max(0,(p.A||0)-1)+Math.max(0,(p.B||0)-1)};
const updateReplay=id=>{document.querySelector(`[data-replays="${id}"]`).textContent=String(replayCount(id))};
const required=['difference_strength','category_judgment','confidence','interference','artifact'];
const updateCompletion=()=>{let complete=0;for(const trial of R){const row=getTrial(trial.trial_id);if(required.every(k=>row.responses[k])&&(row.play_starts.A||0)>0&&(row.play_starts.B||0)>0)complete++}document.getElementById('completion').textContent=`${complete}/${R.length} comparisons complete`;document.getElementById('download').disabled=complete!==R.length};
document.querySelectorAll('[data-field]').forEach(el=>{const row=getTrial(el.dataset.trialId);el.value=row.responses[el.dataset.field]??'';el.addEventListener('input',()=>{row.responses[el.dataset.field]=el.value;save();updateCompletion()})});
document.querySelectorAll('audio').forEach(audio=>{const playerBox=audio.closest('.player'),band=playerBox.querySelector('.target-band'),head=playerBox.querySelector('.playhead'),status=playerBox.querySelector('.target-status'),start=Number(audio.dataset.cueStart),end=Number(audio.dataset.cueEnd);const draw=()=>{if(!Number.isFinite(audio.duration)||audio.duration<=0)return;band.style.left=`${100*start/audio.duration}%`;band.style.width=`${100*(end-start)/audio.duration}%`;head.style.left=`${Math.min(100,100*audio.currentTime/audio.duration)}%`;const now=audio.currentTime>=start&&audio.currentTime<=end&&!audio.paused;playerBox.classList.toggle('target-now',now);status.textContent=now?'TARGET NOW':'Target position'};audio.addEventListener('loadedmetadata',draw);audio.addEventListener('timeupdate',draw);audio.addEventListener('pause',draw);audio.addEventListener('ended',draw);audio.addEventListener('play',()=>{document.querySelectorAll('audio').forEach(other=>{if(other!==audio&&!other.paused)other.pause()});const row=getTrial(audio.dataset.trial);const side=audio.dataset.side;row.play_starts[side]=(row.play_starts[side]||0)+1;save();updateReplay(audio.dataset.trial);updateCompletion();draw()})});
for(const trial of R)updateReplay(trial.trial_id);updateCompletion();save();
document.getElementById('download').addEventListener('click',()=>{const responses=R.map(trial=>{const row=getTrial(trial.trial_id);return{trial_id:trial.trial_id,difference_strength:Number(row.responses.difference_strength),category_judgment:row.responses.category_judgment,confidence:Number(row.responses.confidence),play_starts:row.play_starts,replay_count:replayCount(trial.trial_id),interference:row.responses.interference,artifact:row.responses.artifact,notes:row.responses.notes??''}});const payload={schema_version:1,run_id:'__RUN_ID__',protocol_sha256:PROTOCOL,session_id:S.session_id,saved_at:new Date().toISOString(),responses};const blob=new Blob([JSON.stringify(payload,null,2),String.fromCharCode(10)],{type:'application/json'});const link=document.createElement('a');link.href=URL.createObjectURL(blob);link.download='salience-attribution-v1-response.json';link.click()});
</script>
</body>
</html>
"""
    return (
        html.replace("__TRIALS__", json.dumps(public, ensure_ascii=False))
        .replace("__PROTOCOL__", protocol_sha256)
        .replace("__RUN_ID__", RUN_ID)
    )


def _ensure_opaque_links(layout: list[dict[str, Any]], run_dir: Path) -> None:
    parent = _parent_dir()
    for trial in layout:
        for side in trial["sides"]:
            source = parent / side["parent_audio_relative_path"]
            destination = run_dir / side["opaque_audio_relative_path"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            relative_target = os.path.relpath(source, destination.parent)
            if os.path.lexists(destination):
                if not destination.is_symlink() or os.readlink(destination) != relative_target:
                    raise RuntimeError(f"opaque audio path differs from freeze: {destination}")
            else:
                destination.symlink_to(relative_target)
            if sha256_file(destination) != side["audio_sha256"]:
                raise RuntimeError(f"opaque audio link hash mismatch: {destination}")


def prepare() -> dict[str, Any]:
    protocol = protocol_record()
    run_dir = Paths().artifacts / "phoneme-renderer" / RUN_ID
    protocol_path = run_dir / "protocol.json"
    if protocol_path.is_file():
        existing = json.loads(protocol_path.read_text(encoding="utf-8"))
        if stable_json(existing) != stable_json(protocol):
            raise RuntimeError("existing salience-attribution protocol differs from freeze")
    else:
        atomic_write_json(protocol_path, protocol)

    layout = blinded_layout(protocol)
    _ensure_opaque_links(layout, run_dir)
    blind_key = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "protocol_sha256": protocol["protocol_sha256"],
        "trials": [
            {
                "trial_id": trial["trial_id"],
                "logical_id": trial["logical_id"],
                "sides": {
                    side["side"]: {
                        "source_slot_id": side["source_slot_id"],
                        "audio_sha256": side["audio_sha256"],
                    }
                    for side in trial["sides"]
                },
            }
            for trial in layout
        ],
    }
    atomic_write_json(run_dir / "blind-key.json", blind_key)
    public = public_review_manifest(protocol)
    atomic_write_json(
        run_dir / "review-manifest.json",
        {
            "schema_version": 1,
            "run_id": RUN_ID,
            "protocol_sha256": protocol["protocol_sha256"],
            "trials": public,
        },
    )
    atomic_write_text(run_dir / "review.html", _review_html(public, protocol["protocol_sha256"]))
    return protocol


if __name__ == "__main__":
    result = prepare()
    print(json.dumps({"run_id": RUN_ID, "protocol_sha256": result["protocol_sha256"]}, indent=2))
