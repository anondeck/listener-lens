from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

from .config import Paths, sha256_json, stable_json
from .util import atomic_write_json, atomic_write_text, sha256_file


RUN_ID = "20260716-kokoro-stronger-span-product-qc-v1"
PARENT_RUN_ID = "20260716-kokoro-common-rng-confirmation-v4"
BLIND_SEED = 20_260_716_02
CUE_START_S = 1.425
CUE_END_S = 1.550
NEUTRAL_SLOT_ID = "common-neutral"
IDENTITY_SLOT_ID = "common-neutral-identity"
CANDIDATES = (
    ("target-word", "common-lens-target-word"),
    ("target-word-plus-boundaries", "common-lens-target-word-plus-boundaries"),
    ("full-contextual-state", "common-lens-full-contextual-state"),
)
SELECTION_ORDER = tuple(candidate_id for candidate_id, _ in CANDIDATES)
EXCLUDED = (
    {
        "candidate_id": "target-only",
        "source_slot_id": "common-lens-target-only",
        "reason": "frozen automatic acoustic failure; listening cannot rescue it",
    },
    {
        "candidate_id": "stress-plus-target",
        "source_slot_id": "common-lens-stress-plus-target",
        "reason": "completed frozen manual QC at 4/7; omitted with no repeat or reliability trial",
    },
)


def _parent_dir() -> Path:
    return Paths().artifacts / "phoneme-renderer" / PARENT_RUN_ID


def _parent_records() -> dict[str, dict[str, Any]]:
    rows = json.loads((_parent_dir() / "records.json").read_text(encoding="utf-8"))
    records = {row["slot_id"]: row for row in rows}
    required = {
        NEUTRAL_SLOT_ID,
        IDENTITY_SLOT_ID,
        *(slot_id for _, slot_id in CANDIDATES),
        *(row["source_slot_id"] for row in EXCLUDED),
    }
    missing = required - records.keys()
    if missing:
        raise RuntimeError(
            f"frozen v4 records are missing source slots: {sorted(missing)}"
        )
    return records


def _parent_summary() -> dict[str, Any]:
    return json.loads((_parent_dir() / "summary.json").read_text(encoding="utf-8"))


def _verify_shared_alignment(record: dict[str, Any]) -> dict[str, Any]:
    alignment = record["target"]["alignment"]
    if alignment["start_s"] != CUE_START_S or alignment["end_s"] != CUE_END_S:
        raise RuntimeError(
            f"source slot does not inherit the shared target interval: {record['slot_id']}"
        )
    return alignment


def _source_record(
    slot_id: str, *, role: str, candidate_id: str | None = None
) -> dict[str, Any]:
    parent = _parent_dir()
    record = _parent_records()[slot_id]
    path = parent / record["audio_relative_path"]
    actual_hash = sha256_file(path)
    if actual_hash != record["audio_sha256"]:
        raise RuntimeError(f"frozen source WAV hash mismatch: {slot_id}")
    alignment = _verify_shared_alignment(record)
    payload: dict[str, Any] = {
        "role": role,
        "slot_id": slot_id,
        "parent_audio_relative_path": record["audio_relative_path"],
        "audio_sha256": actual_hash,
        "sample_rate_hz": record["timing"]["sample_rate_hz"],
        "decoded_sample_count": record["timing"]["decoded_sample_count"],
        "target_start_s": alignment["start_s"],
        "target_end_s": alignment["end_s"],
    }
    if candidate_id is not None:
        pair_result = _parent_summary()["pair_results"][candidate_id]
        if pair_result["pass"] is not True:
            raise RuntimeError(
                f"candidate did not pass the frozen automatic gate: {candidate_id}"
            )
        localization = pair_result["difference_localization"]
        payload.update(
            {
                "candidate_id": candidate_id,
                "automatic_acoustic_pass": True,
                "known_descriptive_metrics": {
                    "inside_difference_energy_fraction": localization[
                        "inside_difference_energy_fraction"
                    ],
                    "inside_window_start_s": localization["inside_window_start_s"],
                    "inside_window_end_s": localization["inside_window_end_s"],
                    "outside_rms_pcm": localization["outside_rms_pcm"],
                    "maximum_absolute_pcm_delta": localization[
                        "maximum_absolute_pcm_delta"
                    ],
                    "mean_absolute_pcm_delta": localization["mean_absolute_pcm_delta"],
                    "sample_count_equal": localization["sample_count_equal"],
                    "selection_authority": False,
                    "threshold": None,
                },
            }
        )
    return payload


def _source_manifest() -> list[dict[str, Any]]:
    return [
        _source_record(NEUTRAL_SLOT_ID, role="shared-neutral"),
        _source_record(IDENTITY_SLOT_ID, role="bit-identical-control"),
        *(
            _source_record(
                slot_id, role="eligible-stronger-candidate", candidate_id=candidate_id
            )
            for candidate_id, slot_id in CANDIDATES
        ),
    ]


def protocol_record() -> dict[str, Any]:
    parent = _parent_dir()
    parent_protocol = json.loads((parent / "protocol.json").read_text(encoding="utf-8"))
    source_wavs = _source_manifest()
    neutral = next(row for row in source_wavs if row["slot_id"] == NEUTRAL_SLOT_ID)
    identity = next(row for row in source_wavs if row["slot_id"] == IDENTITY_SLOT_ID)
    if neutral["audio_sha256"] != identity["audio_sha256"]:
        raise RuntimeError("frozen identity control is not bit-identical")

    payload = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "status": "frozen_before_one_informed_creator_product_qc_session",
        "question": (
            "Does the first eligible stronger shared-state span in the fixed selection order satisfy every frozen "
            "manual product gate against the same v4 neutral?"
        ),
        "scope": {
            "sessions": 1,
            "listener": "informed creator-listener",
            "interpretation": "product-tuning evidence only",
            "independent_evidence": False,
            "population_evidence": False,
            "brazilian_portuguese_listener_validation": False,
            "can_reclassify_parent_v4": False,
            "api_calls": 0,
            "new_renders": 0,
            "audio_edits": 0,
            "source_audio": "existing frozen v4 WAVs and hashes only",
        },
        "parent": {
            "run_id": PARENT_RUN_ID,
            "protocol_sha256": parent_protocol["protocol_sha256"],
            "protocol_file_sha256": sha256_file(parent / "protocol.json"),
            "records_sha256": sha256_file(parent / "records.json"),
            "summary_sha256": sha256_file(parent / "summary.json"),
            "manual_result_sha256": sha256_file(parent / "manual-result.json"),
            "classification": (
                "controlled, correctly directed perceptual near-pass; not a full manual-gate pass"
            ),
            "classification_is_immutable": True,
        },
        "candidate_set": {
            "selection_order": list(SELECTION_ORDER),
            "candidates": [
                {
                    "candidate_id": candidate_id,
                    "neutral_slot_id": NEUTRAL_SLOT_ID,
                    "lens_slot_id": slot_id,
                }
                for candidate_id, slot_id in CANDIDATES
            ],
            "excluded": list(EXCLUDED),
        },
        "source_wavs": source_wavs,
        "trial_contract": {
            "logical_trials": [
                *(
                    {
                        "logical_id": candidate_id,
                        "source_slots": [NEUTRAL_SLOT_ID, slot_id],
                    }
                    for candidate_id, slot_id in CANDIDATES
                ),
                {
                    "logical_id": "identity-control",
                    "source_slots": [NEUTRAL_SLOT_ID, IDENTITY_SLOT_ID],
                },
            ],
            "blinding_seed": BLIND_SEED,
            "trial_order": "deterministically shuffled",
            "side_order": "A/B independently and deterministically shuffled within each trial",
            "hidden": [
                "candidate identity",
                "source slots",
                "phonemes",
                "filenames",
                "conditions",
                "blind key",
            ],
            "all_ratings_before_decode": True,
            "raw_response_preservation": "byte-for-byte; decoding belongs in a separate result artifact",
        },
        "cue_contract": {
            "target_start_s": CUE_START_S,
            "target_end_s": CUE_END_S,
            "sides_use_union_of_frozen_target_intervals": True,
            "same_interval_for_every_trial": True,
            "latent_span_width_exposed": False,
            "cropping": False,
            "looping": False,
            "editing": False,
            "isolated_replay": False,
            "audio_preprocessing": False,
            "replay_starts": "descriptive telemetry only",
        },
        "response_schema": {
            "per_side": {
                "naturalness": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 5,
                    "pass": [4, 5],
                },
                "delivery": {
                    "enum": [
                        "sentence-like",
                        "slightly list-like",
                        "dominantly list-like",
                        "other",
                    ],
                    "pass": ["sentence-like"],
                },
                "stable_recoverable_meaning": {
                    "enum": [
                        "none",
                        "isolated possible word",
                        "coherent phrase",
                        "clear source sentence",
                    ],
                    "pass": ["none"],
                },
                "artifact": {
                    "enum": ["none", "minor", "major", "uncertain"],
                    "pass": ["none", "minor"],
                },
            },
            "pair": {
                "difference_strength": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 7,
                    "anchors": {
                        "1": "no audible difference",
                        "4": "moderate",
                        "7": "very strong",
                    },
                    "candidate_pass": ">=5",
                },
                "direction": {
                    "enum": ["A", "B", "same", "uncertain", "neither"],
                    "candidate_pass": "expected lens side as closer to the vowel in bet",
                },
                "confidence": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 5,
                    "candidate_pass": ">=3",
                },
                "unrelated_interference": {
                    "enum": ["none", "manageable", "dominant", "uncertain"],
                    "candidate_pass": ["none", "manageable"],
                },
                "notes": {"type": "string", "required": False},
                "replay_count": {"type": "integer", "selection_authority": False},
            },
        },
        "manual_gate": {
            "candidate_gate": (
                "conjunctive across both side-level naturalness, delivery, opacity, and artifact gates plus "
                "pair-level strength, expected direction, confidence, and interference gates; no compensation"
            ),
            "catch_clean": {
                "rule": "difference strength == 1 and direction == neither",
                "interpretation": "No false alarm occurred on this session's catch trial.",
            },
            "catch_flagged": {
                "rule": "difference strength >=2 or direction is A or B",
                "interpretation": (
                    "Do not invalidate or characterize the listener; bar the session from promoting any stronger "
                    "candidate."
                ),
            },
            "selection": (
                "after all ratings are frozen and decoded, select the first complete candidate pass in the fixed "
                "selection order; never select the largest effect"
            ),
            "known_localization_metrics": (
                "report every candidate's localization fraction and outside RMS PCM with no threshold and no "
                "selection authority"
            ),
        },
        "predetermined_outcomes": {
            "clean_catch_candidate_passes": (
                "freeze the selected span as the versioned candidate configuration for replication; it becomes "
                "the production choice only after a separately frozen typed-engine replication/generalization "
                "protocol reproduces it across 2-3 engine-generated carrier fixtures"
            ),
            "clean_catch_no_candidate_passes": (
                "close this Build Week ae-to-eh span-strength search; preserve stress-plus-target as the documented "
                "controlled correctly directed perceptual near-pass and handle salience through transparent UX "
                "without causal claims; do not close other rules or post-hackathon research"
            ),
            "catch_flagged": (
                "classify control-flagged/inconclusive, promote no stronger candidate, rerun no ratings, and retain "
                "stress-plus-target as the provisional engineering fallback without new supporting evidence"
            ),
        },
        "engine_sequence": {
            "scaffolding_before_outcome": "permitted only if span-parameterized",
            "fixtures_before_outcome": "forbidden",
            "after_outcome_1": "fixtures use the selected stronger candidate",
            "after_outcome_2_or_3": (
                "fixtures use stress-plus-target while retaining its controlled correctly directed perceptual "
                "near-pass status"
            ),
        },
        "stopping_rule": (
            "Freeze the protocol and build the review package, but do not open it or begin listening. Stop for a "
            "read-only protocol check before the one authorized session."
        ),
    }
    return {**payload, "protocol_sha256": sha256_json(payload)}


def blinded_layout(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    sources = {row["slot_id"]: row for row in protocol["source_wavs"]}
    rng = random.Random(BLIND_SEED)
    pending: list[dict[str, Any]] = []
    for trial in protocol["trial_contract"]["logical_trials"]:
        slots = list(trial["source_slots"])
        rng.shuffle(slots)
        pending.append({"logical_id": trial["logical_id"], "source_slots": slots})
    rng.shuffle(pending)

    layout: list[dict[str, Any]] = []
    for index, row in enumerate(pending, start=1):
        trial_id = f"comparison-{index:02d}"
        sides = []
        for side, slot_id in zip(("A", "B"), row["source_slots"], strict=True):
            source = sources[slot_id]
            sides.append(
                {
                    "side": side,
                    "source_slot_id": slot_id,
                    "audio_sha256": source["audio_sha256"],
                    "parent_audio_relative_path": source["parent_audio_relative_path"],
                    "opaque_audio_relative_path": f"audio/{trial_id}-{side.lower()}.wav",
                }
            )
        layout.append(
            {
                "trial_id": trial_id,
                "logical_id": row["logical_id"],
                "cue_start_s": CUE_START_S,
                "cue_end_s": CUE_END_S,
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
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Blind product QC</title>
<style>
:root{color-scheme:light;--ink:#17221c;--muted:#59665f;--paper:#f5f2e9;--card:#fff;--line:#d7d4ca;--target:#d87b35;--now:#16704f}
*{box-sizing:border-box}body{font:17px/1.5 system-ui,-apple-system,sans-serif;max-width:980px;margin:auto;padding:26px;background:var(--paper);color:var(--ink)}
h1{font-size:2rem;margin-bottom:.3rem}.intro{max-width:820px}.trial{background:var(--card);border:1px solid var(--line);border-radius:18px;padding:22px;margin:24px 0;box-shadow:0 2px 9px #17221c0a}
.pair{display:grid;grid-template-columns:1fr 1fr;gap:16px}.player{border:1px solid var(--line);border-radius:14px;padding:15px;background:#fbfbf8}.player h3{margin:.1rem 0 .6rem}audio{width:100%}
.positions{display:flex;gap:6px;margin:13px 0 8px}.positions i{width:15px;height:15px;border-radius:50%;background:#d6dcd7}.positions i:nth-child(8){background:var(--target);outline:2px solid #f0c49e}.target-now .positions i:nth-child(8){background:var(--now);outline:4px solid #9ed3bf}
.timeline{height:13px;border-radius:999px;background:#e4e8e4;position:relative;overflow:hidden}.target-band{position:absolute;top:0;bottom:0;background:#e8a36eaa}.playhead{position:absolute;top:0;bottom:0;width:3px;background:#17221c;left:0}.target-status{font-size:.85rem;color:var(--muted);font-weight:700;margin-top:5px}.target-now .target-status{color:var(--now)}
.fields{display:grid;grid-template-columns:1fr 1fr;gap:12px 18px;margin-top:14px}label{display:block;font-weight:650}select,textarea{width:100%;font:inherit;margin-top:5px;padding:9px;border:1px solid #b9c0ba;border-radius:9px;background:white}textarea{min-height:80px}.wide{grid-column:1/-1}.replays,.scale{font-size:.88rem;color:var(--muted);font-weight:400}.pair-fields{border-top:1px solid var(--line);margin-top:20px;padding-top:16px}
.download{padding:12px 20px;border:0;border-radius:999px;background:#154f3e;color:white;font-weight:750;font-size:1rem}.download:disabled{opacity:.42}.completion{margin-left:12px;color:var(--muted)}
@media(max-width:760px){.pair,.fields{grid-template-columns:1fr}.wide{grid-column:auto}}
</style></head>
<body><h1>Blind product QC</h1>
<p class="intro">Complete every rating before downloading. Candidate identities, conditions, spellings, filenames, and comparison types are hidden. The orange eighth position is the target in every clip; <strong>TARGET NOW</strong> marks the same frozen interval for every side.</p>
<p class="intro"><strong>Meaning</strong> asks whether you can recover stable English words or a coherent English phrase—not whether the audio merely sounds sentence-like. Replay counts are descriptive only.</p>
<main id="trials"></main><button class="download" id="download" disabled>Download response.json</button><span class="completion" id="completion"></span>
<script>
const R=__TRIALS__,PROTOCOL='__PROTOCOL__',KEY='kokoro-stronger-span-product-qc-v1';
const fresh=()=>({session_id:(globalThis.crypto&&crypto.randomUUID)?crypto.randomUUID():'session-'+Date.now()+'-'+Math.random().toString(16).slice(2),protocol_sha256:PROTOCOL,trials:{}});
let S;try{S=JSON.parse(localStorage.getItem(KEY))}catch(_){S=null}if(!S||S.protocol_sha256!==PROTOCOL)S=fresh();
const save=()=>localStorage.setItem(KEY,JSON.stringify(S));
const getTrial=id=>S.trials[id]??=({sides:{A:{},B:{}},pair:{},play_starts:{A:0,B:0}});
const option=(value,label)=>`<option value="${value}">${label}</option>`;
const positions='<div class="positions" aria-label="Ten spoken positions; position eight is the target">'+Array.from({length:10},()=>'<i></i>').join('')+'</div>';
const sideFields=(trial,side)=>`<div class="fields"><label>Naturalness <span class="scale">1 poor · 5 fully natural</span><select data-trial-id="${trial.trial_id}" data-side="${side}" data-field="naturalness"><option value="">—</option>${[1,2,3,4,5].map(n=>option(String(n),String(n))).join('')}</select></label><label>Delivery<select data-trial-id="${trial.trial_id}" data-side="${side}" data-field="delivery"><option value="">—</option>${option('sentence-like','Sentence-like')}${option('slightly list-like','Slightly list-like')}${option('dominantly list-like','Dominantly list-like')}${option('other','Other')}</select></label><label>Stable recoverable meaning<select data-trial-id="${trial.trial_id}" data-side="${side}" data-field="meaning"><option value="">—</option>${option('none','None')}${option('isolated possible word','Isolated possible word')}${option('coherent phrase','Coherent phrase')}${option('clear source sentence','Clear source sentence')}</select></label><label>Artifact<select data-trial-id="${trial.trial_id}" data-side="${side}" data-field="artifact"><option value="">—</option>${option('none','None')}${option('minor','Minor')}${option('major','Major')}${option('uncertain','Uncertain')}</select></label></div>`;
const player=(trial,side)=>`<section class="player" data-player="${trial.trial_id}-${side.side}"><h3>Clip ${side.side}</h3><audio controls preload="auto" data-trial="${trial.trial_id}" data-side="${side.side}" data-cue-start="${trial.cue_start_s}" data-cue-end="${trial.cue_end_s}" src="${side.audio}"></audio>${positions}<div class="timeline" aria-hidden="true"><span class="target-band"></span><i class="playhead"></i></div><div class="target-status" aria-live="polite">Target position</div>${sideFields(trial,side.side)}</section>`;
const pairFields=trial=>`<div class="fields pair-fields"><label>Difference strength <span class="scale">1 none · 4 moderate · 7 very strong</span><select data-trial-id="${trial.trial_id}" data-pair data-field="difference_strength"><option value="">—</option>${[1,2,3,4,5,6,7].map(n=>option(String(n),String(n))).join('')}</select></label><label>Which side, if either, sounds closer to the vowel in “bet”?<select data-trial-id="${trial.trial_id}" data-pair data-field="direction"><option value="">—</option>${option('A','A')}${option('B','B')}${option('same','Same / no category difference')}${option('uncertain','Uncertain')}${option('neither','Neither')}</select></label><label>Confidence <span class="scale">1 guessing · 5 highly confident</span><select data-trial-id="${trial.trial_id}" data-pair data-field="confidence"><option value="">—</option>${[1,2,3,4,5].map(n=>option(String(n),String(n))).join('')}</select></label><label>Unrelated delivery interference<select data-trial-id="${trial.trial_id}" data-pair data-field="interference"><option value="">—</option>${option('none','None')}${option('manageable','Manageable')}${option('dominant','Dominant')}${option('uncertain','Uncertain')}</select></label><label class="wide">Notes<textarea data-trial-id="${trial.trial_id}" data-pair data-field="notes" placeholder="Optional notes"></textarea></label></div>`;
document.getElementById('trials').innerHTML=R.map((trial,index)=>`<section class="trial"><h2>Comparison ${index+1} of ${R.length}</h2><div class="pair">${trial.sides.map(side=>player(trial,side)).join('')}</div><p class="replays">Recorded replays: <strong data-replays="${trial.trial_id}">0</strong></p>${pairFields(trial)}</section>`).join('');
const replayCount=id=>{const p=getTrial(id).play_starts;return Math.max(0,(p.A||0)-1)+Math.max(0,(p.B||0)-1)};
const updateReplay=id=>document.querySelector(`[data-replays="${id}"]`).textContent=String(replayCount(id));
const sideRequired=['naturalness','delivery','meaning','artifact'],pairRequired=['difference_strength','direction','confidence','interference'];
const updateCompletion=()=>{let complete=0;for(const trial of R){const row=getTrial(trial.trial_id);if(['A','B'].every(side=>sideRequired.every(k=>row.sides[side][k]))&&pairRequired.every(k=>row.pair[k])&&(row.play_starts.A||0)>0&&(row.play_starts.B||0)>0)complete++}document.getElementById('completion').textContent=`${complete}/${R.length} comparisons complete`;document.getElementById('download').disabled=complete!==R.length};
document.querySelectorAll('[data-field]').forEach(el=>{const row=getTrial(el.dataset.trialId);const target=el.hasAttribute('data-pair')?row.pair:row.sides[el.dataset.side];el.value=target[el.dataset.field]??'';el.addEventListener('input',()=>{target[el.dataset.field]=el.value;save();updateCompletion()})});
document.querySelectorAll('audio').forEach(audio=>{const box=audio.closest('.player'),band=box.querySelector('.target-band'),head=box.querySelector('.playhead'),status=box.querySelector('.target-status'),start=Number(audio.dataset.cueStart),end=Number(audio.dataset.cueEnd);const draw=()=>{if(!Number.isFinite(audio.duration)||audio.duration<=0)return;band.style.left=`${100*start/audio.duration}%`;band.style.width=`${100*(end-start)/audio.duration}%`;head.style.left=`${Math.min(100,100*audio.currentTime/audio.duration)}%`;const now=audio.currentTime>=start&&audio.currentTime<=end&&!audio.paused;box.classList.toggle('target-now',now);status.textContent=now?'TARGET NOW':'Target position'};audio.addEventListener('loadedmetadata',draw);audio.addEventListener('timeupdate',draw);audio.addEventListener('pause',draw);audio.addEventListener('ended',draw);audio.addEventListener('play',()=>{document.querySelectorAll('audio').forEach(other=>{if(other!==audio&&!other.paused)other.pause()});const row=getTrial(audio.dataset.trial),side=audio.dataset.side;row.play_starts[side]=(row.play_starts[side]||0)+1;save();updateReplay(audio.dataset.trial);updateCompletion();draw()})});
for(const trial of R)updateReplay(trial.trial_id);updateCompletion();save();
document.getElementById('download').addEventListener('click',()=>{const responses=R.map(trial=>{const row=getTrial(trial.trial_id);return{trial_id:trial.trial_id,sides:{A:{naturalness:Number(row.sides.A.naturalness),delivery:row.sides.A.delivery,stable_recoverable_meaning:row.sides.A.meaning,artifact:row.sides.A.artifact},B:{naturalness:Number(row.sides.B.naturalness),delivery:row.sides.B.delivery,stable_recoverable_meaning:row.sides.B.meaning,artifact:row.sides.B.artifact}},pair:{difference_strength:Number(row.pair.difference_strength),direction:row.pair.direction,confidence:Number(row.pair.confidence),unrelated_interference:row.pair.interference,notes:row.pair.notes??''},play_starts:row.play_starts,replay_count:replayCount(trial.trial_id)}});const payload={schema_version:1,run_id:'__RUN_ID__',protocol_sha256:PROTOCOL,session_id:S.session_id,saved_at:new Date().toISOString(),responses};const blob=new Blob([JSON.stringify(payload,null,2),String.fromCharCode(10)],{type:'application/json'}),link=document.createElement('a');link.href=URL.createObjectURL(blob);link.download='stronger-span-product-qc-v1-response.json';link.click()});
</script></body></html>"""
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
                if (
                    not destination.is_symlink()
                    or os.readlink(destination) != relative_target
                ):
                    raise RuntimeError(
                        f"opaque audio path differs from freeze: {destination}"
                    )
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
            raise RuntimeError(
                "existing stronger-span product-QC protocol differs from freeze"
            )
    else:
        atomic_write_json(protocol_path, protocol)

    layout = blinded_layout(protocol)
    _ensure_opaque_links(layout, run_dir)
    blind_key = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "protocol_sha256": protocol["protocol_sha256"],
        "selection_order": list(SELECTION_ORDER),
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
    atomic_write_text(
        run_dir / "review.html", _review_html(public, protocol["protocol_sha256"])
    )
    return protocol


if __name__ == "__main__":
    result = prepare()
    print(
        json.dumps(
            {"run_id": RUN_ID, "protocol_sha256": result["protocol_sha256"]}, indent=2
        )
    )
