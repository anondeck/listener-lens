from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import shutil
from typing import Any, Iterable
import wave

import numpy as np

from earshift_bakeoff.config import ROOT, stable_json
from earshift_bakeoff.util import atomic_write_json, atomic_write_text, sha256_file


PROTOCOL_VERSION = "bilingual-vowel-unseen-blind-qc-v1"
RUN_ID = "20260718-bilingual-vowel-unseen-blind-qc-v1"
PROTOCOL_PATH = ROOT / "rules" / f"{PROTOCOL_VERSION}.json"
RUN_DIR = ROOT / "artifacts" / "product-matrix" / RUN_ID
SESSION_LABELS = ("session-a", "session-b", "session-c", "session-d")
CONTEXT_ORDER = (
    "real_g2p_phrase_medial",
    "real_g2p_phrase_final",
    "real_g2p_repeated_target",
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _semantic_hash(payload: dict[str, Any]) -> str:
    semantic = dict(payload)
    semantic.pop("record_sha256", None)
    return hashlib.sha256(stable_json(semantic).encode("utf-8")).hexdigest()


def _record(payload: dict[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result["record_sha256"] = _semantic_hash(result)
    return result


def load_protocol() -> dict[str, Any]:
    protocol = _load_json(PROTOCOL_PATH)
    if (
        protocol.get("schema_version") != 1
        or protocol.get("protocol_version") != PROTOCOL_VERSION
        or protocol.get("status") != "frozen_before_first_human_review"
        or protocol.get("production_enabled") is not False
    ):
        raise RuntimeError("unseen blind-QC protocol drifted")
    for label in ("unseen_confirmation", "typed_manifest", "automatic_protocol"):
        path = ROOT / protocol["parent_bindings"][f"{label}_path"]
        if sha256_file(path) != protocol["parent_bindings"][f"{label}_sha256"]:
            raise RuntimeError(f"unseen blind-QC parent drifted: {label}")
    return protocol


def _read_wav(path: Path) -> tuple[int, np.ndarray]:
    with wave.open(str(path), "rb") as handle:
        if handle.getnchannels() != 1 or handle.getsampwidth() != 2:
            raise RuntimeError(f"blind QC requires mono PCM16: {path}")
        sample_rate = handle.getframerate()
        pcm = np.frombuffer(
            handle.readframes(handle.getnframes()), dtype="<i2"
        ).copy()
    if sample_rate <= 0 or pcm.size == 0:
        raise RuntimeError(f"blind QC found invalid WAV: {path}")
    return sample_rate, pcm


def cue_intervals_from_difference(
    neutral_path: Path,
    lens_path: Path,
    *,
    merge_gap_ms: float = 20.0,
    pad_ms: float = 40.0,
) -> tuple[float, list[dict[str, float]]]:
    sample_rate, neutral = _read_wav(neutral_path)
    lens_rate, lens = _read_wav(lens_path)
    if lens_rate != sample_rate or lens.shape != neutral.shape:
        raise RuntimeError("blind-QC pair has unequal sample shape or rate")
    changed = np.flatnonzero(neutral != lens)
    if changed.size == 0:
        raise RuntimeError("candidate comparison contains no changed PCM")

    gap_samples = round(sample_rate * merge_gap_ms / 1000.0)
    pad_samples = round(sample_rate * pad_ms / 1000.0)
    runs: list[tuple[int, int]] = []
    start = previous = int(changed[0])
    for value in changed[1:]:
        current = int(value)
        if current - previous > gap_samples:
            runs.append((start, previous + 1))
            start = current
        previous = current
    runs.append((start, previous + 1))

    padded: list[tuple[int, int]] = []
    for start, end in runs:
        start = max(0, start - pad_samples)
        end = min(neutral.size, end + pad_samples)
        if padded and start <= padded[-1][1]:
            padded[-1] = (padded[-1][0], max(padded[-1][1], end))
        else:
            padded.append((start, end))
    intervals = [
        {
            "start_s": round(start / sample_rate, 6),
            "end_s": round(end / sample_rate, 6),
        }
        for start, end in padded
    ]
    return neutral.size / sample_rate, intervals


def _blind_id(seed: int, *parts: str) -> str:
    material = "|".join((str(seed), *parts)).encode("utf-8")
    return "trial-" + hashlib.sha256(material).hexdigest()[:16]


def _session_voice_map(protocol: dict[str, Any]) -> dict[str, str]:
    voices = sorted(protocol["scope"]["voice_session_counts"])
    random.Random(protocol["presentation"]["session_label_seed"]).shuffle(voices)
    return dict(zip(SESSION_LABELS, voices, strict=True))


def _identity_context(cell_id: str) -> str:
    index = int(hashlib.sha256(cell_id.encode("utf-8")).hexdigest(), 16) % len(
        CONTEXT_ORDER
    )
    return CONTEXT_ORDER[index]


def _passing_cells(result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    cells = {
        row["cell_id"]: row
        for row in result["cell_summaries"]
        if row["unseen_automatic_pass"]
        and row["blind_human_qc_eligible"]
        and row["replicated_anchor"]["directional_pass"]
        and row["product_enabled"] is False
    }
    if len(cells) != 18:
        raise RuntimeError("blind-QC automatic candidate denominator drifted")
    return cells


def _selected_outcomes(
    result: dict[str, Any], cells: dict[str, dict[str, Any]]
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in result["outcomes"]:
        if row["cell_id"] not in cells:
            continue
        if (
            not row["aggregate"]["directional_pass"]
            or not row["candidate_integrity"]["integrity_pass"]
            or row["audio"]["lens"] is None
            or row["product_enabled"] is not False
        ):
            raise RuntimeError("blind-QC candidate lost an automatic gate")
        grouped[row["cell_id"]].append(row)
    for cell_id, rows in grouped.items():
        rows.sort(key=lambda row: CONTEXT_ORDER.index(row["context"]))
        if [row["context"] for row in rows] != list(CONTEXT_ORDER):
            raise RuntimeError(f"blind-QC contexts drifted: {cell_id}")
    if set(grouped) != set(cells):
        raise RuntimeError("blind-QC result is missing a passing cell")
    return grouped


def _copy_blind_audio(source: Path, target: Path) -> dict[str, Any]:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    if sha256_file(source) != sha256_file(target):
        raise RuntimeError("blind audio copy changed its source")
    return {
        "source_path": str(source.relative_to(ROOT)),
        "source_sha256": sha256_file(source),
        "blind_path": str(target.relative_to(RUN_DIR)),
        "blind_sha256": sha256_file(target),
    }


def _html(
    *,
    session_id: str,
    session_number: int,
    public_trials: list[dict[str, Any]],
    protocol_sha256: str,
    public_manifest_sha256: str,
    response_filename: str,
) -> str:
    public = json.dumps(public_trials, ensure_ascii=False).replace("</", "<\\/")
    return r'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="Content-Security-Policy" content="default-src 'self'; media-src 'self'; script-src 'unsafe-inline'; style-src 'unsafe-inline'"><title>Blind vowel QC</title><style>:root{font-family:ui-sans-serif,system-ui,sans-serif;color:#18221d;background:#f3f0e8}*{box-sizing:border-box}body{margin:0}.wrap{max-width:980px;margin:auto;padding:26px 18px 90px}.intro,.trial,.side,.meta{background:#fff;border:1px solid #d6d1c6;border-radius:18px;padding:20px;margin:16px 0;box-shadow:0 8px 25px #302d2410}h1{font-size:clamp(2rem,6vw,4rem);line-height:1;letter-spacing:-.045em;margin:.25em 0}.lede,.muted{color:#5d675f}.grid,.pair{display:grid;grid-template-columns:1fr 1fr;gap:14px}.side{margin:0;box-shadow:none}.timeline{height:17px;background:#dce1dc;border-radius:999px;position:relative;margin:11px 0 7px;overflow:hidden}.cue{position:absolute;top:0;bottom:0;background:#da7b37}.head{position:absolute;top:0;bottom:0;width:2px;background:#174b38}.player.active .timeline{outline:3px solid #e7a878}.target-status{min-height:1.4em;font-size:.9rem;font-weight:750;color:#84502b}audio,select,textarea,input{width:100%;font:inherit;box-sizing:border-box}select,textarea,input{padding:9px;border:1px solid #aaa69b;border-radius:9px;background:#fff}label{display:block;font-weight:700;margin:11px 0}textarea{min-height:70px}.progress{position:sticky;top:0;z-index:3;background:#f3f0e8ee;backdrop-filter:blur(7px);padding:10px 0}.bar{height:9px;background:#d7dcd7;border-radius:99px;overflow:hidden}.fill{height:100%;width:0;background:#174b38}.actions{position:sticky;bottom:0;background:#f3f0e8ee;backdrop-filter:blur(7px);padding:13px 0}button{font:inherit;font-weight:800;color:#fff;background:#174b38;border:0;border-radius:999px;padding:12px 18px}button:disabled{opacity:.42}.error{color:#9b2c2c;font-weight:750}.anchor-note{padding:10px 12px;background:#eef2ec;border-radius:10px}@media(max-width:720px){.grid,.pair{grid-template-columns:1fr}}</style></head><body><main class="wrap"><section class="intro"><p><b>Blinded vowel QC · Session __SESSION_NUMBER__ of 4</b></p><h1>Listen for one controlled change.</h1><p class="lede">Judge each clip on its own before comparing A with B. Conditions, source text, carrier spelling, target word, rule, voice name, and filenames are hidden. Orange bands mark the same target positions in every condition.</p><p class="anchor-note">The category question names a reference vowel only; it is not a transcription of the hidden carrier. A bit-identical control may appear anywhere.</p></section><section class="meta"><div class="grid"><label>Reviewer code<input id="reviewer" autocomplete="off" placeholder="e.g. max"></label><label>Language background<input id="background" autocomplete="off" placeholder="e.g. native English; conversational Portuguese"></label><label>Listening setup<select id="setup"><option value="">—</option><option value="speakers">Speakers</option><option value="headphones">Headphones</option><option value="other">Other</option></select></label></div></section><div class="progress"><p><b id="progressText">0 complete</b></p><div class="bar"><div class="fill" id="fill"></div></div></div><div id="trials"></div><p id="error" class="error"></p><div class="actions"><button id="download" disabled>Download this session’s response</button></div></main><script>const T=__TRIALS__;const RUN='__RUN_ID__',SESSION='__SESSION_ID__',PROTOCOL='__PROTOCOL__',PUBLIC='__PUBLIC__',F='__RESPONSE__',K=`${RUN}:${SESSION}`;let S=JSON.parse(localStorage.getItem(K)||'{"meta":{},"ratings":{}}');S.meta??={};S.ratings??={};S.session_uuid??=crypto.randomUUID();const save=()=>localStorage.setItem(K,JSON.stringify(S));const esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'}[c]));const options=(items,value)=>'<option value="">—</option>'+items.map(([v,l])=>`<option value="${v}" ${v===value?'selected':''}>${l}</option>`).join('');const state=id=>(S.ratings[id]??={sides:{A:{},B:{}},pair:{},play_starts:{A:0,B:0}});const sel=(id,scope,side,field,label,items)=>{const x=state(id),target=scope==='side'?x.sides[side]:x.pair;return `<label>${label}<select data-id="${id}" data-scope="${scope}" data-side="${side}" data-field="${field}">${options(items,target[field]??'')}</select></label>`};const side=(t,name)=>{const x=state(t.trial_id).sides[name];return `<section class="side"><h3>Clip ${name}</h3><div class="player"><audio controls preload="metadata" src="${t.audio[name]}" data-id="${t.trial_id}" data-side="${name}"></audio><div class="timeline">${t.target_intervals.map(v=>`<i class="cue" style="left:${100*v.start_s/t.duration_s}%;width:${100*(v.end_s-v.start_s)/t.duration_s}%"></i>`).join('')}<i class="head"></i></div><div class="target-status">Target position</div></div>${sel(t.trial_id,'side',name,'naturalness','Naturalness (1 unusable · 5 fully natural)',[['1','1'],['2','2'],['3','3'],['4','4'],['5','5']])}${sel(t.trial_id,'side',name,'sentence_delivery','Sentence-like delivery',[['sentence_like','Sentence-like'],['partly_sentence_like','Partly sentence-like'],['not_sentence_like','Not sentence-like']])}${sel(t.trial_id,'side',name,'stable_meaning','Stable recoverable meaning',[['none','None'],['isolated_possible_word','Isolated possible word'],['coherent_phrase','Coherent phrase'],['clear_source_sentence','Clear source sentence']])}${sel(t.trial_id,'side',name,'artifact','Artifact or defect',[['none','None'],['minor','Minor'],['major','Major'],['uncertain','Uncertain']])}</section>`};document.getElementById('trials').innerHTML=T.map((t,i)=>`<article class="trial"><h2>Comparison ${i+1} of ${T.length}</h2><div class="pair">${side(t,'A')}${side(t,'B')}</div><p class="muted">Total play starts: <b data-plays="${t.trial_id}">0</b></p>${sel(t.trial_id,'pair','','difference_strength','Difference strength (1 none · 7 very strong)',[['1','1'],['2','2'],['3','3'],['4','4'],['5','5'],['6','6'],['7','7']])}${sel(t.trial_id,'pair','','target_direction',esc(t.direction_prompt),[['A','A'],['B','B'],['same','Same'],['uncertain','Uncertain']])}${sel(t.trial_id,'pair','','confidence','Confidence (1 guessing · 5 highly confident)',[['1','1'],['2','2'],['3','3'],['4','4'],['5','5']])}${sel(t.trial_id,'pair','','unrelated_interference','Did unrelated delivery differences interfere?',[['none','None'],['manageable','Manageable'],['dominant','Dominant'],['uncertain','Uncertain']])}<label>Notes (optional)<textarea data-id="${t.trial_id}" data-scope="pair" data-side="" data-field="notes">${esc(state(t.trial_id).pair.notes??'')}</textarea></label></article>`).join('');const requiredSide=['naturalness','sentence_delivery','stable_meaning','artifact'],requiredPair=['difference_strength','target_direction','confidence','unrelated_interference'];const trialComplete=t=>{const x=state(t.trial_id);return ['A','B'].every(s=>requiredSide.every(f=>String(x.sides[s][f]??'')!==''))&&requiredPair.every(f=>String(x.pair[f]??'')!=='')};const update=()=>{const complete=T.filter(trialComplete).length;document.getElementById('progressText').textContent=`${complete} of ${T.length} complete`;document.getElementById('fill').style.width=`${100*complete/T.length}%`;document.getElementById('download').disabled=complete!==T.length||!S.meta.reviewer?.trim()||!S.meta.language_background?.trim()||!S.meta.listening_setup;for(const t of T){const x=state(t.trial_id);document.querySelector(`[data-plays="${t.trial_id}"]`).textContent=String((x.play_starts.A||0)+(x.play_starts.B||0))}save()};for(const [id,key] of [['reviewer','reviewer'],['background','language_background'],['setup','listening_setup']]){const el=document.getElementById(id);el.value=S.meta[key]??'';el.oninput=()=>{S.meta[key]=el.value;update()}}document.querySelectorAll('[data-field]').forEach(el=>{el.oninput=()=>{const x=state(el.dataset.id),target=el.dataset.scope==='side'?x.sides[el.dataset.side]:x.pair;target[el.dataset.field]=el.value;update()}});document.querySelectorAll('audio').forEach(audio=>{const t=T.find(x=>x.trial_id===audio.dataset.id),box=audio.closest('.player'),head=box.querySelector('.head'),status=box.querySelector('.target-status');const draw=()=>{if(!Number.isFinite(audio.duration)||audio.duration<=0)return;head.style.left=`${Math.min(100,100*audio.currentTime/audio.duration)}%`;const active=t.target_intervals.some(v=>audio.currentTime>=v.start_s&&audio.currentTime<=v.end_s)&&!audio.paused;box.classList.toggle('active',active);status.textContent=active?'TARGET NOW':'Target position'};for(const event of ['loadedmetadata','timeupdate','pause','ended'])audio.addEventListener(event,draw);audio.addEventListener('error',()=>{status.textContent='Audio error';status.classList.add('error')});audio.addEventListener('play',()=>{document.querySelectorAll('audio').forEach(other=>{if(other!==audio&&!other.paused)other.pause()});state(audio.dataset.id).play_starts[audio.dataset.side]++;draw();update()})});update();document.getElementById('download').onclick=()=>{if(document.getElementById('download').disabled)return;const ratings=T.map(t=>{const x=state(t.trial_id);return{trial_id:t.trial_id,sides:x.sides,difference_strength:Number(x.pair.difference_strength),target_direction:x.pair.target_direction,confidence:Number(x.pair.confidence),unrelated_interference:x.pair.unrelated_interference,notes:String(x.pair.notes??''),play_starts:x.play_starts,replay_count:(x.play_starts.A||0)+(x.play_starts.B||0)}});const payload={schema_version:1,run_id:RUN,session_id:SESSION,protocol_sha256:PROTOCOL,public_manifest_sha256:PUBLIC,session_uuid:S.session_uuid,saved_at:new Date().toISOString(),reviewer:S.meta,ratings};const blob=new Blob([JSON.stringify(payload,null,2)+'\n'],{type:'application/json'}),a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download=F;a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)};</script></body></html>'''.replace(
        "__TRIALS__", public
    ).replace("__RUN_ID__", RUN_ID).replace("__SESSION_ID__", session_id).replace(
        "__SESSION_NUMBER__", str(session_number)
    ).replace(
        "__PROTOCOL__", protocol_sha256
    ).replace(
        "__PUBLIC__", public_manifest_sha256
    ).replace(
        "__RESPONSE__", response_filename
    )


def _hub(sessions: Iterable[dict[str, Any]]) -> str:
    cards = "".join(
        f'<li><a href="{row["session_id"]}/review.html">Session {index}</a> '
        f'({row["trial_count"]} comparisons)</li>'
        for index, row in enumerate(sessions, 1)
    )
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Blind vowel QC sessions</title><style>body{{font:18px/1.55 system-ui;max-width:760px;margin:auto;padding:28px;background:#f3f0e8;color:#18221d}}main{{background:#fff;border:1px solid #d6d1c6;border-radius:18px;padding:24px}}li{{margin:14px 0}}a{{color:#174b38;font-weight:800}}</style></head><body><main><h1>Blind vowel QC</h1><p>Four independent, resumable sessions cover every automatic unseen candidate. Complete them in order. Each page saves locally until you download its response JSON.</p><ol>{cards}</ol><p>Do not inspect the private manifest until every session is complete.</p></main></body></html>'''


def build_review_package(run_dir: Path = RUN_DIR) -> dict[str, Any]:
    if run_dir.exists():
        raise RuntimeError(f"refusing to overwrite blind-QC package: {run_dir}")
    protocol = load_protocol()
    protocol_sha256 = sha256_file(PROTOCOL_PATH)
    result_path = ROOT / protocol["parent_bindings"]["unseen_confirmation_path"]
    result = _load_json(result_path)
    if result["record_sha256"] != protocol["parent_bindings"][
        "unseen_confirmation_record_sha256"
    ]:
        raise RuntimeError("unseen automatic result record drifted")
    cells = _passing_cells(result)
    outcomes = _selected_outcomes(result, cells)
    source_root = result_path.parent
    session_voice = _session_voice_map(protocol)
    voice_session = {voice: session for session, voice in session_voice.items()}
    private_trials: list[dict[str, Any]] = []
    public_by_session: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for cell_id in sorted(cells):
        cell = cells[cell_id]
        session_id = voice_session[cell["voice_id"]]
        direction_prompt = protocol["direction_prompts"][cell["rule_id"]]
        identity_context = _identity_context(cell_id)
        for outcome in outcomes[cell_id]:
            neutral_path = source_root / outcome["audio"]["neutral"]["relative_path"]
            lens_path = source_root / outcome["audio"]["lens"]["relative_path"]
            duration_s, intervals = cue_intervals_from_difference(
                neutral_path, lens_path
            )
            expected = outcome["fixture_spec"]["expected_target_occurrence_count"]
            if len(intervals) != expected:
                raise RuntimeError(
                    f"blind-QC cue count drifted: {outcome['logical_slot_id']}"
                )
            raw_trials = [("candidate", neutral_path, lens_path)]
            if outcome["context"] == identity_context:
                raw_trials.append(("identity", neutral_path, neutral_path))
            for trial_kind, first_path, second_path in raw_trials:
                blind_id = _blind_id(
                    protocol["presentation"]["random_seed"],
                    cell_id,
                    outcome["context"],
                    trial_kind,
                )
                side_rng = random.Random(
                    f"{protocol['presentation']['random_seed']}|{blind_id}|sides"
                )
                second_side = "A" if side_rng.randrange(2) == 0 else "B"
                if trial_kind == "candidate":
                    condition_by_side = {
                        second_side: "lens",
                        ("B" if second_side == "A" else "A"): "neutral",
                    }
                    source_by_side = {
                        side: lens_path if condition == "lens" else neutral_path
                        for side, condition in condition_by_side.items()
                    }
                    expected_direction = second_side
                else:
                    condition_by_side = {"A": "identity", "B": "identity"}
                    source_by_side = {"A": first_path, "B": second_path}
                    expected_direction = "same"
                blind_audio: dict[str, str] = {}
                receipts: dict[str, Any] = {}
                for side in ("A", "B"):
                    target = run_dir / session_id / "audio" / f"{blind_id}-{side.lower()}.wav"
                    receipts[side] = _copy_blind_audio(source_by_side[side], target)
                    blind_audio[side] = f"audio/{target.name}"
                public_trial = {
                    "trial_id": blind_id,
                    "audio": blind_audio,
                    "duration_s": round(duration_s, 6),
                    "target_intervals": intervals,
                    "direction_prompt": direction_prompt,
                }
                public_by_session[session_id].append(public_trial)
                private_trials.append(
                    {
                        "trial_id": blind_id,
                        "session_id": session_id,
                        "trial_kind": trial_kind,
                        "cell_id": cell_id,
                        "profile_id": cell["profile_id"],
                        "voice_id": cell["voice_id"],
                        "rule_id": cell["rule_id"],
                        "source": cell["source"],
                        "target": cell["target"],
                        "context": outcome["context"],
                        "logical_slot_id": outcome["logical_slot_id"],
                        "expected_direction": expected_direction,
                        "condition_by_side": condition_by_side,
                        "fixture_spec": outcome["fixture_spec"],
                        "target_intervals": intervals,
                        "audio_receipts": receipts,
                        "product_enabled": False,
                    }
                )

    public_sessions: list[dict[str, Any]] = []
    private_session_records: list[dict[str, Any]] = []
    for number, session_id in enumerate(SESSION_LABELS, 1):
        trials = public_by_session[session_id]
        random.Random(
            f"{protocol['presentation']['random_seed']}|{session_id}|order"
        ).shuffle(trials)
        response_filename = f"{session_id}-response.json"
        public_manifest = _record(
            {
                "schema_version": 1,
                "run_id": RUN_ID,
                "session_id": session_id,
                "protocol_sha256": protocol_sha256,
                "trial_count": len(trials),
                "response_filename": response_filename,
                "trials": trials,
            }
        )
        manifest_path = run_dir / session_id / "public-manifest.json"
        atomic_write_json(manifest_path, public_manifest)
        public_manifest_sha256 = sha256_file(manifest_path)
        atomic_write_text(
            run_dir / session_id / "review.html",
            _html(
                session_id=session_id,
                session_number=number,
                public_trials=trials,
                protocol_sha256=protocol_sha256,
                public_manifest_sha256=public_manifest_sha256,
                response_filename=response_filename,
            ),
        )
        public_sessions.append(
            {
                "session_id": session_id,
                "trial_count": len(trials),
                "review_path": f"{session_id}/review.html",
                "public_manifest_path": f"{session_id}/public-manifest.json",
                "public_manifest_sha256": public_manifest_sha256,
                "response_filename": response_filename,
            }
        )
        private_session_records.append(
            {
                "session_id": session_id,
                "voice_id": session_voice[session_id],
                "trial_count": len(trials),
            }
        )

    kind_counts = Counter(row["trial_kind"] for row in private_trials)
    voice_counts = Counter(row["voice_id"] for row in private_trials)
    if (
        kind_counts != {"candidate": 54, "identity": 18}
        or dict(voice_counts) != protocol["scope"]["voice_session_counts"]
        or len(private_trials) != protocol["scope"]["total_trial_count"]
    ):
        raise RuntimeError("blind-QC trial denominator drifted")

    private_manifest = _record(
        {
            "schema_version": 1,
            "run_id": RUN_ID,
            "classification": "blind_review_ready_no_human_result_no_product_promotion",
            "protocol_sha256": protocol_sha256,
            "parent_result_record_sha256": result["record_sha256"],
            "automatic_candidate_cell_count": len(cells),
            "trial_kind_counts": dict(sorted(kind_counts.items())),
            "voice_trial_counts": dict(sorted(voice_counts.items())),
            "sessions": private_session_records,
            "trials": sorted(private_trials, key=lambda row: row["trial_id"]),
            "new_audio_renders_made": 0,
            "api_calls_made": 0,
            "human_review_complete": False,
            "production_enabled": False,
        }
    )
    atomic_write_json(run_dir / "private-manifest.json", private_manifest)
    public_hub = _record(
        {
            "schema_version": 1,
            "run_id": RUN_ID,
            "protocol_sha256": protocol_sha256,
            "session_count": len(public_sessions),
            "total_trial_count": sum(row["trial_count"] for row in public_sessions),
            "sessions": public_sessions,
        }
    )
    atomic_write_json(run_dir / "public-hub.json", public_hub)
    atomic_write_text(run_dir / "index.html", _hub(public_sessions))
    return private_manifest


def _validate_choice(value: Any, allowed: set[Any], field: str) -> None:
    if value not in allowed:
        raise ValueError(f"invalid {field}: {value!r}")


def adjudicate_session_response(
    response_path: Path,
    *,
    run_dir: Path = RUN_DIR,
) -> dict[str, Any]:
    protocol = load_protocol()
    private = _load_json(run_dir / "private-manifest.json")
    response = _load_json(response_path)
    if (
        response.get("schema_version") != 1
        or response.get("run_id") != RUN_ID
        or response.get("protocol_sha256") != sha256_file(PROTOCOL_PATH)
        or response.get("session_id") not in SESSION_LABELS
    ):
        raise ValueError("blind-QC response contract mismatch")
    session_id = response["session_id"]
    public_path = run_dir / session_id / "public-manifest.json"
    if response.get("public_manifest_sha256") != sha256_file(public_path):
        raise ValueError("blind-QC response public manifest mismatch")
    expected = {
        row["trial_id"]: row
        for row in private["trials"]
        if row["session_id"] == session_id
    }
    ratings = response.get("ratings")
    if not isinstance(ratings, list) or {row.get("trial_id") for row in ratings} != set(
        expected
    ):
        raise ValueError("blind-QC response trial denominator mismatch")
    reviewer = response.get("reviewer")
    if not isinstance(reviewer, dict) or not all(
        str(reviewer.get(field, "")).strip()
        for field in ("reviewer", "language_background", "listening_setup")
    ):
        raise ValueError("blind-QC response reviewer metadata is incomplete")

    candidate_gate = protocol["candidate_trial_gate"]
    trial_results: list[dict[str, Any]] = []
    for rating in ratings:
        hidden = expected[rating["trial_id"]]
        sides = rating.get("sides")
        if not isinstance(sides, dict) or set(sides) != {"A", "B"}:
            raise ValueError("blind-QC response side denominator mismatch")
        side_passes: dict[str, bool] = {}
        for side in ("A", "B"):
            row = sides[side]
            naturalness = int(row.get("naturalness", 0))
            _validate_choice(naturalness, {1, 2, 3, 4, 5}, "naturalness")
            _validate_choice(
                row.get("sentence_delivery"),
                {"sentence_like", "partly_sentence_like", "not_sentence_like"},
                "sentence_delivery",
            )
            _validate_choice(
                row.get("stable_meaning"),
                {"none", "isolated_possible_word", "coherent_phrase", "clear_source_sentence"},
                "stable_meaning",
            )
            _validate_choice(
                row.get("artifact"), {"none", "minor", "major", "uncertain"}, "artifact"
            )
            side_passes[side] = (
                naturalness >= candidate_gate["both_sides_naturalness_minimum"]
                and row["sentence_delivery"]
                in candidate_gate["both_sides_sentence_delivery_allowed"]
                and row["stable_meaning"]
                in candidate_gate["both_sides_stable_meaning_allowed"]
                and row["artifact"] in candidate_gate["both_sides_artifact_allowed"]
            )
        strength = int(rating.get("difference_strength", 0))
        confidence = int(rating.get("confidence", 0))
        _validate_choice(strength, set(range(1, 8)), "difference_strength")
        _validate_choice(confidence, set(range(1, 6)), "confidence")
        _validate_choice(
            rating.get("target_direction"), {"A", "B", "same", "uncertain"}, "target_direction"
        )
        _validate_choice(
            rating.get("unrelated_interference"),
            {"none", "manageable", "dominant", "uncertain"},
            "unrelated_interference",
        )
        if hidden["trial_kind"] == "candidate":
            passed = (
                all(side_passes.values())
                and strength >= candidate_gate["difference_strength_minimum"]
                and rating["target_direction"] == hidden["expected_direction"]
                and confidence >= candidate_gate["confidence_minimum"]
                and rating["unrelated_interference"]
                in candidate_gate["unrelated_interference_allowed"]
            )
            status = "candidate_trial_pass" if passed else "candidate_trial_fail"
        else:
            passed = (
                strength
                <= protocol["identity_control_handling"][
                    "expected_difference_strength_maximum"
                ]
                and rating["target_direction"]
                == protocol["identity_control_handling"]["expected_direction"]
            )
            status = (
                "identity_control_clean"
                if passed
                else "identity_control_investigation_required"
            )
        trial_results.append(
            {
                "trial_id": rating["trial_id"],
                "cell_id": hidden["cell_id"],
                "voice_id": hidden["voice_id"],
                "rule_id": hidden["rule_id"],
                "context": hidden["context"],
                "trial_kind": hidden["trial_kind"],
                "status": status,
                "gate_pass": passed,
                "side_quality_pass": side_passes,
            }
        )
    by_cell: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in trial_results:
        by_cell[row["cell_id"]].append(row)
    cell_results = []
    for cell_id, rows in sorted(by_cell.items()):
        candidates = [row for row in rows if row["trial_kind"] == "candidate"]
        identities = [row for row in rows if row["trial_kind"] == "identity"]
        if len(candidates) != 3 or len(identities) != 1:
            raise ValueError("blind-QC response cell trial structure drifted")
        candidate_pass = all(row["gate_pass"] for row in candidates)
        identity_clean = identities[0]["gate_pass"]
        eligible = candidate_pass and identity_clean
        cell_results.append(
            {
                "cell_id": cell_id,
                "voice_id": rows[0]["voice_id"],
                "rule_id": rows[0]["rule_id"],
                "candidate_context_gate_pass": candidate_pass,
                "identity_control_clean": identity_clean,
                "human_qc_pass": eligible,
                "product_enabled": False,
            }
        )
    return _record(
        {
            "schema_version": 1,
            "run_id": RUN_ID,
            "session_id": session_id,
            "classification": "human_qc_session_adjudicated_no_product_promotion",
            "protocol_sha256": sha256_file(PROTOCOL_PATH),
            "response_sha256": sha256_file(response_path),
            "reviewer": reviewer,
            "trial_results": sorted(trial_results, key=lambda row: row["trial_id"]),
            "cell_results": cell_results,
            "candidate_trial_pass_count": sum(
                row["gate_pass"] for row in trial_results if row["trial_kind"] == "candidate"
            ),
            "identity_control_clean_count": sum(
                row["gate_pass"] for row in trial_results if row["trial_kind"] == "identity"
            ),
            "human_qc_pass_cell_count": sum(row["human_qc_pass"] for row in cell_results),
            "production_enabled": False,
        }
    )
