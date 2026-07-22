from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from .config import Paths, stable_json
from .util import atomic_write_json, atomic_write_text, sha256_file


RUN_ID = "20260715-ptbr-ae-listener-pilot-v1"
SOURCE_RUN_ID = "20260715-curated-ptbr-ae-matched-pair"
PILOT_SEED = "ptbr-ae-listener-pilot-v1-20260715"
EXPECTED_PROTOCOL_SHA256 = (
    "d3ecfb16b6fbcf7df1ddcd64c737e08a7d91f4454b294fc3d323efaa74aeecda"
)

SOURCE_ARTIFACT_SHA256 = {
    "manifest.json": "7b0bb6ca00a604c98b9d785228c1484d96f091420b5e967e64db494187c65a9a",
    "results.csv": "6e40b558ab7e2e0ebfd87828c134cd475b4ec2604c1d317e6089f0fc8398ab43",
    "pair-selection.json": "860259ffba652f715e304aabeefbe8c457bf2dd6a5a367132b6e6e02a49e135f",
    "summary.json": "ca7ce44a3195f2c7273f4b52c0c34aedc2f6fc30a92cdbf569c539b3e1ee3f88",
}

SOURCE_AUDIO = {
    "neutral-4": {
        "filename": "008__neutral__take-4.wav",
        "sha256": "1052138ac9e9829e28089718bd856a3dd72c63f176241a716ebc743b22613f54",
    },
    "neutral-1": {
        "filename": "003__neutral__take-1.wav",
        "sha256": "c0266afe7cb519ab060b98150f1bb3cdad5f5f0aa87c47fa12b41251e0f3ce14",
    },
    "lens-1": {
        "filename": "004__lens__take-1.wav",
        "sha256": "43b4355211d4702523ab7932cd5bd47ef4cbb2f2c8667d2008f5f188ae2944f8",
    },
}


@dataclass(frozen=True)
class PilotTrial:
    presentation_order: int
    blind_id: str
    condition: Literal["identical", "neutral-variance", "neutral-lens"]
    audio_a_source: str
    audio_b_source: str
    audio_a_filename: str
    audio_b_filename: str


TRIALS = (
    PilotTrial(
        1,
        "d0bbf4c382",
        "neutral-variance",
        "neutral-4",
        "neutral-1",
        "443e3e40ac23.wav",
        "ab81bd3ea524.wav",
    ),
    PilotTrial(
        2,
        "55f78ccfde",
        "neutral-lens",
        "neutral-4",
        "lens-1",
        "89af9595f632.wav",
        "16fbccaf663a.wav",
    ),
    PilotTrial(
        3,
        "4df2d208e7",
        "identical",
        "neutral-4",
        "neutral-4",
        "f6075c922051.wav",
        "527df3dceca6.wav",
    ),
)


def _source_run_dir() -> Path:
    return Paths().artifacts / "matched-pairs" / SOURCE_RUN_ID


def _verify_sources() -> None:
    source_dir = _source_run_dir()
    for filename, expected in SOURCE_ARTIFACT_SHA256.items():
        if sha256_file(source_dir / filename) != expected:
            raise RuntimeError(f"Listener-pilot source changed: {filename}")
    for source, item in SOURCE_AUDIO.items():
        path = source_dir / "audio" / item["filename"]
        if sha256_file(path) != item["sha256"]:
            raise RuntimeError(f"Listener-pilot source audio changed: {source}")


def protocol_record() -> dict[str, Any]:
    _verify_sources()
    protocol: dict[str, Any] = {
        "schema_version": 1,
        "status": "preregistered_listener_pilot",
        "run_id": RUN_ID,
        "source_run_id": SOURCE_RUN_ID,
        "pilot_seed": PILOT_SEED,
        "source_artifact_sha256": SOURCE_ARTIFACT_SHA256,
        "source_audio": SOURCE_AUDIO,
        "trials": [asdict(trial) for trial in TRIALS],
        "presentation": {
            "same_blinded_randomized_ui": True,
            "condition_identity_in_review_html": False,
            "source_script_visible": False,
            "focus": "repeated middle-word vowel category",
            "required_responses": [
                "same/different/uncertain",
                "difference strength 1-5",
                "confidence 1-5",
                "delivery interference yes/no/uncertain",
            ],
        },
        "condition_roles": {
            "identical": "false-alarm baseline",
            "neutral-variance": "renderer take-variance control",
            "neutral-lens": "listener-lens signal",
        },
        "single_listener_interpretation": {
            "internally_clean_pattern": [
                "identical=same with confidence >=3",
                "neutral-variance=same with confidence >=3",
                "neutral-lens=different with strength >=3 and confidence >=3",
            ],
            "mixed_or_inconclusive": (
                "Any other response pattern; a single listener never establishes "
                "population validity."
            ),
            "stratify_by_language_background": True,
        },
        "api_calls": 0,
        "new_renders": 0,
    }
    protocol["protocol_sha256"] = hashlib.sha256(
        stable_json(protocol).encode("utf-8")
    ).hexdigest()
    return protocol


REVIEW_TEMPLATE = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Blinded vowel comparison pilot</title>
<style>
:root{font-family:ui-sans-serif,system-ui,-apple-system,sans-serif;color:#20251f;background:#f1eee5}*{box-sizing:border-box}body{margin:0}.wrap{max-width:860px;margin:auto;padding:28px 20px 70px}h1{font-size:clamp(2rem,6vw,4rem);line-height:.95;letter-spacing:-.05em;margin:.3em 0}.lede{max-width:680px;color:#596258;font-size:1.05rem}.panel,.trial{background:#fff;border:1px solid #d6d2c7;border-radius:18px;padding:20px;margin:18px 0;box-shadow:0 8px 24px #302d2410}.meta{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:14px}label,.label{font-weight:650;display:block;margin:8px 0}input[type=text],select,textarea{width:100%;font:inherit;padding:10px;border:1px solid #aaa99f;border-radius:9px;background:#fff}.clips{display:grid;grid-template-columns:1fr 1fr;gap:14px}.clip{background:#eef1eb;border-radius:13px;padding:14px}.clip b{display:block;margin-bottom:8px}audio{width:100%}.options{display:flex;flex-wrap:wrap;gap:12px}.options label{font-weight:500;margin:0;padding:9px 12px;background:#f3f1ea;border-radius:999px}.scales{display:grid;grid-template-columns:1fr 1fr;gap:14px}.anchors{font-size:.9rem;color:#5c655c}.error{color:#9b2c2c;font-weight:700;min-height:1.5em}.actions{display:flex;gap:12px;align-items:center;position:sticky;bottom:12px;background:#f1eee5e8;backdrop-filter:blur(8px);padding:12px 0}button{font:inherit;font-weight:750;border:0;border-radius:999px;padding:12px 18px;cursor:pointer}.primary{background:#174634;color:#fff}.secondary{background:#ddd9ce;color:#252a25}@media(max-width:600px){.clips,.scales{grid-template-columns:1fr}}
</style></head><body><main class="wrap">
<p><b>Listener pilot · blinded</b></p><h1>Compare the vowel, not the voice.</h1>
<p class="lede">Use headphones in a quiet room at one comfortable volume. In each trial, focus on the three repeated middle words. Decide whether their repeated vowel sounds belong to the same category or different categories. Pace and delivery may vary; report when that gets in the way.</p>
<section class="panel"><div class="meta"><div><label for="listener">Listener code</label><input id="listener" type="text" placeholder="e.g. max-01"></div><div><label for="background">Language background</label><select id="background"><option value="">Choose…</option><option>Native Brazilian Portuguese</option><option>Advanced Brazilian Portuguese</option><option>Intermediate Brazilian Portuguese</option><option>Basic Brazilian Portuguese</option><option>No Brazilian Portuguese</option></select></div><div><label for="headphones">Listening setup</label><select id="headphones"><option value="">Choose…</option><option>Headphones</option><option>Speakers</option></select></div></div></section>
<div id="trials"></div><p class="error" id="error"></p><div class="actions"><button class="primary" id="download">Download pilot-ratings.csv</button><button class="secondary" id="reset">Reset for another listener</button></div>
</main><script>
const PROTOCOL=__PROTOCOL__;const TRIALS=__TRIALS__;const KEY=`listener-pilot-${PROTOCOL}`;let state=JSON.parse(localStorage.getItem(KEY)||'{"meta":{},"trials":{}}');
const esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'}[c]));
function persist(){localStorage.setItem(KEY,JSON.stringify(state))}function setMeta(k,v){state.meta[k]=v;persist()}function setTrial(id,k,v){state.trials[id]??={plays_a:0,plays_b:0};state.trials[id][k]=v;persist()}function checked(id,k,v){return state.trials[id]?.[k]===v?'checked':''}function selected(id,k,v){return String(state.trials[id]?.[k]??'')===String(v)?'selected':''}
document.getElementById('listener').value=state.meta.listener??'';document.getElementById('background').value=state.meta.background??'';document.getElementById('headphones').value=state.meta.headphones??'';document.getElementById('listener').oninput=e=>setMeta('listener',e.target.value);document.getElementById('background').onchange=e=>setMeta('background',e.target.value);document.getElementById('headphones').onchange=e=>setMeta('headphones',e.target.value);
document.getElementById('trials').innerHTML=TRIALS.map((t,i)=>`<article class="trial"><h2>Trial ${i+1}</h2><div class="clips"><div class="clip"><b>Clip A</b><audio controls preload="metadata" data-id="${t.blind_id}" data-side="a" src="audio/${t.audio_a_filename}"></audio></div><div class="clip"><b>Clip B</b><audio controls preload="metadata" data-id="${t.blind_id}" data-side="b" src="audio/${t.audio_b_filename}"></audio></div></div><p class="label">The repeated middle-word vowels sound:</p><div class="options">${['same','different','uncertain'].map(v=>`<label><input type="radio" name="${t.blind_id}-judgment" value="${v}" ${checked(t.blind_id,'judgment',v)} onchange="setTrial('${t.blind_id}','judgment','${v}')"> ${v}</label>`).join('')}</div><div class="scales"><div><label>Difference strength</label><select onchange="setTrial('${t.blind_id}','strength',this.value)"><option value="">—</option>${[1,2,3,4,5].map(v=>`<option ${selected(t.blind_id,'strength',v)}>${v}</option>`).join('')}</select><p class="anchors">1 no audible difference · 3 clear/moderate · 5 very strong</p></div><div><label>Confidence</label><select onchange="setTrial('${t.blind_id}','confidence',this.value)"><option value="">—</option>${[1,2,3,4,5].map(v=>`<option ${selected(t.blind_id,'confidence',v)}>${v}</option>`).join('')}</select><p class="anchors">1 guessing · 3 moderately sure · 5 highly sure</p></div></div><p class="label">Did pace, rhythm, or delivery interfere?</p><div class="options">${['yes','no','uncertain'].map(v=>`<label><input type="radio" name="${t.blind_id}-interference" value="${v}" ${checked(t.blind_id,'delivery_interference',v)} onchange="setTrial('${t.blind_id}','delivery_interference','${v}')"> ${v}</label>`).join('')}</div><label>Optional note</label><textarea oninput="setTrial('${t.blind_id}','notes',this.value)">${esc(state.trials[t.blind_id]?.notes??'')}</textarea></article>`).join('');
document.querySelectorAll('audio').forEach(el=>el.addEventListener('play',()=>{const id=el.dataset.id,k=`plays_${el.dataset.side}`;setTrial(id,k,Number(state.trials[id]?.[k]??0)+1)}));
const q=v=>`"${String(v??'').replaceAll('"','""')}"`;document.getElementById('download').onclick=()=>{const err=document.getElementById('error');err.textContent='';if(!state.meta.listener?.trim()||!state.meta.background||!state.meta.headphones){err.textContent='Complete the listener details first.';return}for(const t of TRIALS){const r=state.trials[t.blind_id]??{};if(!r.judgment||!r.strength||!r.confidence||!r.delivery_interference){err.textContent='Complete every required judgment before downloading.';return}}const f=['protocol_sha256','listener_code','language_background','listening_setup','presentation_order','blind_id','judgment','strength','confidence','delivery_interference','plays_a','plays_b','notes'];const rows=TRIALS.map((t,i)=>{const r=state.trials[t.blind_id];return[PROTOCOL,state.meta.listener,state.meta.background,state.meta.headphones,i+1,t.blind_id,r.judgment,r.strength,r.confidence,r.delivery_interference,r.plays_a??0,r.plays_b??0,r.notes??'']});const csv=[f.join(','),...rows.map(row=>row.map(q).join(','))].join('\n')+'\n';const blob=new Blob([csv],{type:'text/csv'});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download=`pilot-ratings-${state.meta.listener.trim().replaceAll(/[^a-zA-Z0-9_-]/g,'_')}.csv`;a.click();URL.revokeObjectURL(a.href)};
document.getElementById('reset').onclick=()=>{if(confirm('Clear this listener’s answers and start fresh?')){localStorage.removeItem(KEY);location.reload()}};
</script></body></html>'''


def build_listener_pilot(run_id: str = RUN_ID) -> dict[str, Any]:
    if run_id != RUN_ID:
        raise RuntimeError(f"The frozen listener-pilot run id is {RUN_ID}")
    protocol = protocol_record()
    if protocol["protocol_sha256"] != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError("Listener-pilot protocol does not match its freeze")
    run_dir = Paths().artifacts / "listener-pilot" / run_id
    manifest_path = run_dir / "manifest.json"
    key_path = run_dir / "condition-key.json"
    if manifest_path.is_file():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != protocol:
            raise RuntimeError("Existing listener-pilot manifest does not match freeze")
    else:
        if run_dir.exists() and any(run_dir.iterdir()):
            raise RuntimeError("Listener-pilot directory exists without its manifest")
        atomic_write_json(manifest_path, protocol)

    source_dir = _source_run_dir() / "audio"
    copied: dict[str, str] = {}
    for trial in TRIALS:
        for source_key, filename in (
            (trial.audio_a_source, trial.audio_a_filename),
            (trial.audio_b_source, trial.audio_b_filename),
        ):
            source = source_dir / SOURCE_AUDIO[source_key]["filename"]
            destination = run_dir / "audio" / filename
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.is_file():
                shutil.copyfile(source, destination)
            expected = SOURCE_AUDIO[source_key]["sha256"]
            if sha256_file(destination) != expected:
                raise RuntimeError("A blinded listener-pilot audio copy changed")
            copied[filename] = expected

    key = {
        "protocol_sha256": protocol["protocol_sha256"],
        "trials": [asdict(trial) for trial in TRIALS],
        "blinded_audio_sha256": copied,
    }
    atomic_write_json(key_path, key)
    public_trials = [
        {
            "blind_id": trial.blind_id,
            "audio_a_filename": trial.audio_a_filename,
            "audio_b_filename": trial.audio_b_filename,
        }
        for trial in TRIALS
    ]
    html = REVIEW_TEMPLATE.replace(
        "__PROTOCOL__", json.dumps(protocol["protocol_sha256"])
    ).replace("__TRIALS__", json.dumps(public_trials).replace("</", "<\\/"))
    if any(condition in html for condition in ("neutral-variance", "neutral-lens", "identical")):
        raise RuntimeError("Condition identity leaked into blinded review HTML")
    review_path = run_dir / "review.html"
    atomic_write_text(review_path, html)
    return {
        "run_id": run_id,
        "protocol_sha256": protocol["protocol_sha256"],
        "trials": 3,
        "api_calls": 0,
        "new_renders": 0,
        "manifest": str(manifest_path),
        "condition_key": str(key_path),
        "review_html": str(review_path),
    }
