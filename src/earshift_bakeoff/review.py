from __future__ import annotations

import json
import random
from pathlib import Path

from .config import Paths
from .util import atomic_write_text, read_csv


HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Renderer blind review</title>
<style>
:root { color-scheme: light; font-family: ui-sans-serif, system-ui, sans-serif; }
body { max-width: 900px; margin: 0 auto; padding: 24px; background:#f3f0e8; color:#1f2521; }
h1 { margin-bottom:4px; } .muted { color:#5c665f; }
.card { background:#fff; border:1px solid #d7d3c8; border-radius:14px; padding:18px; margin:16px 0; }
.meta { display:flex; justify-content:space-between; gap:12px; font-size:14px; }
.script { font-size:18px; line-height:1.55; margin:12px 0; }
audio { width:100%; margin:8px 0 14px; }
fieldset { border:0; padding:0; margin:10px 0; }
label { margin-right:14px; display:inline-block; }
select, textarea { font:inherit; } textarea { width:100%; min-height:58px; }
button { background:#183f32; color:white; border:0; border-radius:999px; padding:12px 18px; font-weight:700; }
.sticky { position:sticky; bottom:12px; display:flex; justify-content:flex-end; }
mark { background:#ffe18a; padding:0 2px; }
.guide { background:#e7eee8; border-left:4px solid #183f32; padding:14px 18px; margin:18px 0; }
.guide li { margin:7px 0; }
.anchors { font-size:14px; color:#5c665f; margin:8px 0 14px; }
</style>
</head>
<body>
<h1>Renderer blind review</h1>
<p class="muted">Review fluency first. Renderer identities are intentionally hidden. Your answers stay in this browser until you download the rating sheet.</p>
<section class="guide">
<strong>Quick start</strong>
<ol>
  <li>Play one clip once or twice. Ignore meaning: every word is intentionally invented.</li>
  <li>Choose <b>yes</b> only if the whole clip sounds like naturally connected speech in the labeled language. Use <b>uncertain</b> when you cannot judge that language reliably.</li>
  <li>Rate pace, prosody, coherence, and your confidence from 1–5. These scores are diagnostic; they do not replace the yes/no gate.</li>
  <li>Flag added commentary, letter-by-letter reading, broken audio, or replacement with obvious real words.</li>
</ol>
<div class="anchors"><b>Pace:</b> 1 very slow/halting · 3 plausible but uneven · 5 natural conversational speed<br>
<b>Prosody:</b> 1 robotic/wrong stress · 3 mixed · 5 natural rhythm and intonation<br>
<b>Coherence:</b> 1 disconnected list/syllables · 3 partly connected · 5 one cohesive utterance<br>
<b>Confidence:</b> 1 guessing · 3 moderately sure · 5 highly sure</div>
</section>
<div id="fluency"></div>
<h2>Contextual eSpeak comparison</h2>
<p class="muted">Only start this section after completing every fluency judgment. Compare the highlighted token in context with the isolated eSpeak reference. This audit is report-only.</p>
<div id="g2p"></div>
<div class="sticky"><button id="download">Download ratings.csv</button></div>
<script>
const RUN_ID = __RUN_ID__;
const ROWS = __ROWS__;
const storeKey = `earshift-review-${RUN_ID}`;
let state = JSON.parse(localStorage.getItem(storeKey) || '{}');
function save(id, key, value) { state[id] ||= {}; state[id][key] = value; localStorage.setItem(storeKey, JSON.stringify(state)); }
function checked(id,key,value) { return state[id]?.[key] === value ? 'checked' : ''; }
function selected(id,key,value) { return String(state[id]?.[key] ?? '') === String(value) ? 'selected' : ''; }
function esc(s) { return String(s ?? '').replace(/[&<>\"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;',"'":'&#39;'}[c])); }
function radios(id,key,values) { return values.map(v => `<label><input type="radio" name="${id}-${key}" value="${v}" ${checked(id,key,v)} onchange="save('${id}','${key}',this.value)">${v}</label>`).join(''); }
function scale(id,key) { return `<select onchange="save('${id}','${key}',this.value)"><option value="">—</option>${[1,2,3,4,5].map(v=>`<option ${selected(id,key,v)}>${v}</option>`).join('')}</select>`; }
document.getElementById('fluency').innerHTML = ROWS.map((r,i) => `<article class="card">
  <div class="meta"><strong>${i+1}. ${esc(r.language)} · ${esc(r.profile_id)}</strong><span>Blind ID ${esc(r.blind_id)}</span></div>
  <div class="script">${esc(r.script_text)}</div>
  <audio controls preload="none" src="audio/raw/${encodeURIComponent(r.audio_filename)}"></audio>
  <fieldset><legend>Fluent at native pace and prosody?</legend>${radios(r.blind_id,'human_fluent',['yes','no','uncertain'])}</fieldset>
  <fieldset>Pace ${scale(r.blind_id,'human_pace')} &nbsp; Prosody ${scale(r.blind_id,'human_prosody')} &nbsp; Coherence ${scale(r.blind_id,'human_coherence')}</fieldset>
  <fieldset>Confidence in this judgment ${scale(r.blind_id,'human_confidence')}</fieldset>
  <fieldset><legend>Glitch or spelling-out?</legend>${radios(r.blind_id,'human_glitch_or_spelling',['yes','no'])}</fieldset>
  <fieldset><legend>Obvious real-word autocorrection?</legend>${radios(r.blind_id,'human_real_word_autocorrection',['yes','no','uncertain'])}</fieldset>
  <textarea placeholder="Notes" oninput="save('${r.blind_id}','human_notes',this.value)">${esc(state[r.blind_id]?.human_notes || '')}</textarea>
</article>`).join('');
const probes = ROWS.filter(r => r.g2p_sampled === 'True' || r.g2p_sampled === 'true' || r.g2p_sampled === true);
document.getElementById('g2p').innerHTML = probes.map((r,i) => {
  const highlighted = esc(r.script_text).replace(esc(r.g2p_token), `<mark>${esc(r.g2p_token)}</mark>`);
  const ref = `g2p_reference/${encodeURIComponent(r.script_id + '__' + r.g2p_token + '.wav')}`;
  return `<article class="card"><div class="meta"><strong>${i+1}. ${esc(r.language)}</strong><span>Blind ID ${esc(r.blind_id)}</span></div>
  <div class="script">${highlighted}</div><p>Predicted IPA: <code>${esc(r.espeak_ipa)}</code></p>
  <p>TTS in context</p><audio controls preload="none" src="audio/raw/${encodeURIComponent(r.audio_filename)}"></audio>
  <p>eSpeak reference</p><audio controls preload="none" src="${ref}"></audio>
  <fieldset><legend>Agreement</legend>${radios(r.blind_id,'g2p_judgment',['match','near','mismatch','unclear'])}</fieldset></article>`;
}).join('');
document.getElementById('download').onclick = () => {
  const fields = ['blind_id','human_fluent','human_pace','human_prosody','human_coherence','human_confidence','human_glitch_or_spelling','human_real_word_autocorrection','human_notes','g2p_judgment'];
  const quote = v => `"${String(v ?? '').replaceAll('"','""')}"`;
  const lines = [fields.join(','), ...ROWS.map(r => fields.map(f => quote(f === 'blind_id' ? r.blind_id : state[r.blind_id]?.[f])).join(','))];
  const blob = new Blob([lines.join('\\n')+'\\n'], {type:'text/csv'}); const a=document.createElement('a'); a.href=URL.createObjectURL(blob); a.download='ratings.csv'; a.click(); URL.revokeObjectURL(a.href);
};
</script>
</body></html>
"""


def build_review(run_id: str) -> Path:
    run_dir = Paths().run_dir(run_id)
    rows = read_csv(run_dir / "results.csv")
    successful = [row for row in rows if row.get("render_status") == "ok"]
    random.Random(run_id).shuffle(successful)
    rendered = HTML_TEMPLATE.replace("__RUN_ID__", json.dumps(run_id)).replace(
        "__ROWS__", json.dumps(successful, ensure_ascii=False).replace("</", "<\\/")
    )
    destination = run_dir / "review.html"
    atomic_write_text(destination, rendered)
    return destination
