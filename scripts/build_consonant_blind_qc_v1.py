from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
import shutil
from typing import Any

from earshift_bakeoff.config import ROOT, stable_json
from earshift_bakeoff.util import atomic_write_json, sha256_file


RUN_ID = "20260717-consonant-blind-qc-v1"
RUN_DIR = ROOT / "artifacts" / "consonant-qc" / RUN_ID
THETA_DIR = (
    ROOT
    / "artifacts"
    / "consonant-calibration"
    / "20260717-theta-controlled-confirmation-v1"
)
PALATAL_DIR = (
    ROOT / "artifacts" / "consonant-calibration" / "20260717-consonant-calibration-v1"
)
THETA_IDS = (
    "enpt.theta_t__intervocalic__08",
    "enpt.theta_t__word_final__14",
    "enpt.theta_t__word_final__17",
    "enpt.theta_t__word_initial__05",
)
PALATAL_IDS = (
    "pten.palatal_lateral_yod__intervocalic_a",
    "pten.palatal_lateral_yod__intervocalic_u",
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _protocol() -> dict[str, Any]:
    protocol = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "classification": "zero_render_blind_consonant_creator_qc",
        "api_calls_authorized": 0,
        "renders_authorized": 0,
        "parents": {
            "theta": {
                "records_sha256": sha256_file(THETA_DIR / "records.json"),
                "selected_ids": THETA_IDS,
                "mode": "adjacent",
                "frozen_interpretation": (
                    "Four controlled /theta/ to /t/ fixtures passed engineering and "
                    "auxiliary target-label gates; no production promotion."
                ),
            },
            "palatal_lateral": {
                "records_sha256": sha256_file(PALATAL_DIR / "records.json"),
                "selected_ids": PALATAL_IDS,
                "frozen_interpretation": (
                    "Two source-and-target endpoint-retaining /palatal lateral/ to "
                    "/yod/ contexts are eligible for blind QC; no production promotion."
                ),
            },
        },
        "presentation": (
            "For every fixture, randomize one bit-identical control and one neutral/lens "
            "comparison. Hide condition, script, language, rule, token, and source "
            "filename. Show the same five-position cue with position three highlighted."
        ),
        "creator_qc_gate": {
            "identity_control": "difference strength <=2/7 and no major artifact",
            "effect": (
                "correct direction, strength >=5/7, confidence >=3/5, naturalness "
                ">=4/5 on both sides, sentence-like delivery on both sides, no major "
                "artifact, and no dominant unrelated interference"
            ),
            "aggregate": "every selected fixture passes its control and effect trial",
        },
        "interpretation": (
            "The creator review can screen audibility, direction, naturalness, and "
            "artifacts. It is not Brazilian-Portuguese population evidence for the "
            "English-interdental mapping and does not replace the cited listener study."
        ),
        "stopping_rule": (
            "Preserve each response unchanged. A failure cannot be rescued by removing "
            "a fixture, revealing scripts, or relaxing thresholds."
        ),
        "builder_sha256": sha256_file(Path(__file__).resolve()),
    }
    protocol["protocol_sha256"] = hashlib.sha256(
        stable_json(protocol).encode("utf-8")
    ).hexdigest()
    return protocol


def _selected_rows() -> list[dict[str, Any]]:
    theta = _load(THETA_DIR / "records.json")
    theta_rows = {
        row["candidate_id"]: row
        for row in theta["records"]
        if row["mode"] == "adjacent"
    }
    palatal = _load(PALATAL_DIR / "records.json")
    palatal_rows = {row["fixture_id"]: row for row in palatal["fixtures"]}
    selected: list[dict[str, Any]] = []
    for fixture_id in THETA_IDS:
        row = theta_rows[fixture_id]
        if (
            not row["engineering_pass"]
            or not row["universal_phone_recognizer"]["target_match"]
        ):
            raise RuntimeError("selected theta fixture lost its frozen automatic gate")
        selected.append(
            {
                "fixture_id": fixture_id,
                "rule_id": "enpt.theta_t",
                "neutral_path": THETA_DIR / row["audio"]["neutral"]["relative_path"],
                "lens_path": THETA_DIR
                / row["audio"]["spliced_candidate"]["relative_path"],
                "direction_prompt": "Which clip sounds more like the consonant at the start of “top”?",
                "expected_role": "lens",
            }
        )
    for fixture_id in PALATAL_IDS:
        row = palatal_rows[fixture_id]
        instrument = row["universal_phone_recognizer"]
        if not (
            row["engineering"]["pass"]
            and instrument["source_anchor_match"]
            and instrument["target_anchor_match"]
        ):
            raise RuntimeError(
                "selected palatal fixture lost its frozen automatic gate"
            )
        selected.append(
            {
                "fixture_id": fixture_id,
                "rule_id": "pten.palatal_lateral_yod",
                "neutral_path": PALATAL_DIR / row["audio"]["identity"]["relative_path"],
                "lens_path": PALATAL_DIR / row["audio"]["candidate"]["relative_path"],
                "direction_prompt": "Which clip sounds more like the consonant at the start of “yes”?",
                "expected_role": "lens",
            }
        )
    return selected


def _html(trials: list[dict[str, Any]]) -> str:
    public = json.dumps(trials, ensure_ascii=False).replace("</", "<\\/")
    return r"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Blind consonant QC</title><style>body{font:17px/1.5 system-ui;max-width:900px;margin:auto;padding:24px;background:#f5f2e9;color:#17221c}.card{background:white;padding:20px;border:1px solid #d6d3c9;border-radius:16px;margin:18px 0}.pair{display:grid;grid-template-columns:1fr 1fr;gap:14px}.cue{display:flex;gap:8px;margin:10px 0}.cue i{width:16px;height:16px;border-radius:50%;background:#ccd2cc}.cue i:nth-child(3){background:#d87b35;outline:2px solid #f1c59e}audio,select,textarea{width:100%;box-sizing:border-box}label{display:block;margin:10px 0}textarea{min-height:64px}button{padding:11px 18px;border:0;border-radius:999px;background:#154f3e;color:white;font-weight:700}@media(max-width:650px){.pair{grid-template-columns:1fr}}</style></head><body><h1>Blind consonant QC</h1><p>Judge each clip’s naturalness and sentence-like delivery before choosing a direction. Conditions, scripts, languages, rules, and filenames are hidden.</p><div id="trials"></div><button id="download">Download consonant-review.json</button><script>const T=__TRIALS__,K='__RUN_ID__',S=JSON.parse(localStorage.getItem(K)||'{}');const save=()=>localStorage.setItem(K,JSON.stringify(S));const opts=(xs,v)=>'<option value="">—</option>'+xs.map(x=>`<option ${x===v?'selected':''}>${x}</option>`).join('');document.getElementById('trials').innerHTML=T.map((t,i)=>{const s=S[t.blind_id]??{};return `<section class="card"><h2>Trial ${i+1}</h2><div class="cue"><i></i><i></i><i></i><i></i><i></i></div><div class="pair"><div><b>A</b><audio controls data-id="${t.blind_id}" src="${t.audio_a}"></audio></div><div><b>B</b><audio controls data-id="${t.blind_id}" src="${t.audio_b}"></audio></div></div><label>Naturalness A (1–5)<select data-id="${t.blind_id}" data-field="naturalness_a">${opts(['1','2','3','4','5'],s.naturalness_a)}</select></label><label>Naturalness B (1–5)<select data-id="${t.blind_id}" data-field="naturalness_b">${opts(['1','2','3','4','5'],s.naturalness_b)}</select></label><label>Sentence-like A<select data-id="${t.blind_id}" data-field="sentence_a">${opts(['yes','partly','no'],s.sentence_a)}</select></label><label>Sentence-like B<select data-id="${t.blind_id}" data-field="sentence_b">${opts(['yes','partly','no'],s.sentence_b)}</select></label><label>Difference strength (1–7)<select data-id="${t.blind_id}" data-field="strength">${opts(['1','2','3','4','5','6','7'],s.strength)}</select></label><label>${t.direction_prompt}<select data-id="${t.blind_id}" data-field="direction">${opts(['A','B','same','uncertain'],s.direction)}</select></label><label>Confidence (1–5)<select data-id="${t.blind_id}" data-field="confidence">${opts(['1','2','3','4','5'],s.confidence)}</select></label><label>Artifact<select data-id="${t.blind_id}" data-field="artifact">${opts(['none','minor','major','uncertain'],s.artifact)}</select></label><label>Unrelated interference<select data-id="${t.blind_id}" data-field="interference">${opts(['none','manageable','dominant','uncertain'],s.interference)}</select></label><textarea data-id="${t.blind_id}" data-field="notes" placeholder="Optional notes">${s.notes??''}</textarea></section>`}).join('');document.querySelectorAll('[data-field]').forEach(el=>{el.oninput=()=>{S[el.dataset.id]??={};S[el.dataset.id][el.dataset.field]=el.value;save()}});document.querySelectorAll('audio').forEach(el=>{el.onplay=()=>{S[el.dataset.id]??={};S[el.dataset.id].replay_count=(S[el.dataset.id].replay_count??0)+1;save()}});document.getElementById('download').onclick=()=>{const blob=new Blob([JSON.stringify({schema_version:1,run_id:'__RUN_ID__',saved_at:new Date().toISOString(),ratings:S},null,2)+'\n'],{type:'application/json'});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='consonant-review.json';a.click()};</script></body></html>""".replace(
        "__TRIALS__", public
    ).replace("__RUN_ID__", RUN_ID)


def main() -> None:
    if RUN_DIR.exists():
        raise RuntimeError(f"consonant review already exists: {RUN_DIR}")
    protocol = _protocol()
    RUN_DIR.mkdir(parents=True)
    atomic_write_json(RUN_DIR / "protocol.json", protocol)
    rng = random.Random(int(hashlib.sha256(RUN_ID.encode()).hexdigest()[:16], 16))
    public: list[dict[str, Any]] = []
    private: list[dict[str, Any]] = []
    audio_dir = RUN_DIR / "review" / "audio"
    audio_dir.mkdir(parents=True)
    for row in _selected_rows():
        for comparison, roles in (
            ("identity_control", ["neutral", "neutral"]),
            ("consonant_effect", ["neutral", "lens"]),
        ):
            rng.shuffle(roles)
            blind_id = hashlib.sha256(
                f"{RUN_ID}:{row['fixture_id']}:{comparison}".encode()
            ).hexdigest()[:12]
            filenames: dict[str, str] = {}
            for side, role in zip(("A", "B"), roles, strict=True):
                filename = (
                    hashlib.sha256(f"{blind_id}:{side}:{role}".encode()).hexdigest()[
                        :16
                    ]
                    + ".wav"
                )
                shutil.copy2(row[f"{role}_path"], audio_dir / filename)
                filenames[side] = f"audio/{filename}"
            public.append(
                {
                    "blind_id": blind_id,
                    "audio_a": filenames["A"],
                    "audio_b": filenames["B"],
                    "direction_prompt": row["direction_prompt"],
                }
            )
            private.append(
                {
                    "blind_id": blind_id,
                    "fixture_id": row["fixture_id"],
                    "rule_id": row["rule_id"],
                    "comparison": comparison,
                    "side_roles": {side: role for side, role in zip(("A", "B"), roles)},
                    "expected_direction_side": (
                        next(
                            side
                            for side, role in zip(("A", "B"), roles)
                            if role == row["expected_role"]
                        )
                        if comparison == "consonant_effect"
                        else "same"
                    ),
                }
            )
    rng.shuffle(public)
    atomic_write_json(
        RUN_DIR / "review" / "private-manifest.json",
        {"schema_version": 1, "run_id": RUN_ID, "trials": private},
    )
    (RUN_DIR / "review" / "review.html").write_text(_html(public), encoding="utf-8")
    result = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "protocol_sha256": protocol["protocol_sha256"],
        "api_calls_made": 0,
        "renders_made": 0,
        "fixture_count": 6,
        "trial_count": len(public),
        "identity_control_count": 6,
        "effect_trial_count": 6,
        "classification": "blind_creator_qc_ready_no_claim_promotion",
        "review_html_sha256": sha256_file(RUN_DIR / "review" / "review.html"),
        "private_manifest_sha256": sha256_file(
            RUN_DIR / "review" / "private-manifest.json"
        ),
    }
    result["record_sha256"] = hashlib.sha256(
        stable_json(result).encode("utf-8")
    ).hexdigest()
    atomic_write_json(RUN_DIR / "record.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
