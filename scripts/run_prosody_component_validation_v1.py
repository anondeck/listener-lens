from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import wave
from typing import Any

import numpy as np

from earshift_bakeoff.bilingual_listener_engine import (
    BILINGUAL_LISTENER_CANDIDATE_VERSION,
    BILINGUAL_LISTENER_RULES_PATH,
    BilingualListenerRuntime,
)
from earshift_bakeoff.config import Paths, stable_json
from earshift_bakeoff.kokoro_synthesis import SAMPLE_RATE_HZ
from earshift_bakeoff.prosody_acoustics import (
    measure_question_component,
    measure_stress_component,
    run_praat_probe,
)
from earshift_bakeoff.prosody_component import (
    PROSODY_COMPONENT_VERSION,
    QUESTION_RULE_ID,
    STRESS_RULE_ID,
    render_prosody_component,
)
from earshift_bakeoff.util import atomic_write_json, sha256_file


RUN_ID = "20260717-prosody-component-validation-v1"
RUN_DIR = Paths().artifacts / "prosody-validation" / RUN_ID
PRAAT = Path("/opt/homebrew/bin/praat")
PRAAT_PROBE = Paths().root / "scripts" / "praat_prosody_probe.praat"
COMPONENT_SOURCE = Paths().root / "src" / "earshift_bakeoff" / "prosody_component.py"
ACOUSTIC_SOURCE = Paths().root / "src" / "earshift_bakeoff" / "prosody_acoustics.py"
RUNNER_SOURCE = Path(__file__).resolve()

# Frozen before the first saved validation render. The texts were selected by
# deterministic structural position and plan-gate checks without listening.
FIXTURES = (
    {
        "fixture_id": "stress-initial-word",
        "profile_id": "en-US-to-pt-BR-listener-v2",
        "rule_id": STRESS_RULE_ID,
        "text": "International decisions matter.",
        "target_structure": "stress-bearing word is phrase-initial",
    },
    {
        "fixture_id": "stress-medial-word",
        "profile_id": "en-US-to-pt-BR-listener-v2",
        "rule_id": STRESS_RULE_ID,
        "text": "Today independence matters.",
        "target_structure": "stress-bearing word is phrase-medial",
    },
    {
        "fixture_id": "stress-final-word",
        "profile_id": "en-US-to-pt-BR-listener-v2",
        "rule_id": STRESS_RULE_ID,
        "text": "People discuss organization.",
        "target_structure": "stress-bearing word is phrase-final",
    },
    {
        "fixture_id": "question-final-nasal",
        "profile_id": "pt-BR-to-en-US-listener-v2",
        "rule_id": QUESTION_RULE_ID,
        "text": "A menina comprou pão?",
        "target_structure": "four-word question ending in a nasal diphthong",
    },
    {
        "fixture_id": "question-final-stressed-nasal",
        "profile_id": "pt-BR-to-en-US-listener-v2",
        "rule_id": QUESTION_RULE_ID,
        "text": "Você trabalha amanhã?",
        "target_structure": "three-word question ending in a stressed nasal vowel",
    },
    {
        "fixture_id": "question-final-unstressed-vowel",
        "profile_id": "pt-BR-to-en-US-listener-v2",
        "rule_id": QUESTION_RULE_ID,
        "text": "O professor chegou cedo?",
        "target_structure": "four-word question ending in an unstressed vowel",
    },
)

STRESS_MINIMUM_FRAMES = 3
STRESS_MINIMUM_DURATION_DELTA_MS = 20.0
STRESS_MINIMUM_PROMOTED_RMS_RATIO = 1.10
STRESS_MAXIMUM_DEMOTED_RMS_RATIO = 0.90
QUESTION_MINIMUM_FRAMES = 5
QUESTION_MINIMUM_VOICED_FRACTION = 0.50
QUESTION_MINIMUM_NEUTRAL_RISE_RATIO = 1.05
QUESTION_MAXIMUM_NEUTRAL_END_TO_PEAK_RATIO = 0.90
QUESTION_MAXIMUM_LENS_END_TO_START_RATIO = 0.90
QUESTION_MAXIMUM_LENS_MIDDLE_TO_START_RATIO = 1.05


def _write_wav(path: Path, values: np.ndarray) -> dict[str, Any]:
    if path.exists():
        raise RuntimeError(f"validation WAV already exists: {path}")
    pcm = np.asarray(values, dtype="<i2").reshape(-1)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    try:
        with wave.open(str(temporary), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(SAMPLE_RATE_HZ)
            handle.writeframes(pcm.tobytes())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "relative_path": str(path.relative_to(RUN_DIR)),
        "wav_sha256": sha256_file(path),
        "pcm_sha256": hashlib.sha256(pcm.tobytes()).hexdigest(),
        "sample_count": int(pcm.size),
        "duration_s": pcm.size / SAMPLE_RATE_HZ,
    }


def _protocol() -> dict[str, Any]:
    if not PRAAT.is_file() or not PRAAT_PROBE.is_file():
        raise RuntimeError("standalone Praat or the frozen prosody probe is missing")
    version = subprocess.run(
        [str(PRAAT), "--version"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout.strip()
    protocol = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "classification": "prosody_only_multi_fixture_validation_preregistration",
        "api_calls_authorized": 0,
        "replacement_fixtures_authorized": 0,
        "component_version": PROSODY_COMPONENT_VERSION,
        "listener_candidate_version": BILINGUAL_LISTENER_CANDIDATE_VERSION,
        "fixtures": FIXTURES,
        "fixture_order_sha256": hashlib.sha256(
            stable_json(FIXTURES).encode("utf-8")
        ).hexdigest(),
        "isolation_contract": (
            "Neutral and lens use identical segment symbols. Lexical-stress trials "
            "change only paired stress markers, decoder duration allocation, and the "
            "frozen local intensity cue. Question trials change only the frozen final "
            "F0 contour. Output-domain splicing localizes returned differences."
        ),
        "instrument": {
            "name": "standalone Praat",
            "version": version,
            "executable_sha256": sha256_file(PRAAT),
            "probe_sha256": sha256_file(PRAAT_PROBE),
            "probe_sampling_step_ms": 5.0,
            "pitch_floor_hz": 75,
            "pitch_ceiling_hz": 500,
            "rms_window_ms": 10.0,
        },
        "measurement_gates": {
            "stress": {
                "minimum_frames_per_vowel": STRESS_MINIMUM_FRAMES,
                "minimum_decoder_duration_delta_ms": (STRESS_MINIMUM_DURATION_DELTA_MS),
                "minimum_promoted_rms_ratio": STRESS_MINIMUM_PROMOTED_RMS_RATIO,
                "maximum_demoted_rms_ratio": STRESS_MAXIMUM_DEMOTED_RMS_RATIO,
                "aggregate": "every promoted and demoted occurrence in all three fixtures passes",
            },
            "question": {
                "minimum_frames_per_contour_third": QUESTION_MINIMUM_FRAMES,
                "minimum_voiced_fraction": QUESTION_MINIMUM_VOICED_FRACTION,
                "minimum_neutral_middle_to_start_ratio": (
                    QUESTION_MINIMUM_NEUTRAL_RISE_RATIO
                ),
                "maximum_neutral_end_to_middle_ratio": (
                    QUESTION_MAXIMUM_NEUTRAL_END_TO_PEAK_RATIO
                ),
                "maximum_lens_end_to_start_ratio": (
                    QUESTION_MAXIMUM_LENS_END_TO_START_RATIO
                ),
                "maximum_lens_middle_to_start_ratio": (
                    QUESTION_MAXIMUM_LENS_MIDDLE_TO_START_RATIO
                ),
                "aggregate": "all four contour checks pass in all three fixtures",
            },
        },
        "automatic_pass": (
            "All six fixtures must pass plan gates, segment-identity isolation, "
            "bit-exact identity, PCM/integrity/splice gates, and their frozen acoustic "
            "gate. A pass remains pending blind creator QC and does not establish a "
            "population-level perception result."
        ),
        "human_gate_if_automatic_pass": {
            "identity_controls": "difference strength <=2 and no major artifact",
            "effect_trials": (
                "correct direction, strength >=5/7, confidence >=3/5, naturalness "
                ">=4/5 on both sides, sentence-like delivery on both sides, no stable "
                "recoverable meaning, no major artifact, and no dominant interference"
            ),
            "interpretation": (
                "Creator QC establishes product audibility/naturalness for these six "
                "fixtures only; cited listener evidence remains necessary for the "
                "listener-lens interpretation."
            ),
        },
        "stopping_rule": (
            "Preserve this run unchanged. Failure identifies the exact fixture and "
            "mechanism; it cannot be rescued by threshold changes, fixture replacement, "
            "or selective rerendering."
        ),
        "source_hashes": {
            "listener_rules": sha256_file(BILINGUAL_LISTENER_RULES_PATH),
            "component": sha256_file(COMPONENT_SOURCE),
            "acoustics": sha256_file(ACOUSTIC_SOURCE),
            "praat_probe": sha256_file(PRAAT_PROBE),
            "runner": sha256_file(RUNNER_SOURCE),
        },
    }
    protocol["protocol_sha256"] = hashlib.sha256(
        stable_json(protocol).encode("utf-8")
    ).hexdigest()
    return protocol


def _strip_stress(value: str) -> str:
    return value.replace("ˈ", "").replace("ˌ", "")


def _review(records: list[dict[str, Any]]) -> None:
    rng = random.Random(int(hashlib.sha256(RUN_ID.encode()).hexdigest()[:16], 16))
    trials: list[dict[str, Any]] = []
    private: list[dict[str, Any]] = []
    review_audio = RUN_DIR / "review" / "audio"
    review_audio.mkdir(parents=True, exist_ok=True)
    for record in records:
        for comparison, left_role, right_role in (
            ("identity_control", "neutral", "identity"),
            ("prosody_effect", "neutral", "lens"),
        ):
            roles = [left_role, right_role]
            rng.shuffle(roles)
            blind_id = hashlib.sha256(
                f"{RUN_ID}:{record['fixture_id']}:{comparison}".encode()
            ).hexdigest()[:12]
            public_audio: dict[str, str] = {}
            for side, role in zip(("A", "B"), roles, strict=True):
                source = RUN_DIR / record["audio"][role]["relative_path"]
                filename = (
                    hashlib.sha256(f"{blind_id}:{side}:{role}".encode()).hexdigest()[
                        :16
                    ]
                    + ".wav"
                )
                destination = review_audio / filename
                shutil.copy2(source, destination)
                public_audio[side] = f"audio/{filename}"
            question = record["rule_id"] == QUESTION_RULE_ID
            target_index = (
                len(record["safe_plan"]["word_roles"]) - 1
                if question
                else record["target_word_indexes"][0]
            )
            trials.append(
                {
                    "blind_id": blind_id,
                    "audio_a": public_audio["A"],
                    "audio_b": public_audio["B"],
                    "cue_count": len(record["safe_plan"]["word_roles"]),
                    "cue_index": target_index,
                    "direction_prompt": (
                        "Which clip has the stronger rise-then-fall ending?"
                        if question
                        else "Which clip puts stronger prominence earlier in the highlighted word?"
                    ),
                }
            )
            private.append(
                {
                    "blind_id": blind_id,
                    "fixture_id": record["fixture_id"],
                    "rule_id": record["rule_id"],
                    "comparison": comparison,
                    "side_roles": {side: role for side, role in zip(("A", "B"), roles)},
                    "expected_direction_side": (
                        next(
                            side
                            for side, role in zip(("A", "B"), roles)
                            if role == ("neutral" if question else "lens")
                        )
                        if comparison == "prosody_effect"
                        else "same"
                    ),
                }
            )
    rng.shuffle(trials)
    atomic_write_json(
        RUN_DIR / "review" / "private-manifest.json",
        {
            "schema_version": 1,
            "run_id": RUN_ID,
            "trials": private,
        },
    )
    public = json.dumps(trials, ensure_ascii=False).replace("</", "<\\/")
    page = r"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Blind prosody validation</title><style>body{font:17px/1.5 system-ui;max-width:900px;margin:auto;padding:24px;background:#f5f2e9;color:#17221c}.card{background:white;padding:20px;border:1px solid #d6d3c9;border-radius:16px;margin:18px 0}.pair{display:grid;grid-template-columns:1fr 1fr;gap:14px}.cue{display:flex;gap:7px;margin:10px 0}.cue i{width:15px;height:15px;border-radius:50%;background:#ccd2cc}.cue i.on{background:#d87b35;outline:2px solid #f1c59e}audio,select,textarea{width:100%;box-sizing:border-box}label{display:block;margin:10px 0}textarea{min-height:64px}button{padding:11px 18px;border:0;border-radius:999px;background:#154f3e;color:white;font-weight:700}@media(max-width:650px){.pair{grid-template-columns:1fr}}</style></head><body><h1>Blind prosody QC</h1><p>Compare the highlighted word or ending. Scripts, language, rule, and condition are hidden. Judge naturalness before deciding which side changed.</p><div id="trials"></div><button id="download">Download prosody-review.json</button><script>const T=__TRIALS__,K='__RUN_ID__',S=JSON.parse(localStorage.getItem(K)||'{}');const save=()=>localStorage.setItem(K,JSON.stringify(S));const opts=(xs,v)=>'<option value="">—</option>'+xs.map(x=>`<option ${x===v?'selected':''}>${x}</option>`).join('');document.getElementById('trials').innerHTML=T.map((t,i)=>{const s=S[t.blind_id]??{};const cue=Array.from({length:t.cue_count},(_,j)=>`<i class="${j===t.cue_index?'on':''}"></i>`).join('');return `<section class="card"><h2>Trial ${i+1}</h2><div class="cue">${cue}</div><div class="pair"><div><b>A</b><audio controls data-id="${t.blind_id}" src="${t.audio_a}"></audio></div><div><b>B</b><audio controls data-id="${t.blind_id}" src="${t.audio_b}"></audio></div></div><label>Naturalness A (1–5)<select data-id="${t.blind_id}" data-field="naturalness_a">${opts(['1','2','3','4','5'],s.naturalness_a)}</select></label><label>Naturalness B (1–5)<select data-id="${t.blind_id}" data-field="naturalness_b">${opts(['1','2','3','4','5'],s.naturalness_b)}</select></label><label>Sentence-like delivery A<select data-id="${t.blind_id}" data-field="sentence_a">${opts(['yes','partly','no'],s.sentence_a)}</select></label><label>Sentence-like delivery B<select data-id="${t.blind_id}" data-field="sentence_b">${opts(['yes','partly','no'],s.sentence_b)}</select></label><label>Difference strength (1–7)<select data-id="${t.blind_id}" data-field="strength">${opts(['1','2','3','4','5','6','7'],s.strength)}</select></label><label>${t.direction_prompt}<select data-id="${t.blind_id}" data-field="direction">${opts(['A','B','same','uncertain'],s.direction)}</select></label><label>Confidence (1–5)<select data-id="${t.blind_id}" data-field="confidence">${opts(['1','2','3','4','5'],s.confidence)}</select></label><label>Stable recoverable meaning<select data-id="${t.blind_id}" data-field="meaning">${opts(['none','isolated possible word','coherent phrase'],s.meaning)}</select></label><label>Artifact<select data-id="${t.blind_id}" data-field="artifact">${opts(['none','minor','major','uncertain'],s.artifact)}</select></label><label>Unrelated interference<select data-id="${t.blind_id}" data-field="interference">${opts(['none','manageable','dominant','uncertain'],s.interference)}</select></label><textarea data-id="${t.blind_id}" data-field="notes" placeholder="Optional notes">${s.notes??''}</textarea></section>`}).join('');document.querySelectorAll('[data-field]').forEach(el=>{el.oninput=()=>{S[el.dataset.id]??={};S[el.dataset.id][el.dataset.field]=el.value;save()}});document.querySelectorAll('audio[data-id]').forEach(el=>{el.onplay=()=>{S[el.dataset.id]??={};S[el.dataset.id].replay_count=(S[el.dataset.id].replay_count??0)+1;save()}});document.getElementById('download').onclick=()=>{const blob=new Blob([JSON.stringify({schema_version:1,run_id:'__RUN_ID__',saved_at:new Date().toISOString(),ratings:S},null,2)+'\n'],{type:'application/json'});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='prosody-review.json';a.click()};</script></body></html>""".replace(
        "__TRIALS__", public
    ).replace("__RUN_ID__", RUN_ID)
    (RUN_DIR / "review" / "review.html").write_text(page, encoding="utf-8")


def main() -> None:
    if RUN_DIR.exists():
        raise RuntimeError(f"prosody validation already exists: {RUN_DIR}")
    protocol = _protocol()
    RUN_DIR.mkdir(parents=True)
    atomic_write_json(RUN_DIR / "protocol.json", protocol)

    runtimes: dict[str, BilingualListenerRuntime] = {}
    records: list[dict[str, Any]] = []
    for fixture in FIXTURES:
        profile_id = fixture["profile_id"]
        if profile_id not in runtimes:
            runtimes[profile_id] = BilingualListenerRuntime.load(profile_id)
        runtime = runtimes[profile_id]
        render = render_prosody_component(runtime, fixture["text"])
        if render.rule_id != fixture["rule_id"]:
            raise RuntimeError("fixture activated the wrong prosody rule")
        segment_identity = _strip_stress(render.neutral_phonemes) == _strip_stress(
            render.lens_phonemes
        )
        if not segment_identity:
            raise RuntimeError("prosody-only fixture changed a segment category")
        audio = {
            role: _write_wav(
                RUN_DIR / "audio" / f"{fixture['fixture_id']}__{role}.wav", pcm
            )
            for role, pcm in (
                ("neutral", render.neutral_pcm),
                ("identity", render.identity_pcm),
                ("full_lens_diagnostic", render.full_lens_pcm),
                ("lens", render.lens_pcm),
            )
        }
        probes = {
            role: run_praat_probe(
                RUN_DIR / audio[role]["relative_path"],
                RUN_DIR / "analysis" / f"{fixture['fixture_id']}__{role}.tsv",
                praat_path=PRAAT,
                probe_path=PRAAT_PROBE,
            )
            for role in ("neutral", "lens")
        }
        if render.rule_id == STRESS_RULE_ID:
            measurement = measure_stress_component(
                render,
                probes["neutral"],
                probes["lens"],
                minimum_frames=STRESS_MINIMUM_FRAMES,
                minimum_duration_delta_ms=STRESS_MINIMUM_DURATION_DELTA_MS,
                minimum_promoted_rms_ratio=STRESS_MINIMUM_PROMOTED_RMS_RATIO,
                maximum_demoted_rms_ratio=STRESS_MAXIMUM_DEMOTED_RMS_RATIO,
            )
        else:
            measurement = measure_question_component(
                render,
                probes["neutral"],
                probes["lens"],
                minimum_frames=QUESTION_MINIMUM_FRAMES,
                minimum_voiced_fraction=QUESTION_MINIMUM_VOICED_FRACTION,
                minimum_neutral_rise_ratio=QUESTION_MINIMUM_NEUTRAL_RISE_RATIO,
                maximum_neutral_end_to_peak_ratio=(
                    QUESTION_MAXIMUM_NEUTRAL_END_TO_PEAK_RATIO
                ),
                maximum_lens_end_to_start_ratio=(
                    QUESTION_MAXIMUM_LENS_END_TO_START_RATIO
                ),
                maximum_lens_middle_to_start_ratio=(
                    QUESTION_MAXIMUM_LENS_MIDDLE_TO_START_RATIO
                ),
            )
        plan_gate_pass = bool(
            render.plan.gates.written_and_espeak_gate_pass
            and render.plan.gates.model_representable
            and render.plan.gates.punctuation_preserved
            and render.plan.gates.repeated_word_invariant_pass
        )
        automatic_pass = bool(
            plan_gate_pass
            and segment_identity
            and render.verification.integrity_pass
            and measurement["gate_pass"]
        )
        records.append(
            {
                **fixture,
                "plan_sha256": render.plan.plan_sha256,
                "safe_plan": {
                    "word_count": len(render.plan.words),
                    "word_roles": [word.carrier_role for word in render.plan.words],
                    "neutral_phone_sha256": hashlib.sha256(
                        render.neutral_phonemes.encode("utf-8")
                    ).hexdigest(),
                    "lens_phone_sha256": hashlib.sha256(
                        render.lens_phonemes.encode("utf-8")
                    ).hexdigest(),
                },
                "target_word_indexes": render.target_word_indexes,
                "plan_gate_pass": plan_gate_pass,
                "segment_identity_isolation_pass": segment_identity,
                "verification": asdict(render.verification),
                "splice_windows": render.splice_windows,
                "target_intervals": render.target_intervals,
                "stress_interventions": render.stress_interventions,
                "neutral_prosody": render.neutral_prosody,
                "lens_prosody": render.lens_prosody,
                "boundary": render.boundary,
                "localization": render.localization,
                "measurement": measurement,
                "audio": audio,
                "automatic_pass": automatic_pass,
            }
        )

    all_pass = all(record["automatic_pass"] for record in records)
    by_rule = {
        rule_id: {
            "passed": sum(
                record["automatic_pass"]
                for record in records
                if record["rule_id"] == rule_id
            ),
            "attempted": sum(record["rule_id"] == rule_id for record in records),
            "automatic_pass": all(
                record["automatic_pass"]
                for record in records
                if record["rule_id"] == rule_id
            ),
        }
        for rule_id in (STRESS_RULE_ID, QUESTION_RULE_ID)
    }
    result = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "protocol_sha256": protocol["protocol_sha256"],
        "api_calls_made": 0,
        "paid_calls_made": 0,
        "fixtures": records,
        "summary": by_rule,
        "classification": (
            "automatic_pass_pending_blind_creator_qc"
            if all_pass
            else "automatic_fail_preserved_no_human_review"
        ),
    }
    result["records_sha256"] = hashlib.sha256(
        stable_json(result).encode("utf-8")
    ).hexdigest()
    atomic_write_json(RUN_DIR / "records.json", result)
    if all_pass:
        _review(records)
    print(
        json.dumps(
            {
                "classification": result["classification"],
                "summary": by_rule,
                "records": str(RUN_DIR / "records.json"),
                "review": (
                    str(RUN_DIR / "review" / "review.html") if all_pass else None
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
