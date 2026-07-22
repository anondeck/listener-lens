from __future__ import annotations

import hashlib
import html
import json
import wave
from dataclasses import asdict
from pathlib import Path

import numpy as np

from earshift_bakeoff.bilingual_vowel_engine import (
    BILINGUAL_RULES_PATH,
    BilingualVowelRender,
    BilingualVowelRuntime,
)
from earshift_bakeoff.config import Paths, stable_json
from earshift_bakeoff.kokoro_synthesis import SAMPLE_RATE_HZ
from earshift_bakeoff.util import sha256_file


RUN_ID = "20260717-bidirectional-broad-vowel-smoke-v1"
RUN_DIR = Paths().artifacts / "bilingual-vowel-engine" / RUN_ID
FIXTURES = (
    {
        "fixture_id": "english-to-brazilian-portuguese-listener",
        "profile_id": "en-US-to-pt-BR-vowels-v1",
        "label": "English through a Brazilian-Portuguese-linked vowel profile",
        "text": "What a great day it is to catch some sun.",
    },
    {
        "fixture_id": "brazilian-portuguese-to-american-english-listener",
        "profile_id": "pt-BR-to-en-US-vowels-v1",
        "label": "Brazilian Portuguese through an American-English-linked vowel profile",
        "text": "Que dia bonito para pegar um pouco de sol.",
    },
)


def _write_wav(path: Path, pcm: np.ndarray) -> dict[str, object]:
    values = np.asarray(pcm, dtype="<i2").reshape(-1)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE_HZ)
        handle.writeframes(values.tobytes())
    return {
        "relative_path": str(path.relative_to(RUN_DIR)),
        "wav_sha256": sha256_file(path),
        "pcm_sha256": hashlib.sha256(values.tobytes()).hexdigest(),
        "sample_count": int(values.size),
        "duration_s": values.size / SAMPLE_RATE_HZ,
    }


def _review_html(records: list[dict[str, object]]) -> str:
    sections = []
    for index, record in enumerate(records, 1):
        coverage = record["coverage"]
        verification = record["verification"]
        audio = record["audio"]
        sections.append(
            f"""
            <section>
              <h2>{index}. {html.escape(str(record['label']))}</h2>
              <p><strong>Typed source:</strong> {html.escape(str(record['text']))}</p>
              <div class="pair">
                <article><h3>A — meaning-off source-vowel carrier</h3>
                  <audio controls src="{html.escape(str(audio['neutral']['relative_path']))}"></audio>
                </article>
                <article><h3>B — broad listener-vowel projection</h3>
                  <audio controls src="{html.escape(str(audio['lens']['relative_path']))}"></audio>
                </article>
              </div>
              <p>{coverage['mapped_vowel_occurrences']}/{coverage['source_vowel_occurrences']}
              vowel occurrences mapped; {coverage['changed_vowel_occurrences']} changed;
              {coverage['directly_observed_occurrences']} tied to direct perception results;
              {coverage['pending_acoustic_changed_occurrences']} changed occurrences still
              require rule-level acoustic validation.</p>
              <p><strong>Automatic engineering checks:</strong>
              {html.escape(str(verification['evidence_status']))}.</p>
              <details><summary>Debug carrier text</summary>
                <p>A: {html.escape(str(record['neutral_script']))}</p>
                <p>B: {html.escape(str(record['lens_script']))}</p>
              </details>
            </section>
            """
        )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Bidirectional broad-vowel engineering smoke</title>
<style>
body{{font:17px/1.5 system-ui;max-width:980px;margin:auto;padding:24px;background:#f4f1e8;color:#17241e}}
section{{background:white;border:1px solid #d5d1c5;border-radius:16px;padding:20px;margin:18px 0}}
.pair{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}article{{background:#f7f7f2;border-radius:12px;padding:14px}}audio{{width:100%}}
.warning{{border-left:5px solid #b65f2c;padding:10px 14px;background:#fff8ef}}details{{margin-top:14px}}
@media(max-width:700px){{.pair{{grid-template-columns:1fr}}}}
</style></head><body>
<h1>Bidirectional broad-vowel engineering smoke</h1>
<p class="warning"><strong>This is not a validated product claim.</strong> It tests whether the new broad engine actually produces natural, meaning-opaque sentence audio in both directions. Most changed vowel rules still need acoustic validation.</p>
{''.join(sections)}
</body></html>"""


def main() -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    for fixture in FIXTURES:
        runtime = BilingualVowelRuntime.load(fixture["profile_id"])
        result = runtime.render(fixture["text"])
        if not isinstance(result, BilingualVowelRender):
            raise RuntimeError(f"fixture has no changed vowels: {fixture['fixture_id']}")
        audio = {
            role: _write_wav(
                RUN_DIR / "audio" / f"{fixture['fixture_id']}__{role}.wav",
                values,
            )
            for role, values in (
                ("neutral", result.neutral_pcm),
                ("identity", result.identity_pcm),
                ("full_lens_diagnostic", result.full_lens_pcm),
                ("lens", result.lens_pcm),
            )
        }
        records.append(
            {
                **fixture,
                "plan_sha256": result.plan.plan_sha256,
                "voice_id": result.plan.voice_id,
                "neutral_script": result.plan.neutral_script,
                "lens_script": result.plan.lens_script,
                "coverage": asdict(result.plan.coverage),
                "gates": asdict(result.plan.gates),
                "verification": asdict(result.verification),
                "splice_windows": list(result.splice_windows),
                "target_occurrences": result.alignment["target_occurrences"],
                "audio": audio,
            }
        )
    payload: dict[str, object] = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "classification": "engineering_smoke_not_acoustic_or_listener_validation",
        "api_calls_made": 0,
        "paid_calls_made": 0,
        "rules_path": str(BILINGUAL_RULES_PATH.relative_to(Paths().root)),
        "rules_sha256": sha256_file(BILINGUAL_RULES_PATH),
        "fixtures": records,
    }
    payload["record_sha256"] = hashlib.sha256(
        stable_json(payload).encode("utf-8")
    ).hexdigest()
    (RUN_DIR / "records.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (RUN_DIR / "review.html").write_text(_review_html(records), encoding="utf-8")
    print(RUN_DIR / "review.html")


if __name__ == "__main__":
    main()
