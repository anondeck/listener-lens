from __future__ import annotations

from collections import defaultdict
import hashlib
import html
import json
import os
from pathlib import Path
import subprocess
import wave
from typing import Any

import numpy as np

from earshift_bakeoff.bilingual_vowel_engine import _load_pinned_synthesis_voice
from earshift_bakeoff.config import ROOT, stable_json
from earshift_bakeoff.consonant_acoustics import (
    consonant_acoustic_metrics,
    decoder_column_interval,
    descriptive_distance,
    expanded_interval,
)
from earshift_bakeoff.consonant_calibration import (
    ALLOSAURUS_LICENSE,
    ALLOSAURUS_SCIPY_VERSION,
    ALLOSAURUS_SOURCE,
    ALLOSAURUS_VERSION,
    CONSONANT_CALIBRATION_VERSION,
    aggregate_rule_instrument,
    calibration_fixtures,
    labels_support_expected,
    overlapping_upr_labels,
    parse_allosaurus_timestamps,
    protocol_hash,
)
from earshift_bakeoff.controlled_listener_synthesis import (
    render_controlled_listener_triplet,
)
from earshift_bakeoff.kokoro_output_domain_splice import (
    boundary_artifact_report,
    output_domain_splice,
)
from earshift_bakeoff.kokoro_specs import VOICE_SPECS_BY_ID
from earshift_bakeoff.kokoro_synthesis import (
    CONFIG_FILE,
    KOKORO_VERSION,
    MODEL_FILE,
    MODEL_HASHES,
    MODEL_REPO,
    MODEL_REVISION,
    PairPlan,
    SAMPLE_RATE_HZ,
    _filtered_symbols,
    pcm16_bytes,
)
from earshift_bakeoff.kokoro_typed_diagnostic import localization_report
from earshift_bakeoff.util import atomic_write_json, sha256_file


RUN_ID = "20260717-consonant-calibration-v1"
RUN_DIR = ROOT / "artifacts" / "consonant-calibration" / RUN_ID
SPLICE_CONTEXT_MS = 25.0
ALLOSAURUS_BATCH = ROOT / "scripts" / "run_allosaurus_batch.py"


def _subsequence_start(values: tuple[str, ...], target: tuple[str, ...]) -> int:
    matches = [
        index
        for index in range(len(values) - len(target) + 1)
        if values[index : index + len(target)] == target
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one target sequence {target!r}; found {len(matches)}"
        )
    return matches[0]


def _columns(runtime: Any, neutral: str, lens: str, source: str) -> tuple[tuple[int, ...], tuple[int, ...]]:
    neutral_symbols = _filtered_symbols(runtime.model, neutral)
    lens_symbols = _filtered_symbols(runtime.model, lens)
    if len(neutral_symbols) != len(lens_symbols):
        raise RuntimeError("calibration fixture changed model-token count")
    changed = tuple(
        index + 1
        for index, (left, right) in enumerate(
            zip(neutral_symbols, lens_symbols, strict=True)
        )
        if left != right
    )
    source_symbols = tuple(
        symbol for symbol in source if runtime.model.vocab.get(symbol) is not None
    )
    start = _subsequence_start(neutral_symbols, source_symbols)
    target = tuple(range(start + 1, start + len(source_symbols) + 1))
    if not changed or any(column not in target for column in changed):
        raise RuntimeError("changed columns escaped the frozen source target")
    return changed, target


def _pcm(audio: np.ndarray) -> np.ndarray:
    return np.frombuffer(pcm16_bytes(audio), dtype="<i2").copy()


def _pcm_hash(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype="<i2").tobytes()).hexdigest()


def _write_wav(path: Path, values: np.ndarray) -> dict[str, Any]:
    if path.exists():
        raise RuntimeError(f"calibration output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = np.asarray(values, dtype="<i2").reshape(-1).tobytes()
    temporary = path.with_name(path.name + ".partial")
    try:
        with wave.open(str(temporary), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(SAMPLE_RATE_HZ)
            handle.writeframes(payload)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "relative_path": str(path.relative_to(RUN_DIR)),
        "pcm_sha256": _pcm_hash(values),
        "wav_sha256": sha256_file(path),
        "sample_count": int(np.asarray(values).size),
        "duration_s": np.asarray(values).size / SAMPLE_RATE_HZ,
    }


def _integrity(
    neutral: np.ndarray,
    identity: np.ndarray,
    target: np.ndarray,
    candidate: np.ndarray,
    weights: np.ndarray,
    window: dict[str, Any],
    target_interval: dict[str, Any],
) -> dict[str, Any]:
    arrays = (neutral, identity, target, candidate)
    equal_nonempty = bool(neutral.size and len({values.size for values in arrays}) == 1)
    finite = all(np.isfinite(values.astype(np.float64)).all() for values in arrays)
    unclipped = all(
        float(np.mean(np.abs(values.astype(np.int64)) >= 32767)) < 0.001
        for values in arrays
    )
    outside_exact = bool(
        np.array_equal(candidate[weights == 0.0], neutral[weights == 0.0])
    )
    interior_exact = bool(
        np.any(weights == 1.0)
        and np.array_equal(candidate[weights == 1.0], target[weights == 1.0])
    )
    start = int(target_interval["start_sample"])
    end = int(target_interval["end_sample_exclusive"])
    target_exact = bool(np.array_equal(candidate[start:end], target[start:end]))
    boundary = boundary_artifact_report(neutral, target, candidate, (window,))
    localization = localization_report(neutral, candidate, (window,))
    checks = {
        "neutral_identity_bit_exact": bool(np.array_equal(neutral, identity)),
        "equal_nonempty_samples": equal_nonempty,
        "finite": finite,
        "unclipped": unclipped,
        "outside_splice_exact_neutral": outside_exact,
        "full_weight_interior_exact_target": interior_exact,
        "target_interval_exact_target": target_exact,
        "boundary_metrics_pass": boundary.get("pass") is True,
        "localization_pass": localization.get("pass") is True,
    }
    return {
        **checks,
        "pass": all(checks.values()),
        "boundary": boundary,
        "localization": localization,
        "localization_note": (
            "The splice-window localization fraction is expected to be 1.0 by "
            "construction; it is an implementation-integrity check, not phonetic evidence."
        ),
    }


def _protocol() -> dict[str, Any]:
    voices = {
        voice_id: {
            "filename": VOICE_SPECS_BY_ID[voice_id].filename,
            "sha256": VOICE_SPECS_BY_ID[voice_id].sha256,
        }
        for voice_id in ("af_heart", "pf_dora")
    }
    payload = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "version": CONSONANT_CALIBRATION_VERSION,
        "classification": "exploratory_zero_api_consonant_renderer_calibration",
        "api_calls_authorized": 0,
        "fixture_count": 18,
        "conditions_per_fixture": [
            "source_anchor",
            "bit_identical_identity",
            "target_anchor",
            "output_domain_spliced_candidate",
        ],
        "fixtures": [row.protocol_record() for row in calibration_fixtures()],
        "intervention": {
            "renderer": "shared-state/common-RNG Kokoro synthesis",
            "candidate": "unchanged output-domain splice",
            "splice_context_ms_each_side": SPLICE_CONTEXT_MS,
            "taper_ms": 10.0,
            "measurement": (
                "exact decoder-column interval; descriptive PCM features; auxiliary "
                "Allosaurus timestamps within +/-30 ms"
            ),
        },
        "interpretation": {
            "upr": (
                "Auxiliary research instrument only. Two of three matching source and "
                "target anchors makes a direct rule eligible for blind human QC, never "
                "automatically promoted."
            ),
            "acoustics": (
                "Descriptors expose source/target separation and splice fidelity; they "
                "do not independently assign a consonant category."
            ),
            "derived_rules": "Remain research-only regardless of this run.",
        },
        "renderer_provenance": {
            "kokoro_version": KOKORO_VERSION,
            "model_repo": MODEL_REPO,
            "model_revision": MODEL_REVISION,
            "model_hashes": {
                CONFIG_FILE: MODEL_HASHES[CONFIG_FILE],
                MODEL_FILE: MODEL_HASHES[MODEL_FILE],
            },
            "voices": voices,
        },
        "upr_provenance": {
            "tool": "Allosaurus universal phone recognizer",
            "version": ALLOSAURUS_VERSION,
            "scipy_version": ALLOSAURUS_SCIPY_VERSION,
            "license": ALLOSAURUS_LICENSE,
            "source": ALLOSAURUS_SOURCE,
            "use": "research-only auxiliary instrument; not a shipped dependency",
        },
    }
    payload["protocol_sha256"] = protocol_hash(payload)
    return payload


def _write_review(records: list[dict[str, Any]]) -> None:
    order = sorted(
        records,
        key=lambda row: hashlib.sha256(
            (RUN_ID + row["fixture_id"]).encode("utf-8")
        ).hexdigest(),
    )
    sections = []
    for index, record in enumerate(order, 1):
        swapped = int(hashlib.sha256(record["fixture_id"].encode()).hexdigest(), 16) % 2
        roles = ("source_anchor", "candidate")
        if swapped:
            roles = roles[::-1]
        audio = record["audio"]
        sections.append(
            f"""<section><h2>{index}</h2><p>Judge the highlighted middle consonant only. "
            "Phone plans, rule, language, and condition are hidden.</p><div class=pair>
            <article><h3>A</h3><audio controls src=\"{html.escape(audio[roles[0]]["relative_path"])}\"></audio></article>
            <article><h3>B</h3><audio controls src=\"{html.escape(audio[roles[1]]["relative_path"])}\"></audio></article>
            </div><label>Difference strength 1–7 <input type=number min=1 max=7></label>
            <label>Unrelated delivery interference <select><option></option><option>none</option><option>manageable</option><option>major</option></select></label>
            <label>Notes <textarea></textarea></label></section>"""
        )
    page = f"""<!doctype html><html><head><meta charset=utf-8><meta name=viewport content=\"width=device-width,initial-scale=1\"><title>Blind consonant QC</title><style>body{{font:16px/1.5 system-ui;max-width:950px;margin:auto;padding:24px;background:#f4f1e8;color:#17241e}}section{{background:white;padding:20px;margin:16px 0;border-radius:14px}}.pair{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}audio,textarea{{width:100%}}label{{display:block;margin:12px 0}}@media(max-width:650px){{.pair{{grid-template-columns:1fr}}}}</style></head><body><h1>Blind consonant-renderer QC</h1><p><strong>Do not use this yet unless the automatic record marks its direct rules eligible.</strong> This is creator QC, not population evidence.</p>{''.join(sections)}</body></html>"""
    (RUN_DIR / "review.html").write_text(page, encoding="utf-8")


def main() -> None:
    if RUN_DIR.exists():
        raise RuntimeError(f"run directory already exists: {RUN_DIR}")
    RUN_DIR.mkdir(parents=True)
    protocol = _protocol()
    atomic_write_json(RUN_DIR / "protocol.json", protocol)

    runtimes: dict[str, Any] = {}
    records: list[dict[str, Any]] = []
    recognizer_inputs: list[dict[str, str]] = []
    for fixture in calibration_fixtures():
        runtime = runtimes.setdefault(
            fixture.voice_id, _load_pinned_synthesis_voice(fixture.voice_id)
        )
        lens = fixture.lens_phonemes
        changed_columns, target_columns = _columns(
            runtime, fixture.neutral_phonemes, lens, fixture.source
        )
        pair = PairPlan(
            source_phonemes=fixture.neutral_phonemes,
            neutral_phonemes=fixture.neutral_phonemes,
            lens_phonemes=lens,
            target_word_indexes=(2,),
        )
        rendered = render_controlled_listener_triplet(
            runtime,
            pair,
            consonant_columns=changed_columns,
        )
        neutral = _pcm(rendered.neutral)
        identity = _pcm(rendered.identity)
        target = _pcm(rendered.full_lens)
        target_interval = decoder_column_interval(
            rendered.predicted_durations,
            target_columns,
            sample_count=neutral.size,
            sample_rate_hz=SAMPLE_RATE_HZ,
        )
        splice_interval = expanded_interval(
            target_interval,
            context_ms=SPLICE_CONTEXT_MS,
            sample_count=neutral.size,
        )
        window = splice_interval.as_record()
        candidate, weights = output_domain_splice(neutral, target, (window,))
        interval_record = target_interval.as_record()
        audio: dict[str, Any] = {}
        for role, values in (
            ("source_anchor", neutral),
            ("identity", identity),
            ("target_anchor", target),
            ("candidate", candidate),
        ):
            path = RUN_DIR / "audio" / f"{fixture.fixture_id}__{role}.wav"
            audio[role] = _write_wav(path, values)
            if role in {"source_anchor", "target_anchor"}:
                recognizer_inputs.append(
                    {"id": f"{fixture.fixture_id}::{role}", "path": str(path)}
                )
        metrics = {
            role: consonant_acoustic_metrics(values, target_interval)
            for role, values in (
                ("source_anchor", neutral),
                ("target_anchor", target),
                ("candidate", candidate),
            )
        }
        records.append(
            {
                **fixture.protocol_record(),
                "changed_model_columns": changed_columns,
                "target_model_columns": target_columns,
                "predicted_durations": rendered.predicted_durations,
                "predicted_durations_sha256": hashlib.sha256(
                    stable_json(rendered.predicted_durations).encode()
                ).hexdigest(),
                "target_interval": interval_record,
                "splice_window": window,
                "audio": audio,
                "acoustic_descriptors": metrics,
                "source_target_descriptive_distance": descriptive_distance(
                    metrics["source_anchor"], metrics["target_anchor"]
                ),
                "candidate_target_descriptive_distance": descriptive_distance(
                    metrics["candidate"], metrics["target_anchor"]
                ),
                "engineering": _integrity(
                    neutral,
                    identity,
                    target,
                    candidate,
                    weights,
                    window,
                    interval_record,
                ),
            }
        )

    manifest = {
        "run_id": RUN_ID,
        "inputs": recognizer_inputs,
    }
    manifest_path = RUN_DIR / "analysis" / "allosaurus-inputs.json"
    manifest_path.parent.mkdir(parents=True)
    atomic_write_json(manifest_path, manifest)
    allosaurus_path = RUN_DIR / "analysis" / "allosaurus-output.json"
    command = [
        "uvx",
        "--python",
        "3.10",
        "--from",
        "allosaurus",
        "--with",
        f"scipy=={ALLOSAURUS_SCIPY_VERSION}",
        "python",
        str(ALLOSAURUS_BATCH),
        str(manifest_path),
        str(allosaurus_path),
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
    )
    if completed.returncode != 0 or not allosaurus_path.is_file():
        raise RuntimeError(
            "Allosaurus batch failed: "
            + (completed.stderr.strip() or completed.stdout.strip())
        )
    recognized = json.loads(allosaurus_path.read_text(encoding="utf-8"))
    if (
        recognized["allosaurus_version"] != ALLOSAURUS_VERSION
        or recognized["scipy_version"] != ALLOSAURUS_SCIPY_VERSION
    ):
        raise RuntimeError("Allosaurus research environment drifted")
    outputs = {row["id"]: row for row in recognized["rows"]}
    rule_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        interval = record["target_interval"]
        source_upr = parse_allosaurus_timestamps(
            outputs[f"{record['fixture_id']}::source_anchor"]["timestamp_output"]
        )
        target_upr = parse_allosaurus_timestamps(
            outputs[f"{record['fixture_id']}::target_anchor"]["timestamp_output"]
        )
        source_labels = overlapping_upr_labels(
            source_upr, start_s=interval["start_s"], end_s=interval["end_s"]
        )
        target_labels = overlapping_upr_labels(
            target_upr, start_s=interval["start_s"], end_s=interval["end_s"]
        )
        record["universal_phone_recognizer"] = {
            "instrument_role": "auxiliary_research_only",
            "source_anchor_all_timestamps": source_upr,
            "target_anchor_all_timestamps": target_upr,
            "source_anchor_overlapping_labels": source_labels,
            "target_anchor_overlapping_labels": target_labels,
            "source_anchor_match": labels_support_expected(
                source_labels, record["expected_source_labels"]
            ),
            "target_anchor_match": labels_support_expected(
                target_labels, record["expected_target_labels"]
            ),
        }
        rule_rows[record["rule_id"]].append(
            {
                "source_anchor_upr_match": record["universal_phone_recognizer"][
                    "source_anchor_match"
                ],
                "target_anchor_upr_match": record["universal_phone_recognizer"][
                    "target_anchor_match"
                ],
                "engineering_integrity_pass": record["engineering"]["pass"],
            }
        )
    fixture_by_rule = {row.rule_id: row for row in calibration_fixtures()}
    rule_results = {
        rule_id: aggregate_rule_instrument(
            rows, evidence_tier=fixture_by_rule[rule_id].evidence_tier
        )
        for rule_id, rows in sorted(rule_rows.items())
    }
    payload = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "protocol_sha256": protocol["protocol_sha256"],
        "api_calls_made": 0,
        "paid_calls_made": 0,
        "fixtures": records,
        "rules": rule_results,
        "classification": (
            "automatic_consonant_instruments_complete_human_qc_not_started"
        ),
        "product_status": "all_consonant_rules_remain_disabled_pending_adjudication",
    }
    payload["records_sha256"] = hashlib.sha256(
        stable_json(payload).encode("utf-8")
    ).hexdigest()
    atomic_write_json(RUN_DIR / "records.json", payload)
    _write_review(records)
    print(
        json.dumps(
            {
                "protocol": str(RUN_DIR / "protocol.json"),
                "records": str(RUN_DIR / "records.json"),
                "review": str(RUN_DIR / "review.html"),
                "rules": rule_results,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
