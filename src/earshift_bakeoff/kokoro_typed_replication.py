from __future__ import annotations

import hashlib
import json
import os
import time
import wave
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .kokoro_synthesis import (
    SAMPLE_RATE_HZ,
    PairRender,
    _word_column_spans,
    pcm16_bytes,
    pcm_sha256,
    target_word_columns,
)
from .kokoro_typed_engine import (
    MAX_CLIPPED_FRACTION,
    KokoroTypedPlanner,
    TypedPlan,
    inspect_render,
)
from .kokoro_typed_replication_protocol import (
    FIXTURES,
    LOCALIZATION_MINIMUM,
    MEASUREMENT_SCRIPT,
    PRAAT,
    RUN_ID,
    TARGET_CUE_PADDING_S,
    blinded_trial_plan,
    prepare,
    protocol_record,
    run_dir,
)
from .sentence_pair_v2_analysis import CEILINGS, _measure
from .util import atomic_write_json, atomic_write_text, sha256_file


RECORDS_FILE = "render-records.json"
ANALYSIS_FILE = "analysis.json"
BLIND_KEY_FILE = "blind-key.json"
REVIEW_MANIFEST_FILE = "review-manifest.json"
REVIEW_FILE = "review.html"
RAW_RESPONSE_FILE = "response.json"
MANUAL_RESULT_FILE = "manual-result.json"


def _pcm_record(audio: np.ndarray) -> dict[str, Any]:
    values = np.asarray(audio).reshape(-1)
    finite = bool(values.size and np.isfinite(values).all())
    clipped_fraction = float(np.mean(np.abs(values) >= 1.0)) if finite else 1.0
    return {
        "sample_count": int(values.size),
        "finite": finite,
        "clipped_fraction": clipped_fraction,
        "clipping_pass": bool(clipped_fraction < MAX_CLIPPED_FRACTION),
        "pcm_sha256": pcm_sha256(values) if finite else None,
    }


def _write_wav(path: Path, audio: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    try:
        with wave.open(str(temporary), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(SAMPLE_RATE_HZ)
            handle.writeframes(pcm16_bytes(audio))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _model_symbol_count(model: Any, phonemes: str) -> int:
    symbols = [symbol for symbol in phonemes if model.vocab.get(symbol) is not None]
    if len(symbols) != len(phonemes):
        raise RuntimeError("validated phone plan lost a model symbol")
    return len(symbols)


def _sample_interval(
    columns: Sequence[int], durations: Sequence[int], samples_per_frame: int
) -> dict[str, Any]:
    selected = tuple(sorted(set(int(value) for value in columns)))
    if not selected or selected != tuple(range(selected[0], selected[-1] + 1)):
        raise RuntimeError("alignment interval columns must be nonempty and contiguous")
    start_sample = sum(durations[: selected[0]]) * samples_per_frame
    end_sample = sum(durations[: selected[-1] + 1]) * samples_per_frame
    if end_sample <= start_sample:
        raise RuntimeError("alignment interval has nonpositive length")
    return {
        "columns": list(selected),
        "start_sample": start_sample,
        "end_sample_exclusive": end_sample,
        "start_s": start_sample / SAMPLE_RATE_HZ,
        "end_s": end_sample / SAMPLE_RATE_HZ,
    }


def alignment_record(
    *, model: Any, plan: TypedPlan, durations: Sequence[int], sample_count: int
) -> dict[str, Any]:
    expected_duration_count = _model_symbol_count(model, plan.source_phonemes) + 2
    if len(durations) != expected_duration_count:
        raise RuntimeError(
            "predicted-duration count does not match the model-token plan"
        )
    total_frames = sum(int(value) for value in durations)
    if total_frames <= 0 or sample_count % total_frames:
        raise RuntimeError("decoded samples do not map to an integer alignment frame")
    samples_per_frame = sample_count // total_frames
    if samples_per_frame <= 0:
        raise RuntimeError("samples per alignment frame must be positive")

    word_spans = _word_column_spans(model, plan.neutral_phonemes)
    if len(word_spans) != len(plan.words):
        raise RuntimeError("carrier word spans do not match typed words")
    expected_replaced = target_word_columns(
        model, plan.neutral_phonemes, plan.target_word_indexes
    )

    occurrences: list[dict[str, Any]] = []
    target_words: list[dict[str, Any]] = []
    for word_index in plan.target_word_indexes:
        word = plan.words[word_index]
        span = word_spans[word_index]
        if len(span) != len(word.neutral_phone):
            raise RuntimeError(
                "carrier word columns drifted from the frozen phone plan"
            )
        word_interval = _sample_interval(span, durations, samples_per_frame)
        target_words.append(
            {
                "word_index": word_index,
                "interval": word_interval,
            }
        )
        for within_word_index, target_offset in enumerate(word.target_offsets):
            if word.neutral_phone[target_offset] != "æ":
                raise RuntimeError("target offset no longer points to neutral /ae/")
            if word.lens_phone[target_offset] != "ɛ":
                raise RuntimeError("target offset no longer points to lens /eh/")
            if target_offset < 1 or word.neutral_phone[target_offset - 1] not in {
                "ˈ",
                "ˌ",
            }:
                raise RuntimeError(
                    "replication target lacks the frozen preceding stress marker"
                )
            stress_column = span[target_offset - 1]
            target_column = span[target_offset]
            occurrences.append(
                {
                    "occurrence_index": len(occurrences),
                    "within_word_index": within_word_index,
                    "word_index": word_index,
                    "stress_column": stress_column,
                    "target_column": target_column,
                    "neutral_symbol": "æ",
                    "lens_symbol": "ɛ",
                    "measurement_interval": _sample_interval(
                        (stress_column, target_column),
                        durations,
                        samples_per_frame,
                    ),
                    "target_word_interval": word_interval,
                }
            )

    if len(occurrences) != plan.target_occurrence_count:
        raise RuntimeError("alignment lost a preregistered target occurrence")
    return {
        "duration_count": len(durations),
        "total_alignment_frames": total_frames,
        "samples_per_alignment_frame": samples_per_frame,
        "expected_replaced_columns": list(expected_replaced),
        "target_occurrences": occurrences,
        "target_words": target_words,
    }


def _runtime_record(
    *,
    fixture_id: str,
    plan: TypedPlan,
    rendered: Any,
    model: Any,
    paths: dict[str, Path],
) -> dict[str, Any]:
    arrays = {
        "neutral": rendered.neutral,
        "identity": rendered.identity,
        "lens": rendered.lens,
    }
    metrics = {role: _pcm_record(audio) for role, audio in arrays.items()}
    counts = {record["sample_count"] for record in metrics.values()}
    identity_equal = pcm16_bytes(rendered.neutral) == pcm16_bytes(rendered.identity)
    pair_integrity = inspect_render(
        PairRender(
            neutral=rendered.neutral,
            lens=rendered.lens,
            predicted_durations=rendered.predicted_durations,
            replaced_columns=rendered.replaced_columns,
        )
    )
    alignment = alignment_record(
        model=model,
        plan=plan,
        durations=rendered.predicted_durations,
        sample_count=len(rendered.neutral),
    )
    replaced_match = (
        list(rendered.replaced_columns) == alignment["expected_replaced_columns"]
    )
    runtime_pass = bool(
        len(counts) == 1
        and next(iter(counts), 0) > 0
        and identity_equal
        and replaced_match
        and pair_integrity.pass_all
        and all(
            record["finite"] and record["clipping_pass"] for record in metrics.values()
        )
    )
    return {
        "fixture_id": fixture_id,
        "plan_sha256": plan.plan_sha256,
        "plan_safe_metadata": plan.safe_metadata(),
        "predicted_durations": list(rendered.predicted_durations),
        "replaced_columns": list(rendered.replaced_columns),
        "alignment": alignment,
        "audio": {
            role: {
                **metrics[role],
                "relative_path": str(path.relative_to(run_dir())),
                "wav_sha256": sha256_file(path),
            }
            for role, path in paths.items()
        },
        "neutral_identity_bit_identical": identity_equal,
        "replaced_columns_match_complete_target_words": replaced_match,
        "pair_integrity": asdict(pair_integrity),
        "runtime_pass": runtime_pass,
    }


def render() -> dict[str, Any]:
    protocol = prepare()
    destination = run_dir() / RECORDS_FILE
    audio_dir = run_dir() / "audio"
    if destination.exists() or (audio_dir.exists() and any(audio_dir.iterdir())):
        raise RuntimeError(
            "the one-pass replication render already started; rerendering is forbidden"
        )

    from .kokoro_synthesis import KokoroSynthesisRuntime

    started = time.perf_counter()
    planner = KokoroTypedPlanner.load()
    synthesis = KokoroSynthesisRuntime.load(download=False)
    fixture_protocol = {row["fixture_id"]: row for row in protocol["fixtures"]}
    records: list[dict[str, Any]] = []
    request_order = 1
    for fixture in FIXTURES:
        plan = planner.plan(fixture.text)
        frozen = fixture_protocol[fixture.fixture_id]
        if plan.plan_sha256 != frozen["plan_sha256"]:
            raise RuntimeError(f"{fixture.fixture_id} plan drifted after freeze")
        pair_plan = plan.pair_plan()
        if pair_plan is None:
            raise RuntimeError(f"{fixture.fixture_id} unexpectedly has no comparison")
        rendered = synthesis.render_parity_triplet(pair_plan)
        paths: dict[str, Path] = {}
        for role, audio in (
            ("neutral", rendered.neutral),
            ("identity", rendered.identity),
            ("lens", rendered.lens),
        ):
            path = audio_dir / f"{request_order:02d}__{fixture.fixture_id}__{role}.wav"
            _write_wav(path, audio)
            paths[role] = path
            request_order += 1
        record = _runtime_record(
            fixture_id=fixture.fixture_id,
            plan=plan,
            rendered=rendered,
            model=synthesis.model,
            paths=paths,
        )
        records.append(record)
        print(
            f"typed replication {len(records)}/3 {fixture.fixture_id}: "
            f"{record['audio']['neutral']['sample_count']} samples",
            flush=True,
        )

    payload = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "protocol_sha256": protocol["protocol_sha256"],
        "status": "render_complete",
        "api_calls_made": 0,
        "paid_calls_made": 0,
        "triplets_rendered": len(records),
        "logical_wav_outputs": 3 * len(records),
        "wall_seconds": time.perf_counter() - started,
        "records": records,
        "all_runtime_gates_pass": all(record["runtime_pass"] for record in records),
        "one_pass_stopping_rule_satisfied": len(records) == 3,
    }
    atomic_write_json(destination, payload)
    return payload


def _read_pcm(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as handle:
        if (
            handle.getnchannels() != 1
            or handle.getsampwidth() != 2
            or handle.getframerate() != SAMPLE_RATE_HZ
        ):
            raise RuntimeError("replication WAV does not match frozen PCM format")
        samples = np.frombuffer(
            handle.readframes(handle.getnframes()), dtype="<i2"
        ).astype(np.float64)
    return samples, SAMPLE_RATE_HZ


def merge_sample_intervals(
    intervals: Sequence[tuple[int, int]], sample_count: int
) -> tuple[tuple[int, int], ...]:
    normalized = sorted(
        (max(0, start), min(sample_count, end))
        for start, end in intervals
        if end > 0 and start < sample_count
    )
    merged: list[tuple[int, int]] = []
    for start, end in normalized:
        if end <= start:
            continue
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return tuple(merged)


def localization_report(
    neutral: np.ndarray,
    lens: np.ndarray,
    target_word_intervals: Sequence[dict[str, Any]],
    *,
    sample_rate_hz: int = SAMPLE_RATE_HZ,
) -> dict[str, Any]:
    left = np.asarray(neutral, dtype=np.float64).reshape(-1)
    right = np.asarray(lens, dtype=np.float64).reshape(-1)
    if left.shape != right.shape or not left.size:
        return {"sample_count_equal": False, "pass": False}
    padding = round(TARGET_CUE_PADDING_S * sample_rate_hz)
    windows = merge_sample_intervals(
        [
            (
                int(row["start_sample"]) - padding,
                int(row["end_sample_exclusive"]) + padding,
            )
            for row in target_word_intervals
        ],
        len(left),
    )
    mask = np.zeros(len(left), dtype=bool)
    for start, end in windows:
        mask[start:end] = True
    delta = right - left
    energy = delta * delta
    total_energy = float(np.sum(energy))
    inside_energy = float(np.sum(energy[mask]))
    fraction = inside_energy / total_energy if total_energy else 1.0
    outside = delta[~mask]
    return {
        "sample_count_equal": True,
        "inside_windows": [
            {
                "start_sample": start,
                "end_sample_exclusive": end,
                "start_s": start / sample_rate_hz,
                "end_s": end / sample_rate_hz,
            }
            for start, end in windows
        ],
        "inside_difference_energy_fraction": fraction,
        "minimum_inside_difference_energy_fraction": LOCALIZATION_MINIMUM,
        "outside_rms_pcm": (
            float(np.sqrt(np.mean(outside**2))) if outside.size else 0.0
        ),
        "maximum_absolute_pcm_delta": float(np.max(np.abs(delta), initial=0.0)),
        "mean_absolute_pcm_delta": float(np.mean(np.abs(delta))),
        "pass": bool(fraction >= LOCALIZATION_MINIMUM),
    }


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator else -1.0


def classify_measurements(
    neutral: dict[str, Any], lens: dict[str, Any], geometry: dict[str, Any]
) -> dict[str, Any]:
    families: dict[str, Any] = {}
    for ceiling in CEILINGS:
        key = str(ceiling)
        anchor = geometry["families"][key]
        source = np.asarray(anchor["full_ae_bark"], dtype=float)
        target = np.asarray(anchor["full_eh_bark"], dtype=float)
        expected = np.asarray(anchor["full_vector_bark"], dtype=float)
        neutral_point = np.asarray(
            [neutral[key]["f1_bark"], neutral[key]["f2_bark"]], dtype=float
        )
        lens_point = np.asarray(
            [lens[key]["f1_bark"], lens[key]["f2_bark"]], dtype=float
        )
        vector = lens_point - neutral_point
        magnitude = float(np.linalg.norm(vector))
        cosine = _cosine(vector, expected)
        source_category = float(np.linalg.norm(neutral_point - source)) < float(
            np.linalg.norm(neutral_point - target)
        )
        target_category = float(np.linalg.norm(lens_point - target)) < float(
            np.linalg.norm(lens_point - source)
        )
        passed = bool(
            anchor["direction_sanity_pass"]
            and neutral[key]["plausibility_pass"]
            and lens[key]["plausibility_pass"]
            and source_category
            and target_category
            and cosine >= 0.5
            and magnitude >= anchor["product_magnitude_threshold_bark"]
        )
        families[key] = {
            "neutral_bark": neutral_point.tolist(),
            "lens_bark": lens_point.tolist(),
            "vector_bark": vector.tolist(),
            "magnitude_bark": magnitude,
            "threshold_bark": anchor["product_magnitude_threshold_bark"],
            "direction_cosine": cosine,
            "neutral_category_pass": source_category,
            "lens_category_pass": target_category,
            "neutral_plausibility_pass": neutral[key]["plausibility_pass"],
            "lens_plausibility_pass": lens[key]["plausibility_pass"],
            "pass": passed,
        }
    return {
        "families": families,
        "pass": all(record["pass"] for record in families.values()),
    }


def _measure_occurrence(
    neutral_path: Path,
    lens_path: Path,
    occurrence: dict[str, Any],
    geometry: dict[str, Any],
) -> dict[str, Any]:
    interval = occurrence["measurement_interval"]
    result: dict[str, Any] = {
        "occurrence_index": occurrence["occurrence_index"],
        "word_index": occurrence["word_index"],
        "measurement_interval": interval,
        "status": "measurement_failed",
        "exclusion_reasons": [],
    }
    try:
        neutral = {
            str(ceiling): _measure(neutral_path, interval, ceiling)
            for ceiling in CEILINGS
        }
        lens = {
            str(ceiling): _measure(lens_path, interval, ceiling) for ceiling in CEILINGS
        }
        classification = classify_measurements(neutral, lens, geometry)
        result.update(
            {
                "status": "measurable",
                "neutral_measurements": neutral,
                "lens_measurements": lens,
                "classification": classification,
                "pass": classification["pass"],
            }
        )
    except Exception as exc:
        result["exclusion_reasons"].append(f"{type(exc).__name__}: {str(exc)[:500]}")
        result["pass"] = False
    return result


def analyze() -> dict[str, Any]:
    protocol = protocol_record()
    records_path = run_dir() / RECORDS_FILE
    records_payload = json.loads(records_path.read_text(encoding="utf-8"))
    if records_payload["protocol_sha256"] != protocol["protocol_sha256"]:
        raise RuntimeError("render records do not match the frozen protocol")
    acoustic = protocol["replication_only_acoustic_gate"]
    if sha256_file(PRAAT) != acoustic["praat_sha256"]:
        raise RuntimeError("standalone Praat executable changed after freeze")
    if sha256_file(MEASUREMENT_SCRIPT) != acoustic["measurement_script_sha256"]:
        raise RuntimeError("Praat measurement script changed after freeze")

    geometry = acoustic["anchor_geometry"]
    analyzed: list[dict[str, Any]] = []
    measurement_failed = False
    for record in records_payload["records"]:
        neutral_path = run_dir() / record["audio"]["neutral"]["relative_path"]
        lens_path = run_dir() / record["audio"]["lens"]["relative_path"]
        neutral_pcm, rate = _read_pcm(neutral_path)
        lens_pcm, lens_rate = _read_pcm(lens_path)
        if rate != lens_rate:
            raise RuntimeError("replication pair sample rates differ")
        occurrences = [
            _measure_occurrence(neutral_path, lens_path, occurrence, geometry)
            for occurrence in record["alignment"]["target_occurrences"]
        ]
        measurement_failed |= any(
            occurrence["status"] != "measurable" for occurrence in occurrences
        )
        target_intervals = [
            row["interval"] for row in record["alignment"]["target_words"]
        ]
        localization = localization_report(
            neutral_pcm,
            lens_pcm,
            target_intervals,
            sample_rate_hz=rate,
        )
        acoustic_pass = bool(
            occurrences and all(occurrence["pass"] for occurrence in occurrences)
        )
        fixture_pass = bool(
            record["runtime_pass"] and acoustic_pass and localization["pass"]
        )
        analyzed.append(
            {
                "fixture_id": record["fixture_id"],
                "runtime_pass": record["runtime_pass"],
                "target_occurrences": occurrences,
                "acoustic_pass": acoustic_pass,
                "localization": localization,
                "automatic_replication_pass": fixture_pass,
            }
        )
        print(
            f"typed acoustic {len(analyzed)}/3 {record['fixture_id']}: "
            f"{'pass' if fixture_pass else 'fail'}",
            flush=True,
        )

    automatic_pass = bool(
        len(analyzed) == len(FIXTURES)
        and all(record["automatic_replication_pass"] for record in analyzed)
    )
    if automatic_pass:
        classification = "automatic_replication_pass_ready_for_blind_creator_qc"
    elif measurement_failed:
        classification = "inconclusive_measurement_failure"
    else:
        classification = "automatic_replication_failed_no_promotion"
    result = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "protocol_sha256": protocol["protocol_sha256"],
        "render_records_sha256": sha256_file(records_path),
        "status": "analysis_complete",
        "classification": classification,
        "fixture_count": len(analyzed),
        "target_occurrence_count": sum(
            len(record["target_occurrences"]) for record in analyzed
        ),
        "automatic_replication_pass": automatic_pass,
        "api_calls_made": 0,
        "fixtures": analyzed,
    }
    atomic_write_json(run_dir() / ANALYSIS_FILE, result)
    if automatic_pass:
        build_review()
    else:
        atomic_write_text(
            run_dir() / REVIEW_FILE,
            "<!doctype html><meta charset='utf-8'><title>Replication did not advance</title>"
            "<p>The frozen automatic replication gate did not pass, so the preregistered "
            "listener review was not opened.</p>",
        )
    return result


def _records_by_fixture() -> dict[str, dict[str, Any]]:
    payload = json.loads((run_dir() / RECORDS_FILE).read_text(encoding="utf-8"))
    return {record["fixture_id"]: record for record in payload["records"]}


def _ensure_review_audio(
    layout: Sequence[dict[str, Any]], records: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    public: list[dict[str, Any]] = []
    for trial in layout:
        record = records[trial["fixture_id"]]
        sides: list[dict[str, Any]] = []
        for side in ("A", "B"):
            role = trial["side_roles"][side]
            source = run_dir() / record["audio"][role]["relative_path"]
            destination = (
                run_dir() / "review-audio" / f"{trial['trial_id']}-{side.lower()}.wav"
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            relative = os.path.relpath(source, destination.parent)
            if os.path.lexists(destination):
                if not destination.is_symlink() or os.readlink(destination) != relative:
                    raise RuntimeError("opaque review audio path differs from freeze")
            else:
                destination.symlink_to(relative)
            if sha256_file(destination) != record["audio"][role]["wav_sha256"]:
                raise RuntimeError("opaque review audio hash mismatch")
            sides.append(
                {
                    "side": side,
                    "audio": str(destination.relative_to(run_dir())),
                    "audio_sha256": record["audio"][role]["wav_sha256"],
                }
            )
        public.append(
            {
                "trial_id": trial["trial_id"],
                "duration_s": (
                    record["audio"]["neutral"]["sample_count"] / SAMPLE_RATE_HZ
                ),
                "target_intervals": [
                    {
                        "start_s": row["interval"]["start_s"],
                        "end_s": row["interval"]["end_s"],
                    }
                    for row in record["alignment"]["target_words"]
                ],
                "sides": sides,
            }
        )
    return public


def _review_html(public: Sequence[dict[str, Any]], protocol_sha256: str) -> str:
    template = r"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Typed-engine blind replication QC</title><style>:root{color-scheme:light}body{font:17px/1.5 system-ui;max-width:920px;margin:auto;padding:24px;background:#f5f2e9;color:#17221c}.intro,.trial,.side{background:white;border:1px solid #d6d3c9;border-radius:16px;padding:20px;margin:16px 0}.pair{display:grid;grid-template-columns:1fr 1fr;gap:14px}.side{margin:0}.timeline{height:16px;background:#d8ddd8;border-radius:99px;position:relative;margin:10px 0 16px;overflow:hidden}.cue{position:absolute;height:100%;background:#d87b35;box-shadow:0 0 0 2px #f1c59e inset}.playhead{position:absolute;top:0;bottom:0;width:2px;background:#154f3e}.target-now .timeline{outline:3px solid #d87b35}.target-status{font-weight:700;color:#805025;min-height:1.5em}audio,select,textarea{width:100%;box-sizing:border-box}label{display:block;margin:10px 0}textarea{min-height:78px}button{padding:12px 20px;border:0;border-radius:999px;background:#154f3e;color:white;font-weight:700}button:disabled{opacity:.45}.muted{color:#57645e}.scale{font-size:.9em;color:#57645e}@media(max-width:680px){.pair{grid-template-columns:1fr}}</style></head><body><section class="intro"><h1>Blind typed-engine replication QC</h1><p>There are six randomized comparisons. Three are exact identity checks and three contain the candidate vowel change; their order is hidden. Judge what you hear, not what you expect.</p><p>The orange band marks the target word position. Every branch and both sides receive the same cue. Some comparisons have two target positions.</p><p class="muted">“Stable English meaning” asks whether the gibberish itself consistently communicates an English word or phrase—not whether you can guess or remember a source sentence.</p><p><strong>Complete every field before downloading.</strong> Notes are optional.</p></section><div id="trials"></div><button id="download" disabled>Download review.json</button><script>const R=__PUBLIC__;const PROTOCOL='__PROTOCOL__';const RUN='__RUN__';const K='typed-engine-replication-v1-review';const S=JSON.parse(localStorage.getItem(K)||'{}');S.session_id??=crypto.randomUUID();S.trials??={};const save=()=>localStorage.setItem(K,JSON.stringify(S));const trialState=id=>(S.trials[id]??={sides:{A:{},B:{}},pair:{},play_starts:{A:0,B:0}});const option=(value,label)=>`<option value="${value}">${label}</option>`;const selector=(id,scope,side,field,options,label)=>`<label>${label}<select data-id="${id}" data-scope="${scope}" data-side="${side||''}" data-field="${field}"><option value="">—</option>${options}</select></label>`;const sideCard=(trial,side)=>`<section class="side"><h3>Clip ${side.side}</h3><div class="player"><audio controls preload="metadata" src="${side.audio}" data-id="${trial.trial_id}" data-side="${side.side}"></audio><div class="timeline">${trial.target_intervals.map(x=>`<i class="cue" style="left:${100*x.start_s/trial.duration_s}%;width:${100*(x.end_s-x.start_s)/trial.duration_s}%"></i>`).join('')}<i class="playhead"></i></div><div class="target-status">Target position</div></div>${selector(trial.trial_id,'side',side.side,'naturalness',[1,2,3,4,5].map(n=>option(n,n)).join(''),'Naturalness (1 unusable · 5 fully natural)')}${selector(trial.trial_id,'side',side.side,'delivery',option('sentence-like','Sentence-like')+option('slightly-list-like','Slightly list-like')+option('dominantly-list-like','Dominantly list-like')+option('other','Other'),'Delivery')}${selector(trial.trial_id,'side',side.side,'meaning',option('none','None')+option('isolated-possible-word','Isolated possible word')+option('coherent-phrase','Coherent phrase')+option('clear-source-sentence','Clear source sentence'),'Stable English meaning in the gibberish')}${selector(trial.trial_id,'side',side.side,'artifact',option('none','None')+option('minor','Minor')+option('major','Major')+option('uncertain','Uncertain'),'Artifact or defect')}</section>`;document.getElementById('trials').innerHTML=R.map((trial,index)=>`<section class="trial"><h2>Comparison ${index+1} of ${R.length}</h2><div class="pair">${trial.sides.map(side=>sideCard(trial,side)).join('')}</div><p class="muted">Recorded play starts: <strong data-replays="${trial.trial_id}">0</strong></p>${selector(trial.trial_id,'pair','','difference_strength',[1,2,3,4,5,6,7].map(n=>option(n,n)).join(''),'Difference strength (1 none · 4 moderate · 7 very strong)')}${selector(trial.trial_id,'pair','','category_judgment',option('A','A')+option('B','B')+option('same','Same / no category difference')+option('uncertain','Uncertain')+option('neither','Neither'),'Which side, if either, sounds closer to the vowel in “bet”?')}${selector(trial.trial_id,'pair','','confidence',[1,2,3,4,5].map(n=>option(n,n)).join(''),'Confidence (1 guessing · 5 highly confident)')}${selector(trial.trial_id,'pair','','interference',option('none','None')+option('manageable','Manageable')+option('dominant','Dominant')+option('uncertain','Uncertain'),'Unrelated delivery interference')}<label>Notes (optional)<textarea data-id="${trial.trial_id}" data-scope="pair" data-side="" data-field="notes"></textarea></label></section>`).join('');const requiredSide=['naturalness','delivery','meaning','artifact'],requiredPair=['difference_strength','category_judgment','confidence','interference'];const complete=()=>R.every(t=>{const x=trialState(t.trial_id);return ['A','B'].every(s=>requiredSide.every(f=>String(x.sides[s][f]??'')!==''))&&requiredPair.every(f=>String(x.pair[f]??'')!=='')});const update=()=>{for(const t of R){const x=trialState(t.trial_id);document.querySelector(`[data-replays="${t.trial_id}"]`).textContent=String((x.play_starts.A||0)+(x.play_starts.B||0))}document.getElementById('download').disabled=!complete();save()};document.querySelectorAll('[data-field]').forEach(el=>{const x=trialState(el.dataset.id),target=el.dataset.scope==='side'?x.sides[el.dataset.side]:x.pair;el.value=target[el.dataset.field]??'';el.addEventListener('input',()=>{target[el.dataset.field]=el.value;update()})});document.querySelectorAll('audio').forEach(audio=>{const box=audio.closest('.player'),head=box.querySelector('.playhead'),status=box.querySelector('.target-status'),trial=R.find(x=>x.trial_id===audio.dataset.id);const draw=()=>{if(!Number.isFinite(audio.duration)||audio.duration<=0)return;head.style.left=`${Math.min(100,100*audio.currentTime/audio.duration)}%`;const active=trial.target_intervals.some(x=>audio.currentTime>=x.start_s&&audio.currentTime<=x.end_s)&&!audio.paused;box.classList.toggle('target-now',active);status.textContent=active?'TARGET NOW':'Target position'};for(const event of ['loadedmetadata','timeupdate','pause','ended'])audio.addEventListener(event,draw);audio.addEventListener('play',()=>{document.querySelectorAll('audio').forEach(other=>{if(other!==audio&&!other.paused)other.pause()});trialState(audio.dataset.id).play_starts[audio.dataset.side]++;draw();update()})});update();document.getElementById('download').addEventListener('click',()=>{if(!complete())return;const responses=R.map(t=>{const x=trialState(t.trial_id);return{trial_id:t.trial_id,sides:x.sides,difference_strength:Number(x.pair.difference_strength),category_judgment:x.pair.category_judgment,confidence:Number(x.pair.confidence),interference:x.pair.interference,notes:x.pair.notes??'',play_starts:x.play_starts,replay_count:(x.play_starts.A||0)+(x.play_starts.B||0)}});const payload={schema_version:1,run_id:RUN,protocol_sha256:PROTOCOL,session_id:S.session_id,saved_at:new Date().toISOString(),responses};const blob=new Blob([JSON.stringify(payload,null,2),String.fromCharCode(10)],{type:'application/json'}),link=document.createElement('a');link.href=URL.createObjectURL(blob);link.download='typed-replication-v1-response.json';link.click()});</script></body></html>"""
    return (
        template.replace("__PUBLIC__", json.dumps(public, ensure_ascii=False))
        .replace("__PROTOCOL__", protocol_sha256)
        .replace("__RUN__", RUN_ID)
    )


def build_review() -> dict[str, Any]:
    protocol = protocol_record()
    analysis = json.loads((run_dir() / ANALYSIS_FILE).read_text(encoding="utf-8"))
    if analysis.get("automatic_replication_pass") is not True:
        raise RuntimeError("automatic replication did not advance to blind review")
    layout = blinded_trial_plan()
    records = _records_by_fixture()
    public = _ensure_review_audio(layout, records)
    key = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "protocol_sha256": protocol["protocol_sha256"],
        "trials": [
            {
                "trial_id": trial["trial_id"],
                "fixture_id": trial["fixture_id"],
                "condition": trial["condition"],
                "side_roles": trial["side_roles"],
                "expected_lens_side": next(
                    (
                        side
                        for side, role in trial["side_roles"].items()
                        if role == "lens"
                    ),
                    None,
                ),
            }
            for trial in layout
        ],
    }
    manifest = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "protocol_sha256": protocol["protocol_sha256"],
        "trial_count": len(public),
        "public_trials": public,
        "hidden_fields_absent": True,
    }
    atomic_write_json(run_dir() / BLIND_KEY_FILE, key)
    atomic_write_json(run_dir() / REVIEW_MANIFEST_FILE, manifest)
    atomic_write_text(
        run_dir() / REVIEW_FILE,
        _review_html(public, protocol["protocol_sha256"]),
    )
    return manifest


def _side_gate(side: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "naturalness": int(side["naturalness"]) in {4, 5},
        "delivery": side["delivery"] == "sentence-like",
        "stable_recoverable_meaning": side["meaning"] == "none",
        "artifact": side["artifact"] in {"none", "minor"},
    }
    return {"checks": checks, "pass": all(checks.values())}


def decode_response(path: Path) -> dict[str, Any]:
    protocol = protocol_record()
    raw = path.read_bytes()
    response = json.loads(raw)
    if response.get("run_id") != RUN_ID:
        raise RuntimeError("review response belongs to a different run")
    if response.get("protocol_sha256") != protocol["protocol_sha256"]:
        raise RuntimeError("review response belongs to a different protocol")
    key = json.loads((run_dir() / BLIND_KEY_FILE).read_text(encoding="utf-8"))
    key_by_trial = {row["trial_id"]: row for row in key["trials"]}
    rows = response.get("responses")
    if not isinstance(rows, list) or len(rows) != len(key_by_trial):
        raise RuntimeError("review response is incomplete")
    if {row.get("trial_id") for row in rows} != set(key_by_trial):
        raise RuntimeError("review response trial set differs from the blind key")

    decoded: list[dict[str, Any]] = []
    fixture_results: dict[str, dict[str, bool]] = {
        fixture.fixture_id: {} for fixture in FIXTURES
    }
    for row in rows:
        key_row = key_by_trial[row["trial_id"]]
        side_results = {side: _side_gate(row["sides"][side]) for side in ("A", "B")}
        side_pass = all(record["pass"] for record in side_results.values())
        interference_pass = row["interference"] in {"none", "manageable"}
        if key_row["condition"] == "lens-candidate":
            pair_checks = {
                "difference_strength": int(row["difference_strength"]) >= 5,
                "category_direction": row["category_judgment"]
                == key_row["expected_lens_side"],
                "confidence": int(row["confidence"]) >= 3,
                "interference": interference_pass,
            }
        else:
            pair_checks = {
                "difference_strength": int(row["difference_strength"]) == 1,
                "category_direction": row["category_judgment"] in {"same", "neither"},
                "confidence": True,
                "interference": interference_pass,
            }
        passed = bool(side_pass and all(pair_checks.values()))
        fixture_results[key_row["fixture_id"]][key_row["condition"]] = passed
        decoded.append(
            {
                "trial_id": row["trial_id"],
                "fixture_id": key_row["fixture_id"],
                "condition": key_row["condition"],
                "side_roles": key_row["side_roles"],
                "expected_lens_side": key_row["expected_lens_side"],
                "side_gates": side_results,
                "pair_checks": pair_checks,
                "pass": passed,
                "raw_response": row,
            }
        )

    fixture_summary = {
        fixture_id: {
            **conditions,
            "manual_fixture_pass": conditions
            == {"identity-catch": True, "lens-candidate": True},
        }
        for fixture_id, conditions in fixture_results.items()
    }
    manual_pass = all(
        record["manual_fixture_pass"] for record in fixture_summary.values()
    )
    analysis = json.loads((run_dir() / ANALYSIS_FILE).read_text(encoding="utf-8"))
    promotion = bool(analysis["automatic_replication_pass"] and manual_pass)
    raw_destination = run_dir() / RAW_RESPONSE_FILE
    if raw_destination.exists() and raw_destination.read_bytes() != raw:
        raise RuntimeError("a different raw response is already frozen")
    if not raw_destination.exists():
        raw_destination.write_bytes(raw)
    result = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "protocol_sha256": protocol["protocol_sha256"],
        "raw_response_sha256": hashlib.sha256(raw).hexdigest(),
        "status": "manual_result_complete",
        "decoded_trials": decoded,
        "fixtures": fixture_summary,
        "manual_replication_pass": manual_pass,
        "automatic_replication_pass": analysis["automatic_replication_pass"],
        "production_candidate_promoted": promotion,
        "classification": (
            "typed_target_word_replication_passed_production_candidate"
            if promotion
            else "typed_target_word_replication_failed_no_promotion"
        ),
        "interpretation": (
            "bounded one-rule typed-engine product evidence from one informed "
            "creator-listener; not Brazilian-Portuguese population evidence"
        ),
    }
    atomic_write_json(run_dir() / MANUAL_RESULT_FILE, result)
    return result
