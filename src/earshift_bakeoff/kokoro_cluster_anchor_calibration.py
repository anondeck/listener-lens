from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .bilingual_vowel_replicated_anchors import (
    REPLICATED_ANCHOR_VERSION,
    TRAINING_SEEDS,
    render_seeded_natural_conditions,
)
from .config import Paths, sha256_json
from .kokoro_synthesis import (
    KOKORO_VERSION,
    MODEL_HASHES,
    MODEL_REPO,
    MODEL_REVISION,
    RNG_SEED,
    SAMPLE_RATE_HZ,
    _filtered_symbols,
    _word_column_spans,
    pcm16_bytes,
    pcm_sha256,
)
from .kokoro_typed_confirmation import _sample_interval
from .kokoro_typed_diagnostic import measure_interval_windows
from .kokoro_typed_engine import MAX_CLIPPED_FRACTION
from .kokoro_typed_confirmation_protocol import (
    CEILINGS_HZ,
    DESCRIPTIVE_WINDOW_PERCENTS,
    MEASUREMENT_SCRIPT,
    PRIMARY_WINDOW_PERCENT,
)
from .util import atomic_write_json, sha256_file


CALIBRATION_VERSION = "kokoro-cluster-anchor-calibration-v1"
RUN_ID = "20260718-kokoro-cluster-anchor-calibration-v1"
PARENT_RUN_ID = "20260717-kokoro-strict-shell-confirmation-v1"

# The carrier frame is the strict-shell confirmation's frozen medial fixture;
# only the target word is swapped for each anchor form, so endpoints are
# measured in exactly the sentence context rung-1b candidates will occupy.
EXPECTED_FRAME_TEXT = "Quiet voices map distant roads."
EXPECTED_FRAME_NEUTRAL_PHONEMES = "ʒʃˈɪɪd ʒˈəɡWp vˈæʒ ʧˈWɹwəsh ʤˈʌpθ."
EXPECTED_TARGET_WORD_INDEX = 2

BASELINE_SEED = RNG_SEED
MIN_CROSS_EXTRA_COSINE = 0.5
SEPARATION_FLOOR_BARK = 0.25
MAX_SEED_SPREAD_BARK = 0.25

# These constants belong to the already-rendered v1/v2 evidence and must never
# follow the active product shell.  Keeping them local prevents a later shell
# version from silently changing a frozen protocol when it is reconstructed.
LEGACY_CLUSTER_SHELL_VERSION = 2
LEGACY_CLUSTER_NEUTRAL_SHELL = "vˈæs"
LEGACY_CLUSTER_LENS_SHELL = "vˈɛs"
LEGACY_CLUSTER_EXTRA_CONSONANTS = ("t", "k", "p")
LEGACY_REAL_WORD_ANCHORS = ("vˈæst", "vˈɛst")

# Backwards-compatible public name used by the v1 tests and artifact readers.
REAL_WORD_ANCHORS = LEGACY_REAL_WORD_ANCHORS

# The rendered v3 evidence used the ʒ-onset shell with the full t/k/p pool.
# The same pinning rule applies: these constants belong to the frozen v3
# artifacts and must never follow the active product shell.
V3_CLUSTER_SHELL_VERSION = 3
V3_CLUSTER_NEUTRAL_SHELL = "ʒˈæs"
V3_CLUSTER_LENS_SHELL = "ʒˈɛs"
V3_CLUSTER_EXTRA_CONSONANTS = ("t", "k", "p")
V3_REAL_WORD_ANCHORS: tuple[str, ...] = ()


def run_dir() -> Path:
    return Paths().artifacts / "typed-engine" / RUN_ID


def parent_dir() -> Path:
    return Paths().artifacts / "typed-engine" / PARENT_RUN_ID


def _load_json(path: Path) -> Any:
    if not path.is_file():
        raise RuntimeError(f"required frozen artifact is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _verify_internal_hash(
    record: dict[str, Any], digest_key: str, *, label: str
) -> None:
    expected = record.get(digest_key)
    if not isinstance(expected, str) or len(expected) != 64:
        raise RuntimeError(f"{label} is missing {digest_key}")
    payload = {key: value for key, value in record.items() if key != digest_key}
    if sha256_json(payload) != expected:
        raise RuntimeError(f"{label} has an invalid {digest_key}")


def _verified_parent_fixture() -> dict[str, Any]:
    protocol = _load_json(parent_dir() / "protocol.json")
    fixture = protocol["new_fixture"]
    if (
        fixture["text"] != EXPECTED_FRAME_TEXT
        or fixture["neutral_phonemes"] != EXPECTED_FRAME_NEUTRAL_PHONEMES
        or fixture["target_word_indexes"] != [EXPECTED_TARGET_WORD_INDEX]
    ):
        raise RuntimeError("strict-shell parent frame drifted from frozen constants")
    analysis = _load_json(parent_dir() / "analysis.json")
    if not analysis.get("automatic_pass"):
        raise RuntimeError("strict-shell parent is not an automatic pass")
    return {
        "run_id": PARENT_RUN_ID,
        "protocol_sha256": protocol["protocol_sha256"],
        "protocol_file_sha256": sha256_file(parent_dir() / "protocol.json"),
        "analysis_sha256": sha256_file(parent_dir() / "analysis.json"),
        "fixture_plan_sha256": fixture["plan_sha256"],
        "classification": analysis["classification"],
        "frame_words": [word["neutral_phone"] for word in fixture["words"]],
    }


def _anchor_conditions(
    *,
    neutral_shell: str,
    lens_shell: str,
    extras: Sequence[str],
    real_word_anchors: Sequence[str],
) -> list[dict[str, Any]]:
    parent = _verified_parent_fixture()
    frame = list(parent["frame_words"])
    conditions: list[dict[str, Any]] = []
    for endpoint, shell in (("ae", neutral_shell), ("eh", lens_shell)):
        for extra in extras:
            target_phone = shell + extra
            words = list(frame)
            words[EXPECTED_TARGET_WORD_INDEX] = target_phone
            conditions.append(
                {
                    "condition_id": f"{endpoint}-{extra}",
                    "endpoint": endpoint,
                    "extra_consonant": extra,
                    "target_phone": target_phone,
                    "target_word_index": EXPECTED_TARGET_WORD_INDEX,
                    "phonemes": " ".join(words[:-1]) + " " + words[-1] + ".",
                    "is_real_english_word": target_phone in real_word_anchors,
                }
            )
    return conditions


def anchor_conditions() -> list[dict[str, Any]]:
    """Reconstruct the immutable v1/v2 anchor manifest."""

    return _anchor_conditions(
        neutral_shell=LEGACY_CLUSTER_NEUTRAL_SHELL,
        lens_shell=LEGACY_CLUSTER_LENS_SHELL,
        extras=LEGACY_CLUSTER_EXTRA_CONSONANTS,
        real_word_anchors=LEGACY_REAL_WORD_ANCHORS,
    )


def anchor_conditions_v3() -> list[dict[str, Any]]:
    """Reconstruct the immutable v3 anchor manifest."""

    return _anchor_conditions(
        neutral_shell=V3_CLUSTER_NEUTRAL_SHELL,
        lens_shell=V3_CLUSTER_LENS_SHELL,
        extras=V3_CLUSTER_EXTRA_CONSONANTS,
        real_word_anchors=V3_REAL_WORD_ANCHORS,
    )


def protocol_record() -> dict[str, Any]:
    parent = _verified_parent_fixture()
    conditions = anchor_conditions()
    payload: dict[str, Any] = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "version": CALIBRATION_VERSION,
        "status": "frozen_before_any_render",
        "purpose": (
            "Context-matched /ae/ and /eh/ endpoint anchors for the voiceless "
            "cluster shell, rendered inside the frozen strict-shell medial "
            "frame, for rung-1b use only."
        ),
        "parent": {
            key: parent[key]
            for key in (
                "run_id",
                "protocol_sha256",
                "protocol_file_sha256",
                "analysis_sha256",
                "fixture_plan_sha256",
                "classification",
            )
        },
        "cluster_shell_version": LEGACY_CLUSTER_SHELL_VERSION,
        "replicated_anchor_library_version": REPLICATED_ANCHOR_VERSION,
        "renderer": {
            "kokoro_version": KOKORO_VERSION,
            "model_repo": MODEL_REPO,
            "model_revision": MODEL_REVISION,
            "model_hashes": dict(MODEL_HASHES),
            "voice": "af_heart",
        },
        "seeds": {
            "training": list(TRAINING_SEEDS),
            "baseline": BASELINE_SEED,
            "baseline_double_decode_bit_identity_required": True,
        },
        "conditions": conditions,
        "measurement": {
            "praat_script_sha256": sha256_file(MEASUREMENT_SCRIPT),
            "ceilings_hz": list(CEILINGS_HZ),
            "primary_window_percent": PRIMARY_WINDOW_PERCENT,
            "descriptive_window_percents": list(DESCRIPTIVE_WINDOW_PERCENTS),
            "interval": "stress-plus-vowel span of the swapped target word",
        },
        "gates": {
            "per_condition_all_training_seeds_measurement_valid": True,
            "max_training_seed_spread_bark": MAX_SEED_SPREAD_BARK,
            "endpoint_separation_floor_bark": SEPARATION_FLOOR_BARK,
            "min_cross_extra_direction_cosine": MIN_CROSS_EXTRA_COSINE,
            "leave_one_extra_out_direction_agreement_required": True,
            "endpoint_definition": "mean of the three training-seed points",
        },
        "predetermined_outcomes": {
            "pass": (
                "The per-family endpoints freeze for rung-1b consumption only; "
                "no product claim, shape enablement, or listener inference."
            ),
            "fail": (
                "Record the failure, route by mechanism, and design a new "
                "versioned calibration; nothing is reclassified or rerun."
            ),
        },
        "scope": {
            "api_calls": 0,
            "listening": "none — instrument media only",
            "logical_decodes": len(conditions) * (len(TRAINING_SEEDS) + 2),
            "production_enabled": False,
        },
    }
    return {**payload, "protocol_sha256": sha256_json(payload)}


def prepare() -> dict[str, Any]:
    protocol = protocol_record()
    destination = run_dir() / "protocol.json"
    if destination.is_file():
        existing = _load_json(destination)
        if existing != protocol:
            raise RuntimeError("existing cluster anchor protocol differs from freeze")
        return existing
    destination.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(destination, protocol)
    return protocol


def _write_wav_once(path: Path, audio: "np.ndarray", base_dir: Path) -> dict[str, Any]:
    import os
    import wave

    values = np.asarray(audio, dtype=np.float64).reshape(-1)
    if not values.size or not np.isfinite(values).all():
        raise RuntimeError(f"anchor decode produced invalid audio for {path.name}")
    if not path.exists():
        temporary = path.with_name(path.name + ".partial")
        try:
            with wave.open(str(temporary), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(SAMPLE_RATE_HZ)
                handle.writeframes(pcm16_bytes(values))
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
    clipped = float(np.mean(np.abs(values) >= 1.0))
    return {
        "relative_path": str(path.relative_to(base_dir)),
        "sample_count": int(values.size),
        "clipped_fraction": clipped,
        "clipping_pass": bool(clipped < MAX_CLIPPED_FRACTION),
        "pcm_sha256": pcm_sha256(values),
        "wav_sha256": sha256_file(path),
    }


def _target_interval(
    model: Any, phonemes: str, durations: Sequence[int], sample_count: int
) -> dict[str, Any]:
    expected = len(_filtered_symbols(model, phonemes)) + 2
    if len(durations) != expected:
        raise RuntimeError("anchor duration count differs from the phoneme plan")
    total = sum(int(value) for value in durations)
    if total <= 0 or sample_count % total:
        raise RuntimeError("anchor samples do not map to integral alignment frames")
    spans = _word_column_spans(model, phonemes)
    span = spans[EXPECTED_TARGET_WORD_INDEX]
    if len(span) != 5:
        raise RuntimeError("anchor target word span drifted from the CˈVCC shell")
    stress_vowel = span[1:3]
    return _sample_interval(stress_vowel, durations, sample_count // total)


def _seed_point(
    path: Path, interval: dict[str, Any], ceiling: int
) -> dict[str, Any] | None:
    windows = measure_interval_windows(path, interval, ceiling)
    primary = windows[str(PRIMARY_WINDOW_PERCENT)]
    if not (primary.get("measurement_valid") and primary.get("plausibility_pass")):
        return None
    return {
        "point": [float(primary["f1_bark"]), float(primary["f2_bark"])],
        "windows": windows,
    }


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    a = np.asarray(left, dtype=float)
    b = np.asarray(right, dtype=float)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denominator) if denominator else -1.0


def run() -> dict[str, Any]:
    protocol = prepare()
    analysis_path = run_dir() / "analysis.json"
    if analysis_path.exists():
        raise RuntimeError("cluster anchor calibration has already been analyzed")
    from .kokoro_synthesis import KokoroSynthesisRuntime

    runtime = KokoroSynthesisRuntime.load()
    audio_dir = run_dir() / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    seeds = tuple(protocol["seeds"]["training"]) + (protocol["seeds"]["baseline"],)
    conditions: dict[str, Any] = {}
    for condition in protocol["conditions"]:
        cid = condition["condition_id"]
        seeded = render_seeded_natural_conditions(
            runtime,
            phonemes=condition["phonemes"],
            reference_phonemes=condition["phonemes"],
            seeds=seeds,
        )
        baseline_repeat = render_seeded_natural_conditions(
            runtime,
            phonemes=condition["phonemes"],
            reference_phonemes=condition["phonemes"],
            seeds=(protocol["seeds"]["baseline"],),
        )
        first = seeded.audio_by_seed[protocol["seeds"]["baseline"]]
        second = baseline_repeat.audio_by_seed[protocol["seeds"]["baseline"]]
        baseline_identical = bool(
            first.shape == second.shape and np.array_equal(first, second)
        )
        sample_count = int(first.size)
        interval = _target_interval(
            runtime.model,
            condition["phonemes"],
            seeded.predicted_durations,
            sample_count,
        )
        per_seed: dict[str, Any] = {}
        for seed in seeds:
            wav_path = audio_dir / f"{cid}-seed-{seed}.wav"
            record = _write_wav_once(wav_path, seeded.audio_by_seed[seed], run_dir())
            record["families"] = {
                str(ceiling): _seed_point(wav_path, interval, ceiling)
                for ceiling in CEILINGS_HZ
            }
            per_seed[str(seed)] = record
        conditions[cid] = {
            "condition": condition,
            "measurement_interval": interval,
            "baseline_double_decode_bit_identical": baseline_identical,
            "seeds": per_seed,
        }
    gates: dict[str, Any] = {"families": {}}
    endpoints_by_extra: dict[str, Any] = {}
    training = [str(seed) for seed in protocol["seeds"]["training"]]
    overall = True
    for ceiling in CEILINGS_HZ:
        key = str(ceiling)
        family_rows: dict[str, Any] = {}
        vectors: dict[str, Any] = {}
        family_pass = True
        for extra in LEGACY_CLUSTER_EXTRA_CONSONANTS:
            row: dict[str, Any] = {}
            for endpoint in ("ae", "eh"):
                cid = f"{endpoint}-{extra}"
                points = [
                    conditions[cid]["seeds"][seed]["families"][key] for seed in training
                ]
                valid = all(point is not None for point in points)
                if valid:
                    values = np.asarray(
                        [point["point"] for point in points], dtype=float
                    )
                    spread = float(
                        max(
                            np.linalg.norm(values[i] - values[j])
                            for i in range(len(values))
                            for j in range(i + 1, len(values))
                        )
                    )
                    mean_point = [float(v) for v in values.mean(axis=0)]
                else:
                    spread, mean_point = None, None
                row[endpoint] = {
                    "all_training_seeds_valid": valid,
                    "seed_spread_bark": spread,
                    "spread_pass": bool(
                        valid and spread is not None and spread <= MAX_SEED_SPREAD_BARK
                    ),
                    "endpoint_bark": mean_point,
                }
            both_valid = row["ae"]["spread_pass"] and row["eh"]["spread_pass"]
            if both_valid:
                separation = float(
                    np.linalg.norm(
                        np.asarray(row["eh"]["endpoint_bark"])
                        - np.asarray(row["ae"]["endpoint_bark"])
                    )
                )
                vectors[extra] = [
                    float(v)
                    for v in (
                        np.asarray(row["eh"]["endpoint_bark"])
                        - np.asarray(row["ae"]["endpoint_bark"])
                    )
                ]
            else:
                separation = None
            row["separation_bark"] = separation
            row["separation_pass"] = bool(
                separation is not None and separation >= SEPARATION_FLOOR_BARK
            )
            family_rows[extra] = row
            family_pass = family_pass and row["separation_pass"]
        pair_cosines = {}
        extras = list(LEGACY_CLUSTER_EXTRA_CONSONANTS)
        for i in range(len(extras)):
            for j in range(i + 1, len(extras)):
                left, right = extras[i], extras[j]
                value = (
                    _cosine(vectors[left], vectors[right])
                    if left in vectors and right in vectors
                    else None
                )
                pair_cosines[f"{left}-{right}"] = value
                family_pass = family_pass and bool(
                    value is not None and value >= MIN_CROSS_EXTRA_COSINE
                )
        gates["families"][key] = {
            "extras": family_rows,
            "cross_extra_cosines": pair_cosines,
            "pass": family_pass,
        }
        overall = overall and family_pass
        if family_pass:
            for extra in LEGACY_CLUSTER_EXTRA_CONSONANTS:
                endpoints_by_extra.setdefault(extra, {})[key] = {
                    "ae_bark": family_rows[extra]["ae"]["endpoint_bark"],
                    "eh_bark": family_rows[extra]["eh"]["endpoint_bark"],
                }
    baseline_ok = all(
        row["baseline_double_decode_bit_identical"] for row in conditions.values()
    )
    overall = overall and baseline_ok
    analysis = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "protocol_sha256": protocol["protocol_sha256"],
        "api_calls_made": 0,
        "baseline_double_decode_bit_identical_all": baseline_ok,
        "conditions": conditions,
        "gates": gates,
        "endpoints_by_extra": endpoints_by_extra if overall else {},
        "classification": (
            "cluster_anchor_calibration_pass"
            if overall
            else "cluster_anchor_calibration_fail"
        ),
        "production_enabled": False,
    }
    analysis["analysis_sha256"] = sha256_json(
        {key: value for key, value in analysis.items() if key != "analysis_sha256"}
    )
    atomic_write_json(analysis_path, analysis)
    return analysis


def _analyze_extras_families(
    conditions: dict[str, Any], *, extras: Sequence[str], training: Sequence[int]
) -> tuple[dict[str, Any], dict[str, Any], bool]:
    """Shared v2-design per-extra gate evaluation over rendered conditions."""

    gates: dict[str, Any] = {"families": {}}
    endpoints_by_extra: dict[str, Any] = {}
    training_keys = [str(seed) for seed in training]
    overall = True
    for ceiling in CEILINGS_HZ:
        key = str(ceiling)
        family_rows: dict[str, Any] = {}
        vectors: dict[str, Any] = {}
        family_pass = True
        for extra in extras:
            row: dict[str, Any] = {}
            for endpoint in ("ae", "eh"):
                cid = f"{endpoint}-{extra}"
                points = [
                    conditions[cid]["seeds"][seed]["families"][key]
                    for seed in training_keys
                ]
                valid = all(point is not None for point in points)
                if valid:
                    values = np.asarray(
                        [point["point"] for point in points], dtype=float
                    )
                    pairwise = [
                        float(np.linalg.norm(values[i] - values[j]))
                        for i in range(len(values))
                        for j in range(i + 1, len(values))
                    ]
                    spread = float(np.median(pairwise))
                    mean_point = [float(v) for v in values.mean(axis=0)]
                else:
                    spread, mean_point = None, None
                row[endpoint] = {
                    "all_training_seeds_valid": valid,
                    "median_pairwise_spread_bark": spread,
                    "spread_pass": bool(
                        valid and spread is not None and spread <= MAX_SEED_SPREAD_BARK
                    ),
                    "endpoint_bark": mean_point,
                }
            both = row["ae"]["spread_pass"] and row["eh"]["spread_pass"]
            if both:
                vector = np.asarray(row["eh"]["endpoint_bark"]) - np.asarray(
                    row["ae"]["endpoint_bark"]
                )
                separation = float(np.linalg.norm(vector))
                vectors[extra] = [float(v) for v in vector]
                f2_positive = bool(vector[1] > 0.0)
            else:
                separation, f2_positive = None, False
            row["separation_bark"] = separation
            row["separation_pass"] = bool(
                separation is not None and separation >= SEPARATION_FLOOR_BARK
            )
            row["f2_direction_positive"] = f2_positive
            extra_pass = row["separation_pass"] and f2_positive
            row["pass"] = extra_pass
            family_rows[extra] = row
            family_pass = family_pass and extra_pass
        descriptive = {}
        ordered = list(extras)
        for i in range(len(ordered)):
            for j in range(i + 1, len(ordered)):
                left, right = ordered[i], ordered[j]
                descriptive[f"{left}-{right}"] = (
                    _cosine(vectors[left], vectors[right])
                    if left in vectors and right in vectors
                    else None
                )
        gates["families"][key] = {
            "extras": family_rows,
            "descriptive_cross_extra_cosines": descriptive,
            "pass": family_pass,
        }
        overall = overall and family_pass
        if family_pass:
            for extra in extras:
                endpoints_by_extra.setdefault(extra, {})[key] = {
                    "ae_bark": family_rows[extra]["ae"]["endpoint_bark"],
                    "eh_bark": family_rows[extra]["eh"]["endpoint_bark"],
                }
    return gates, endpoints_by_extra, overall


V2_CALIBRATION_VERSION = "kokoro-cluster-anchor-calibration-v2"
V2_RUN_ID = "20260718-kokoro-cluster-anchor-calibration-v2"
V2_TRAINING_SEEDS = TRAINING_SEEDS + (202_607_174, 202_607_175)


def run_dir_v2() -> Path:
    return Paths().artifacts / "typed-engine" / V2_RUN_ID


def _verified_v1_failure() -> dict[str, Any]:
    protocol_path = run_dir() / "protocol.json"
    analysis_path = run_dir() / "analysis.json"
    protocol = _load_json(protocol_path)
    analysis = _load_json(analysis_path)
    _verify_internal_hash(protocol, "protocol_sha256", label="v1 protocol")
    _verify_internal_hash(analysis, "analysis_sha256", label="v1 analysis")
    if analysis["classification"] != "cluster_anchor_calibration_fail":
        raise RuntimeError("v1 parent must remain the frozen failed calibration")
    return {
        "run_id": RUN_ID,
        "protocol_sha256": protocol["protocol_sha256"],
        "analysis_sha256": analysis["analysis_sha256"],
        "classification": analysis["classification"],
    }


def protocol_record_v2() -> dict[str, Any]:
    base = protocol_record()
    payload: dict[str, Any] = {
        key: value
        for key, value in base.items()
        if key not in ("protocol_sha256", "run_id", "version", "seeds", "gates")
    }
    payload["run_id"] = V2_RUN_ID
    payload["version"] = V2_CALIBRATION_VERSION
    payload["v1_parent"] = _verified_v1_failure()
    payload["seeds"] = {
        "training": list(V2_TRAINING_SEEDS),
        "baseline": BASELINE_SEED,
        "baseline_double_decode_bit_identity_required": True,
    }
    payload["gates"] = {
        "per_condition_all_training_seeds_measurement_valid": True,
        "seed_spread_statistic": "median_pairwise_distance",
        "max_training_seed_spread_bark": MAX_SEED_SPREAD_BARK,
        "endpoint_separation_floor_bark": SEPARATION_FLOOR_BARK,
        "per_extra_f2_direction_positive_required": True,
        "cross_extra_cosines": "descriptive_only",
        "endpoint_definition": "mean of the five training-seed points",
    }
    payload["scope"] = dict(base["scope"])
    payload["scope"]["logical_decodes"] = len(base["conditions"]) * (
        len(V2_TRAINING_SEEDS) + 2
    )
    return {**payload, "protocol_sha256": sha256_json(payload)}


def prepare_v2() -> dict[str, Any]:
    protocol = protocol_record_v2()
    destination = run_dir_v2() / "protocol.json"
    if destination.is_file():
        existing = _load_json(destination)
        if existing != protocol:
            raise RuntimeError("existing v2 anchor protocol differs from freeze")
        return existing
    destination.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(destination, protocol)
    return protocol


def run_v2() -> dict[str, Any]:
    protocol = prepare_v2()
    analysis_path = run_dir_v2() / "analysis.json"
    if analysis_path.exists():
        raise RuntimeError("v2 cluster anchor calibration has already been analyzed")
    from .kokoro_synthesis import KokoroSynthesisRuntime

    runtime = KokoroSynthesisRuntime.load()
    audio_dir = run_dir_v2() / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    training = tuple(protocol["seeds"]["training"])
    seeds = training + (protocol["seeds"]["baseline"],)
    conditions: dict[str, Any] = {}
    for condition in protocol["conditions"]:
        cid = condition["condition_id"]
        seeded = render_seeded_natural_conditions(
            runtime,
            phonemes=condition["phonemes"],
            reference_phonemes=condition["phonemes"],
            seeds=seeds,
        )
        repeat = render_seeded_natural_conditions(
            runtime,
            phonemes=condition["phonemes"],
            reference_phonemes=condition["phonemes"],
            seeds=(protocol["seeds"]["baseline"],),
        )
        first = seeded.audio_by_seed[protocol["seeds"]["baseline"]]
        second = repeat.audio_by_seed[protocol["seeds"]["baseline"]]
        baseline_identical = bool(
            first.shape == second.shape and np.array_equal(first, second)
        )
        interval = _target_interval(
            runtime.model,
            condition["phonemes"],
            seeded.predicted_durations,
            int(first.size),
        )
        per_seed: dict[str, Any] = {}
        for seed in seeds:
            wav_path = audio_dir / f"{cid}-seed-{seed}.wav"
            record = _write_wav_once(wav_path, seeded.audio_by_seed[seed], run_dir_v2())
            record["families"] = {
                str(ceiling): _seed_point(wav_path, interval, ceiling)
                for ceiling in CEILINGS_HZ
            }
            per_seed[str(seed)] = record
        conditions[cid] = {
            "condition": condition,
            "measurement_interval": interval,
            "baseline_double_decode_bit_identical": baseline_identical,
            "seeds": per_seed,
        }
    gates: dict[str, Any] = {"families": {}}
    endpoints_by_extra: dict[str, Any] = {}
    training_keys = [str(seed) for seed in training]
    overall = True
    for ceiling in CEILINGS_HZ:
        key = str(ceiling)
        family_rows: dict[str, Any] = {}
        vectors: dict[str, Any] = {}
        family_pass = True
        for extra in LEGACY_CLUSTER_EXTRA_CONSONANTS:
            row: dict[str, Any] = {}
            for endpoint in ("ae", "eh"):
                cid = f"{endpoint}-{extra}"
                points = [
                    conditions[cid]["seeds"][seed]["families"][key]
                    for seed in training_keys
                ]
                valid = all(point is not None for point in points)
                if valid:
                    values = np.asarray(
                        [point["point"] for point in points], dtype=float
                    )
                    pairwise = [
                        float(np.linalg.norm(values[i] - values[j]))
                        for i in range(len(values))
                        for j in range(i + 1, len(values))
                    ]
                    spread = float(np.median(pairwise))
                    mean_point = [float(v) for v in values.mean(axis=0)]
                else:
                    spread, mean_point = None, None
                row[endpoint] = {
                    "all_training_seeds_valid": valid,
                    "median_pairwise_spread_bark": spread,
                    "spread_pass": bool(
                        valid and spread is not None and spread <= MAX_SEED_SPREAD_BARK
                    ),
                    "endpoint_bark": mean_point,
                }
            both = row["ae"]["spread_pass"] and row["eh"]["spread_pass"]
            if both:
                vector = np.asarray(row["eh"]["endpoint_bark"]) - np.asarray(
                    row["ae"]["endpoint_bark"]
                )
                separation = float(np.linalg.norm(vector))
                vectors[extra] = [float(v) for v in vector]
                f2_positive = bool(vector[1] > 0.0)
            else:
                separation, f2_positive = None, False
            row["separation_bark"] = separation
            row["separation_pass"] = bool(
                separation is not None and separation >= SEPARATION_FLOOR_BARK
            )
            row["f2_direction_positive"] = f2_positive
            extra_pass = row["separation_pass"] and f2_positive
            row["pass"] = extra_pass
            family_rows[extra] = row
            family_pass = family_pass and extra_pass
        descriptive = {}
        extras = list(LEGACY_CLUSTER_EXTRA_CONSONANTS)
        for i in range(len(extras)):
            for j in range(i + 1, len(extras)):
                left, right = extras[i], extras[j]
                descriptive[f"{left}-{right}"] = (
                    _cosine(vectors[left], vectors[right])
                    if left in vectors and right in vectors
                    else None
                )
        gates["families"][key] = {
            "extras": family_rows,
            "descriptive_cross_extra_cosines": descriptive,
            "pass": family_pass,
        }
        overall = overall and family_pass
        if family_pass:
            for extra in LEGACY_CLUSTER_EXTRA_CONSONANTS:
                endpoints_by_extra.setdefault(extra, {})[key] = {
                    "ae_bark": family_rows[extra]["ae"]["endpoint_bark"],
                    "eh_bark": family_rows[extra]["eh"]["endpoint_bark"],
                }
    baseline_ok = all(
        row["baseline_double_decode_bit_identical"] for row in conditions.values()
    )
    overall = overall and baseline_ok
    analysis = {
        "schema_version": 1,
        "run_id": V2_RUN_ID,
        "protocol_sha256": protocol["protocol_sha256"],
        "api_calls_made": 0,
        "baseline_double_decode_bit_identical_all": baseline_ok,
        "conditions": conditions,
        "gates": gates,
        "endpoints_by_extra": endpoints_by_extra if overall else {},
        "classification": (
            "cluster_anchor_calibration_v2_pass"
            if overall
            else "cluster_anchor_calibration_v2_fail"
        ),
        "production_enabled": False,
    }
    analysis["analysis_sha256"] = sha256_json(
        {key: value for key, value in analysis.items() if key != "analysis_sha256"}
    )
    atomic_write_json(analysis_path, analysis)
    return analysis


V3_CALIBRATION_VERSION = "kokoro-cluster-anchor-calibration-v3"
V3_RUN_ID = "20260718-kokoro-cluster-anchor-calibration-v3"


def run_dir_v3() -> Path:
    return Paths().artifacts / "typed-engine" / V3_RUN_ID


def _verified_v2_pass() -> dict[str, Any]:
    protocol_path = run_dir_v2() / "protocol.json"
    analysis_path = run_dir_v2() / "analysis.json"
    protocol = _load_json(protocol_path)
    analysis = _load_json(analysis_path)
    _verify_internal_hash(protocol, "protocol_sha256", label="v2 protocol")
    _verify_internal_hash(analysis, "analysis_sha256", label="v2 analysis")
    if analysis["classification"] != "cluster_anchor_calibration_v2_pass":
        raise RuntimeError("v2 parent must remain the frozen passed calibration")
    return {
        "run_id": V2_RUN_ID,
        "protocol_sha256": protocol["protocol_sha256"],
        "protocol_file_sha256": sha256_file(protocol_path),
        "analysis_sha256": analysis["analysis_sha256"],
        "analysis_file_sha256": sha256_file(analysis_path),
        "classification": analysis["classification"],
        "note": (
            "The v2 instrument design carries forward unchanged; its vˈæs-shell "
            "endpoints are superseded because that shell is lexically saturated "
            "and can never produce gate-clean carriers."
        ),
    }


def _shell_gate_probe(
    *, neutral_shell: str, lens_shell: str, extras: Sequence[str]
) -> dict[str, Any]:
    """Prove the exact given shell forms clear both pinned lexical gates."""

    from .kokoro_cluster_shell import ClusterShellPlanner
    from .kokoro_typed_engine import (
        CarrierAssignment,
        _surface_for,
        local_engine_assets,
    )

    planner = ClusterShellPlanner.load()
    assets = local_engine_assets()
    rows: list[dict[str, Any]] = []
    for extra in extras:
        neutral_phone = neutral_shell + extra
        lens_phone = lens_shell + extra
        assignment = CarrierAssignment(
            neutral_surface=_surface_for(neutral_phone),
            lens_surface=_surface_for(lens_phone),
            neutral_phone=neutral_phone,
            lens_phone=lens_phone,
            candidate_attempt=0,
        )
        reasons = sorted(planner._isolated_reasons(assignment))
        rows.append(
            {
                "extra_consonant": extra,
                "neutral_surface": assignment.neutral_surface,
                "lens_surface": assignment.lens_surface,
                "neutral_phone": neutral_phone,
                "lens_phone": lens_phone,
                "rejection_reasons": reasons,
                "pass": not reasons,
            }
        )
    if not all(row["pass"] for row in rows):
        raise RuntimeError("cluster shell forms do not pass the pinned lexical gates")
    return {
        "method": "KokoroTypedPlanner._isolated_reasons",
        "gate_assets": {
            "engine_version": assets["engine_version"],
            "dependency_lock_sha256": assets["dependency_lock_sha256"],
            "gate_database_sha256": assets["gate_database_sha256"],
            "kokoro_gate_database_sha256": assets["kokoro_gate_database_sha256"],
        },
        "all_forms_pass": True,
        "rows": rows,
    }


def _v3_gate_probe() -> dict[str, Any]:
    """Reproduce the frozen v3 probe from the pinned v3 shell forms."""

    return _shell_gate_probe(
        neutral_shell=V3_CLUSTER_NEUTRAL_SHELL,
        lens_shell=V3_CLUSTER_LENS_SHELL,
        extras=V3_CLUSTER_EXTRA_CONSONANTS,
    )


def protocol_record_v3() -> dict[str, Any]:
    # Inherit the frozen artifact, not the live v2 builder. The latter is kept
    # reproducible too, but this file binding makes the evidence dependency
    # explicit and prevents future code evolution from changing v3's parent.
    base_path = run_dir_v2() / "protocol.json"
    base = _load_json(base_path)
    _verify_internal_hash(base, "protocol_sha256", label="v2 protocol")
    payload: dict[str, Any] = {
        key: value
        for key, value in base.items()
        if key
        not in (
            "protocol_sha256",
            "run_id",
            "version",
            "v1_parent",
            "conditions",
            "cluster_shell_version",
            "purpose",
            "predetermined_outcomes",
        )
    }
    payload["run_id"] = V3_RUN_ID
    payload["version"] = V3_CALIBRATION_VERSION
    payload["purpose"] = (
        "Context-matched /ae/ and /eh/ endpoint anchors for the gate-clean "
        "v3 voiceless cluster shell. The v2 measurement design is unchanged; "
        "only the shell forms and their endpoints are newly rendered."
    )
    payload["cluster_shell_version"] = V3_CLUSTER_SHELL_VERSION
    payload["conditions"] = anchor_conditions_v3()
    payload["v2_parent"] = _verified_v2_pass()
    payload["v2_parent"]["protocol_file_sha256"] = sha256_file(base_path)
    payload["isolated_gate_probe"] = _v3_gate_probe()
    payload["predetermined_outcomes"] = {
        "pass": (
            "Freeze the v3 per-extra endpoints for one bounded cluster "
            "confirmation; no product enablement follows from calibration alone."
        ),
        "fail": (
            "Freeze the failure and do not run the cluster confirmation or "
            "enable the cluster planner."
        ),
    }
    payload["scope"] = dict(payload["scope"])
    payload["scope"]["logical_decodes"] = len(payload["conditions"]) * (
        len(payload["seeds"]["training"]) + 2
    )
    return {**payload, "protocol_sha256": sha256_json(payload)}


def prepare_v3() -> dict[str, Any]:
    protocol = protocol_record_v3()
    destination = run_dir_v3() / "protocol.json"
    if destination.is_file():
        existing = _load_json(destination)
        if existing != protocol:
            raise RuntimeError("existing v3 anchor protocol differs from freeze")
        return existing
    destination.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(destination, protocol)
    return protocol


def run_v3() -> dict[str, Any]:
    protocol = prepare_v3()
    analysis_path = run_dir_v3() / "analysis.json"
    if analysis_path.exists():
        raise RuntimeError("v3 cluster anchor calibration has already been analyzed")
    from .kokoro_synthesis import KokoroSynthesisRuntime

    runtime = KokoroSynthesisRuntime.load()
    audio_dir = run_dir_v3() / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    training = tuple(protocol["seeds"]["training"])
    seeds = training + (protocol["seeds"]["baseline"],)
    conditions: dict[str, Any] = {}
    for condition in protocol["conditions"]:
        cid = condition["condition_id"]
        seeded = render_seeded_natural_conditions(
            runtime,
            phonemes=condition["phonemes"],
            reference_phonemes=condition["phonemes"],
            seeds=seeds,
        )
        repeat = render_seeded_natural_conditions(
            runtime,
            phonemes=condition["phonemes"],
            reference_phonemes=condition["phonemes"],
            seeds=(protocol["seeds"]["baseline"],),
        )
        first = seeded.audio_by_seed[protocol["seeds"]["baseline"]]
        second = repeat.audio_by_seed[protocol["seeds"]["baseline"]]
        baseline_identical = bool(
            first.shape == second.shape and np.array_equal(first, second)
        )
        interval = _target_interval(
            runtime.model,
            condition["phonemes"],
            seeded.predicted_durations,
            int(first.size),
        )
        per_seed: dict[str, Any] = {}
        for seed in seeds:
            wav_path = audio_dir / f"{cid}-seed-{seed}.wav"
            record = _write_wav_once(wav_path, seeded.audio_by_seed[seed], run_dir_v3())
            record["families"] = {
                str(ceiling): _seed_point(wav_path, interval, ceiling)
                for ceiling in CEILINGS_HZ
            }
            per_seed[str(seed)] = record
        conditions[cid] = {
            "condition": condition,
            "measurement_interval": interval,
            "baseline_double_decode_bit_identical": baseline_identical,
            "seeds": per_seed,
        }
    gates: dict[str, Any] = {"families": {}}
    endpoints_by_extra: dict[str, Any] = {}
    training_keys = [str(seed) for seed in training]
    overall = True
    for ceiling in CEILINGS_HZ:
        key = str(ceiling)
        family_rows: dict[str, Any] = {}
        vectors: dict[str, Any] = {}
        family_pass = True
        for extra in V3_CLUSTER_EXTRA_CONSONANTS:
            row: dict[str, Any] = {}
            for endpoint in ("ae", "eh"):
                cid = f"{endpoint}-{extra}"
                points = [
                    conditions[cid]["seeds"][seed]["families"][key]
                    for seed in training_keys
                ]
                valid = all(point is not None for point in points)
                if valid:
                    values = np.asarray(
                        [point["point"] for point in points], dtype=float
                    )
                    pairwise = [
                        float(np.linalg.norm(values[i] - values[j]))
                        for i in range(len(values))
                        for j in range(i + 1, len(values))
                    ]
                    spread = float(np.median(pairwise))
                    mean_point = [float(v) for v in values.mean(axis=0)]
                else:
                    spread, mean_point = None, None
                row[endpoint] = {
                    "all_training_seeds_valid": valid,
                    "median_pairwise_spread_bark": spread,
                    "spread_pass": bool(
                        valid and spread is not None and spread <= MAX_SEED_SPREAD_BARK
                    ),
                    "endpoint_bark": mean_point,
                }
            both = row["ae"]["spread_pass"] and row["eh"]["spread_pass"]
            if both:
                vector = np.asarray(row["eh"]["endpoint_bark"]) - np.asarray(
                    row["ae"]["endpoint_bark"]
                )
                separation = float(np.linalg.norm(vector))
                vectors[extra] = [float(v) for v in vector]
                f2_positive = bool(vector[1] > 0.0)
            else:
                separation, f2_positive = None, False
            row["separation_bark"] = separation
            row["separation_pass"] = bool(
                separation is not None and separation >= SEPARATION_FLOOR_BARK
            )
            row["f2_direction_positive"] = f2_positive
            extra_pass = row["separation_pass"] and f2_positive
            row["pass"] = extra_pass
            family_rows[extra] = row
            family_pass = family_pass and extra_pass
        descriptive = {}
        extras = list(V3_CLUSTER_EXTRA_CONSONANTS)
        for i in range(len(extras)):
            for j in range(i + 1, len(extras)):
                left, right = extras[i], extras[j]
                descriptive[f"{left}-{right}"] = (
                    _cosine(vectors[left], vectors[right])
                    if left in vectors and right in vectors
                    else None
                )
        gates["families"][key] = {
            "extras": family_rows,
            "descriptive_cross_extra_cosines": descriptive,
            "pass": family_pass,
        }
        overall = overall and family_pass
        if family_pass:
            for extra in V3_CLUSTER_EXTRA_CONSONANTS:
                endpoints_by_extra.setdefault(extra, {})[key] = {
                    "ae_bark": family_rows[extra]["ae"]["endpoint_bark"],
                    "eh_bark": family_rows[extra]["eh"]["endpoint_bark"],
                }
    baseline_ok = all(
        row["baseline_double_decode_bit_identical"] for row in conditions.values()
    )
    overall = overall and baseline_ok
    analysis = {
        "schema_version": 1,
        "run_id": V3_RUN_ID,
        "protocol_sha256": protocol["protocol_sha256"],
        "api_calls_made": 0,
        "baseline_double_decode_bit_identical_all": baseline_ok,
        "conditions": conditions,
        "gates": gates,
        "endpoints_by_extra": endpoints_by_extra if overall else {},
        "classification": (
            "cluster_anchor_calibration_v3_pass"
            if overall
            else "cluster_anchor_calibration_v3_fail"
        ),
        "production_enabled": False,
    }
    analysis["analysis_sha256"] = sha256_json(
        {key: value for key, value in analysis.items() if key != "analysis_sha256"}
    )
    atomic_write_json(analysis_path, analysis)
    return analysis


V4_CALIBRATION_VERSION = "kokoro-cluster-anchor-calibration-v4"
V4_RUN_ID = "20260718-kokoro-cluster-anchor-calibration-v4"


def run_dir_v4() -> Path:
    return Paths().artifacts / "typed-engine" / V4_RUN_ID


def _verified_v3_failure() -> dict[str, Any]:
    protocol_path = run_dir_v3() / "protocol.json"
    analysis_path = run_dir_v3() / "analysis.json"
    protocol = _load_json(protocol_path)
    analysis = _load_json(analysis_path)
    _verify_internal_hash(protocol, "protocol_sha256", label="v3 protocol")
    _verify_internal_hash(analysis, "analysis_sha256", label="v3 analysis")
    if analysis["classification"] != "cluster_anchor_calibration_v3_fail":
        raise RuntimeError("v3 parent must remain the frozen failed calibration")
    return {
        "run_id": V3_RUN_ID,
        "protocol_sha256": protocol["protocol_sha256"],
        "protocol_file_sha256": sha256_file(protocol_path),
        "analysis_sha256": analysis["analysis_sha256"],
        "analysis_file_sha256": sha256_file(analysis_path),
        "classification": analysis["classification"],
        "note": (
            "The v3 aggregate stays failed under its own frozen gates. Its "
            "passing /st/ and /sp/ cells carry no selection authority here; "
            "every v4 endpoint is newly rendered under the narrowed pool."
        ),
    }


def anchor_conditions_v4() -> list[dict[str, Any]]:
    """Build the active v4 shell manifest without mutating frozen evidence."""

    from .kokoro_cluster_shell import (
        CLUSTER_EXTRA_CONSONANTS,
        CLUSTER_LENS_SHELL,
        CLUSTER_NEUTRAL_SHELL,
    )

    return _anchor_conditions(
        neutral_shell=CLUSTER_NEUTRAL_SHELL,
        lens_shell=CLUSTER_LENS_SHELL,
        extras=CLUSTER_EXTRA_CONSONANTS,
        real_word_anchors=(),
    )


def protocol_record_v4() -> dict[str, Any]:
    # Inherit the frozen v3 artifact rather than the live v3 builder, the
    # same file binding v3 used against v2: the evidence dependency is
    # explicit and later code evolution cannot change this parent.
    from .kokoro_cluster_shell import (
        CLUSTER_EXTRA_CONSONANTS,
        CLUSTER_LENS_SHELL,
        CLUSTER_NEUTRAL_SHELL,
        CLUSTER_SHELL_VERSION,
    )

    base_path = run_dir_v3() / "protocol.json"
    base = _load_json(base_path)
    _verify_internal_hash(base, "protocol_sha256", label="v3 protocol")
    payload: dict[str, Any] = {
        key: value
        for key, value in base.items()
        if key
        not in (
            "protocol_sha256",
            "run_id",
            "version",
            "v2_parent",
            "conditions",
            "cluster_shell_version",
            "purpose",
            "predetermined_outcomes",
            "isolated_gate_probe",
        )
    }
    payload["run_id"] = V4_RUN_ID
    payload["version"] = V4_CALIBRATION_VERSION
    payload["purpose"] = (
        "Context-matched /ae/ and /eh/ endpoint anchors for the v4 cluster "
        "shell, whose extra pool is narrowed to the alveolar and bilabial "
        "stops. The v2 measurement design is unchanged; every endpoint is "
        "newly rendered."
    )
    payload["cluster_shell_version"] = CLUSTER_SHELL_VERSION
    payload["conditions"] = anchor_conditions_v4()
    payload["v3_parent"] = _verified_v3_failure()
    payload["pool_change"] = {
        "from_extras": list(V3_CLUSTER_EXTRA_CONSONANTS),
        "to_extras": list(CLUSTER_EXTRA_CONSONANTS),
        "rationale": (
            "The frozen v3 calibration failed only its /sk/ cell, whose "
            "eh-minus-ae F2 component is stably negative at every ceiling — "
            "a coda-conditioned realization, not estimator noise. Extra-pool "
            "membership is a free planner design choice and anchors only "
            "need to cover what the planner can emit, so v4 excludes the "
            "velar extra by design instead of shipping an anchor-less "
            "context. Nothing in v3 is rescued or reclassified."
        ),
    }
    payload["isolated_gate_probe"] = _shell_gate_probe(
        neutral_shell=CLUSTER_NEUTRAL_SHELL,
        lens_shell=CLUSTER_LENS_SHELL,
        extras=CLUSTER_EXTRA_CONSONANTS,
    )
    payload["predetermined_outcomes"] = {
        "pass": (
            "Freeze the v4 per-extra endpoints for one bounded cluster "
            "confirmation; no product enablement follows from calibration "
            "alone."
        ),
        "fail": (
            "Freeze the failure and do not run the cluster confirmation or "
            "enable the cluster planner."
        ),
    }
    payload["scope"] = dict(payload["scope"])
    payload["scope"]["logical_decodes"] = len(payload["conditions"]) * (
        len(payload["seeds"]["training"]) + 2
    )
    return {**payload, "protocol_sha256": sha256_json(payload)}


def prepare_v4() -> dict[str, Any]:
    protocol = protocol_record_v4()
    destination = run_dir_v4() / "protocol.json"
    if destination.is_file():
        existing = _load_json(destination)
        if existing != protocol:
            raise RuntimeError("existing v4 anchor protocol differs from freeze")
        return existing
    destination.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(destination, protocol)
    return protocol


def run_v4() -> dict[str, Any]:
    protocol = prepare_v4()
    analysis_path = run_dir_v4() / "analysis.json"
    if analysis_path.exists():
        raise RuntimeError("v4 cluster anchor calibration has already been analyzed")
    from .kokoro_synthesis import KokoroSynthesisRuntime

    runtime = KokoroSynthesisRuntime.load()
    audio_dir = run_dir_v4() / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    training = tuple(protocol["seeds"]["training"])
    seeds = training + (protocol["seeds"]["baseline"],)
    # The runner is bound to its frozen protocol: the extras it evaluates are
    # exactly the distinct extras of the frozen conditions, in manifest order.
    extras: list[str] = []
    for condition in protocol["conditions"]:
        if condition["extra_consonant"] not in extras:
            extras.append(condition["extra_consonant"])
    conditions: dict[str, Any] = {}
    for condition in protocol["conditions"]:
        cid = condition["condition_id"]
        seeded = render_seeded_natural_conditions(
            runtime,
            phonemes=condition["phonemes"],
            reference_phonemes=condition["phonemes"],
            seeds=seeds,
        )
        repeat = render_seeded_natural_conditions(
            runtime,
            phonemes=condition["phonemes"],
            reference_phonemes=condition["phonemes"],
            seeds=(protocol["seeds"]["baseline"],),
        )
        first = seeded.audio_by_seed[protocol["seeds"]["baseline"]]
        second = repeat.audio_by_seed[protocol["seeds"]["baseline"]]
        baseline_identical = bool(
            first.shape == second.shape and np.array_equal(first, second)
        )
        interval = _target_interval(
            runtime.model,
            condition["phonemes"],
            seeded.predicted_durations,
            int(first.size),
        )
        per_seed: dict[str, Any] = {}
        for seed in seeds:
            wav_path = audio_dir / f"{cid}-seed-{seed}.wav"
            record = _write_wav_once(wav_path, seeded.audio_by_seed[seed], run_dir_v4())
            record["families"] = {
                str(ceiling): _seed_point(wav_path, interval, ceiling)
                for ceiling in CEILINGS_HZ
            }
            per_seed[str(seed)] = record
        conditions[cid] = {
            "condition": condition,
            "measurement_interval": interval,
            "baseline_double_decode_bit_identical": baseline_identical,
            "seeds": per_seed,
        }
    gates, endpoints_by_extra, overall = _analyze_extras_families(
        conditions, extras=extras, training=training
    )
    baseline_ok = all(
        row["baseline_double_decode_bit_identical"] for row in conditions.values()
    )
    overall = overall and baseline_ok
    analysis = {
        "schema_version": 1,
        "run_id": V4_RUN_ID,
        "protocol_sha256": protocol["protocol_sha256"],
        "api_calls_made": 0,
        "baseline_double_decode_bit_identical_all": baseline_ok,
        "conditions": conditions,
        "gates": gates,
        "endpoints_by_extra": endpoints_by_extra if overall else {},
        "classification": (
            "cluster_anchor_calibration_v4_pass"
            if overall
            else "cluster_anchor_calibration_v4_fail"
        ),
        "production_enabled": False,
    }
    analysis["analysis_sha256"] = sha256_json(
        {key: value for key, value in analysis.items() if key != "analysis_sha256"}
    )
    atomic_write_json(analysis_path, analysis)
    return analysis
