from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import math
import os
import secrets
import shutil
import subprocess
import tempfile
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .config import stable_json
from .kokoro_specs import VOICE_SPECS_BY_ID, resolve_pinned_file
from .kokoro_synthesis import (
    CONFIG_FILE,
    KOKORO_VERSION,
    MODEL_FILE,
    MODEL_HASHES,
    MODEL_REPO,
    RNG_SEED,
    SAMPLE_RATE_HZ,
    SPEED,
    _f0_noise,
    _INFERENCE_LOCK,
    _input_ids,
    _predicted_alignment,
    _text_features,
    pcm16_bytes,
    pcm_sha256,
)
from .ptbr_listener_lens_feasibility_protocol import (
    ANCHOR_DISTANCE_MULTIPLIER,
    CEILINGS_HZ,
    LOCALIZATION_MINIMUM,
    LOCALIZATION_PADDING_S,
    MAX_CLIPPED_FRACTION,
    MAX_DECODER_CALLS,
    MEASUREMENT_SCRIPT,
    MIDDLE_FRACTION,
    MIN_DIRECTION_COSINE,
    MIN_ANCHOR_DISTANCE_BARK,
    MIN_MAGNITUDE_BARK,
    MIN_VALID_FRAME_FRACTION,
    MIN_VALID_FRAMES,
    PRAAT,
    REQUIRED_REVIEW_FIELDS,
    RESPONSE_FILENAME,
    RESPONSE_SCHEMA_PATH,
    RUN_ID,
    TECHNICAL_PROBE_VOICE_ID,
    ReciprocalFeasibilityProtocolError,
    ReciprocalProfilePhonePlan,
    _profile_plan,
    render_manifest,
    run_dir,
    verify_frozen_protocol,
)
from .util import atomic_write_json, atomic_write_text, sha256_file


RENDER_ATTEMPT_FILE = "render-attempt.json"
RENDER_RECORDS_FILE = "render-records.json"
ANALYSIS_FILE = "analysis.json"
REVIEW_FILE = "review.html"
REVIEW_MANIFEST_FILE = "review-manifest.json"
BLIND_KEY_FILE = "blind-key.json"
REVIEW_FAILURE_FILE = "review-generation-failure.json"
PUBLIC_REVIEW_ROOT = Path("public") / "review"
PRIVATE_REVIEW_ROOT = Path("private")


@dataclass(frozen=True)
class _RenderedState:
    audio: np.ndarray
    durations: tuple[int, ...]
    alignment_sha256: str
    f0_sha256: str
    noise_sha256: str
    target_intervals: tuple[dict[str, Any], ...]


def verify_repo_bound_inputs_at_head(
    bindings: dict[str, str],
    *,
    repository: Path | None = None,
) -> dict[str, Any]:
    repository = (repository or Path(__file__).resolve().parents[2]).resolve()
    head_result = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
    )
    if head_result.returncode != 0:
        raise ReciprocalFeasibilityProtocolError(
            "repo-bound inputs require a repository HEAD and tracked files"
        )
    head = head_result.stdout.strip()
    if not bindings:
        raise ReciprocalFeasibilityProtocolError("repo-bound input list is empty")
    verified: list[dict[str, Any]] = []
    for relative_text, expected_sha256 in sorted(bindings.items()):
        relative = Path(relative_text)
        if relative.is_absolute() or ".." in relative.parts:
            raise ReciprocalFeasibilityProtocolError(
                f"invalid repo-bound input path: {relative_text}"
            )
        path = (repository / relative).resolve()
        try:
            path.relative_to(repository)
        except ValueError as exc:
            raise ReciprocalFeasibilityProtocolError(
                f"repo-bound input is outside repository: {relative_text}"
            ) from exc
        if not path.is_file() or sha256_file(path) != expected_sha256:
            raise ReciprocalFeasibilityProtocolError(
                f"repo-bound input hash drifted: {relative_text}"
            )
        tracked = subprocess.run(
            [
                "git",
                "-C",
                str(repository),
                "ls-files",
                "--error-unmatch",
                "--",
                relative.as_posix(),
            ],
            capture_output=True,
            text=True,
        )
        if tracked.returncode != 0:
            raise ReciprocalFeasibilityProtocolError(
                f"repo-bound input must be tracked at HEAD: {relative_text}"
            )
        unstaged = subprocess.run(
            [
                "git",
                "-C",
                str(repository),
                "diff",
                "--quiet",
                "--",
                relative.as_posix(),
            ]
        )
        if unstaged.returncode != 0:
            raise ReciprocalFeasibilityProtocolError(
                f"repo-bound input has unstaged drift: {relative_text}"
            )
        staged = subprocess.run(
            [
                "git",
                "-C",
                str(repository),
                "diff",
                "--cached",
                "--quiet",
                "HEAD",
                "--",
                relative.as_posix(),
            ]
        )
        if staged.returncode != 0:
            raise ReciprocalFeasibilityProtocolError(
                f"repo-bound input has staged drift: {relative_text}"
            )
        committed = subprocess.run(
            [
                "git",
                "-C",
                str(repository),
                "show",
                f"{head}:{relative.as_posix()}",
            ],
            check=False,
            capture_output=True,
        )
        if committed.returncode != 0 or committed.stdout != path.read_bytes():
            raise ReciprocalFeasibilityProtocolError(
                f"repo-bound input is not byte-identical to HEAD: {relative_text}"
            )
        verified.append({"path": relative.as_posix(), "sha256": expected_sha256})
    final_head = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if final_head != head:
        raise ReciprocalFeasibilityProtocolError(
            "repository HEAD changed during bound-input verification"
        )
    return {
        "repository_head": head,
        "all_inputs_tracked_clean_and_byte_identical": True,
        "verified_input_count": len(verified),
        "inputs": verified,
    }


def verify_protocol_committed_at_head(
    protocol_path: Path | None = None,
    *,
    repository: Path | None = None,
) -> dict[str, Any]:
    repository = (repository or Path(__file__).resolve().parents[2]).resolve()
    protocol_path = (protocol_path or (run_dir() / "protocol.json")).resolve()
    try:
        relative = protocol_path.relative_to(repository).as_posix()
    except ValueError as exc:
        raise ReciprocalFeasibilityProtocolError(
            "protocol path is outside the repository"
        ) from exc
    receipt = verify_repo_bound_inputs_at_head(
        {relative: sha256_file(protocol_path)}, repository=repository
    )
    return {
        **receipt,
        "protocol_path": relative,
        "tracked_at_head": True,
        "clean_at_head": True,
    }


def _tensor_sha256(value: Any) -> str:
    array = value.detach().cpu().contiguous().numpy()
    header = stable_json(
        {"dtype": str(array.dtype), "shape": list(array.shape)}
    ).encode("utf-8")
    return hashlib.sha256(header + b"\0" + array.tobytes()).hexdigest()


def _reference_style(voice_pack: Any, phonemes: str) -> Any:
    style_index = len(phonemes) - 1
    if not 0 <= style_index < len(voice_pack):
        raise ReciprocalFeasibilityProtocolError(
            "profile phone length has no technical-probe voice style"
        )
    style = voice_pack[style_index]
    if style.ndim == 1:
        style = style.unsqueeze(0)
    if style.ndim != 2 or style.shape[-1] < 256:
        raise ReciprocalFeasibilityProtocolError(
            "technical-probe voice style has an unexpected shape"
        )
    return style


def _decode(
    model: Any,
    *,
    state: Any,
    alignment: Any,
    f0: Any,
    noise: Any,
    ref_s: Any,
    torch: Any,
) -> np.ndarray:
    torch.manual_seed(RNG_SEED)
    audio = model.decoder(state @ alignment, f0, noise, ref_s[:, :128])
    return audio.squeeze().detach().cpu().numpy()


def _sample_interval(
    columns: Sequence[int],
    durations: Sequence[int],
    samples_per_frame: int,
) -> dict[str, Any]:
    selected = tuple(sorted(set(int(value) for value in columns)))
    if not selected or selected != tuple(range(selected[0], selected[-1] + 1)):
        raise ReciprocalFeasibilityProtocolError(
            "target measurement columns must be contiguous"
        )
    start_sample = sum(durations[: selected[0]]) * samples_per_frame
    end_sample = sum(durations[: selected[-1] + 1]) * samples_per_frame
    if end_sample <= start_sample:
        raise ReciprocalFeasibilityProtocolError(
            "target measurement interval has nonpositive duration"
        )
    return {
        "columns": list(selected),
        "start_sample": start_sample,
        "end_sample_exclusive": end_sample,
        "start_s": start_sample / SAMPLE_RATE_HZ,
        "end_s": end_sample / SAMPLE_RATE_HZ,
    }


def _alignment_intervals(
    plan: ReciprocalProfilePhonePlan,
    *,
    durations: Sequence[int],
    sample_count: int,
) -> tuple[dict[str, Any], ...]:
    expected_duration_count = plan.equal_model_token_count + 2
    if len(durations) != expected_duration_count:
        raise ReciprocalFeasibilityProtocolError(
            "duration count does not match profile model-token plan"
        )
    total_frames = sum(int(value) for value in durations)
    if total_frames <= 0 or sample_count <= 0 or sample_count % total_frames:
        raise ReciprocalFeasibilityProtocolError(
            "decoded samples do not map to integer alignment frames"
        )
    samples_per_frame = sample_count // total_frames
    intervals: list[dict[str, Any]] = []
    for occurrence in plan.target_occurrences:
        target_interval = _sample_interval(
            (occurrence.model_column,),
            durations,
            samples_per_frame,
        )
        descriptive_interval = _sample_interval(
            (occurrence.stress_model_column, occurrence.model_column),
            durations,
            samples_per_frame,
        )
        intervals.append(
            {
                "occurrence_index": occurrence.occurrence_index,
                "source_word_index": occurrence.source_word_index,
                "stress_model_column": occurrence.stress_model_column,
                "target_model_column": occurrence.model_column,
                "samples_per_alignment_frame": samples_per_frame,
                "total_alignment_frames": total_frames,
                "target_interval": target_interval,
                "primary_measurement_interval": target_interval,
                "stress_plus_target_descriptive_interval": descriptive_interval,
            }
        )
    return tuple(intervals)


def _ordinary_render(
    *,
    model: Any,
    voice_pack: Any,
    phonemes: str,
    profile_plan: ReciprocalProfilePhonePlan,
    torch: Any,
) -> _RenderedState:
    ref_s = _reference_style(voice_pack, phonemes)
    features = _text_features(model, _input_ids(model, phonemes, torch), ref_s, torch)
    durations, alignment = _predicted_alignment(model, features, SPEED, torch)
    f0, noise = _f0_noise(model, features, alignment, torch)
    audio = _decode(
        model,
        state=features["t_en"],
        alignment=alignment,
        f0=f0,
        noise=noise,
        ref_s=ref_s,
        torch=torch,
    )
    duration_values = tuple(int(value) for value in durations.cpu().tolist())
    return _RenderedState(
        audio=audio,
        durations=duration_values,
        alignment_sha256=_tensor_sha256(alignment),
        f0_sha256=_tensor_sha256(f0),
        noise_sha256=_tensor_sha256(noise),
        target_intervals=_alignment_intervals(
            profile_plan,
            durations=duration_values,
            sample_count=len(audio),
        ),
    )


def _controlled_renders(
    *,
    model: Any,
    voice_pack: Any,
    profile_plan: ReciprocalProfilePhonePlan,
    torch: Any,
) -> tuple[_RenderedState, _RenderedState, _RenderedState, tuple[int, ...]]:
    ref_s = _reference_style(voice_pack, profile_plan.source_alignment_phonemes)
    source_features = _text_features(
        model,
        _input_ids(model, profile_plan.source_alignment_phonemes, torch),
        ref_s,
        torch,
    )
    neutral_features = _text_features(
        model,
        _input_ids(model, profile_plan.neutral_phonemes, torch),
        ref_s,
        torch,
    )
    lens_features = _text_features(
        model,
        _input_ids(model, profile_plan.lens_phonemes, torch),
        ref_s,
        torch,
    )
    durations, alignment = _predicted_alignment(model, source_features, SPEED, torch)
    f0, noise = _f0_noise(model, neutral_features, alignment, torch)
    columns = tuple(
        occurrence.model_column for occurrence in profile_plan.target_occurrences
    )
    neutral_state = neutral_features["t_en"]
    lens_state = neutral_state.clone()
    lens_state[:, :, list(columns)] = lens_features["t_en"][:, :, list(columns)]
    neutral_audio = _decode(
        model,
        state=neutral_state,
        alignment=alignment,
        f0=f0,
        noise=noise,
        ref_s=ref_s,
        torch=torch,
    )
    identity_audio = _decode(
        model,
        state=neutral_state,
        alignment=alignment,
        f0=f0,
        noise=noise,
        ref_s=ref_s,
        torch=torch,
    )
    lens_audio = _decode(
        model,
        state=lens_state,
        alignment=alignment,
        f0=f0,
        noise=noise,
        ref_s=ref_s,
        torch=torch,
    )
    if not (neutral_audio.shape == identity_audio.shape == lens_audio.shape):
        raise ReciprocalFeasibilityProtocolError(
            "controlled triplet has unequal sample counts"
        )
    duration_values = tuple(int(value) for value in durations.cpu().tolist())
    intervals = _alignment_intervals(
        profile_plan,
        durations=duration_values,
        sample_count=len(neutral_audio),
    )
    common = {
        "durations": duration_values,
        "alignment_sha256": _tensor_sha256(alignment),
        "f0_sha256": _tensor_sha256(f0),
        "noise_sha256": _tensor_sha256(noise),
        "target_intervals": intervals,
    }
    return (
        _RenderedState(audio=neutral_audio, **common),
        _RenderedState(audio=identity_audio, **common),
        _RenderedState(audio=lens_audio, **common),
        columns,
    )


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


def _audio_record(audio: np.ndarray, path: Path, destination: Path) -> dict[str, Any]:
    values = np.asarray(audio).reshape(-1)
    finite = bool(values.size and np.isfinite(values).all())
    clipped_fraction = float(np.mean(np.abs(values) >= 1.0)) if finite else 1.0
    pcm = (
        np.frombuffer(pcm16_bytes(values), dtype="<i2")
        if finite
        else np.asarray([], dtype="<i2")
    )
    pcm_full_scale_fraction = (
        float(np.mean(np.abs(pcm.astype(np.int32)) >= 32767)) if pcm.size else 1.0
    )
    return {
        "relative_path": str(path.relative_to(destination)),
        "wav_sha256": sha256_file(path),
        "pcm_sha256": pcm_sha256(values) if finite else None,
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "sample_count": int(values.size),
        "finite": finite,
        "clipped_fraction": clipped_fraction,
        "clipping_pass": bool(clipped_fraction < MAX_CLIPPED_FRACTION),
        "pcm_full_scale_fraction": pcm_full_scale_fraction,
        "pcm_full_scale_pass": bool(pcm_full_scale_fraction < MAX_CLIPPED_FRACTION),
    }


def _render_record(
    *,
    manifest_row: dict[str, Any],
    rendered: _RenderedState,
    path: Path,
    destination: Path,
    profile_plan: ReciprocalProfilePhonePlan,
    phone_role: str,
    replaced_columns: Sequence[int],
) -> dict[str, Any]:
    return {
        **manifest_row,
        "profile_plan_sha256": profile_plan.plan_sha256,
        "phone_role": phone_role,
        "predicted_durations": list(rendered.durations),
        "alignment_sha256": rendered.alignment_sha256,
        "f0_sha256": rendered.f0_sha256,
        "noise_sha256": rendered.noise_sha256,
        "target_intervals": list(rendered.target_intervals),
        "replaced_columns": list(replaced_columns),
        "audio": _audio_record(rendered.audio, path, destination),
    }


def _load_renderer() -> tuple[Any, Any, Any]:
    if importlib.metadata.version("kokoro") != KOKORO_VERSION:
        raise ReciprocalFeasibilityProtocolError(f"Kokoro {KOKORO_VERSION} is required")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    import torch
    from kokoro import KModel

    files = {
        filename: resolve_pinned_file(filename)
        for filename in (CONFIG_FILE, MODEL_FILE)
    }
    if {filename: sha256_file(path) for filename, path in files.items()} != {
        filename: MODEL_HASHES[filename] for filename in files
    }:
        raise ReciprocalFeasibilityProtocolError("Kokoro model assets drifted")
    voice = VOICE_SPECS_BY_ID[TECHNICAL_PROBE_VOICE_ID]
    voice_path = resolve_pinned_file(voice.filename)
    if sha256_file(voice_path) != voice.sha256:
        raise ReciprocalFeasibilityProtocolError("technical-probe voice drifted")
    with _INFERENCE_LOCK:
        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
        torch.backends.mkldnn.enabled = False
        torch.backends.nnpack.set_flags(False)
        torch.use_deterministic_algorithms(True)
        model = (
            KModel(
                repo_id=MODEL_REPO,
                config=str(files[CONFIG_FILE]),
                model=str(files[MODEL_FILE]),
            )
            .to("cpu")
            .eval()
        )
        voice_pack = torch.load(voice_path, map_location="cpu", weights_only=True)
    return model, voice_pack, torch


def _stage_artifact_paths(destination: Path) -> tuple[Path, ...]:
    return (
        destination / RENDER_ATTEMPT_FILE,
        destination / RENDER_RECORDS_FILE,
        destination / "audio",
        destination / ANALYSIS_FILE,
        destination / PUBLIC_REVIEW_ROOT,
        destination / PRIVATE_REVIEW_ROOT,
        destination / REVIEW_FAILURE_FILE,
        destination / "review-generation.partial",
        destination / REVIEW_FILE,
        destination / REVIEW_MANIFEST_FILE,
        destination / BLIND_KEY_FILE,
    )


def _reject_existing(paths: Sequence[Path], *, stage: str) -> None:
    present = [str(path) for path in paths if path.exists()]
    if present:
        raise ReciprocalFeasibilityProtocolError(
            f"{stage} refuses stale or pre-eligibility artifacts: " + ", ".join(present)
        )


def _verify_current_repo_inputs(protocol: dict[str, Any]) -> dict[str, Any]:
    protocol_path = run_dir() / "protocol.json"
    repository = Path(__file__).resolve().parents[2]
    bindings = dict(protocol.get("bindings", {}).get("repo_bound_inputs", {}))
    try:
        protocol_relative = protocol_path.resolve().relative_to(repository.resolve())
    except ValueError as exc:
        raise ReciprocalFeasibilityProtocolError(
            "frozen protocol is outside the repository"
        ) from exc
    bindings[protocol_relative.as_posix()] = sha256_file(protocol_path)
    return verify_repo_bound_inputs_at_head(bindings, repository=repository)


def render() -> dict[str, Any]:
    protocol = verify_frozen_protocol()
    destination = run_dir()
    _reject_existing(_stage_artifact_paths(destination), stage="render")
    committed_inputs = _verify_current_repo_inputs(protocol)
    attempt_path = destination / RENDER_ATTEMPT_FILE
    records_path = destination / RENDER_RECORDS_FILE
    audio_dir = destination / "audio"
    manifest = render_manifest()
    if len(manifest) != MAX_DECODER_CALLS:
        raise ReciprocalFeasibilityProtocolError("decoder manifest bound drifted")
    atomic_write_json(
        attempt_path,
        {
            "schema_version": 1,
            "run_id": RUN_ID,
            "protocol_sha256": protocol["protocol_sha256"],
            "maximum_decoder_calls": MAX_DECODER_CALLS,
            "attempts_per_slot": 1,
            "retries_allowed": 0,
            "committed_inputs": committed_inputs,
        },
    )
    audio_dir.mkdir(parents=False, exist_ok=False)

    _, profile_plan = _profile_plan()
    if (
        profile_plan.plan_sha256
        != protocol["fixture"]["profile_phone_plan"]["plan_sha256"]
    ):
        raise ReciprocalFeasibilityProtocolError("profile plan drifted after freeze")
    model, voice_pack, torch = _load_renderer()
    started = time.perf_counter()
    records: list[dict[str, Any]] = []
    with _INFERENCE_LOCK, torch.no_grad():
        anchor_neutral = _ordinary_render(
            model=model,
            voice_pack=voice_pack,
            phonemes=profile_plan.neutral_phonemes,
            profile_plan=profile_plan,
            torch=torch,
        )
        anchor_lens = _ordinary_render(
            model=model,
            voice_pack=voice_pack,
            phonemes=profile_plan.lens_phonemes,
            profile_plan=profile_plan,
            torch=torch,
        )
        controlled_neutral, controlled_identity, controlled_lens, columns = (
            _controlled_renders(
                model=model,
                voice_pack=voice_pack,
                profile_plan=profile_plan,
                torch=torch,
            )
        )
    outputs = (
        (anchor_neutral, "neutral", ()),
        (anchor_lens, "lens", ()),
        (controlled_neutral, "neutral", ()),
        (controlled_identity, "identity", ()),
        (controlled_lens, "lens", columns),
    )
    for row, (rendered, phone_role, replaced) in zip(manifest, outputs, strict=True):
        path = audio_dir / f"{row['order']:02d}__{row['slot_id']}.wav"
        _write_wav(path, rendered.audio)
        records.append(
            _render_record(
                manifest_row=row,
                rendered=rendered,
                path=path,
                destination=destination,
                profile_plan=profile_plan,
                phone_role=phone_role,
                replaced_columns=replaced,
            )
        )

    by_slot = {record["slot_id"]: record for record in records}
    controlled = [
        by_slot[slot]
        for slot in ("controlled-neutral", "controlled-identity", "controlled-lens")
    ]
    controlled_counts = {record["audio"]["sample_count"] for record in controlled}
    common_latents = all(
        stable_json(record[field]) == stable_json(controlled[0][field])
        for record in controlled[1:]
        for field in (
            "predicted_durations",
            "alignment_sha256",
            "f0_sha256",
            "noise_sha256",
            "target_intervals",
        )
    )
    identity_equal = (
        by_slot["controlled-neutral"]["audio"]["pcm_sha256"]
        == by_slot["controlled-identity"]["audio"]["pcm_sha256"]
    )
    declared_columns = [
        occurrence.model_column for occurrence in profile_plan.target_occurrences
    ]
    runtime_pass = bool(
        len(records) == MAX_DECODER_CALLS
        and len(controlled_counts) == 1
        and next(iter(controlled_counts), 0) > 0
        and common_latents
        and identity_equal
        and by_slot["controlled-lens"]["replaced_columns"] == declared_columns
        and all(
            record["audio"]["finite"]
            and record["audio"]["clipping_pass"]
            and record["audio"]["pcm_full_scale_pass"]
            for record in records
        )
    )
    payload = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "status": "single_bounded_render_complete",
        "protocol_sha256": protocol["protocol_sha256"],
        "profile_plan_sha256": profile_plan.plan_sha256,
        "decoder_call_count": len(records),
        "decoder_call_limit": MAX_DECODER_CALLS,
        "retries_made": 0,
        "variants_rendered": 0,
        "selection_performed": False,
        "render_seconds": time.perf_counter() - started,
        "records": records,
        "runtime_integrity": {
            "controlled_equal_nonzero_sample_counts": len(controlled_counts) == 1
            and next(iter(controlled_counts), 0) > 0,
            "controlled_common_alignment_f0_noise_exact": common_latents,
            "controlled_neutral_identity_bit_identical": identity_equal,
            "replaced_columns_exact": by_slot["controlled-lens"]["replaced_columns"]
            == declared_columns,
            "all_audio_finite_and_below_clipping_limit": all(
                record["audio"]["finite"]
                and record["audio"]["clipping_pass"]
                and record["audio"]["pcm_full_scale_pass"]
                for record in records
            ),
            "pass": runtime_pass,
        },
        "api_calls": 0,
        "paid_calls": 0,
    }
    atomic_write_json(records_path, payload)
    return payload


def _number(value: str) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _bark(hz: float) -> float:
    if not math.isfinite(hz) or hz <= 0:
        raise ReciprocalFeasibilityProtocolError(
            "Bark conversion requires a finite positive frequency"
        )
    return 26.81 / (1 + 1960 / hz) - 0.53


def _measure_rows(
    rows: Sequence[dict[str, str]],
    interval: dict[str, Any],
    ceiling_hz: int,
) -> dict[str, Any]:
    duration = float(interval["end_s"]) - float(interval["start_s"])
    if not math.isfinite(duration) or duration <= 0:
        raise ReciprocalFeasibilityProtocolError(
            "primary target-column interval has nonpositive duration"
        )
    margin = duration * (1 - MIDDLE_FRACTION) / 2
    middle_start = float(interval["start_s"]) + margin
    middle_end = float(interval["end_s"]) - margin
    queried_count = 0
    finite_count = 0
    positive_ordered_count = 0
    pairs: list[tuple[float, float]] = []
    for row in rows:
        time_s = _number(row.get("time_s", ""))
        if time_s is None or not middle_start <= time_s <= middle_end:
            continue
        queried_count += 1
        f1_hz = _number(row.get("f1_hz", ""))
        f2_hz = _number(row.get("f2_hz", ""))
        if f1_hz is None or f2_hz is None:
            continue
        finite_count += 1
        if f1_hz <= 0 or f2_hz <= 0 or f2_hz <= f1_hz:
            continue
        positive_ordered_count += 1
        pairs.append((f1_hz, f2_hz))
    retained_count = len(pairs)
    valid_fraction = retained_count / queried_count if queried_count else 0.0
    if (
        queried_count < MIN_VALID_FRAMES
        or retained_count < MIN_VALID_FRAMES
        or valid_fraction < MIN_VALID_FRAME_FRACTION
    ):
        raise ReciprocalFeasibilityProtocolError(
            f"frame retention failed at {ceiling_hz} Hz: "
            f"retained={retained_count}/queried={queried_count}; "
            f"finite={finite_count}; positive_ordered={positive_ordered_count}"
        )
    f1_hz, f2_hz = np.median(np.asarray(pairs), axis=0)
    plausible = bool(
        180 <= f1_hz <= 1200 and 600 <= f2_hz <= 3500 and f2_hz - f1_hz >= 250
    )
    return {
        "ceiling_hz": ceiling_hz,
        "middle_fraction": MIDDLE_FRACTION,
        "queried_frame_count": queried_count,
        "finite_f1_f2_frame_count": finite_count,
        "positive_ordered_f1_f2_frame_count": positive_ordered_count,
        "retained_f1_f2_frame_count": retained_count,
        "retained_f1_f2_fraction": valid_fraction,
        "f1_hz": float(f1_hz),
        "f2_hz": float(f2_hz),
        "f1_bark": _bark(float(f1_hz)),
        "f2_bark": _bark(float(f2_hz)),
        "plausibility_pass": plausible,
    }


def _measure(path: Path, interval: dict[str, Any], ceiling_hz: int) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="ptbr-reciprocal-burg-") as temp:
        output = Path(temp) / "frames.tsv"
        subprocess.run(
            [
                str(PRAAT),
                "--run",
                str(MEASUREMENT_SCRIPT),
                str(path),
                str(output),
                f"{float(interval['start_s']):.9f}",
                f"{float(interval['end_s']):.9f}",
                str(ceiling_hz),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        with output.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
    return _measure_rows(rows, interval, ceiling_hz)


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator else -1.0


def _meets_inclusive_threshold(value: float, threshold: float) -> bool:
    return bool(
        value >= threshold or math.isclose(value, threshold, rel_tol=0.0, abs_tol=1e-12)
    )


def classify_local_acoustics(
    *,
    anchor_neutral: dict[str, dict[str, Any]],
    anchor_lens: dict[str, dict[str, Any]],
    controlled_neutral: dict[str, dict[str, Any]],
    controlled_lens: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    families: dict[str, Any] = {}
    for ceiling_hz in CEILINGS_HZ:
        key = str(ceiling_hz)
        an_measure = anchor_neutral[key]
        al_measure = anchor_lens[key]
        cn_measure = controlled_neutral[key]
        cl_measure = controlled_lens[key]
        anchor_source = np.asarray(
            [an_measure["f1_bark"], an_measure["f2_bark"]], dtype=float
        )
        anchor_target = np.asarray(
            [al_measure["f1_bark"], al_measure["f2_bark"]], dtype=float
        )
        neutral = np.asarray(
            [cn_measure["f1_bark"], cn_measure["f2_bark"]], dtype=float
        )
        lens = np.asarray([cl_measure["f1_bark"], cl_measure["f2_bark"]], dtype=float)
        anchor_vector = anchor_target - anchor_source
        candidate_vector = lens - neutral
        anchor_distance = float(np.linalg.norm(anchor_vector))
        magnitude = float(np.linalg.norm(candidate_vector))
        threshold = max(
            MIN_MAGNITUDE_BARK, ANCHOR_DISTANCE_MULTIPLIER * anchor_distance
        )
        direction_cosine = _cosine(candidate_vector, anchor_vector)
        neutral_source_distance = float(np.linalg.norm(neutral - anchor_source))
        neutral_target_distance = float(np.linalg.norm(neutral - anchor_target))
        lens_source_distance = float(np.linalg.norm(lens - anchor_source))
        lens_target_distance = float(np.linalg.norm(lens - anchor_target))
        anchor_valid = bool(
            an_measure["plausibility_pass"]
            and al_measure["plausibility_pass"]
            and _meets_inclusive_threshold(anchor_distance, MIN_ANCHOR_DISTANCE_BARK)
        )
        neutral_category = neutral_source_distance < neutral_target_distance
        lens_category = lens_target_distance < lens_source_distance
        passed = bool(
            anchor_valid
            and cn_measure["plausibility_pass"]
            and cl_measure["plausibility_pass"]
            and neutral_category
            and lens_category
            and _meets_inclusive_threshold(direction_cosine, MIN_DIRECTION_COSINE)
            and _meets_inclusive_threshold(magnitude, threshold)
        )
        families[key] = {
            "anchor_neutral_bark": anchor_source.tolist(),
            "anchor_lens_bark": anchor_target.tolist(),
            "anchor_vector_bark": anchor_vector.tolist(),
            "anchor_distance_bark": anchor_distance,
            "minimum_anchor_distance_bark": MIN_ANCHOR_DISTANCE_BARK,
            "anchor_valid": anchor_valid,
            "controlled_neutral_bark": neutral.tolist(),
            "controlled_lens_bark": lens.tolist(),
            "controlled_vector_bark": candidate_vector.tolist(),
            "controlled_magnitude_bark": magnitude,
            "magnitude_threshold_bark": threshold,
            "numeric_threshold_interpretation": ("engineering_nonperceptual_criterion"),
            "direction_cosine": direction_cosine,
            "neutral_to_anchor_neutral_bark": neutral_source_distance,
            "neutral_to_anchor_lens_bark": neutral_target_distance,
            "lens_to_anchor_neutral_bark": lens_source_distance,
            "lens_to_anchor_lens_bark": lens_target_distance,
            "neutral_category_pass": neutral_category,
            "lens_category_pass": lens_category,
            "controlled_plausibility_pass": bool(
                cn_measure["plausibility_pass"] and cl_measure["plausibility_pass"]
            ),
            "pass": passed,
        }
    return {
        "families": families,
        "all_three_ceilings_pass": all(family["pass"] for family in families.values()),
    }


def _read_pcm(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as handle:
        if (
            handle.getnchannels() != 1
            or handle.getsampwidth() != 2
            or handle.getframerate() != SAMPLE_RATE_HZ
        ):
            raise ReciprocalFeasibilityProtocolError("rendered WAV format drifted")
        return np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2").copy()


def localization_report(
    neutral_pcm: np.ndarray,
    lens_pcm: np.ndarray,
    intervals: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    neutral = np.asarray(neutral_pcm, dtype=np.float64).reshape(-1)
    lens = np.asarray(lens_pcm, dtype=np.float64).reshape(-1)
    if neutral.shape != lens.shape or not neutral.size:
        raise ReciprocalFeasibilityProtocolError(
            "localization requires equal nonempty PCM arrays"
        )
    delta = lens - neutral
    squared = delta * delta
    total_energy = float(np.sum(squared))
    mask = np.zeros(neutral.size, dtype=bool)
    padding = int(round(LOCALIZATION_PADDING_S * SAMPLE_RATE_HZ))
    windows: list[dict[str, int]] = []
    for item in intervals:
        interval = item["target_interval"]
        start = max(0, int(interval["start_sample"]) - padding)
        end = min(neutral.size, int(interval["end_sample_exclusive"]) + padding)
        if end <= start:
            raise ReciprocalFeasibilityProtocolError(
                "localization target window has nonpositive duration"
            )
        mask[start:end] = True
        windows.append({"start_sample": start, "end_sample_exclusive": end})
    inside_energy = float(np.sum(squared[mask]))
    outside = delta[~mask]
    outside_rms = float(np.sqrt(np.mean(outside * outside))) if outside.size else 0.0
    fraction = inside_energy / total_energy if total_energy > 0 else 0.0
    return {
        "windows": windows,
        "total_squared_difference_energy": total_energy,
        "inside_squared_difference_energy": inside_energy,
        "inside_energy_fraction": fraction,
        "outside_rms_pcm": outside_rms,
        "zero_total_difference": total_energy == 0,
        "minimum_inside_fraction": LOCALIZATION_MINIMUM,
        "pass": bool(total_energy > 0 and fraction >= LOCALIZATION_MINIMUM),
    }


def _verify_render_evidence(
    payload: dict[str, Any], destination: Path, protocol: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    if payload.get("protocol_sha256") != protocol["protocol_sha256"]:
        raise ReciprocalFeasibilityProtocolError("render protocol binding drifted")
    records = payload.get("records", [])
    if len(records) != MAX_DECODER_CALLS:
        raise ReciprocalFeasibilityProtocolError("render record count drifted")
    if [
        {key: record[key] for key in ("order", "slot_id", "mode", "role")}
        for record in records
    ] != render_manifest():
        raise ReciprocalFeasibilityProtocolError("render manifest drifted")
    profile_record = protocol["fixture"]["profile_phone_plan"]
    expected_profile_sha256 = profile_record["plan_sha256"]
    if payload.get("profile_plan_sha256") != expected_profile_sha256:
        raise ReciprocalFeasibilityProtocolError("render profile-plan binding drifted")
    pcm_by_slot: dict[str, np.ndarray] = {}
    for record in records:
        if record.get("profile_plan_sha256") != expected_profile_sha256:
            raise ReciprocalFeasibilityProtocolError(
                f"record profile-plan binding drifted: {record['slot_id']}"
            )
        path = destination / record["audio"]["relative_path"]
        if not path.is_file() or sha256_file(path) != record["audio"]["wav_sha256"]:
            raise ReciprocalFeasibilityProtocolError(
                f"rendered WAV hash drifted: {record['slot_id']}"
            )
        pcm = _read_pcm(path)
        if (
            hashlib.sha256(pcm.astype("<i2", copy=False).tobytes()).hexdigest()
            != record["audio"]["pcm_sha256"]
        ):
            raise ReciprocalFeasibilityProtocolError(
                f"rendered PCM hash drifted: {record['slot_id']}"
            )
        if len(pcm) != record["audio"]["sample_count"]:
            raise ReciprocalFeasibilityProtocolError(
                f"rendered sample count drifted: {record['slot_id']}"
            )
        durations = tuple(int(value) for value in record["predicted_durations"])
        expected_duration_count = int(profile_record["equal_model_token_count"]) + 2
        total_frames = sum(durations)
        if (
            len(durations) != expected_duration_count
            or total_frames <= 0
            or len(pcm) % total_frames
        ):
            raise ReciprocalFeasibilityProtocolError(
                f"record alignment geometry drifted: {record['slot_id']}"
            )
        samples_per_frame = len(pcm) // total_frames
        expected_intervals: list[dict[str, Any]] = []
        for occurrence in profile_record["target_occurrences"]:
            target_interval = _sample_interval(
                (int(occurrence["model_column"]),),
                durations,
                samples_per_frame,
            )
            descriptive_interval = _sample_interval(
                (
                    int(occurrence["stress_model_column"]),
                    int(occurrence["model_column"]),
                ),
                durations,
                samples_per_frame,
            )
            expected_intervals.append(
                {
                    "occurrence_index": int(occurrence["occurrence_index"]),
                    "source_word_index": int(occurrence["source_word_index"]),
                    "stress_model_column": int(occurrence["stress_model_column"]),
                    "target_model_column": int(occurrence["model_column"]),
                    "samples_per_alignment_frame": samples_per_frame,
                    "total_alignment_frames": total_frames,
                    "target_interval": target_interval,
                    "primary_measurement_interval": target_interval,
                    "stress_plus_target_descriptive_interval": descriptive_interval,
                }
            )
        if stable_json(record["target_intervals"]) != stable_json(expected_intervals):
            raise ReciprocalFeasibilityProtocolError(
                f"record target intervals drifted: {record['slot_id']}"
            )
        if not all(
            isinstance(record.get(key), str) and len(record[key]) == 64
            for key in ("alignment_sha256", "f0_sha256", "noise_sha256")
        ):
            raise ReciprocalFeasibilityProtocolError(
                f"record latent hashes are invalid: {record['slot_id']}"
            )
        pcm_by_slot[record["slot_id"]] = pcm

    by_slot = {record["slot_id"]: record for record in records}
    controlled_slots = (
        "controlled-neutral",
        "controlled-identity",
        "controlled-lens",
    )
    controlled = [by_slot[slot] for slot in controlled_slots]
    controlled_counts = {record["audio"]["sample_count"] for record in controlled}
    shared_fields = (
        "predicted_durations",
        "alignment_sha256",
        "f0_sha256",
        "noise_sha256",
        "target_intervals",
    )
    common_latents = all(
        stable_json(record[field]) == stable_json(controlled[0][field])
        for record in controlled[1:]
        for field in shared_fields
    )
    identity_equal = bool(
        np.array_equal(
            pcm_by_slot["controlled-neutral"],
            pcm_by_slot["controlled-identity"],
        )
    )
    declared_columns = [
        int(item["model_column"]) for item in profile_record["target_occurrences"]
    ]
    replacements_exact = bool(
        by_slot["controlled-lens"]["replaced_columns"] == declared_columns
        and all(
            not by_slot[slot]["replaced_columns"]
            for slot in (
                "ordinary-anchor-neutral",
                "ordinary-anchor-lens",
                "controlled-neutral",
                "controlled-identity",
            )
        )
    )
    audio_integrity = True
    for record in records:
        pcm_full_scale_fraction = float(
            np.mean(np.abs(pcm_by_slot[record["slot_id"]].astype(np.int32)) >= 32767)
        )
        audio_integrity = bool(
            audio_integrity
            and record["audio"]["finite"]
            and record["audio"]["clipping_pass"]
            and float(record["audio"]["clipped_fraction"]) < MAX_CLIPPED_FRACTION
            and math.isclose(
                float(record["audio"]["pcm_full_scale_fraction"]),
                pcm_full_scale_fraction,
                rel_tol=0,
                abs_tol=1e-15,
            )
            and record["audio"]["pcm_full_scale_pass"]
            == (pcm_full_scale_fraction < MAX_CLIPPED_FRACTION)
        )
    bounds_exact = bool(
        payload.get("decoder_call_count") == MAX_DECODER_CALLS
        and payload.get("decoder_call_limit") == MAX_DECODER_CALLS
        and payload.get("retries_made") == 0
        and payload.get("variants_rendered") == 0
        and payload.get("selection_performed") is False
    )
    integrity = {
        "bounded_manifest_exact": bounds_exact,
        "controlled_equal_nonzero_sample_counts": len(controlled_counts) == 1
        and next(iter(controlled_counts), 0) > 0,
        "controlled_common_alignment_f0_noise_exact": common_latents,
        "controlled_neutral_identity_bit_identical": identity_equal,
        "replaced_columns_exact": replacements_exact,
        "all_audio_hash_format_finite_and_below_clipping_limit": audio_integrity,
    }
    integrity["pass"] = all(integrity.values())
    if bool(payload.get("runtime_integrity", {}).get("pass")) != integrity["pass"]:
        raise ReciprocalFeasibilityProtocolError(
            "stored and independently recomputed runtime integrity disagree"
        )
    return by_slot, integrity


def _measure_record(
    record: dict[str, Any], destination: Path
) -> dict[str, dict[str, dict[str, Any]]]:
    path = destination / record["audio"]["relative_path"]
    output: dict[str, dict[str, dict[str, Any]]] = {}
    for occurrence in record["target_intervals"]:
        occurrence_key = str(occurrence["occurrence_index"])
        output[occurrence_key] = {
            str(ceiling): _measure(
                path, occurrence["primary_measurement_interval"], ceiling
            )
            for ceiling in CEILINGS_HZ
        }
    return output


def _automatic_branch(
    *,
    runtime_pass: bool,
    measurement_error: str | None,
    acoustic_pass: bool,
    localization_error: str | None,
    localization_pass: bool | None,
) -> str:
    if not runtime_pass:
        return "automatic_acoustic_feasibility_failed"
    if measurement_error is not None:
        return "automatic_measurement_inconclusive"
    if not acoustic_pass:
        return "automatic_acoustic_feasibility_failed"
    if localization_error is not None:
        return "automatic_measurement_inconclusive"
    if localization_pass is True:
        return "automatic_acoustic_feasibility_pass__blind_prototype_review_pending"
    return "automatic_acoustic_feasibility_failed"


def validate_review_response(
    response: dict[str, Any] | Path,
    *,
    analysis_path: Path,
    public_manifest_path: Path,
) -> dict[str, Any]:
    value = (
        json.loads(response.read_text(encoding="utf-8"))
        if isinstance(response, Path)
        else response
    )
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "run_id",
        "review_kind",
        "complete",
        "bindings",
        "ratings",
    }:
        raise ReciprocalFeasibilityProtocolError(
            "review response top-level schema is not exact"
        )
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    manifest = json.loads(public_manifest_path.read_text(encoding="utf-8"))
    if not (
        analysis.get("classification")
        == "automatic_acoustic_feasibility_pass__blind_prototype_review_pending"
        and analysis.get("automatic_acoustic_feasibility_pass") is True
        and manifest.get("protocol_sha256") == analysis.get("protocol_sha256")
        and manifest.get("analysis_file_sha256") == sha256_file(analysis_path)
        and manifest.get("render_records_file_sha256")
        == analysis.get("render_records_sha256")
        and manifest.get("response_schema_sha256") == sha256_file(RESPONSE_SCHEMA_PATH)
        and isinstance(manifest.get("clips"), list)
        and len(manifest["clips"]) == 3
        and len({clip.get("blind_id") for clip in manifest["clips"]}) == 3
        and all(
            isinstance(clip, dict)
            and set(clip) == {"blind_id", "audio", "wav_sha256"}
            and isinstance(clip["blind_id"], str)
            and len(clip["blind_id"]) == 24
            and clip["audio"] == f"audio/{clip['blind_id']}.wav"
            and isinstance(clip["wav_sha256"], str)
            and len(clip["wav_sha256"]) == 64
            for clip in manifest["clips"]
        )
    ):
        raise ReciprocalFeasibilityProtocolError(
            "public review manifest is not bound to the eligible analysis and schema"
        )
    expected_bindings = {
        "protocol_sha256": analysis["protocol_sha256"],
        "analysis_file_sha256": sha256_file(analysis_path),
        "public_manifest_file_sha256": sha256_file(public_manifest_path),
    }
    if not (
        value["schema_version"] == 1
        and value["run_id"] == RUN_ID
        and value["review_kind"] == "blind_prototype_qc_not_perceptual_validation"
        and value["complete"] is True
        and value["bindings"] == expected_bindings
    ):
        raise ReciprocalFeasibilityProtocolError(
            "review response session binding is invalid"
        )
    expected_ids = [clip["blind_id"] for clip in manifest["clips"]]
    ratings = value["ratings"]
    if not isinstance(ratings, dict) or set(ratings) != set(expected_ids):
        raise ReciprocalFeasibilityProtocolError(
            "review response clip set is not exact"
        )
    allowed = {
        "naturalness": {"1", "2", "3", "4", "5"},
        "artifact": {"none", "minor", "major", "uncertain"},
        "meaning": {
            "none",
            "isolated possible word",
            "coherent phrase",
            "uncertain",
        },
    }
    for clip_id in expected_ids:
        rating = ratings[clip_id]
        if not isinstance(rating, dict) or set(rating) != {
            "naturalness",
            "artifact",
            "meaning",
            "notes",
        }:
            raise ReciprocalFeasibilityProtocolError(
                f"review response fields are not exact for {clip_id}"
            )
        if any(
            not isinstance(rating[field], str) or rating[field] not in values
            for field, values in allowed.items()
        ):
            raise ReciprocalFeasibilityProtocolError(
                f"review response enum is invalid for {clip_id}"
            )
        if not isinstance(rating["notes"], str):
            raise ReciprocalFeasibilityProtocolError(
                f"review response notes must be a string for {clip_id}"
            )
    return value


def _record_review_generation_failure(
    *,
    destination: Path,
    analysis: dict[str, Any],
    analysis_hash: str,
    render_hash: str,
    error: Exception,
) -> None:
    atomic_write_json(
        destination / REVIEW_FAILURE_FILE,
        {
            "schema_version": 1,
            "run_id": RUN_ID,
            "classification": (
                "automatic_acoustic_feasibility_pass__review_generation_failed"
            ),
            "protocol_sha256": analysis.get("protocol_sha256"),
            "analysis_file_sha256": analysis_hash,
            "render_records_file_sha256": render_hash,
            "error": f"{type(error).__name__}: {str(error)[:500]}",
            "review_published": False,
        },
    )


def generate_review(
    *,
    analysis_path: Path,
    render_records_path: Path,
    destination: Path,
) -> dict[str, Any]:
    expected = "automatic_acoustic_feasibility_pass__blind_prototype_review_pending"
    if not (
        analysis_path.resolve() == (destination / ANALYSIS_FILE).resolve()
        and render_records_path.resolve()
        == (destination / RENDER_RECORDS_FILE).resolve()
    ):
        raise ReciprocalFeasibilityProtocolError(
            "review generation requires exact persisted run-directory paths"
        )
    if any(
        path.exists()
        for path in (
            destination / PUBLIC_REVIEW_ROOT,
            destination / PRIVATE_REVIEW_ROOT,
            destination / REVIEW_FAILURE_FILE,
            destination / "review-generation.partial",
            destination / REVIEW_FILE,
            destination / REVIEW_MANIFEST_FILE,
            destination / BLIND_KEY_FILE,
        )
    ):
        raise ReciprocalFeasibilityProtocolError(
            "blind review artifacts already exist; overwrite is forbidden"
        )
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    render_payload = json.loads(render_records_path.read_text(encoding="utf-8"))
    render_hash = sha256_file(render_records_path)
    analysis_hash = sha256_file(analysis_path)
    if not (
        analysis.get("classification") == expected
        and analysis.get("automatic_acoustic_feasibility_pass") is True
        and analysis.get("render_evidence_verified") is True
        and analysis.get("render_records_sha256") == render_hash
        and render_payload.get("protocol_sha256") == analysis.get("protocol_sha256")
    ):
        raise ReciprocalFeasibilityProtocolError(
            "blind prototype review requires persisted verified eligible analysis"
        )
    records_by_slot = {
        record["slot_id"]: record for record in render_payload.get("records", [])
    }
    source_rows: list[dict[str, Any]] = []
    for role, slot in (
        ("neutral", "controlled-neutral"),
        ("identity", "controlled-identity"),
        ("lens", "controlled-lens"),
    ):
        record = records_by_slot.get(slot)
        if record is None:
            error = ReciprocalFeasibilityProtocolError(
                "persisted render is missing a controlled review slot"
            )
            _record_review_generation_failure(
                destination=destination,
                analysis=analysis,
                analysis_hash=analysis_hash,
                render_hash=render_hash,
                error=error,
            )
            raise error
        source_path = destination / record["audio"]["relative_path"]
        if (
            not source_path.is_file()
            or sha256_file(source_path) != record["audio"]["wav_sha256"]
        ):
            error = ReciprocalFeasibilityProtocolError(
                "controlled review source WAV hash drifted"
            )
            _record_review_generation_failure(
                destination=destination,
                analysis=analysis,
                analysis_hash=analysis_hash,
                render_hash=render_hash,
                error=error,
            )
            raise error
        source_rows.append(
            {
                "role": role,
                "slot_id": slot,
                "source_path": source_path,
                "source_relative_path": record["audio"]["relative_path"],
                "wav_sha256": record["audio"]["wav_sha256"],
            }
        )

    try:
        secret = secrets.token_hex(32)
        secrets.SystemRandom().shuffle(source_rows)
    except Exception as exc:
        _record_review_generation_failure(
            destination=destination,
            analysis=analysis,
            analysis_hash=analysis_hash,
            render_hash=render_hash,
            error=exc,
        )
        raise
    partial = destination / "review-generation.partial"
    partial_public = partial / "public" / "review"
    partial_audio = partial_public / "audio"
    partial_private = partial / "private"
    try:
        partial_audio.mkdir(parents=True, exist_ok=False)
        partial_private.mkdir(parents=True, exist_ok=False)
        public_clips: list[dict[str, str]] = []
        private_mappings: list[dict[str, str]] = []
        for index, row in enumerate(source_rows, start=1):
            blind_id = hashlib.sha256(
                f"{secret}:{index}:{row['wav_sha256']}".encode("utf-8")
            ).hexdigest()[:24]
            filename = f"{blind_id}.wav"
            copy_path = partial_audio / filename
            shutil.copy2(row["source_path"], copy_path)
            if sha256_file(copy_path) != row["wav_sha256"]:
                raise ReciprocalFeasibilityProtocolError(
                    "opaque public review copy is not hash-identical"
                )
            public_clips.append(
                {
                    "blind_id": blind_id,
                    "audio": f"audio/{filename}",
                    "wav_sha256": row["wav_sha256"],
                }
            )
            private_mappings.append(
                {
                    "blind_id": blind_id,
                    "role": row["role"],
                    "slot_id": row["slot_id"],
                    "source_relative_path": row["source_relative_path"],
                    "wav_sha256": row["wav_sha256"],
                }
            )
        manifest = {
            "schema_version": 1,
            "run_id": RUN_ID,
            "status": "pending_blind_prototype_review",
            "purpose": "prototype QC after fixed-probe acoustic tests only",
            "claim_boundary": (
                "This review cannot establish perceptual efficacy, select a voice, "
                "enable a feature, or promote a candidate."
            ),
            "protocol_sha256": analysis["protocol_sha256"],
            "analysis_file_sha256": analysis_hash,
            "render_records_file_sha256": render_hash,
            "response_schema_sha256": sha256_file(RESPONSE_SCHEMA_PATH),
            "response_filename": RESPONSE_FILENAME,
            "required_fields_per_clip": list(REQUIRED_REVIEW_FIELDS),
            "clips": public_clips,
        }
        manifest_path = partial_public / REVIEW_MANIFEST_FILE
        atomic_write_json(manifest_path, manifest)
        manifest_hash = sha256_file(manifest_path)
        bindings = {
            "protocol_sha256": analysis["protocol_sha256"],
            "analysis_file_sha256": analysis_hash,
            "public_manifest_file_sha256": manifest_hash,
        }
        atomic_write_json(
            partial_private / BLIND_KEY_FILE,
            {
                "schema_version": 1,
                "run_id": RUN_ID,
                "blind_secret": secret,
                "public_manifest_file_sha256": manifest_hash,
                "mappings": private_mappings,
            },
        )
        html = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Blind prototype QC</title><style>body{font:17px/1.5 system-ui;max-width:820px;margin:auto;padding:24px;background:#f3f0e8;color:#18231e}.card{background:#fff;border:1px solid #d5d0c4;border-radius:16px;padding:18px;margin:16px 0}audio,select,textarea{width:100%;box-sizing:border-box}label{display:block;margin:12px 0}textarea{min-height:80px}button{border:0;border-radius:999px;padding:12px 18px;background:#174b3a;color:#fff;font-weight:700}</style></head><body><h1>Blind Portuguese technical-prototype QC</h1><p>This follows fixed engineering acoustic tests on one technical probe. It cannot prove perceptual efficacy, select a voice, enable a feature, or promote a mapping.</p><div id="clips"></div><p id="completion">Complete every required field to enable download.</p><button id="download" disabled>Download response JSON</button><script>const R=__ROWS__,B=__BINDINGS__,Q=['naturalness','artifact','meaning'],K='ptbr-blind-qc-v1:'+B.public_manifest_file_sha256,S=JSON.parse(localStorage.getItem(K)||'{"ratings":{}}');S.ratings??={};const esc=x=>String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));const opts=(a,v)=>'<option value="">—</option>'+a.map(x=>`<option value="${x}" ${x===v?'selected':''}>${x}</option>`).join('');document.getElementById('clips').innerHTML=R.map(r=>{const s=S.ratings[r.blind_id]??{};return `<section class="card"><h2>${r.blind_id}</h2><audio controls preload="metadata" src="${r.audio}"></audio><label>Naturalness (1–5)<select required data-id="${r.blind_id}" data-field="naturalness">${opts(['1','2','3','4','5'],s.naturalness)}</select></label><label>Audible artifact<select required data-id="${r.blind_id}" data-field="artifact">${opts(['none','minor','major','uncertain'],s.artifact)}</select></label><label>Stable recoverable meaning<select required data-id="${r.blind_id}" data-field="meaning">${opts(['none','isolated possible word','coherent phrase','uncertain'],s.meaning)}</select></label><label>Optional note<textarea data-id="${r.blind_id}" data-field="notes">${esc(s.notes)}</textarea></label></section>`}).join('');const D=document.getElementById('download'),M=document.getElementById('completion'),complete=()=>R.every(r=>Q.every(f=>String(S.ratings[r.blind_id]?.[f]??'').trim())),update=()=>{const ok=complete();D.disabled=!ok;M.textContent=ok?'Response complete and ready to download.':'Complete every required field to enable download.'};document.querySelectorAll('[data-id]').forEach(el=>{el.oninput=()=>{const id=el.dataset.id;S.ratings[id]??={notes:''};S.ratings[id][el.dataset.field]=el.value;localStorage.setItem(K,JSON.stringify(S));update()}});update();D.onclick=()=>{if(!complete())return;const ratings=Object.fromEntries(R.map(r=>{const s=S.ratings[r.blind_id];return[r.blind_id,{naturalness:s.naturalness,artifact:s.artifact,meaning:s.meaning,notes:String(s.notes??'')}]})),p={schema_version:1,run_id:'__RUN_ID__',review_kind:'blind_prototype_qc_not_perceptual_validation',complete:true,bindings:B,ratings},b=new Blob([JSON.stringify(p,null,2)+'\\n'],{type:'application/json'}),a=document.createElement('a');a.href=URL.createObjectURL(b);a.download='__RESPONSE__';a.click()};</script></body></html>"""
        html = (
            html.replace("__ROWS__", json.dumps(public_clips, ensure_ascii=False))
            .replace("__BINDINGS__", json.dumps(bindings, ensure_ascii=False))
            .replace("__RUN_ID__", RUN_ID)
            .replace("__RESPONSE__", RESPONSE_FILENAME)
        )
        atomic_write_text(partial_public / REVIEW_FILE, html)
        os.replace(partial / "public", destination / "public")
        os.replace(partial / "private", destination / "private")
        partial.rmdir()
        return {
            "status": "pending_blind_prototype_review",
            "public_manifest": str(PUBLIC_REVIEW_ROOT / REVIEW_MANIFEST_FILE),
            "public_manifest_file_sha256": manifest_hash,
            "analysis_file_sha256": analysis_hash,
            "render_records_file_sha256": render_hash,
            "opaque_copy_count": len(public_clips),
        }
    except Exception as exc:
        shutil.rmtree(partial, ignore_errors=True)
        shutil.rmtree(destination / "public", ignore_errors=True)
        shutil.rmtree(destination / "private", ignore_errors=True)
        _record_review_generation_failure(
            destination=destination,
            analysis=analysis,
            analysis_hash=analysis_hash,
            render_hash=render_hash,
            error=exc,
        )
        raise


def analyze() -> dict[str, Any]:
    protocol = verify_frozen_protocol()
    destination = run_dir()
    analysis_path = destination / ANALYSIS_FILE
    forbidden = (
        analysis_path,
        destination / PUBLIC_REVIEW_ROOT,
        destination / PRIVATE_REVIEW_ROOT,
        destination / REVIEW_FAILURE_FILE,
        destination / "review-generation.partial",
        destination / REVIEW_FILE,
        destination / REVIEW_MANIFEST_FILE,
        destination / BLIND_KEY_FILE,
    )
    _reject_existing(forbidden, stage="analysis")
    attempt_path = destination / RENDER_ATTEMPT_FILE
    render_path = destination / RENDER_RECORDS_FILE
    if not (
        attempt_path.is_file()
        and render_path.is_file()
        and (destination / "audio").is_dir()
    ):
        raise ReciprocalFeasibilityProtocolError(
            "analysis requires one persisted completed render attempt"
        )
    committed_inputs = _verify_current_repo_inputs(protocol)
    attempt = json.loads(attempt_path.read_text(encoding="utf-8"))
    if not (
        attempt.get("protocol_sha256") == protocol["protocol_sha256"]
        and attempt.get("committed_inputs") == committed_inputs
    ):
        raise ReciprocalFeasibilityProtocolError(
            "attempt marker protocol, input, or common-HEAD receipt drifted"
        )
    render_payload = json.loads(render_path.read_text(encoding="utf-8"))
    records, runtime_integrity = _verify_render_evidence(
        render_payload, destination, protocol
    )
    runtime_pass = bool(runtime_integrity["pass"])

    measurements: dict[str, Any] = {}
    occurrences: dict[str, Any] = {}
    acoustic_pass = False
    measurement_error: str | None = None
    localization_error: str | None = None
    if not runtime_pass:
        measurement_status = "unavailable_runtime_integrity_failure"
        localization: dict[str, Any] = {
            "status": "skipped_runtime_integrity_failure",
            "pass": False,
        }
    else:
        try:
            for slot_id in (
                "ordinary-anchor-neutral",
                "ordinary-anchor-lens",
                "controlled-neutral",
                "controlled-lens",
            ):
                measurements[slot_id] = _measure_record(records[slot_id], destination)
        except Exception as exc:
            measurement_error = f"{type(exc).__name__}: {str(exc)[:500]}"
        if measurement_error is not None:
            measurement_status = "inconclusive_measurement_error"
            localization = {
                "status": "skipped_measurement_inconclusive",
                "pass": False,
            }
        else:
            measurement_status = "complete_target_column_primary_middle_50_only"
            occurrence_keys = tuple(measurements["controlled-neutral"].keys())
            for key in occurrence_keys:
                occurrences[key] = classify_local_acoustics(
                    anchor_neutral=measurements["ordinary-anchor-neutral"][key],
                    anchor_lens=measurements["ordinary-anchor-lens"][key],
                    controlled_neutral=measurements["controlled-neutral"][key],
                    controlled_lens=measurements["controlled-lens"][key],
                )
            acoustic_pass = bool(
                occurrences
                and all(
                    record["all_three_ceilings_pass"] for record in occurrences.values()
                )
            )
            if not acoustic_pass:
                localization = {
                    "status": "skipped_conclusive_acoustic_failure",
                    "pass": False,
                }
            else:
                try:
                    neutral_path = (
                        destination
                        / records["controlled-neutral"]["audio"]["relative_path"]
                    )
                    lens_path = (
                        destination
                        / records["controlled-lens"]["audio"]["relative_path"]
                    )
                    localization = {
                        "status": "complete",
                        **localization_report(
                            _read_pcm(neutral_path),
                            _read_pcm(lens_path),
                            records["controlled-neutral"]["target_intervals"],
                        ),
                    }
                except Exception as exc:
                    localization_error = f"{type(exc).__name__}: {str(exc)[:500]}"
                    localization = {
                        "status": "unavailable_localization_tool_error",
                        "error": localization_error,
                        "pass": False,
                    }
    classification = _automatic_branch(
        runtime_pass=runtime_pass,
        measurement_error=measurement_error,
        acoustic_pass=acoustic_pass,
        localization_error=localization_error,
        localization_pass=(
            bool(localization["pass"])
            if localization.get("status") == "complete"
            else None
        ),
    )
    automatic_pass = bool(
        classification
        == "automatic_acoustic_feasibility_pass__blind_prototype_review_pending"
    )
    payload = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "status": "analysis_complete",
        "protocol_sha256": protocol["protocol_sha256"],
        "committed_inputs": committed_inputs,
        "attempt_marker_sha256": sha256_file(attempt_path),
        "render_records_sha256": sha256_file(render_path),
        "render_evidence_verified": True,
        "classification": classification,
        "branch_precedence": (
            "runtime fail => failed; preclassification measurement error => "
            "inconclusive; conclusive acoustic fail => failed and localization "
            "skipped; acoustic pass plus localization-tool error => inconclusive; "
            "otherwise localization pass/fail decides"
        ),
        "runtime_integrity": runtime_integrity,
        "measurement_status": measurement_status,
        "measurement_window_policy": {
            "primary_middle_fraction": MIDDLE_FRACTION,
            "exploratory_windows_computed": [],
            "middle_40_computed": False,
            "middle_60_computed": False,
        },
        "measurements": measurements,
        "measurement_error": measurement_error,
        "occurrences": occurrences,
        "all_occurrences_all_three_ceilings_pass": acoustic_pass,
        "localization": localization,
        "localization_error": localization_error,
        "automatic_acoustic_feasibility_pass": automatic_pass,
        "claim": (
            "local median-F1/F2 shift plus localized waveform difference on one "
            "fixed technical probe only"
            if automatic_pass
            else "no positive acoustic-feasibility claim"
        ),
        "numeric_thresholds": "engineering_nonperceptual_criteria_only",
        "perceptual_efficacy_established": False,
        "voice_selected": False,
        "technical_probe_result_transferable_to_selected_voice": False,
        "candidate_enabled": False,
        "production_route_available": False,
        "api_calls": 0,
        "paid_calls": 0,
    }
    atomic_write_json(analysis_path, payload)
    if automatic_pass:
        generate_review(
            analysis_path=analysis_path,
            render_records_path=render_path,
            destination=destination,
        )
    elif any(
        path.exists()
        for path in (
            destination / PUBLIC_REVIEW_ROOT,
            destination / PRIVATE_REVIEW_ROOT,
            destination / REVIEW_FAILURE_FILE,
        )
    ):
        raise ReciprocalFeasibilityProtocolError(
            "failed or inconclusive analysis must not publish review artifacts"
        )
    return payload
