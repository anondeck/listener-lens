from __future__ import annotations

import concurrent.futures
import hashlib
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from .api import (
    ChatAudioRenderer,
    Generator,
    OpenAIGenerator,
    Renderer,
    SpeechRenderer,
    api_contract_fingerprint,
    renderer_instruction,
    require_api_key,
)
from .config import Paths, load_config, sha256_json, stable_json, verify_criteria_hash
from .corpus import CodexCorpusGenerator
from .gates import CandidateGate, EspeakPhonemizer, GateResult
from .models import RenderResult, ScriptCandidate, VerificationResult
from .util import atomic_write_json, sha256_file, write_csv
from .verifier import WhisperVerifier


RESULT_FIELDS = [
    "run_id",
    "language",
    "profile_id",
    "attempt",
    "script_id",
    "script_text",
    "script_sha256",
    "token_count",
    "syllable_count",
    "generator_model",
    "generator_source",
    "generator_response_id",
    "generation_round",
    "gate_pass",
    "gate_reasons",
    "renderer_slug",
    "renderer_model",
    "voice",
    "audio_filename",
    "raw_audio_path",
    "normalized_audio_path",
    "render_status",
    "render_retry_count",
    "request_id",
    "resolved_model",
    "latency_ms",
    "provider_transcript",
    "audio_sha256",
    "duration_s",
    "sample_rate_hz",
    "clipped_fraction",
    "whisper_variant",
    "whisper_top_language",
    "whisper_target_score",
    "whisper_runner_up_language",
    "whisper_margin",
    "whisper_language_scores_json",
    "whisper_transcript",
    "transcript_nonword_rate",
    "no_speech_probability",
    "avg_logprob",
    "compression_ratio",
    "sister_language_split",
    "machine_pass",
    "blind_id",
    "human_fluent",
    "human_pace",
    "human_prosody",
    "human_coherence",
    "human_confidence",
    "human_glitch_or_spelling",
    "human_real_word_autocorrection",
    "human_notes",
    "human_pass",
    "g2p_sampled",
    "g2p_token",
    "g2p_token_index",
    "espeak_voice",
    "espeak_ipa",
    "g2p_reference_path",
    "g2p_judgment",
    "clip_pass",
    "failure_stage",
    "failure_code",
    "failure_detail",
    "created_at_utc",
    "completed_at_utc",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class RunError(RuntimeError):
    pass


@dataclass
class Services:
    generator: Generator
    renderers: Sequence[Renderer]
    gate: Any
    verifier: Any
    phonemizer: Any
    whisper_variant: str


def _language_for_profile(profile_id: str) -> str:
    return {"en-US-mae": "en", "es-MX-cdmx": "es", "pt-BR-sp": "pt"}[profile_id]


def _candidate_map(batch: Any) -> dict[str, list[ScriptCandidate]]:
    mapped: dict[str, list[ScriptCandidate]] = {}
    for language_batch in batch.languages:
        mapped.setdefault(language_batch.profile_id, []).extend(language_batch.candidates)
    return mapped


def collect_accepted_scripts(
    generator: Generator,
    gate: Any,
    config: dict[str, Any],
) -> tuple[dict[str, list[ScriptCandidate]], list[dict[str, Any]], dict[str, list[dict[str, str]]]]:
    profiles = config["profiles"]
    target = config["generator"]["target_scripts"]
    accepted: dict[str, list[ScriptCandidate]] = {profile: [] for profile in profiles}
    gate_log: list[dict[str, Any]] = []
    generation_meta: dict[str, list[dict[str, str]]] = {profile: [] for profile in profiles}

    for generation_round in range(config["generator"]["max_refill_calls"] + 1):
        deficient = [profile for profile in profiles if len(accepted[profile]) < target]
        if not deficient:
            break
        count = (
            config["generator"]["initial_candidates_per_language"]
            if generation_round == 0
            else max(10, (target - min(len(accepted[p]) for p in deficient)) * 2)
        )
        batch = generator.generate(deficient, count, refill_index=generation_round)
        response_id = getattr(generator, "last_response_id", None)
        resolved_model = getattr(generator, "last_resolved_model", None)
        source = "codex" if hasattr(generator, "corpus_sha256") else "responses_api"
        mapped = _candidate_map(batch)
        for profile in deficient:
            generation_meta[profile].append(
                {
                    "round": str(generation_round),
                    "response_id": response_id or "",
                    "resolved_model": resolved_model or config["generator"]["model"],
                    "source": source,
                }
            )
            for candidate in mapped.get(profile, []):
                if len(accepted[profile]) >= target:
                    break
                if candidate.profile_id != profile:
                    gate_log.append(
                        {
                            "candidate_id": candidate.candidate_id,
                            "profile_id": profile,
                            "round": generation_round,
                            "passed": False,
                            "reasons": ["profile_mismatch"],
                        }
                    )
                    continue
                result: GateResult = gate.gate(candidate)
                gate_log.append(
                    {
                        "candidate_id": candidate.candidate_id,
                        "profile_id": profile,
                        "round": generation_round,
                        "passed": result.passed,
                        "reasons": result.reasons,
                    }
                )
                if result.passed:
                    candidate.candidate_id = f"{_language_for_profile(profile)}-{len(accepted[profile]) + 1:02d}"
                    accepted[profile].append(candidate)

    return accepted, gate_log, generation_meta


def _blind_id(run_id: str, language: str, renderer: str, attempt: int) -> str:
    return hashlib.sha256(
        f"{run_id}\0{language}\0{renderer}\0{attempt}".encode("utf-8")
    ).hexdigest()[:12]


def _is_retryable(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    return isinstance(exc, (TimeoutError, ConnectionError)) or status == 429 or (
        isinstance(status, int) and status >= 500
    )


def _render_with_retries(
    renderer: Renderer,
    script: str,
    instruction: str,
    voice: str,
    output: Path,
    max_retries: int,
) -> RenderResult:
    for retry in range(max_retries + 1):
        try:
            result = renderer.render(script, instruction, voice, output)
            result.retry_count = retry
            return result
        except Exception as exc:
            if retry >= max_retries or not _is_retryable(exc):
                return RenderResult(
                    renderer_slug=renderer.slug,
                    renderer_model=renderer.model,
                    status="failed",
                    retry_count=retry,
                    error_code=type(exc).__name__,
                    error_detail=str(exc),
                )
            time.sleep(min(4.0, 0.5 * (2**retry)))
    raise AssertionError("retry loop exhausted unexpectedly")


def _select_g2p_probes(
    accepted: dict[str, list[ScriptCandidate]], run_id: str
) -> dict[str, tuple[int, str, int]]:
    selected: dict[str, tuple[int, str, int]] = {}
    for profile, candidates in accepted.items():
        if len(candidates) < 5:
            continue
        candidate_indices = sorted({round(i * (len(candidates) - 1) / 4) for i in range(5)})
        while len(candidate_indices) < 5:
            candidate_indices.append(next(i for i in range(len(candidates)) if i not in candidate_indices))
            candidate_indices.sort()
        for sample_index, candidate_index in enumerate(candidate_indices[:5]):
            candidate = candidates[candidate_index]
            eligible = [
                (index, token)
                for index, token in enumerate(candidate.tokens[1:-1], start=1)
                if 4 <= len(token.surface) <= 12 and token.role == "content"
            ]
            if not eligible:
                eligible = list(enumerate(candidate.tokens))
            if sample_index >= 3:
                token_index, token = max(
                    eligible, key=lambda item: (len(item[1].rule_ids), -item[0])
                )
            else:
                seed = int(
                    hashlib.sha256(
                        f"{run_id}\0{profile}\0{candidate.candidate_id}".encode("utf-8")
                    ).hexdigest()[:8],
                    16,
                )
                token_index, token = eligible[seed % len(eligible)]
            selected[candidate.candidate_id] = (token_index, token.surface, sample_index)
    return selected


def _base_row(
    run_id: str,
    candidate: ScriptCandidate,
    attempt: int,
    renderer: Renderer,
    voice: str,
    generation_meta: dict[str, list[dict[str, str]]],
    created_at: str,
) -> dict[str, Any]:
    language = candidate.language
    filename_language = "pt-BR" if language == "pt" else language
    filename = f"{filename_language}__{renderer.slug}__attempt-{attempt:02d}.wav"
    latest_meta = generation_meta[candidate.profile_id][-1] if generation_meta[candidate.profile_id] else {}
    return {
        "run_id": run_id,
        "language": language,
        "profile_id": candidate.profile_id,
        "attempt": attempt,
        "script_id": candidate.candidate_id,
        "script_text": candidate.text,
        "script_sha256": hashlib.sha256(candidate.text.encode("utf-8")).hexdigest(),
        "token_count": len(candidate.tokens),
        "syllable_count": candidate.syllable_count,
        "generator_model": latest_meta.get("resolved_model", ""),
        "generator_source": latest_meta.get("source", ""),
        "generator_response_id": latest_meta.get("response_id", ""),
        "generation_round": latest_meta.get("round", ""),
        "gate_pass": True,
        "gate_reasons": "[]",
        "renderer_slug": renderer.slug,
        "renderer_model": renderer.model,
        "voice": voice,
        "audio_filename": filename,
        "blind_id": _blind_id(run_id, language, renderer.slug, attempt),
        "g2p_sampled": False,
        "created_at_utc": created_at,
    }


def execute_run(
    run_id: str,
    *,
    services: Services | None = None,
    require_live_prerequisites: bool = True,
    voice: str | None = None,
) -> Path:
    config = load_config()
    criteria_hash = verify_criteria_hash(config)
    paths = Paths()

    if require_live_prerequisites:
        require_api_key()
        if not paths.prepare_receipt.is_file():
            raise RunError("Local prepare receipt is missing")
        if not paths.smoke_receipt.is_file():
            raise RunError("Smoke receipt is missing; do not start T0")
        prepare = json.loads(paths.prepare_receipt.read_text(encoding="utf-8"))
        if prepare.get("criteria_sha256") != criteria_hash:
            raise RunError("Prepare receipt does not match the locked pass criteria")
        gate_receipt = prepare.get("gate", {})
        if gate_receipt.get("database_sha256") != sha256_file(paths.gate_db):
            raise RunError("Prepared word/G2P database checksum changed")
        whisper_receipt = prepare.get("whisper", {})
        model_path = Path(whisper_receipt.get("model_path", ""))
        weights_path = model_path / "weights.npz"
        if (
            not weights_path.is_file()
            or whisper_receipt.get("weights_sha256") != sha256_file(weights_path)
        ):
            raise RunError("Prepared Whisper weights are missing or changed")

        smoke = json.loads(paths.smoke_receipt.read_text(encoding="utf-8"))
        if smoke.get("criteria_sha256") != criteria_hash:
            raise RunError("Smoke receipt does not match the locked pass criteria")
        voice = smoke["effective_voice"]
        if smoke.get("api_contract_fingerprint") != api_contract_fingerprint(voice):
            raise RunError("Smoke receipt API contract is stale; rerun smoke before T0")
        verifier = WhisperVerifier(Path(whisper_receipt["model_path"]))
        generator_config = config["generator"]
        if generator_config.get("mode") == "codex_corpus":
            corpus_path = paths.root / generator_config["corpus_path"]
            generator: Generator = CodexCorpusGenerator(corpus_path)
        else:
            generator = OpenAIGenerator()
        services = Services(
            generator=generator,
            renderers=[SpeechRenderer(), ChatAudioRenderer()],
            gate=CandidateGate(),
            verifier=verifier,
            phonemizer=EspeakPhonemizer(),
            whisper_variant=whisper_receipt["variant"],
        )
    if services is None:
        raise RunError("Services are required")
    voice = voice or config["preferred_voice"]

    run_dir = paths.run_dir(run_id)
    if run_dir.exists() and any(run_dir.iterdir()):
        raise RunError(f"Run directory is not empty: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)

    started_at = utc_now()
    started_monotonic = time.monotonic()
    manifest = {
        "run_id": run_id,
        "status": "generating",
        "started_at_utc": started_at,
        "timebox_seconds": config["timebox_seconds"],
        "criteria_sha256": criteria_hash,
        "voice": voice,
        "config_fingerprint": sha256_json(config),
        "api_contract_fingerprint": api_contract_fingerprint(voice),
        "generator_source": config["generator"].get("mode", "responses_api"),
        "corpus_sha256": getattr(services.generator, "corpus_sha256", ""),
        "manual_pause_seconds": 0,
    }
    atomic_write_json(run_dir / "run.json", manifest)

    accepted, gate_log, generation_meta = collect_accepted_scripts(
        services.generator, services.gate, config
    )
    atomic_write_json(run_dir / "gate-log.json", gate_log)
    atomic_write_json(
        run_dir / "scripts.json",
        {
            profile: [candidate.model_dump(mode="json") for candidate in candidates]
            for profile, candidates in accepted.items()
        },
    )

    minimum = config["generator"]["minimum_shortfall_scripts"]
    active = {
        profile: candidates for profile, candidates in accepted.items() if len(candidates) >= minimum
    }
    manifest["generation_counts"] = {
        profile: len(candidates) for profile, candidates in accepted.items()
    }
    manifest["generation_failures"] = [
        profile for profile, candidates in accepted.items() if len(candidates) < minimum
    ]
    manifest["status"] = "rendering"
    atomic_write_json(run_dir / "run.json", manifest)

    rows: list[dict[str, Any]] = []
    raw_dir = run_dir / "audio" / "raw"
    created_at = utc_now()
    for profile, candidates in active.items():
        for attempt, candidate in enumerate(candidates, start=1):
            instruction = renderer_instruction(profile)
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                jobs = []
                job_rows = []
                for renderer in services.renderers:
                    row = _base_row(
                        run_id,
                        candidate,
                        attempt,
                        renderer,
                        voice,
                        generation_meta,
                        created_at,
                    )
                    output = raw_dir / row["audio_filename"]
                    jobs.append(
                        executor.submit(
                            _render_with_retries,
                            renderer,
                            candidate.text,
                            instruction,
                            voice,
                            output,
                            config["max_transport_retries"],
                        )
                    )
                    job_rows.append(row)
                for row, future in zip(job_rows, jobs):
                    result = future.result()
                    row.update(
                        {
                            "render_status": result.status,
                            "render_retry_count": result.retry_count,
                            "request_id": result.request_id or "",
                            "resolved_model": result.resolved_model or "",
                            "latency_ms": result.latency_ms or "",
                            "provider_transcript": result.provider_transcript or "",
                            "raw_audio_path": result.output_path or "",
                            "failure_stage": "" if result.status == "ok" else "render",
                            "failure_code": result.error_code or "",
                            "failure_detail": result.error_detail or "",
                        }
                    )
                    if result.status == "ok" and result.output_path:
                        row["audio_sha256"] = sha256_file(Path(result.output_path))
                    rows.append(row)

    manifest["status"] = "verifying"
    atomic_write_json(run_dir / "run.json", manifest)
    normalized_dir = run_dir / "audio" / "normalized"
    for row in rows:
        if row.get("render_status") != "ok":
            row["machine_pass"] = False
            continue
        normalized = normalized_dir / row["audio_filename"]
        verification: VerificationResult = services.verifier.verify(
            Path(row["raw_audio_path"]), row["language"], normalized
        )
        row.update(
            {
                "normalized_audio_path": str(normalized),
                "duration_s": verification.duration_s,
                "sample_rate_hz": verification.sample_rate_hz,
                "clipped_fraction": verification.clipped_fraction,
                "whisper_variant": services.whisper_variant,
                "whisper_top_language": verification.top_language or "",
                "whisper_target_score": verification.target_score,
                "whisper_runner_up_language": verification.runner_up_language or "",
                "whisper_margin": verification.margin,
                "whisper_language_scores_json": stable_json(verification.language_scores),
                "whisper_transcript": verification.transcript or "",
                "transcript_nonword_rate": verification.transcript_nonword_rate,
                "no_speech_probability": verification.no_speech_probability,
                "avg_logprob": verification.avg_logprob,
                "compression_ratio": verification.compression_ratio,
                "sister_language_split": verification.sister_language_split,
                "machine_pass": verification.machine_pass,
            }
        )
        if verification.error_detail:
            row.update(
                {
                    "failure_stage": "verify",
                    "failure_code": "verification_error",
                    "failure_detail": verification.error_detail,
                }
            )

    probes = _select_g2p_probes(active, run_id)
    profile_language = {profile: _language_for_profile(profile) for profile in active}
    reference_dir = run_dir / "g2p_reference"
    for profile, candidates in active.items():
        language = profile_language[profile]
        voice_id = config["word_gate"]["voices"][language]
        for candidate in candidates:
            if candidate.candidate_id not in probes:
                continue
            token_index, token, _ = probes[candidate.candidate_id]
            ipa = services.phonemizer.phonemize([token], voice_id)[0]
            reference_path = reference_dir / f"{candidate.candidate_id}__{token}.wav"
            if hasattr(services.phonemizer, "reference_wav"):
                services.phonemizer.reference_wav(token, voice_id, reference_path)
            for row in rows:
                if row["script_id"] == candidate.candidate_id:
                    row.update(
                        {
                            "g2p_sampled": True,
                            "g2p_token": token,
                            "g2p_token_index": token_index,
                            "espeak_voice": voice_id,
                            "espeak_ipa": ipa,
                            "g2p_reference_path": str(reference_path),
                        }
                    )

    completed_at = utc_now()
    for row in rows:
        row.setdefault("human_pass", "")
        row.setdefault("clip_pass", "")
        row["completed_at_utc"] = completed_at
    write_csv(run_dir / "results.csv", rows, RESULT_FIELDS)
    manifest.update(
        {
            "status": "awaiting_review",
            "automated_completed_at_utc": completed_at,
            "active_elapsed_seconds_at_automation_end": time.monotonic() - started_monotonic,
            "result_rows": len(rows),
        }
    )
    atomic_write_json(run_dir / "run.json", manifest)
    return run_dir / "results.csv"
