from __future__ import annotations

import resource
import subprocess
import time
import wave
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
from huggingface_hub import snapshot_download
from mlx_whisper.audio import N_FRAMES, N_SAMPLES, load_audio, log_mel_spectrogram, pad_or_trim
from mlx_whisper.decoding import DecodingOptions
from mlx_whisper.load_models import load_model

from .config import Paths, load_config
from .gates import transcript_nonword_rate
from .models import VerificationResult
from .util import atomic_write_json, sha256_file


class VerifierError(RuntimeError):
    pass


CONTROL_TEXT = {
    "en": "The teacher reads a short passage while the students listen carefully for rhythm, stress, and natural intonation.",
    "es": "La profesora lee un pasaje breve mientras los estudiantes escuchan con atención el ritmo y la entonación natural.",
    "pt": "A professora lê uma passagem curta enquanto os alunos escutam com atenção o ritmo e a entonação natural.",
}
CONTROL_VOICES = {"en": "Samantha", "es": "Paulina", "pt": "Luciana"}


def normalize_wav(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.stem + ".partial" + destination.suffix)
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(partial),
        ],
        check=True,
    )
    partial.replace(destination)


def wav_integrity(path: Path) -> tuple[float, int, float]:
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        sample_rate = handle.getframerate()
        sample_width = handle.getsampwidth()
        frames = handle.readframes(handle.getnframes())
    if channels != 1 or sample_width != 2:
        raise VerifierError(f"Expected mono 16-bit PCM WAV, got {channels}ch/{sample_width * 8}bit")
    samples = np.frombuffer(frames, dtype="<i2")
    clipped = float(np.mean(np.abs(samples.astype(np.int32)) >= 32767)) if samples.size else 1.0
    duration = samples.size / sample_rate
    return duration, sample_rate, clipped


def _download_checkpoint(config: dict[str, Any], fallback: bool = False) -> Path:
    whisper = config["whisper"]
    repo_key = "fallback_repo" if fallback else "model_repo"
    revision_key = "fallback_revision" if fallback else "model_revision"
    label = "large-v3-4bit" if fallback else "large-v3-full"
    target = Paths().whisper_cache / label
    target.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=whisper[repo_key],
        revision=whisper[revision_key],
        allow_patterns=["README.md", "config.json", "weights.npz"],
        local_dir=target,
    )
    expected_weights = (
        whisper["fallback_weights_sha256"] if fallback else whisper["weights_sha256"]
    )
    actual_weights = sha256_file(target / "weights.npz")
    if actual_weights != expected_weights:
        raise VerifierError(
            f"Whisper weights checksum mismatch: {actual_weights} != {expected_weights}"
        )
    if not fallback:
        actual_config = sha256_file(target / "config.json")
        if actual_config != whisper["config_sha256"]:
            raise VerifierError(
                f"Whisper config checksum mismatch: {actual_config} != {whisper['config_sha256']}"
            )
    return target


class WhisperVerifier:
    def __init__(self, model_path: Path, gate_db: Path | None = None) -> None:
        self.model_path = model_path
        self.gate_db = gate_db or Paths().gate_db
        self.model = load_model(str(model_path), dtype=mx.float16)

    def _decode(self, audio_path: Path, target_language: str) -> dict[str, Any]:
        audio = load_audio(str(audio_path))
        mel = log_mel_spectrogram(
            audio, n_mels=self.model.dims.n_mels, padding=N_SAMPLES
        )
        mel_segment = pad_or_trim(mel, N_FRAMES, axis=-2).astype(mx.float16)
        # MLX Conv1d expects [batch, time, mel]. Reuse the single encoded
        # feature matrix for both language ID and forced-language decoding.
        features = self.model.encoder(mel_segment[None])[0]
        _, scores = self.model.detect_language(features)
        result = self.model.decode(
            features,
            DecodingOptions(
                language=target_language,
                task="transcribe",
                temperature=0.0,
                sample_len=128,
            ),
        )
        ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        top_language, top_score = ordered[0]
        runner_language, runner_score = ordered[1]
        return {
            "scores": scores,
            "top_language": top_language,
            "top_score": float(top_score),
            "target_score": float(scores.get(target_language, 0.0)),
            "runner_language": runner_language,
            "margin": float(top_score - runner_score),
            "text": result.text.strip(),
            "avg_logprob": float(result.avg_logprob),
            "no_speech_probability": float(result.no_speech_prob),
            "compression_ratio": float(result.compression_ratio),
        }

    def verify(
        self, source: Path, target_language: str, normalized: Path
    ) -> VerificationResult:
        try:
            normalize_wav(source, normalized)
            duration, sample_rate, clipped = wav_integrity(normalized)
            decoded = self._decode(normalized, target_language)
            no_speech_ok = decoded["no_speech_probability"] < 0.60
            audio_ok = 5.0 <= duration <= 20.0 and clipped < 0.001 and no_speech_ok
            top_is_target = decoded["top_language"] == target_language
            full_gate = (
                top_is_target
                and decoded["target_score"] >= 0.70
                and decoded["margin"] >= 0.25
            )
            sister_split = (
                target_language in {"es", "pt"}
                and top_is_target
                and 0.50 <= decoded["target_score"] < 0.70
            )
            machine_pass = audio_ok and (full_gate or sister_split)
            return VerificationResult(
                audio_ok=audio_ok,
                duration_s=duration,
                sample_rate_hz=sample_rate,
                clipped_fraction=clipped,
                top_language=decoded["top_language"],
                target_score=decoded["target_score"],
                runner_up_language=decoded["runner_language"],
                margin=decoded["margin"],
                language_scores=decoded["scores"],
                transcript=decoded["text"],
                transcript_nonword_rate=transcript_nonword_rate(decoded["text"], self.gate_db),
                no_speech_probability=decoded["no_speech_probability"],
                avg_logprob=decoded["avg_logprob"],
                compression_ratio=decoded["compression_ratio"],
                sister_language_split=sister_split,
                machine_pass=machine_pass,
            )
        except Exception as exc:
            return VerificationResult(
                audio_ok=False,
                duration_s=None,
                sample_rate_hz=None,
                clipped_fraction=None,
                top_language=None,
                target_score=None,
                runner_up_language=None,
                margin=None,
                language_scores={},
                transcript=None,
                transcript_nonword_rate=None,
                no_speech_probability=None,
                avg_logprob=None,
                compression_ratio=None,
                sister_language_split=False,
                machine_pass=False,
                error_detail=str(exc),
            )


def _peak_rss_gib() -> float:
    # macOS reports ru_maxrss in bytes.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**3)


def _make_say_controls(directory: Path) -> dict[str, Path]:
    controls: dict[str, Path] = {}
    directory.mkdir(parents=True, exist_ok=True)
    for language, text in CONTROL_TEXT.items():
        aiff = directory / f"{language}.aiff"
        wav = directory / f"{language}.wav"
        subprocess.run(
            ["say", "-v", CONTROL_VOICES[language], "-r", "170", "-o", str(aiff), text],
            check=True,
        )
        normalize_wav(aiff, wav)
        controls[language] = wav
    return controls


def prepare_whisper(config: dict[str, Any] | None = None) -> dict[str, Any]:
    config = config or load_config()
    model_path = _download_checkpoint(config, fallback=False)
    control_dir = Paths().whisper_cache / "controls"
    controls = _make_say_controls(control_dir)

    try:
        verifier = WhisperVerifier(model_path)
        # Warm the exact model/code path once before timed measurements.
        verifier._decode(controls["en"], "en")
        measurements: dict[str, Any] = {}
        all_ok = True
        for language, path in controls.items():
            started = time.monotonic()
            decoded = verifier._decode(path, language)
            elapsed = time.monotonic() - started
            ok = (
                decoded["top_language"] == language
                and decoded["target_score"] >= 0.70
                and decoded["margin"] >= 0.30
                and bool(decoded["text"])
                and decoded["no_speech_probability"] < 0.60
                and elapsed <= 15.0
            )
            all_ok = all_ok and ok
            measurements[language] = {
                **decoded,
                "elapsed_s": elapsed,
                "ok": ok,
                "control_sha256": sha256_file(path),
            }
        rss = _peak_rss_gib()
        all_ok = all_ok and rss < 12.0
        if not all_ok:
            raise VerifierError(
                f"Full-precision preflight failed (peak RSS {rss:.2f} GiB): {measurements}"
            )
        receipt = {
            "variant": "large-v3-full",
            "model_path": str(model_path),
            "revision": config["whisper"]["model_revision"],
            "weights_sha256": sha256_file(model_path / "weights.npz"),
            "peak_rss_gib": rss,
            "measurements": measurements,
        }
    except (MemoryError, VerifierError) as full_error:
        fallback_path = _download_checkpoint(config, fallback=True)
        verifier = WhisperVerifier(fallback_path)
        verifier._decode(controls["en"], "en")
        measurements = {}
        all_ok = True
        for language, path in controls.items():
            started = time.monotonic()
            decoded = verifier._decode(path, language)
            elapsed = time.monotonic() - started
            ok = (
                decoded["top_language"] == language
                and decoded["target_score"] >= 0.70
                and decoded["margin"] >= 0.30
                and bool(decoded["text"])
                and decoded["no_speech_probability"] < 0.60
                and elapsed <= 15.0
            )
            all_ok = all_ok and ok
            measurements[language] = {
                **decoded,
                "elapsed_s": elapsed,
                "ok": ok,
                "control_sha256": sha256_file(path),
            }
        rss = _peak_rss_gib()
        if not all_ok or rss >= 12.0:
            raise VerifierError(
                f"Both verifier variants failed. Full error: {full_error}; fallback: {measurements}"
            )
        receipt = {
            "variant": "large-v3-4bit",
            "model_path": str(fallback_path),
            "revision": config["whisper"]["fallback_revision"],
            "weights_sha256": sha256_file(fallback_path / "weights.npz"),
            "peak_rss_gib": rss,
            "measurements": measurements,
            "full_precision_error": str(full_error),
        }
    atomic_write_json(Paths().whisper_cache / "preflight-receipt.json", receipt)
    return receipt
