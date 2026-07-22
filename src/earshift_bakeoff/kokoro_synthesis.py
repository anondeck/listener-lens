from __future__ import annotations

import hashlib
import importlib.metadata
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from huggingface_hub import hf_hub_download

from .util import sha256_file


KOKORO_VERSION = "0.9.4"
MODEL_REPO = "hexgrad/Kokoro-82M"
MODEL_REVISION = "f3ff3571791e39611d31c381e3a41a3af07b4987"
MODEL_FILE = "kokoro-v1_0.pth"
CONFIG_FILE = "config.json"
VOICE_FILE = "voices/af_heart.pt"
MODEL_HASHES = {
    CONFIG_FILE: "5abb01e2403b072bf03d04fde160443e209d7a0dad49a423be15196b9b43c17f",
    MODEL_FILE: "496dba118d1a58f5f3db2efc88dbdc216e0483fc89fe6e47ee1f2c53f18ad1e4",
    VOICE_FILE: "0ab5709b8ffab19bfd849cd11d98f75b60af7733253ad0d67b12382a102cb4ff",
}
SAMPLE_RATE_HZ = 24_000
SPEED = 1.0
RNG_SEED = 20_260_716
MAX_PHONEME_CHARACTERS = 510

# torch.manual_seed controls process-global CPU RNG state. This lock is shared by
# every runtime instance so callers cannot accidentally interleave state
# generation or decoder excitation through two independently constructed models.
_INFERENCE_LOCK = threading.RLock()
_WORD_BOUNDARIES = frozenset(' ;:,.!?—…()[]{}"“”')


class KokoroSynthesisError(RuntimeError):
    """The controlled synthesis contract could not be satisfied."""


@dataclass(frozen=True)
class PairPlan:
    source_phonemes: str
    neutral_phonemes: str
    lens_phonemes: str
    target_word_indexes: tuple[int, ...]
    speed: float = SPEED


@dataclass(frozen=True)
class PairRender:
    neutral: np.ndarray
    lens: np.ndarray
    predicted_durations: tuple[int, ...]
    replaced_columns: tuple[int, ...]
    sample_rate_hz: int = SAMPLE_RATE_HZ


@dataclass(frozen=True)
class ParityRender:
    neutral: np.ndarray
    identity: np.ndarray
    lens: np.ndarray
    predicted_durations: tuple[int, ...]
    replaced_columns: tuple[int, ...]
    sample_rate_hz: int = SAMPLE_RATE_HZ


def verify_model_files(*, download: bool = False) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for filename, expected_hash in MODEL_HASHES.items():
        path = Path(
            hf_hub_download(
                repo_id=MODEL_REPO,
                revision=MODEL_REVISION,
                filename=filename,
                local_files_only=not download,
            )
        )
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            raise KokoroSynthesisError(
                f"Kokoro artifact hash mismatch for {filename}: {actual_hash}"
            )
        paths[filename] = path
    return paths


def pcm16_bytes(audio: np.ndarray) -> bytes:
    values = np.asarray(audio, dtype=np.float64).reshape(-1)
    if not values.size or not np.isfinite(values).all():
        raise KokoroSynthesisError("audio must contain finite samples")
    return np.rint(np.clip(values, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


def pcm_sha256(audio: np.ndarray) -> str:
    return hashlib.sha256(pcm16_bytes(audio)).hexdigest()


def _input_ids(model: Any, phonemes: str, torch: Any) -> Any:
    values = [model.vocab.get(symbol) for symbol in phonemes]
    values = [value for value in values if value is not None]
    return torch.LongTensor([[0, *values, 0]]).to(model.device)


def _text_features(
    model: Any, input_ids: Any, ref_s: Any, torch: Any
) -> dict[str, Any]:
    input_lengths = torch.full(
        (input_ids.shape[0],),
        input_ids.shape[-1],
        device=input_ids.device,
        dtype=torch.long,
    )
    text_mask = (
        torch.arange(input_lengths.max())
        .unsqueeze(0)
        .expand(input_lengths.shape[0], -1)
        .type_as(input_lengths)
    )
    text_mask = torch.gt(text_mask + 1, input_lengths.unsqueeze(1)).to(model.device)
    bert_dur = model.bert(input_ids, attention_mask=(~text_mask).int())
    d_en = model.bert_encoder(bert_dur).transpose(-1, -2)
    style = ref_s[:, 128:]
    d = model.predictor.text_encoder(d_en, style, input_lengths, text_mask)
    t_en = model.text_encoder(input_ids, input_lengths, text_mask)
    return {
        "input_lengths": input_lengths,
        "style": style,
        "d": d,
        "t_en": t_en,
    }


def _predicted_alignment(
    model: Any, features: dict[str, Any], speed: float, torch: Any
) -> tuple[Any, Any]:
    x, _ = model.predictor.lstm(features["d"])
    duration = model.predictor.duration_proj(x)
    pred_dur = (
        torch.round(torch.sigmoid(duration).sum(axis=-1) / speed)
        .clamp(min=1)
        .long()
        .squeeze()
    )
    indices = torch.repeat_interleave(
        torch.arange(features["input_lengths"].item(), device=model.device), pred_dur
    )
    alignment = torch.zeros(
        (features["input_lengths"].item(), indices.shape[0]), device=model.device
    )
    alignment[indices, torch.arange(indices.shape[0])] = 1
    return pred_dur, alignment.unsqueeze(0)


def _validate_projection(module: Any, values: Any) -> None:
    if values.ndim != 3 or values.shape[0] != 1:
        raise KokoroSynthesisError(
            "controlled F0/noise projection requires batch size one"
        )
    if module.weight.shape != (1, values.shape[1], 1) or module.bias.shape != (1,):
        raise KokoroSynthesisError("Kokoro F0/noise projection has an unexpected shape")
    if values.shape[1] % 2:
        raise KokoroSynthesisError("Kokoro F0/noise projection channels must be even")


def _project_f0(module: Any, values: Any, torch: Any) -> Any:
    """Reproduce frozen-v4 F0 projection with a fixed CPU reduction schedule.

    The pinned arm64 slow-Conv1d path used different vector kernels for its
    128-frame block, 64-frame block, and tail. Encoding that schedule explicitly
    removes cold-process dispatch variation while preserving the frozen PCM.
    """
    _validate_projection(module, values)
    weight = module.weight[0, :, 0]
    frames = values.shape[-1]
    offset = 0
    segments: list[Any] = []
    while frames - offset >= 128:
        end = offset + 128
        accumulator = module.bias[0].expand(128).clone()
        for channel in range(values.shape[1]):
            accumulator.add_(
                values[0, channel, offset:end], alpha=float(weight[channel])
            )
        segments.append(accumulator)
        offset = end
    if frames - offset >= 64:
        end = offset + 64
        even = module.bias[0].expand(64).clone()
        odd = torch.zeros_like(even)
        for channel in range(0, values.shape[1], 2):
            even.add_(values[0, channel, offset:end], alpha=float(weight[channel]))
            odd.add_(
                values[0, channel + 1, offset:end],
                alpha=float(weight[channel + 1]),
            )
        segments.append(even + odd)
        offset = end
    if offset < frames:
        tail = (
            torch.matmul(weight.view(1, -1), values[0, :, offset:]).reshape(-1)
            + module.bias[0]
        )
        segments.append(tail)
    return torch.cat(segments).unsqueeze(0)


def _project_noise(module: Any, values: Any, torch: Any) -> Any:
    """Apply the frozen-v4 noise projection through a stable matrix product."""
    _validate_projection(module, values)
    weight = module.weight[0, :, 0]
    projected = torch.matmul(weight.view(1, -1), values[0]).reshape(-1)
    return (projected + module.bias[0]).unsqueeze(0)


def _f0_noise(
    model: Any, features: dict[str, Any], alignment: Any, torch: Any
) -> tuple[Any, Any]:
    encoded = features["d"].transpose(-1, -2) @ alignment
    shared, _ = model.predictor.shared(encoded.transpose(-1, -2))
    f0 = shared.transpose(-1, -2)
    for block in model.predictor.F0:
        f0 = block(f0, features["style"])
    noise = shared.transpose(-1, -2)
    for block in model.predictor.N:
        noise = block(noise, features["style"])
    return (
        _project_f0(model.predictor.F0_proj, f0, torch),
        _project_noise(model.predictor.N_proj, noise, torch),
    )


def _filtered_symbols(model: Any, phonemes: str) -> tuple[str, ...]:
    return tuple(symbol for symbol in phonemes if model.vocab.get(symbol) is not None)


def _word_column_spans(model: Any, phonemes: str) -> tuple[tuple[int, ...], ...]:
    symbols = _filtered_symbols(model, phonemes)
    spans: list[tuple[int, ...]] = []
    current: list[int] = []
    for filtered_index, symbol in enumerate(symbols):
        if symbol in _WORD_BOUNDARIES:
            if current:
                spans.append(tuple(current))
                current = []
            continue
        # +1 accounts for the model's start boundary token.
        current.append(filtered_index + 1)
    if current:
        spans.append(tuple(current))
    return tuple(spans)


def target_word_columns(
    model: Any, phonemes: str, target_word_indexes: Sequence[int]
) -> tuple[int, ...]:
    spans = _word_column_spans(model, phonemes)
    requested = tuple(sorted(set(target_word_indexes)))
    if not requested:
        raise KokoroSynthesisError("at least one target word is required")
    if requested[0] < 0 or requested[-1] >= len(spans):
        raise KokoroSynthesisError(
            f"target word index is outside the {len(spans)}-word phoneme plan"
        )
    return tuple(sorted({column for index in requested for column in spans[index]}))


def _validate_plan(model: Any, plan: PairPlan) -> tuple[int, ...]:
    if plan.speed != SPEED:
        raise KokoroSynthesisError(f"only the frozen speed {SPEED} is supported")
    for label, phonemes in (
        ("source", plan.source_phonemes),
        ("neutral", plan.neutral_phonemes),
        ("lens", plan.lens_phonemes),
    ):
        if not phonemes or len(phonemes) > MAX_PHONEME_CHARACTERS:
            raise KokoroSynthesisError(
                f"{label} phoneme plan must contain 1-{MAX_PHONEME_CHARACTERS} characters"
            )
        unsupported = sorted(set(phonemes) - set(model.vocab))
        if unsupported:
            raise KokoroSynthesisError(
                f"{label} phoneme plan contains unsupported symbols: "
                f"{''.join(unsupported)}"
            )
    source = _filtered_symbols(model, plan.source_phonemes)
    neutral = _filtered_symbols(model, plan.neutral_phonemes)
    lens = _filtered_symbols(model, plan.lens_phonemes)
    if not (len(source) == len(neutral) == len(lens)):
        raise KokoroSynthesisError(
            "source, neutral, and lens plans must have equal model-token counts"
        )
    columns = target_word_columns(
        model, plan.neutral_phonemes, plan.target_word_indexes
    )
    changed = tuple(
        index + 1
        for index, (neutral_symbol, lens_symbol) in enumerate(
            zip(neutral, lens, strict=True)
        )
        if neutral_symbol != lens_symbol
    )
    if not changed:
        raise KokoroSynthesisError("neutral and lens plans do not differ")
    if not set(changed).issubset(columns):
        raise KokoroSynthesisError(
            "neutral/lens differences escape the complete target-word columns"
        )
    if not any(column in changed for column in columns):
        raise KokoroSynthesisError("target-word columns contain no lens change")
    return columns


class KokoroSynthesisRuntime:
    def __init__(self, model: Any, voice_pack: Any, torch: Any) -> None:
        self.model = model
        self.voice_pack = voice_pack
        self.torch = torch

    @classmethod
    def load(cls, *, download: bool = False) -> KokoroSynthesisRuntime:
        if importlib.metadata.version("kokoro") != KOKORO_VERSION:
            raise KokoroSynthesisError(
                f"Kokoro {KOKORO_VERSION} is required for controlled synthesis"
            )
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        files = verify_model_files(download=download)
        import torch
        from kokoro import KModel

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
            voice_pack = torch.load(
                files[VOICE_FILE], map_location="cpu", weights_only=True
            )
        return cls(model, voice_pack, torch)

    def _reference_style(self, source_phonemes: str) -> Any:
        style_index = len(source_phonemes) - 1
        if style_index < 0 or style_index >= len(self.voice_pack):
            raise KokoroSynthesisError(
                f"source phoneme length {len(source_phonemes)} has no frozen voice style"
            )
        ref_s = self.voice_pack[style_index]
        if ref_s.ndim == 1:
            ref_s = ref_s.unsqueeze(0)
        if ref_s.ndim != 2 or ref_s.shape[-1] < 256:
            raise KokoroSynthesisError("voice style has an unexpected shape")
        return ref_s

    def _decode(
        self,
        state: Any,
        alignment: Any,
        f0: Any,
        noise: Any,
        ref_s: Any,
    ) -> np.ndarray:
        self.torch.manual_seed(RNG_SEED)
        asr = state @ alignment
        audio = self.model.decoder(asr, f0, noise, ref_s[:, :128])
        return audio.squeeze().detach().cpu().numpy()

    def render_pair(self, plan: PairPlan) -> PairRender:
        rendered = self.render_parity_triplet(plan)
        return PairRender(
            neutral=rendered.neutral,
            lens=rendered.lens,
            predicted_durations=rendered.predicted_durations,
            replaced_columns=rendered.replaced_columns,
        )

    def render_parity_triplet(self, plan: PairPlan) -> ParityRender:
        # The lock intentionally covers plan validation, state generation, and
        # every decode. No global torch RNG consumer can interleave with a pair.
        with _INFERENCE_LOCK, self.torch.no_grad():
            columns = _validate_plan(self.model, plan)
            ref_s = self._reference_style(plan.source_phonemes)
            source_features = _text_features(
                self.model,
                _input_ids(self.model, plan.source_phonemes, self.torch),
                ref_s,
                self.torch,
            )
            pred_dur, alignment = _predicted_alignment(
                self.model, source_features, plan.speed, self.torch
            )
            neutral_features = _text_features(
                self.model,
                _input_ids(self.model, plan.neutral_phonemes, self.torch),
                ref_s,
                self.torch,
            )
            lens_features = _text_features(
                self.model,
                _input_ids(self.model, plan.lens_phonemes, self.torch),
                ref_s,
                self.torch,
            )
            f0, noise = _f0_noise(self.model, neutral_features, alignment, self.torch)
            neutral_state = neutral_features["t_en"]
            lens_state = neutral_state.clone()
            lens_state[:, :, list(columns)] = lens_features["t_en"][:, :, list(columns)]
            neutral = self._decode(neutral_state, alignment, f0, noise, ref_s)
            identity = self._decode(neutral_state, alignment, f0, noise, ref_s)
            lens = self._decode(lens_state, alignment, f0, noise, ref_s)

        if neutral.shape != identity.shape or neutral.shape != lens.shape:
            raise KokoroSynthesisError("controlled pair has unequal sample counts")
        if pcm16_bytes(neutral) != pcm16_bytes(identity):
            raise KokoroSynthesisError(
                "common-RNG neutral identity control is not bit-identical"
            )
        return ParityRender(
            neutral=neutral,
            identity=identity,
            lens=lens,
            predicted_durations=tuple(int(value) for value in pred_dur.cpu().tolist()),
            replaced_columns=columns,
        )
