from __future__ import annotations

import hashlib
import math
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Literal, Protocol, Sequence

from .audio_conformance import (
    FLOW_DEVELOPER_PROMPT,
    AudioTiming,
    analyze_audio_timing,
    check_transcript,
)
from .config import stable_json
from .models import RenderResult
from .util import sha256_file


Side = Literal["neutral", "lens"]


@dataclass(frozen=True)
class PairingThresholds:
    max_relative_duration_difference: float = 0.03
    max_pause_position_difference: float = 0.06
    pause_duration_scale: float = 0.05
    duration_weight: float = 0.50
    pause_position_weight: float = 0.30
    pause_duration_weight: float = 0.20
    calibration_status: str = "provisional_pending_empirical_calibration"
    algorithm_version: str = "matched-pair-v1"

    def __post_init__(self) -> None:
        positive = (
            self.max_relative_duration_difference,
            self.max_pause_position_difference,
            self.pause_duration_scale,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("Pairing thresholds must be positive.")
        weight_sum = (
            self.duration_weight
            + self.pause_position_weight
            + self.pause_duration_weight
        )
        if not math.isclose(weight_sum, 1.0, abs_tol=1e-9):
            raise ValueError("Pairing score weights must sum to 1.0.")


@dataclass(frozen=True)
class TakeCandidate:
    side: Side
    take_index: int
    audio_sha256: str
    renderer_model: str
    voice: str
    prompt_contract_fingerprint: str
    transcript_exact: bool
    integrity_ok: bool
    timing: AudioTiming
    failure_detail: str = ""
    audio_path: str = ""
    request_id: str = ""
    resolved_model: str = ""

    def to_record(self) -> dict:
        payload = asdict(self)
        payload["timing"] = self.timing.to_result_fields()
        return payload


@dataclass(frozen=True)
class PairScore:
    neutral_take_index: int
    lens_take_index: int
    neutral_audio_sha256: str
    lens_audio_sha256: str
    qualified: bool
    failure_reasons: tuple[str, ...]
    relative_duration_difference: float
    pause_count_equal: bool
    max_pause_position_difference: float
    mean_pause_position_difference: float
    relative_pause_duration_difference: float
    score: float


@dataclass(frozen=True)
class PairSelection:
    mode: Literal["live", "curated"]
    selected: PairScore
    neutral_takes: tuple[TakeCandidate, ...]
    lens_takes: tuple[TakeCandidate, ...]
    all_pair_scores: tuple[PairScore, ...]
    thresholds: PairingThresholds
    retry_side: Side | None
    total_renders: int

    @property
    def degraded(self) -> bool:
        return not self.selected.qualified

    def to_record(self) -> dict:
        return {
            "mode": self.mode,
            "algorithm_version": self.thresholds.algorithm_version,
            "thresholds": asdict(self.thresholds),
            "total_renders": self.total_renders,
            "retry_side": self.retry_side,
            "degraded": self.degraded,
            "selected_pair": asdict(self.selected),
            "candidate_takes": [
                take.to_record() for take in (*self.neutral_takes, *self.lens_takes)
            ],
            "pair_scores": [asdict(score) for score in self.all_pair_scores],
        }


def _relative_difference(left: float, right: float) -> float:
    denominator = max((left + right) / 2, 1e-9)
    return abs(left - right) / denominator


def _pause_midpoints(timing: AudioTiming) -> tuple[float, ...]:
    return tuple(
        (pause.start_fraction + pause.end_fraction) / 2
        for pause in timing.interior_pauses
    )


def evaluate_pair(
    neutral: TakeCandidate,
    lens: TakeCandidate,
    thresholds: PairingThresholds | None = None,
) -> PairScore:
    thresholds = thresholds or PairingThresholds()
    if neutral.side != "neutral" or lens.side != "lens":
        raise ValueError("Pair evaluation requires neutral then lens candidates.")

    failures: list[str] = []
    if not neutral.integrity_ok or not lens.integrity_ok:
        failures.append("integrity_failure")
    if not neutral.transcript_exact or not lens.transcript_exact:
        failures.append("transcript_mismatch")
    if neutral.renderer_model != lens.renderer_model:
        failures.append("renderer_mismatch")
    if neutral.voice != lens.voice:
        failures.append("voice_mismatch")
    if neutral.prompt_contract_fingerprint != lens.prompt_contract_fingerprint:
        failures.append("prompt_contract_mismatch")

    duration_difference = _relative_difference(
        neutral.timing.utterance_duration_s,
        lens.timing.utterance_duration_s,
    )
    if duration_difference > thresholds.max_relative_duration_difference:
        failures.append("duration_difference")

    neutral_positions = _pause_midpoints(neutral.timing)
    lens_positions = _pause_midpoints(lens.timing)
    pause_count_equal = len(neutral_positions) == len(lens_positions)
    if not pause_count_equal:
        failures.append("pause_count_mismatch")
        position_differences = (1.0,)
    else:
        position_differences = tuple(
            abs(left - right)
            for left, right in zip(neutral_positions, lens_positions)
        )
    max_position_difference = max(position_differences, default=0.0)
    mean_position_difference = (
        statistics.mean(position_differences) if position_differences else 0.0
    )
    if max_position_difference > thresholds.max_pause_position_difference:
        failures.append("pause_position_difference")

    pause_duration_difference = _relative_difference(
        neutral.timing.interior_pause_s,
        lens.timing.interior_pause_s,
    )
    duration_component = (
        duration_difference / thresholds.max_relative_duration_difference
    )
    position_component = (
        mean_position_difference / thresholds.max_pause_position_difference
    )
    pause_duration_component = (
        pause_duration_difference / thresholds.pause_duration_scale
    )
    score = (
        thresholds.duration_weight * duration_component
        + thresholds.pause_position_weight * position_component
        + thresholds.pause_duration_weight * pause_duration_component
    )
    fundamental_failures = {
        "integrity_failure",
        "transcript_mismatch",
        "renderer_mismatch",
        "voice_mismatch",
        "prompt_contract_mismatch",
    }
    score += 100 * len(fundamental_failures.intersection(failures))
    if not pause_count_equal:
        score += 10

    return PairScore(
        neutral_take_index=neutral.take_index,
        lens_take_index=lens.take_index,
        neutral_audio_sha256=neutral.audio_sha256,
        lens_audio_sha256=lens.audio_sha256,
        qualified=not failures,
        failure_reasons=tuple(failures),
        relative_duration_difference=round(duration_difference, 6),
        pause_count_equal=pause_count_equal,
        max_pause_position_difference=round(max_position_difference, 6),
        mean_pause_position_difference=round(mean_position_difference, 6),
        relative_pause_duration_difference=round(pause_duration_difference, 6),
        score=round(score, 6),
    )


def _score_sort_key(score: PairScore) -> tuple:
    return (
        not score.qualified,
        score.score,
        score.neutral_take_index,
        score.lens_take_index,
        score.neutral_audio_sha256,
        score.lens_audio_sha256,
    )


def select_best_pair(
    neutral_takes: Sequence[TakeCandidate],
    lens_takes: Sequence[TakeCandidate],
    thresholds: PairingThresholds | None = None,
) -> tuple[PairScore, tuple[PairScore, ...]]:
    thresholds = thresholds or PairingThresholds()
    if not neutral_takes or not lens_takes:
        raise ValueError("Pair selection requires at least one take on each side.")
    scores = tuple(
        evaluate_pair(neutral, lens, thresholds)
        for neutral in neutral_takes
        for lens in lens_takes
    )
    return min(scores, key=_score_sort_key), scores


def choose_retry_side(
    neutral_takes: Sequence[TakeCandidate],
    lens_takes: Sequence[TakeCandidate],
    scores: Sequence[PairScore],
) -> Side:
    invalid_by_side = {
        "neutral": sum(
            not take.integrity_ok or not take.transcript_exact for take in neutral_takes
        ),
        "lens": sum(
            not take.integrity_ok or not take.transcript_exact for take in lens_takes
        ),
    }
    if invalid_by_side["neutral"] != invalid_by_side["lens"]:
        return max(invalid_by_side, key=invalid_by_side.get)  # type: ignore[return-value]

    neutral_best = [
        min(score.score for score in scores if score.neutral_take_index == take.take_index)
        for take in neutral_takes
    ]
    lens_best = [
        min(score.score for score in scores if score.lens_take_index == take.take_index)
        for take in lens_takes
    ]
    neutral_mean = statistics.mean(neutral_best)
    lens_mean = statistics.mean(lens_best)
    if not math.isclose(neutral_mean, lens_mean, abs_tol=1e-9):
        return "neutral" if neutral_mean > lens_mean else "lens"

    def duration_spread(takes: Sequence[TakeCandidate]) -> float:
        durations = [take.timing.utterance_duration_s for take in takes]
        return (max(durations) - min(durations)) / max(statistics.mean(durations), 1e-9)

    neutral_spread = duration_spread(neutral_takes)
    lens_spread = duration_spread(lens_takes)
    if not math.isclose(neutral_spread, lens_spread, abs_tol=1e-9):
        return "neutral" if neutral_spread > lens_spread else "lens"
    return "lens"


RenderTake = Callable[[Side, int], TakeCandidate]


class ScriptAudioRenderer(Protocol):
    model: str

    def render(
        self, script: str, instruction: str, voice: str, output: Path
    ) -> RenderResult: ...


def _empty_timing() -> AudioTiming:
    return AudioTiming(
        duration_s=0.0,
        sample_rate_hz=0,
        decoded_sample_count=0,
        clipped_fraction=1.0,
        utterance_duration_s=0.0,
        estimated_syllables_per_second=None,
        interior_pause_count=0,
        interior_pause_s=0.0,
        interior_pauses=(),
    )


class MatchedTakeFactory:
    """Adapter from a script renderer to provenance-bearing pair candidates."""

    def __init__(
        self,
        *,
        renderer: ScriptAudioRenderer,
        neutral_script: str,
        lens_script: str,
        delivery: str,
        voice: str,
        output_dir: Path,
        max_clipped_fraction: float = 0.001,
        max_duration_s: float = 45.0,
    ) -> None:
        self.renderer = renderer
        self.scripts = {"neutral": neutral_script, "lens": lens_script}
        self.delivery = delivery
        self.voice = voice
        self.output_dir = output_dir
        self.max_clipped_fraction = max_clipped_fraction
        self.max_duration_s = max_duration_s
        contract = {
            "model": renderer.model,
            "voice": voice,
            "delivery": delivery,
            "developer_prompt": FLOW_DEVELOPER_PROMPT,
            "protocol": "json-flow-v2",
        }
        self.prompt_contract_fingerprint = hashlib.sha256(
            stable_json(contract).encode("utf-8")
        ).hexdigest()

    def __call__(self, side: Side, take_index: int) -> TakeCandidate:
        script = self.scripts[side]
        output = self.output_dir / f"{side}__take-{take_index:02d}.wav"
        try:
            result = self.renderer.render(
                script, self.delivery, self.voice, output
            )
            timing = analyze_audio_timing(output, intended_syllables=None)
            transcript_exact = check_transcript(
                script, result.provider_transcript or ""
            ).exact_token_match
            integrity_ok = (
                result.status == "ok"
                and timing.decoded_sample_count > 0
                and 0 < timing.duration_s <= self.max_duration_s
                and timing.clipped_fraction < self.max_clipped_fraction
            )
            return TakeCandidate(
                side=side,
                take_index=take_index,
                audio_sha256=sha256_file(output),
                renderer_model=result.renderer_model,
                voice=self.voice,
                prompt_contract_fingerprint=self.prompt_contract_fingerprint,
                transcript_exact=transcript_exact,
                integrity_ok=integrity_ok,
                timing=timing,
                failure_detail=result.error_detail or "",
                audio_path=str(output),
                request_id=result.request_id or "",
                resolved_model=result.resolved_model or result.renderer_model,
            )
        except Exception as exc:
            output.unlink(missing_ok=True)
            return TakeCandidate(
                side=side,
                take_index=take_index,
                audio_sha256="",
                renderer_model=self.renderer.model,
                voice=self.voice,
                prompt_contract_fingerprint=self.prompt_contract_fingerprint,
                transcript_exact=False,
                integrity_ok=False,
                timing=_empty_timing(),
                failure_detail=f"{type(exc).__name__}: {str(exc)[:400]}",
                audio_path="",
                resolved_model=self.renderer.model,
            )


def _render_checked(render_take: RenderTake, side: Side, index: int) -> TakeCandidate:
    candidate = render_take(side, index)
    if candidate.side != side or candidate.take_index != index:
        raise ValueError("Take renderer returned mismatched side or index metadata.")
    return candidate


def render_live_selection(
    render_take: RenderTake,
    thresholds: PairingThresholds | None = None,
) -> PairSelection:
    thresholds = thresholds or PairingThresholds()
    neutral = [_render_checked(render_take, "neutral", index) for index in (1, 2)]
    lens = [_render_checked(render_take, "lens", index) for index in (1, 2)]
    selected, scores = select_best_pair(neutral, lens, thresholds)
    retry_side: Side | None = None
    if not selected.qualified:
        retry_side = choose_retry_side(neutral, lens, scores)
        target = neutral if retry_side == "neutral" else lens
        target.append(_render_checked(render_take, retry_side, 3))
        selected, scores = select_best_pair(neutral, lens, thresholds)
    total_renders = len(neutral) + len(lens)
    if total_renders > 5:
        raise RuntimeError("Live matched-pair selection exceeded five renders.")
    return PairSelection(
        mode="live",
        selected=selected,
        neutral_takes=tuple(neutral),
        lens_takes=tuple(lens),
        all_pair_scores=scores,
        thresholds=thresholds,
        retry_side=retry_side,
        total_renders=total_renders,
    )


def render_curated_selection(
    render_take: RenderTake,
    thresholds: PairingThresholds | None = None,
) -> PairSelection:
    thresholds = thresholds or PairingThresholds()
    neutral = [
        _render_checked(render_take, "neutral", index) for index in range(1, 5)
    ]
    lens = [_render_checked(render_take, "lens", index) for index in range(1, 5)]
    selected, scores = select_best_pair(neutral, lens, thresholds)
    return PairSelection(
        mode="curated",
        selected=selected,
        neutral_takes=tuple(neutral),
        lens_takes=tuple(lens),
        all_pair_scores=scores,
        thresholds=thresholds,
        retry_side=None,
        total_renders=8,
    )
