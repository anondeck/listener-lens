from __future__ import annotations

import math
import struct
import wave
from pathlib import Path

from earshift_bakeoff.audio_conformance import AudioTiming, PauseInterval
from earshift_bakeoff.matched_pairs import (
    MatchedTakeFactory,
    PairingThresholds,
    TakeCandidate,
    evaluate_pair,
    render_curated_selection,
    render_live_selection,
    select_best_pair,
)
from earshift_bakeoff.models import RenderResult


def timing(
    duration: float,
    pauses: tuple[tuple[float, float], ...] = ((0.48, 0.52),),
) -> AudioTiming:
    intervals = tuple(
        PauseInterval(
            start_s=round(start * duration, 3),
            end_s=round(end * duration, 3),
            duration_s=round((end - start) * duration, 3),
            start_fraction=start,
            end_fraction=end,
        )
        for start, end in pauses
    )
    return AudioTiming(
        duration_s=duration,
        sample_rate_hz=24000,
        decoded_sample_count=round(duration * 24000),
        clipped_fraction=0.0,
        utterance_duration_s=duration,
        estimated_syllables_per_second=None,
        interior_pause_count=len(intervals),
        interior_pause_s=round(sum(pause.duration_s for pause in intervals), 3),
        interior_pauses=intervals,
    )


def take(
    side: str,
    index: int,
    duration: float,
    *,
    transcript_exact: bool = True,
    integrity_ok: bool = True,
    pauses: tuple[tuple[float, float], ...] = ((0.48, 0.52),),
    prompt: str = "flow-v2-fingerprint",
) -> TakeCandidate:
    return TakeCandidate(
        side=side,  # type: ignore[arg-type]
        take_index=index,
        audio_sha256=f"{side}-{index}-{duration}",
        renderer_model="gpt-audio-1.5",
        voice="marin",
        prompt_contract_fingerprint=prompt,
        transcript_exact=transcript_exact,
        integrity_ok=integrity_ok,
        timing=timing(duration, pauses),
    )


def test_pair_qualifies_on_exact_transcripts_timing_and_pause_alignment() -> None:
    score = evaluate_pair(take("neutral", 1, 10.0), take("lens", 1, 10.2))

    assert score.qualified
    assert score.failure_reasons == ()
    assert score.relative_duration_difference < 0.03
    assert score.pause_count_equal


def test_default_threshold_is_explicitly_provisional() -> None:
    thresholds = PairingThresholds()

    assert thresholds.max_relative_duration_difference == 0.03
    assert thresholds.calibration_status == "provisional_pending_empirical_calibration"


def test_pair_rejects_duration_pause_count_and_pause_location_failures() -> None:
    duration = evaluate_pair(take("neutral", 1, 10), take("lens", 1, 11))
    count = evaluate_pair(
        take("neutral", 1, 10), take("lens", 1, 10, pauses=())
    )
    location = evaluate_pair(
        take("neutral", 1, 10),
        take("lens", 1, 10, pauses=((0.70, 0.74),)),
    )

    assert "duration_difference" in duration.failure_reasons
    assert "pause_count_mismatch" in count.failure_reasons
    assert "pause_position_difference" in location.failure_reasons


def test_pair_rejects_renderer_contract_drift() -> None:
    score = evaluate_pair(
        take("neutral", 1, 10, prompt="contract-a"),
        take("lens", 1, 10, prompt="contract-b"),
    )

    assert not score.qualified
    assert "prompt_contract_mismatch" in score.failure_reasons


def test_pair_tie_breaking_is_deterministic() -> None:
    selected, scores = select_best_pair(
        [take("neutral", 2, 10), take("neutral", 1, 10)],
        [take("lens", 2, 10), take("lens", 1, 10)],
    )

    assert len(scores) == 4
    assert (selected.neutral_take_index, selected.lens_take_index) == (1, 1)


def test_live_selection_stops_at_four_when_initial_pair_qualifies() -> None:
    calls: list[tuple[str, int]] = []

    def render(side, index):
        calls.append((side, index))
        return take(side, index, 10 + index * 0.01)

    selection = render_live_selection(render)

    assert selection.total_renders == 4
    assert selection.retry_side is None
    assert selection.selected.qualified
    assert len(calls) == 4


def test_live_selection_retries_only_the_worse_side_and_caps_at_five() -> None:
    calls: list[tuple[str, int]] = []

    def render(side, index):
        calls.append((side, index))
        transcript_exact = not (side == "lens" and index < 3)
        return take(
            side,
            index,
            10 + index * 0.01,
            transcript_exact=transcript_exact,
        )

    selection = render_live_selection(render)

    assert selection.total_renders == 5
    assert selection.retry_side == "lens"
    assert calls == [
        ("neutral", 1),
        ("neutral", 2),
        ("lens", 1),
        ("lens", 2),
        ("lens", 3),
    ]
    assert selection.selected.qualified
    assert len(selection.all_pair_scores) == 6


def test_live_selection_returns_honest_degraded_pair_after_bounded_retry() -> None:
    def render(side, index):
        return take(side, index, 10, transcript_exact=False)

    selection = render_live_selection(render)

    assert selection.total_renders == 5
    assert selection.degraded
    assert "transcript_mismatch" in selection.selected.failure_reasons


def test_curated_selection_renders_four_per_side_and_scores_all_pairs() -> None:
    calls: list[tuple[str, int]] = []

    def render(side, index):
        calls.append((side, index))
        return take(side, index, 10 + index * 0.01)

    selection = render_curated_selection(render)
    record = selection.to_record()

    assert selection.total_renders == 8
    assert len(selection.neutral_takes) == 4
    assert len(selection.lens_takes) == 4
    assert len(selection.all_pair_scores) == 16
    assert len(calls) == 8
    assert len(record["candidate_takes"]) == 8
    assert len(record["pair_scores"]) == 16
    assert record["thresholds"]["calibration_status"].startswith("provisional")


class FakeAudioRenderer:
    model = "gpt-audio-1.5"

    def render(
        self, script: str, instruction: str, voice: str, output: Path
    ) -> RenderResult:
        assert instruction == "Natural connected English."
        assert voice == "marin"
        output.parent.mkdir(parents=True, exist_ok=True)
        rate = 24000
        samples = [
            int(5000 * math.sin(2 * math.pi * 220 * index / rate))
            for index in range(rate)
        ]
        with wave.open(str(output), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(rate)
            wav.writeframes(struct.pack(f"<{len(samples)}h", *samples))
        return RenderResult(
            renderer_slug=self.model,
            renderer_model=self.model,
            status="ok",
            output_path=str(output),
            request_id="req-test",
            resolved_model=self.model,
            provider_transcript=script,
        )


def test_take_factory_records_exactness_integrity_timing_and_provenance(
    tmp_path: Path,
) -> None:
    factory = MatchedTakeFactory(
        renderer=FakeAudioRenderer(),
        neutral_script="naver pellik.",
        lens_script="never pellik.",
        delivery="Natural connected English.",
        voice="marin",
        output_dir=tmp_path,
    )

    candidate = factory("neutral", 1)

    assert candidate.transcript_exact
    assert candidate.integrity_ok
    assert candidate.timing.duration_s == 1.0
    assert candidate.audio_sha256
    assert candidate.request_id == "req-test"
    assert candidate.resolved_model == "gpt-audio-1.5"
    assert len(candidate.prompt_contract_fingerprint) == 64
    assert Path(candidate.audio_path).is_file()


def test_take_factory_turns_renderer_failure_into_a_rejected_candidate(
    tmp_path: Path,
) -> None:
    class FailingRenderer(FakeAudioRenderer):
        def render(self, script, instruction, voice, output):
            raise RuntimeError("provider rejected test request")

    factory = MatchedTakeFactory(
        renderer=FailingRenderer(),
        neutral_script="naver pellik.",
        lens_script="never pellik.",
        delivery="Natural connected English.",
        voice="marin",
        output_dir=tmp_path,
    )

    candidate = factory("lens", 1)

    assert not candidate.transcript_exact
    assert not candidate.integrity_ok
    assert candidate.audio_sha256 == ""
    assert candidate.timing.decoded_sample_count == 0
    assert candidate.failure_detail.startswith("RuntimeError:")
