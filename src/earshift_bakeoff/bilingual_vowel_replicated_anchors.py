from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .controlled_listener_synthesis import NaturalConditionRender
from .kokoro_synthesis import (
    MAX_PHONEME_CHARACTERS,
    SAMPLE_RATE_HZ,
    SPEED,
    KokoroSynthesisError,
    KokoroSynthesisRuntime,
    _f0_noise,
    _INFERENCE_LOCK,
    _input_ids,
    _predicted_alignment,
    _text_features,
)


REPLICATED_ANCHOR_VERSION = "bilingual-vowel-replicated-anchors-v1"
TRAINING_SEEDS = (202_607_171, 202_607_172, 202_607_173)
MINIMUM_EXACT_SEED_PAIRS_PER_OCCURRENCE = 2
MAXIMUM_REVERSED_SEED_PAIRS_PER_OCCURRENCE = 0


@dataclass(frozen=True)
class SeededNaturalConditionRender:
    audio_by_seed: dict[int, np.ndarray]
    predicted_durations: tuple[int, ...]
    f0: np.ndarray
    sample_rate_hz: int = SAMPLE_RATE_HZ
    version: str = REPLICATED_ANCHOR_VERSION


def validate_seed_order(seeds: Sequence[int]) -> tuple[int, ...]:
    values = tuple(seeds)
    if (
        not values
        or len(set(values)) != len(values)
        or any(
            isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63
            for seed in values
        )
    ):
        raise ValueError("decoder seeds must be unique integers in frozen order")
    return values


def render_seeded_natural_conditions(
    runtime: KokoroSynthesisRuntime,
    *,
    phonemes: str,
    reference_phonemes: str,
    seeds: Sequence[int],
) -> SeededNaturalConditionRender:
    """Render one fully conditioned natural anchor under fixed decoder seeds.

    Text state, duration, F0, and predicted noise are computed once. Only the
    decoder's process-global random seed varies, and the shared inference lock
    covers every decode.
    """

    seed_order = validate_seed_order(seeds)
    if not phonemes or len(phonemes) > MAX_PHONEME_CHARACTERS:
        raise KokoroSynthesisError(
            f"phoneme plan must contain 1-{MAX_PHONEME_CHARACTERS} characters"
        )
    unsupported = sorted(set(phonemes) - set(runtime.model.vocab))
    if unsupported:
        raise KokoroSynthesisError(
            "phoneme plan contains unsupported symbols: " + "".join(unsupported)
        )
    torch = runtime.torch
    audio_by_seed: dict[int, np.ndarray] = {}
    with _INFERENCE_LOCK, torch.no_grad():
        ref_s = runtime._reference_style(reference_phonemes)
        features = _text_features(
            runtime.model,
            _input_ids(runtime.model, phonemes, torch),
            ref_s,
            torch,
        )
        durations, alignment = _predicted_alignment(
            runtime.model, features, SPEED, torch
        )
        f0, noise = _f0_noise(runtime.model, features, alignment, torch)
        asr = features["t_en"] @ alignment
        for seed in seed_order:
            torch.manual_seed(seed)
            audio = runtime.model.decoder(asr, f0, noise, ref_s[:, :128])
            values = audio.squeeze().detach().cpu().numpy()
            if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
                raise KokoroSynthesisError(
                    f"seeded natural decoder returned invalid audio for seed {seed}"
                )
            audio_by_seed[seed] = values
    return SeededNaturalConditionRender(
        audio_by_seed=audio_by_seed,
        predicted_durations=tuple(int(value) for value in durations.cpu().tolist()),
        f0=f0.detach().cpu().numpy(),
    )


def baseline_natural_render(
    seeded: SeededNaturalConditionRender,
    *,
    seed: int,
) -> NaturalConditionRender:
    try:
        audio = seeded.audio_by_seed[seed]
    except KeyError as exc:
        raise ValueError("baseline seed was not rendered") from exc
    return NaturalConditionRender(
        audio=audio,
        predicted_durations=seeded.predicted_durations,
        f0=seeded.f0,
    )


def aggregate_replicated_anchor_occurrence(
    *,
    natural_seed_records: Sequence[dict[str, Any]],
    candidate_record: dict[str, Any] | None,
) -> dict[str, Any]:
    if len(natural_seed_records) != len(TRAINING_SEEDS):
        raise ValueError("every frozen natural anchor seed must be evaluated")
    exact_count = sum(
        bool(record["exact_category_pass"]) for record in natural_seed_records
    )
    directional_count = sum(
        bool(record["directional_pass"]) for record in natural_seed_records
    )
    reversed_count = sum(
        float(record["direction_cosine"]) < 0.0 for record in natural_seed_records
    )
    anchor_validation_pass = bool(
        exact_count >= MINIMUM_EXACT_SEED_PAIRS_PER_OCCURRENCE
        and reversed_count <= MAXIMUM_REVERSED_SEED_PAIRS_PER_OCCURRENCE
    )
    if candidate_record is None:
        classification = "not_evaluated"
    elif not anchor_validation_pass:
        classification = "anchor_validation_fail"
    else:
        classification = str(candidate_record["classification"])
    return {
        "natural_seed_pair_count": len(natural_seed_records),
        "natural_exact_seed_pair_count": exact_count,
        "natural_directional_seed_pair_count": directional_count,
        "natural_reversed_seed_pair_count": reversed_count,
        "minimum_natural_exact_seed_pair_count": (
            MINIMUM_EXACT_SEED_PAIRS_PER_OCCURRENCE
        ),
        "maximum_natural_reversed_seed_pair_count": (
            MAXIMUM_REVERSED_SEED_PAIRS_PER_OCCURRENCE
        ),
        "anchor_validation_pass": anchor_validation_pass,
        "candidate_evaluated": candidate_record is not None,
        "classification": classification,
        "directional_pass": classification
        in ("exact_category_pass", "directional_only_pass"),
        "exact_category_pass": classification == "exact_category_pass",
    }


def aggregate_replicated_anchor_cell(
    occurrence_records: Sequence[dict[str, Any]],
    *,
    expected_occurrence_count: int = 4,
) -> dict[str, Any]:
    if len(occurrence_records) != expected_occurrence_count:
        raise ValueError("replicated anchor cell requires every frozen occurrence")
    anchor_pass_count = sum(
        bool(record["anchor_validation_pass"]) for record in occurrence_records
    )
    candidate_records = [
        record for record in occurrence_records if record["candidate_evaluated"]
    ]
    if not candidate_records:
        classification = "anchors_only"
    elif len(candidate_records) != expected_occurrence_count:
        raise ValueError("candidate coverage must be complete or absent per cell")
    elif anchor_pass_count != expected_occurrence_count:
        classification = "anchor_validation_fail"
    elif all(record["exact_category_pass"] for record in candidate_records):
        classification = "exact_category_pass"
    elif all(record["directional_pass"] for record in candidate_records):
        classification = "directional_only_pass"
    else:
        classification = "fail"
    return {
        "expected_occurrence_count": expected_occurrence_count,
        "anchor_valid_occurrence_count": anchor_pass_count,
        "all_anchor_occurrences_valid": anchor_pass_count == expected_occurrence_count,
        "candidate_occurrence_count": len(candidate_records),
        "classification": classification,
        "directional_pass": classification
        in ("exact_category_pass", "directional_only_pass"),
        "exact_category_pass": classification == "exact_category_pass",
    }
