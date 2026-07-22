from __future__ import annotations

import numpy as np

from earshift_bakeoff.bilingual_candidate_runtime import (
    _candidate_classification,
    _natural_anchor_classifications,
)
from earshift_bakeoff.bilingual_vowel_replicated_anchors import TRAINING_SEEDS


def _scaler() -> dict:
    return {
        "feature_size": 36,
        "observation_count": 120,
        "scale_floor": 0.05,
        "center": [0.0] * 36,
        "scale": [1.0] * 36,
    }


def _features(value: float) -> dict[int, list[float]]:
    return {
        seed: (np.full(36, value + offset, dtype=float)).tolist()
        for seed, offset in zip(TRAINING_SEEDS, (-0.01, 0.0, 0.01), strict=True)
    }


def test_current_context_classifier_accepts_clean_directed_endpoint() -> None:
    source = _features(0.0)
    target = _features(1.0)

    natural = _natural_anchor_classifications(
        source_features=source,
        target_features=target,
        scaler=_scaler(),
    )
    candidate = _candidate_classification(
        source_features=source,
        target_features=target,
        neutral_feature=[0.0] * 36,
        lens_feature=[0.8] * 36,
        scaler=_scaler(),
    )

    assert all(row["exact_category_pass"] for row in natural)
    assert candidate["exact_category_pass"] is True
    assert candidate["direction_cosine"] > 0.99


def test_current_context_classifier_rejects_identity_and_reverse_motion() -> None:
    source = _features(0.0)
    target = _features(1.0)

    identity = _candidate_classification(
        source_features=source,
        target_features=target,
        neutral_feature=[0.0] * 36,
        lens_feature=[0.0] * 36,
        scaler=_scaler(),
    )
    reverse = _candidate_classification(
        source_features=source,
        target_features=target,
        neutral_feature=[0.0] * 36,
        lens_feature=[-0.8] * 36,
        scaler=_scaler(),
    )

    assert identity["directional_pass"] is False
    assert reverse["directional_pass"] is False
    assert reverse["direction_cosine"] < -0.99
