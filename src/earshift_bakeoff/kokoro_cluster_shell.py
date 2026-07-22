from __future__ import annotations

from .kokoro_typed_engine import (
    SOURCE_PHONE,
    TARGET_PHONE,
    CarrierAssignment,
    KokoroTypedEngineError,
    KokoroTypedPlanner,
    MappingKey,
    _CONSONANT_SYMBOLS,
    _surface_for,
)
from .kokoro_validated_shell import ValidatedShellPlanner


CLUSTER_SHELL_VERSION = 4

# Rung-1 scope is stressed monosyllables with a single-consonant onset and a
# coda cluster. The coda must be voiceless (voiced codas are the lengthening
# context behind the frozen neutral-category failure) and s+stop keeps the
# clusters English-legal. The v2 shell vˈæs was lexically saturated — vast,
# vest, vasp, and a vask homophone all collide with the pinned word/phone
# gates — so v3 pinned the onset to ʒ. This is a receipt about the tested
# forms, not a claim that English has no word-initial /ʒ/. Onset clusters
# remain out of scope until a rung-2 design with onset-legal pinning.
#
# v4 shrinks the extra pool from (t, k, p) to (t, p): the frozen v3
# calibration failed only its /sk/ cell, where the /ɛ/-minus-/ae/ F2
# component is stably negative at every ceiling — a coda-conditioned
# realization, not noise. Pool membership is a planner design freedom, and
# anchors only need to cover what the planner can emit, so v4 excludes the
# velar extra by design rather than carrying an anchor-less context. The
# frozen v3 aggregate stays failed; nothing is rescued or reclassified.
CLUSTER_NEUTRAL_SHELL = "ʒˈæs"
CLUSTER_LENS_SHELL = "ʒˈɛs"
CLUSTER_LEFT_CONSONANT = "ʒ"
CLUSTER_RIGHT_CONSONANT = "s"
CLUSTER_EXTRA_CONSONANTS = ("t", "p")


class ClusterShellPlanner(ValidatedShellPlanner):
    """Coda-cluster planner with a voiceless, category-preserving target shell."""

    @classmethod
    def load(cls) -> ClusterShellPlanner:
        base = ValidatedShellPlanner.load()
        return cls(
            g2p=base.g2p,
            model_vocab=base.model_vocab,
            source_analyzer=base.source_analyzer,
            nonce_checker=base.nonce_checker,
            phone_index=base.phone_index,
            rules_path=base.rules_path,
        )

    def _candidate(self, key: MappingKey, attempt: int) -> CarrierAssignment:
        base = KokoroTypedPlanner._candidate(self, key, attempt)
        if not key.target_offsets:
            return base
        if len(key.target_offsets) != 1:
            raise KokoroTypedEngineError(
                "cluster_shell_unsupported_target_shape",
                "The cluster shell supports one target per source word.",
            )
        target = key.target_offsets[0]
        phone = key.source_phone
        onset = phone[: target - 1]
        coda = phone[target + 1 :]
        if (
            target < 2
            or phone[target - 1] not in {"ˈ", "ˌ"}
            or len(onset) != 1
            or onset not in _CONSONANT_SYMBOLS
            or len(coda) < 2
            or any(symbol not in _CONSONANT_SYMBOLS for symbol in coda)
        ):
            raise KokoroTypedEngineError(
                "cluster_shell_unsupported_target_shape",
                "This version supports /ae/ only in C-stress-vowel-CC+ words.",
            )
        neutral = list(base.neutral_phone)
        lens = list(base.lens_phone)
        neutral[target - 2] = lens[target - 2] = CLUSTER_LEFT_CONSONANT
        neutral[target] = SOURCE_PHONE
        lens[target] = TARGET_PHONE
        neutral[target + 1] = lens[target + 1] = CLUSTER_RIGHT_CONSONANT
        for order, position in enumerate(range(target + 2, len(phone))):
            selected = CLUSTER_EXTRA_CONSONANTS[
                (attempt + order) % len(CLUSTER_EXTRA_CONSONANTS)
            ]
            neutral[position] = selected
            lens[position] = selected
        neutral_phone = "".join(neutral)
        lens_phone = "".join(lens)
        if (
            neutral_phone[target - 2 : target + 2] != CLUSTER_NEUTRAL_SHELL
            or lens_phone[target - 2 : target + 2] != CLUSTER_LENS_SHELL
            or len(neutral_phone) != len(phone)
        ):
            raise KokoroTypedEngineError(
                "cluster_shell_drift", "The cluster target shell drifted."
            )
        return CarrierAssignment(
            neutral_surface=_surface_for(neutral_phone),
            lens_surface=_surface_for(lens_phone),
            neutral_phone=neutral_phone,
            lens_phone=lens_phone,
            candidate_attempt=attempt,
        )
