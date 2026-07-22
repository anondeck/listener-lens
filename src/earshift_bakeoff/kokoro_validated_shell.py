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


VALIDATED_SHELL_VERSION = 1
VALIDATED_LEFT_CONSONANT = "v"
VALIDATED_RIGHT_CONSONANT = "ʒ"
VALIDATED_NEUTRAL_SHELL = "vˈæʒ"
VALIDATED_LENS_SHELL = "vˈɛʒ"


class ValidatedShellPlanner(KokoroTypedPlanner):
    """Candidate planner that fixes only the immediate context of /ae/ targets."""

    @classmethod
    def load(cls) -> ValidatedShellPlanner:
        base = KokoroTypedPlanner.load()
        return cls(
            g2p=base.g2p,
            model_vocab=base.model_vocab,
            source_analyzer=base.source_analyzer,
            nonce_checker=base.nonce_checker,
            phone_index=base.phone_index,
            rules_path=base.rules_path,
        )

    def _candidate(self, key: MappingKey, attempt: int) -> CarrierAssignment:
        candidate = super()._candidate(key, attempt)
        if not key.target_offsets:
            return candidate
        if len(key.target_offsets) != 1:
            raise KokoroTypedEngineError(
                "validated_shell_unsupported_target_shape",
                "The validated carrier shell supports one target per source word.",
            )
        target = key.target_offsets[0]
        if (
            target < 2
            or target + 1 >= len(key.source_phone)
            or key.source_phone[target - 1] not in {"ˈ", "ˌ"}
            or key.source_phone[target - 2] not in _CONSONANT_SYMBOLS
            or key.source_phone[target + 1] not in _CONSONANT_SYMBOLS
        ):
            raise KokoroTypedEngineError(
                "validated_shell_unsupported_target_shape",
                "The target lacks the frozen C-stress-vowel-C shell.",
            )
        neutral = list(candidate.neutral_phone)
        lens = list(candidate.lens_phone)
        neutral[target - 2] = lens[target - 2] = VALIDATED_LEFT_CONSONANT
        neutral[target] = SOURCE_PHONE
        lens[target] = TARGET_PHONE
        neutral[target + 1] = lens[target + 1] = VALIDATED_RIGHT_CONSONANT
        neutral_phone = "".join(neutral)
        lens_phone = "".join(lens)
        if (
            neutral_phone[target - 2 : target + 2] != VALIDATED_NEUTRAL_SHELL
            or lens_phone[target - 2 : target + 2] != VALIDATED_LENS_SHELL
        ):
            raise KokoroTypedEngineError(
                "validated_shell_drift", "The immediate target shell drifted."
            )
        return CarrierAssignment(
            neutral_surface=_surface_for(neutral_phone),
            lens_surface=_surface_for(lens_phone),
            neutral_phone=neutral_phone,
            lens_phone=lens_phone,
            candidate_attempt=attempt,
        )
