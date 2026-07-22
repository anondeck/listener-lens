from __future__ import annotations

from .kokoro_typed_engine import (
    KokoroTypedEngineError,
    MappingKey,
    _CONSONANT_SYMBOLS,
)
from .kokoro_validated_shell import ValidatedShellPlanner


STRICT_SHELL_VERSION = 1


class StrictShellPlanner(ValidatedShellPlanner):
    """Validated-shell planner limited to exact C-stress-/ae/-C target words."""

    @classmethod
    def load(cls) -> StrictShellPlanner:
        base = ValidatedShellPlanner.load()
        return cls(
            g2p=base.g2p,
            model_vocab=base.model_vocab,
            source_analyzer=base.source_analyzer,
            nonce_checker=base.nonce_checker,
            phone_index=base.phone_index,
            rules_path=base.rules_path,
        )

    def _candidate(self, key: MappingKey, attempt: int):  # type: ignore[no-untyped-def]
        if key.target_offsets:
            exact_shape = bool(
                len(key.source_phone) == 4
                and key.target_offsets == (2,)
                and key.source_phone[0] in _CONSONANT_SYMBOLS
                and key.source_phone[1] in {"ˈ", "ˌ"}
                and key.source_phone[3] in _CONSONANT_SYMBOLS
            )
            if not exact_shape:
                raise KokoroTypedEngineError(
                    "strict_shell_unsupported_target_word",
                    "This version supports /æ/ only in exact C-stress-vowel-C words.",
                )
        return super()._candidate(key, attempt)
