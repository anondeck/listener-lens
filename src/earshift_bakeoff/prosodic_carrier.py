from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Sequence

from .listener_lens import (
    WORD_RE,
    LensResult,
    ListenerLensError,
    NonceChecker,
    NonceDecision,
)


PROSODIC_CARRIER_VERSION = 1
PUNCTUATION_RE = re.compile(r"[^\s]")


@dataclass(frozen=True)
class ProsodicGateAttempt:
    side: str
    stage: str
    group_index: int
    surface: str
    previous_surface: str | None
    accepted: bool
    predicted_ipa: str
    rejection_reason: str | None


@dataclass(frozen=True)
class ProsodicRuleSlot:
    group_index: int
    source_word_index: int
    neutral_character_span: tuple[int, int]
    lens_character_span: tuple[int, int]
    rule_id: str
    source_ipa: str
    target_ipa: str
    neutral_grapheme: str
    lens_grapheme: str


@dataclass(frozen=True)
class ProsodicGroup:
    group_index: int
    source_word_indices: tuple[int, ...]
    source_words: tuple[str, ...]
    head_source_word_index: int
    neutral_surface: str
    lens_surface: str
    syllables: int
    slots: tuple[ProsodicRuleSlot, ...]


@dataclass(frozen=True)
class ProsodicCarrierResult:
    version: int
    source_word_count: int
    group_count: int
    total_syllables: int
    neutral_script: str
    lens_script: str
    groups: tuple[ProsodicGroup, ...]
    slots: tuple[ProsodicRuleSlot, ...]
    gate_attempts: tuple[ProsodicGateAttempt, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _decision(
    checker: NonceChecker,
    surface: str,
    language: str,
    previous_surface: str | None,
) -> NonceDecision:
    check = getattr(checker, "check", None)
    if callable(check):
        return check(surface, language, previous_surface)
    accepted, predicted_ipa = checker.accepts(surface, language, previous_surface)
    return NonceDecision(
        accepted=accepted,
        predicted_ipa=predicted_ipa,
        rejection_reason=(
            None
            if accepted
            else (
                "adjacency_nonce_gate_rejected"
                if previous_surface is not None
                else "nonce_gate_rejected"
            )
        ),
    )


def _source_segments(text: str, word_count: int) -> tuple[list[re.Match[str]], list[tuple[int, int]]]:
    matches = list(WORD_RE.finditer(text))
    if len(matches) != word_count:
        raise ListenerLensError("The source text no longer aligns with its carrier words.")
    if not matches:
        raise ListenerLensError("A prosodic carrier requires at least one source word.")

    segments: list[tuple[int, int]] = []
    start = 0
    for index in range(len(matches) - 1):
        gap = text[matches[index].end() : matches[index + 1].start()]
        if PUNCTUATION_RE.search(gap):
            segments.append((start, index))
            start = index + 1
    segments.append((start, len(matches) - 1))
    return matches, segments


def _group_indices(words: Sequence[Any], segments: Sequence[tuple[int, int]]) -> list[tuple[int, ...]]:
    groups: list[tuple[int, ...]] = []
    for segment_start, segment_end in segments:
        content = [
            index
            for index in range(segment_start, segment_end + 1)
            if words[index].carrier_role == "content"
        ]
        if not content:
            groups.append(tuple(range(segment_start, segment_end + 1)))
            continue
        for content_offset, head_index in enumerate(content):
            group_start = segment_start if content_offset == 0 else head_index
            group_end = (
                content[content_offset + 1] - 1
                if content_offset + 1 < len(content)
                else segment_end
            )
            groups.append(tuple(range(group_start, group_end + 1)))
    return groups


def _normalized_gap(value: str) -> str:
    return re.sub(r"\s+", " ", value)


def _render_script(
    source_text: str,
    matches: Sequence[re.Match[str]],
    groups: Sequence[ProsodicGroup],
    side: str,
) -> str:
    pieces = [_normalized_gap(source_text[: matches[0].start()])]
    for group_offset, group in enumerate(groups):
        pieces.append(getattr(group, f"{side}_surface"))
        last_word_index = group.source_word_indices[-1]
        if group_offset + 1 < len(groups):
            next_word_index = groups[group_offset + 1].source_word_indices[0]
            gap = source_text[
                matches[last_word_index].end() : matches[next_word_index].start()
            ]
            pieces.append(_normalized_gap(gap))
        else:
            pieces.append(_normalized_gap(source_text[matches[last_word_index].end() :]))
    return "".join(pieces)


def build_prosodic_carrier(
    result: LensResult,
    checker: NonceChecker,
    *,
    language: str = "en",
) -> ProsodicCarrierResult:
    """Regroup carrier syllables into accentual feet without changing them.

    A content carrier is the head of a group. Leading weak carriers attach to
    the first head in a punctuation-bounded segment; following weak carriers
    attach to the preceding head. Adjacent content carriers remain separate.
    The operation removes only whitespace between members of a group.
    """

    matches, segments = _source_segments(result.original_text, len(result.words))
    index_groups = _group_indices(result.words, segments)
    groups: list[ProsodicGroup] = []
    all_slots: list[ProsodicRuleSlot] = []

    for group_index, source_indices in enumerate(index_groups):
        neutral_offset = 0
        lens_offset = 0
        group_slots: list[ProsodicRuleSlot] = []
        for source_word_index in source_indices:
            word = result.words[source_word_index]
            for slot in word.slots:
                neutral_start, neutral_end = slot.neutral_character_span
                lens_start, lens_end = slot.lens_character_span
                group_slots.append(
                    ProsodicRuleSlot(
                        group_index=group_index,
                        source_word_index=source_word_index,
                        neutral_character_span=(
                            neutral_offset + neutral_start,
                            neutral_offset + neutral_end,
                        ),
                        lens_character_span=(
                            lens_offset + lens_start,
                            lens_offset + lens_end,
                        ),
                        rule_id=slot.rule_id,
                        source_ipa=slot.source_ipa,
                        target_ipa=slot.target_ipa,
                        neutral_grapheme=slot.neutral_grapheme,
                        lens_grapheme=slot.lens_grapheme,
                    )
                )
            neutral_offset += len(word.neutral_surface)
            lens_offset += len(word.lens_surface)

        content_heads = [
            index
            for index in source_indices
            if result.words[index].carrier_role == "content"
        ]
        group = ProsodicGroup(
            group_index=group_index,
            source_word_indices=source_indices,
            source_words=tuple(result.words[index].source for index in source_indices),
            head_source_word_index=content_heads[0] if content_heads else source_indices[0],
            neutral_surface="".join(
                result.words[index].neutral_surface for index in source_indices
            ),
            lens_surface="".join(
                result.words[index].lens_surface for index in source_indices
            ),
            syllables=sum(result.words[index].syllables for index in source_indices),
            slots=tuple(group_slots),
        )
        groups.append(group)
        all_slots.extend(group_slots)

    attempts: list[ProsodicGateAttempt] = []
    for side in ("neutral", "lens"):
        previous_surface: str | None = None
        for group in groups:
            surface = getattr(group, f"{side}_surface")
            isolated = _decision(checker, surface, language, None)
            attempts.append(
                ProsodicGateAttempt(
                    side=side,
                    stage="isolated",
                    group_index=group.group_index,
                    surface=surface,
                    previous_surface=None,
                    accepted=isolated.accepted,
                    predicted_ipa=isolated.predicted_ipa,
                    rejection_reason=isolated.rejection_reason,
                )
            )
            if not isolated.accepted:
                raise ListenerLensError(
                    f"The {side} prosodic group {group.group_index} failed the "
                    f"local opacity gate: {isolated.rejection_reason or 'rejected'}."
                )
            if previous_surface is not None:
                adjacent = _decision(checker, surface, language, previous_surface)
                attempts.append(
                    ProsodicGateAttempt(
                        side=side,
                        stage="adjacency",
                        group_index=group.group_index,
                        surface=surface,
                        previous_surface=previous_surface,
                        accepted=adjacent.accepted,
                        predicted_ipa=adjacent.predicted_ipa,
                        rejection_reason=adjacent.rejection_reason,
                    )
                )
                if not adjacent.accepted:
                    raise ListenerLensError(
                        f"The {side} prosodic boundary before group "
                        f"{group.group_index} failed the local opacity gate: "
                        f"{adjacent.rejection_reason or 'rejected'}."
                    )
            previous_surface = surface

    neutral_script = _render_script(result.original_text, matches, groups, "neutral")
    lens_script = _render_script(result.original_text, matches, groups, "lens")
    total_syllables = sum(group.syllables for group in groups)
    source_syllables = sum(word.syllables for word in result.words)
    if total_syllables != source_syllables:
        raise ListenerLensError("Prosodic regrouping changed the carrier syllable count.")
    if len(all_slots) != len(result.slots):
        raise ListenerLensError("Prosodic regrouping lost a listener-rule slot.")
    if len(WORD_RE.findall(neutral_script)) != len(groups):
        raise ListenerLensError("The neutral prosodic script lost group alignment.")
    if len(WORD_RE.findall(lens_script)) != len(groups):
        raise ListenerLensError("The listener prosodic script lost group alignment.")

    for group in groups:
        for slot in group.slots:
            if (
                group.neutral_surface[slice(*slot.neutral_character_span)]
                != slot.neutral_grapheme
            ):
                raise ListenerLensError("A neutral slot moved during prosodic regrouping.")
            if (
                group.lens_surface[slice(*slot.lens_character_span)]
                != slot.lens_grapheme
            ):
                raise ListenerLensError("A listener slot moved during prosodic regrouping.")

    return ProsodicCarrierResult(
        version=PROSODIC_CARRIER_VERSION,
        source_word_count=len(result.words),
        group_count=len(groups),
        total_syllables=total_syllables,
        neutral_script=neutral_script,
        lens_script=lens_script,
        groups=tuple(groups),
        slots=tuple(all_slots),
        gate_attempts=tuple(attempts),
    )
