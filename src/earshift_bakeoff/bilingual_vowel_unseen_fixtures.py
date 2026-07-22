from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict
import importlib.metadata
from typing import Any, Sequence

from wordfreq import iter_wordlist

from .bilingual_g2p_reachability import scan_vowel_rule_ids
from .bilingual_listener_engine import (
    BilingualListenerPlanner,
    load_listener_profiles,
)
from .bilingual_listener_engine_v8 import BilingualListenerPlannerV8
from .bilingual_product_isolation import (
    active_changed_rule_ids,
    isolate_listener_profile,
)
from .bilingual_vowel_engine import BilingualVowelEngineError
from .gates import canonical_token


UNSEEN_TYPED_FIXTURE_VERSION = "bilingual-vowel-unseen-typed-fixture-v1"
CONTEXT_ORDER = (
    "real_g2p_phrase_medial",
    "real_g2p_phrase_final",
    "real_g2p_repeated_target",
)
MINIMUM_CANONICAL_RANK = 256
MAXIMUM_CANONICAL_RANK = 75_000
CANDIDATES_PER_RULE = 128
MINIMUM_WORD_LENGTH = 3
MAXIMUM_WORD_LENGTH = 14

_FRAME_WORDS = {
    "en-US": frozenset(("they", "see", "more", "now", "say", "then")),
    "pt-BR": frozenset(("quem", "quer", "mais", "diz")),
}
_EXCLUDED_WORDS = frozenset(("mora", "tavi", "nelo"))


def fixture_text(
    language_id: str, context: str, target_word: str
) -> tuple[str, tuple[int, ...]]:
    if language_id == "en-US":
        if context == "real_g2p_phrase_medial":
            return f"They see {target_word} more now.", (2,)
        if context == "real_g2p_phrase_final":
            return f"They say more, then {target_word}.", (4,)
        if context == "real_g2p_repeated_target":
            return f"They say {target_word}, then {target_word}.", (2, 4)
    elif language_id == "pt-BR":
        if context == "real_g2p_phrase_medial":
            return f"Quem quer {target_word} mais?", (2,)
        if context == "real_g2p_phrase_final":
            return f"Quem quer mais, diz {target_word}.", (4,)
        if context == "real_g2p_repeated_target":
            return f"Quem diz {target_word}, diz {target_word}.", (2, 4)
    raise ValueError(f"unsupported unseen fixture frame: {language_id} {context}")


def _changed_ids(planner: BilingualListenerPlanner, phone: str) -> tuple[str, ...]:
    changed = {rule.rule_id for rule in planner.rules.values() if rule.changed}
    return tuple(
        rule_id
        for rule_id in scan_vowel_rule_ids(
            phone,
            rule_sources=planner.rule_sources,
            rules=planner.rules,
        )
        if rule_id in changed
    )


def _isolated_planner(
    *,
    base: BilingualListenerPlanner,
    profile_id: str,
    voice_id: str,
    rule_id: str,
) -> BilingualListenerPlannerV8:
    profile = isolate_listener_profile(load_listener_profiles()[profile_id], rule_id)
    return BilingualListenerPlannerV8(
        profile={
            **profile,
            "voice_id": voice_id,
            "voice_registry_version": base.profile["voice_registry_version"],
            "voice_registry_sha256": base.profile["voice_registry_sha256"],
        },
        adapter=base.adapter,
        model_vocab=set(base.model_vocab),
        nonce_checker=base.nonce_checker,
        phone_indexes=base.phone_indexes,
    )


def _candidate_contract(
    *,
    plan: Any,
    rule_id: str,
    target_word: str,
    expected_target_word_indexes: tuple[int, ...],
) -> tuple[bool, str | None]:
    expected_occurrences = len(expected_target_word_indexes)
    if active_changed_rule_ids(plan) != (rule_id,):
        return False, "active_rule_drift"
    if tuple(plan.target_word_indexes) != expected_target_word_indexes:
        return False, "target_word_index_drift"
    if plan.coverage.changed_vowel_occurrences != expected_occurrences:
        return False, "target_occurrence_count_drift"
    if (
        plan.coverage.changed_consonant_occurrences
        or plan.coverage.changed_insertion_occurrences
        or plan.coverage.changed_prosody_occurrences
        or plan.active_prosody_rule_ids
    ):
        return False, "nonvowel_change_present"
    if not plan.comparison_available:
        return False, "comparison_unavailable"
    if any(
        plan.words[index].source.casefold() != target_word.casefold()
        for index in expected_target_word_indexes
    ):
        return False, "target_source_word_drift"
    changed = [
        occurrence
        for word in plan.words
        for occurrence in word.vowel_occurrences
        if occurrence.changed
    ]
    if len(changed) != expected_occurrences or any(
        occurrence.rule_id != rule_id for occurrence in changed
    ):
        return False, "changed_occurrence_drift"
    gates = plan.gates
    if not (
        gates.written_and_espeak_gate_pass
        and gates.supplemental_phone_gates_pass
        and gates.model_representable
        and gates.punctuation_preserved
        and gates.repeated_word_invariant_pass
    ):
        return False, "planner_gate_fail"
    return True, None


def _word_candidates(
    *,
    language: str,
    rule_ids: Sequence[str],
    base_planners: dict[str, BilingualListenerPlanner],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    primary_voice = sorted(base_planners)[0]
    primary = base_planners[primary_voice]
    desired = frozenset(rule_ids)
    rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rejections: Counter[str] = Counter()
    canonical_seen: set[str] = set()
    analyzed = 0
    last_rank = 0
    for raw_word in iter_wordlist(language, wordlist="large"):
        word = canonical_token(raw_word)
        if word is None or word in canonical_seen:
            continue
        canonical_seen.add(word)
        rank = len(canonical_seen)
        last_rank = rank
        if rank < MINIMUM_CANONICAL_RANK:
            continue
        if rank > MAXIMUM_CANONICAL_RANK:
            break
        if not (
            MINIMUM_WORD_LENGTH <= len(word) <= MAXIMUM_WORD_LENGTH
            and word.isalpha()
            and word.casefold() == word
            and word not in _FRAME_WORDS[primary.profile["source_language"]]
            and word not in _EXCLUDED_WORDS
        ):
            rejections["surface_filter"] += 1
            continue
        try:
            primary_analysis = primary.adapter.analyze(word)
        except Exception as exc:
            rejections[f"primary_g2p:{getattr(exc, 'code', type(exc).__name__)}"] += 1
            continue
        analyzed += 1
        if len(primary_analysis.words) != 1:
            rejections["primary_word_alignment"] += 1
            continue
        primary_ids = _changed_ids(primary, primary_analysis.words[0].phone)
        if len(primary_ids) != 1 or primary_ids[0] not in desired:
            rejections["not_exactly_one_desired_changed_rule"] += 1
            continue
        rule_id = primary_ids[0]
        if len(rows[rule_id]) >= CANDIDATES_PER_RULE:
            continue
        phone_by_voice = {primary_voice: primary_analysis.words[0].phone}
        cross_voice_pass = True
        for voice_id, planner in sorted(base_planners.items()):
            if voice_id == primary_voice:
                continue
            try:
                analysis = planner.adapter.analyze(word)
            except Exception as exc:
                rejections[
                    f"cross_voice_g2p:{getattr(exc, 'code', type(exc).__name__)}"
                ] += 1
                cross_voice_pass = False
                break
            if len(analysis.words) != 1 or _changed_ids(
                planner, analysis.words[0].phone
            ) != (rule_id,):
                rejections["cross_voice_rule_drift"] += 1
                cross_voice_pass = False
                break
            phone_by_voice[voice_id] = analysis.words[0].phone
        if cross_voice_pass:
            rows[rule_id].append(
                {
                    "canonical_rank": rank,
                    "word": word,
                    "source_phone_by_voice": dict(sorted(phone_by_voice.items())),
                }
            )
        if all(len(rows[rule_id]) >= CANDIDATES_PER_RULE for rule_id in desired):
            break
    missing = {
        rule_id: len(rows[rule_id])
        for rule_id in sorted(desired)
        if len(rows[rule_id]) < 3
    }
    if missing:
        raise RuntimeError(f"unseen typed word inventory is insufficient: {missing}")
    return dict(rows), {
        "language": language,
        "wordfreq_version": importlib.metadata.version("wordfreq"),
        "minimum_canonical_rank": MINIMUM_CANONICAL_RANK,
        "maximum_canonical_rank": MAXIMUM_CANONICAL_RANK,
        "last_canonical_rank_examined": last_rank,
        "canonical_word_count_examined": len(canonical_seen),
        "primary_g2p_analyzed_word_count": analyzed,
        "candidate_count_by_rule": {
            rule_id: len(rows[rule_id]) for rule_id in sorted(desired)
        },
        "rejection_counts": dict(sorted(rejections.items())),
    }


def select_unseen_typed_fixtures(
    candidate_cells: Sequence[dict[str, str]],
) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for cell in candidate_cells:
        grouped[(cell["profile_id"], cell["rule_id"])].append(dict(cell))

    slots: list[dict[str, Any]] = []
    inventory_receipts: list[dict[str, Any]] = []
    selection_rejections: Counter[str] = Counter()
    selected_words: dict[str, dict[str, str]] = {}
    profile_groups: dict[str, list[tuple[str, list[dict[str, str]]]]] = defaultdict(
        list
    )
    for (profile_id, rule_id), cells in sorted(grouped.items()):
        profile_groups[profile_id].append((rule_id, cells))

    for profile_id, rule_groups in sorted(profile_groups.items()):
        voices = sorted(
            {cell["voice_id"] for _, cells in rule_groups for cell in cells}
        )
        base_planners = {
            voice: BilingualListenerPlanner.load(profile_id, voice_id=voice)
            for voice in voices
        }
        source_language = next(iter(base_planners.values())).profile["source_language"]
        isolated = {
            (voice, rule_id): _isolated_planner(
                base=base_planners[voice],
                profile_id=profile_id,
                voice_id=voice,
                rule_id=rule_id,
            )
            for rule_id, cells in rule_groups
            for voice in sorted(cell["voice_id"] for cell in cells)
        }
        inventory, receipt = _word_candidates(
            language="en" if source_language == "en-US" else "pt",
            rule_ids=[rule_id for rule_id, _ in rule_groups],
            base_planners=base_planners,
        )
        inventory_receipts.append(receipt)
        for rule_id, cells in sorted(rule_groups):
            used: set[str] = set()
            selected_words[rule_id] = {}
            for context in CONTEXT_ORDER:
                selected: dict[str, Any] | None = None
                for candidate in inventory[rule_id]:
                    word = candidate["word"]
                    if word in used:
                        continue
                    text, target_indexes = fixture_text(source_language, context, word)
                    plans: dict[str, Any] = {}
                    rejected = False
                    for cell in sorted(cells, key=lambda row: row["voice_id"]):
                        voice_id = cell["voice_id"]
                        try:
                            plan = isolated[(voice_id, rule_id)].plan(text)
                        except BilingualVowelEngineError as exc:
                            selection_rejections[f"planner:{exc.code}"] += 1
                            rejected = True
                            break
                        passed, reason = _candidate_contract(
                            plan=plan,
                            rule_id=rule_id,
                            target_word=word,
                            expected_target_word_indexes=target_indexes,
                        )
                        if not passed:
                            selection_rejections[f"contract:{reason}"] += 1
                            rejected = True
                            break
                        plans[voice_id] = plan
                    if rejected:
                        continue
                    selected = {
                        "candidate": candidate,
                        "text": text,
                        "target_indexes": target_indexes,
                        "plans": plans,
                    }
                    break
                if selected is None:
                    raise RuntimeError(
                        f"no gate-clean unseen fixture: {profile_id} {rule_id} {context}"
                    )
                word = selected["candidate"]["word"]
                used.add(word)
                selected_words[rule_id][context] = word
                for cell in sorted(cells, key=lambda row: row["voice_id"]):
                    voice_id = cell["voice_id"]
                    plan = selected["plans"][voice_id]
                    slots.append(
                        {
                            "logical_slot_id": (
                                f"{profile_id}__{voice_id}__{rule_id}__{context}"
                            ),
                            "cell_id": cell["cell_id"],
                            "profile_id": profile_id,
                            "voice_id": voice_id,
                            "rule_id": rule_id,
                            "candidate_rung": cell["candidate_rung"],
                            "source": cell["source"],
                            "target": cell["target"],
                            "context": context,
                            "fixture_spec": {
                                "text": selected["text"],
                                "target_word": word,
                                "target_word_canonical_rank": selected["candidate"][
                                    "canonical_rank"
                                ],
                                "expected_target_word_indexes": list(
                                    selected["target_indexes"]
                                ),
                                "expected_target_occurrence_count": len(
                                    selected["target_indexes"]
                                ),
                                "target_source_phone": selected["candidate"][
                                    "source_phone_by_voice"
                                ][voice_id],
                            },
                            "plan_sha256": plan.plan_sha256,
                            "carrier_scripts": {
                                "neutral": plan.neutral_script,
                                "lens": plan.lens_script,
                            },
                            "gate_receipt": asdict(plan.gates),
                            "word_roles": [word.carrier_role for word in plan.words],
                            "product_enabled": False,
                        }
                    )
    return {
        "fixture_selection_version": UNSEEN_TYPED_FIXTURE_VERSION,
        "context_order": list(CONTEXT_ORDER),
        "cell_count": len(candidate_cells),
        "rule_group_count": len(grouped),
        "logical_slot_count": len(slots),
        "expected_occurrence_count": sum(
            row["fixture_spec"]["expected_target_occurrence_count"] for row in slots
        ),
        "selected_words_by_rule": selected_words,
        "inventory_receipts": inventory_receipts,
        "selection_rejection_counts": dict(sorted(selection_rejections.items())),
        "slots": slots,
    }
