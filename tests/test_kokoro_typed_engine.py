from __future__ import annotations

from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from earshift_bakeoff.kokoro_synthesis import PairRender
from earshift_bakeoff.kokoro_typed_engine import (
    KokoroTypedEngineError,
    KokoroTypedPlanner,
    inspect_render,
    merge_spans,
)
from earshift_bakeoff.listener_lens import NonceDecision


@dataclass
class _Token:
    text: str
    phonemes: str
    whitespace: str
    tag: str = "NN"


class _G2P:
    def __init__(self, tokens: list[_Token]) -> None:
        self.tokens = tokens

    def __call__(self, text: str) -> tuple[str, list[_Token]]:
        return (
            "".join(token.phonemes + token.whitespace for token in self.tokens),
            self.tokens,
        )


class _Analyzer:
    def __init__(self, phones: list[str]) -> None:
        self.phones = phones

    def phonemize_words(self, words: list[str], voice: str) -> list[str]:
        assert voice == "en-us"
        assert len(words) == len(self.phones)
        return self.phones


class _NonceChecker:
    enabled = True

    def check(
        self, surface: str, language: str, previous_surface: str | None
    ) -> NonceDecision:
        assert surface.isalpha()
        assert language == "en"
        return NonceDecision(True, "nˈɑns", None)


class _PhoneIndex:
    def phone_match(self, phone: str) -> bool:
        return False


class _OneAdjacencyConflictIndex:
    def __init__(self, single_phone_length: int) -> None:
        self.single_phone_length = single_phone_length
        self.rejected = False

    def phone_match(self, phone: str) -> bool:
        if len(phone) > self.single_phone_length and not self.rejected:
            self.rejected = True
            return True
        return False


VOCAB = set(' ;:,.!?—…"()“”ˈˌːʰʲ̃aAIiOuæɑɒɔəɚɜɐɛɪʊʌWQᵊᵻɨɯɤøœbdfɡhjklmnŋpɹrstTvwzʃʒθðʧʤ')


def _planner(
    tokens: list[_Token],
    espeak: list[str],
    *,
    phone_index: _PhoneIndex | _OneAdjacencyConflictIndex | None = None,
) -> KokoroTypedPlanner:
    return KokoroTypedPlanner(
        g2p=_G2P(tokens),
        model_vocab=VOCAB,
        source_analyzer=_Analyzer(espeak),
        nonce_checker=_NonceChecker(),
        phone_index=phone_index or _PhoneIndex(),
    )


def test_plans_multiple_target_words_and_preserves_punctuation() -> None:
    planner = _planner(
        [
            _Token("The", "ðə", " ", "DT"),
            _Token("cat", "kˈæt", " "),
            _Token("sat", "sˈæt", "", "VBD"),
            _Token(".", ".", "", "."),
        ],
        ["ðə", "kæt", "sæt"],
    )

    plan = planner.plan("The cat sat.")

    assert plan.comparison_available is True
    assert plan.target_word_indexes == (1, 2)
    assert plan.target_word_count == 2
    assert plan.target_occurrence_count == 2
    assert plan.coverage_count == 2
    assert plan.neutral_script.endswith(".")
    assert plan.lens_script.endswith(".")
    assert len(plan.source_phonemes) == len(plan.neutral_phonemes)
    assert len(plan.neutral_phonemes) == len(plan.lens_phonemes)
    assert plan.pair_plan() is not None


def test_repeated_source_mapping_is_identical_within_utterance() -> None:
    planner = _planner(
        [
            _Token("Cat", "kˈæt", " "),
            _Token("cat", "kˈæt", ""),
            _Token(".", ".", ""),
        ],
        ["kæt", "kæt"],
    )

    plan = planner.plan("Cat cat.")

    assert plan.words[0].neutral_surface == plan.words[1].neutral_surface
    assert plan.words[0].lens_surface == plan.words[1].lens_surface
    assert plan.words[0].neutral_phone == plan.words[1].neutral_phone
    assert plan.words[0].lens_phone == plan.words[1].lens_phone


def test_multiple_targets_inside_one_word_share_one_target_word_span() -> None:
    planner = _planner(
        [
            _Token("Catamaran", "kˈætəmæɹən", " "),
            _Token("moves", "mˈuvz", ""),
            _Token(".", ".", ""),
        ],
        ["kætəmæɹən", "muvz"],
    )

    plan = planner.plan("Catamaran moves.")

    assert plan.target_word_indexes == (0,)
    assert plan.target_word_count == 1
    assert plan.target_occurrence_count == 2
    assert plan.words[0].lens_phone.count("ɛ") == 2


def test_no_target_disables_comparison_without_render_plan() -> None:
    planner = _planner(
        [
            _Token("Green", "ɡɹˈin", " "),
            _Token("trees", "tɹˈiz", ""),
            _Token(".", ".", ""),
        ],
        ["ɡɹin", "tɹiz"],
    )

    plan = planner.plan("Green trees.")

    assert plan.comparison_available is False
    assert plan.coverage_count == 0
    assert plan.pair_plan() is None
    assert plan.neutral_phonemes == plan.lens_phonemes


def test_source_gate_disagreement_fails_closed() -> None:
    planner = _planner(
        [_Token("Catch", "kˈɛʧ", " "), _Token("sun", "sˈʌn", "")],
        ["kætʃ", "sʌn"],
    )

    with pytest.raises(KokoroTypedEngineError) as exc_info:
        planner.plan("Catch sun")

    assert exc_info.value.code == "eligible_target_disagreement"


def test_adjacency_conflict_is_deterministically_reresolved() -> None:
    planner = _planner(
        [_Token("Cat", "kˈæt", " "), _Token("sat", "sˈæt", "")],
        ["kæt", "sæt"],
        phone_index=_OneAdjacencyConflictIndex(single_phone_length=4),
    )

    plan = planner.plan("Cat sat")

    assert plan.gate_summary.candidate_attempts > 2
    assert (
        plan.gate_summary.candidate_rejection_counts[
            "neutral_kokoro_adjacency_predicted_homophone"
        ]
        == 1
    )


def test_unclassified_source_phone_fails_closed() -> None:
    planner = _planner(
        [_Token("Odd", "x☃", " "), _Token("word", "wɜɹd", "")],
        ["x", "wɜɹd"],
    )

    with pytest.raises(KokoroTypedEngineError) as exc_info:
        planner.plan("Odd word")

    assert exc_info.value.code == "unrepresentable_phone_plan"


def test_merge_spans_is_order_independent_and_merges_touching_ranges() -> None:
    assert merge_spans(((4, 8), (1, 3), (3, 5), (10, 11))) == (
        (1, 8),
        (10, 11),
    )


def test_render_integrity_checks_finite_count_and_clipping() -> None:
    good = inspect_render(
        PairRender(
            neutral=np.array([0.0, 0.5], dtype=np.float32),
            lens=np.array([0.0, -0.5], dtype=np.float32),
            predicted_durations=(1,),
            replaced_columns=(1,),
        )
    )
    bad = inspect_render(
        PairRender(
            neutral=np.array([0.0, 1.1], dtype=np.float32),
            lens=np.array([0.0], dtype=np.float32),
            predicted_durations=(1,),
            replaced_columns=(1,),
        )
    )

    assert good.pass_all is True
    assert bad.pass_all is False


def test_planner_is_deterministic_under_concurrent_identical_and_mixed_inputs() -> None:
    first = _planner(
        [_Token("Cat", "kˈæt", " "), _Token("sat", "sˈæt", "")],
        ["kæt", "sæt"],
    )
    second = _planner(
        [_Token("Green", "ɡɹˈin", " "), _Token("trees", "tɹˈiz", "")],
        ["ɡɹin", "tɹiz"],
    )
    baseline_first = first.plan("Cat sat").plan_sha256
    baseline_second = second.plan("Green trees").plan_sha256

    jobs = ((first, "Cat sat"), (first, "Cat sat"), (second, "Green trees"))
    with ThreadPoolExecutor(max_workers=3) as executor:
        actual = list(
            executor.map(lambda item: item[0].plan(item[1]).plan_sha256, jobs)
        )

    assert actual == [baseline_first, baseline_first, baseline_second]
