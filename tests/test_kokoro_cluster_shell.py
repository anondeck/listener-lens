from __future__ import annotations

import pytest

from earshift_bakeoff.kokoro_cluster_shell import (
    CLUSTER_EXTRA_CONSONANTS,
    CLUSTER_LENS_SHELL,
    CLUSTER_NEUTRAL_SHELL,
    ClusterShellPlanner,
)
from earshift_bakeoff.kokoro_typed_engine import KokoroTypedEngineError, MappingKey


@pytest.fixture(scope="module")
def planner() -> ClusterShellPlanner:
    return ClusterShellPlanner.load()


def _key(source_phone: str, target_offsets: tuple[int, ...]) -> MappingKey:
    return MappingKey(
        source_casefold="fixture",
        source_phone=source_phone,
        target_offsets=target_offsets,
        carrier_role="content",
    )


def test_coda_cluster_gets_voiceless_shell_and_extras(planner) -> None:
    key = _key("hˈænd", (2,))
    assignment = planner._candidate(key, attempt=0)
    assert assignment.neutral_phone[0:4] == CLUSTER_NEUTRAL_SHELL
    assert assignment.lens_phone[0:4] == CLUSTER_LENS_SHELL
    assert assignment.neutral_phone[4] in CLUSTER_EXTRA_CONSONANTS
    assert assignment.lens_phone[4] == assignment.neutral_phone[4]
    assert len(assignment.neutral_phone) == len(key.source_phone)


def test_triple_coda_extras_all_voiceless(planner) -> None:
    key = _key("tˈæmpt", (2,))
    assignment = planner._candidate(key, attempt=0)
    assert assignment.neutral_phone[0:4] == CLUSTER_NEUTRAL_SHELL
    for position in (4, 5):
        assert assignment.neutral_phone[position] in CLUSTER_EXTRA_CONSONANTS


def test_attempt_rotation_stays_inside_extra_pool(planner) -> None:
    key = _key("hˈænd", (2,))
    picks = {
        planner._candidate(key, attempt=attempt).neutral_phone[4]
        for attempt in range(len(CLUSTER_EXTRA_CONSONANTS))
    }
    assert picks <= set(CLUSTER_EXTRA_CONSONANTS)
    assert len(picks) > 1


def test_onset_clusters_and_simple_shapes_are_rejected(planner) -> None:
    with pytest.raises(KokoroTypedEngineError):
        planner._candidate(_key("ɡɹˈæb", (3,)), attempt=0)
    with pytest.raises(KokoroTypedEngineError):
        planner._candidate(_key("kˈæt", (2,)), attempt=0)
    with pytest.raises(KokoroTypedEngineError):
        planner._candidate(_key("ˈæsk", (1,)), attempt=0)


def test_nontarget_words_pass_through_unchanged(planner) -> None:
    key = _key("tˈok", ())
    assignment = planner._candidate(key, attempt=0)
    assert "æ" not in assignment.neutral_phone
    assert assignment.neutral_phone == assignment.lens_phone
