from __future__ import annotations

from dataclasses import dataclass

from earshift_bakeoff.bilingual_g2p_reachability import scan_vowel_rule_ids


@dataclass(frozen=True)
class _Rule:
    rule_id: str
    source: str


def test_scan_uses_longest_rule_for_nasal_vowels() -> None:
    rules = {
        "ɐ̃": _Rule("nasal", "ɐ̃"),
        "ɐ": _Rule("oral", "ɐ"),
        "ʊ": _Rule("foot", "ʊ"),
    }

    assert scan_vowel_rule_ids(
        "nˈɐ̃ʊ̃",
        rule_sources=("ɐ̃", "ɐ", "ʊ"),
        rules=rules,
    ) == ("nasal", "foot")


def test_scan_counts_repeated_occurrences() -> None:
    rules = {"æ": _Rule("trap", "æ")}

    assert scan_vowel_rule_ids(
        "bˈæd bˈæd",
        rule_sources=("æ",),
        rules=rules,
    ) == ("trap", "trap")
