from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from typing import Any, Iterable

from .config import stable_json


CONSONANT_CALIBRATION_VERSION = "consonant-calibration-v1"
ALLOSAURUS_VERSION = "1.0.2"
ALLOSAURUS_SCIPY_VERSION = "1.11.4"
ALLOSAURUS_LICENSE = "GPL-3.0"
ALLOSAURUS_SOURCE = "https://github.com/xinjli/allosaurus"
UPR_TIMESTAMP_CONTEXT_S = 0.030


@dataclass(frozen=True)
class ConsonantCalibrationFixture:
    fixture_id: str
    rule_id: str
    evidence_tier: str
    voice_id: str
    context_id: str
    neutral_phonemes: str
    source: str
    target: str
    expected_source_labels: tuple[str, ...]
    expected_target_labels: tuple[str, ...]

    @property
    def lens_phonemes(self) -> str:
        if self.neutral_phonemes.count(self.source) != 1:
            raise ValueError(
                f"{self.fixture_id} must contain its source sequence exactly once"
            )
        return self.neutral_phonemes.replace(self.source, self.target, 1)

    def protocol_record(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "lens_phonemes": self.lens_phonemes,
        }


def calibration_fixtures() -> tuple[ConsonantCalibrationFixture, ...]:
    rows: list[ConsonantCalibrationFixture] = []

    def add(
        *,
        rule_id: str,
        evidence_tier: str,
        voice_id: str,
        source: str,
        target: str,
        expected_source_labels: tuple[str, ...],
        expected_target_labels: tuple[str, ...],
        contexts: Iterable[tuple[str, str]],
    ) -> None:
        for context_id, phonemes in contexts:
            rows.append(
                ConsonantCalibrationFixture(
                    fixture_id=f"{rule_id}__{context_id}",
                    rule_id=rule_id,
                    evidence_tier=evidence_tier,
                    voice_id=voice_id,
                    context_id=context_id,
                    neutral_phonemes=phonemes,
                    source=source,
                    target=target,
                    expected_source_labels=expected_source_labels,
                    expected_target_labels=expected_target_labels,
                )
            )

    english_prefix = "mˈɑl vˈɑn "
    english_suffix = " nˈɑl sˈɑv."
    add(
        rule_id="enpt.theta_t",
        evidence_tier="direct_listener_assimilation",
        voice_id="af_heart",
        source="θ",
        target="t",
        expected_source_labels=("θ",),
        expected_target_labels=("t", "tʰ", "t̪"),
        contexts=(
            ("word_initial_a", english_prefix + "θˈɑf" + english_suffix),
            ("intervocalic_a", english_prefix + "fˈɑθɑm" + english_suffix),
            ("word_final_a", english_prefix + "fˈɑθ" + english_suffix),
        ),
    )
    add(
        rule_id="enpt.eth_d",
        evidence_tier="direct_listener_assimilation",
        voice_id="af_heart",
        source="ð",
        target="d",
        expected_source_labels=("ð",),
        expected_target_labels=("d", "d̪"),
        contexts=(
            ("word_initial_a", english_prefix + "ðˈɑf" + english_suffix),
            ("intervocalic_a", english_prefix + "fˈɑðɑm" + english_suffix),
            ("word_final_a", english_prefix + "fˈɑð" + english_suffix),
        ),
    )

    portuguese_prefix = "mˈal vˈan "
    portuguese_suffix = " nˈal sˈav."
    for rule_id, evidence_tier, source, target, source_labels, target_labels in (
        (
            "pten.palatal_lateral_yod",
            "direct_listener_identification_renderer_projection",
            "lj",
            "jj",
            ("ʎ", "lʲ"),
            ("j", "ʝ"),
        ),
        (
            "pten.palatal_nasal_n",
            "derived_nearest_listener_category",
            "ɲ",
            "n",
            ("ɲ", "nʲ"),
            ("n", "n̪"),
        ),
        (
            "pten.dorsal_r_h",
            "derived_nearest_listener_category",
            "x",
            "h",
            ("x", "χ", "ɣ"),
            ("h", "ɦ"),
        ),
        (
            "pten.tap_flap",
            "derived_listener_allophone_correspondence",
            "ɾ",
            "T",
            ("ɾ", "ɽ"),
            ("ɾ", "ɽ", "t", "d"),
        ),
    ):
        add(
            rule_id=rule_id,
            evidence_tier=evidence_tier,
            voice_id="pf_dora",
            source=source,
            target=target,
            expected_source_labels=source_labels,
            expected_target_labels=target_labels,
            contexts=tuple(
                (
                    f"intervocalic_{vowel}",
                    portuguese_prefix
                    + f"fˈ{vowel}{source}{vowel}m"
                    + portuguese_suffix,
                )
                for vowel in ("a", "i", "u")
            ),
        )
    fixtures = tuple(rows)
    if len(fixtures) != 18 or len({row.fixture_id for row in fixtures}) != 18:
        raise RuntimeError("the consonant calibration manifest drifted")
    for fixture in fixtures:
        if len(fixture.source) != len(fixture.target):
            raise RuntimeError(
                f"{fixture.fixture_id} violates shared-state token parity"
            )
        fixture.lens_phonemes
    return fixtures


def protocol_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(stable_json(payload).encode("utf-8")).hexdigest()


def parse_allosaurus_timestamps(text: str) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        values = line.strip().split(maxsplit=2)
        if not values:
            continue
        if len(values) != 3:
            raise ValueError(f"unexpected Allosaurus timestamp row: {line!r}")
        start_s = float(values[0])
        duration_s = float(values[1])
        if start_s < 0.0 or duration_s <= 0.0 or not values[2]:
            raise ValueError(f"invalid Allosaurus timestamp row: {line!r}")
        rows.append(
            {
                "start_s": start_s,
                "duration_s": duration_s,
                "end_s": start_s + duration_s,
                "label": values[2],
            }
        )
    return tuple(rows)


def overlapping_upr_labels(
    rows: Iterable[dict[str, Any]],
    *,
    start_s: float,
    end_s: float,
    context_s: float = UPR_TIMESTAMP_CONTEXT_S,
) -> tuple[str, ...]:
    lower = max(0.0, start_s - context_s)
    upper = end_s + context_s
    return tuple(
        str(row["label"])
        for row in rows
        if float(row["start_s"]) < upper and float(row["end_s"]) > lower
    )


def labels_support_expected(
    observed: Iterable[str], expected: Iterable[str]
) -> bool:
    expected_values = set(expected)
    return any(value in expected_values for value in observed)


def aggregate_rule_instrument(
    rows: Iterable[dict[str, Any]], *, evidence_tier: str
) -> dict[str, Any]:
    values = tuple(rows)
    source_matches = sum(bool(row["source_anchor_upr_match"]) for row in values)
    target_matches = sum(bool(row["target_anchor_upr_match"]) for row in values)
    minimum = 2
    supportive = bool(
        len(values) == 3
        and source_matches >= minimum
        and target_matches >= minimum
        and all(row["engineering_integrity_pass"] for row in values)
    )
    direct = evidence_tier.startswith("direct_")
    return {
        "fixture_count": len(values),
        "source_anchor_upr_matches": source_matches,
        "target_anchor_upr_matches": target_matches,
        "supportive_context_minimum": minimum,
        "engineering_integrity_pass": bool(
            values and all(row["engineering_integrity_pass"] for row in values)
        ),
        "auxiliary_instrument_status": (
            "supportive" if supportive else "mixed_or_inconclusive"
        ),
        "claim_status": (
            "eligible_for_blind_human_qc_not_promoted"
            if supportive and direct
            else (
                "research_only_derived_rule_not_promoted"
                if supportive
                else "disabled_pending_better_realization_or_measurement"
            )
        ),
    }
