from __future__ import annotations

from types import SimpleNamespace

from earshift_bakeoff.bilingual_listener_engine_v8 import (
    VOWEL_MEASUREMENT_ALIGNMENT_VERSION,
    bilingual_alignment_record_v8,
)
from earshift_bakeoff.bilingual_vowel_engine import VowelOccurrence


def test_v8_alignment_includes_preceding_stress_and_complete_vowel() -> None:
    occurrence = VowelOccurrence(
        source="æ",
        target="ɛ",
        rule_id="enpt.ae_eh",
        evidence_tier="direct_assimilation",
        acoustic_status="pending",
        changed=True,
        phone_offset=2,
        phone_length=1,
    )
    word = SimpleNamespace(
        neutral_phone="bˈæz",
        lens_phone="bˈɛz",
        vowel_occurrences=(occurrence,),
        consonant_occurrences=(),
        prosody_occurrences=(),
        insertion_occurrences=(),
    )
    plan = SimpleNamespace(
        render_reference_phonemes="bˈæz",
        neutral_phonemes="bˈæz",
        target_word_indexes=(0,),
        words=(word,),
        coverage=SimpleNamespace(
            changed_vowel_occurrences=1,
            changed_consonant_occurrences=0,
            changed_prosody_occurrences=0,
            changed_insertion_occurrences=0,
        ),
    )
    model = SimpleNamespace(
        vocab={symbol: index for index, symbol in enumerate("bˈæz")}
    )
    durations = (1, 1, 2, 3, 2, 1)
    sample_count = sum(durations) * 600

    record = bilingual_alignment_record_v8(
        model=model,
        plan=plan,
        durations=durations,
        sample_count=sample_count,
    )

    row = record["target_occurrences"][0]
    assert row["control_interval"]["columns"] == [3]
    assert row["measurement_interval"]["columns"] == [2, 3]
    assert row["stress_context_column"] == 2
    assert row["vowel_state_columns"] == [3]
    assert row["measurement_alignment_version"] == VOWEL_MEASUREMENT_ALIGNMENT_VERSION
    assert row["measurement_interval"]["start_sample"] == 1200
    assert row["measurement_interval"]["end_sample_exclusive"] == 4200


def test_v8_unstressed_vowel_keeps_vowel_only_measurement_span() -> None:
    occurrence = VowelOccurrence(
        source="a",
        target="æ",
        rule_id="pten.a_ae",
        evidence_tier="direct",
        acoustic_status="pending",
        changed=True,
        phone_offset=1,
        phone_length=1,
    )
    word = SimpleNamespace(
        neutral_phone="baz",
        lens_phone="bæz",
        vowel_occurrences=(occurrence,),
        consonant_occurrences=(),
        prosody_occurrences=(),
        insertion_occurrences=(),
    )
    plan = SimpleNamespace(
        render_reference_phonemes="baz",
        neutral_phonemes="baz",
        target_word_indexes=(0,),
        words=(word,),
        coverage=SimpleNamespace(
            changed_vowel_occurrences=1,
            changed_consonant_occurrences=0,
            changed_prosody_occurrences=0,
            changed_insertion_occurrences=0,
        ),
    )
    model = SimpleNamespace(vocab={symbol: index for index, symbol in enumerate("baz")})
    durations = (1, 2, 3, 2, 1)

    row = bilingual_alignment_record_v8(
        model=model,
        plan=plan,
        durations=durations,
        sample_count=sum(durations) * 600,
    )["target_occurrences"][0]

    assert row["measurement_interval"]["columns"] == [2]
    assert row["stress_context_column"] is None
