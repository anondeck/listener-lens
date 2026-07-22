from __future__ import annotations

from types import SimpleNamespace

import pytest

from earshift_bakeoff import kokoro_typed_confirmation as confirmation
from earshift_bakeoff.kokoro_typed_confirmation import (
    _analysis_payload,
    _begin_fixture_attempts,
    _family_gate,
    _review_html,
    alignment_record,
    build_review,
)
from earshift_bakeoff.kokoro_typed_confirmation_protocol import (
    REVIEW_RESPONSE_FILENAME,
)


def test_alignment_uses_own_source_map_and_maps_every_occurrence() -> None:
    phones = "zɪ vˈæʒ zɪ vˈæʒ."
    model = SimpleNamespace(
        vocab={symbol: index + 1 for index, symbol in enumerate(set(phones))}
    )
    words = (
        SimpleNamespace(neutral_phone="zɪ", lens_phone="zɪ", target_offsets=()),
        SimpleNamespace(neutral_phone="vˈæʒ", lens_phone="vˈɛʒ", target_offsets=(2,)),
        SimpleNamespace(neutral_phone="zɪ", lens_phone="zɪ", target_offsets=()),
        SimpleNamespace(neutral_phone="vˈæʒ", lens_phone="vˈɛʒ", target_offsets=(2,)),
    )
    plan = SimpleNamespace(
        source_phonemes=phones,
        neutral_phonemes=phones,
        words=words,
        target_word_indexes=(1, 3),
        target_occurrence_count=2,
    )
    duration_count = len(phones) + 2
    durations = tuple(range(1, duration_count + 1))
    result = alignment_record(
        model=model,
        plan=plan,
        durations=durations,
        sample_count=sum(durations) * 600,
        anchor_occurrence_map=(0, 1),
    )
    assert result["samples_per_alignment_frame"] == 600
    assert result["own_source_derived_durations"] is True
    assert result["own_source_derived_alignment"] is True
    assert result["own_fixture_neutral_f0_noise"] is True
    assert [row["anchor_occurrence_index"] for row in result["target_occurrences"]] == [
        0,
        1,
    ]
    assert [row["position"] for row in result["target_occurrences"]] == [
        "medial",
        "phrase-final",
    ]
    assert [
        len(row["target_word_interval"]["columns"])
        for row in result["target_occurrences"]
    ] == [4, 4]


def _measurement(point: tuple[float, float]) -> dict[str, float | bool]:
    return {
        "measurement_valid": True,
        "plausibility_pass": True,
        "f1_bark": point[0],
        "f2_bark": point[1],
    }


def test_local_anchor_family_gate_is_conjunctive_and_directional() -> None:
    anchor = {"ae_bark": [6.0, 11.0], "eh_bark": [5.0, 12.0]}
    passed = _family_gate(_measurement((6.0, 11.0)), _measurement((5.0, 12.0)), anchor)
    reversed_result = _family_gate(
        _measurement((5.0, 12.0)), _measurement((6.0, 11.0)), anchor
    )
    invalid = _family_gate(
        {**_measurement((6.0, 11.0)), "measurement_valid": False},
        _measurement((5.0, 12.0)),
        anchor,
    )
    assert passed["pass"] is True
    assert all(passed["checks"].values())
    assert reversed_result["pass"] is False
    assert invalid["pass"] is False


def test_attempt_markers_consume_whole_fixed_triplet(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(confirmation, "run_dir", lambda: tmp_path)
    records = {
        "protocol_sha256": "a" * 64,
        "slots": [
            {
                "slot_id": f"fixture__{role}",
                "fixture_id": "fixture",
                "role": role,
                "status": "pending",
            }
            for role in ("neutral", "identity", "lens")
        ],
    }
    rows = _begin_fixture_attempts(records, "fixture")
    assert [row["status"] for row in rows] == ["attempt_started"] * 3
    assert sorted(path.stem for path in (tmp_path / "attempts").iterdir()) == [
        "fixture__identity",
        "fixture__lens",
        "fixture__neutral",
    ]
    with pytest.raises(RuntimeError, match="already attempted"):
        _begin_fixture_attempts(records, "fixture")
    assert all(row["status"] == "interrupted_no_retry" for row in records["slots"])


def test_public_review_hides_fixtures_roles_and_exact_source_text() -> None:
    public = [
        {
            "trial_id": "comparison-01",
            "duration_s": 1.0,
            "target_intervals": [{"start_s": 0.2, "end_s": 0.4}],
            "sides": [
                {"side": "A", "audio": "review-audio/01-a.wav"},
                {"side": "B", "audio": "review-audio/01-b.wav"},
            ],
        }
    ]
    html = _review_html(public, "a" * 64)
    for hidden in (
        "The cap turns near the cap.",
        "We rest near the cap.",
        "new-repeated-phrase-final",
        "independent-phrase-final-only",
        "identity-catch",
        "lens-candidate",
        "side_roles",
    ):
        assert hidden not in html
    assert REVIEW_RESPONSE_FILENAME in html
    assert "TARGET NOW" in html


def test_review_cannot_be_built_before_automatic_pass(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(confirmation, "run_dir", lambda: tmp_path)
    with pytest.raises(RuntimeError, match="did not authorize"):
        build_review({}, {"automatic_confirmation_pass": False})
    assert list(tmp_path.glob("**/*")) == []


def test_measurement_failure_preserves_other_fixture_evidence(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(confirmation, "run_dir", lambda: tmp_path)
    (tmp_path / "render-records.json").write_text("{}\n", encoding="utf-8")

    def analyze(record, _protocol):
        if record["fixture_id"] == "bad":
            raise RuntimeError("instrument failed")
        return {"fixture_id": record["fixture_id"], "automatic_pass": True}

    monkeypatch.setattr(confirmation, "_analyze_fixture", analyze)
    result = _analysis_payload(
        {
            "protocol_sha256": "a" * 64,
            "parents": {"diagnostic": {"classification": "fixed"}},
        },
        {"fixtures": [{"fixture_id": "good"}, {"fixture_id": "bad"}]},
    )
    assert result["classification"] == (
        "fresh_unseen_fixture_confirmation_inconclusive_measurement_failure"
    )
    assert result["claim"] == "no_positive_generalization_claim"
    assert result["fixtures"] == [{"fixture_id": "good", "automatic_pass": True}]
    assert result["measurement_failures"] == [
        {"fixture_id": "bad", "failure": "RuntimeError: instrument failed"}
    ]
