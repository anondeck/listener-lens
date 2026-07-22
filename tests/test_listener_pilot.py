from earshift_bakeoff.listener_pilot import (
    EXPECTED_PROTOCOL_SHA256,
    TRIALS,
    protocol_record,
)


def test_listener_pilot_has_all_three_conditions_once() -> None:
    assert len(TRIALS) == 3
    assert {trial.condition for trial in TRIALS} == {
        "identical",
        "neutral-variance",
        "neutral-lens",
    }
    assert [trial.blind_id for trial in TRIALS] == [
        "d0bbf4c382",
        "55f78ccfde",
        "4df2d208e7",
    ]
    identical = next(trial for trial in TRIALS if trial.condition == "identical")
    assert identical.audio_a_source == identical.audio_b_source == "neutral-4"
    variance = next(
        trial for trial in TRIALS if trial.condition == "neutral-variance"
    )
    assert (variance.audio_a_source, variance.audio_b_source) == (
        "neutral-4",
        "neutral-1",
    )
    signal = next(trial for trial in TRIALS if trial.condition == "neutral-lens")
    assert (signal.audio_a_source, signal.audio_b_source) == (
        "neutral-4",
        "lens-1",
    )


def test_listener_pilot_protocol_hash_is_frozen() -> None:
    assert EXPECTED_PROTOCOL_SHA256
    assert protocol_record()["protocol_sha256"] == EXPECTED_PROTOCOL_SHA256
