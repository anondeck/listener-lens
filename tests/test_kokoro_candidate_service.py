from __future__ import annotations

import pytest

from earshift_bakeoff.kokoro_candidate_service import (
    KokoroCandidateRuntime,
    load_candidate_state,
)
from earshift_bakeoff.kokoro_strict_shell import StrictShellPlanner
from earshift_bakeoff.kokoro_typed_engine import KokoroTypedEngineError


class MustNotSynthesize:
    def render_parity_triplet(self, _plan):  # pragma: no cover - failure sentinel
        raise AssertionError("unsupported input reached synthesis")


def test_candidate_state_binds_the_automatic_pass_and_stays_disabled() -> None:
    state = load_candidate_state()
    assert state["candidate_id"] == ("kokoro-en-ae-to-eh-strict-shell-output-splice-v1")
    assert state["enabled_by_default"] is False
    assert state["production_enabled"] is False
    assert state["evidence"]["automatic_status"] == "pass"
    assert state["evidence"]["human_status"] == "pending"
    assert state["evidence"]["production_promotion"] is False
    assert len(state["local_anchor_geometry"]) == 3
    assert state["renderer"]["voice"] == "af_heart"
    assert state["voice_registry_version"] == "kokoro-product-voices-v1"
    assert len(state["voice_registry_sha256"]) == 64
    assert state["voice_registry"] == {
        "version": state["voice_registry_version"],
        "sha256": state["voice_registry_sha256"],
    }


def test_no_rule_and_unvalidated_positions_stop_before_synthesis() -> None:
    runtime = KokoroCandidateRuntime(
        state=load_candidate_state(),
        planner=StrictShellPlanner.load(),
        synthesis=MustNotSynthesize(),  # type: ignore[arg-type]
    )

    no_rule = runtime.render("This word is good.")
    assert no_rule["status"] == "no_supported_sounds"
    assert no_rule["transform"]["comparison_available"] is False
    assert no_rule["api_calls_made"] == 0

    with pytest.raises(KokoroTypedEngineError) as caught:
        runtime.render("Cat moves quietly.")
    assert caught.value.code == "strict_shell_unsupported_target_position"

    with pytest.raises(KokoroTypedEngineError) as caught:
        runtime.render("We move slowly past fields.")
    assert caught.value.code == "strict_shell_unsupported_target_word"


def test_candidate_rejects_unvalidated_voice_before_synthesis() -> None:
    runtime = KokoroCandidateRuntime(
        state=load_candidate_state(),
        planner=StrictShellPlanner.load(),
        synthesis=MustNotSynthesize(),  # type: ignore[arg-type]
    )

    with pytest.raises(RuntimeError, match="no evidence"):
        runtime.render("Quiet voices map distant roads.", voice_id="am_michael")
