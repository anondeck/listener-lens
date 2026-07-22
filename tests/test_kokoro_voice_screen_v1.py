from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from earshift_bakeoff.kokoro_specs import (
    ENGLISH_SCREEN_SHORTLIST,
    PORTUGUESE_SCREEN_VOICES,
)
from earshift_bakeoff.kokoro_voice_screen_v1 import (
    ENGLISH_RESPONSE_FILENAME,
    PORTUGUESE_RESPONSE_FILENAME,
    RUN_ID,
    FixturePlan,
    RenderOutput,
    VoiceScreenError,
    assemble_protocol,
    build_fixture_plans,
    inspect_audio,
    render_screen,
    verify_protocol,
)
from earshift_bakeoff.util import atomic_write_json, sha256_bytes


_VOCAB = {"a": 1, "b": 2}


def _fixture(
    fixture_id: str,
    language_id: str,
    fixture_kind: str,
) -> dict:
    text = f"{fixture_id} source"
    return FixturePlan(
        fixture_id=fixture_id,
        language_id=language_id,
        fixture_kind=fixture_kind,
        source_text=text,
        render_text=text,
        phonemes="ab",
        g2p_tokens=(),
        gate_receipt={"pass": True} if fixture_kind == "opaque-carrier" else None,
    ).record(_VOCAB)


def _fixtures() -> list[dict]:
    return [
        _fixture("en-real-1", "en-US", "real-passage"),
        _fixture("en-real-2", "en-US", "real-passage"),
        _fixture("en-opaque", "en-US", "opaque-carrier"),
        _fixture("pt-real-1", "pt-BR", "real-passage"),
        _fixture("pt-real-2", "pt-BR", "real-passage"),
        _fixture("pt-opaque", "pt-BR", "opaque-carrier"),
    ]


def _protocol() -> dict:
    return assemble_protocol(
        fixtures=_fixtures(),
        asset_bindings={
            "source": {"sha256": "1" * 64},
            "model": {"sha256": "2" * 64},
            "voices": {"sha256": "3" * 64},
            "g2p": {"sha256": "4" * 64},
            "gates": {"sha256": "5" * 64},
            "code": {"sha256": "6" * 64},
        },
    )


def _sine() -> np.ndarray:
    sample_count = 6_000
    return np.asarray(
        [
            0.15 * math.sin(2 * math.pi * 220 * index / 24_000)
            for index in range(sample_count)
        ],
        dtype=np.float32,
    )


class _Runtime:
    model_vocab = _VOCAB

    def render(self, *, voice_id: str, phonemes: str) -> RenderOutput:
        assert voice_id in (*ENGLISH_SCREEN_SHORTLIST, *PORTUGUESE_SCREEN_VOICES)
        assert phonemes == "ab"
        return RenderOutput(audio=_sine(), predicted_durations=(1, 2, 3, 4))


def test_protocol_freezes_exact_voice_fixture_and_repeat_counts() -> None:
    protocol = _protocol()

    assert verify_protocol(protocol) == protocol["protocol_sha256"]
    assert protocol["status"] == "frozen-before-render"
    assert len(protocol["render_manifest"]) == 54
    assert protocol["scope"]["english_voices"] == list(ENGLISH_SCREEN_SHORTLIST)
    assert protocol["scope"]["portuguese_voices"] == list(PORTUGUESE_SCREEN_VOICES)
    assert protocol["candidate_flags"] == {
        "KOKORO_ENGLISH_CANDIDATE_ENABLED": False,
        "PORTUGUESE_RENDERER_CANDIDATE_ENABLED": False,
        "production_selection_performed": False,
    }
    assert all(slot["maximum_attempts"] == 1 for slot in protocol["render_manifest"])
    assert all(not slot["retry_allowed"] for slot in protocol["render_manifest"])
    assert all(not slot["replacement_allowed"] for slot in protocol["render_manifest"])
    groups: dict[str, list[str]] = {}
    for slot in protocol["render_manifest"]:
        groups.setdefault(slot["determinism_group"], []).append(slot["render_role"])
    assert len(groups) == 27
    assert set(map(tuple, groups.values())) == {("primary", "determinism-repeat")}


def test_protocol_hash_rejects_any_post_freeze_change() -> None:
    protocol = _protocol()
    protocol["renderer"]["speed"] = 1.01

    with pytest.raises(VoiceScreenError, match="hash mismatch"):
        verify_protocol(protocol)


def test_signal_metrics_include_spectral_and_exact_duration_adherence() -> None:
    metrics = inspect_audio(
        audio=_sine(),
        predicted_durations=(1, 2, 3, 4),
        phonemes="ab",
        model_vocab=_VOCAB,
    )

    assert metrics["integrity_pass"] is True
    assert metrics["duration_s"] == 0.25
    assert metrics["samples_per_duration_frame"] == 600
    assert metrics["expected_sample_count_from_duration_plan"] == 6_000
    assert 219 <= metrics["spectral_centroid_hz"] <= 221
    assert metrics["spectral_rolloff_95_hz"] > 0
    assert metrics["rms"] > 0
    assert metrics["peak"] > 0
    assert abs(metrics["dc_offset"]) < 1e-3


@pytest.mark.parametrize(
    ("audio", "durations", "failed_check"),
    [
        (np.full(6_000, np.nan), (1, 2, 3, 4), "finite_pass"),
        (np.ones(6_000), (1, 2, 3, 4), "clipping_pass"),
        (_sine(), (1, 2, 3, 3), "duration_plan_pass"),
    ],
)
def test_signal_integrity_is_fail_closed(
    audio: np.ndarray,
    durations: tuple[int, ...],
    failed_check: str,
) -> None:
    metrics = inspect_audio(
        audio=audio,
        predicted_durations=durations,
        phonemes="ab",
        model_vocab=_VOCAB,
    )

    assert metrics["integrity_pass"] is False
    assert metrics["checks"][failed_check] is False


def test_render_is_one_shot_and_builds_separate_blind_sessions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol = _protocol()
    atomic_write_json(tmp_path / "protocol.json", protocol)
    monkeypatch.setattr(
        "earshift_bakeoff.kokoro_voice_screen_v1.protocol_record",
        lambda screen_dir: protocol,
    )

    summary = render_screen(
        screen_dir=tmp_path,
        runtime=_Runtime(),
        require_committed_protocol=False,
    )

    assert summary["status"] == "pending-human-review"
    assert summary["automatic_integrity_pass"] is True
    assert summary["integrity_pass_count"] == 54
    assert summary["determinism"]["pair_count"] == 27
    assert summary["determinism"]["pass_count"] == 27
    assert summary["voice_selection_performed"] is False
    assert [review["response_filename"] for review in summary["human_reviews"]] == [
        ENGLISH_RESPONSE_FILENAME,
        PORTUGUESE_RESPONSE_FILENAME,
    ]

    english_public_text = (tmp_path / "review/en/public-manifest.json").read_text()
    portuguese_public_text = (tmp_path / "review/ptbr/public-manifest.json").read_text()
    english_html = (tmp_path / "review/en/review.html").read_text()
    portuguese_html = (tmp_path / "review/ptbr/review.html").read_text()
    assert ENGLISH_RESPONSE_FILENAME in english_html
    assert PORTUGUESE_RESPONSE_FILENAME in portuguese_html
    assert "naturalness" in english_html
    assert "accent_fit" in english_html
    assert "sentence_flow" in english_html
    assert "clarity" in english_html
    assert "artifacts" in english_html
    assert "nonce_handling" in english_html
    assert "language_id:'en-US'" in english_html
    assert "language_id:'pt-BR'" in portuguese_html
    assert len(json.loads(english_public_text)["clips"]) == 18
    assert len(json.loads(portuguese_public_text)["clips"]) == 9
    for voice_id in (*ENGLISH_SCREEN_SHORTLIST, *PORTUGUESE_SCREEN_VOICES):
        assert voice_id not in english_public_text
        assert voice_id not in portuguese_public_text
        assert voice_id not in english_html
        assert voice_id not in portuguese_html

    english_key = json.loads((tmp_path / "private/en-blind-key.json").read_text())
    portuguese_key = json.loads((tmp_path / "private/ptbr-blind-key.json").read_text())
    assert {row["voice_id"] for row in english_key["mapping"].values()} == set(
        ENGLISH_SCREEN_SHORTLIST
    )
    assert {row["voice_id"] for row in portuguese_key["mapping"].values()} == set(
        PORTUGUESE_SCREEN_VOICES
    )

    with pytest.raises(VoiceScreenError, match="already started"):
        render_screen(
            screen_dir=tmp_path,
            runtime=_Runtime(),
            require_committed_protocol=False,
        )


def test_bit_determinism_is_pcm_and_complete_wav_exact(tmp_path: Path) -> None:
    protocol = _protocol()
    atomic_write_json(tmp_path / "protocol.json", protocol)

    first = _Runtime().render(voice_id="af_heart", phonemes="ab")
    second = _Runtime().render(voice_id="af_heart", phonemes="ab")

    assert first.predicted_durations == second.predicted_durations
    assert sha256_bytes(first.audio.tobytes()) == sha256_bytes(second.audio.tobytes())


def test_fixture_plan_rejects_unknown_g2p_symbols() -> None:
    fixture = FixturePlan(
        fixture_id="bad",
        language_id="en-US",
        fixture_kind="real-passage",
        source_text="text",
        render_text="text",
        phonemes="a☃",
        g2p_tokens=(),
        gate_receipt=None,
    )

    with pytest.raises(VoiceScreenError, match="unsupported symbols"):
        fixture.record(_VOCAB)


def test_english_and_portuguese_negative_index_semantics_stay_distinct() -> None:
    fixtures = {fixture.fixture_id: fixture for fixture in build_fixture_plans()}
    english = fixtures["en-opaque-carrier"].gate_receipt
    portuguese = fixtures["ptbr-opaque-carrier"].gate_receipt

    assert english is not None
    assert english["english_complete_kokoro_index_negative_used_for_clearance"] is True
    assert "native_negative_used_for_clearance" not in english
    assert "Fallback-only negatives" in english["english_complete_kokoro_index_scope"]
    assert portuguese is not None
    assert all(
        receipt["gate_receipt"]["native_negative_used_for_clearance"] is False
        for receipt in portuguese["per_voice_receipts"].values()
    )
    assert all(
        receipt["gate_receipt"]["native_index_scope"] == "partial_positive_only_index"
        for receipt in portuguese["per_voice_receipts"].values()
    )


def test_run_identity_and_response_names_are_frozen() -> None:
    assert RUN_ID == "20260717-kokoro-bilingual-voice-screen-v1"
    assert ENGLISH_RESPONSE_FILENAME == "kokoro-en-voice-screen-v1-response.json"
    assert PORTUGUESE_RESPONSE_FILENAME == ("kokoro-ptbr-voice-screen-v1-response.json")
