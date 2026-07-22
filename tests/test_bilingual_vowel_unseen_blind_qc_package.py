from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from earshift_bakeoff.bilingual_vowel_unseen_blind_qc import (
    PROTOCOL_PATH,
    RUN_DIR,
    RUN_ID,
    SESSION_LABELS,
    adjudicate_session_response,
)
from earshift_bakeoff.config import stable_json
from earshift_bakeoff.util import sha256_file


PRIVATE = RUN_DIR / "private-manifest.json"
HUB = RUN_DIR / "public-hub.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _semantic_hash(payload: dict) -> str:
    semantic = dict(payload)
    semantic.pop("record_sha256", None)
    return hashlib.sha256(stable_json(semantic).encode("utf-8")).hexdigest()


def test_blind_qc_package_has_exact_frozen_denominators() -> None:
    private = _load(PRIVATE)
    hub = _load(HUB)

    assert private["record_sha256"] == _semantic_hash(private)
    assert private["record_sha256"] == (
        "0976cce0396b8715b562bd7f039ece0faeb6e553e3073ca8ac88c1f6660efba4"
    )
    assert private["classification"] == (
        "blind_review_ready_no_human_result_no_product_promotion"
    )
    assert private["automatic_candidate_cell_count"] == 18
    assert private["trial_kind_counts"] == {"candidate": 54, "identity": 18}
    assert private["voice_trial_counts"] == {
        "af_heart": 24,
        "am_michael": 32,
        "pf_dora": 8,
        "pm_alex": 8,
    }
    assert len(private["trials"]) == hub["total_trial_count"] == 72
    assert private["new_audio_renders_made"] == 0
    assert private["api_calls_made"] == 0
    assert private["human_review_complete"] is False
    assert private["production_enabled"] is False
    assert all(row["product_enabled"] is False for row in private["trials"])
    assert hub["record_sha256"] == _semantic_hash(hub)
    assert hub["session_count"] == 4


def test_blind_qc_package_has_three_candidates_and_one_identity_per_cell() -> None:
    private = _load(PRIVATE)
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in private["trials"]:
        grouped[row["cell_id"]].append(row)

    assert len(grouped) == 18
    for rows in grouped.values():
        assert Counter(row["trial_kind"] for row in rows) == {
            "candidate": 3,
            "identity": 1,
        }
        assert {
            row["context"] for row in rows if row["trial_kind"] == "candidate"
        } == {
            "real_g2p_phrase_medial",
            "real_g2p_phrase_final",
            "real_g2p_repeated_target",
        }
        assert len({row["voice_id"] for row in rows}) == 1
        assert len({row["rule_id"] for row in rows}) == 1


def test_blind_audio_copies_and_target_cues_preserve_frozen_sources() -> None:
    private = _load(PRIVATE)

    for row in private["trials"]:
        for side in ("A", "B"):
            receipt = row["audio_receipts"][side]
            assert receipt["source_sha256"] == receipt["blind_sha256"]
            assert sha256_file(RUN_DIR / receipt["blind_path"]) == receipt[
                "blind_sha256"
            ]
        hashes = {
            side: row["audio_receipts"][side]["blind_sha256"]
            for side in ("A", "B")
        }
        if row["trial_kind"] == "identity":
            assert hashes["A"] == hashes["B"]
            assert row["expected_direction"] == "same"
        else:
            assert hashes["A"] != hashes["B"]
            assert row["expected_direction"] in {"A", "B"}
        assert len(row["target_intervals"]) == row["fixture_spec"][
            "expected_target_occurrence_count"
        ]
        assert all(
            interval["end_s"] > interval["start_s"]
            for interval in row["target_intervals"]
        )


def test_public_review_contains_no_private_condition_identifiers() -> None:
    private = _load(PRIVATE)
    hub = _load(HUB)
    public_text = (RUN_DIR / "index.html").read_text(encoding="utf-8")
    for session in hub["sessions"]:
        public_manifest = _load(RUN_DIR / session["public_manifest_path"])
        assert public_manifest["record_sha256"] == _semantic_hash(public_manifest)
        assert set(public_manifest["trials"][0]) == {
            "trial_id",
            "audio",
            "duration_s",
            "target_intervals",
            "direction_prompt",
        }
        public_text += (RUN_DIR / session["review_path"]).read_text(encoding="utf-8")
        public_text += (RUN_DIR / session["public_manifest_path"]).read_text(
            encoding="utf-8"
        )

    for row in private["trials"]:
        assert row["cell_id"] not in public_text
        assert row["rule_id"] not in public_text
        assert row["voice_id"] not in public_text
        assert row["logical_slot_id"] not in public_text
        assert row["fixture_spec"]["text"] not in public_text
        assert Path(row["audio_receipts"]["A"]["source_path"]).name not in public_text
        assert Path(row["audio_receipts"]["B"]["source_path"]).name not in public_text
    assert "shit" not in public_text.casefold()
    assert "neutral" not in public_text.casefold()
    assert "lens" not in public_text.casefold()


def _passing_response(session_id: str) -> dict:
    private = _load(PRIVATE)
    public_path = RUN_DIR / session_id / "public-manifest.json"
    rows = [row for row in private["trials"] if row["session_id"] == session_id]
    return {
        "schema_version": 1,
        "run_id": RUN_ID,
        "session_id": session_id,
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "public_manifest_sha256": sha256_file(public_path),
        "session_uuid": "test-session",
        "saved_at": "2026-07-18T00:00:00Z",
        "reviewer": {
            "reviewer": "test",
            "language_background": "test background",
            "listening_setup": "speakers",
        },
        "ratings": [
            {
                "trial_id": row["trial_id"],
                "sides": {
                    "A": {
                        "naturalness": "4",
                        "sentence_delivery": "sentence_like",
                        "stable_meaning": "none",
                        "artifact": "none",
                    },
                    "B": {
                        "naturalness": "4",
                        "sentence_delivery": "sentence_like",
                        "stable_meaning": "none",
                        "artifact": "none",
                    },
                },
                "difference_strength": 1 if row["trial_kind"] == "identity" else 5,
                "target_direction": row["expected_direction"],
                "confidence": 3,
                "unrelated_interference": "none",
                "notes": "",
                "play_starts": {"A": 1, "B": 1},
                "replay_count": 2,
            }
            for row in rows
        ],
    }


def test_adjudicator_accepts_complete_passing_sessions(tmp_path: Path) -> None:
    expected_cells = {"session-a": 2, "session-b": 2, "session-c": 8, "session-d": 6}

    for session_id in SESSION_LABELS:
        response_path = tmp_path / f"{session_id}.json"
        response_path.write_text(
            json.dumps(_passing_response(session_id)), encoding="utf-8"
        )
        result = adjudicate_session_response(response_path)
        assert result["record_sha256"] == _semantic_hash(result)
        assert result["classification"] == (
            "human_qc_session_adjudicated_no_product_promotion"
        )
        assert result["human_qc_pass_cell_count"] == expected_cells[session_id]
        assert result["identity_control_clean_count"] == expected_cells[session_id]
        assert result["production_enabled"] is False


def test_identity_false_alarm_flags_only_its_cell(tmp_path: Path) -> None:
    response = _passing_response("session-a")
    private = _load(PRIVATE)
    identity_id = next(
        row["trial_id"]
        for row in private["trials"]
        if row["session_id"] == "session-a" and row["trial_kind"] == "identity"
    )
    rating = next(row for row in response["ratings"] if row["trial_id"] == identity_id)
    rating["difference_strength"] = 3
    rating["target_direction"] = "A"
    path = tmp_path / "false-alarm.json"
    path.write_text(json.dumps(response), encoding="utf-8")

    result = adjudicate_session_response(path)

    assert result["identity_control_clean_count"] == 1
    assert result["human_qc_pass_cell_count"] == 1
    flagged = [
        row
        for row in result["trial_results"]
        if row["status"] == "identity_control_investigation_required"
    ]
    assert len(flagged) == 1
