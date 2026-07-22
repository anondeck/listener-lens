import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SHORTLIST_PATH = ROOT / "rules" / "kokoro-en-voice-shortlist.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_english_voice_shortlist_preserves_creator_defaults() -> None:
    shortlist = json.loads(SHORTLIST_PATH.read_text(encoding="utf-8"))

    assert shortlist["schema_version"] == 1
    assert shortlist["status"] == "creator_selected_defaults_pending_rule_qc"
    assert shortlist["selected_defaults"] == {
        "female_voice_id": "af_heart",
        "male_voice_id": "am_michael",
    }
    assert shortlist["deferred_stylistic_alternates"] == [
        {
            "voice_id": "am_puck",
            "reason_tag": "stylized_theatrical_delivery_not_default",
        }
    ]
    assert [row["voice_id"] for row in shortlist["eliminated_voices"]] == [
        "af_nicole",
        "af_bella",
        "am_fenrir",
    ]
    assert shortlist["production_candidate_enabled"] is False


def test_english_voice_shortlist_binds_frozen_screen() -> None:
    shortlist = json.loads(SHORTLIST_PATH.read_text(encoding="utf-8"))
    source = shortlist["source_screen"]
    run_root = ROOT / "artifacts" / "voice-screen" / source["run_id"]
    summary_path = run_root / "summary.json"
    manifest_path = run_root / "review" / "en" / "public-manifest.json"

    assert _sha256(summary_path) == source["summary_sha256"]
    assert _sha256(manifest_path) == source["public_manifest_sha256"]

    frozen_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert frozen_summary["voice_selection_performed"] is False


def test_existing_heart_candidate_does_not_claim_michael_evidence() -> None:
    candidate = json.loads(
        (ROOT / "rules" / "kokoro-candidate-state.json").read_text(
            encoding="utf-8"
        )
    )

    assert candidate["renderer"]["voice"] == "af_heart"
    assert candidate["production_enabled"] is False
