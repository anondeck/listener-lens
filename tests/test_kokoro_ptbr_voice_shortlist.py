import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SHORTLIST_PATH = ROOT / "rules" / "kokoro-ptbr-voice-shortlist.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_ptbr_voice_shortlist_preserves_creator_selection() -> None:
    shortlist = json.loads(SHORTLIST_PATH.read_text(encoding="utf-8"))

    assert shortlist["schema_version"] == 1
    assert shortlist["status"] == "creator_selected_pending_renderer_qc"
    assert shortlist["active_voice_ids"] == ["pm_alex", "pf_dora"]
    assert shortlist["selected_voice_id"] == "pm_alex"
    assert shortlist["selection"] == {
        "decision_basis": "creator_comparative_listening_qc",
        "preference_strength": "slight",
        "reason_tag": "preferred_expressive_dramatic_delivery",
        "alternate_voice_ids": ["pf_dora"],
        "scope": "product_voice_preference_pending_renderer_and_rule_qc",
    }
    assert shortlist["production_candidate_enabled"] is False
    assert shortlist["eliminated_voices"] == [
        {
            "voice_id": "pm_santa",
            "decision_basis": "creator_listening_qc",
            "reason_tag": "raspy_timbre",
            "scope": "product_voice_preference_not_automatic_renderer_failure",
        }
    ]


def test_ptbr_voice_shortlist_binds_frozen_screen_without_rewriting_it() -> None:
    shortlist = json.loads(SHORTLIST_PATH.read_text(encoding="utf-8"))
    source = shortlist["source_screen"]

    assert source == {
        "run_id": "20260717-kokoro-bilingual-voice-screen-v1",
        "summary_sha256": (
            "0b198eb28afc534c059ce051c306312e2605765d4cd7554c4f126548f241def1"
        ),
        "public_manifest_sha256": (
            "7016d3837d6ebfae7256acd16dc29dbf0d0b75afa307029a1f9abcdcda8362c5"
        ),
    }

    run_root = ROOT / "artifacts" / "voice-screen" / source["run_id"]
    summary_path = run_root / "summary.json"
    manifest_path = run_root / "review" / "ptbr" / "public-manifest.json"
    assert _sha256(summary_path) == source["summary_sha256"]
    assert _sha256(manifest_path) == source["public_manifest_sha256"]

    frozen_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert frozen_summary["voice_selection_performed"] is False
