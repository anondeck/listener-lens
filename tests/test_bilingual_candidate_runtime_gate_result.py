from __future__ import annotations

from collections import Counter
import hashlib
import json

from earshift_bakeoff.config import ROOT, stable_json
from earshift_bakeoff.util import sha256_file


RUN_DIR = (
    ROOT
    / "artifacts"
    / "product-matrix"
    / "20260718-bilingual-candidate-runtime-gate-v1"
)
RESULT = RUN_DIR / "results.json"
SCALERS = RUN_DIR / "voice-scalers.json"


def _load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _semantic_hash(payload: dict) -> str:
    semantic = dict(payload)
    semantic.pop("record_sha256", None)
    return hashlib.sha256(stable_json(semantic).encode("utf-8")).hexdigest()


def test_runtime_gate_result_is_complete_integral_and_nonpromotional() -> None:
    result = _load(RESULT)

    assert result["record_sha256"] == _semantic_hash(result)
    assert result["record_sha256"] == (
        "d53f4f2172c443d82834524d3621f1cfec48c72af8b0d6b0fbe4b82a268f8964"
    )
    assert result["classification"] == (
        "runtime_gate_complete_no_product_promotion"
    )
    assert result["production_enabled"] is False
    assert result["api_calls_made"] == 0
    assert result["candidate_audio_rerenders_made"] == 0
    assert result["natural_decoder_render_count"] == 504
    assert result["logical_slot_count"] == 84
    assert result["target_occurrence_count"] == 112


def test_runtime_gate_preserves_all_eighteen_prior_passes_without_rescue() -> None:
    result = _load(RESULT)

    assert result["prior_unseen_pass_count"] == 18
    assert result["runtime_gate_pass_count"] == 18
    assert result["lost_prior_pass_cell_ids"] == []
    assert Counter(
        row["voice_id"]
        for row in result["cell_results"]
        if row["runtime_gate_pass"]
    ) == {
        "af_heart": 6,
        "am_michael": 8,
        "pm_alex": 2,
        "pf_dora": 2,
    }
    assert not any(
        row["runtime_gate_pass"] and not row["prior_unseen_pass"]
        for row in result["cell_results"]
    )
    assert all(row["product_enabled"] is False for row in result["cell_results"])


def test_runtime_gate_has_zero_identity_false_positives_and_complete_scalers() -> None:
    result = _load(RESULT)
    scalers = _load(SCALERS)

    assert result["identity_negative_control_count"] == 112
    assert result["identity_negative_control_false_positive_count"] == 0
    assert all(
        not row["identity_negative_control"]["directional_pass"]
        for row in result["occurrence_results"]
    )
    assert result["voice_scaler_sha256"] == sha256_file(SCALERS)
    assert result["voice_scaler_record_sha256"] == scalers["record_sha256"]
    assert scalers["record_sha256"] == _semantic_hash(scalers)
    assert set(scalers["voice_scalers"]) == {
        "af_heart",
        "am_michael",
        "pm_alex",
        "pf_dora",
    }
    assert all(
        scaler["feature_size"] == 36
        and scaler["observation_count"] >= 120
        and len(scaler["center"]) == len(scaler["scale"]) == 36
        and all(value > 0 for value in scaler["scale"])
        for scaler in scalers["voice_scalers"].values()
    )
