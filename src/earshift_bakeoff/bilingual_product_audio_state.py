from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .bilingual_product_matrix import BilingualProductMatrixError
from .config import ROOT
from .util import sha256_file


AUDIO_INTEGRITY_STATE_PATH = (
    ROOT / "rules" / "bilingual-product-audio-integrity-state-v1.json"
)
AUDIO_INTEGRITY_STATE_VERSION = "bilingual-product-audio-integrity-state-v1"
ISOLATED_AUDIO_STATE_PATH = (
    ROOT / "rules" / "bilingual-product-isolated-audio-state-v1.json"
)
ISOLATED_AUDIO_STATE_VERSION = "bilingual-product-isolated-audio-state-v1"


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise BilingualProductMatrixError(
            "invalid_audio_integrity_state", "Audio integrity state must be an object."
        )
    return value


def load_bilingual_audio_integrity_state(
    *,
    matrix_version: str,
    matrix_sha256: str,
    path: Path = AUDIO_INTEGRITY_STATE_PATH,
    verify_result_artifact: bool = True,
) -> dict[str, Any]:
    state = _load(path)
    expected_keys = {
        "schema_version",
        "state_version",
        "matrix_version",
        "matrix_sha256",
        "protocol_binding",
        "result_binding",
        "classification",
        "slot_count",
        "universal_integrity_pass_count",
        "universal_integrity_fail_count",
        "universal_integrity_yield",
        "family_counts",
        "voice_counts",
        "api_calls_made",
        "audio_render_sets_made",
        "family_acoustic_validation_status",
        "human_validation_status",
        "production_enabled",
    }
    if set(state) != expected_keys:
        raise BilingualProductMatrixError(
            "invalid_audio_integrity_state", "Audio integrity state schema drifted."
        )
    if (
        state["schema_version"] != 1
        or state["state_version"] != AUDIO_INTEGRITY_STATE_VERSION
        or state["matrix_version"] != matrix_version
        or state["matrix_sha256"] != matrix_sha256
        or state["classification"]
        != "all_cells_universal_integrity_pass_family_acoustics_pending"
        or state["slot_count"] != 98
        or state["universal_integrity_pass_count"] != 98
        or state["universal_integrity_fail_count"] != 0
        or state["universal_integrity_yield"] != 1.0
        or state["family_counts"]
        != {"vowel": 80, "consonant": 12, "insertion": 2, "prosody": 4}
        or state["voice_counts"]
        != {"af_heart": 23, "am_michael": 23, "pm_alex": 26, "pf_dora": 26}
        or state["api_calls_made"] != 0
        or state["audio_render_sets_made"] != 98
        or state["family_acoustic_validation_status"] != "pending"
        or state["human_validation_status"]
        != "pending_automatic_family_acoustics"
        or state["production_enabled"] is not False
    ):
        raise BilingualProductMatrixError(
            "audio_integrity_state_drift",
            "Audio integrity state no longer binds its universal pass.",
        )
    protocol_binding = state["protocol_binding"]
    result_binding = state["result_binding"]
    if set(protocol_binding) != {"path", "sha256", "protocol_version"} or set(
        result_binding
    ) != {"path", "sha256", "record_sha256"}:
        raise BilingualProductMatrixError(
            "invalid_audio_integrity_binding", "Audio integrity bindings drifted."
        )
    protocol_path = (ROOT / protocol_binding["path"]).resolve()
    result_path = (ROOT / result_binding["path"]).resolve()
    root = ROOT.resolve()
    if (
        not protocol_path.is_relative_to(root)
        or not result_path.is_relative_to(root)
        or sha256_file(protocol_path) != protocol_binding["sha256"]
    ):
        raise BilingualProductMatrixError(
            "audio_integrity_binding_drift", "Audio integrity protocol drifted."
        )
    protocol = _load(protocol_path)
    if protocol.get("protocol_version") != protocol_binding["protocol_version"]:
        raise BilingualProductMatrixError(
            "audio_integrity_protocol_drift", "Audio integrity protocol version drifted."
        )
    if not verify_result_artifact:
        return state
    if sha256_file(result_path) != result_binding["sha256"]:
        raise BilingualProductMatrixError(
            "audio_integrity_result_drift", "Audio integrity result file drifted."
        )
    result = _load(result_path)
    if (
        result.get("record_sha256") != result_binding["record_sha256"]
        or result.get("matrix_sha256") != matrix_sha256
        or result.get("classification") != state["classification"]
        or result.get("slot_count") != state["slot_count"]
        or result.get("universal_integrity_pass_count")
        != state["universal_integrity_pass_count"]
        or result.get("universal_integrity_fail_count")
        != state["universal_integrity_fail_count"]
        or result.get("universal_integrity_yield")
        != state["universal_integrity_yield"]
        or result.get("api_calls_made") != 0
        or result.get("audio_render_sets_made") != 98
        or result.get("production_enabled") is not False
    ):
        raise BilingualProductMatrixError(
            "audio_integrity_result_semantic_drift",
            "Audio integrity result semantics no longer match the state record.",
        )
    return state


def load_bilingual_isolated_audio_state(
    *,
    matrix_version: str,
    matrix_sha256: str,
    path: Path = ISOLATED_AUDIO_STATE_PATH,
    verify_result_artifact: bool = True,
) -> dict[str, Any]:
    state = _load(path)
    expected_keys = {
        "schema_version",
        "state_version",
        "matrix_version",
        "matrix_sha256",
        "protocol_binding",
        "result_binding",
        "classification",
        "slot_count",
        "isolated_universal_integrity_pass_count",
        "isolated_universal_integrity_fail_count",
        "isolated_universal_integrity_yield",
        "family_counts",
        "voice_counts",
        "api_calls_made",
        "audio_render_sets_made",
        "family_acoustic_validation_status",
        "human_validation_status",
        "production_enabled",
    }
    if set(state) != expected_keys:
        raise BilingualProductMatrixError(
            "invalid_isolated_audio_state", "Isolated audio state schema drifted."
        )
    if (
        state["schema_version"] != 1
        or state["state_version"] != ISOLATED_AUDIO_STATE_VERSION
        or state["matrix_version"] != matrix_version
        or state["matrix_sha256"] != matrix_sha256
        or state["classification"]
        != "all_isolated_slots_universal_integrity_pass_family_acoustics_pending"
        or state["slot_count"] != 280
        or state["isolated_universal_integrity_pass_count"] != 280
        or state["isolated_universal_integrity_fail_count"] != 0
        or state["isolated_universal_integrity_yield"] != 1.0
        or state["family_counts"]
        != {"vowel": 240, "consonant": 32, "insertion": 4, "prosody": 4}
        or state["voice_counts"]
        != {"af_heart": 66, "am_michael": 66, "pm_alex": 74, "pf_dora": 74}
        or state["api_calls_made"] != 0
        or state["audio_render_sets_made"] != 280
        or state["family_acoustic_validation_status"] != "pending"
        or state["human_validation_status"]
        != "pending_automatic_family_acoustics"
        or state["production_enabled"] is not False
    ):
        raise BilingualProductMatrixError(
            "isolated_audio_state_drift",
            "Isolated audio state no longer binds its universal pass.",
        )
    protocol_binding = state["protocol_binding"]
    result_binding = state["result_binding"]
    if set(protocol_binding) != {"path", "sha256", "protocol_version"} or set(
        result_binding
    ) != {"path", "sha256", "record_sha256"}:
        raise BilingualProductMatrixError(
            "invalid_isolated_audio_binding", "Isolated audio bindings drifted."
        )
    protocol_path = (ROOT / protocol_binding["path"]).resolve()
    result_path = (ROOT / result_binding["path"]).resolve()
    root = ROOT.resolve()
    if (
        not protocol_path.is_relative_to(root)
        or not result_path.is_relative_to(root)
        or sha256_file(protocol_path) != protocol_binding["sha256"]
    ):
        raise BilingualProductMatrixError(
            "isolated_audio_binding_drift", "Isolated audio protocol drifted."
        )
    protocol = _load(protocol_path)
    if protocol.get("protocol_version") != protocol_binding["protocol_version"]:
        raise BilingualProductMatrixError(
            "isolated_audio_protocol_drift", "Isolated audio protocol version drifted."
        )
    if not verify_result_artifact:
        return state
    if sha256_file(result_path) != result_binding["sha256"]:
        raise BilingualProductMatrixError(
            "isolated_audio_result_drift", "Isolated audio result file drifted."
        )
    result = _load(result_path)
    if (
        result.get("record_sha256") != result_binding["record_sha256"]
        or result.get("matrix_sha256") != matrix_sha256
        or result.get("classification") != state["classification"]
        or result.get("slot_count") != state["slot_count"]
        or result.get("isolated_universal_integrity_pass_count")
        != state["isolated_universal_integrity_pass_count"]
        or result.get("isolated_universal_integrity_fail_count")
        != state["isolated_universal_integrity_fail_count"]
        or result.get("isolated_universal_integrity_yield")
        != state["isolated_universal_integrity_yield"]
        or result.get("api_calls_made") != 0
        or result.get("audio_render_sets_made") != 280
        or result.get("production_enabled") is not False
    ):
        raise BilingualProductMatrixError(
            "isolated_audio_result_semantic_drift",
            "Isolated audio result semantics no longer match the state record.",
        )
    return state
