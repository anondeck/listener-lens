#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence
import unicodedata
import wave

import numpy as np

from earshift_bakeoff.bilingual_listener_engine import (
    BilingualListenerPlanner,
    BilingualListenerRuntime,
    load_listener_profiles,
)
from earshift_bakeoff.bilingual_product_matrix import (
    BILINGUAL_PRODUCT_MATRIX_PATH,
    BILINGUAL_PRODUCT_STRUCTURAL_STATE_PATH,
    load_bilingual_product_matrix,
    load_bilingual_structural_state,
)
from earshift_bakeoff.bilingual_vowel_engine import (
    BilingualVowelEngineError,
    BilingualVowelRender,
    SourceAnalysis,
    SourceWord,
    _load_pinned_synthesis_voice,
)
from earshift_bakeoff.config import Paths, stable_json
from earshift_bakeoff.kokoro_gate_bridge import KokoroGateIndex
from earshift_bakeoff.kokoro_synthesis import (
    CONFIG_FILE,
    SAMPLE_RATE_HZ,
    verify_model_files,
)
from earshift_bakeoff.listener_lens import DatabaseNonceChecker
from earshift_bakeoff.portuguese_carrier_planner_v1 import (
    PortuguesePositiveOnlyIndexV1,
)
from earshift_bakeoff.util import atomic_write_json, sha256_file


PROTOCOL_VERSION = "bilingual-product-audio-integrity-screen-v1"
PROTOCOL_PATH = Paths().root / "rules" / f"{PROTOCOL_VERSION}.json"
RUN_ID = "20260717-bilingual-product-audio-integrity-screen-v1"
RUN_DIR = Paths().artifacts / "product-matrix" / RUN_ID


@dataclass(frozen=True)
class FixtureAdapter:
    language_id: str
    source_words: tuple[str, ...]
    source_phones: tuple[str, ...]
    punctuation: str

    def analyze(self, normalized_text: str) -> SourceAnalysis:
        expected_text = " ".join(self.source_words) + self.punctuation
        if normalized_text != expected_text:
            raise BilingualVowelEngineError(
                "fixture_text_drift", "Audio fixture text changed before analysis."
            )
        if len(self.source_words) != len(self.source_phones):
            raise BilingualVowelEngineError(
                "fixture_alignment_drift", "Fixture word and phone counts differ."
            )
        words = tuple(
            SourceWord(
                word_index=index,
                source=source,
                phone=unicodedata.normalize("NFD", phone),
            )
            for index, (source, phone) in enumerate(
                zip(self.source_words, self.source_phones, strict=True)
            )
        )
        separators = ("", *(" " for _ in words[1:]), self.punctuation)
        chunks = [separators[0]]
        for index, word in enumerate(words):
            chunks.extend((word.phone, separators[index + 1]))
        return SourceAnalysis(
            language_id=self.language_id,
            normalized_text=normalized_text,
            source_phonemes="".join(chunks),
            words=words,
            phone_separators=tuple(separators),
        )


def _tuple(value: Sequence[str]) -> tuple[str, ...]:
    return tuple(str(item) for item in value)


def _semantic_hash(payload: Any) -> str:
    return hashlib.sha256(stable_json(payload).encode("utf-8")).hexdigest()


def _selected_slots(manifest: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    selected: dict[str, dict[str, Any]] = {}
    for slot in manifest["slots"]:
        selected.setdefault(slot["cell_id"], slot)
    return tuple(selected.values())


def _load_protocol(
    matrix_sha256: str,
    selected_slots: tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if set(protocol) != {
        "schema_version",
        "protocol_version",
        "status",
        "production_enabled",
        "matrix_binding",
        "structural_state_binding",
        "source_manifest_binding",
        "slot_selection",
        "renderer_policy",
        "universal_gates",
        "classification_policy",
        "source_bindings",
    }:
        raise RuntimeError("audio screen protocol schema drifted")
    matrix_binding = protocol["matrix_binding"]
    structural_binding = protocol["structural_state_binding"]
    manifest_binding = protocol["source_manifest_binding"]
    manifest_path = Paths().root / manifest_binding["path"]
    if (
        protocol["schema_version"] != 1
        or protocol["protocol_version"] != PROTOCOL_VERSION
        or protocol["status"] != "frozen_before_first_audio_render"
        or protocol["production_enabled"] is not False
        or matrix_binding["path"]
        != str(BILINGUAL_PRODUCT_MATRIX_PATH.relative_to(Paths().root))
        or matrix_binding["sha256"] != sha256_file(BILINGUAL_PRODUCT_MATRIX_PATH)
        or matrix_binding["matrix_sha256"] != matrix_sha256
        or structural_binding["path"]
        != str(
            BILINGUAL_PRODUCT_STRUCTURAL_STATE_PATH.relative_to(Paths().root)
        )
        or structural_binding["sha256"]
        != sha256_file(BILINGUAL_PRODUCT_STRUCTURAL_STATE_PATH)
        or manifest_binding["sha256"] != sha256_file(manifest_path)
        or protocol["slot_selection"]["policy"]
        != "first_frozen_context_per_changed_cell"
        or protocol["slot_selection"]["logical_slot_count"] != 98
        or protocol["slot_selection"]["logical_slots_sha256"]
        != _semantic_hash([row["logical_slot_id"] for row in selected_slots])
        or protocol["renderer_policy"]["api_calls_allowed"] != 0
        or protocol["renderer_policy"]["one_valid_render_set_per_slot"] is not True
        or protocol["renderer_policy"]["replacement_slots_allowed"] is not False
        or protocol["classification_policy"]["product_promotion_allowed"]
        is not False
    ):
        raise RuntimeError("audio screen protocol binding drifted")
    for binding in protocol["source_bindings"]:
        source = Paths().root / binding["path"]
        if sha256_file(source) != binding["sha256"]:
            raise RuntimeError(f"audio source binding drifted: {binding['path']}")
    return protocol


def _write_wav(path: Path, pcm: np.ndarray) -> dict[str, Any]:
    values = np.asarray(pcm, dtype="<i2").reshape(-1)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with wave.open(str(temporary), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE_HZ)
        handle.writeframes(values.tobytes())
    temporary.replace(path)
    return {
        "relative_path": str(path.relative_to(RUN_DIR)),
        "wav_sha256": sha256_file(path),
        "pcm_sha256": hashlib.sha256(values.tobytes()).hexdigest(),
        "sample_count": int(values.size),
        "duration_s": values.size / SAMPLE_RATE_HZ,
    }


def _safe_name(value: str) -> str:
    return "".join(character if character.isalnum() else "-" for character in value)


def _target_rows(result: BilingualVowelRender, rule_id: str) -> list[dict[str, Any]]:
    rows = [
        row
        for row in result.alignment["target_occurrences"]
        if row["rule_id"] == rule_id
    ]
    if not rows and rule_id in result.plan.active_prosody_rule_ids:
        interval = (result.prosody or {}).get("sample_window")
        if interval is not None:
            rows = [
                {
                    "occurrence_index": 0,
                    "word_index": result.plan.target_word_indexes[-1],
                    "within_word_index": 0,
                    "segment_type": "prosody",
                    "rule_id": rule_id,
                    "source": None,
                    "target": None,
                    "evidence_tier": "direct_delexicalized_listener_classification",
                    "acoustic_status": "controlled_f0_candidate",
                    "control_interval": interval,
                    "measurement_interval": interval,
                }
            ]
    return rows


def _universal_pass(result: BilingualVowelRender, rows: list[dict[str, Any]]) -> bool:
    verification = result.verification
    return bool(
        rows
        and verification.neutral_identity_bit_exact
        and verification.equal_nonempty_samples
        and verification.finite
        and verification.unclipped
        and verification.outside_splice_exact_neutral
        and verification.full_weight_interior_exact_lens
        and verification.boundary_metrics_pass
        and verification.localization_pass
        and verification.integrity_pass
    )


def _render_slot(
    *,
    slot: dict[str, Any],
    profile: dict[str, Any],
    synthesis: Any,
    model_vocab: set[str],
    nonce_checker: DatabaseNonceChecker,
    phone_indexes: tuple[Any, ...],
) -> dict[str, Any]:
    fixture = slot["fixture_spec"]
    planner = BilingualListenerPlanner(
        profile={**profile, "voice_id": slot["voice_id"]},
        adapter=FixtureAdapter(
            language_id=profile["source_language"],
            source_words=_tuple(fixture["source_words"]),
            source_phones=_tuple(fixture["source_phones"]),
            punctuation=fixture["punctuation"],
        ),
        model_vocab=model_vocab,
        nonce_checker=nonce_checker,
        phone_indexes=phone_indexes,
    )
    runtime = BilingualListenerRuntime(planner=planner, synthesis=synthesis)
    rendered = runtime.render(fixture["text"])
    if not isinstance(rendered, BilingualVowelRender):
        raise BilingualVowelEngineError(
            "audio_fixture_no_comparison", "Audio fixture produced no comparison."
        )
    rows = _target_rows(rendered, slot["rule_id"])
    stem = _safe_name(slot["logical_slot_id"])
    audio = {
        "neutral": _write_wav(
            RUN_DIR / "audio" / f"{stem}__neutral.wav",
            rendered.neutral_pcm,
        ),
        "lens": _write_wav(
            RUN_DIR / "audio" / f"{stem}__lens.wav", rendered.lens_pcm
        ),
        "identity_pcm_sha256": hashlib.sha256(
            np.asarray(rendered.identity_pcm, dtype="<i2").tobytes()
        ).hexdigest(),
        "full_lens_pcm_sha256": hashlib.sha256(
            np.asarray(rendered.full_lens_pcm, dtype="<i2").tobytes()
        ).hexdigest(),
    }
    return {
        "logical_slot_id": slot["logical_slot_id"],
        "cell_id": slot["cell_id"],
        "profile_id": slot["profile_id"],
        "voice_id": slot["voice_id"],
        "family": slot["family"],
        "rule_id": slot["rule_id"],
        "context": slot["context"],
        "status": "universal_integrity_pass"
        if _universal_pass(rendered, rows)
        else "universal_integrity_fail",
        "plan_sha256": rendered.plan.plan_sha256,
        "target_occurrences": rows,
        "splice_windows": rendered.splice_windows,
        "verification": asdict(rendered.verification),
        "audio": audio,
        "family_acoustic_status": "not_classified_by_integrity_screen",
        "product_enabled": False,
        "api_calls_made": 0,
    }


def _error_outcome(
    slot: dict[str, Any],
    *,
    status: str,
    exc: Exception,
) -> dict[str, Any]:
    return {
        "logical_slot_id": slot["logical_slot_id"],
        "cell_id": slot["cell_id"],
        "profile_id": slot["profile_id"],
        "voice_id": slot["voice_id"],
        "family": slot["family"],
        "rule_id": slot["rule_id"],
        "context": slot["context"],
        "status": status,
        "error_code": getattr(exc, "code", type(exc).__name__),
        "error": str(exc),
        "product_enabled": False,
        "api_calls_made": 0,
    }


def main() -> None:
    if RUN_DIR.exists():
        raise RuntimeError(
            f"refusing to overwrite frozen audio screen directory: {RUN_DIR}"
        )
    matrix = load_bilingual_product_matrix()
    load_bilingual_structural_state(matrix)
    manifest_path = (
        Paths().artifacts
        / "product-matrix"
        / "20260717-bilingual-product-matrix-v1"
        / "manifest.json"
    )
    source_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))[
        "validation_manifest"
    ]
    slots = _selected_slots(source_manifest)
    protocol = _load_protocol(matrix.matrix_sha256, slots)
    profiles = load_listener_profiles()
    files = verify_model_files(download=False)
    model_vocab = set(
        json.loads(files[CONFIG_FILE].read_text(encoding="utf-8"))["vocab"]
    )
    nonce_checker = DatabaseNonceChecker()
    phone_indexes = (KokoroGateIndex(), PortuguesePositiveOnlyIndexV1())
    outcomes: list[dict[str, Any]] = []
    RUN_DIR.mkdir(parents=True, exist_ok=False)
    for voice_id in ("af_heart", "am_michael", "pm_alex", "pf_dora"):
        voice_slots = [slot for slot in slots if slot["voice_id"] == voice_id]
        try:
            synthesis = _load_pinned_synthesis_voice(voice_id)
        except Exception as exc:
            outcomes.extend(
                _error_outcome(
                    slot,
                    status="renderer_load_error",
                    exc=exc,
                )
                for slot in voice_slots
            )
            continue
        for slot in voice_slots:
            try:
                outcomes.append(
                    _render_slot(
                        slot=slot,
                        profile=profiles[slot["profile_id"]],
                        synthesis=synthesis,
                        model_vocab=model_vocab,
                        nonce_checker=nonce_checker,
                        phone_indexes=phone_indexes,
                    )
                )
            except Exception as exc:
                outcomes.append(
                    _error_outcome(
                        slot,
                        status="render_or_integrity_error",
                        exc=exc,
                    )
                )
    passed = sum(row["status"] == "universal_integrity_pass" for row in outcomes)
    failed = len(outcomes) - passed
    result = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "protocol_version": PROTOCOL_VERSION,
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "matrix_version": matrix.matrix_version,
        "matrix_sha256": matrix.matrix_sha256,
        "classification": (
            "all_cells_universal_integrity_pass_family_acoustics_pending"
            if failed == 0
            else "universal_integrity_yield_incomplete_family_acoustics_pending"
        ),
        "scope": "one_frozen_context_per_changed_voice_rule_cell",
        "slot_count": len(outcomes),
        "universal_integrity_pass_count": passed,
        "universal_integrity_fail_count": failed,
        "universal_integrity_yield": passed / len(outcomes),
        "family_acoustic_classification_status": "pending",
        "api_calls_made": 0,
        "audio_render_sets_made": sum("audio" in row for row in outcomes),
        "replacement_slots_used": 0,
        "production_enabled": False,
        "protocol": protocol,
        "outcomes": outcomes,
    }
    result["record_sha256"] = _semantic_hash(result)
    atomic_write_json(RUN_DIR / "results.json", result)
    print(
        json.dumps(
            {
                "output": str(RUN_DIR / "results.json"),
                "classification": result["classification"],
                "slot_count": len(outcomes),
                "universal_integrity_pass_count": passed,
                "universal_integrity_fail_count": failed,
                "api_calls_made": 0,
                "audio_render_sets_made": result["audio_render_sets_made"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
