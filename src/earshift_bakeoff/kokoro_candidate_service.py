from __future__ import annotations

import base64
import hashlib
import io
import json
import re
import tempfile
import time
import wave
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .config import ROOT
from .kokoro_output_domain_splice import (
    LOCALIZATION_MINIMUM,
    MAX_BOUNDARY_DERIVATIVE_RATIO,
    MAX_EDGE_DELTA_STEP_PCM,
    MAX_LOCALIZATION_P95_MS,
    TAPER_MS,
    boundary_artifact_report,
    output_domain_splice,
)
from .kokoro_output_splice_unseen import (
    _acoustic_report,
    _word_intervals,
    phrase_medial_edge_gate,
)
from .kokoro_strict_shell import STRICT_SHELL_VERSION, StrictShellPlanner
from .kokoro_synthesis import (
    KOKORO_VERSION,
    MODEL_REPO,
    MODEL_REVISION,
    RNG_SEED,
    SAMPLE_RATE_HZ,
    KokoroSynthesisRuntime,
    PairRender,
    ParityRender,
    pcm16_bytes,
    target_word_columns,
)
from .kokoro_typed_confirmation import alignment_record
from .kokoro_typed_confirmation_protocol import MEASUREMENT_SCRIPT, PRAAT
from .kokoro_typed_diagnostic import localization_report
from .kokoro_typed_engine import (
    MAX_CLIPPED_FRACTION,
    RULE_ID,
    KokoroTypedEngineError,
    TypedPlan,
    inspect_render,
)
from .product_voices import load_product_voice_registry
from .util import sha256_file


CANDIDATE_STATE_PATH = ROOT / "rules" / "kokoro-candidate-state.json"
NO_RULE_MESSAGE = (
    "We don't yet support any sounds in this sentence. Try a sentence with the "
    "vowel in “cat,” such as “Quiet voices map distant roads.”"
)


class KokoroCandidateError(RuntimeError):
    """The exact controlled candidate failed closed."""


class KokoroCandidateGateError(KokoroCandidateError):
    """A runtime evidence gate rejected the generated pair."""


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise KokoroCandidateError(f"expected an object in {path}")
    return value


def load_candidate_state() -> dict[str, Any]:
    state = _load_json(CANDIDATE_STATE_PATH)
    expected_top_level = {
        "schema_version",
        "candidate_id",
        "feature_flag",
        "enabled_by_default",
        "production_enabled",
        "service_contract_version",
        "profile_id",
        "voice_registry",
        "rule_ids",
        "planner",
        "renderer",
        "splice",
        "evidence",
    }
    if set(state) != expected_top_level:
        raise KokoroCandidateError("candidate state has an unexpected schema")
    if (
        state["schema_version"] != 1
        or state["feature_flag"] != "KOKORO_ENGLISH_CANDIDATE_ENABLED"
        or state["enabled_by_default"] is not False
        or state["production_enabled"] is not False
        or state["rule_ids"] != [RULE_ID]
        or state["planner"].get("version") != STRICT_SHELL_VERSION
        or state["renderer"].get("version") != KOKORO_VERSION
        or state["renderer"].get("model_repo") != MODEL_REPO
        or state["renderer"].get("model_revision") != MODEL_REVISION
        or state["renderer"].get("sample_rate_hz") != SAMPLE_RATE_HZ
        or state["renderer"].get("rng_seed") != RNG_SEED
        or state["splice"].get("taper_ms_each_edge") != TAPER_MS
        or state["splice"].get("localization_minimum") != LOCALIZATION_MINIMUM
        or state["splice"].get("maximum_edge_delta_step_pcm")
        != MAX_EDGE_DELTA_STEP_PCM
        or state["splice"].get("maximum_boundary_derivative_ratio")
        != MAX_BOUNDARY_DERIVATIVE_RATIO
        or state["evidence"].get("automatic_status") != "pass"
        or state["evidence"].get("human_status") not in {"pending", "pass"}
        or state["evidence"].get("production_promotion") is not False
    ):
        raise KokoroCandidateError("candidate state does not bind the disabled pass")
    voice_registry = load_product_voice_registry()
    if state["voice_registry"] != {
        "version": voice_registry.registry_version,
        "sha256": voice_registry.registry_sha256,
    }:
        raise KokoroCandidateError("candidate voice registry binding drifted")
    selected_voice = voice_registry.resolve("en-US", state["renderer"].get("voice"))
    if (
        selected_voice.voice_id != "af_heart"
        or selected_voice.evidence_status != "existing_rule_specific_evidence_only"
    ):
        raise KokoroCandidateError("candidate state does not bind its evidence voice")

    evidence = state["evidence"]
    protocol_path = ROOT / evidence["protocol_path"]
    analysis_path = ROOT / evidence["analysis_path"]
    if sha256_file(protocol_path) != evidence["protocol_sha256"]:
        raise KokoroCandidateError("candidate protocol hash mismatch")
    if sha256_file(analysis_path) != evidence["analysis_sha256"]:
        raise KokoroCandidateError("candidate analysis hash mismatch")
    protocol = _load_json(protocol_path)
    analysis = _load_json(analysis_path)
    if (
        protocol.get("protocol_sha256") != evidence["protocol_record_sha256"]
        or analysis.get("analysis_sha256") != evidence["analysis_record_sha256"]
        or analysis.get("classification") != evidence["automatic_classification"]
        or analysis.get("automatic_pass") is not True
        or analysis.get("production_enabled") is not False
    ):
        raise KokoroCandidateError("candidate evidence binding mismatch")
    measurement = protocol["implementation"]["measurement"]
    if sha256_file(PRAAT) != measurement["praat_sha256"]:
        raise KokoroCandidateError("frozen Praat binary hash mismatch")
    if sha256_file(MEASUREMENT_SCRIPT) != measurement["script_sha256"]:
        raise KokoroCandidateError("frozen measurement script hash mismatch")
    return {
        **state,
        "candidate_state_sha256": sha256_file(CANDIDATE_STATE_PATH),
        "voice_registry_version": voice_registry.registry_version,
        "voice_registry_sha256": voice_registry.registry_sha256,
        "local_anchor_geometry": protocol["parents"]["diagnostic_anchor_geometry"][
            "local_anchor_geometry"
        ],
    }


def _pcm(values: np.ndarray) -> np.ndarray:
    return np.frombuffer(pcm16_bytes(values), dtype="<i2").copy()


def _pcm_hash(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype="<i2").tobytes()).hexdigest()


def _wav_bytes(values: np.ndarray) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE_HZ)
        handle.writeframes(np.asarray(values, dtype="<i2").tobytes())
    return output.getvalue()


def _audio_record(values: np.ndarray) -> dict[str, Any]:
    pcm = np.asarray(values, dtype="<i2").reshape(-1)
    wav = _wav_bytes(pcm)
    return {
        "mime_type": "audio/wav",
        "base64": base64.b64encode(wav).decode("ascii"),
        "sha256": hashlib.sha256(wav).hexdigest(),
        "pcm_sha256": _pcm_hash(pcm),
        "sample_count": int(pcm.size),
        "duration_s": pcm.size / SAMPLE_RATE_HZ,
    }


def _target_span(surface: str, grapheme: str) -> list[int]:
    start = surface.find(grapheme)
    if start < 0 or surface.find(grapheme, start + 1) >= 0:
        raise KokoroCandidateError("validated shell has an ambiguous target grapheme")
    return [start, start + len(grapheme)]


def _transform(
    plan: TypedPlan, state: dict[str, Any], *, voice_id: str
) -> dict[str, Any]:
    slots: list[dict[str, Any]] = []
    for word_index in plan.target_word_indexes:
        word = plan.words[word_index]
        slots.append(
            {
                "word_index": word_index,
                "rule_id": RULE_ID,
                "source_ipa": "æ",
                "target_ipa": "ɛ",
                "neutral_character_span": _target_span(word.neutral_surface, "a"),
                "lens_character_span": _target_span(word.lens_surface, "eh"),
            }
        )
    return {
        "schema_version": 1,
        "profile_id": state["profile_id"],
        "voice_id": voice_id,
        "original_text": plan.normalized_text,
        "neutral_script": plan.neutral_script,
        "lens_script": plan.lens_script,
        "comparison_available": plan.comparison_available,
        "plan_sha256": plan.plan_sha256,
        "applied_rules": (
            [{"rule_id": RULE_ID, "occurrences": plan.target_occurrence_count}]
            if plan.comparison_available
            else []
        ),
        "slots": slots,
        "carrier_roles": [
            {"word_index": word.word_index, "role": word.carrier_role}
            for word in plan.words
        ],
    }


def _write_temp_wav(directory: Path, name: str, values: np.ndarray) -> Path:
    path = directory / name
    path.write_bytes(_wav_bytes(values))
    return path


class KokoroCandidateRuntime:
    def __init__(
        self,
        *,
        state: dict[str, Any],
        planner: StrictShellPlanner,
        synthesis: KokoroSynthesisRuntime,
        acoustic_reporter: Callable[
            [Path, Path, list[dict[str, Any]], dict[str, Any]], dict[str, Any]
        ] = _acoustic_report,
    ) -> None:
        if getattr(synthesis, "voice_id", state["renderer"]["voice"]) != state[
            "renderer"
        ]["voice"]:
            raise KokoroCandidateError("candidate renderer voice does not match evidence")
        self.state = state
        self.planner = planner
        self.synthesis = synthesis
        self.acoustic_reporter = acoustic_reporter

    @classmethod
    def load(cls) -> KokoroCandidateRuntime:
        return cls(
            state=load_candidate_state(),
            planner=StrictShellPlanner.load(),
            synthesis=KokoroSynthesisRuntime.load(download=False),
        )

    def contract(self) -> dict[str, Any]:
        return {
            "service_contract_version": self.state["service_contract_version"],
            "candidate_id": self.state["candidate_id"],
            "candidate_state_sha256": self.state["candidate_state_sha256"],
            "profile_id": self.state["profile_id"],
            "voice_id": self.state["renderer"]["voice"],
            "voice_registry_version": self.state["voice_registry_version"],
            "voice_registry_sha256": self.state["voice_registry_sha256"],
            "rule_ids": self.state["rule_ids"],
            "planner_version": self.state["planner"]["version"],
            "splice_version": self.state["splice"]["version"],
            "sample_rate_hz": self.state["renderer"]["sample_rate_hz"],
            "production_enabled": False,
            "human_qc_status": self.state["evidence"]["human_status"],
        }

    def _validate_scope(self, plan: TypedPlan) -> None:
        sentence_marks = re.findall(r"[.!?]+", plan.normalized_text)
        if len(sentence_marks) > 1:
            raise KokoroTypedEngineError(
                "strict_shell_unsupported_sentence_count",
                "This candidate supports one short sentence at a time.",
            )
        if any(index == 0 for index in plan.target_word_indexes):
            raise KokoroTypedEngineError(
                "strict_shell_unsupported_target_position",
                "This candidate has not validated phrase-initial target words.",
            )

    def render(self, text: str, *, voice_id: str | None = None) -> dict[str, Any]:
        requested_voice = voice_id or self.state["renderer"]["voice"]
        if requested_voice != self.state["renderer"]["voice"]:
            raise KokoroCandidateError(
                "requested voice has no evidence bound to this candidate"
            )
        plan = self.planner.plan(text)
        transform = _transform(plan, self.state, voice_id=requested_voice)
        contract = self.contract()
        if not plan.comparison_available:
            return {
                "schema_version": 1,
                "status": "no_supported_sounds",
                "message": NO_RULE_MESSAGE,
                "candidate_contract": contract,
                "transform": transform,
                "api_calls_made": 0,
            }
        self._validate_scope(plan)
        pair = plan.pair_plan()
        if pair is None:  # pragma: no cover - guarded by comparison_available
            raise KokoroCandidateError("comparison plan disappeared")

        rendered: ParityRender = self.synthesis.render_parity_triplet(pair)
        expected_columns = target_word_columns(
            self.synthesis.model, plan.neutral_phonemes, plan.target_word_indexes
        )
        neutral = _pcm(rendered.neutral)
        identity = _pcm(rendered.identity)
        full_lens = _pcm(rendered.lens)
        anchor_map = [
            1 if index == len(plan.words) - 1 else 0
            for index in plan.target_word_indexes
        ]
        alignment = alignment_record(
            model=self.synthesis.model,
            plan=plan,
            durations=rendered.predicted_durations,
            sample_count=neutral.size,
            anchor_occurrence_map=anchor_map,
        )
        targets = [row["interval"] for row in alignment["target_words"]]
        full_state_localization = localization_report(neutral, full_lens, targets)
        windows = full_state_localization.get("inside_windows", [])
        if len(windows) != len(plan.target_word_indexes):
            raise KokoroCandidateGateError(
                "target splice windows overlap or disappeared"
            )
        lens, weights = output_domain_splice(neutral, full_lens, windows)

        word_intervals = _word_intervals(
            self.synthesis.model,
            plan,
            rendered.predicted_durations,
            neutral.size,
        )
        position_checks: list[dict[str, Any]] = []
        for target_index, window in zip(plan.target_word_indexes, windows, strict=True):
            if target_index == len(plan.words) - 1:
                position_checks.append(
                    {
                        "position": "phrase-final",
                        "target_word_index": target_index,
                        "pass": True,
                    }
                )
            else:
                position_checks.append(
                    {
                        "position": "phrase-medial",
                        **phrase_medial_edge_gate(target_index, word_intervals, window),
                    }
                )

        raw_integrity = inspect_render(
            PairRender(
                neutral=rendered.neutral,
                lens=rendered.lens,
                predicted_durations=rendered.predicted_durations,
                replaced_columns=rendered.replaced_columns,
            )
        )
        clipped = [
            float(np.mean(np.abs(values.astype(np.int64)) >= 32767))
            for values in (neutral, identity, full_lens, lens)
        ]
        boundary = boundary_artifact_report(neutral, full_lens, lens, windows)
        started = time.perf_counter_ns()
        localization = localization_report(neutral, lens, targets)
        localization_ms = (time.perf_counter_ns() - started) / 1_000_000.0
        fail_closed = bool(
            localization_report(neutral, neutral, targets).get("pass") is False
            and localization_report(neutral, lens[:-1], targets).get("pass") is False
        )

        with tempfile.TemporaryDirectory(prefix="kokoro-candidate-gates-") as temp:
            directory = Path(temp)
            neutral_path = _write_temp_wav(directory, "neutral.wav", neutral)
            lens_path = _write_temp_wav(directory, "lens.wav", lens)
            acoustic = self.acoustic_reporter(
                neutral_path,
                lens_path,
                alignment["target_occurrences"],
                self.state["local_anchor_geometry"],
            )

        integrity_checks = {
            "exact_replaced_columns": rendered.replaced_columns == expected_columns,
            "raw_render_integrity": raw_integrity.pass_all,
            "neutral_identity_bit_exact": np.array_equal(neutral, identity),
            "equal_nonempty_samples": bool(
                neutral.size
                and neutral.size == identity.size == full_lens.size == lens.size
            ),
            "finite": all(
                np.isfinite(values.astype(np.float64)).all()
                for values in (neutral, identity, full_lens, lens)
            ),
            "unclipped": all(value < MAX_CLIPPED_FRACTION for value in clipped),
            "outside_exact_neutral": np.array_equal(
                lens[weights == 0.0], neutral[weights == 0.0]
            ),
            "interior_exact_full_lens": bool(
                np.any(weights == 1.0)
                and np.array_equal(lens[weights == 1.0], full_lens[weights == 1.0])
            ),
        }
        automatic_checks = {
            "plan_and_pcm_integrity": all(integrity_checks.values()),
            "target_positions": all(row["pass"] for row in position_checks),
            "boundary_click_metrics": bool(boundary["pass"]),
            "primary_50_acoustic_gate": bool(acoustic["primary_gate_pass"]),
            "localization_at_least_0_80": bool(localization["pass"]),
            "localization_runtime_cheap": localization_ms <= MAX_LOCALIZATION_P95_MS,
            "localization_fail_closed": fail_closed,
        }
        if not all(automatic_checks.values()):
            failed = sorted(key for key, value in automatic_checks.items() if not value)
            raise KokoroCandidateGateError(
                "automatic candidate gates failed: " + ",".join(failed)
            )

        return {
            "schema_version": 1,
            "status": "ready",
            "claim_tier": (
                "controlled_candidate_human_qc_pass"
                if self.state["evidence"]["human_status"] == "pass"
                else "controlled_candidate_pending_human_qc"
            ),
            "candidate_contract": contract,
            "transform": transform,
            "audio": {
                "neutral": _audio_record(neutral),
                "lens": _audio_record(lens),
            },
            "verification": {
                "status": "automatic_gates_passed",
                "plan_sha256": plan.plan_sha256,
                "target_occurrence_count": plan.target_occurrence_count,
                "neutral_pcm_sha256": _pcm_hash(neutral),
                "identity_pcm_sha256": _pcm_hash(identity),
                "lens_pcm_sha256": _pcm_hash(lens),
                "identity_bit_exact": True,
                "outside_exact_neutral": True,
                "interior_exact_full_lens": True,
                "inside_difference_energy_fraction": localization[
                    "inside_difference_energy_fraction"
                ],
                "localization_expected_by_construction": True,
                "localization_runtime_ms": localization_ms,
                "boundary_maximum_edge_delta_step_pcm": boundary[
                    "maximum_edge_delta_step_pcm"
                ],
                "boundary_maximum_derivative_ratio": boundary[
                    "maximum_candidate_to_reference_derivative_ratio"
                ],
                "acoustic_primary_window_percent": acoustic["primary_window_percent"],
                "acoustic_primary_gate_pass": True,
                "descriptive_window_sensitivity": acoustic[
                    "descriptive_window_sensitivity"
                ],
                "automatic_checks": automatic_checks,
                "api_calls_made": 0,
            },
            "cache_hit": False,
            "api_calls_made": 0,
        }
