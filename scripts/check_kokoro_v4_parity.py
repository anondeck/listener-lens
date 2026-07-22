#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import io
import json
import wave
from pathlib import Path

from earshift_bakeoff.config import Paths
from earshift_bakeoff.kokoro_synthesis import (
    CONFIG_FILE,
    KOKORO_VERSION,
    MODEL_FILE,
    MODEL_HASHES,
    MODEL_REPO,
    MODEL_REVISION,
    RNG_SEED,
    SAMPLE_RATE_HZ,
    VOICE_FILE,
    KokoroSynthesisRuntime,
    PairPlan,
    pcm16_bytes,
    pcm_sha256,
)
from earshift_bakeoff.util import atomic_write_json, sha256_file


RUN_ID = "20260716-kokoro-typed-engine-v1"
PARENT_RUN_ID = "20260716-kokoro-common-rng-confirmation-v4"
PARITY_FILES = {
    "neutral": "01__common-neutral.wav",
    "identity": "02__common-neutral-identity.wav",
    "lens": "05__common-lens-target-word.wav",
}
TARGET_WORD_INDEX = 7


def _wav_record(path: Path) -> dict[str, object]:
    with wave.open(str(path), "rb") as handle:
        frames = handle.readframes(handle.getnframes())
        return {
            "wav_sha256": sha256_file(path),
            "pcm_sha256": hashlib.sha256(frames).hexdigest(),
            "sample_count": handle.getnframes(),
            "sample_rate_hz": handle.getframerate(),
            "sample_width_bytes": handle.getsampwidth(),
            "channels": handle.getnchannels(),
        }


def _serialized_wav_sha256(audio: object) -> str:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE_HZ)
        handle.writeframes(pcm16_bytes(audio))
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def main() -> None:
    paths = Paths()
    parent = paths.artifacts / "phoneme-renderer" / PARENT_RUN_ID
    protocol_path = parent / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    fixed = protocol["fixed_product_state"]
    expected = {
        label: _wav_record(parent / "audio" / filename)
        for label, filename in PARITY_FILES.items()
    }
    plan = PairPlan(
        source_phonemes=fixed["source_alignment"],
        neutral_phonemes=fixed["neutral"],
        lens_phonemes=fixed["lens"],
        target_word_indexes=(TARGET_WORD_INDEX,),
    )
    runtime = KokoroSynthesisRuntime.load(download=False)
    rendered = runtime.render_parity_triplet(plan)
    arrays = {
        "neutral": rendered.neutral,
        "identity": rendered.identity,
        "lens": rendered.lens,
    }
    actual = {
        label: {
            "pcm_sha256": pcm_sha256(audio),
            "wav_sha256": _serialized_wav_sha256(audio),
            "sample_count": len(audio),
        }
        for label, audio in arrays.items()
    }
    comparisons = {
        label: {
            "pcm_sha256_match": actual[label]["pcm_sha256"]
            == expected[label]["pcm_sha256"],
            "sample_count_match": actual[label]["sample_count"]
            == expected[label]["sample_count"],
            "wav_sha256_match": actual[label]["wav_sha256"]
            == expected[label]["wav_sha256"],
        }
        for label in PARITY_FILES
    }
    identity_bit_identical = pcm16_bytes(rendered.neutral) == pcm16_bytes(
        rendered.identity
    )
    passed = identity_bit_identical and all(
        result["pcm_sha256_match"] and result["sample_count_match"]
        for result in comparisons.values()
    )
    module_path = paths.root / "src" / "earshift_bakeoff" / "kokoro_synthesis.py"
    receipt = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "kind": "implementation_regression_not_research_evidence",
        "status": "passed" if passed else "failed_architectural_drift",
        "api_calls_made": 0,
        "audio_artifacts_retained": 0,
        "parity_definition": (
            "decoded PCM16 SHA-256 and sample count are blocking; full-WAV SHA-256 "
            "is recorded because the current serialization is deterministic"
        ),
        "parent": {
            "run_id": PARENT_RUN_ID,
            "protocol_sha256": protocol["protocol_sha256"],
            "protocol_file_sha256": sha256_file(protocol_path),
            "files": PARITY_FILES,
        },
        "implementation": {
            "module_sha256": sha256_file(module_path),
            "synthesis_contract": "source-alignment_neutral-f0n_separate-state_target-word_common-rng-v1",
            "projection_schedule": (
                "explicit frozen-v4-equivalent arm64 float32 reduction: F0 128/64/tail; "
                "noise matrix product; removes cold-process slow-Conv1d dispatch drift"
            ),
            "target_word_index_zero_based": TARGET_WORD_INDEX,
            "replaced_columns": list(rendered.replaced_columns),
            "predicted_durations": list(rendered.predicted_durations),
            "rng_seed": RNG_SEED,
        },
        "renderer": {
            "package": "kokoro",
            "package_version": KOKORO_VERSION,
            "model_repo": MODEL_REPO,
            "model_revision": MODEL_REVISION,
            "asset_hashes": {
                CONFIG_FILE: MODEL_HASHES[CONFIG_FILE],
                MODEL_FILE: MODEL_HASHES[MODEL_FILE],
                VOICE_FILE: MODEL_HASHES[VOICE_FILE],
            },
        },
        "expected": expected,
        "actual": actual,
        "comparisons": comparisons,
        "neutral_identity_bit_identical": identity_bit_identical,
        "pass": passed,
    }
    destination = paths.artifacts / "typed-engine" / RUN_ID / "parity.json"
    atomic_write_json(destination, receipt)
    print(json.dumps(receipt, indent=2))
    if not passed:
        raise SystemExit("frozen-v4 decoded-PCM parity failed; fixture work is blocked")


if __name__ == "__main__":
    main()
