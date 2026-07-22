from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from earshift_bakeoff.config import Paths, stable_json
from earshift_bakeoff.prosody_acoustics import measure_stress_unit_component
from earshift_bakeoff.util import sha256_file

import run_prosody_component_validation_v1 as base


RUN_ID = "20260717-prosody-component-validation-v2"
RUN_DIR = Paths().artifacts / "prosody-validation" / RUN_ID
PARENT_FAILURE = (
    Paths().artifacts
    / "prosody-validation"
    / "20260717-prosody-component-validation-v1"
    / "failure.json"
)
RUNNER_SOURCE = Path(__file__).resolve()
_BASE_PROTOCOL = base._protocol


def _protocol_v2() -> dict[str, Any]:
    protocol = _BASE_PROTOCOL()
    protocol.pop("protocol_sha256", None)
    protocol.update(
        {
            "run_id": RUN_ID,
            "classification": (
                "bounded_stress_unit_donor_correction_multi_fixture_validation"
            ),
            "parent_run": {
                "run_id": "20260717-prosody-component-validation-v1",
                "failure_sha256": sha256_file(PARENT_FAILURE),
                "frozen_outcome": ("automatic_execution_failed_closed_no_human_review"),
            },
            "corrected_mechanism": (
                "When a demoted vowel has one decoder frame, v2 donates one frame "
                "from the adjacent demoted stress-marker column instead of deleting "
                "the vowel. The recipient remains the promoted vowel. All other "
                "control, splice, acoustic, fixture, and human thresholds are unchanged."
            ),
            "isolation_contract": (
                "Neutral and lens use identical segment symbols. Lexical-stress trials "
                "change only paired stress markers, duration allocation within their "
                "marker-plus-vowel stress units, and the frozen local intensity cue. "
                "Question trials change only the frozen final F0 contour. Output-domain "
                "splicing localizes returned differences."
            ),
            "stopping_rule": (
                "Preserve this corrective run unchanged. Failure cannot be rescued by "
                "threshold changes, fixture replacement, selective rerendering, or a "
                "second stress-duration donor redesign in this validation chain."
            ),
        }
    )
    stress = protocol["measurement_gates"]["stress"]
    stress.pop("minimum_decoder_duration_delta_ms", None)
    stress["minimum_stress_unit_duration_delta_ms"] = (
        base.STRESS_MINIMUM_DURATION_DELTA_MS
    )
    stress["duration_unit"] = "stress marker through its following vowel"
    protocol["source_hashes"]["runner"] = sha256_file(RUNNER_SOURCE)
    protocol["source_hashes"]["parent_failure"] = sha256_file(PARENT_FAILURE)
    protocol["protocol_sha256"] = hashlib.sha256(
        stable_json(protocol).encode("utf-8")
    ).hexdigest()
    return protocol


def main() -> None:
    base.RUN_ID = RUN_ID
    base.RUN_DIR = RUN_DIR
    base.RUNNER_SOURCE = RUNNER_SOURCE
    base._protocol = _protocol_v2
    base.measure_stress_component = measure_stress_unit_component
    base.main()


if __name__ == "__main__":
    main()
