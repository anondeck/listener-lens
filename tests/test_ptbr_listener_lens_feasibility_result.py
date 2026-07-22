from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "artifacts" / "research" / "20260717-ptbr-to-ae-listener-lens-v1"

EXPECTED_FILE_HASHES = {
    "protocol.json": "9ba316338f511dcf4752275a28488883e873087538df39ce01beae82a7a02cc1",
    "render-attempt.json": "75175eaab5363cd5f2385b1bf0d682c4ed3431a792e4f04f1c5c5024ea5f7857",
    "render-records.json": "3fa16b937fd7c772e2661f086809c9c387a381c822db6085422e722c21b41f20",
    "analysis.json": "24c99df1a04087f84752c8420720638c281593aea27583a307037ea283c928e0",
    "audio/01__ordinary-anchor-neutral.wav": (
        "c0689b112ea5db769e8937ad6660e6e14857c29f9a3686c549b3a625b2c29bd1"
    ),
    "audio/02__ordinary-anchor-lens.wav": (
        "837bf2aaf0332b07d274bc508b672851500ea47e8373b79dc14953917bd3dd14"
    ),
    "audio/03__controlled-neutral.wav": (
        "c0689b112ea5db769e8937ad6660e6e14857c29f9a3686c549b3a625b2c29bd1"
    ),
    "audio/04__controlled-identity.wav": (
        "c0689b112ea5db769e8937ad6660e6e14857c29f9a3686c549b3a625b2c29bd1"
    ),
    "audio/05__controlled-lens.wav": (
        "97569e7086ff027ce28f1e5f1c7ab275576d9fae14c0fa2bf4c835cd8f0bf90d"
    ),
}


def _load(name: str) -> dict[str, object]:
    return json.loads((RUN_ROOT / name).read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_result_files_are_exact_and_bounded() -> None:
    actual_files = {
        path.relative_to(RUN_ROOT).as_posix()
        for path in RUN_ROOT.rglob("*")
        if path.is_file()
    }
    assert actual_files == set(EXPECTED_FILE_HASHES)
    assert {
        relative: _sha256(RUN_ROOT / relative)
        for relative in sorted(EXPECTED_FILE_HASHES)
    } == EXPECTED_FILE_HASHES


def test_single_render_receipt_and_controlled_integrity_are_exact() -> None:
    protocol = _load("protocol.json")
    attempt = _load("render-attempt.json")
    render = _load("render-records.json")

    assert protocol["protocol_sha256"] == (
        "002fef936f04c293624046badc8d6f5c58b5bc3ab2858a24b3bee3bb68db2a69"
    )
    assert protocol["status"] == "independent_rechecks_approved_ready_for_freeze"
    assert protocol["claim_boundary"]["candidate_enabled"] is False
    assert protocol["independent_protocol_review"]["freeze_authorized"] is True

    assert attempt["protocol_sha256"] == protocol["protocol_sha256"]
    assert attempt["maximum_decoder_calls"] == 5
    assert attempt["attempts_per_slot"] == 1
    assert attempt["retries_allowed"] == 0
    assert attempt["committed_inputs"]["repository_head"] == (
        "5b04ff073be434d7309554f26985672beb398485"
    )
    assert (
        attempt["committed_inputs"]["all_inputs_tracked_clean_and_byte_identical"]
        is True
    )

    assert render["status"] == "single_bounded_render_complete"
    assert render["decoder_call_count"] == render["decoder_call_limit"] == 5
    assert render["retries_made"] == 0
    assert render["variants_rendered"] == 0
    assert render["selection_performed"] is False
    assert render["api_calls"] == render["paid_calls"] == 0
    assert render["runtime_integrity"]["pass"] is True
    assert (
        render["runtime_integrity"]["controlled_neutral_identity_bit_identical"] is True
    )
    assert [record["order"] for record in render["records"]] == [1, 2, 3, 4, 5]
    assert [record["slot_id"] for record in render["records"]] == [
        "ordinary-anchor-neutral",
        "ordinary-anchor-lens",
        "controlled-neutral",
        "controlled-identity",
        "controlled-lens",
    ]
    assert (
        render["records"][2]["audio"]["pcm_sha256"]
        == (render["records"][3]["audio"]["pcm_sha256"])
    )
    assert (
        render["records"][4]["audio"]["pcm_sha256"]
        != (render["records"][2]["audio"]["pcm_sha256"])
    )


def test_inconclusive_classification_has_no_review_or_positive_claim() -> None:
    analysis = _load("analysis.json")

    assert analysis["status"] == "analysis_complete"
    assert analysis["classification"] == "automatic_measurement_inconclusive"
    assert analysis["measurement_status"] == "inconclusive_measurement_error"
    assert analysis["automatic_acoustic_feasibility_pass"] is False
    assert analysis["all_occurrences_all_three_ceilings_pass"] is False
    assert analysis["claim"] == "no positive acoustic-feasibility claim"
    assert analysis["measurement_error"] == (
        "ReciprocalFeasibilityProtocolError: frame retention failed at 5500 Hz: "
        "retained=2/queried=2; finite=2; positive_ordered=2"
    )
    assert analysis["localization"] == {
        "pass": False,
        "status": "skipped_measurement_inconclusive",
    }
    assert analysis["localization_error"] is None
    assert analysis["perceptual_efficacy_established"] is False
    assert analysis["technical_probe_result_transferable_to_selected_voice"] is False
    assert analysis["voice_selected"] is False
    assert analysis["candidate_enabled"] is False
    assert analysis["production_route_available"] is False
    assert analysis["api_calls"] == analysis["paid_calls"] == 0

    for relative in (
        "public",
        "private",
        "review-generation-failure.json",
        "review-generation.partial",
        "review.html",
        "review-manifest.json",
        "blind-key.json",
    ):
        assert not (RUN_ROOT / relative).exists()
