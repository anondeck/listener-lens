from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Literal

from .audio_conformance import FLOW_DEVELOPER_PROMPT
from .config import stable_json
from .matched_pairs import PairingThresholds


RUN_ID = "20260715-sentence-pair-v2"
MODEL = "gpt-audio-1.5"
VOICE = "marin"
FORMAT = "wav"
MANIFEST_VERSION = "sentence-pair-v2-manifest-v1"
REQUEST_ORDER_METHOD = "frozen_balanced_four-block_order-v1"
PROMPT_PROTOCOL = "json-flow-v2"
DELIVERY = (
    "Fluent natural mainstream U.S. English. Perform this invented sentence at "
    "a normal conversational pace with connected speech, natural reductions, "
    "and one continuous intonation contour. Do not read it as a list."
)
RULE_ID = "ptbr.vowel.ae_to_eh"
RULE_SOURCE = "æ"
RULE_TARGET = "ɛ"
PRAAT_SHA256 = "b0311abb9ae606a5204715b0ab861d4ab863b932cbe3f0faecb7cce80b609d8d"
MEASUREMENT_SCRIPT_SHA256 = (
    "518cce752ba7907db2589d02e4ce8388e5d0eafa6e26849bfa7bc2ea2c13010b"
)
ANCHOR_SOURCE_FREEZE_SHA256 = (
    "d27e2e1142bbb0cce7ef0950bb68f7e38f3a5f5063b8a26213bda3b2d3fbb282"
)
GATE_DATABASE_SHA256 = (
    "cae4b5c9545d1577e9c3ac5892824a9540b234836354fca820e52d0e00567697"
)


@dataclass(frozen=True)
class SentenceCarrier:
    carrier_id: str
    shell: str
    neutral_token: str
    lens_token: str
    neutral_script: str
    lens_script: str
    target_word_index_zero_based: int
    neutral_character_span: tuple[int, int]
    lens_character_span: tuple[int, int]


@dataclass(frozen=True)
class SentenceSlot:
    request_order: int
    slot_id: str
    carrier_id: str
    shell: str
    side: Literal["neutral", "lens"]
    take_index: int
    script: str
    target_token: str
    target_word_index_zero_based: int
    target_character_span: tuple[int, int]


CARRIERS = (
    SentenceCarrier(
        carrier_id="carrier-zvf",
        shell="z_V_f",
        neutral_token="zaf",
        lens_token="zehf",
        neutral_script="frohr nushvot zaf tadril prohk.",
        lens_script="frohr nushvot zehf tadril prohk.",
        target_word_index_zero_based=2,
        neutral_character_span=(14, 17),
        lens_character_span=(14, 18),
    ),
    SentenceCarrier(
        carrier_id="carrier-vvp",
        shell="v_V_p",
        neutral_token="vap",
        lens_token="vehp",
        neutral_script="frohr nushvot vap tadril prohk.",
        lens_script="frohr nushvot vehp tadril prohk.",
        target_word_index_zero_based=2,
        neutral_character_span=(14, 17),
        lens_character_span=(14, 18),
    ),
    SentenceCarrier(
        carrier_id="carrier-bvvd",
        shell="b_V_vd",
        neutral_token="bavd",
        lens_token="behvd",
        neutral_script="frohr nushvot bavd tadril prohk.",
        lens_script="frohr nushvot behvd tadril prohk.",
        target_word_index_zero_based=2,
        neutral_character_span=(14, 18),
        lens_character_span=(14, 19),
    ),
)


# Four six-request blocks. Every block contains all three carriers and both
# sides once, limiting a time/order drift from becoming a side or carrier drift.
_ORDER = (
    ("carrier-zvf", "neutral", 1),
    ("carrier-vvp", "lens", 1),
    ("carrier-bvvd", "neutral", 1),
    ("carrier-zvf", "lens", 1),
    ("carrier-vvp", "neutral", 1),
    ("carrier-bvvd", "lens", 1),
    ("carrier-vvp", "neutral", 2),
    ("carrier-bvvd", "lens", 2),
    ("carrier-zvf", "neutral", 2),
    ("carrier-vvp", "lens", 2),
    ("carrier-bvvd", "neutral", 2),
    ("carrier-zvf", "lens", 2),
    ("carrier-bvvd", "neutral", 3),
    ("carrier-zvf", "lens", 3),
    ("carrier-vvp", "neutral", 3),
    ("carrier-bvvd", "lens", 3),
    ("carrier-zvf", "neutral", 3),
    ("carrier-vvp", "lens", 3),
    ("carrier-zvf", "lens", 4),
    ("carrier-vvp", "neutral", 4),
    ("carrier-bvvd", "lens", 4),
    ("carrier-zvf", "neutral", 4),
    ("carrier-vvp", "lens", 4),
    ("carrier-bvvd", "neutral", 4),
)


ANCHOR_GATE = {
    "instrument": "standalone Praat Burg",
    "maximum_formant_hz_family": [5500, 5750, 6000],
    "number_of_formants": 5.0,
    "time_step_s": 0.005,
    "window_s": 0.025,
    "pre_emphasis_from_hz": 50,
    "summary": "coordinate-wise median over middle 50% of frozen interval",
    "family_vector_cosine_minimum": 0.75,
    "direction_cosine_minimum": 0.50,
    "families": {
        "5500": {
            "source_centroid_bark": [9.073299, 11.798604],
            "target_centroid_bark": [7.532537, 13.063028],
            "anchor_vector_bark": [-1.540762, 1.264424],
            "magnitude_bark": 1.993167,
            "endpoint_take_variance_bark": 0.180808,
            "magnitude_threshold_bark": 0.361616,
            "minimum_cross_take_cosine": 0.993479,
        },
        "5750": {
            "source_centroid_bark": [9.080427, 12.182049],
            "target_centroid_bark": [7.528473, 13.032787],
            "anchor_vector_bark": [-1.551955, 0.850738],
            "magnitude_bark": 1.769836,
            "endpoint_take_variance_bark": 0.157748,
            "magnitude_threshold_bark": 0.315495,
            "minimum_cross_take_cosine": 0.992146,
        },
        "6000": {
            "source_centroid_bark": [9.065453, 12.015478],
            "target_centroid_bark": [7.379249, 13.011287],
            "anchor_vector_bark": [-1.686203, 0.995809],
            "magnitude_bark": 1.958294,
            "endpoint_take_variance_bark": 0.289625,
            "magnitude_threshold_bark": 0.579249,
            "minimum_cross_take_cosine": 0.973475,
        },
    },
    "pairwise_anchor_vector_cosines": [0.982795, 0.988203, 0.999488],
    "passed_before_rendering": True,
}


def prompt_contract_fingerprint() -> str:
    contract = {
        "model": MODEL,
        "voice": VOICE,
        "delivery": DELIVERY,
        "developer_prompt": FLOW_DEVELOPER_PROMPT,
        "protocol": PROMPT_PROTOCOL,
    }
    return hashlib.sha256(stable_json(contract).encode("utf-8")).hexdigest()


def build_manifest() -> tuple[SentenceSlot, ...]:
    carriers = {carrier.carrier_id: carrier for carrier in CARRIERS}
    slots: list[SentenceSlot] = []
    for request_order, (carrier_id, side, take_index) in enumerate(_ORDER, start=1):
        carrier = carriers[carrier_id]
        neutral = side == "neutral"
        slots.append(
            SentenceSlot(
                request_order=request_order,
                slot_id=f"{carrier.shell}__{side}__take-{take_index}",
                carrier_id=carrier_id,
                shell=carrier.shell,
                side=side,  # type: ignore[arg-type]
                take_index=take_index,
                script=carrier.neutral_script if neutral else carrier.lens_script,
                target_token=carrier.neutral_token if neutral else carrier.lens_token,
                target_word_index_zero_based=carrier.target_word_index_zero_based,
                target_character_span=(
                    carrier.neutral_character_span
                    if neutral
                    else carrier.lens_character_span
                ),
            )
        )
    if len(slots) != 24 or len({slot.slot_id for slot in slots}) != 24:
        raise AssertionError("sentence-pair-v2 requires exactly 24 unique slots")
    return tuple(slots)


def protocol_record() -> dict:
    thresholds = asdict(PairingThresholds())
    protocol = {
        "schema_version": 1,
        "status": "preregistered_awaiting_paid_call_approval",
        "run_id": RUN_ID,
        "question": (
            "Can the existing /æ/→/ɛ/ rule produce a natural sentence-level "
            "neutral/lens difference that listeners distinguish above ordinary "
            "GPT Audio take variance?"
        ),
        "rule": {
            "rule_id": RULE_ID,
            "source_ipa": RULE_SOURCE,
            "target_ipa": RULE_TARGET,
            "isolated_evidence": "frozen exact-category pass",
            "other_rules": "disabled and out of scope",
        },
        "renderer": {
            "model": MODEL,
            "voice": VOICE,
            "format": FORMAT,
            "modalities": ["text", "audio"],
            "store": False,
            "protocol": PROMPT_PROTOCOL,
            "delivery": DELIVERY,
            "prompt_contract_fingerprint": prompt_contract_fingerprint(),
            "claim_boundary": (
                "The prompt requests natural delivery; the transformation preserves "
                "structural carrier properties only and does not claim validated "
                "stress, rhythm, or prosody preservation."
            ),
        },
        "carriers": [asdict(carrier) for carrier in CARRIERS],
        "semantic_opacity_audit": {
            "common_gate_clean_tokens": ["frohr", "nushvot", "tadril", "prohk"],
            "gate_database_sha256": GATE_DATABASE_SHA256,
            "target_surface_exceptions": {
                "zaf": ["written_word_match", "predicted_homophone"],
                "vap": ["written_word_match", "predicted_homophone"],
                "vehp": ["predicted_homophone"],
            },
            "reason": (
                "These exact target surfaces are retained because they are the "
                "validated calibration stimuli. This is a research-only exception, "
                "not a relaxation of the product zero-real-word gate."
            ),
        },
        "manifest": {
            "version": MANIFEST_VERSION,
            "order_method": REQUEST_ORDER_METHOD,
            "logical_slots": 24,
            "takes_per_side_per_carrier": 4,
            "slots": [asdict(slot) for slot in build_manifest()],
        },
        "request_policy": {
            "sdk_automatic_retries": 0,
            "manual_retries": (
                "one retry for the same slot only after HTTP 429, HTTP 5xx, "
                "timeout, or connection failure returning no audio"
            ),
            "replacement_takes": 0,
            "logical_slots": 24,
            "successful_audio_ceiling": 24,
            "maximum_attempts_per_slot": 2,
            "maximum_total_attempts_if_every_first_attempt_is_external_failure": 48,
            "successfully_returned_audio_makes_slot_final": True,
            "evidentiary_failure_after_audio_never_triggers_retry": True,
            "external_retryable_failures": [
                "HTTP 429 with no audio",
                "HTTP 5xx with no audio",
                "timeout with no audio",
                "connection failure with no audio",
            ],
            "no_listening_before_acoustic_eligibility_and_timing_selection": True,
        },
        "anchor_gate": ANCHOR_GATE,
        "in_context_eligibility": {
            "analyze_every_returned_take": True,
            "integrity": {
                "valid_decodable_mono_pcm_wav_required": True,
                "sample_rate_hz": 24000,
                "duration_range_s": [0.5, 10.0],
                "maximum_clipped_fraction_exclusive": 0.001,
                "exact_provider_transcript_required": True,
            },
            "alignment": {
                "local_model": "whisper-large-v3-full",
                "language": "en",
                "temperature": 0,
                "condition_on_previous_text": False,
                "initial_prompt": "exact expected script",
                "required_monotonic_word_intervals": 5,
                "target_word_index_zero_based": 2,
                "word_label_policy": (
                    "Provider transcript must be exact. Whisper supplies timing only; "
                    "exactly five monotonic word intervals are required and the frozen "
                    "third interval is used even if Whisper's nonce label differs."
                ),
                "target_search_fraction": [0.10, 0.75],
                "core_duration_range_s": [0.060, 0.100],
                "core_selection": (
                    "highest-RMS formant-valid voiced frame, contiguous expansion, "
                    "sample-snapped; no manual boundary adjustment"
                ),
            },
            "measurement": {
                **ANCHOR_GATE,
                "minimum_middle_frames": 5,
                "minimum_valid_frame_fraction": 0.60,
                "plausibility_hz": {
                    "f1": [180, 1200],
                    "f2": [600, 3500],
                    "minimum_f2_minus_f1": 250,
                },
            },
            "individual_take_category": (
                "Under every family member, neutral is closer to the Marin /æ/ "
                "centroid than /ɛ/; lens is closer to /ɛ/ than /æ/."
            ),
            "carrier_level_contrast": (
                "Use every measurable take, including category failures. Under every "
                "family member require at least two measurable takes per side; neutral "
                "centroid closer to /æ/; lens centroid closer to /ɛ/; neutral-to-lens "
                "magnitude greater than max(0.15 Bark, 1.5 × the larger within-side RMS "
                "take variance); and cosine at least 0.50 to the Marin anchor vector."
            ),
            "complete_family_required": True,
            "acoustics_are_eligibility_only": True,
            "no_effect_size_ranking": True,
        },
        "pair_selection": {
            "thresholds": thresholds,
            "thresholds_remain_provisional": True,
            "eligible_inputs_only": True,
            "joint_block": (
                "For each carrier enumerate a shared eligible neutral baseline, one "
                "different eligible neutral, and one eligible lens. Both neutral/lens "
                "and neutral/neutral comparisons must pass the same timing and pause "
                "thresholds. Select the minimum combined timing score; ties use "
                "baseline, lens, then control take index."
            ),
            "selection_inputs": [
                "utterance duration",
                "pause count",
                "normalized pause position",
                "pause duration",
            ],
            "acoustic_values_used_for_ranking": False,
            "human_listening_used_for_ranking": False,
            "no_complete_block": "carrier fails without replacement",
        },
        "listener_pilot": {
            "conditions_per_eligible_carrier": [
                "identical baseline neutral versus itself",
                "baseline neutral versus a different eligible neutral",
                "baseline neutral versus eligible lens",
            ],
            "blinding": (
                "No neutral/lens spelling, script, filename, condition label, token "
                "identity, target grapheme, or shell label is visible. Show only five "
                "unlabeled position markers with position 3 highlighted; condition is "
                "hidden and A/B order is counterbalanced."
            ),
            "visible_structure": ["position 1", "position 2", "highlighted position 3", "position 4", "position 5"],
            "randomization": "deterministic per listener code after carrier yield freezes",
            "questions": [
                "Did the highlighted target position change? same/different/uncertain",
                "Strength of the target change: 1-5",
                "Confidence: 1-5",
                "Did unrelated pace, rhythm, or delivery differences interfere? yes/no/uncertain",
                "Did both clips sound like natural flowing utterances rather than a list? yes/no/uncertain",
            ],
            "language_strata": {
                "primary": "native Brazilian Portuguese family listeners",
                "sensitivity": "project owner; intermediate Brazilian Portuguese",
                "other": "record and report separately; never pool automatically",
            },
            "intended_primary_n": 4,
            "minimum_primary_n_for_pass_fail": 3,
            "maximum_primary_n": 4,
            "trial_count_per_listener": "3 × number of eligible carrier blocks; maximum 9",
            "uncertain_scoring": "non-positive, not missing",
            "interference_trials_are_not_dropped": True,
        },
        "listener_decision_rule": {
            "minimum_eligible_carrier_blocks": 2,
            "minimum_native_bp_listeners": 3,
            "identical_false_alarm_rate_maximum": 0.20,
            "neutral_lens_minus_neutral_neutral_change_rate_minimum": 0.30,
            "neutral_lens_minus_neutral_neutral_median_strength_minimum": 1.0,
            "carrier_generalization": (
                "At least two carriers each have neutral/lens different-response rate "
                "at least 0.25 above their neutral/neutral rate."
            ),
            "listener_generalization": (
                "At least ceil(2N/3) native BP listeners report more neutral/lens "
                "changes than neutral/neutral changes across eligible carriers."
            ),
            "neutral_lens_delivery_interference_yes_rate_maximum": 1 / 3,
            "neutral_lens_natural_flow_yes_rate_minimum": 2 / 3,
            "neutral_lens_median_confidence_minimum": 3,
            "all_conditions_required_for_pass": True,
            "primary_population": "native Brazilian Portuguese stratum only",
            "interpretation": (
                "Native-BP different responses test whether the controlled sentence "
                "signal is perceptually detectable above renderer variance. They do "
                "not by themselves prove that the lens recreates Brazilian-Portuguese "
                "perception; that stronger interpretation also requires the acoustic "
                "direction, cited profile evidence, and later profile-fit validation."
            ),
        },
        "cost": {
            "official_price_source": "https://developers.openai.com/api/docs/pricing",
            "rates_usd_per_million_tokens": {
                "text_input": 2.5,
                "text_output": 10.0,
                "audio_output": 64.0,
            },
            "prior_eight_sentence_render_cost_usd": 0.053528,
            "expected_total_usd_range": [0.10, 0.15],
            "approval_cap_usd": 0.25,
            "api_calls_already_made_for_this_run": 0,
        },
        "stopping_rule": {
            "before_render": (
                "If source bindings, anchor family, prompt contract, carrier manifest, "
                "or price cap cannot be satisfied, stop for approval."
            ),
            "render": (
                "Stop after all 24 logical slots are final, with at most one bounded "
                "same-slot retry for a listed external no-audio failure and never more "
                "than 24 successfully returned audio files."
            ),
            "yield": (
                "If fewer than two carriers yield a complete blinded trial block, "
                "classify inconclusive_external_failure when unresolved retryable "
                "transport failures caused the insufficiency; otherwise classify the "
                "sentence architecture as failed/insufficient. Do not recruit family "
                "listeners for a pass claim in either case."
            ),
            "listeners": (
                "Stop after the owner and all available consented native BP family "
                "listeners up to four complete. Fewer than three native BP listeners "
                "permits feasibility reporting only, not pass/fail."
            ),
            "after_result": (
                "A pass authorizes only a bounded different-take sentence approximation "
                "for /æ/→/ɛ/. A failure does not authorize more rules, UI compensation, "
                "new carriers, threshold changes, or more paid calls."
            ),
        },
        "claim_if_pass": (
            "A bounded sentence-level, different-take approximation for /æ/→/ɛ/ in "
            "these carrier contexts; not sample-level identity, validated prosody, "
            "arbitrary-sentence validity, or a universal BP perceptual mapping."
        ),
        "claim_if_fail": (
            "The current GPT-rendered sentence architecture did not isolate the "
            "listener-lens signal sufficiently above take variance."
        ),
        "downstream_work_held": [
            "typed-audio connection",
            "rule-profile expansion",
            "Portuguese GPT Audio nonce-renderer study",
        ],
        "pre_render_amendment": {
            "status": "frozen_before_any_paid_sentence_pair_v2_call",
            "presentation_blinding": "five positions only; third highlighted; no stimulus identity visible",
            "external_failure_policy": "one bounded same-slot retry; unresolved insufficiency is inconclusive_external_failure",
            "interpretation_boundary": "detectability above renderer variance is not proof of BP perceptual recreation",
            "unchanged": [
                "carriers",
                "acoustic gates",
                "listener thresholds",
                "24 successful-audio ceiling",
                "$0.25 approval cap",
            ],
        },
    }
    protocol["protocol_sha256"] = hashlib.sha256(
        stable_json(protocol).encode("utf-8")
    ).hexdigest()
    return protocol
