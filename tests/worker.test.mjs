import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import test from 'node:test';
import bilingualCandidateState from '../rules/bilingual-kokoro-candidate-state-v1.json' with { type: 'json' };
import bilingualCompositionState from '../rules/bilingual-kokoro-composition-candidate-v3.json' with { type: 'json' };
import bilingualRuleDisplay from '../rules/bilingual-rule-display-v1.json' with { type: 'json' };
import candidateState from '../rules/kokoro-candidate-state.json' with { type: 'json' };
import { FALLBACK_ACTIVITY, fallbackActivity, handleRequest } from '../worker/app.js';
import {
  AUDIO_MODEL,
  AUDIO_PROTOCOL_SHA256,
  AUDIO_VOICE,
  audioProtocolFingerprintInput,
  BILINGUAL_CANDIDATE_STATE_SHA256,
  BILINGUAL_COMPOSITION_STATE_SHA256,
  buildFlowPlan,
  candidateFlagsAreExactlyFalse,
  compareProsody,
  CURRENT_EVIDENCE_VOICE_ID,
  EXPECTED_TRANSFORM_CONTRACT,
  handleLegacyTypedAudio,
  handleTypedAudio,
  KOKORO_NO_RULE_MESSAGE,
  MAX_RENDER_CONCURRENCY,
  NO_RULE_MESSAGE,
  OVERALL_SERVER_DEADLINE_MS,
  PRODUCT_VOICE_REGISTRY_SHA256,
  PROFILE_ID,
  productCapabilityCatalog,
  productVoiceCatalog
} from '../worker/typed-audio.js';

const assets = { fetch: async () => new Response('asset', { status: 200 }) };
const ANCHOR_AUDIO = 'QU5DSE9SLVdBVg==';
const NEUTRAL_AUDIO = 'TkVVVFJBTC1XQVY=';
const LENS_AUDIO = 'TEVOUy1XQVY=';
const ANCHOR_SHA = 'd'.repeat(64);
const NEUTRAL_SHA = 'a'.repeat(64);
const LENS_SHA = 'b'.repeat(64);
const ACTIVITY_PROFILE_ID = 'en-US-to-pt-BR-listener-v2';

function byteSha(base64) {
  return createHash('sha256').update(Buffer.from(base64, 'base64')).digest('hex');
}

function displayRule(ruleId) {
  return bilingualRuleDisplay.rules.find(row => row.rule_id === ruleId);
}

function candidateContract(voiceId = CURRENT_EVIDENCE_VOICE_ID) {
  return {
    service_contract_version: candidateState.service_contract_version,
    candidate_id: candidateState.candidate_id,
    candidate_state_sha256: 'c'.repeat(64),
    profile_id: candidateState.profile_id,
    voice_id: voiceId,
    voice_registry_version: candidateState.voice_registry.version,
    voice_registry_sha256: candidateState.voice_registry.sha256,
    rule_ids: candidateState.rule_ids,
    planner_version: candidateState.planner.version,
    splice_version: candidateState.splice.version,
    sample_rate_hz: candidateState.renderer.sample_rate_hz,
    production_enabled: false,
    human_qc_status: candidateState.evidence.human_status
  };
}

function candidateTransform(comparisonAvailable = true, originalText = 'Happy cat.', voiceId = CURRENT_EVIDENCE_VOICE_ID) {
  const sourceWordCount = (originalText.match(/[A-Za-z]+(?:['’][A-Za-z]+)?/g) || []).length;
  const opaque = Array.from({ length: sourceWordCount }, (_, index) =>
    ['dohm', 'fihsh', 'nahl', 'vohm'][index % 4]).join(' ') + '.';
  return {
    schema_version: 1,
    profile_id: PROFILE_ID,
    voice_id: voiceId,
    original_text: originalText,
    neutral_script: comparisonAvailable ? 'dohm vazh.' : opaque,
    lens_script: comparisonAvailable ? 'dohm vehzh.' : opaque,
    comparison_available: comparisonAvailable,
    plan_sha256: 'd'.repeat(64),
    applied_rules: comparisonAvailable
      ? [{ rule_id: 'ptbr.vowel.ae_to_eh', occurrences: 1 }]
      : [],
    slots: comparisonAvailable
      ? [{
          word_index: 1,
          rule_id: 'ptbr.vowel.ae_to_eh',
          source_ipa: 'æ',
          target_ipa: 'ɛ',
          neutral_character_span: [1, 2],
          lens_character_span: [1, 3]
        }]
      : [],
    carrier_roles: Array.from({ length: sourceWordCount }, (_, wordIndex) => ({
      word_index: wordIndex,
      role: 'content'
    }))
  };
}

function candidateReadyPayload(originalText = 'Happy cat.', voiceId = CURRENT_EVIDENCE_VOICE_ID) {
  const neutral = 'TkVVVFJBTC1LT0tPUk8=';
  const lens = 'TEVOUy1LT0tPUk8=';
  const neutralPcm = 'e'.repeat(64);
  const lensPcm = 'f'.repeat(64);
  const automaticChecks = {
    plan_and_pcm_integrity: true,
    target_positions: true,
    boundary_click_metrics: true,
    primary_50_acoustic_gate: true,
    localization_at_least_0_80: true,
    localization_runtime_cheap: true,
    localization_fail_closed: true
  };
  return {
    schema_version: 1,
    status: 'ready',
    claim_tier: 'controlled_candidate_pending_human_qc',
    candidate_contract: candidateContract(voiceId),
    transform: candidateTransform(true, originalText, voiceId),
    audio: {
      neutral: {
        mime_type: 'audio/wav', base64: neutral, sha256: byteSha(neutral),
        pcm_sha256: neutralPcm, sample_count: 4, duration_s: 4 / 24000
      },
      lens: {
        mime_type: 'audio/wav', base64: lens, sha256: byteSha(lens),
        pcm_sha256: lensPcm, sample_count: 4, duration_s: 4 / 24000
      }
    },
    verification: {
      status: 'automatic_gates_passed',
      plan_sha256: 'd'.repeat(64),
      target_occurrence_count: 1,
      neutral_pcm_sha256: neutralPcm,
      identity_pcm_sha256: neutralPcm,
      lens_pcm_sha256: lensPcm,
      identity_bit_exact: true,
      outside_exact_neutral: true,
      interior_exact_full_lens: true,
      inside_difference_energy_fraction: 1,
      localization_expected_by_construction: true,
      localization_runtime_ms: 0.2,
      boundary_maximum_edge_delta_step_pcm: 0,
      boundary_maximum_derivative_ratio: 1,
      acoustic_primary_window_percent: 50,
      acoustic_primary_gate_pass: true,
      descriptive_window_sensitivity: { 40: false, 60: false },
      automatic_checks: automaticChecks,
      api_calls_made: 0
    },
    cache_hit: false,
    api_calls_made: 0
  };
}

function candidateNoRulePayload(originalText = 'This word is good.', voiceId = CURRENT_EVIDENCE_VOICE_ID) {
  return {
    schema_version: 1,
    status: 'no_supported_sounds',
    message: KOKORO_NO_RULE_MESSAGE,
    candidate_contract: candidateContract(voiceId),
    transform: candidateTransform(false, originalText, voiceId),
    api_calls_made: 0
  };
}

function bilingualRequest(
  text = 'The cat naps.',
  profileId = 'en-US-to-pt-BR-listener-v2',
  voiceId = 'af_heart'
) {
  return new Request('https://example.com/api/listener-lens', {
    method: 'POST',
    headers: {
      'content-type': 'application/json',
      origin: 'https://example.com',
      'cf-connecting-ip': '192.0.2.8'
    },
    body: JSON.stringify({ text, profile_id: profileId, voice_id: voiceId })
  });
}

function bilingualContract(profileId, voiceId) {
  return {
    service_contract_version: bilingualCompositionState.service_contract_version,
    candidate_id: bilingualCandidateState.candidate_id,
    candidate_state_sha256: BILINGUAL_CANDIDATE_STATE_SHA256,
    composition_candidate_id: bilingualCompositionState.candidate_id,
    composition_state_sha256: BILINGUAL_COMPOSITION_STATE_SHA256,
    composition_human_qc_status: bilingualCompositionState.evidence.human_status,
    composition_unseen_status: bilingualCompositionState.evidence.unseen_composition_status,
    runtime_gate_result_sha256: bilingualCandidateState.evidence.runtime_gate_result_sha256,
    runtime_gate_scaler_sha256: bilingualCandidateState.evidence.runtime_gate_scaler_sha256,
    profile_id: profileId,
    voice_id: voiceId,
    voice_registry_version: 'kokoro-product-voices-v1',
    voice_registry_sha256: PRODUCT_VOICE_REGISTRY_SHA256,
    production_enabled: false,
    human_qc_status: 'pending'
  };
}

function bilingualEndpoint() {
  return {
    anchor_separation_scaled_rms: 0.8,
    minimum_anchor_separation_scaled_rms: 0.25,
    controlled_movement_scaled_rms: 0.7,
    controlled_movement_fraction_of_anchor: 0.875,
    minimum_directional_movement_fraction: 0.25,
    minimum_exact_movement_fraction: 0.5,
    direction_cosine: 0.9,
    minimum_direction_cosine: 0.5,
    neutral_source_distance_scaled_rms: 0.1,
    neutral_target_distance_scaled_rms: 0.8,
    lens_source_distance_scaled_rms: 0.7,
    lens_target_distance_scaled_rms: 0.2,
    anchor_gate_pass: true,
    direction_gate_pass: true,
    directional_movement_gate_pass: true,
    exact_movement_gate_pass: true,
    neutral_endpoint_gate_pass: true,
    lens_endpoint_gate_pass: true,
    target_gain_gate_pass: true,
    source_departure_gate_pass: true,
    directional_pass: true,
    exact_category_pass: true,
    classification: 'exact_category_pass'
  };
}

function bilingualAggregate() {
  return {
    natural_seed_pair_count: 3,
    natural_exact_seed_pair_count: 3,
    natural_directional_seed_pair_count: 3,
    natural_reversed_seed_pair_count: 0,
    minimum_natural_exact_seed_pair_count: 2,
    maximum_natural_reversed_seed_pair_count: 0,
    anchor_validation_pass: true,
    candidate_evaluated: true,
    classification: 'exact_category_pass',
    directional_pass: true,
    exact_category_pass: true
  };
}

function bilingualReadyPayload({
  text = 'The cat naps.',
  profileId = 'en-US-to-pt-BR-listener-v2',
  voiceId = 'af_heart',
  ruleId = 'enpt.ae_eh',
  occurrenceCount = 2
} = {}) {
  const display = displayRule(ruleId);
  const neutral = 'TkVVVFJBTC1CSUxJTkdVQUw=';
  const lens = 'TEVOUy1CSUxJTkdVQUw=';
  const neutralPcm = '1'.repeat(64);
  const lensPcm = '2'.repeat(64);
  const plan = '3'.repeat(64);
  const audio = {
    neutral: {
      mime_type: 'audio/wav', base64: neutral, sha256: byteSha(neutral),
      pcm_sha256: neutralPcm, sample_count: 4, duration_s: 4 / 24000
    },
    lens: {
      mime_type: 'audio/wav', base64: lens, sha256: byteSha(lens),
      pcm_sha256: lensPcm, sample_count: 4, duration_s: 4 / 24000
    }
  };
  const transform = {
    schema_version: 1,
    profile_id: profileId,
    voice_id: voiceId,
    original_text: text,
    neutral_script: 'Dava nifa soma.',
    lens_script: 'Dehva nifa soma.',
    comparison_available: true,
    plan_sha256: plan,
    composition_mode: 'single_rule',
    applied_rules: [{
      rule_id: ruleId,
      source_ipa: display.display_source,
      target_ipa: display.display_target,
      display_label: display.display_label,
      occurrences: occurrenceCount
    }],
    omitted_rule_ids: [],
    partial_profile_coverage: false
  };
  return {
    schema_version: 1,
    status: 'ready_pending_human_qc',
    claim_tier: 'runtime_acoustic_pass_human_qc_pending',
    candidate_contract: bilingualContract(profileId, voiceId),
    transform,
    audio,
    verification: {
      status: 'runtime_acoustic_gates_passed',
      plan_sha256: plan,
      target_occurrence_count: occurrenceCount,
      neutral_pcm_sha256: neutralPcm,
      identity_pcm_sha256: neutralPcm,
      lens_pcm_sha256: lensPcm,
      render_integrity: {
        neutral_identity_bit_exact: true,
        equal_nonempty_samples: true,
        finite: true,
        unclipped: true,
        outside_splice_exact_neutral: true,
        full_weight_interior_exact_lens: true,
        boundary_metrics_pass: true,
        localization_pass: true,
        localization_fraction: 1,
        integrity_pass: true,
        changed_rules_acoustically_validated: false,
        evidence_status: 'integrity_pass_acoustic_validation_pending',
        prosody_control_pass: true,
        active_prosody_rule_ids: []
      },
      acoustic: {
        version: 'bilingual-candidate-runtime-gate-v1',
        rule_id: ruleId,
        voice_id: voiceId,
        occurrence_count: occurrenceCount,
        natural_decoder_render_count: 6,
        identity_false_positive_count: 0,
        classification: 'exact_category_pass',
        directional_pass: true,
        exact_category_pass: true,
        integrity_pass: true,
        pass: true,
        occurrences: Array.from({ length: occurrenceCount }, (_, index) => ({
          occurrence_index: index,
          aggregate: bilingualAggregate(),
          candidate: bilingualEndpoint(),
          identity_negative_control_directional: false
        }))
      },
      elapsed_ms: 12.5,
      api_calls_made: 0
    },
    cache_hit: false,
    api_calls_made: 0
  };
}

function bilingualCompositionPayload({
  text = 'We once took books.',
  profileId = 'en-US-to-pt-BR-listener-v2',
  voiceId = 'af_heart'
} = {}) {
  const payload = bilingualReadyPayload({
    text, profileId, voiceId, ruleId: 'enpt.ah_a', occurrenceCount: 1
  });
  const first = payload.verification.acoustic;
  const second = {
    ...structuredClone(first),
    rule_id: 'enpt.uh_u',
    occurrence_count: 2,
    occurrences: Array.from({ length: 2 }, (_, index) => ({
      occurrence_index: index,
      aggregate: bilingualAggregate(),
      candidate: bilingualEndpoint(),
      identity_negative_control_directional: false
    }))
  };
  payload.status = 'ready_automatic_only';
  payload.claim_tier = 'runtime_adaptive_composition_acoustic_pass_unseen_algorithm_pass_human_qc_pending';
  payload.transform.composition_mode = 'multi_rule_v8';
  payload.transform.applied_rules = [
    {
      rule_id: 'enpt.ah_a',
      source_ipa: displayRule('enpt.ah_a').display_source,
      target_ipa: displayRule('enpt.ah_a').display_target,
      display_label: displayRule('enpt.ah_a').display_label,
      occurrences: 1
    },
    {
      rule_id: 'enpt.uh_u',
      source_ipa: displayRule('enpt.uh_u').display_source,
      target_ipa: displayRule('enpt.uh_u').display_target,
      display_label: displayRule('enpt.uh_u').display_label,
      occurrences: 2
    }
  ];
  payload.verification.target_occurrence_count = 3;
  payload.verification.acoustic = {
    version: 'bilingual-candidate-v8-composition-gate-v1',
    voice_id: voiceId,
    rule_count: 2,
    rule_ids: ['enpt.ah_a', 'enpt.uh_u'],
    occurrence_count: 3,
    shared_natural_decoder_render_count: 6,
    identity_false_positive_count: 0,
    integrity_pass: true,
    pass: true,
    cells: [first, second]
  };
  payload.verification.adaptive_carrier = {
    version: 'v8-adaptive-carrier-v1',
    attempt_count: 2,
    selected_round_index: 1,
    rescued_after_retry: true,
    maximum_retry_rounds: 5
  };
  return payload;
}

function bilingualNoRulePayload({ text, profileId, voiceId }) {
  return {
    schema_version: 1,
    status: 'no_supported_sounds',
    message: 'No independently confirmed oral-vowel rule is available for this voice and sentence yet.',
    candidate_contract: bilingualContract(profileId, voiceId),
    coverage: {
      status: 'no_supported_sounds',
      profile_id: profileId,
      voice_id: voiceId,
      changed_rule_ids: [],
      omitted_rule_ids: [],
      cell: null,
      blockers: ['no_changed_listener_rules'],
      render_eligible: false
    },
    api_calls_made: 0
  };
}

function bilingualEnv({ payloadFactory = null } = {}) {
  const counters = { serviceCalls: 0 };
  return {
    env: {
      ASSETS: assets,
      TYPED_AUDIO_SERVE_ENABLED: 'true',
      TYPED_AUDIO_RENDER_ENABLED: 'true',
      KOKORO_ENGLISH_CANDIDATE_ENABLED: 'false',
      KOKORO_BILINGUAL_CANDIDATE_ENABLED: 'true',
      PORTUGUESE_RENDERER_CANDIDATE_ENABLED: 'false',
      RECIPROCAL_LISTENER_LENS_RESEARCH_ENABLED: 'false',
    AZURE_LENS_CANDIDATE_ENABLED: 'false',
    GIBBERISH_CANDIDATE_ENABLED: 'false',
      TRANSFORM_SERVICE: {
        async fetch(request) {
          counters.serviceCalls += 1;
          if (new URL(request.url).pathname !== '/bilingual-kokoro-listener-lens') {
            return Response.json({ error: 'not_found' }, { status: 404 });
          }
          const body = await request.json();
          const payload = payloadFactory
            ? payloadFactory(body)
            : bilingualReadyPayload({
                text: body.text,
                profileId: body.profile_id,
                voiceId: body.voice_id,
                ruleId: body.profile_id.startsWith('en-') ? 'enpt.ae_eh' : 'pten.ao_aa',
                occurrenceCount: body.profile_id.startsWith('en-') ? 2 : 1
              });
          return Response.json(payload);
        }
      }
    },
    counters
  };
}

class MemoryCache {
  constructor() {
    this.entries = new Map();
    this.matchCalls = 0;
  }
  async match(request) {
    this.matchCalls += 1;
    const response = this.entries.get(request.url);
    return response ? response.clone() : undefined;
  }
  async put(request, response) {
    this.entries.set(request.url, response.clone());
  }
  async payloads() {
    return Promise.all([...this.entries.values()].map(response => response.clone().json()));
  }
}

function makeCtx() {
  const pending = [];
  return {
    waitUntil(promise) { pending.push(Promise.resolve(promise)); },
    flush() { return Promise.allSettled(pending); }
  };
}

test.beforeEach(() => {
  globalThis.caches = { default: new MemoryCache() };
});

test('prosody-transfer protocol fingerprint is frozen before paid calls', async () => {
  const digest = await crypto.subtle.digest(
    'SHA-256',
    new TextEncoder().encode(audioProtocolFingerprintInput())
  );
  assert.equal(Buffer.from(digest).toString('hex'), AUDIO_PROTOCOL_SHA256);
});

test('prosody-transfer flow plan exposes only positional delivery guidance', () => {
  const weak = new Set([2, 5, 6, 7, 9]);
  const words = Array.from({ length: 10 }, (_, index) => ({
    carrier_role: weak.has(index + 1) ? 'weak' : 'content'
  }));
  assert.deepEqual(buildFlowPlan(words), {
    token_count: 10,
    weak_positions_one_based: [2, 5, 6, 7, 9],
    main_prominence_position_one_based: 10,
    boundary_policy: 'punctuation_only',
    grouping_policy: 'no_repeating_token_pairs',
    target_rate_wpm: [165, 185]
  });
});

function typedRequest(text = 'Happy cat.', voiceId = CURRENT_EVIDENCE_VOICE_ID) {
  return new Request('https://example.com/api/listener-lens', {
    method: 'POST',
    headers: { 'content-type': 'application/json', origin: 'https://example.com', 'cf-connecting-ip': '192.0.2.8' },
    body: JSON.stringify({ text, profile_id: PROFILE_ID, voice_id: voiceId })
  });
}

function prosodyFingerprint({ energySign = 1, pitchSign = 1, medianF0 = 220, voicedFraction = 0.75 } = {}) {
  return {
    version: 'prosody-fingerprint-v1',
    bin_count: 32,
    frame_count: 80,
    energy_contour_db: Array.from({ length: 32 }, (_, index) => energySign * (index - 15.5) / 4),
    pitch_contour_semitones: Array.from({ length: 32 }, (_, index) => pitchSign * (index - 15.5) / 8),
    median_f0_hz: medianF0,
    voiced_fraction: voicedFraction,
    energy_span_db: 8
  };
}

function requestRecord(body) {
  const userContent = body.messages[1].content;
  const serialized = typeof userContent === 'string'
    ? userContent
    : userContent.find(item => item.type === 'text')?.text;
  return JSON.parse(serialized);
}

test('frozen prosody comparison accepts an identity and rejects inverted delivery', () => {
  const timing = {
    duration_s: 2,
    sample_rate_hz: 24000,
    decoded_sample_count: 48000,
    clipped_fraction: 0,
    utterance_duration_s: 2,
    interior_pause_count: 0,
    interior_pause_s: 0,
    interior_pauses: []
  };
  const reference = { timing, prosody: prosodyFingerprint() };
  const identity = compareProsody(reference, { timing: { ...timing }, prosody: prosodyFingerprint() });
  assert.equal(identity.eligible, true);
  assert.equal(identity.score, 0);

  const inverted = compareProsody(reference, {
    timing: { ...timing },
    prosody: prosodyFingerprint({ energySign: -1, pitchSign: -1 })
  });
  assert.equal(inverted.eligible, false);
  assert.ok(inverted.reasons.includes('energy_correlation'));
  assert.ok(inverted.reasons.includes('pitch_correlation'));
});

function slot(wordIndex) {
  return {
    word_index: wordIndex,
    neutral_character_span: [1, 2],
    lens_character_span: [1, 3],
    rule_id: 'ptbr.vowel.ae_to_eh',
    source_ipa: 'æ',
    target_ipa: 'ɛ',
    neutral_grapheme: 'a',
    lens_grapheme: 'eh'
  };
}

function transformPayload(comparisonAvailable = true, originalText = 'Happy cat.') {
  const slots = comparisonAvailable ? [slot(0), slot(1)] : [];
  const makeWord = index => ({
    source: 'cat',
    source_ipa: 'kæt',
    listener_ipa: comparisonAvailable ? 'kɛt' : 'kæt',
    carrier_role: 'content',
    neutral_surface: comparisonAvailable ? 'bavd' : 'frayr',
    lens_surface: comparisonAvailable ? 'behvd' : 'frayr',
    syllables: 1,
    applied_rule_ids: comparisonAvailable ? ['ptbr.vowel.ae_to_eh'] : [],
    slots: comparisonAvailable ? [slot(index)] : [],
    pair_generation_attempt: 1
  });
  return {
    schema_version: 4,
    cache_key: 'c'.repeat(64),
    profile_id: PROFILE_ID,
    profile_label: 'English through a Brazilian Portuguese vowel-category lens',
    claim_label: 'Evidence-informed approximation',
    original_text: originalText,
    neutral_script: comparisonAvailable ? 'bavd bavd.' : 'frayr frayr.',
    lens_script: comparisonAvailable ? 'behvd behvd.' : 'frayr frayr.',
    comparison_available: comparisonAvailable,
    words: [makeWord(0), makeWord(1)],
    weak_form_report: {
      policy_version: 1,
      eligible_word_count: 0,
      eligible_mapping_count: 0,
      selected_mapping_count: 0,
      candidate_attempt_count: 0,
      candidate_gate_yield: null,
      rejected_attempt_count: 0,
      rejection_reason_counts: {},
      attempts: []
    },
    slots,
    applied_rules: comparisonAvailable ? [{
      rule_id: 'ptbr.vowel.ae_to_eh',
      source: 'æ',
      target: 'ɛ',
      occurrences: 2,
      confidence: 'passed-frozen-exact-category-acoustic-gates',
      description: 'Bounded test rule.',
      source_ids: ['rauber2006']
    }] : [],
    warnings: comparisonAvailable ? ['bounded_runtime_approximation'] : ['no_supported_listener_rule'],
    sources: [{ id: 'rauber2006', title: 'Vowel study', url: 'https://example.test/source' }],
    renderer_status: 'endpoint_implemented_pending_live_smoke',
    api_calls_made: 0,
    nonce_gate_enabled: true,
    deploy_contract: {
      ...EXPECTED_TRANSFORM_CONTRACT,
      enabled_rule_ids: [...EXPECTED_TRANSFORM_CONTRACT.enabled_rule_ids]
    }
  };
}

function typedEnv({ comparisonAvailable = true, inspect = () => true, transform = null } = {}) {
  const counters = { transforms: 0, inspections: 0, reserves: 0 };
  const service = {
    async fetch(request) {
      const encoded = new Uint8Array(await request.clone().arrayBuffer());
      assert.equal(Number(request.headers.get('content-length')), encoded.byteLength);
      const path = new URL(request.url).pathname;
      if (path === '/transform') {
        counters.transforms += 1;
        const body = await request.json();
        return Response.json(transform ? transform(body) : transformPayload(comparisonAvailable, body.text));
      }
      if (path === '/inspect-audio') {
        counters.inspections += 1;
        const body = await request.json();
        const inspection = inspect(body);
        const accepted = typeof inspection === 'boolean' ? inspection : inspection.accepted;
        const isLens = body.expected_script.startsWith('behvd');
        const isNeutral = body.expected_script.startsWith('bavd');
        const duration = isLens ? 0.79 : 0.8;
        return Response.json({
          accepted,
          reasons: accepted ? [] : ['provider_transcript_mismatch'],
          transcript: { exact_token_match: accepted },
          timing: {
            duration_s: duration,
            sample_rate_hz: 24000,
            decoded_sample_count: Math.round(24000 * duration),
            clipped_fraction: 0,
            utterance_duration_s: duration,
            interior_pause_count: 0,
            interior_pause_s: 0,
            interior_pauses: []
          },
          prosody: typeof inspection === 'object' && inspection.prosody
            ? inspection.prosody
            : prosodyFingerprint(),
          audio_sha256: isLens ? LENS_SHA : isNeutral ? NEUTRAL_SHA : ANCHOR_SHA
        });
      }
      return Response.json({ error: 'not_found' }, { status: 404 });
    }
  };
  const budget = {
    released: [],
    async reserveRenders(_clientHash, requested) {
      counters.reserves += 1;
      return { allowed: true, reserved: requested };
    },
    async releaseRenders(_clientHash, released) { this.released.push(released); }
  };
  return {
    env: {
      ASSETS: assets,
      OPENAI_API_KEY: 'test-only',
      TYPED_AUDIO_SERVE_ENABLED: 'true',
      TYPED_AUDIO_RENDER_ENABLED: 'true',
      KOKORO_ENGLISH_CANDIDATE_ENABLED: 'false',
      KOKORO_BILINGUAL_CANDIDATE_ENABLED: 'false',
      PORTUGUESE_RENDERER_CANDIDATE_ENABLED: 'false',
      RECIPROCAL_LISTENER_LENS_RESEARCH_ENABLED: 'false',
    AZURE_LENS_CANDIDATE_ENABLED: 'false',
    GIBBERISH_CANDIDATE_ENABLED: 'false',
      TRANSFORM_SERVICE: service,
      RENDER_BUDGET: budget
    },
    counters,
    budget
  };
}

function candidateEnv({ noRule = false, payloadFactory = null } = {}) {
  const counters = { serviceCalls: 0 };
  return {
    env: {
      ASSETS: assets,
      TYPED_AUDIO_SERVE_ENABLED: 'true',
      TYPED_AUDIO_RENDER_ENABLED: 'true',
      KOKORO_ENGLISH_CANDIDATE_ENABLED: 'true',
      KOKORO_BILINGUAL_CANDIDATE_ENABLED: 'false',
      PORTUGUESE_RENDERER_CANDIDATE_ENABLED: 'false',
      RECIPROCAL_LISTENER_LENS_RESEARCH_ENABLED: 'false',
    AZURE_LENS_CANDIDATE_ENABLED: 'false',
    GIBBERISH_CANDIDATE_ENABLED: 'false',
      TRANSFORM_SERVICE: {
        async fetch(request) {
          counters.serviceCalls += 1;
          if (new URL(request.url).pathname !== '/kokoro-listener-lens') {
            return Response.json({ error: 'not_found' }, { status: 404 });
          }
          const body = await request.json();
          const payload = payloadFactory
            ? payloadFactory(body)
            : noRule
              ? candidateNoRulePayload(body.text, body.voice_id)
              : candidateReadyPayload(body.text, body.voice_id);
          return Response.json(payload);
        }
      }
    },
    counters
  };
}

function activityRequest(body) {
  return new Request('https://example.com/api/activity', {
    method: 'POST',
    headers: { 'content-type': 'application/json', origin: 'https://example.com' },
    body: JSON.stringify(body)
  });
}

function resultMetadata() {
  return {
    profile_id: ACTIVITY_PROFILE_ID,
    source_locale: 'en-US',
    listener_locale: 'pt-BR',
    rule_ids: ['enpt.ae_eh'],
    changed_word_count: 1,
    comparison_status: 'ready',
    renderer_verification: 'azure_ssml_pair_returned'
  };
}

function openAiAudioResponse(script, index = 1, audioBase64 = 'UklGRg==') {
  return Response.json({
    id: `chatcmpl-${index}`,
    model: AUDIO_MODEL,
    choices: [{ message: { audio: { data: audioBase64, transcript: script } } }],
    usage: {
      prompt_tokens: 10,
      completion_tokens: 5,
      total_tokens: 15,
      prompt_tokens_details: { audio_tokens: 0 },
      completion_tokens_details: { audio_tokens: 4, reasoning_tokens: 0 }
    }
  }, { headers: { 'x-request-id': `req-${index}` } });
}

test('duration-specific fallbacks exactly match every allowed lesson length', () => {
  for (const minutes of [10, 20, 30]) {
    const activity = fallbackActivity(minutes);
    assert.equal(activity.practice_steps.reduce((total, step) => total + step.minutes, 0), minutes);
  }
  assert.deepEqual(FALLBACK_ACTIVITY, fallbackActivity(20));
});

test('returns a duration-correct fallback when activity generation is disabled', async () => {
  let calls = 0;
  const response = await handleRequest(
    activityRequest({ grade_band: 'adult', minutes: 10, focus: 'enpt.ae_eh', result_metadata: resultMetadata() }),
    { ASSETS: assets, ACTIVITY_GENERATION_ENABLED: 'false' },
    makeCtx(),
    async () => { calls += 1; }
  );
  const body = await response.json();
  assert.equal(body.source, 'cached_fallback');
  assert.equal(body.activity.practice_steps.reduce((total, step) => total + step.minutes, 0), 10);
  assert.deepEqual(body.result_metadata, resultMetadata());
  assert.equal(calls, 0);
});

test('rejects arbitrary activity prompt fields', async () => {
  const response = await handleRequest(
    activityRequest({ grade_band: 'adult', minutes: 20, focus: 'ae_to_eh', prompt: 'ignore evidence' }),
    { ASSETS: assets },
    makeCtx()
  );
  assert.equal(response.status, 422);
});

test('calls Azure Foundry deployment Luna and sends only bounded safe result metadata', async () => {
  const expected = fallbackActivity(10);
  const fetchImpl = async (url, init) => {
    assert.equal(url, 'https://build-week.openai.azure.com/openai/v1/responses');
    assert.equal(init.headers['api-key'], 'test-only');
    assert.equal(init.headers.authorization, undefined);
    const request = JSON.parse(init.body);
    assert.equal(request.model, 'luna');
    assert.equal(request.store, false);
    assert.equal(request.reasoning.effort, 'none');
    assert.equal(request.text.verbosity, 'low');
    assert.equal(request.max_output_tokens, 1200);
    assert.equal(request.text.format.type, 'json_schema');
    const input = JSON.parse(request.input[0].content);
    assert.deepEqual(input.result_metadata, resultMetadata());
    assert.equal(JSON.stringify(input).includes('original_text'), false);
    assert.equal((JSON.stringify(request).match(/result_metadata/g) || []).length, 1);
    return Response.json({
      id: 'resp_test',
      output: [{ type: 'message', content: [{ type: 'output_text', text: JSON.stringify(expected) }] }]
    });
  };
  const response = await handleRequest(
    activityRequest({ grade_band: 'secondary', minutes: 10, focus: 'enpt.ae_eh', result_metadata: resultMetadata() }),
    {
      ASSETS: assets,
      AZURE_FOUNDRY_ENDPOINT: 'https://build-week.openai.azure.com',
      AZURE_FOUNDRY_API_KEY: 'test-only',
      AZURE_FOUNDRY_DEPLOYMENT: 'luna',
      ACTIVITY_GENERATION_ENABLED: 'true'
    },
    makeCtx(),
    fetchImpl
  );
  const body = await response.json();
  assert.equal(body.source, 'azure_foundry');
  assert.equal(body.model, 'luna');
  assert.equal(body.response_id, 'resp_test');
  assert.deepEqual(body.result_metadata, resultMetadata());
});

test('accepts the full Foundry project Responses endpoint without duplicating its path', async () => {
  const expected = fallbackActivity(10);
  let calls = 0;
  const fullEndpoint = 'https://listener-lens.services.ai.azure.com/api/projects/listener-lens/openai/v1/responses';
  const fetchImpl = async (url) => {
    calls += 1;
    assert.equal(url, fullEndpoint);
    return Response.json({
      id: 'resp_project_endpoint',
      output: [{ type: 'message', content: [{ type: 'output_text', text: JSON.stringify(expected) }] }]
    });
  };
  const response = await handleRequest(
    activityRequest({ grade_band: 'secondary', minutes: 10, focus: 'enpt.ae_eh', result_metadata: resultMetadata() }),
    {
      ASSETS: assets,
      AZURE_FOUNDRY_ENDPOINT: fullEndpoint,
      AZURE_FOUNDRY_API_KEY: 'test-only',
      AZURE_FOUNDRY_DEPLOYMENT: 'gpt-5.6-luna',
      ACTIVITY_GENERATION_ENABLED: 'true'
    },
    makeCtx(),
    fetchImpl
  );
  const body = await response.json();
  assert.equal(response.status, 200);
  assert.equal(calls, 1);
  assert.equal(body.source, 'azure_foundry');
  assert.equal(body.model, 'gpt-5.6-luna');
});

test('rejects a generated activity with the wrong total duration and returns the exact fallback', async () => {
  const invalid = fallbackActivity(20);
  const response = await handleRequest(
    activityRequest({ grade_band: 'secondary', minutes: 10, focus: 'enpt.ae_eh', result_metadata: resultMetadata() }),
    {
      ASSETS: assets,
      AZURE_FOUNDRY_ENDPOINT: 'https://build-week.openai.azure.com',
      AZURE_FOUNDRY_API_KEY: 'test-only',
      AZURE_FOUNDRY_DEPLOYMENT: 'luna',
      ACTIVITY_GENERATION_ENABLED: 'true'
    },
    makeCtx(),
    async () => Response.json({ output: [{ type: 'message', content: [{ type: 'output_text', text: JSON.stringify(invalid) }] }] })
  );
  const body = await response.json();
  assert.equal(body.source, 'cached_fallback');
  assert.deepEqual(body.activity, fallbackActivity(10));
});

test('uses the duration-correct fallback when the activity limiter is exhausted', async () => {
  const response = await handleRequest(
    activityRequest({ grade_band: 'adult', minutes: 30, focus: 'enpt.ae_eh', result_metadata: resultMetadata() }),
    {
      AZURE_FOUNDRY_ENDPOINT: 'https://build-week.openai.azure.com',
      AZURE_FOUNDRY_API_KEY: 'test-only',
      ACTIVITY_GENERATION_ENABLED: 'true',
      ACTIVITY_RATE_LIMITER: { limit: async () => ({ success: false }) }
    },
    makeCtx()
  );
  const payload = await response.json();
  assert.equal(payload.source, 'cached_fallback');
  assert.equal(payload.rate_limited, true);
  assert.equal(payload.activity.practice_steps.reduce((total, step) => total + step.minutes, 0), 30);
});

test('serves assets outside the API namespace', async () => {
  const response = await handleRequest(new Request('https://example.com/'), { ASSETS: assets }, makeCtx());
  assert.equal(await response.text(), 'asset');
});

test('public voice catalog exposes exactly the four selected voices without asset hashes', async () => {
  const catalog = productVoiceCatalog();
  assert.equal(catalog.registry_version, 'kokoro-product-voices-v1');
  assert.equal(catalog.renderer, 'kokoro');
  assert.equal(catalog.same_voice_pair_required, true);
  assert.equal(catalog.production_enabled, false);
  assert.deepEqual(
    catalog.languages.flatMap(language => language.voices.map(voice => voice.voice_id)).sort(),
    ['af_heart', 'am_michael', 'pf_dora', 'pm_alex'].sort()
  );
  assert.equal(
    catalog.languages.flatMap(language => language.voices)
      .filter(voice => voice.current_narrow_runtime_available).length,
    1
  );
  assert.equal(
    catalog.languages.flatMap(language => language.voices)
      .filter(voice => voice.current_runtime_available).length,
    0
  );
  const broadCatalog = productVoiceCatalog({ bilingualCandidateEnabled: true });
  assert.equal(
    broadCatalog.languages.flatMap(language => language.voices)
      .filter(voice => voice.current_runtime_available).length,
    4
  );
  assert.equal(
    broadCatalog.languages.flatMap(language => language.voices)
      .reduce((total, voice) => total + voice.bilingual_automatic_candidate_rule_count, 0),
    18
  );
  const serialized = JSON.stringify(catalog);
  assert.equal(serialized.includes('voice_sha256'), false);
  assert.equal(serialized.includes('selection_bindings'), false);

  const response = await handleRequest(
    new Request('https://example.com/api/voices'),
    { ASSETS: assets },
    makeCtx()
  );
  assert.equal(response.status, 200);
  assert.deepEqual(await response.json(), catalog);

  const rejected = await handleRequest(
    new Request('https://example.com/api/voices', { method: 'POST' }),
    { ASSETS: assets },
    makeCtx()
  );
  assert.equal(rejected.status, 405);
  assert.equal(rejected.headers.get('allow'), 'GET');
});

test('public capability catalog exposes the full structural matrix without promotion', async () => {
  const catalog = productCapabilityCatalog();
  assert.equal(catalog.matrix_version, 'bilingual-product-matrix-v1');
  assert.equal(catalog.production_enabled, false);
  assert.equal(catalog.evidence_transfer_between_voices, false);
  assert.equal(catalog.evidence_transfer_between_rules, false);
  assert.equal(catalog.rule_cell_count, 166);
  assert.equal(catalog.changed_rule_cell_count, 98);
  assert.equal(catalog.product_enabled_cell_count, 0);
  assert.equal(catalog.structural_planner_slot_count, 280);
  assert.equal(catalog.structural_planner_gate_yield, 1);
  assert.equal(catalog.audio_validation_status, 'pending');
  assert.deepEqual(
    catalog.directions.flatMap(direction => direction.voices.map(voice => voice.voice_id)).sort(),
    ['af_heart', 'am_michael', 'pf_dora', 'pm_alex'].sort()
  );
  const serialized = JSON.stringify(catalog);
  assert.equal(serialized.includes('sha256'), false);
  assert.equal(serialized.includes('path'), false);

  const response = await handleRequest(
    new Request('https://example.com/api/capabilities'),
    { ASSETS: assets },
    makeCtx()
  );
  assert.equal(response.status, 200);
  assert.deepEqual(await response.json(), catalog);

  const rejected = await handleRequest(
    new Request('https://example.com/api/capabilities', { method: 'POST' }),
    { ASSETS: assets },
    makeCtx()
  );
  assert.equal(rejected.status, 405);
  assert.equal(rejected.headers.get('allow'), 'GET');
});

test('both POST endpoints accept a valid body without a Content-Length header', async () => {
  const activity = activityRequest({
    grade_band: 'adult',
    minutes: 20,
    focus: 'enpt.ae_eh',
    result_metadata: resultMetadata()
  });
  const typed = typedRequest();
  assert.equal(activity.headers.get('content-length'), null);
  assert.equal(typed.headers.get('content-length'), null);
  const activityResponse = await handleRequest(activity, { ASSETS: assets }, makeCtx());
  assert.equal(activityResponse.status, 200);
  const { env } = candidateEnv({ noRule: true });
  const typedResponse = await handleRequest(typed, env, makeCtx());
  assert.equal(typedResponse.status, 200);
});

function oversizedStreamRequest(path, onCancel) {
  let chunks = 0;
  const stream = new ReadableStream({
    pull(controller) {
      chunks += 1;
      controller.enqueue(new Uint8Array(1024).fill(32));
      if (chunks >= 4) controller.close();
    },
    cancel() { onCancel(); }
  });
  return new Request(`https://example.com${path}`, {
    method: 'POST',
    headers: { 'content-type': 'application/json', 'content-length': '1' },
    body: stream,
    duplex: 'half'
  });
}

test('activity body reader terminates an oversized stream even when Content-Length lies', async () => {
  let cancelled = false;
  const response = await handleRequest(
    oversizedStreamRequest('/api/activity', () => { cancelled = true; }),
    { ASSETS: assets },
    makeCtx()
  );
  assert.equal(response.status, 413);
  assert.equal(cancelled, true);
});

test('typed body reader terminates an oversized stream without transformation', async () => {
  let cancelled = false;
  const { env, counters } = candidateEnv();
  const response = await handleRequest(
    oversizedStreamRequest('/api/listener-lens', () => { cancelled = true; }),
    env,
    makeCtx()
  );
  assert.equal(response.status, 413);
  assert.equal(cancelled, true);
  assert.equal(counters.serviceCalls, 0);
});

test('serve-disabled typed audio stops before body reading, transformation, and cache lookup', async () => {
  const { env, counters } = candidateEnv();
  env.TYPED_AUDIO_SERVE_ENABLED = 'false';
  const response = await handleRequest(typedRequest(), env, makeCtx());
  const payload = await response.json();
  assert.equal(response.status, 503);
  assert.equal(payload.error, 'typed_audio_serve_disabled');
  assert.equal(counters.serviceCalls, 0);
  assert.equal(globalThis.caches.default.matchCalls, 0);
});

test('candidate configuration permits only the single Kokoro flag and fails closed otherwise', async () => {
  const invalidValues = [undefined, 'FALSE', '', '0'];
  for (const value of invalidValues) {
    globalThis.caches = { default: new MemoryCache() };
    const { env, counters } = candidateEnv();
    if (value === undefined) delete env.KOKORO_ENGLISH_CANDIDATE_ENABLED;
    else env.KOKORO_ENGLISH_CANDIDATE_ENABLED = value;
    let modelCalls = 0;
    const response = await handleRequest(
      typedRequest(), env, makeCtx(), async () => { modelCalls += 1; }
    );
    const payload = await response.json();
    assert.equal(candidateFlagsAreExactlyFalse(env), false);
    assert.equal(response.status, 503);
    assert.equal(payload.error, 'candidate_configuration_invalid');
    assert.equal(payload.api_calls_made, 0);
    assert.equal(counters.serviceCalls, 0);
    assert.equal(globalThis.caches.default.matchCalls, 0);
    assert.equal(modelCalls, 0);
  }

  const { env, counters } = candidateEnv();
  env.KOKORO_ENGLISH_CANDIDATE_ENABLED = 'false';
  const disabled = await handleRequest(typedRequest(), env, makeCtx());
  assert.equal(disabled.status, 503);
  assert.equal((await disabled.json()).error, 'kokoro_candidate_disabled');
  assert.equal(counters.serviceCalls, 0);

  for (const flag of ['PORTUGUESE_RENDERER_CANDIDATE_ENABLED', 'RECIPROCAL_LISTENER_LENS_RESEARCH_ENABLED']) {
    const configured = candidateEnv();
    configured.env[flag] = 'true';
    const response = await handleRequest(typedRequest(), configured.env, makeCtx());
    assert.equal(response.status, 503);
    assert.equal((await response.json()).error, 'candidate_configuration_invalid');
    assert.equal(configured.counters.serviceCalls, 0);
  }
});

test('bilingual candidate serves strict zero-API English and Portuguese contracts', async () => {
  for (const request of [
    bilingualRequest(),
    bilingualRequest('O povo corre.', 'pt-BR-to-en-US-listener-v2', 'pm_alex')
  ]) {
    const { env, counters } = bilingualEnv();
    let externalCalls = 0;
    const response = await handleRequest(
      request, env, makeCtx(), async () => { externalCalls += 1; }
    );
    const payload = await response.json();
    assert.equal(response.status, 200);
    assert.equal(payload.status, 'ready_pending_human_qc');
    assert.equal(payload.verification.acoustic.pass, true);
    assert.equal(payload.api_calls_made, 0);
    assert.equal(counters.serviceCalls, 1);
    assert.equal(externalCalls, 0);
  }
});

test('bilingual candidate accepts a strict multi-rule v8 composition contract', async () => {
  const { env, counters } = bilingualEnv({
    payloadFactory: body => bilingualCompositionPayload({
      text: body.text, profileId: body.profile_id, voiceId: body.voice_id
    })
  });
  const response = await handleRequest(
    bilingualRequest('We once took books.'), env, makeCtx()
  );
  const payload = await response.json();

  assert.equal(response.status, 200);
  assert.equal(payload.status, 'ready_automatic_only');
  assert.equal(
    payload.claim_tier,
    'runtime_adaptive_composition_acoustic_pass_unseen_algorithm_pass_human_qc_pending'
  );
  assert.equal(payload.transform.composition_mode, 'multi_rule_v8');
  assert.deepEqual(
    payload.transform.applied_rules.map(rule => rule.rule_id),
    ['enpt.ah_a', 'enpt.uh_u']
  );
  assert.equal(payload.verification.target_occurrence_count, 3);
  assert.equal(payload.verification.acoustic.pass, true);
  assert.deepEqual(payload.verification.adaptive_carrier, {
    version: 'v8-adaptive-carrier-v1',
    attempt_count: 2,
    selected_round_index: 1,
    rescued_after_retry: true,
    maximum_retry_rounds: 5
  });
  assert.equal(counters.serviceCalls, 1);
});

test('multi-rule adaptive composition rejects incomplete or contradictory retry metadata', async () => {
  const mutations = [
    payload => { delete payload.verification.adaptive_carrier; },
    payload => { payload.verification.adaptive_carrier.version = 'unexpected'; },
    payload => { payload.verification.adaptive_carrier.maximum_retry_rounds = 6; },
    payload => { payload.verification.adaptive_carrier.attempt_count = 7; },
    payload => { payload.verification.adaptive_carrier.selected_round_index = 0; },
    payload => { payload.verification.adaptive_carrier.rescued_after_retry = false; },
    payload => { payload.verification.adaptive_carrier.unexpected = true; }
  ];
  for (const mutate of mutations) {
    const { env, counters } = bilingualEnv({ payloadFactory(body) {
      const payload = bilingualCompositionPayload({
        text: body.text, profileId: body.profile_id, voiceId: body.voice_id
      });
      mutate(payload);
      return payload;
    } });
    const response = await handleRequest(
      bilingualRequest('We once took books.'), env, makeCtx()
    );
    assert.equal(response.status, 503);
    assert.equal((await response.json()).error, 'bilingual_kokoro_contract_mismatch');
    assert.equal(counters.serviceCalls, 1);
  }
});

test('multi-rule latent-state audio may share carrier spelling after strict gates', async () => {
  const { env } = bilingualEnv({
    payloadFactory: body => {
      const payload = bilingualCompositionPayload({
        text: body.text, profileId: body.profile_id, voiceId: body.voice_id
      });
      payload.transform.lens_script = payload.transform.neutral_script;
      return payload;
    }
  });
  const response = await handleRequest(
    bilingualRequest('We once took books.'), env, makeCtx()
  );

  assert.equal(response.status, 200);
  assert.equal((await response.json()).status, 'ready_automatic_only');
});

test('single-rule response still rejects identical neutral and lens scripts', async () => {
  const { env } = bilingualEnv({
    payloadFactory: body => {
      const payload = bilingualReadyPayload({
        text: body.text, profileId: body.profile_id, voiceId: body.voice_id
      });
      payload.transform.lens_script = payload.transform.neutral_script;
      return payload;
    }
  });
  const response = await handleRequest(bilingualRequest(), env, makeCtx());

  assert.equal(response.status, 503);
  assert.equal((await response.json()).error, 'bilingual_kokoro_contract_mismatch');
});

test('bilingual candidate rejects invalid direction, voice, and flag combinations before inference', async () => {
  const invalidRequests = [
    bilingualRequest('The cat naps.', 'pt-BR-to-en-US-listener-v2', 'af_heart'),
    bilingualRequest('O povo corre.', 'en-US-to-pt-BR-listener-v2', 'pm_alex'),
    bilingualRequest('The cat naps.', 'unknown-profile', 'af_heart')
  ];
  for (const request of invalidRequests) {
    const { env, counters } = bilingualEnv();
    const response = await handleRequest(request, env, makeCtx());
    assert.equal(response.status, 422);
    assert.equal((await response.json()).error, 'unsupported_options');
    assert.equal(counters.serviceCalls, 0);
  }

  const both = bilingualEnv();
  both.env.KOKORO_ENGLISH_CANDIDATE_ENABLED = 'true';
  const conflict = await handleRequest(bilingualRequest(), both.env, makeCtx());
  assert.equal(conflict.status, 503);
  assert.equal((await conflict.json()).error, 'candidate_configuration_invalid');
  assert.equal(both.counters.serviceCalls, 0);
});

test('bilingual candidate accepts only the complete hash-bound runtime contract', async () => {
  const mutations = [
    payload => { payload.candidate_contract.candidate_state_sha256 = '0'.repeat(64); },
    payload => { payload.candidate_contract.voice_registry_sha256 = '0'.repeat(64); },
    payload => { payload.transform.omitted_rule_ids = [payload.transform.applied_rules[0].rule_id]; },
    payload => { payload.verification.neutral_pcm_sha256 = '0'.repeat(64); },
    payload => { payload.verification.render_integrity.localization_pass = false; },
    payload => { payload.verification.acoustic.rule_id = 'unexpected.rule'; },
    payload => { payload.verification.acoustic.occurrences[0].candidate.direction_gate_pass = false; },
    payload => { payload.verification.acoustic.occurrences[0].unexpected = true; },
    payload => { payload.unexpected = true; }
  ];
  for (const mutate of mutations) {
    const { env, counters } = bilingualEnv({ payloadFactory(body) {
      const payload = bilingualReadyPayload({
        text: body.text, profileId: body.profile_id, voiceId: body.voice_id
      });
      mutate(payload);
      return payload;
    } });
    const response = await handleRequest(bilingualRequest(), env, makeCtx());
    assert.equal(response.status, 503);
    assert.equal((await response.json()).error, 'bilingual_kokoro_contract_mismatch');
    assert.equal(counters.serviceCalls, 1);
  }
});

test('bilingual no-supported-sounds result is explicit and render-free', async () => {
  const { env, counters } = bilingualEnv({
    payloadFactory: body => bilingualNoRulePayload({
      text: body.text, profileId: body.profile_id, voiceId: body.voice_id
    })
  });
  const response = await handleRequest(
    bilingualRequest('This input has no eligible rule.'), env, makeCtx()
  );
  const payload = await response.json();
  assert.equal(response.status, 200);
  assert.equal(payload.status, 'no_supported_sounds');
  assert.equal(payload.coverage.render_eligible, false);
  assert.equal(payload.api_calls_made, 0);
  assert.equal(counters.serviceCalls, 1);
});

test('health reports candidate exact-false configuration without exposing raw values', async () => {
  const { env } = typedEnv();
  let response = await handleRequest(new Request('https://example.com/api/health'), env, makeCtx());
  let payload = await response.json();
  assert.equal(payload.candidate_flags_exactly_false, true);
  assert.equal(payload.core_audio_renderer, 'kokoro');
  assert.equal(payload.kokoro_candidate_enabled, false);
  assert.equal(payload.bilingual_kokoro_candidate_enabled, false);
  assert.equal(payload.voice_registry_version, 'kokoro-product-voices-v1');
  assert.equal(payload.configured_voice_count, 4);
  assert.equal(payload.bilingual_matrix_version, 'bilingual-product-matrix-v1');
  assert.equal(payload.bilingual_changed_rule_cell_count, 98);
  assert.equal(payload.bilingual_audio_validation_status, 'pending');
  assert.equal(JSON.stringify(payload).includes('KOKORO_ENGLISH_CANDIDATE_ENABLED'), false);

  env.RECIPROCAL_LISTENER_LENS_RESEARCH_ENABLED = 'TRUE';
  response = await handleRequest(new Request('https://example.com/api/health'), env, makeCtx());
  payload = await response.json();
  assert.equal(payload.candidate_flags_exactly_false, false);
});

test('transform-service details are not relayed across the public Worker boundary', async () => {
  const { env } = candidateEnv();
  env.TRANSFORM_SERVICE = {
    async fetch() {
      return Response.json(
        { error: 'transform_rejected', detail: 'PRIVATE_TYPED_SOURCE_TEXT' },
        { status: 422 }
      );
    }
  };
  const response = await handleRequest(typedRequest(), env, makeCtx());
  const payload = await response.json();
  assert.equal(response.status, 422);
  assert.deepEqual(payload, {
    status: 'unavailable',
    error: 'kokoro_service_rejected',
    api_calls_made: 0
  });
  assert.equal(JSON.stringify(payload).includes('PRIVATE_TYPED_SOURCE_TEXT'), false);
});

test('every Kokoro service-contract boundary fails before audio reaches the browser', async () => {
  const mutations = [
    payload => { payload.candidate_contract.service_contract_version = 'unexpected-service'; },
    payload => { payload.candidate_contract.candidate_id = 'unexpected-candidate'; },
    payload => { payload.candidate_contract.voice_id = 'am_michael'; },
    payload => { payload.candidate_contract.voice_registry_version = 'unexpected-registry'; },
    payload => { payload.candidate_contract.voice_registry_sha256 = 'not-a-hash'; },
    payload => { payload.candidate_contract.production_enabled = true; },
    payload => { payload.transform.schema_version = 999; },
    payload => { payload.transform.profile_id = 'unexpected-profile'; },
    payload => { payload.transform.voice_id = 'am_michael'; },
    payload => { payload.transform.neutral_script = ''; },
    payload => { payload.transform.slots = []; },
    payload => { payload.verification.identity_bit_exact = false; },
    payload => { payload.verification.acoustic_primary_gate_pass = false; },
    payload => { payload.verification.automatic_checks.target_positions = false; },
    payload => { payload.audio.neutral.sha256 = '0'.repeat(64); },
    payload => { payload.unexpected = true; }
  ];
  for (const mutate of mutations) {
    globalThis.caches = { default: new MemoryCache() };
    const { env, counters } = candidateEnv({ payloadFactory(body) {
      const payload = candidateReadyPayload(body.text);
      mutate(payload);
      return payload;
    } });
    let calls = 0;
    const response = await handleRequest(typedRequest(), env, makeCtx(), async () => { calls += 1; });
    const payload = await response.json();
    assert.equal(response.status, 503);
    assert.equal(payload.error, 'kokoro_contract_mismatch');
    assert.equal(globalThis.caches.default.matchCalls, 0);
    assert.equal(counters.serviceCalls, 1);
    assert.equal(calls, 0);
  }
});

test('Michael is configured but current Heart evidence is never transferred to him', async () => {
  const { env, counters } = candidateEnv();
  let openAiCalls = 0;
  const response = await handleRequest(
    typedRequest('Happy cat.', 'am_michael'),
    env,
    makeCtx(),
    async () => { openAiCalls += 1; }
  );
  const payload = await response.json();
  assert.equal(response.status, 409);
  assert.deepEqual(payload, {
    status: 'unavailable',
    error: 'voice_evidence_unavailable',
    voice_id: 'am_michael',
    api_calls_made: 0
  });
  assert.equal(counters.serviceCalls, 0);
  assert.equal(openAiCalls, 0);
});

test('no-rule input returns the exact explanation and makes no paid call', async () => {
  const { env } = candidateEnv({ noRule: true });
  let calls = 0;
  const response = await handleRequest(typedRequest('This word is good.'), env, makeCtx(), async () => { calls += 1; });
  const payload = await response.json();
  assert.equal(response.status, 200);
  assert.equal(payload.status, 'no_supported_sounds');
  assert.equal(payload.message, KOKORO_NO_RULE_MESSAGE);
  assert.equal(payload.api_calls_made, 0);
  assert.equal(calls, 0);
});

test('active typed route serves only the zero-API Kokoro candidate contract', async () => {
  const { env, counters } = candidateEnv();
  let openAiCalls = 0;
  const response = await handleRequest(
    typedRequest(),
    env,
    makeCtx(),
    async () => { openAiCalls += 1; }
  );
  const payload = await response.json();
  assert.equal(response.status, 200);
  assert.equal(payload.status, 'ready');
  assert.equal(payload.api_calls_made, 0);
  assert.equal(payload.candidate_contract.production_enabled, false);
  assert.equal(payload.verification.acoustic_primary_gate_pass, true);
  assert.equal(payload.verification.identity_bit_exact, true);
  assert.equal(counters.serviceCalls, 1);
  assert.equal(openAiCalls, 0);
});

test('active Kokoro route fails before service work when rendering is disabled', async () => {
  const { env, counters } = candidateEnv();
  env.TYPED_AUDIO_RENDER_ENABLED = 'false';
  const response = await handleRequest(typedRequest(), env, makeCtx());
  assert.equal(response.status, 503);
  assert.equal((await response.json()).error, 'typed_audio_render_disabled');
  assert.equal(counters.serviceCalls, 0);
});

test('render-disabled mode serves a compatible derived cache hit with the fresh request transform', async () => {
  const { env } = typedEnv();
  const ctx = makeCtx();
  let calls = 0;
  const fetchImpl = async (_url, init) => {
    calls += 1;
    const record = requestRecord(JSON.parse(init.body));
    const audio = record.task === 'natural_source_anchor'
      ? ANCHOR_AUDIO
      : record.reference_kind === 'source_anchor' ? NEUTRAL_AUDIO : LENS_AUDIO;
    return openAiAudioResponse(record.script, calls, audio);
  };
  const first = await handleLegacyTypedAudio(typedRequest('The Cat.'), env, ctx, fetchImpl);
  assert.equal(first.status, 200);
  await ctx.flush();
  const cachePayloads = await globalThis.caches.default.payloads();
  assert.equal(cachePayloads.length, 1);
  assert.deepEqual(Object.keys(cachePayloads[0]).sort(), ['attempts', 'audio', 'renderer', 'selection', 'verification']);
  const serialized = JSON.stringify(cachePayloads[0]);
  assert.equal(serialized.includes('original_text'), false);
  assert.equal(serialized.includes('source_ipa'), false);
  assert.equal(serialized.includes('listener_ipa'), false);
  assert.equal(serialized.includes('"words"'), false);
  assert.equal(serialized.includes(ANCHOR_AUDIO), false);

  env.TYPED_AUDIO_RENDER_ENABLED = 'false';
  const second = await handleLegacyTypedAudio(typedRequest('the cat.'), env, makeCtx(), async () => { throw new Error('must not render'); });
  const secondPayload = await second.json();
  assert.equal(second.status, 200);
  assert.equal(secondPayload.cache_hit, true);
  assert.equal(secondPayload.api_calls_made, 0);
  assert.equal(secondPayload.transform.original_text, 'the cat.');
  assert.equal(calls, 5);
});

test('render-disabled mode fails closed on a cache miss after returning the fresh transform', async () => {
  const { env, counters } = typedEnv();
  env.TYPED_AUDIO_RENDER_ENABLED = 'false';
  let calls = 0;
  const response = await handleLegacyTypedAudio(typedRequest(), env, makeCtx(), async () => { calls += 1; });
  const payload = await response.json();
  assert.equal(response.status, 503);
  assert.equal(payload.error, 'typed_audio_render_disabled');
  assert.equal(payload.transform.original_text, 'Happy cat.');
  assert.equal(counters.reserves, 0);
  assert.equal(calls, 0);
});

test('typed flow creates one anchor then bounded 2+2 audio-conditioned transfers', async () => {
  const { env } = typedEnv();
  const requests = [];
  let active = 0;
  let peak = 0;
  const fetchImpl = async (_url, init) => {
    active += 1;
    peak = Math.max(peak, active);
    const body = JSON.parse(init.body);
    requests.push(body);
    assert.equal(body.model, AUDIO_MODEL);
    assert.equal(body.audio.voice, AUDIO_VOICE);
    assert.deepEqual(body.modalities, ['text', 'audio']);
    assert.equal(body.store, false);
    const renderRecord = requestRecord(body);
    const script = renderRecord.script;
    if (renderRecord.task === 'natural_source_anchor') {
      assert.equal(typeof body.messages[1].content, 'string');
      assert.equal(script, 'Happy cat.');
      assert.equal(JSON.stringify(body).includes('input_audio'), false);
    } else {
      assert.equal(renderRecord.task, 'verbatim_prosody_transfer');
      assert.equal(renderRecord.flow_plan.token_count, 2);
      assert.deepEqual(renderRecord.flow_plan.weak_positions_one_based, []);
      assert.equal(JSON.stringify(renderRecord.flow_plan).includes('cat'), false);
      const inputAudio = body.messages[1].content.find(item => item.type === 'input_audio');
      assert.equal(inputAudio.input_audio.format, 'wav');
      assert.equal(JSON.stringify(body).includes('Happy cat.'), false);
      assert.equal(
        inputAudio.input_audio.data,
        renderRecord.reference_kind === 'source_anchor' ? ANCHOR_AUDIO : NEUTRAL_AUDIO
      );
    }
    await new Promise(resolve => setTimeout(resolve, 10));
    active -= 1;
    const audio = renderRecord.task === 'natural_source_anchor'
      ? ANCHOR_AUDIO
      : renderRecord.reference_kind === 'source_anchor' ? NEUTRAL_AUDIO : LENS_AUDIO;
    return openAiAudioResponse(script, requests.length, audio);
  };
  const response = await handleLegacyTypedAudio(typedRequest(), env, makeCtx(), fetchImpl);
  const payload = await response.json();
  assert.equal(response.status, 200);
  assert.equal(payload.status, 'ready');
  assert.equal(payload.api_calls_made, 5);
  assert.equal(payload.attempts.length, 5);
  assert.deepEqual(payload.attempts.map(attempt => attempt.side), ['anchor', 'neutral', 'neutral', 'lens', 'lens']);
  assert.equal(payload.selection.method, 'audio_reference_prosody_chain_v1');
  assert.equal(payload.selection.anchor_take, 1);
  assert.equal(payload.selection.neutral_take, 1);
  assert.equal(payload.selection.lens_take, 1);
  assert.equal(payload.audio.neutral.sha256, NEUTRAL_SHA);
  assert.equal(payload.audio.lens.sha256, LENS_SHA);
  assert.equal(requests.length, 5);
  assert.equal(peak, MAX_RENDER_CONCURRENCY);
});

test('valid but rejected neutral slots are final and do not trigger replacement or lens calls', async () => {
  const { env } = typedEnv({ inspect(body) {
    return !body.expected_script.startsWith('bavd');
  } });
  let calls = 0;
  const fetchImpl = async (_url, init) => {
    calls += 1;
    const script = requestRecord(JSON.parse(init.body)).script;
    return openAiAudioResponse(script, calls);
  };
  const response = await handleLegacyTypedAudio(typedRequest(), env, makeCtx(), fetchImpl);
  const payload = await response.json();
  assert.equal(response.status, 503);
  assert.equal(payload.error, 'no_reference_matched_neutral');
  assert.equal(payload.api_calls_made, 3);
  assert.equal(calls, 3);
  assert.deepEqual(payload.attempts.map(attempt => attempt.side), ['anchor', 'neutral', 'neutral']);
});

test('an invalid anchor stops the dependency chain after one paid call', async () => {
  const { env } = typedEnv({ inspect: () => false });
  let calls = 0;
  const fetchImpl = async (_url, init) => {
    calls += 1;
    const script = requestRecord(JSON.parse(init.body)).script;
    return openAiAudioResponse(script, calls);
  };
  const response = await handleLegacyTypedAudio(typedRequest(), env, makeCtx(), fetchImpl);
  const payload = await response.json();
  assert.equal(response.status, 503);
  assert.equal(payload.error, 'no_verified_anchor');
  assert.equal(payload.api_calls_made, 1);
  assert.equal(calls, 1);
});

test('a transport failure is inconclusive rather than evidence against the architecture', async () => {
  const { env } = typedEnv();
  const response = await handleLegacyTypedAudio(
    typedRequest(), env, makeCtx(),
    async () => new Response('temporary', { status: 503, headers: { 'x-request-id': 'req-external' } })
  );
  const payload = await response.json();
  assert.equal(response.status, 503);
  assert.equal(payload.error, 'inconclusive_external_failure');
  assert.equal(payload.api_calls_made, 1);
});

test('overall typed deadline is below the browser timeout and aborts outstanding bounded work', async () => {
  assert.ok(OVERALL_SERVER_DEADLINE_MS < 150_000);
  const { env } = candidateEnv();
  let calls = 0;
  let aborted = 0;
  env.TRANSFORM_SERVICE = {
    async fetch(request) {
      calls += 1;
      return new Promise((resolve, reject) => {
        request.signal.addEventListener('abort', () => {
          aborted += 1;
          reject(request.signal.reason);
        }, { once: true });
      });
    }
  };
  const started = Date.now();
  const response = await handleTypedAudio(
    typedRequest(), env, makeCtx(), fetch,
    { deadlineMs: 60 }
  );
  const payload = await response.json();
  assert.equal(response.status, 503);
  assert.equal(payload.error, 'server_deadline_exceeded');
  assert.equal(calls, 1);
  assert.equal(aborted, calls);
  assert.ok(Date.now() - started < 1_000);
});

function azureSideFixture(label) {
  const bytes = Buffer.alloc(64);
  bytes.write('RIFF', 0);
  bytes.write(label, 8);
  return {
    wav_base64: bytes.toString('base64'),
    wav_sha256: createHash('sha256').update(bytes).digest('hex'),
    byte_count: bytes.length
  };
}

function azureReadyPayload(overrides = {}) {
  return {
    schema_version: 1,
    status: 'ready_azure_lane',
    lane_version: 'azure-lens-lane-v1',
    profile_id: 'en-US-to-pt-BR-listener-v2',
    locale: 'en-US',
    voice: 'en-US-AvaNeural',
    listener_locale: 'pt-BR',
    speaker_voice: 'pt-BR-FranciscaNeural',
    normalized_text: 'the cat naps',
    words: [{ written: 'cat', source_phone: 'kæt', lens_phone: 'kɛt', applied_rule_ids: ['enpt.ae_eh'] }],
    applied_rule_ids: ['enpt.ae_eh'],
    map_neutralized_rule_ids: ['enpt.schwa_reduced_a'],
    context_absent_rule_ids: ['enpt.lexical_stress_initial_bias'],
    renderer_inaudible_rule_ids: [],
    omitted_rule_ids: [],
    prosody: { polar_question: false, contour_applied: false },
    affected_word_count: 1,
    audio: { neutral: azureSideFixture('neutral'), lens: azureSideFixture('lens'), speaker: azureSideFixture('speaker') },
    api_calls_made: 3,
    cache_hit: false,
    ...overrides
  };
}

function azureEnv(payload, counters = { calls: 0 }) {
  return {
    AZURE_LENS_CANDIDATE_ENABLED: 'true',
    TRANSFORM_SERVICE: {
      async fetch(request) {
        counters.calls += 1;
        if (new URL(request.url).pathname !== '/azure-lens') {
          return Response.json({ error: 'not_found' }, { status: 404 });
        }
        return payload instanceof Response ? payload : Response.json(payload);
      }
    }
  };
}

test('azure lens route is fail-closed while the flag is false', async () => {
  const response = await handleRequest(
    new Request('https://example.com/api/azure-lens', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ text: 'The cat naps', profile_id: 'en-US-to-pt-BR-listener-v2' })
    }),
    { AZURE_LENS_CANDIDATE_ENABLED: 'false', GIBBERISH_CANDIDATE_ENABLED: 'false' },
    makeCtx()
  );
  assert.equal(response.status, 503);
  assert.equal((await response.json()).error, 'azure_lens_disabled');
});

test('azure lens rejects malformed requests before any upstream call', async () => {
  const counters = { calls: 0 };
  const env = azureEnv(azureReadyPayload(), counters);
  for (const body of [
    { text: 'The cat naps' },
    { text: 'The cat naps', profile_id: 'unknown' },
    { text: '', profile_id: 'en-US-to-pt-BR-listener-v2' },
    { text: 'x'.repeat(201), profile_id: 'en-US-to-pt-BR-listener-v2' }
  ]) {
    const response = await handleRequest(
      new Request('https://example.com/api/azure-lens', {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify(body)
      }),
      env,
      makeCtx()
    );
    assert.equal(response.status, 422);
  }
  assert.equal(counters.calls, 0);
});

test('azure lens passes a verified upstream pair through', async () => {
  const response = await handleRequest(
    new Request('https://example.com/api/azure-lens', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ text: 'The cat naps', profile_id: 'en-US-to-pt-BR-listener-v2' })
    }),
    azureEnv(azureReadyPayload()),
    makeCtx()
  );
  assert.equal(response.status, 200);
  const payload = await response.json();
  assert.equal(payload.status, 'ready_azure_lane');
  assert.equal(payload.applied_rule_ids[0], 'enpt.ae_eh');
  assert.deepEqual(payload.map_neutralized_rule_ids, ['enpt.schwa_reduced_a']);
  assert.ok(payload.audio.neutral.wav_sha256 !== payload.audio.lens.wav_sha256);
});

test('azure lens rejects a tampered upstream hash', async () => {
  const tampered = azureReadyPayload();
  tampered.audio.lens.wav_sha256 = tampered.audio.neutral.wav_sha256;
  const response = await handleRequest(
    new Request('https://example.com/api/azure-lens', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ text: 'The cat naps', profile_id: 'en-US-to-pt-BR-listener-v2' })
    }),
    azureEnv(tampered),
    makeCtx()
  );
  assert.equal(response.status, 502);
  assert.equal((await response.json()).error, 'azure_lens_contract_invalid');
});

test('azure lens surfaces upstream unavailability without inventing audio', async () => {
  const response = await handleRequest(
    new Request('https://example.com/api/azure-lens', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ text: 'The cat naps', profile_id: 'en-US-to-pt-BR-listener-v2' })
    }),
    azureEnv(Response.json({ status: 'unavailable', error: 'azure_lens_key_missing', api_calls_made: 0 }, { status: 503 })),
    makeCtx()
  );
  assert.equal(response.status, 503);
  assert.equal((await response.json()).error, 'azure_lens_key_missing');
});

test('azure lens forwards the actionable refusal detail, bounded', async () => {
  const longDetail = 'x'.repeat(500);
  const response = await handleRequest(
    new Request('https://example.com/api/azure-lens', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ text: 'Pi is 3.14', profile_id: 'en-US-to-pt-BR-listener-v2' })
    }),
    azureEnv(Response.json({
      status: 'unavailable', error: 'azure_lens_rejected',
      detail: "The token '3.14' is not a word this lane can verify; " + longDetail,
      api_calls_made: 0
    }, { status: 422 })),
    makeCtx()
  );
  assert.equal(response.status, 422);
  const payload = await response.json();
  assert.equal(payload.error, 'azure_lens_rejected');
  assert.ok(payload.detail.includes("token '3.14'"));
  assert.ok(payload.detail.length <= 200, 'detail must stay bounded');
});

// ---- Gibberish mode ----
// The request carries a source locale, never a listener direction. The
// contract check that matters is that the two sides differ: a
// gibberish pair whose halves render identically has done nothing.

function gibberishReadyPayload(overrides = {}) {
  return {
    schema_version: 1,
    status: 'ready_gibberish_lane',
    lane_version: 'gibberish-lane-v1',
    locale: 'en-US',
    listener_locale: null,
    profile_id: null,
    voice: 'en-US-AvaNeural',
    normalized_text: 'the cat naps',
    words: [
      { written: 'the', gibberish_phone: 'noʊ', heard_phone: null, syllable_count: 1 },
      { written: 'cat', gibberish_phone: 'kəli', heard_phone: null, syllable_count: 1 },
      { written: 'naps', gibberish_phone: 'ʃən', heard_phone: null, syllable_count: 1 }
    ],
    core_size: 90,
    syllable_shape: 'prefer_open',
    vowel_reduction: true,
    audio: { neutral: azureSideFixture('neutral'), gibberish: azureSideFixture('gibbrsh') },
    api_calls_made: 2,
    cache_hit: false,
    ...overrides
  };
}

function gibberishHeardPayload(overrides = {}) {
  return gibberishReadyPayload({
    listener_locale: 'es-ES',
    profile_id: 'en-US-to-es-ES-listener-v1',
    words: [
      { written: 'the', gibberish_phone: 'noʊ', heard_phone: 'no', syllable_count: 1 },
      { written: 'cat', gibberish_phone: 'kəli', heard_phone: 'kali', syllable_count: 1 },
      { written: 'naps', gibberish_phone: 'ʃən', heard_phone: 'ʃan', syllable_count: 1 }
    ],
    ...overrides
  });
}

function gibberishEnv(payload, counters = { calls: 0 }) {
  return {
    GIBBERISH_CANDIDATE_ENABLED: 'true',
    TRANSFORM_SERVICE: {
      async fetch(request) {
        counters.calls += 1;
        if (new URL(request.url).pathname !== '/gibberish') {
          return Response.json({ error: 'not_found' }, { status: 404 });
        }
        return payload instanceof Response ? payload : Response.json(payload);
      }
    }
  };
}

function gibberishRequest(body) {
  return new Request('https://example.com/api/gibberish', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(body)
  });
}

test('gibberish route is fail-closed while its own flag is false', async () => {
  const response = await handleRequest(
    gibberishRequest({ text: 'The cat naps', source_locale: 'en-US' }),
    { GIBBERISH_CANDIDATE_ENABLED: 'false' },
    makeCtx()
  );
  assert.equal(response.status, 503);
  assert.equal((await response.json()).error, 'gibberish_disabled');
});

test('gibberish gates on its own flag, not the lens flag', async () => {
  const response = await handleRequest(
    gibberishRequest({ text: 'The cat naps', source_locale: 'en-US' }),
    { AZURE_LENS_CANDIDATE_ENABLED: 'true' },
    makeCtx()
  );
  assert.equal(response.status, 503);
  assert.equal((await response.json()).error, 'gibberish_disabled');
});

test('gibberish rejects malformed requests before any upstream call', async () => {
  const counters = { calls: 0 };
  const env = gibberishEnv(gibberishReadyPayload(), counters);
  for (const body of [
    { text: 'The cat naps' },
    { source_locale: 'en-US' },
    { text: 'The cat naps', source_locale: 'xx-XX' },
    { text: '   ', source_locale: 'en-US' },
    { text: 'x'.repeat(201), source_locale: 'en-US' },
    { text: 'The cat naps', source_locale: 'en-US', core_size: 40 },
    // Sound Minus Meaning is source-only. Any listener field is rejected.
    { text: 'The cat naps', source_locale: 'en-US', listener_locale: 'xx-XX' },
    { text: 'The cat naps', source_locale: 'en-US', listener_locale: 'en-US' },
    { text: 'The cat naps', source_locale: 'en-US', listener_locale: 42 }
  ]) {
    const response = await handleRequest(gibberishRequest(body), env, makeCtx());
    assert.equal(response.status, 422, JSON.stringify(body));
  }
  assert.equal(counters.calls, 0);
});

test('gibberish forwards only source text and source locale', async () => {
  const seen = [];
  const env = {
    GIBBERISH_CANDIDATE_ENABLED: 'true',
    TRANSFORM_SERVICE: {
      async fetch(request) {
        seen.push(await request.clone().json());
        return Response.json(gibberishReadyPayload());
      }
    }
  };
  const body = { text: 'The cat naps', source_locale: 'en-US' };
  const response = await handleRequest(gibberishRequest(body), env, makeCtx());
  assert.equal(response.status, 200);
  assert.equal((await response.json()).listener_locale, null);
  assert.deepEqual(seen, [body]);
});

test('gibberish refuses an upstream answer coloured by a listener', async () => {
  const response = await handleRequest(
    gibberishRequest({ text: 'The cat naps', source_locale: 'en-US' }),
    gibberishEnv(gibberishHeardPayload()),
    makeCtx()
  );
  assert.equal(response.status, 502);
  assert.equal((await response.json()).error, 'gibberish_contract_invalid');
});

test('gibberish passes a verified upstream pair through', async () => {
  const response = await handleRequest(
    gibberishRequest({ text: 'The cat naps', source_locale: 'en-US' }),
    gibberishEnv(gibberishReadyPayload()),
    makeCtx()
  );
  assert.equal(response.status, 200);
  const payload = await response.json();
  assert.equal(payload.status, 'ready_gibberish_lane');
  assert.equal(payload.locale, 'en-US');
  assert.equal(payload.words.length, 3);
  assert.ok(payload.audio.neutral.wav_sha256 !== payload.audio.gibberish.wav_sha256);
});

test('gibberish refuses a pair whose two sides are the same audio', async () => {
  const identical = gibberishReadyPayload();
  identical.audio.gibberish = identical.audio.neutral;
  const response = await handleRequest(
    gibberishRequest({ text: 'The cat naps', source_locale: 'en-US' }),
    gibberishEnv(identical),
    makeCtx()
  );
  assert.equal(response.status, 502);
  assert.equal((await response.json()).error, 'gibberish_contract_invalid');
});

test('gibberish rejects a tampered upstream hash', async () => {
  const tampered = gibberishReadyPayload();
  tampered.audio.gibberish.wav_sha256 = 'f'.repeat(64);
  const response = await handleRequest(
    gibberishRequest({ text: 'The cat naps', source_locale: 'en-US' }),
    gibberishEnv(tampered),
    makeCtx()
  );
  assert.equal(response.status, 502);
  assert.equal((await response.json()).error, 'gibberish_contract_invalid');
});

test('gibberish rejects an upstream locale that is not the one requested', async () => {
  const response = await handleRequest(
    gibberishRequest({ text: 'The cat naps', source_locale: 'en-US' }),
    gibberishEnv(gibberishReadyPayload({ locale: 'de-DE' })),
    makeCtx()
  );
  assert.equal(response.status, 502);
  assert.equal((await response.json()).error, 'gibberish_contract_invalid');
});

test('gibberish rejects an unknown syllable shape', async () => {
  const response = await handleRequest(
    gibberishRequest({ text: 'The cat naps', source_locale: 'en-US' }),
    gibberishEnv(gibberishReadyPayload({ syllable_shape: 'freestyle' })),
    makeCtx()
  );
  assert.equal(response.status, 502);
  assert.equal((await response.json()).error, 'gibberish_contract_invalid');
});

test('gibberish surfaces upstream unavailability without inventing audio', async () => {
  const response = await handleRequest(
    gibberishRequest({ text: 'The cat naps', source_locale: 'en-US' }),
    gibberishEnv(Response.json(
      { status: 'unavailable', error: 'gibberish_key_missing', api_calls_made: 0 },
      { status: 503 }
    )),
    makeCtx()
  );
  assert.equal(response.status, 503);
  assert.equal((await response.json()).error, 'gibberish_key_missing');
});

test('gibberish forwards the actionable refusal detail, bounded', async () => {
  const response = await handleRequest(
    gibberishRequest({ text: 'Pi is 3.14', source_locale: 'en-US' }),
    gibberishEnv(Response.json({
      status: 'unavailable', error: 'gibberish_rejected',
      detail: "unmapped adapter symbol in gibberish '3.14' " + 'x'.repeat(500),
      api_calls_made: 0
    }, { status: 422 })),
    makeCtx()
  );
  assert.equal(response.status, 422);
  const payload = await response.json();
  assert.equal(payload.error, 'gibberish_rejected');
  assert.ok(payload.detail.length <= 200, 'detail must stay bounded');
});

test('gibberish allowlist matches the frozen syllable bank', async () => {
  const { GIBBERISH_LOCALES } = await import('../worker/gibberish-locales.generated.js');
  const bank = (await import('../rules/gibberish-syllable-cores-v1.json', { with: { type: 'json' } })).default;
  assert.deepEqual([...GIBBERISH_LOCALES].sort(), Object.keys(bank.locales).sort());
  assert.equal(GIBBERISH_LOCALES.length, 30, 'parity: every language the lens offers has a bank');
});
