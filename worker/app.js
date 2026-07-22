import {
  candidateFlagsAreExactlyFalse,
  handleBilingualTypedAudio,
  handleTypedAudio,
  productCapabilityCatalog,
  productVoiceCatalog
} from './typed-audio.js';
import { jsonBodyErrorResponse, readBoundedJson } from './request-utils.js';
import { handleAzureLens } from './azure-lens.js';
import { handleGibberish } from './gibberish.js';
import { AZURE_LENS_PROFILES } from './azure-lens-profiles.generated.js';

const DEFAULT_ACTIVITY_DEPLOYMENT = 'luna';
const ACTIVITY_VERSION = 'teacher-activity-v5-foundry';
const ALLOWED_GRADES = new Set(['elementary', 'middle', 'secondary', 'adult']);
const ALLOWED_MINUTES = new Set([10, 20, 30]);
const PROFILE_IDS = new Set(AZURE_LENS_PROFILES);
const RULE_ID = /^[a-z0-9][a-z0-9._-]{0,79}$/;

const DURATION_SPLITS = new Map([
  [10, [2, 4, 4]],
  [20, [4, 8, 8]],
  [30, [5, 12, 13]]
]);

export function fallbackActivity(minutes) {
  const split = DURATION_SPLITS.get(minutes);
  if (!split) throw new TypeError('unsupported_activity_duration');
  return {
    title: 'Hear, compare, and explain the sound shift',
    objective: 'Use the generated A/B comparison to notice the selected sound changes and connect them to a concrete language-learning goal.',
    warmup: [
      'Play A once without explanation and ask learners to note what sounds different in B.',
      'Reveal the highlighted words and name the selected source and listener languages.'
    ],
    listen_for: [
      'Which highlighted sounds changed most clearly?',
      'Can you imitate A and B without changing the sentence rhythm?'
    ],
    practice_steps: [
      { minutes: split[0], instruction: 'Play A, B, then A again. Learners mark the highlighted words that changed.', teacher_note: 'Keep attention on the returned rule receipt.' },
      { minutes: split[1], instruction: 'Learners alternate the source and listener-lens pronunciations for the selected contrast.', teacher_note: 'Model the change slowly, then return to normal sentence speed.' },
      { minutes: split[2], instruction: 'Pairs create one new example and predict whether the selected rule will apply.', teacher_note: 'Check predictions against the app before discussing the pattern.' }
    ],
    exit_ticket: 'Name one changed sound and one word where you expect to hear it again.',
    evidence_note: 'Use only the rules shown in the generated comparison; the lens is a research-informed approximation, not a claim about every individual listener.'
  };
}

export const FALLBACK_ACTIVITY = fallbackActivity(20);

const ACTIVITY_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  properties: {
    title: { type: 'string', minLength: 4, maxLength: 90 },
    objective: { type: 'string', minLength: 10, maxLength: 240 },
    warmup: { type: 'array', minItems: 2, maxItems: 3, items: { type: 'string', minLength: 5, maxLength: 220 } },
    listen_for: { type: 'array', minItems: 2, maxItems: 4, items: { type: 'string', minLength: 5, maxLength: 220 } },
    practice_steps: {
      type: 'array', minItems: 3, maxItems: 4,
      items: {
        type: 'object', additionalProperties: false,
        properties: {
          minutes: { type: 'integer', minimum: 1, maximum: 20 },
          instruction: { type: 'string', minLength: 8, maxLength: 300 },
          teacher_note: { type: 'string', minLength: 8, maxLength: 240 }
        },
        required: ['minutes', 'instruction', 'teacher_note']
      }
    },
    exit_ticket: { type: 'string', minLength: 8, maxLength: 240 },
    evidence_note: { type: 'string', minLength: 12, maxLength: 300 }
  },
  required: ['title', 'objective', 'warmup', 'listen_for', 'practice_steps', 'exit_ticket', 'evidence_note']
};

function json(value, status = 200, headers = {}) {
  return new Response(JSON.stringify(value), {
    status,
    headers: { 'content-type': 'application/json; charset=utf-8', 'cache-control': 'no-store', ...headers }
  });
}

function securityHeaders(response) {
  const next = new Response(response.body, response);
  next.headers.set('x-content-type-options', 'nosniff');
  next.headers.set('referrer-policy', 'strict-origin-when-cross-origin');
  next.headers.set('x-frame-options', 'DENY');
  next.headers.set('permissions-policy', 'camera=(), microphone=(), geolocation=()');
  next.headers.set('content-security-policy', "default-src 'self'; script-src 'self'; style-src 'self'; media-src 'self' blob:; img-src 'self' data:; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'; form-action 'self'");
  return next;
}

function parseActivityResponse(response) {
  for (const output of response.output || []) {
    if (output.type !== 'message') continue;
    for (const content of output.content || []) {
      if (content.type === 'refusal') throw new Error('model_refusal');
      if (content.type === 'output_text' && content.text) return JSON.parse(content.text);
    }
  }
  throw new Error('missing_structured_output');
}

function exactKeys(value, keys) {
  return value && typeof value === 'object' && !Array.isArray(value)
    && JSON.stringify(Object.keys(value).sort()) === JSON.stringify([...keys].sort());
}

function validResultMetadata(value) {
  return exactKeys(value, ['changed_word_count', 'comparison_status', 'listener_locale', 'profile_id', 'renderer_verification', 'rule_ids', 'source_locale'])
    && PROFILE_IDS.has(value.profile_id)
    && typeof value.source_locale === 'string'
    && typeof value.listener_locale === 'string'
    && value.profile_id.startsWith(`${value.source_locale}-to-${value.listener_locale}-listener-v`)
    && Array.isArray(value.rule_ids)
    && value.rule_ids.length >= 1 && value.rule_ids.length <= 20
    && value.rule_ids.every(ruleId => typeof ruleId === 'string' && RULE_ID.test(ruleId))
    && new Set(value.rule_ids).size === value.rule_ids.length
    && Number.isInteger(value.changed_word_count)
    && value.changed_word_count >= 1 && value.changed_word_count <= 40
    && value.comparison_status === 'ready'
    && value.renderer_verification === 'azure_ssml_pair_returned';
}

function validRequest(body) {
  return exactKeys(body, ['focus', 'grade_band', 'minutes', 'result_metadata'])
    && ALLOWED_GRADES.has(body.grade_band)
    && ALLOWED_MINUTES.has(body.minutes)
    && validResultMetadata(body.result_metadata)
    && body.result_metadata.rule_ids.includes(body.focus);
}

function validActivity(activity, minutes) {
  if (!exactKeys(activity, ['evidence_note', 'exit_ticket', 'listen_for', 'objective', 'practice_steps', 'title', 'warmup'])) return false;
  const boundedStrings = (values, min, max) => Array.isArray(values) && values.length >= min && values.length <= max
    && values.every(value => typeof value === 'string' && value.length >= 5 && value.length <= 300);
  const bounded = (value, min, max) => typeof value === 'string' && value.length >= min && value.length <= max;
  if (!bounded(activity.title, 4, 90) || !bounded(activity.objective, 10, 240)
    || !bounded(activity.exit_ticket, 8, 240) || !bounded(activity.evidence_note, 12, 300)) return false;
  if (!boundedStrings(activity.warmup, 2, 3) || !boundedStrings(activity.listen_for, 2, 4)) return false;
  if (!Array.isArray(activity.practice_steps) || activity.practice_steps.length < 3 || activity.practice_steps.length > 4) return false;
  if (!activity.practice_steps.every(step => exactKeys(step, ['instruction', 'minutes', 'teacher_note'])
    && Number.isInteger(step.minutes) && step.minutes >= 1 && step.minutes <= 20
    && typeof step.instruction === 'string' && step.instruction.length >= 8 && step.instruction.length <= 300
    && typeof step.teacher_note === 'string' && step.teacher_note.length >= 8 && step.teacher_note.length <= 240)) return false;
  return activity.practice_steps.reduce((total, step) => total + step.minutes, 0) === minutes;
}

async function sha256(value) {
  const digest = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(value));
  return [...new Uint8Array(digest)].map(byte => byte.toString(16).padStart(2, '0')).join('');
}

function foundryResponsesUrl(endpoint) {
  const normalized = endpoint.trim().replace(/\/+$/, '');
  if (normalized.endsWith('/responses')) return normalized;
  return normalized.endsWith('/openai/v1')
    ? `${normalized}/responses`
    : `${normalized}/openai/v1/responses`;
}

async function generateActivity(body, resultMetadata, config, fetchImpl) {
  const response = await fetchImpl(foundryResponsesUrl(config.endpoint), {
    method: 'POST',
    headers: { 'api-key': config.apiKey, 'content-type': 'application/json' },
    signal: AbortSignal.timeout(25000),
    body: JSON.stringify({
      model: config.deployment,
      store: false,
      reasoning: { effort: 'none' },
      max_output_tokens: 1200,
      instructions: [
        'You design concise, age-appropriate language-teaching activities.',
        'Ground the activity only in the bounded result metadata supplied by the product. Do not invent sound rules, words, languages, citations, or validation claims.',
        'The audio is a research-informed listener-lens approximation rendered from explicit phoneme rules. It is not private perception and it does not describe every listener.',
        'Create a practical classroom activity that teaches learners to hear and produce the selected contrast while keeping claims proportional to the receipt.',
        `The practice-step minutes must sum to exactly ${body.minutes}.`
      ].join(' '),
      input: [{
        role: 'user',
        content: JSON.stringify({
          grade_band: body.grade_band,
          lesson_minutes: body.minutes,
          focus_rule_id: body.focus,
          result_metadata: resultMetadata
        })
      }],
      text: {
        verbosity: 'low',
        format: { type: 'json_schema', name: 'teacher_activity', strict: true, schema: ACTIVITY_SCHEMA }
      }
    })
  });
  if (!response.ok) throw new Error(`foundry_${response.status}`);
  const payload = await response.json();
  const activity = parseActivityResponse(payload);
  if (!validActivity(activity, body.minutes)) throw new Error('invalid_activity_contract');
  return {
    activity,
    response_id: payload.id || null,
    usage: payload.usage
      ? {
          input_tokens: payload.usage.input_tokens ?? null,
          output_tokens: payload.usage.output_tokens ?? null,
          total_tokens: payload.usage.total_tokens ?? null
        }
      : null
  };
}

export async function handleActivity(request, env, ctx, fetchImpl = fetch) {
  if (!request.headers.get('content-type')?.toLowerCase().startsWith('application/json')) return json({ error: 'content_type_required' }, 415);
  const origin = request.headers.get('origin');
  if (origin && origin !== new URL(request.url).origin) return json({ error: 'origin_not_allowed' }, 403);
  let body;
  try {
    body = await readBoundedJson(request);
  } catch (error) {
    return jsonBodyErrorResponse(error, json) || json({ error: 'invalid_json' }, 400);
  }
  if (!validRequest(body)) return json({ error: 'unsupported_options' }, 422);
  const resultMetadata = body.result_metadata;
  const fallback = fallbackActivity(body.minutes);

  const metadataHash = await sha256(JSON.stringify(resultMetadata));
  const cacheKey = new Request(`https://activity-cache.invalid/${ACTIVITY_VERSION}/${body.grade_band}/${body.minutes}/${body.focus}/${metadataHash}`);
  const cache = globalThis.caches?.default;
  if (cache) {
    const hit = await cache.match(cacheKey);
    if (hit) {
      let cached;
      try { cached = await hit.json(); } catch { cached = null; }
      if (cached && validActivity(cached.activity, body.minutes) && validResultMetadata(cached.result_metadata)) {
        return json({ ...cached, cache_hit: true });
      }
    }
  }
  const foundry = {
    endpoint: env.AZURE_FOUNDRY_ENDPOINT || '',
    apiKey: env.AZURE_FOUNDRY_API_KEY || '',
    deployment: env.AZURE_FOUNDRY_DEPLOYMENT || DEFAULT_ACTIVITY_DEPLOYMENT
  };
  if (env.ACTIVITY_GENERATION_ENABLED !== 'true' || !foundry.endpoint || !foundry.apiKey) {
    return json({ source: 'cached_fallback', activity: fallback, result_metadata: resultMetadata, cache_hit: false });
  }

  if (env.ACTIVITY_RATE_LIMITER) {
    const clientKey = request.headers.get('cf-connecting-ip') || 'unknown-client';
    const { success } = await env.ACTIVITY_RATE_LIMITER.limit({ key: clientKey });
    if (!success) return json({ source: 'cached_fallback', activity: fallback, result_metadata: resultMetadata, cache_hit: false, rate_limited: true });
  }

  try {
    const generated = await generateActivity(body, resultMetadata, foundry, fetchImpl);
    const value = {
      source: 'azure_foundry',
      model: foundry.deployment,
      activity: generated.activity,
      result_metadata: resultMetadata,
      response_id: generated.response_id,
      usage: generated.usage,
      cache_hit: false
    };
    if (cache) {
      const cached = json(value, 200, { 'cache-control': 'public, max-age=604800' });
      ctx.waitUntil(cache.put(cacheKey, cached));
    }
    return json(value);
  } catch (error) {
    console.log(JSON.stringify({ event: 'activity_generation_failed', status: 'fallback', error_name: error instanceof Error ? error.name : 'unknown' }));
    return json({ source: 'cached_fallback', activity: fallback, result_metadata: resultMetadata, cache_hit: false });
  }
}

export async function handleRequest(request, env, ctx, fetchImpl = fetch) {
  const url = new URL(request.url);
  const voiceCatalogOptions = {
    bilingualCandidateEnabled: env.KOKORO_BILINGUAL_CANDIDATE_ENABLED === 'true',
    kokoroCandidateEnabled: env.KOKORO_ENGLISH_CANDIDATE_ENABLED === 'true'
  };
  if (url.pathname === '/api/health' && request.method === 'GET') {
    return json({
      status: 'ok',
      version: ACTIVITY_VERSION,
      activity_provider: 'azure_foundry',
      activity_model: env.AZURE_FOUNDRY_DEPLOYMENT || DEFAULT_ACTIVITY_DEPLOYMENT,
      activity_generation_enabled: env.ACTIVITY_GENERATION_ENABLED === 'true',
      typed_audio_serve_enabled: env.TYPED_AUDIO_SERVE_ENABLED === 'true',
      typed_audio_render_enabled: env.TYPED_AUDIO_RENDER_ENABLED === 'true',
      core_audio_renderer: 'kokoro',
      kokoro_candidate_enabled: env.KOKORO_ENGLISH_CANDIDATE_ENABLED === 'true',
      bilingual_kokoro_candidate_enabled: env.KOKORO_BILINGUAL_CANDIDATE_ENABLED === 'true',
      azure_lens_enabled: env.AZURE_LENS_CANDIDATE_ENABLED === 'true',
      gibberish_enabled: env.GIBBERISH_CANDIDATE_ENABLED === 'true',
      voice_registry_version: productVoiceCatalog(voiceCatalogOptions).registry_version,
      bilingual_matrix_version: productCapabilityCatalog().matrix_version,
      bilingual_changed_rule_cell_count: productCapabilityCatalog().changed_rule_cell_count,
      bilingual_audio_validation_status: productCapabilityCatalog().audio_validation_status,
      configured_voice_count: productVoiceCatalog(voiceCatalogOptions).languages
        .reduce((total, language) => total + language.voices.length, 0),
      candidate_flags_exactly_false: candidateFlagsAreExactlyFalse(env),
      api_configured: Boolean(env.AZURE_FOUNDRY_ENDPOINT && env.AZURE_FOUNDRY_API_KEY)
    });
  }
  if (url.pathname === '/api/voices') {
    if (request.method !== 'GET') return json({ error: 'method_not_allowed' }, 405, { allow: 'GET' });
    return json(productVoiceCatalog(voiceCatalogOptions));
  }
  if (url.pathname === '/api/capabilities') {
    if (request.method !== 'GET') return json({ error: 'method_not_allowed' }, 405, { allow: 'GET' });
    return json(productCapabilityCatalog());
  }
  if (url.pathname === '/api/activity') {
    if (request.method !== 'POST') return json({ error: 'method_not_allowed' }, 405, { allow: 'POST' });
    return handleActivity(request, env, ctx, fetchImpl);
  }
  if (url.pathname === '/api/listener-lens') {
    if (request.method !== 'POST') return json({ error: 'method_not_allowed' }, 405, { allow: 'POST' });
    if (env.KOKORO_BILINGUAL_CANDIDATE_ENABLED === 'true') {
      return handleBilingualTypedAudio(request, env, ctx, fetchImpl);
    }
    return handleTypedAudio(request, env, ctx, fetchImpl);
  }
  if (url.pathname === '/api/azure-lens') {
    if (request.method !== 'POST') return json({ error: 'method_not_allowed' }, 405, { allow: 'POST' });
    return handleAzureLens(request, env);
  }
  if (url.pathname === '/api/gibberish') {
    if (request.method !== 'POST') return json({ error: 'method_not_allowed' }, 405, { allow: 'POST' });
    return handleGibberish(request, env);
  }
  if (url.pathname.startsWith('/api/')) return json({ error: 'not_found' }, 404);
  return env.ASSETS.fetch(request);
}

export default {
  async fetch(request, env, ctx) {
    return securityHeaders(await handleRequest(request, env, ctx));
  }
};
