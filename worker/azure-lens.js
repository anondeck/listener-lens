import { jsonBodyErrorResponse, readBoundedJson } from './request-utils.js';
// Generated from the same registry the service validates against, so the
// edge allowlist cannot drift from what the lane can actually build. Still a
// fail-closed boundary: an unknown profile id is rejected before any
// upstream call. Regenerate with scripts/sync_lane_allowlists_v1.py.
export { AZURE_LENS_PROFILES } from './azure-lens-profiles.generated.js';
import { AZURE_LENS_PROFILES } from './azure-lens-profiles.generated.js';
const MAX_TEXT_CHARS = 200;
const RESPONSE_KEYS = Object.freeze([
  'affected_word_count', 'api_calls_made', 'applied_rule_ids', 'audio',
  'cache_hit', 'context_absent_rule_ids', 'lane_version', 'listener_locale',
  'locale', 'map_neutralized_rule_ids', 'normalized_text',
  'omitted_rule_ids', 'profile_id', 'prosody', 'renderer_inaudible_rule_ids',
  'schema_version', 'speaker_voice', 'status', 'voice', 'words'
]);
const AUDIO_SIDE_KEYS = Object.freeze(['byte_count', 'wav_base64', 'wav_sha256']);

function json(value, status = 200, headers = {}) {
  return new Response(JSON.stringify(value), {
    status,
    headers: { 'content-type': 'application/json; charset=utf-8', 'cache-control': 'no-store', ...headers }
  });
}

function exactKeys(value, keys) {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value)
    && Object.keys(value).sort().join('|') === [...keys].sort().join('|');
}

function decodeBase64(text) {
  const binary = atob(text);
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) bytes[index] = binary.charCodeAt(index);
  return bytes;
}

async function sha256Hex(bytes) {
  const digest = await crypto.subtle.digest('SHA-256', bytes);
  return [...new Uint8Array(digest)].map(value => value.toString(16).padStart(2, '0')).join('');
}

async function validSide(side) {
  if (!exactKeys(side, AUDIO_SIDE_KEYS)) return false;
  if (typeof side.wav_base64 !== 'string' || !/^[a-f0-9]{64}$/.test(side.wav_sha256)) return false;
  let bytes;
  try {
    bytes = decodeBase64(side.wav_base64);
  } catch {
    return false;
  }
  if (bytes.length !== side.byte_count || bytes.length < 44) return false;
  if (String.fromCharCode(...bytes.slice(0, 4)) !== 'RIFF') return false;
  return (await sha256Hex(bytes)) === side.wav_sha256;
}

async function validAzureLensPayload(payload, requested) {
  if (!exactKeys(payload, RESPONSE_KEYS)) return false;
  if (payload.schema_version !== 1 || payload.status !== 'ready_azure_lane') return false;
  if (payload.lane_version !== 'azure-lens-lane-v1') return false;
  if (payload.profile_id !== requested.profile_id) return false;
  if (!Number.isInteger(payload.affected_word_count) || payload.affected_word_count < 1) return false;
  if (![0, 3].includes(payload.api_calls_made) || typeof payload.cache_hit !== 'boolean') return false;
  if (typeof payload.listener_locale !== 'string' || typeof payload.speaker_voice !== 'string') return false;
  if (!Array.isArray(payload.applied_rule_ids) || payload.applied_rule_ids.length < 1) return false;
  if (!Array.isArray(payload.map_neutralized_rule_ids) || !Array.isArray(payload.omitted_rule_ids)) return false;
  if (!Array.isArray(payload.context_absent_rule_ids)) return false;
  // A rule the voice renders identically on both sides must never also be
  // reported as applied; the two buckets are disjoint by construction.
  if (!Array.isArray(payload.renderer_inaudible_rule_ids)) return false;
  if (payload.renderer_inaudible_rule_ids.some(
    ruleId => payload.applied_rule_ids.includes(ruleId)
  )) return false;
  if (!exactKeys(payload.prosody, ['contour_applied', 'polar_question'])
    || typeof payload.prosody.contour_applied !== 'boolean'
    || typeof payload.prosody.polar_question !== 'boolean'
    || (payload.prosody.contour_applied && !payload.prosody.polar_question)) return false;
  if (!Array.isArray(payload.words) || !payload.words.every(word =>
    exactKeys(word, ['applied_rule_ids', 'lens_phone', 'source_phone', 'written'])
    && typeof word.written === 'string'
    && typeof word.source_phone === 'string' && word.source_phone.length > 0
    && typeof word.lens_phone === 'string' && word.lens_phone.length > 0
    && Array.isArray(word.applied_rule_ids))) return false;
  if (!exactKeys(payload.audio, ['lens', 'neutral', 'speaker'])) return false;
  if (!(await validSide(payload.audio.neutral)) || !(await validSide(payload.audio.lens))) return false;
  // The speaker track is the listener language's own voice reading the raw
  // text — production, not perception — and is verified like the other sides.
  if (!(await validSide(payload.audio.speaker))) return false;
  return payload.audio.neutral.wav_sha256 !== payload.audio.lens.wav_sha256;
}

export async function handleAzureLens(request, env) {
  if (env.AZURE_LENS_CANDIDATE_ENABLED !== 'true') {
    return json({ error: 'azure_lens_disabled', api_calls_made: 0 }, 503);
  }
  const body = await readBoundedJson(request, 64 * 1024);
  if (body instanceof Response) return body;
  if (
    !exactKeys(body, ['profile_id', 'text'])
    || typeof body.text !== 'string' || typeof body.profile_id !== 'string'
    || !body.text.trim() || body.text.length > MAX_TEXT_CHARS
    || !AZURE_LENS_PROFILES.includes(body.profile_id)
  ) {
    return json({ error: 'unsupported_azure_lens_request' }, 422);
  }
  const upstreamBody = JSON.stringify({ text: body.text, profile_id: body.profile_id });
  let upstream;
  try {
    upstream = await env.TRANSFORM_SERVICE.fetch(new Request('http://transform-service/azure-lens', {
      method: 'POST',
      headers: {
        'content-type': 'application/json',
        // An explicit length keeps the local dev proxy from re-streaming the
        // body as chunked, which the transform service rejects fail-closed.
        'content-length': String(new TextEncoder().encode(upstreamBody).byteLength)
      },
      body: upstreamBody
    }));
  } catch {
    return json({ error: 'azure_lens_upstream_unreachable' }, 502);
  }
  let payload;
  try {
    payload = await upstream.json();
  } catch {
    return json({ error: 'azure_lens_upstream_invalid' }, 502);
  }
  if (upstream.status !== 200) {
    const error = typeof payload?.error === 'string' ? payload.error : 'azure_lens_upstream_failed';
    // The service names the offending word in `detail` when it refuses a
    // sentence. Forward it — a refusal the user can act on is the difference
    // between "broken" and "write the number out as a word". Bounded so an
    // upstream cannot use it to smuggle arbitrary payloads to the page.
    const detail = typeof payload?.detail === 'string' ? payload.detail.slice(0, 200) : undefined;
    return json({ error, ...(detail ? { detail } : {}), api_calls_made: 0 }, [422, 502, 503].includes(upstream.status) ? upstream.status : 502);
  }
  if (!(await validAzureLensPayload(payload, body))) {
    console.log(JSON.stringify({ event: 'azure_lens_contract_rejected', status: 'invalid_upstream_payload' }));
    return json({ error: 'azure_lens_contract_invalid' }, 502);
  }
  console.log(JSON.stringify({
    event: 'azure_lens_ready',
    profile_id: payload.profile_id,
    voice: payload.voice,
    applied_rule_count: payload.applied_rule_ids.length,
    cache_hit: payload.cache_hit,
    api_calls_made: payload.api_calls_made
  }));
  return json(payload);
}
