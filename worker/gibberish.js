import { readBoundedJson } from './request-utils.js';
// Generated from the frozen syllable bank the generator reads, so the edge
// allowlist cannot drift from the set of languages that actually have a bank.
// Still a fail-closed boundary: an unknown locale is rejected before any
// upstream call. Regenerate with scripts/sync_lane_allowlists_v1.py.
export { GIBBERISH_LOCALES } from './gibberish-locales.generated.js';
import { GIBBERISH_LOCALES } from './gibberish-locales.generated.js';

const MAX_TEXT_CHARS = 200;
const RESPONSE_KEYS = Object.freeze([
  'api_calls_made', 'audio', 'cache_hit', 'core_size', 'lane_version',
  'listener_locale', 'locale', 'normalized_text', 'profile_id',
  'schema_version', 'status', 'syllable_shape', 'vowel_reduction', 'voice',
  'words'
]);
const AUDIO_SIDE_KEYS = Object.freeze(['byte_count', 'wav_base64', 'wav_sha256']);
const WORD_KEYS = Object.freeze([
  'gibberish_phone', 'heard_phone', 'syllable_count', 'written'
]);
const SYLLABLE_SHAPES = Object.freeze(['match_source', 'prefer_open']);

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

async function validGibberishPayload(payload, requested) {
  if (!exactKeys(payload, RESPONSE_KEYS)) return false;
  if (payload.schema_version !== 1 || payload.status !== 'ready_gibberish_lane') return false;
  if (payload.lane_version !== 'gibberish-lane-v1') return false;
  if (payload.locale !== requested.source_locale) return false;
  if (typeof payload.voice !== 'string' || !payload.voice) return false;
  if (typeof payload.normalized_text !== 'string' || !payload.normalized_text) return false;
  if (![0, 2].includes(payload.api_calls_made) || typeof payload.cache_hit !== 'boolean') return false;
  if (!Number.isInteger(payload.core_size) || payload.core_size < 1) return false;
  if (!SYLLABLE_SHAPES.includes(payload.syllable_shape)) return false;
  if (typeof payload.vowel_reduction !== 'boolean') return false;
  if (payload.listener_locale !== null || payload.profile_id !== null) return false;
  if (!Array.isArray(payload.words) || payload.words.length < 1) return false;
  if (!payload.words.every(word => exactKeys(word, WORD_KEYS)
    && typeof word.written === 'string' && word.written
    && typeof word.gibberish_phone === 'string' && word.gibberish_phone
    && Number.isInteger(word.syllable_count) && word.syllable_count >= 1
    && word.heard_phone === null)) return false;
  if (!exactKeys(payload.audio, ['gibberish', 'neutral'])) return false;
  if (!(await validSide(payload.audio.neutral)) || !(await validSide(payload.audio.gibberish))) return false;
  // The whole mode is the difference between the two sides. Identical audio
  // means the phoneme override did nothing and the pair would be a lie, so it
  // is refused here rather than played.
  return payload.audio.neutral.wav_sha256 !== payload.audio.gibberish.wav_sha256;
}

export async function handleGibberish(request, env) {
  if (env.GIBBERISH_CANDIDATE_ENABLED !== 'true') {
    return json({ error: 'gibberish_disabled', api_calls_made: 0 }, 503);
  }
  const body = await readBoundedJson(request, 64 * 1024);
  if (body instanceof Response) return body;
  if (
    Object.keys(body ?? {}).sort().join('|') !== 'source_locale|text'
    || typeof body.text !== 'string' || typeof body.source_locale !== 'string'
    || !body.text.trim() || body.text.length > MAX_TEXT_CHARS
    || !GIBBERISH_LOCALES.includes(body.source_locale)
  ) {
    return json({ error: 'unsupported_gibberish_request' }, 422);
  }
  const upstreamBody = JSON.stringify({
    text: body.text,
    source_locale: body.source_locale
  });
  let upstream;
  try {
    upstream = await env.TRANSFORM_SERVICE.fetch(new Request('http://transform-service/gibberish', {
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
    return json({ error: 'gibberish_upstream_unreachable' }, 502);
  }
  let payload;
  try {
    payload = await upstream.json();
  } catch {
    return json({ error: 'gibberish_upstream_invalid' }, 502);
  }
  if (upstream.status !== 200) {
    const error = typeof payload?.error === 'string' ? payload.error : 'gibberish_upstream_failed';
    // The service names the offending word when it refuses a sentence.
    // Bounded so an upstream cannot use it to smuggle arbitrary payloads.
    const detail = typeof payload?.detail === 'string' ? payload.detail.slice(0, 200) : undefined;
    return json({ error, ...(detail ? { detail } : {}), api_calls_made: 0 }, [422, 502, 503].includes(upstream.status) ? upstream.status : 502);
  }
  if (!(await validGibberishPayload(payload, body))) {
    console.log(JSON.stringify({ event: 'gibberish_contract_rejected', status: 'invalid_upstream_payload' }));
    return json({ error: 'gibberish_contract_invalid' }, 502);
  }
  console.log(JSON.stringify({
    event: 'gibberish_ready',
    locale: payload.locale,
    voice: payload.voice,
    word_count: payload.words.length,
    cache_hit: payload.cache_hit,
    api_calls_made: payload.api_calls_made
  }));
  return json(payload);
}
