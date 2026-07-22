export const MAX_PUBLIC_JSON_BYTES = 2048;

export class JsonBodyError extends Error {
  constructor(code) {
    super(code);
    this.name = 'JsonBodyError';
    this.code = code;
  }
}

export async function readBoundedJson(request, maxBytes = MAX_PUBLIC_JSON_BYTES) {
  if (!Number.isInteger(maxBytes) || maxBytes < 1) throw new TypeError('invalid_json_body_limit');
  if (!request.body) throw new JsonBodyError('invalid_json');

  const reader = request.body.getReader();
  const chunks = [];
  let total = 0;
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      total += value.byteLength;
      if (total > maxBytes) {
        try { await reader.cancel('request_too_large'); } catch { /* The byte gate still wins. */ }
        throw new JsonBodyError('request_too_large');
      }
      chunks.push(value);
    }
  } finally {
    reader.releaseLock();
  }

  const bytes = new Uint8Array(total);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.byteLength;
  }
  try {
    const text = new TextDecoder('utf-8', { fatal: true }).decode(bytes);
    return JSON.parse(text);
  } catch (error) {
    if (error instanceof JsonBodyError) throw error;
    throw new JsonBodyError('invalid_json');
  }
}

export function jsonBodyErrorResponse(error, json) {
  if (!(error instanceof JsonBodyError)) return null;
  return error.code === 'request_too_large'
    ? json({ error: error.code }, 413)
    : json({ error: error.code }, 400);
}
