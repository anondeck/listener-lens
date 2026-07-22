#!/usr/bin/env node
import { spawn } from 'node:child_process';
import { existsSync } from 'node:fs';
import { readFile, stat } from 'node:fs/promises';
import { createServer } from 'node:http';
import { dirname, extname, resolve, sep } from 'node:path';
import { fileURLToPath } from 'node:url';

import worker from '../worker/app.js';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const SITE_ROOT = resolve(ROOT, 'site');
const LOCAL_PORT = Number(process.env.LOCAL_PORT || 8789);
const TRANSFORM_PORT = Number(process.env.TRANSFORM_PORT || 8790);
const PAID_SENTINEL = 'I_UNDERSTAND_THIS_MAKES_PAID_API_CALLS';
const audioServeEnabled = process.env.LOCAL_TYPED_AUDIO_SERVE_ENABLED !== 'false';
const activityGenerationEnabled = process.env.LOCAL_ACTIVITY_GENERATION_ENABLED === PAID_SENTINEL;
const REQUIRED_DISABLED_CANDIDATE_FLAGS = [
  'KOKORO_ENGLISH_CANDIDATE_ENABLED',
  'KOKORO_BILINGUAL_CANDIDATE_ENABLED',
  'PORTUGUESE_RENDERER_CANDIDATE_ENABLED',
  'RECIPROCAL_LISTENER_LENS_RESEARCH_ENABLED'
];
const candidateFlagEnv = Object.fromEntries(REQUIRED_DISABLED_CANDIDATE_FLAGS.map(name => [name, process.env[name] ?? 'false']));
const kokoroCandidateEnabled = candidateFlagEnv.KOKORO_ENGLISH_CANDIDATE_ENABLED === 'true';
const bilingualCandidateEnabled = candidateFlagEnv.KOKORO_BILINGUAL_CANDIDATE_ENABLED === 'true';
// The Azure lane is the owner-selected product path; local dev enables it by
// default (set AZURE_LENS_CANDIDATE_ENABLED=false to opt out). Tracked deploy
// configs keep it exactly false until the owner flips it at deploy time.
const azureLensFlag = process.env.AZURE_LENS_CANDIDATE_ENABLED ?? 'true';
if (!['false', 'true'].includes(azureLensFlag)) {
  throw new Error('AZURE_LENS_CANDIDATE_ENABLED must be exactly true or false.');
}
const azureLensEnabled = azureLensFlag === 'true';
// Gibberish is a second mode with its own claim, gated on its own flag so it
// can be tried locally without the lens and pulled without touching it.
const gibberishFlag = process.env.GIBBERISH_CANDIDATE_ENABLED ?? 'true';
if (!['false', 'true'].includes(gibberishFlag)) {
  throw new Error('GIBBERISH_CANDIDATE_ENABLED must be exactly true or false.');
}
const gibberishEnabled = gibberishFlag === 'true';
const audioRenderEnabled = kokoroCandidateEnabled || bilingualCandidateEnabled;
const paidCallsEnabled = activityGenerationEnabled;

if (!['false', 'true'].includes(candidateFlagEnv.KOKORO_ENGLISH_CANDIDATE_ENABLED)) {
  throw new Error('KOKORO_ENGLISH_CANDIDATE_ENABLED must be exactly true or false.');
}
if (!['false', 'true'].includes(candidateFlagEnv.KOKORO_BILINGUAL_CANDIDATE_ENABLED)) {
  throw new Error('KOKORO_BILINGUAL_CANDIDATE_ENABLED must be exactly true or false.');
}
if (kokoroCandidateEnabled && bilingualCandidateEnabled) {
  throw new Error('Only one Kokoro candidate path may be enabled at a time.');
}
for (const name of ['PORTUGUESE_RENDERER_CANDIDATE_ENABLED', 'RECIPROCAL_LISTENER_LENS_RESEARCH_ENABLED']) {
  if (candidateFlagEnv[name] !== 'false') throw new Error(`${name} must remain exactly false in the local product runtime.`);
}

if (!Number.isInteger(LOCAL_PORT) || !Number.isInteger(TRANSFORM_PORT)) {
  throw new Error('LOCAL_PORT and TRANSFORM_PORT must be integers.');
}

class MemoryCache {
  constructor() { this.entries = new Map(); }
  async match(request) {
    const response = this.entries.get(request.url);
    return response ? response.clone() : undefined;
  }
  async put(request, response) { this.entries.set(request.url, response.clone()); }
}

class WindowLimiter {
  constructor(limit, periodMs) {
    this.limitValue = limit;
    this.periodMs = periodMs;
    this.entries = new Map();
  }
  async limit({ key }) {
    const now = Date.now();
    const recent = (this.entries.get(key) || []).filter(value => now - value < this.periodMs);
    if (recent.length >= this.limitValue) return { success: false };
    recent.push(now);
    this.entries.set(key, recent);
    return { success: true };
  }
}

class MemoryBudget {
  constructor() { this.entries = new Map(); }
  key(day, clientHash) { return `${day}:${clientHash}`; }
  async reserveRenders(clientHash, requested, day) {
    const globalKey = `${day}:global`;
    const clientKey = this.key(day, clientHash);
    const globalUsed = this.entries.get(globalKey) || 0;
    const clientUsed = this.entries.get(clientKey) || 0;
    if (globalUsed + requested > 100) return { allowed: false, reason: 'daily_global_budget' };
    if (clientUsed + requested > 5) return { allowed: false, reason: 'daily_client_budget' };
    this.entries.set(globalKey, globalUsed + requested);
    this.entries.set(clientKey, clientUsed + requested);
    return { allowed: true, reserved: requested };
  }
  async releaseRenders(clientHash, released, day) {
    const globalKey = `${day}:global`;
    const clientKey = this.key(day, clientHash);
    this.entries.set(globalKey, Math.max(0, (this.entries.get(globalKey) || 0) - released));
    this.entries.set(clientKey, Math.max(0, (this.entries.get(clientKey) || 0) - released));
  }
}

globalThis.caches = { default: new MemoryCache() };

const mimeTypes = new Map([
  ['.css', 'text/css; charset=utf-8'],
  ['.html', 'text/html; charset=utf-8'],
  ['.js', 'text/javascript; charset=utf-8'],
  ['.json', 'application/json; charset=utf-8'],
  ['.svg', 'image/svg+xml'],
  ['.wav', 'audio/wav']
]);

async function assetResponse(request) {
  const url = new URL(request.url);
  const decoded = decodeURIComponent(url.pathname);
  const relative = decoded === '/' ? 'index.html' : decoded.replace(/^\/+/, '');
  let path = resolve(SITE_ROOT, relative);
  if (path !== SITE_ROOT && !path.startsWith(SITE_ROOT + sep)) return new Response('Not found', { status: 404 });
  try {
    if ((await stat(path)).isDirectory()) path = resolve(path, 'index.html');
    const body = await readFile(path);
    return new Response(body, {
      status: 200,
      headers: { 'content-type': mimeTypes.get(extname(path)) || 'application/octet-stream' }
    });
  } catch {
    return new Response('Not found', { status: 404 });
  }
}

function transformService() {
  return {
    async fetch(request) {
      const source = new URL(request.url);
      const target = `http://127.0.0.1:${TRANSFORM_PORT}${source.pathname}${source.search}`;
      return fetch(new Request(target, request));
    }
  };
}

const budget = new MemoryBudget();
const env = {
  ASSETS: { fetch: assetResponse },
  TRANSFORM_SERVICE: transformService(),
  RENDER_BUDGET: budget,
  ACTIVITY_RATE_LIMITER: new WindowLimiter(5, 60_000),
  AUDIO_RATE_LIMITER: new WindowLimiter(2, 60_000),
  TYPED_AUDIO_SERVE_ENABLED: audioServeEnabled ? 'true' : 'false',
  TYPED_AUDIO_RENDER_ENABLED: audioRenderEnabled ? 'true' : 'false',
  ACTIVITY_GENERATION_ENABLED: activityGenerationEnabled ? 'true' : 'false',
  AZURE_FOUNDRY_ENDPOINT: activityGenerationEnabled ? process.env.AZURE_FOUNDRY_ENDPOINT : undefined,
  AZURE_FOUNDRY_API_KEY: activityGenerationEnabled ? process.env.AZURE_FOUNDRY_API_KEY : undefined,
  AZURE_FOUNDRY_DEPLOYMENT: process.env.AZURE_FOUNDRY_DEPLOYMENT || 'luna',
  AZURE_LENS_CANDIDATE_ENABLED: azureLensEnabled ? 'true' : 'false',
  GIBBERISH_CANDIDATE_ENABLED: gibberishEnabled ? 'true' : 'false',
  ...candidateFlagEnv,
  OPENAI_API_KEY: paidCallsEnabled ? process.env.OPENAI_API_KEY : undefined
};

if (activityGenerationEnabled && (!env.AZURE_FOUNDRY_ENDPOINT || !env.AZURE_FOUNDRY_API_KEY)) {
  throw new Error('Classroom generation was enabled, but Azure Foundry endpoint or API key is absent.');
}

function pythonCommand() {
  const venvPython = resolve(ROOT, '.venv/bin/python');
  if (existsSync(venvPython)) return { command: venvPython, args: ['scripts/run_deploy_service.py'] };
  return { command: 'uv', args: ['run', 'python', 'scripts/run_deploy_service.py'] };
}

const python = pythonCommand();
const transformEnv = {
  ...process.env,
  HOST: '127.0.0.1',
  PORT: String(TRANSFORM_PORT),
  AZURE_LENS_CANDIDATE_ENABLED: azureLensEnabled ? 'true' : 'false',
  GIBBERISH_CANDIDATE_ENABLED: gibberishEnabled ? 'true' : 'false'
};
delete transformEnv.OPENAI_API_KEY;
const transform = spawn(python.command, python.args, {
  cwd: ROOT,
  env: transformEnv,
  stdio: ['ignore', 'inherit', 'inherit']
});

let transformExit;
transform.once('exit', (code, signal) => { transformExit = { code, signal }; });

async function waitForTransform() {
  for (let attempt = 0; attempt < 100; attempt += 1) {
    if (transformExit) throw new Error(`Transform service exited before readiness: ${JSON.stringify(transformExit)}`);
    try {
      const response = await fetch(`http://127.0.0.1:${TRANSFORM_PORT}/health`);
      const body = await response.json();
      if (response.ok && body.status === 'ok' && body.nonce_gate_enabled === true) return body;
    } catch {
      // The Python process is still starting.
    }
    await new Promise(resolvePromise => setTimeout(resolvePromise, 100));
  }
  throw new Error('Transform service did not become ready within 10 seconds.');
}

function requestBody(incoming) {
  return new Promise((resolvePromise, reject) => {
    const chunks = [];
    let length = 0;
    incoming.on('data', chunk => {
      length += chunk.length;
      if (length > 7 * 1024 * 1024) {
        reject(new Error('Local request exceeds 7 MiB.'));
        incoming.destroy();
        return;
      }
      chunks.push(chunk);
    });
    incoming.on('end', () => resolvePromise(Buffer.concat(chunks)));
    incoming.on('error', reject);
  });
}

const health = await waitForTransform();
const server = createServer(async (incoming, outgoing) => {
  try {
    const host = incoming.headers.host || `127.0.0.1:${LOCAL_PORT}`;
    const url = `http://${host}${incoming.url || '/'}`;
    const body = ['GET', 'HEAD'].includes(incoming.method || 'GET') ? undefined : await requestBody(incoming);
    const request = new Request(url, {
      method: incoming.method,
      headers: incoming.headers,
      body: body?.length ? body : undefined
    });
    const pending = [];
    const ctx = { waitUntil(promise) { pending.push(Promise.resolve(promise)); } };
    const response = await worker.fetch(request, env, ctx);
    outgoing.writeHead(response.status, Object.fromEntries(response.headers.entries()));
    outgoing.end(Buffer.from(await response.arrayBuffer()));
    await Promise.allSettled(pending);
  } catch (error) {
    console.error('local_request_failed', error instanceof Error ? error.message : 'unknown_error');
    if (!outgoing.headersSent) outgoing.writeHead(500, { 'content-type': 'application/json' });
    outgoing.end(JSON.stringify({ error: 'local_gateway_failure' }));
  }
});

server.listen(LOCAL_PORT, '127.0.0.1', () => {
  console.log(`Local product: http://127.0.0.1:${LOCAL_PORT}`);
  console.log(`Transform: ${health.service_version}; eSpeak 1.52.0; nonce gates enabled`);
  console.log(`Typed audio serve: ${audioServeEnabled ? 'enabled' : 'disabled'}`);
  console.log(`Kokoro candidate: ${kokoroCandidateEnabled ? 'EXPLICITLY ENABLED (local only)' : 'disabled'}`);
  console.log(`Bilingual Kokoro candidate: ${bilingualCandidateEnabled ? 'EXPLICITLY ENABLED (local only)' : 'disabled'}`);
  console.log(`Azure lens lane: ${azureLensEnabled ? 'enabled (local default; key read from .env.local)' : 'disabled'}`);
  console.log(`Gibberish mode: ${gibberishEnabled ? 'enabled (local default; key read from .env.local)' : 'disabled'}`);
  console.log(`Typed audio render: ${audioRenderEnabled ? 'enabled with zero API calls' : 'disabled'}`);
  console.log(`Activity generation: ${activityGenerationEnabled ? 'EXPLICITLY ENABLED' : 'disabled'}`);
});

function shutdown() {
  server.close(() => process.exit(0));
  if (!transform.killed) transform.kill('SIGTERM');
  setTimeout(() => process.exit(1), 2_000).unref();
}

process.on('SIGINT', shutdown);
process.on('SIGTERM', shutdown);
transform.on('exit', (code, signal) => {
  if (!server.listening) return;
  console.error(`Transform service stopped unexpectedly (${code ?? signal}).`);
  server.close(() => process.exit(1));
});
