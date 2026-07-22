# Deployment record

## Current state — July 17, 2026

No service is live. `deployment-state.json` records `active: false` and
`state: "deleted"`; no replacement Worker, Pages project, container, or typed
runtime has been deployed. The former URL contained a personal identifier and
has been redacted; it is historical only and returned HTTP `404` after deletion.

Every externally reachable or candidate route is fail-closed in `wrangler.jsonc`:

```json
"TYPED_AUDIO_SERVE_ENABLED": "false",
"TYPED_AUDIO_RENDER_ENABLED": "false",
"ACTIVITY_GENERATION_ENABLED": "false",
"KOKORO_ENGLISH_CANDIDATE_ENABLED": "false",
"PORTUGUESE_RENDERER_CANDIDATE_ENABLED": "false",
"RECIPROCAL_LISTENER_LENS_RESEARCH_ENABLED": "false"
```

The July 17 expansion made zero paid, model-inference, or generative API calls
and incurred no API spend. The Kokoro.js/ONNX audit did make read-only public
HTTP/metadata requests to official npm registry records, immutable GitHub source
and commit records, Hugging Face repository metadata, small configuration files
and LFS metadata, and official license sources. It downloaded no model or voice
payload and ran no synthesis or benchmark. Historical paid renderer and activity
calls remain separately recorded below.

Current research constraints also block promotion:

- the fresh unseen Kokoro confirmation closed as
  `fresh_unseen_fixture_confirmation_automatic_failed_no_review`, with
  `no_positive_generalization_claim` and no human review;
- both English and Portuguese Kokoro voice sessions remain pending and no voice
  is selected;
- Portuguese renderer QC is blocked on voice selection;
- the Portuguese G2P characterization is
  `characterization_complete_nonpromotional`, but its parent index remains
  `partial_positive_only_index` and a negative lookup cannot clear; its
  `characterization.json` SHA-256 is
  `59f67cec5d033868bb0a45d21d9dc1fe43c945fae1f86336b019432f54e1deb0`;
- the container benchmark is frozen but unexecuted because the current Mac has
  no Docker-compatible engine;
- the Kokoro.js/ONNX audit is `complete_source_only_no_runtime_execution`, not a
  runtime or production result; `audit.json` has SHA-256
  `0418ce98f10e7f744608f4336c7bddfe23cee69b1bd7cb5ca06f5c25f4df7532`;
- both independent Track D v2 reviewers approved all 15 source findings before
  the reciprocal PT-BR→AmE protocol was frozen. One bounded run made exactly
  five local decoder calls with zero retries or variants and passed runtime
  integrity, then closed `automatic_measurement_inconclusive` because
  ordinary-anchor-lens at 5500 Hz retained 2/2 frames below the frozen minimum
  of five. The protocol internal SHA-256 is
  `002fef936f04c293624046badc8d6f5c58b5bc3ab2858a24b3bee3bb68db2a69`;
  `analysis.json` has SHA-256
  `24c99df1a04087f84752c8420720638c281593aea27583a307037ea283c928e0`.
  Localization was skipped, no parameters changed, no rerun or review artifact
  exists, and no positive, perceptual, selected-voice, production, or promotion
  claim follows. `pf_dora` remains a nontransferable technical probe and all
  candidate flags remain false.

## Historical deployment verification — July 15, 2026

The following receipt describes a deployment that was subsequently deleted. It
is retained for provenance and does not describe current availability.

- Historical public URL: redacted (deleted deployment)
- Verified at: `2026-07-15T17:02:10Z`
- Source commit: `98fb88b74bac35c6be8fc31587e8440a4c513f98`
- Cloudflare Worker version: `cf0eb6ab-75e1-4cda-88ce-b078bd9fff1e`
- Wrangler: `4.111.0`
- Activity contract: `teacher-activity-v2`

Historical checks recorded:

- `/api/health`: status `ok`, model `gpt-5.6`, API secret configured.
- One uncached paid activity smoke: Responses ID
  `resp_09b337d18fb9b0cb016a57bca1746c8191ace1bab1c088befc`, 241 input
  tokens, 439 output tokens, 680 total tokens.
- The immediate identical request was a cache hit with the same Responses ID;
  it made no second model generation.
- A live Playwright check found the concept-preview label, three isolated
  evidence cards, a rendered GPT-5.6 activity, zero browser errors, and zero
  mobile horizontal overflow.
- Static responses carried CSP, `nosniff`, `DENY` framing, a restrictive
  Permissions Policy, and a strict-origin referrer policy.

The historical audio surface used two static, hash-frozen WAV files. Loading or
using its A/B player made no OpenAI API call.

## Disabled typed-runtime candidate

The repository contains a bounded typed-runtime design. If a cache miss ever
reaches an authorized renderer, its manifest is fixed: one accepted source
anchor, two neutral renders referenced to that anchor, and two lens renders
referenced to the selected neutral. There is no retry or replacement render; an
invalid required stage returns unavailable and may stop the manifest early. A
successful fresh request therefore makes exactly five calls. The proposed Worker
uses `teacher-activity-v4` so its updated mission-grounded fallback and instructions
would not reuse the historical activity cache.

Future deployment would still require all of the following:

- a Docker-compatible engine building the pinned private transform image;
- a Cloudflare Workers Paid account, which Containers require;
- container health confirming eSpeak, rules, nonce-gate, and database hashes;
- approval of the exact first live-smoke manifest and cost cap; and
- verification of the live request, cache hit, rate limit, daily budget, and
  static fallback.

No typed-audio API request was made while implementing this candidate path.

## Remote takedown — July 15, 2026

At `2026-07-15T23:09:52Z`, the project owner directed that nothing remain live.
Wrangler successfully deleted the remote `listener-lens-build-week` Worker. The
former Workers URL subsequently returned HTTP `404`. The local repository and
artifacts remain intact, and the workspace has no configured Git remote.
