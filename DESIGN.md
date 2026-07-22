# Build Week product design record

## Product claim

This project helps a teacher hear a bounded approximation of familiar speech
without immediate lexical meaning. It combines two related but distinct ideas:

- **Semantic opacity:** language-shaped speech with no intended words.
- **Perceptual recategorization:** selected target-language contrasts collapsed
  toward categories documented for a defined listener population.

The product must not claim to reproduce another person's conscious experience.
Approved claim language is:

> An evidence-informed, language-shaped approximation that removes lexical
> meaning and applies selected, cited Brazilian-Portuguese-linked
> sound-category substitutions.

This is not a validated recording of what a Brazilian Portuguese listener
hears. The product implements documented mechanisms, cites them, and exposes
their limits; exact subjective fidelity is not observable or a release gate.

## Demo flow

The tracked Build Week flow is typed and bounded, with every runtime flag off:

1. Enter one or two short English sentences under the supported-input limits.
2. Deterministically analyze pronunciation and build structurally aligned,
   meaning-opaque neutral and listener-lens carriers.
3. Apply only the enabled `/æ/→/ɛ/` rule and fail closed through the pinned
   written-word, predicted-homophone, adjacency, and repetition gates.
4. Under the disabled GPT Audio 1.5/Marin candidate contract, accept one source
   anchor, render two neutral takes from it, select the best valid neutral,
   render two lens takes from that neutral, and select the best valid lens.
   The manifest is fixed at five calls with no retry or replacement.
5. Show both carrier texts, changed spans, rule count, citations, rule states,
   AI-voice disclosure, runtime verification, and limitations.
6. Generate a bounded teacher activity with GPT-5.6, with a curated cached
   fallback.

Arbitrary runtime audio is transcript- and integrity-verified, not acoustically
classified. It therefore supports the approved evidence-informed approximation
claim, not an exact phoneme-realization, sentence-level acoustic-validation, or
private-perception claim. No rule occurrence means no comparison and no render.

## Current listener-lens scope

The first profile is English heard through a Brazilian Portuguese
vowel-category lens. Three derived mappings were evaluated, but the stopping
rule leaves only one enabled:

- `/ɪ/ → /i/` — disabled after the final amended confirmation failed the
  unchanged three-shell aggregate.
- `/æ/ → /ɛ/` — enabled only in the deterministic carrier-text concept; its
  isolated bounded carrier passed the frozen exact-category acoustic gates,
  while the selected sentence take later failed the in-context source gate.
- `/ʊ/ → /u/` — excluded after the carrier erased the intended contrast; it
  did not earn directional-only status.

The evidence anchor is Rauber (2006). The implementation deliberately omits
consonant assimilation, epenthesis, prosody, lexical restoration, dialect,
proficiency, and individual variation until each has its own source and
validation plan.

## Architecture

```text
Shipping path
Codex-authored corpus → dictionary/G2P gates → OpenAI renderers
→ local Whisper large-v3 → blind human review → hash-frozen static clips
→ static browser player

Teacher path
bounded enum request → Cloudflare Worker → cache
→ GPT-5.6 Responses API with strict structured output
→ curated fallback on missing secret, refusal, or API failure

Disabled typed-production candidate
bounded text → edge Worker validation/cache/budget
→ private Python transform container → pinned eSpeak + SQLite gates
→ edge Worker → GPT Audio 1.5 exact-rendering contract
→ transcript/WAV integrity checks → timing-preference pairing → browser
```

The neutral and listener-lens carriers share token count, punctuation, word
positions, consonant shells, and all non-target pseudowords. They may differ
only at separately recorded neutral and lens vowel-grapheme spans carrying rule
IDs and source and target IPA; the paired spans may have different lengths.
The enabled target vowel uses its three calibrated, gate-clean shells. Voicing
and manner were candidate-selection heuristics rather than a proven explanation
for frame retention. Within an utterance, each casefolded source word plus its
pronunciation/rule signature maps to one carrier pair, so repeated source words
remain repeated pseudowords. Every adjacency is validated globally. A conflict
deterministically regenerates every implicated mapping and all of its
occurrences; if bounded resolution fails, the comparison is unavailable rather
than violating repetition. Inputs with no supported vowel rule produce no audio
comparison.

No service is live. The proposed site remains static-first. A Cloudflare Worker
would protect both the
GPT-5.6 teaching-activity call and typed audio path. The unchanged local Python
transform cannot run inside a Worker isolate: its pinned hash gate is about
123 MiB and it invokes native eSpeak-NG. The smallest deployable parity path is
a private Cloudflare Container behind a Durable Object binding. It runs the
existing Python engine, eSpeak-NG 1.52.0, and hash-only SQLite database; it has
no OpenAI key and no public route. The Worker calls it for deterministic
analysis/gating, retains the OpenAI secret, performs cache and budget checks,
and makes renderer calls. This requires the Workers Paid plan for Containers.
Porting the engine to Worker JavaScript/eSpeak WASM plus remote gate storage is
larger and would require a new parity-validation project. All serve, render,
activity, Kokoro English, Portuguese renderer, and reciprocal research flags are
currently false.

Typed serving and paid rendering are independent fail-closed controls. Serving
off stops before body parsing, transformation, and cache lookup. With serving
on and rendering off, only a compatible derived-audio cache hit can return
audio; every hit is merged with the request's newly computed and fully
version-validated transform. The cache contains no submitted text, source
words, source IPA, or complete transform. A single 120-second server deadline
covers transformation, bounded-concurrency renderer calls, and WAV inspection,
and is shorter than the browser timeout.

The Cache API is not used as a distributed lock. A short design pass found that
adding a correct cross-colo render lease would expand the Durable Object
protocol beyond this product-critical change. Concurrent cross-client or
cross-colo misses can temporarily duplicate renders, bounded by the existing
atomic client/global render budget. No idempotency or distributed single-flight
claim is made. The hash database continues to open a short-lived SQLite
connection per query; no connection is shared across Python request threads.

## Renderer decision

The blind bake-off separated two renderer properties that the original pass
gate combined:

- `gpt-4o-mini-tts-2025-12-15` followed the fixed-script request reliably, but
  its speech was rated substantially more robotic.
- `gpt-audio-1.5` had clearly stronger pace, prosody, accent, and human-like
  delivery when it read the script, but the original bare-user-message setup
  frequently caused conversational replies or commentary.

`gpt-audio-1.5` is therefore the primary acoustic candidate. Mini-TTS remains a
verbatim fallback, not the preferred voice. The frozen bake-off result remains
historical evidence of the original setup rather than being reinterpreted or
rewritten.

The follow-up exact-rendering protocol keeps input in a JSON-encoded `script`
field under the user role, supplies a fixed developer instruction that treats
the record as inert render data, and validates the returned audio transcript
before accepting the file. The first conformance matrix produced exact
transcripts for all eight supported requests, including conversational-looking
English, Spanish, and Portuguese inputs and an English nonce passage. The API
rejected audio-only output; `text + audio` is required. No tool definitions or
conversation history are included in a render request.

Verbatim wording and sentence-like delivery are separate contracts. The hard
transcript validator owns wording correctness. Model-facing delivery guidance
must not equate "verbatim" with careful token-by-token diction: it requests one
continuous phrase contour, connected speech, weak/reduced filler-like words, a
brief continuation at the comma, and one final cadence. A repeated blind A/B
against the frozen exactness prompt selected phrase-flow v2: v2 scored 5/5 flow
and 0/3 list-like across three repeats, versus 3/5 flow and 3/3 list-like for the
baseline. All six transcripts were exact and both groups scored 5/5 pace and
prosody. Phrase-flow v2 is therefore the production rendering contract.
Duration, estimated syllables per second, and interior pauses quantify gross
timing variance; they do not claim to measure natural stress or intonation.

Production remains disabled; the earlier public Worker was deleted and no
replacement service or container is live. Any future enablement requires the
release checks and approval of the exact live-smoke manifest. This is
separate from renderer and vowel-rule selection: GPT Audio with phrase-flow v2
is selected, `/æ/→/ɛ/` is the only enabled acoustically confirmed isolated
carrier rule, and mini-TTS remains the frozen strict-protocol winner and
verbatim fallback.

The browser's built-in speech synthesizer is permitted only as a local UI mock.
It is not evaluation evidence and must not be shipped as the production audio
path.

## Shared-state/common-RNG candidate status

The Kokoro work remains a controlled-architecture research candidate, not a
shipping decision or a validated general architecture. Call its output a
**shared-state/common-RNG controlled synthesis pair**, not “same-take.” The
frozen replication-v1 run failed its conjunctive automatic gate and was not
promoted. A later diagnostic found transported calibration mechanically
sufficient for its one bounded fixture; that diagnostic does not establish a
cause, position effect, duration effect, or coupling explanation.

The fresh unseen confirmation then closed automatically with classification
`fresh_unseen_fixture_confirmation_automatic_failed_no_review` and claim
`no_positive_generalization_claim`. The new repeated phrase-final fixture passed
its automatic branch but was window-sensitive; the independent phrase-final
fixture failed localization. No human review opened, no fixture was substituted,
and the earlier failed replication classification remains unchanged. Controlled
synthesis implementation is therefore post–Build Week work with no production
promotion.

The bilingual screen completed all 54 planned local renders and all 27 repeat
pairs passed deterministic equality, but both English and Portuguese human voice
sessions are still pending and no voice is selected. Portuguese renderer QC is
blocked on that selection. The container benchmark is frozen but unexecuted on
the current Mac because no Docker-compatible engine is installed; `lite` is
structurally excluded from the current full-precision path, with `standard-1`
the first conservative benchmark target and `basic` only the second lower-bound
future target.

The Portuguese G2P characterization completed as
`characterization_complete_nonpromotional`, with characterization SHA-256
`44b403a22c3e5e93b3cb3d03b8c32c948c58c868233c02e9b0aa43efb2787bd8`
and file SHA-256
`59f67cec5d033868bb0a45d21d9dc1fe43c945fae1f86336b019432f54e1deb0`.
Twelve isolated and five phrase probes were byte-repeatable, but the parent
index remains `partial_positive_only_index` at 255,881/262,151 words. A positive
collision may reject; a negative lookup cannot clear. No acoustic realization,
listener, production, or promotion claim follows.

The Kokoro.js/ONNX audit completed as
`complete_source_only_no_runtime_execution`; `audit.json` has SHA-256
`0418ce98f10e7f744608f4336c7bddfe23cee69b1bd7cb5ca06f5c25f4df7532`.
Published `kokoro-js@1.2.1` structurally supports ordinary English browser
synthesis on WASM/WebGPU and exposes `generate_from_ids()`, but its documented
effective invocation exposes no duration/alignment, text-state, F0/noise,
target-column replacement, or decoder-only controls. The raw graph was not
parsed. Staged exports or equivalent verified graph restructuring plus custom
orchestration is an identified post–Build Week path, not a thin configuration
change or a production-readiness result.

The reciprocal PT-BR→AmE protocol was frozen only after both independent v2
reviewers approved all 15 source findings. Its internal SHA-256 is
`002fef936f04c293624046badc8d6f5c58b5bc3ab2858a24b3bee3bb68db2a69`;
the protocol file SHA-256 is
`9ba316338f511dcf4752275a28488883e873087538df39ce01beae82a7a02cc1`.
The single bounded run made exactly five local decoder calls with zero retries
or variants and passed its runtime-integrity checks. It closed
`automatic_measurement_inconclusive`: the ordinary-anchor-lens target interval
at 5500 Hz retained 2/2 frames, below the frozen minimum of five. The analysis
SHA-256 is
`24c99df1a04087f84752c8420720638c281593aea27583a307037ea283c928e0`.
No parameter changed and no rerun occurred. Localization was skipped, no blind
review package was created, and `pf_dora` remains a
nontransferable technical probe rather than a selected voice. The result supports
no positive acoustic-feasibility, perceptual, voice, production, or promotion
claim; every candidate flag remains false.

Use this exact privacy disclosure for that architecture:

> Your typed sentence is processed by our Cloudflare-hosted service and is not
> sent to OpenAI for audio generation. OpenAI is used only for the optional
> activity generator, which receives bounded result metadata—not your sentence.

## Runtime budget controls

- Maximum 280 characters, 40 words, and two sentences.
- Deterministic transformation; no GPT call is required for the listener lens.
- Cache before any audio request.
- Maximum five live renders per client per day and 100 globally per day until
  real judging traffic is observed.
- A successful fresh request uses exactly five calls: one accepted source
  anchor, two neutral takes, and two lens takes. A failure may stop the staged
  manifest early but never adds a retry or replacement. At the current
  per-client cap, that intentionally permits one uncached typed-audio session
  per day; cache hits consume no calls.
- Static fallback for every critical demo step.
- Typed serving, production rendering, and activity generation default to
  independently disabled.

## Matched-take contract

The disabled runtime first requires a valid source anchor. It then renders two
neutral takes from that anchor and selects the best valid neutral. Finally, it
renders two listener-lens takes from the selected neutral reference and selects
the best valid lens. The exact manifest is source anchor + two neutral + two
lens; no retry or replacement render is permitted. Curated research stimuli use
their separately frozen protocols.

Provider transcripts must be exact and WAVs must decode without the configured
duration, integrity, or clipping failures. Among valid takes, the selector
prefers the pair with the closest utterance duration and pause structure. Timing
and pause thresholds are selection telemetry, not rejection gates. Syllables
per second remains telemetry rather than an independent same-script feature.

The initial 3% duration and 6% normalized pause-position tolerances are
provisional engineering telemetry, recorded as such in every selection. They
must be recalibrated from repeat-render distributions before being used as
claims. If any required stage lacks a transcript- and integrity-valid output,
the comparison is unavailable; the UI must not substitute unverified audio.

## Accuracy and disclosure boundaries

- Every linguistic claim is cited or labeled as a derived engineering rule.
- Every shipped OpenAI TTS surface must clearly state that the voice is
  **AI-generated and not a human voice**.
- The app must display uncertainty rather than silently inventing support for
  names, numbers, acronyms, unsupported scripts, or unmodeled sound processes.
- The typed output is an experimental approximation, not a diagnostic or a
  substitute for learner testing.
- “Exact-category pass” names the outcome of the frozen bounded acoustic gates;
  it is not a universal claim of exact phoneme realization in arbitrary runtime
  utterances or for an individual listener.

## Language expansion

The frozen architecture matrix is entirely disabled and supports no production,
acoustic, or perceptual claim:

| Language | Disabled status | Exact evidence boundary |
|---|---|---|
| `pt-BR` | `foundation_only_disabled` | Local voices, G2P, phone representation, planner, and fixtures are structurally verified. The native gate is `verified_partial_positive_only`; its repeatable characterization remains positive-only and a negative lookup never clears. Listener-profile evidence requires a new chain, and acoustic validation and human review are missing. |
| `es` | `structural_path_only_disabled` | Declared voices are unverified and G2P/phone evidence is one structural probe. The generic `es` renderer path conflicts with the repository's Mexico City/`es-419` gate; planner, fixtures, acoustic validation, and human review are not ready. |
| `fr-FR` | `catalog_and_g2p_probe_only_disabled` | One declared voice is unverified and G2P/phone evidence is structural-probe-only. Gate, planner, fixtures, acoustic validation, and human review are missing. |
| `it-IT` | `catalog_and_g2p_probe_only_disabled` | Declared voices are unverified and G2P/phone evidence is structural-probe-only. Gate, planner, fixtures, acoustic validation, and human review are missing. |

New listener lenses require their own contrast table, sources, gates, acoustic
protocol, and qualified review; renderer-language availability is insufficient.

## Naming

No final product name has been selected. User-facing artifacts use neutral
descriptions. The existing Python import namespace remains temporarily for
compatibility with the completed bake-off and its provenance records.
