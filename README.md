# Listener / Lens

> Same words. Different ears.

Listener / Lens is a language-education prototype for comparing a familiar sentence with an evidence-informed approximation of how its sounds may be recategorized through another language.

Choose a source language, choose a listener language, and generate three clearly separated tracks:

- **A — As you say it:** the source-language pronunciation.
- **B — As they may hear it:** the same source voice with supported listener-category rules applied.
- **C — Listener voice reading your words:** a production comparison, not part of the A/B perception approximation.

The current product registry contains **30 languages and 835 directed source-to-listener combinations**. That is structural coverage, not a claim that every direction has the same number or strength of linguistic rules. Every result reports which rules applied, which were absent from the sentence, and which became inaudible or neutralized in the renderer.

The separate **Sound Minus Meaning** mode has **30 source-language options**. It replaces words with deterministic, meaning-opaque material while preserving the source language's syllable structure. It has no listener-language selector and makes no listener-perception claim.

This project was built from scratch during the July 13–21, 2026 submission period for the Education track of OpenAI Build Week.

## Classroom use

A teacher can use the A/B comparison to:

- locate sound contrasts that deserve focused practice;
- replay changed words in the context of a full sentence;
- distinguish a listener-category approximation from a listener voice or accent;
- generate a short classroom activity grounded in the rules that actually appeared.

The classroom generator uses an optional GPT-5.6 deployment named `luna` through Azure AI Foundry's Responses API. It receives only bounded result metadata—direction, applied rule IDs, comparison state, and requested lesson settings—not the teacher's sentence or audio. If the live model is disabled or unavailable, the interface returns a duration-correct curated activity instead.

## Architecture

```text
Browser
  │ same-origin /api requests
  ▼
Cloudflare Worker
  │ bounded bodies, rate limits, exact response contracts
  ▼
Python transform service
  │ G2P, listener rules, SSML construction, validation
  ▼
Azure Speech
  │ WAV audio
  ▼
Worker validation → browser audio players

Optional classroom lane
  Cloudflare Worker → Azure AI Foundry Responses API (deployment: luna)
                     ↘ curated fallback when disabled/unavailable
```

The core listening experience does not call OpenAI or Azure AI Foundry. Azure Speech renders the audio. At runtime, GPT-5.6 is called only for the optional classroom activity; GPT-5.6 in Codex was used throughout development.

## Run locally

### Prerequisites

- Node.js 20+
- Python 3.12.12
- [uv](https://docs.astral.sh/uv/)
- eSpeak-NG 1.52.x
- an Azure Speech key and region

```bash
brew install node uv espeak-ng
npm install
uv sync --all-groups
```

Create `.env.local`:

```dotenv
AZURE_SPEECH_KEY=your-speech-key
AZURE_SPEECH_REGION=your-speech-region

# Optional live classroom generator. `luna` is the Foundry deployment name.
AZURE_FOUNDRY_ENDPOINT=https://your-resource.openai.azure.com
AZURE_FOUNDRY_API_KEY=your-foundry-key
AZURE_FOUNDRY_DEPLOYMENT=luna
LOCAL_ACTIVITY_GENERATION_ENABLED=I_UNDERSTAND_THIS_MAKES_PAID_API_CALLS
```

Do not commit `.env.local`.

Start the complete local product:

```bash
npm run dev:local
```

Open `http://127.0.0.1:8789/`.

## Suggested demo

1. Keep **Listener Lens** selected.
2. Choose **English** as the source and **Brazilian Portuguese** as the listener.
3. Use: `The happy cat sat back and laughed at the bad joke.`
4. Select **Generate listener comparison**.
5. Play A and B, then inspect the highlighted words and evidence receipt.
6. Play C only after explaining that it is a separate production comparison.
7. Generate the classroom activity from one applied rule.
8. Switch to **Sound Minus Meaning** and select **Generate sound-only version**.

The default sentence is intentionally rich in currently supported English→Brazilian-Portuguese changes. It exercises three rule families across eleven words while also producing a complete sound-minus-meaning result.

## HTTP surface

| Method | Route | Purpose |
|---|---|---|
| `GET` | `/api/health` | Runtime and feature status |
| `POST` | `/api/azure-lens` | A/B listener comparison plus C production track |
| `POST` | `/api/gibberish` | Source-only Sound Minus Meaning pair |
| `POST` | `/api/activity` | Foundry GPT-5.6 activity or curated fallback |

Listener comparison request:

```bash
curl --request POST http://127.0.0.1:8789/api/azure-lens \
  --header 'content-type: application/json' \
  --data '{
    "text": "The happy cat sat back and laughed at the bad joke.",
    "profile_id": "en-US-to-pt-BR-listener-v2"
  }'
```

Sound Minus Meaning request:

```bash
curl --request POST http://127.0.0.1:8789/api/gibberish \
  --header 'content-type: application/json' \
  --data '{
    "text": "The happy cat sat back and laughed at the bad joke.",
    "source_locale": "en-US"
  }'
```

## Verification

The current product release suite contains **201 Python tests and 58 Worker tests**.

```bash
npm run test:worker
.venv/bin/python -m pytest \
  tests/test_azure_lens_builder.py \
  tests/test_azure_source_adapters.py \
  tests/test_deploy_service.py \
  tests/test_gibberish_generator.py \
  tests/test_release_gate.py \
  -q
npm run deploy:check
```

The release repository contains the current product source, tests, and concise product documentation. Generated audio, local model/cache data, review outputs, and scratch results are intentionally excluded.

## How Codex and GPT-5.6 were used

GPT-5.6 in Codex was used throughout the core build—not added only as a runtime label. Codex accelerated the project by:

- researching and encoding the listener-category rule registry;
- building the transformation, Azure SSML, validation, Worker, and browser layers;
- generating test harnesses for renderer checks, audio validation, and failure analysis;
- tracing privacy, caching, concurrency, licensing, and deployment risks across the complete request path;
- implementing the final interface, responsive layout, release checks, and regression tests;
- maintaining an evidence trail when candidate approaches failed instead of silently promoting them.

The collaboration also shaped several key decisions. GPT-5.6 helped diagnose the list-like delivery and take-to-take confounds in the original GPT Audio approach, evaluate deterministic Kokoro experiments, and investigate renderers with explicit phoneme control. I chose to move the product to Azure Speech, separate Listener Lens from source-only Sound Minus Meaning, keep unsupported results fail-closed, and use GPT-5.6 at runtime only where it provides a clear classroom benefit.

That optional runtime feature uses a GPT-5.6 deployment named `luna` to turn bounded result metadata into a short teaching activity. It is useful but not presented as a requirement for the core audio experience. The product thesis, listening judgments, claim boundaries, renderer choice, visual direction, and final prioritization remained human decisions.

## Privacy and claim boundaries

- This is a research-informed approximation, not access to another person's private perception.
- A rule appears in B only when it matches the sentence and survives the current renderer checks.
- Typed sentences are sent to Azure Speech for audio rendering.
- Typed sentences and audio are not sent to GPT-5.6.
- The optional classroom generator receives bounded result metadata only.
- Service credentials stay server-side.
- Unsupported or unverifiable results fail closed rather than returning misleading audio.

## Licensing and provenance

Project source is licensed under the [MIT License](LICENSE). Third-party software, voices, services, models, and research sources retain their own terms.

- [DESIGN.md](DESIGN.md) records product and claim decisions.
