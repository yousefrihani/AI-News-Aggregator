# AI-News-Aggregator

A small, typed Python library for pulling AI-related news (and, later, papers)
from multiple providers behind a single async interface. Providers fetch and
normalize articles; an aggregator fans out across providers, de-duplicates,
sorts, and isolates partial failures. The continuation layers add provider
routing for LLM calls and a graph store for Graph RAG.

## Architecture

The data flow is one direction, with no import cycles:

```
Provider.fetch_articles() -> list[Article] -> Aggregator -> dedupe -> sort -> list[Article] -> consumer/CLI
```

- **Provider abstraction** (`Articles.BaseProvider`): async ABC exposing
  `fetch_recent`, `search`, and `fetch_articles`. Providers own HTTP details,
  pagination, retries, and normalization, and never import the aggregator.
- **Concrete providers** (`Articles.NewsApi`, later `Articles.PaperApi`):
  thin adapters over a single upstream API.
- **Normalized models** (`Articles.Models`): immutable `Article` / `Paper`
  dataclasses so downstream code never depends on provider payload shapes.
- **Aggregator**: receives providers via injection, runs them concurrently,
  de-duplicates by canonical URL, and returns a sorted result. One failing
  provider is reported but does not abort the whole run.
- **Errors** (`Articles.errors`): typed provider errors for config, auth, HTTP,
  and transport failures.
- **Settings** (`Articles.config`, with `Articles.Vars` kept for compatibility):
  environment-driven configuration for keys, timeouts, and concurrency.
- **LLM layer** (`Articles.llm`): route-based completions. Patterns and agents
  name a route such as `summarize`; the router resolves it to ordered
  provider/model candidates and never exposes vendor details upstream.
- **Graph layer** (`Articles.graph`): a `GraphStore` port with in-memory and
  Neo4j adapters, local `fastembed` embeddings, and a versioned schema.

Key design rules: HTTP lives only in providers, constructors never perform
network I/O, and library code is async-first (the CLI is the only sync boundary).

## Stack

| Layer | Choice |
|---|---|
| Language | Python 3.11+ |
| HTTP | `httpx` (async `AsyncClient`; sync `Client` for legacy aliases) |
| LLM routing | OpenAI-compatible adapters for Groq and OpenRouter |
| Graph | Neo4j 5 Community (optional) or the in-memory store |
| Embeddings | `fastembed` + `BAAI/bge-small-en-v1.5` (384-d) |
| Models | stdlib `dataclasses` |
| Tests | stdlib `unittest` + `unittest.mock`, `IsolatedAsyncioTestCase` |
| Packaging | `pyproject.toml` + `uv.lock` (with generated `requirements.txt`) |

## Install

With [uv](https://docs.astral.sh/uv/) (recommended):

```bash
uv sync
```

This creates `.venv/` and installs the locked runtime and development
dependencies. Graph support is optional:

```bash
uv sync --extra graph --extra embeddings-local
```

If you prefer plain `pip`:

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows (PowerShell)
# .venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in real keys; `.env` is gitignored.

> **Note:** `.venv/` is platform-specific — it contains absolute paths and
> native binaries for the OS that created it. It is gitignored; recreate it on
> each machine rather than copying it between Linux, macOS, and Windows.

## Run tests

```bash
python -m unittest discover -s tests
```

Unit tests are mocked and need no API keys; the offline network guard keeps
real sockets out of the suite. Live end-to-end tests are opt-in via `RUN_E2E=1`.

## Command line

```bash
# after `uv sync`
uv run python -m Articles --limit 10
uv run python -m Articles --limit 10 --json
uv run python -m Articles --limit 5 --query "agentic AI" --json
```

The CLI exits `2` on missing credentials or a provider error, and `0` on a
successful run (including an empty result).

## LLM routing

Copy the example route file and set the keys you intend to use:

```bash
cp config/llm_routes.example.yaml config/llm_routes.yaml
export GROQ_API_KEY=...
export OPENROUTER_API_KEY=...
```

Call a logical route, never a vendor:

```python
import asyncio

from Articles.llm import LLMRouter, Message, load_llm_config


async def main() -> None:
    router = LLMRouter(load_llm_config("config/llm_routes.yaml"))
    response = await router.complete(
        "summarize",
        [Message("user", "Summarize this article in two sentences: ...")],
    )
    print(response.text, response.provider, response.usage.total_tokens)
    await router.aclose()


asyncio.run(main())
```

The router skips providers without credentials, retries retryable failures,
opens a per-provider circuit breaker after repeated failures, and falls back to
the next route candidate. Selection precedence is:

1. `provider=`/`model=` arguments to `complete()`;
2. `AI_NEWS_LLM_PROVIDER`/`AI_NEWS_LLM_MODEL`;
3. the route's ordered candidates;
4. `AI_NEWS_LLM_ROUTE` as the default route name (`summarize`).

A pin fails loudly with `LLMUnavailableError` unless fallback is enabled with
`allow_fallback=True` or `AI_NEWS_LLM_ALLOW_FALLBACK=1`. `LLM_DRY_RUN=1`
returns deterministic fixtures without touching a provider.

Adding a provider requires a YAML entry plus either the generic
`openai_compatible` adapter or an entry point in the
`ai_news_aggregator.llm_providers` group; the router and call sites stay
unchanged.

## Docker

Build and run the runtime image:

```bash
docker build -t ai-news-aggregator:0.4.0 .
docker run --rm --env-file .env ai-news-aggregator:0.4.0 --limit 10 --json
```

Or use Compose (it reads `.env` for interpolation; no secrets are baked in):

```bash
docker compose build
docker compose run --rm app --limit 10 --json
# containerised test suite (sockets blocked, no API keys)
docker compose --profile test run --rm test
```

The runtime image is multi-stage, runs as a non-root user, and is compatible
with a read-only root filesystem (`/tmp` is mounted as tmpfs by Compose).

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `FREENEWSAPI_API_KEY` | — | Primary FreeNewsApi.io key (sent as `x-api-key`). |
| `NEWS_API_KEY` | — | Generic fallback key, used if the primary key is unset. |
| `NEWSAPI_KEY` | — | Second fallback key, checked last. |
| `SEMANTIC_SCHOLAR_API_KEY` | — | Reserved for future paper ingestion; currently unused. |
| `GROQ_API_KEY` | — | Groq adapter key. |
| `OPENROUTER_API_KEY` | — | OpenRouter adapter key. |
| `AI_NEWS_LLM_ROUTES_FILE` | `config/llm_routes.yaml` | Provider/route YAML path. |
| `AI_NEWS_LLM_ROUTE` | `summarize` | Default route when none is passed. |
| `AI_NEWS_LLM_PROVIDER` | — | Pin a provider, overriding route order. |
| `AI_NEWS_LLM_MODEL` | — | Pin a model alias or id. |
| `AI_NEWS_LLM_ALLOW_FALLBACK` | `0` | Allow fallback when a provider is pinned. |
| `LLM_DRY_RUN` | `0` | Fixture responses; never call a provider. |
| `RUN_E2E` | `0` | Set to `1` to enable live end-to-end tests. |
| `AI_NEWS_HTTP_TIMEOUT` | `10.0` | Read/write timeout in seconds. |
| `AI_NEWS_MAX_CONCURRENCY` | `5` | Per-provider concurrent-request limit. |
| `AI_NEWS_MAX_PAGES` | `10` | Upper bound on pagination loops. |
| `AI_NEWS_MAX_ITEMS` | `500` | Upper bound on items fetched per provider call. |
| `AI_NEWS_PAGE_SIZE` | `50` | Items requested per pagination page. |
| `AI_NEWS_TOPICS` | `technology` | Comma-separated default topics. |
| `ARTICLE_LIMIT` | `20` | Compose-only: articles requested by the `app` service. |

API keys are never logged and must never be committed. `.env.example` contains
placeholders only.

## Supported providers

| Provider | Status | Notes |
|---|---|---|
| [FreeNewsApi.io](https://www.freenewsapi.io) | Supported | News search, topics, publishers, and article details. |
| Semantic Scholar / arXiv | Planned | Paper ingestion (`PaperApi`), deferred to a later release. |

## Roadmap

- [x] Provider abstraction and `unittest` baseline.
- [x] Async-first HTTP (`httpx`) with timeouts, connection pooling, retries, and bounded pagination.
- [x] Normalized `Article` / `Paper` models and typed provider errors.
- [x] `Aggregator` with concurrent fan-out, de-duplication, sorting, and partial-failure isolation.
- [x] Offline network guard, integration fixtures, and CI (lint, type-check, tests).
- [x] CLI entry point as the sync boundary.
- [x] Docker image and Compose services.
- [x] LLM routing with Groq/OpenRouter, fallback, pins, and offline tests.
- [ ] Graph RAG (Neo4j + local embeddings) and Docker graph profiles.
- [ ] Enrichment pipeline, retrieval/answer patterns, and agent seam.
- [ ] Paper provider (Semantic Scholar / arXiv).
- [ ] Scheduling, persistence, and delivery adapters.
