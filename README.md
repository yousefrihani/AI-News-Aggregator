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

## Graph RAG

`Articles.graph` exposes a `GraphStore` port with two adapters:

- `InMemoryGraphStore` — dependency-free, deterministic, and used by the
  default offline tests.
- `Neo4jGraphStore` — the production adapter; the driver is imported lazily.

```python
import asyncio

from Articles.graph import Chunk, InMemoryGraphStore, DeterministicEmbedding


async def main() -> None:
    store = InMemoryGraphStore()
    embedder = DeterministicEmbedding(dim=384)
    vector = (await embedder.embed(["A chunk about retrieval"]))[0]
    chunk = Chunk(
        id="chunk-1",
        article_id="article-1",
        ordinal=0,
        text="A chunk about retrieval",
        embedding=tuple(vector),
    )
    await store.upsert_articles([], [chunk])
    results = await store.vector_search(vector, k=3, filters={"article_id": "article-1"})
    print(results[0][0].id, results[0][1])
    await store.aclose()


asyncio.run(main())
```

For Neo4j, set `GRAPH_BACKEND=neo4j`, `NEO4J_URI`, `NEO4J_USER`, and
`NEO4J_PASSWORD`, then apply the idempotent schema or let
`Neo4jGraphStore.initialize()` do it on startup:

```bash
cypher-shell -u neo4j -p "$NEO4J_PASSWORD" -f config/graph_schema.cypher
```

Local embeddings default to `BAAI/bge-small-en-v1.5` at 384 dimensions through
`fastembed`. `EMBEDDING_DIM` must match the vector index; changing it requires
recreating the index. `EMBEDDING_PROVIDER=deterministic` keeps tests and
offline development dependency-free.

## Enrichment pipeline

`Articles.pipeline.EnrichmentPipeline` turns articles into graph-ready records:
it chunks the text, summarizes and classifies it through LLM routes, extracts
entities/relations with the `graph_extraction.v1` schema, embeds chunks locally,
and writes everything in bounded batches.

```python
import asyncio

from Articles.graph import DeterministicEmbedding, InMemoryGraphStore
from Articles.llm import LLMRouter, load_llm_config
from Articles.pipeline import EnrichmentPipeline


async def main(articles) -> None:
    router = LLMRouter(load_llm_config("config/llm_routes.yaml"))
    pipeline = EnrichmentPipeline(
        router,
        InMemoryGraphStore(),
        DeterministicEmbedding(dim=384),
    )
    report = await pipeline.process(articles)
    print(report.succeeded_count, report.skipped_count, report.failed_count)


asyncio.run(main(articles))
```

Re-running over unchanged articles is a no-op: the store keeps each article's
normalized content hash, and only new or changed articles call a provider.
Failures are isolated per article and reported by stage (`summarize`,
`classify`, `extract`, `embed`, or `store`) without aborting the run.
`chunk_tokens`, `chunk_overlap`, `embedding_batch_size`, `write_batch_size`, and
`max_concurrency` are constructor settings; chunk writes stay below the
embedding batch limit of 64 by default.

## Graph RAG retrieval

`GraphRAGRetriever` embeds a question, fetches the top `k` chunks from the
vector index, expands one or two hops through mentions and relations, and
renders a bounded context block. Chunks are grouped under their article
identifier, so answers cite stable `[article_id]` markers rather than opaque
chunk UUIDs.

```python
import asyncio

from Articles.graph import DeterministicEmbedding, GraphRAGRetriever, InMemoryGraphStore


async def main(store: InMemoryGraphStore) -> None:
    retriever = GraphRAGRetriever(store, DeterministicEmbedding(dim=384))
    context = await retriever.retrieve("What did OpenAI announce?")
    print(context.context_text)
    print(context.article_ids)


asyncio.run(main(store))
```

`GraphRAGAnswerPattern` turns that context into a cited answer through the
`graph_rag` route:

```python
from Articles.llm.patterns import GraphRAGAnswerPattern

answer = await GraphRAGAnswerPattern(router).run(question, context)
print(answer.answer)
print(answer.article_ids, answer.unknown_citations)
```

The pattern forbids outside knowledge in both the system and user prompts.
Citations that do not match retrieved articles are removed from `citations`
and surfaced in `unknown_citations`, so callers can log or reject ungrounded
answers. Defaults are `k=8`, `depth=1`, `limit=20`, and
`max_context_chars=12000`; `depth` must be 1 or 2.

The evaluation harness lives in `tests/eval/graph_rag/`. It uses a fixed
21-question corpus with recorded answers and deterministic bag-of-words
embeddings, so it needs no API key and downloads no model. It gates recall@8
at 0.8 and citation precision at 0.9:

```bash
RUN_EVAL=1 python -m unittest tests.eval.graph_rag.test_eval
```

The metric helpers themselves are covered by the default offline suite.

## Docker

Build and run the runtime image:

```bash
docker build -t ai-news-aggregator:0.6.1 .
docker run --rm --env-file .env ai-news-aggregator:0.6.1 --limit 10 --json
```

Or use Compose (it reads `.env` for interpolation; no secrets are baked in):

```bash
docker compose build
docker compose run --rm app --limit 10 --json
# containerised test suite (sockets blocked, no API keys)
docker compose --profile test run --rm test
```

Graph images and profiles are additive:

```bash
# Neo4j on 127.0.0.1 only; requires NEO4J_PASSWORD with no default
docker compose --profile graph up -d neo4j
docker compose --profile graph build app-graph
docker compose --profile graph run --rm app-graph --limit 10 --json

# live Neo4j integration tests
docker compose --profile test-graph run --rm test-graph

# reset graph data (named volumes)
docker compose --profile graph down -v
```

`runtime-graph` adds the Neo4j driver, ONNX Runtime, and the embedding model
runtime, so it is intentionally much larger than `runtime`. Copy
`config/llm_routes.example.yaml` to `config/llm_routes.yaml` before building
`app-graph`; the Compose service reads that path. The embedding model cache is
kept in the `fastembed_cache` volume.

The runtime images are multi-stage, run as a non-root user, and are compatible
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
| `GRAPH_BACKEND` | `memory` | `memory` or `neo4j`. |
| `NEO4J_URI` | `bolt://localhost:7687` | Graph connection URI. |
| `NEO4J_USER` | `neo4j` | Graph user. |
| `NEO4J_PASSWORD` | — | Required when `GRAPH_BACKEND=neo4j`. |
| `NEO4J_DATABASE` | `neo4j` | Target database. |
| `EMBEDDING_PROVIDER` | `fastembed` | `fastembed` or `deterministic`. |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Local embedding model. |
| `EMBEDDING_DIM` | `384` | Must match the vector index. |
| `RUN_GRAPH_TESTS` | `0` | Enable live Neo4j integration tests. |
| `RUN_EVAL` | `0` | Enable the offline Graph RAG evaluation harness. |
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
- [x] Graph store (in-memory + Neo4j), local embeddings, and Docker graph profiles.
- [x] Enrichment pipeline with idempotent re-runs and per-article failure reports.
- [x] Graph RAG retrieval, cited answers, and the offline evaluation harness.
- [ ] Agent seam for the custom workflow specification.
- [ ] Paper provider (Semantic Scholar / arXiv).
- [ ] Scheduling, persistence, and delivery adapters.
