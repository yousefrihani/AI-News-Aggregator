# AI-News-Aggregator

A small, typed Python library for pulling AI-related news (and, later, papers)
from multiple providers behind a single async interface. Providers fetch and
normalize articles; an aggregator fans out across providers, de-duplicates,
sorts, and isolates partial failures.

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

Key design rules: HTTP lives only in providers, constructors never perform
network I/O, and library code is async-first (the CLI is the only sync boundary).

## Stack

| Layer | Choice |
|---|---|
| Language | Python 3.11+ |
| HTTP | `httpx` (async `AsyncClient`; sync `Client` for legacy aliases) |
| Models | stdlib `dataclasses` |
| Tests | stdlib `unittest` + `unittest.mock`, `IsolatedAsyncioTestCase` |
| Packaging | `pyproject.toml` + `uv.lock` (with generated `requirements.txt`) |

## Install

With [uv](https://docs.astral.sh/uv/) (recommended):

```bash
uv sync
```

This creates `.venv/` and installs the locked runtime and development
dependencies. If you prefer plain `pip`:

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

## Docker

Build and run the runtime image:

```bash
docker build -t ai-news-aggregator:0.3.0 .
docker run --rm --env-file .env ai-news-aggregator:0.3.0 --limit 10 --json
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
- [ ] Paper provider (Semantic Scholar / arXiv).
- [ ] Scheduling, persistence, and delivery adapters.
