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

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `FREENEWSAPI_API_KEY` | — | Primary FreeNewsApi.io key (sent as `x-api-key`). |
| `NEWS_API_KEY` | — | Generic fallback key, used if the primary key is unset. |
| `NEWSAPI_KEY` | — | Second fallback key, checked last. |
| `SEMANTIC_SCHOLAR_API_KEY` | — | Reserved for future paper ingestion; currently unused. |
| `RUN_E2E` | `0` | Set to `1` to enable live end-to-end tests. |
| `HTTP_CONNECT_TIMEOUT` | `3.0` | Connect timeout in seconds. |
| `HTTP_READ_TIMEOUT` | `10.0` | Read timeout in seconds. |
| `HTTP_WRITE_TIMEOUT` | `10.0` | Write timeout in seconds. |
| `HTTP_POOL_TIMEOUT` | `3.0` | Connection-pool timeout in seconds. |
| `HTTP_MAX_CONNECTIONS` | `20` | Maximum connections per provider client. |
| `HTTP_MAX_KEEPALIVE_CONNECTIONS` | `10` | Maximum idle keep-alive connections. |
| `HTTP_MAX_CONCURRENCY` | `5` | Per-provider concurrent-request limit. |

API keys are never logged and must never be committed. `.env.example` contains
placeholders only.

## Supported providers

| Provider | Status | Notes |
|---|---|---|
| [FreeNewsApi.io](https://www.freenewsapi.io) | Supported | News search, topics, publishers, and article details. |
| Semantic Scholar / arXiv | Planned | Paper ingestion (`PaperApi`), deferred to a later release. |

## Roadmap

- [x] Provider abstraction and `unittest` baseline.
- [ ] Async-first HTTP (`httpx`) with timeouts, connection pooling, retries, and bounded pagination.
- [ ] Normalized `Article` / `Paper` models and typed provider errors.
- [ ] `Aggregator` with concurrent fan-out, de-duplication, sorting, and partial-failure isolation.
- [ ] Offline network guard, integration fixtures, and CI (lint, type-check, tests).
- [ ] Paper provider (Semantic Scholar / arXiv).
- [ ] Optional CLI entry point as the sync boundary.
