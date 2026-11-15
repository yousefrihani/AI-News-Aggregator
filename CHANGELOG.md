# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.0] - 2026-11-15

### Added

- Graph layer (`Articles.graph`): immutable `Chunk`, `Entity`, `Relation`,
  `Mention`, and `GraphContext` models plus the `GraphStore` port.
- `InMemoryGraphStore` with deterministic upsert, filtered vector search, and
  bounded expansion; used by the default offline suite.
- `Neo4jGraphStore` with lazy driver import, batched writes, idempotent stale
  chunk cleanup, filtered vector search, and bounded expansion.
- Versioned, idempotent Neo4j schema (`config/graph_schema.cypher`) for
  constraints, fulltext indexes, and the 384-d cosine vector index.
- Local embeddings through `FastEmbedEmbedding` (`BAAI/bge-small-en-v1.5`)
  plus a deterministic, dependency-free test embedder.
- Docker `deps-graph`, `runtime-graph`, and `test-graph` targets, and Compose
  `graph`/`test-graph` profiles with a pinned Neo4j 5.26 Community service.
- Opt-in live Neo4j integration tests gated by `RUN_GRAPH_TESTS=1`.

### Changed

- The runtime image now ships the read-only `config/` directory (schema and
  route examples); the default image remains graph-free.
- Raised the package version to `0.5.0`; added `graph` and `embeddings-local`
  optional extras.

## [0.4.0] - 2026-11-15

### Added

- Provider-agnostic LLM layer (`Articles.llm`): `LLMProvider` port,
  OpenAI-compatible transport, and Groq/OpenRouter adapters.
- YAML route configuration (`config/llm_routes.example.yaml`) with ordered
  provider/model candidates, JSON response modes, retries, timeouts, input
  limits, and per-route token budgets.
- `LLMRouter` with health-aware candidate selection, typed error mapping,
  circuit breakers, deterministic caching, dry-run mode, and explicit
  provider/model pins (`AI_NEWS_LLM_PROVIDER`/`AI_NEWS_LLM_MODEL`).
- `ProviderRegistry` for built-in adapters and entry-point plugins, so future
  providers can be added without changing routing or call sites.
- Hermetic LLM suite using `httpx.MockTransport`, scripted providers, and an
  injected clock/sleep; no keys or real backoff in tests.

### Changed

- Raised the package version to `0.4.0`; added `pyyaml` and `types-pyyaml`.

## [0.3.0] - 2026-11-12

### Added

- `python -m Articles` CLI (`Articles.cli`) with `--limit`, `--topic`,
  `--query`, `--json`, and `--verbose`; exits `2` on missing credentials or a
  provider error.
- `Aggregator(..., raise_on_all_failed=True)` so batch callers can surface a
  total provider outage instead of silently returning an empty list.
- Multi-stage `Dockerfile` (`runtime` and `test` targets), `.dockerignore`, and
  `docker-compose.yml` with non-root, read-only runtime hardening.
- Containerised offline test service (`docker compose --profile test`).

### Changed

- Raised the package version to `0.3.0` and documented the Docker workflow.
- Corrected the environment-variable docs to the `AI_NEWS_*` names actually
  read by `Articles.config`.

## [0.2.0] - 2026-11-12

### Added

- Async `httpx`-based `BaseProvider` with retry/backoff, `Retry-After`
  support, semaphore-bounded concurrency, and a sync path for legacy aliases.
- Typed error hierarchy: `ProviderError`, `ProviderConfigError`,
  `ProviderAuthError`, `ProviderHTTPError`, `ProviderTransportError`.
- `FreeNewsApi` provider with lazy topic metadata, offset/cursor pagination,
  `max_pages`/`max_items` guards, UUID de-duplication, and `Article` mapping.
- `Aggregator` with concurrent fan-out, O(N) canonical-URL de-duplication,
  deterministic ordering, and per-provider failure isolation.
- Immutable `Article` / `Paper` models and validated `Settings.from_env()`.
- Hermetic unit and integration suites (`httpx.MockTransport`), an offline
  test runner that blocks sockets, and an opt-in live e2e smoke test.
- CI pipeline: ruff lint + format, mypy, offline tests, coverage gate (80%),
  and dependency audit.
- Packaging: `pyproject.toml`, `uv.lock`, generated `requirements.txt`.

### Changed

- Replaced `requests` with `httpx` as the single HTTP client.
- Removed eager network I/O from provider constructors; `_sources` is now lazy.
- Removed `sys.path` manipulation and import fallbacks; `Articles` is a package.
- Normalized the repository to LF line endings via `.gitattributes`.
- Rewrote the README with architecture, install, test, and environment docs.

### Fixed

- `fetch_recent` / `search` now honour `limit`.
- Unit tests no longer make live HTTP calls.
- Provider failures raise typed exceptions instead of printing and returning
  `None` on auth/config errors.

[0.5.0]: https://github.com/yousefrihani/AI-News-Aggregator/releases/tag/v0.5.0
[0.4.0]: https://github.com/yousefrihani/AI-News-Aggregator/releases/tag/v0.4.0
[0.3.0]: https://github.com/yousefrihani/AI-News-Aggregator/releases/tag/v0.3.0
[0.2.0]: https://github.com/yousefrihani/AI-News-Aggregator/releases/tag/v0.2.0
