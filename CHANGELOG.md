# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[0.2.0]: https://github.com/yousefrihani/AI-News-Aggregator/releases/tag/v0.2.0
