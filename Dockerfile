# syntax=docker/dockerfile:1.7

ARG PYTHON_VERSION=3.11

# ---------------------------------------------------------------------------
# Dependencies: resolve from the lockfile, cache the layer, skip the project
# itself so source changes do not invalidate the dependency install.
# ---------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS deps

COPY --from=ghcr.io/astral-sh/uv:0.13.7 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

# ---------------------------------------------------------------------------
# Runtime: slim, non-root, read-only friendly, venv copied from the deps stage.
# ---------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS runtime

LABEL org.opencontainers.image.title="ai-news-aggregator" \
      org.opencontainers.image.version="0.4.0" \
      org.opencontainers.image.description="Async AI news and paper aggregation"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/tmp \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

RUN groupadd --system app \
    && useradd --system --gid app --home-dir /app --shell /bin/false app

COPY --from=deps --chown=app:app /app/.venv /app/.venv
COPY --chown=app:app Articles ./Articles

USER app

ENTRYPOINT ["python", "-m", "Articles"]
CMD ["--limit", "20"]

# ---------------------------------------------------------------------------
# Test: dev extras + tests, fully offline (sockets blocked by the runner).
# ---------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS test

COPY --from=ghcr.io/astral-sh/uv:0.13.7 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --all-extras --no-install-project

COPY Articles ./Articles
COPY tests ./tests

CMD ["python", "-m", "tests.offline"]
