"""Offline network guard for tests.

Usage::

    from tests.network_guard import blocked_network

    class TestSomething(unittest.TestCase):
        def test_offline(self) -> None:
            with blocked_network():
                ...

While the context manager is active, any HTTP call made through
``requests.Session.request``, ``httpx.Client.request`` or
``httpx.AsyncClient.request`` raises ``AssertionError("unmocked network call")``.
This keeps the unit suite hermetic and makes accidental live calls obvious.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from typing import Any
from unittest import mock


def _raise_unmocked(*_args: Any, **_kwargs: Any) -> None:
    """Fail immediately when an HTTP client is asked to perform a request."""
    raise AssertionError("unmocked network call")


@contextlib.contextmanager
def blocked_network() -> Iterator[None]:
    """Block real HTTP calls for the duration of the context.

    Missing optional HTTP libraries are ignored, so the guard is safe to use
    in environments without ``requests`` or ``httpx`` installed.

    Yields:
        None.

    Raises:
        AssertionError: If code inside the context performs an HTTP request.
    """
    patchers: list[Any] = []

    try:
        from requests import Session as RequestsSession
    except ImportError:
        pass
    else:
        patchers.append(mock.patch.object(RequestsSession, "request", _raise_unmocked))

    try:
        import httpx
    except ImportError:
        pass
    else:
        patchers.append(mock.patch.object(httpx.Client, "request", _raise_unmocked))
        patchers.append(mock.patch.object(httpx.AsyncClient, "request", _raise_unmocked))

    for patcher in patchers:
        patcher.start()
    try:
        yield
    finally:
        for patcher in reversed(patchers):
            patcher.stop()
