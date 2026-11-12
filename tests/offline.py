"""Run the full test suite with real network access disabled.

The runner monkeypatches the low-level socket entry points so that any attempt
to open a real connection fails loudly with ``AssertionError("unmocked network
call")``. Everything else (fixtures, ``httpx.MockTransport``) keeps working, so
the suite stays hermetic without suppressing legitimate in-process tests.

Usage::

    python -m tests.offline
"""

from __future__ import annotations

import os
import socket
import sys
import unittest
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]


def _unmocked_network_call(*_args: Any, **_kwargs: Any) -> None:
    """Raise instead of opening a real network connection."""
    raise AssertionError("unmocked network call")


def main() -> int:
    """Discover and run the test suite with sockets blocked.

    Returns:
        ``0`` when every test passed or was skipped, ``1`` otherwise.
    """
    # The offline suite never runs live end-to-end tests, even when the caller
    # forgot to unset RUN_E2E in their shell.
    os.environ.pop("RUN_E2E", None)

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    original_connect = socket.socket.connect
    original_create_connection = socket.create_connection

    socket.socket.connect = _unmocked_network_call  # type: ignore[method-assign]
    socket.create_connection = _unmocked_network_call  # type: ignore[assignment]
    try:
        suite = unittest.defaultTestLoader.discover(
            start_dir=str(REPO_ROOT / "tests"),
            top_level_dir=str(REPO_ROOT),
        )
        result = unittest.TextTestRunner(verbosity=2, stream=sys.stderr).run(suite)
    finally:
        socket.socket.connect = original_connect  # type: ignore[method-assign]
        socket.create_connection = original_create_connection  # type: ignore[assignment]

    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
