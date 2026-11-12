"""Opt-in live end-to-end test for the FreeNewsApi.io adapter.

The test is skipped unless ``RUN_E2E=1`` and a news API key is present in the
environment. It performs exactly one read-only request and never writes data.
"""

from __future__ import annotations

import inspect
import os
import unittest

from Articles.NewsApi import NewsAPI

RUN_E2E = os.getenv("RUN_E2E") == "1"
API_KEY = os.getenv("FREENEWSAPI_API_KEY") or os.getenv("NEWS_API_KEY") or os.getenv("NEWSAPI_KEY")


@unittest.skipUnless(
    RUN_E2E and API_KEY,
    "live e2e disabled: set RUN_E2E=1 and a FreeNewsApi key to enable",
)
class LiveNewsAPITest(unittest.IsolatedAsyncioTestCase):
    """Single read-only smoke test against the real service."""

    async def test_fetch_recent_read_only(self) -> None:
        """Fetch one recent article without mutating remote state."""
        provider = NewsAPI(apiKey=API_KEY)
        try:
            items = await provider.fetch_recent(limit=1)
        finally:
            closer = getattr(provider, "aclose", None)
            if closer is None:
                closer = getattr(provider, "close", None)
            if closer is not None:
                outcome = closer()
                if inspect.isawaitable(outcome):
                    await outcome

        self.assertIsInstance(items, list)
        self.assertLessEqual(len(items), 1)


if __name__ == "__main__":
    unittest.main()
