import unittest
from unittest.mock import MagicMock, patch

import requests

from Articles.BaseProvider import BaseProvider


class DummyProvider(BaseProvider):
    """Concrete implementation of BaseProvider for unit testing."""

    def fetch_recent(self, limit: int = 10, **kwargs):
        return self._make_request("/recent", params={"limit": limit, **kwargs})

    def search(self, query: str, limit: int = 10, **kwargs):
        return self._make_request("/search", params={"q": query, "limit": limit, **kwargs})


class TestBaseProvider(unittest.TestCase):
    """Unit tests for the BaseProvider abstract class."""

    def test_cannot_instantiate_abstract_class(self):
        """Verify that BaseProvider cannot be instantiated directly."""
        with self.assertRaises(TypeError):
            BaseProvider(api_key="test_key")

    def test_concrete_subclass_initialization(self):
        """Verify initialization of a concrete subclass with default and custom values."""
        provider = DummyProvider(
            api_key="abc123key",
            base_url="https://api.example.com/v1/",
            timeout=15,
        )
        self.assertEqual(provider.api_key, "abc123key")
        self.assertEqual(provider.base_url, "https://api.example.com/v1")
        self.assertEqual(provider.timeout, 15)
        self.assertIsInstance(provider.session, requests.Session)
        provider.close()

    @patch.dict("os.environ", {"API_KEY": "env_secret_key"}, clear=True)
    def test_initialization_falls_back_to_env_var(self):
        """Verify that BaseProvider falls back to API_KEY environment variable when api_key is None."""
        provider = DummyProvider(base_url="https://api.example.com")
        self.assertEqual(provider.api_key, "env_secret_key")
        provider.close()

    @patch.dict("os.environ", {"API_KEY": "env_secret_key"}, clear=True)
    def test_explicit_api_key_overrides_env_var(self):
        """Verify that explicitly passed api_key overrides the environment variable."""
        provider = DummyProvider(api_key="explicit_key", base_url="https://api.example.com")
        self.assertEqual(provider.api_key, "explicit_key")
        provider.close()

    @patch("requests.Session.request")
    def test_make_request_success(self, mock_request):
        """Verify successful request processing and JSON parsing."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"status": "ok", "items": [1, 2, 3]}
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response

        provider = DummyProvider(api_key="key", base_url="https://api.example.com")
        result = provider.search(query="machine learning", limit=5)

        self.assertEqual(result, {"status": "ok", "items": [1, 2, 3]})
        mock_request.assert_called_once_with(
            method="GET",
            url="https://api.example.com/search",
            params={"q": "machine learning", "limit": 5},
            headers=None,
            json=None,
            timeout=10,
        )
        provider.close()

    @patch("requests.Session.request")
    def test_make_request_filters_none_params(self, mock_request):
        """Verify that parameters with None values are filtered out."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"status": "ok"}
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response

        provider = DummyProvider(base_url="https://api.example.com")
        provider._make_request("/test", params={"valid": "yes", "empty": None})

        mock_request.assert_called_once_with(
            method="GET",
            url="https://api.example.com/test",
            params={"valid": "yes"},
            headers=None,
            json=None,
            timeout=10,
        )
        provider.close()

    @patch("requests.Session.request")
    def test_make_request_handles_request_exception(self, mock_request):
        """Verify graceful error handling and returning None on RequestException."""
        mock_request.side_effect = requests.exceptions.ConnectionError("Network down")

        provider = DummyProvider(base_url="https://api.example.com")
        result = provider._make_request("/error-endpoint")

        self.assertIsNone(result)
        provider.close()

    def test_context_manager_closes_session(self):
        """Verify context manager properly invokes close on session."""
        with DummyProvider(base_url="https://api.example.com") as provider:
            mock_session = MagicMock()
            provider.session = mock_session

        mock_session.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
