"""Base provider abstraction for data sources in AI-News-Aggregator.

Defines the contract and common HTTP utilities for all ingestion providers
(e.g., News providers, Paper providers).
"""

import os
from abc import ABC, abstractmethod
from types import TracebackType
from typing import Any, Self

import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


class BaseProvider(ABC):
    """Abstract base class for all news and research paper ingestion providers."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "",
        timeout: int = 10,
        session: requests.Session | None = None,
    ) -> None:
        """Initialize the base provider.

        Args:
            api_key: Optional API key for authenticating with the external service.
                     If None, falls back to the API_KEY environment variable.
            base_url: The root URL for the provider's API.
            timeout: Default timeout in seconds for HTTP requests.
            session: Optional pre-configured requests.Session instance.
        """
        self.api_key = api_key or os.getenv("API_KEY")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = session or requests.Session()

    @abstractmethod
    def fetch_recent(self, limit: int = 10, **kwargs: Any) -> Any:
        """Fetch the most recent items from the provider.

        Args:
            limit: Maximum number of items to retrieve.
            **kwargs: Provider-specific filtering parameters.

        Returns:
            Provider response containing recent items.
        """

    @abstractmethod
    def search(self, query: str, limit: int = 10, **kwargs: Any) -> Any:
        """Search the provider for items matching a specific query string.

        Args:
            query: The search term or topic.
            limit: Maximum number of items to retrieve.
            **kwargs: Provider-specific filtering parameters.

        Returns:
            Provider response containing matching items.
        """

    def _make_request(
        self,
        endpoint: str,
        method: str = "GET",
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        json_data: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Execute an HTTP request with standardized error handling.

        Args:
            endpoint: API endpoint path (e.g. '/v1/news').
            method: HTTP method (default: 'GET').
            params: URL query parameters.
            headers: Additional or override HTTP headers.
            json_data: JSON payload for POST/PUT requests.

        Returns:
            Parsed JSON dictionary on success, or None on failure.
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}" if endpoint else self.base_url

        # Filter out None values from params
        clean_params = {k: v for k, v in (params or {}).items() if v is not None}

        try:
            response = self.session.request(
                method=method,
                url=url,
                params=clean_params,
                headers=headers,
                json=json_data,
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as err:
            print(f"[{self.__class__.__name__}] Request failed for {url}: {err}")
            return None

    def close(self) -> None:
        """Close the underlying HTTP session."""
        if self.session:
            self.session.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()
