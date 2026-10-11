"""Unit tests for the FreeNewsApi-based NewsAPI implementation in Articles.NewsApi."""

import os
import sys
import unittest
from unittest.mock import patch

# Ensure project paths are available for import
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ARTICLES_DIR = os.path.join(PROJECT_ROOT, "Articles")
for path in (PROJECT_ROOT, ARTICLES_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from Articles.BaseProvider import BaseProvider
from Articles.NewsApi import DEFAULT_BASE_URL, NewsAPI
from Articles.Vars import Category


class TestNewsAPI(unittest.TestCase):
    """Unit tests for the FreeNewsApi.io NewsAPI provider."""

    @patch.object(NewsAPI, "_getsource")
    def setUp(self, mock_getsource):
        mock_getsource.return_value = {"data": [{"topic": "technology"}]}
        self.api_key = "test_freenews_api_key"
        self.news_api = NewsAPI(apiKey=self.api_key)

    def test_inherits_from_base_provider(self):
        """Verify NewsAPI is an instance of BaseProvider."""
        self.assertIsInstance(self.news_api, BaseProvider)

    def test_init_sets_api_key_header_and_sources(self):
        """Verify API key is placed into session headers and _sources is populated."""
        self.assertEqual(self.news_api.session.headers.get("x-api-key"), self.api_key)
        self.assertEqual(self.news_api.base_url, DEFAULT_BASE_URL)
        self.assertEqual(self.news_api._sources, {"data": [{"topic": "technology"}]})

    @patch.object(NewsAPI, "_getsource", return_value={"data": []})
    @patch.dict("os.environ", {"FREENEWSAPI_API_KEY": "env_freenews_key"}, clear=True)
    def test_init_resolves_key_from_freenewsapi_env_var(self, mock_getsource):
        """Verify NewsAPI resolves API key from FREENEWSAPI_API_KEY env var when apiKey is None."""
        api = NewsAPI()
        self.assertEqual(api.api_key, "env_freenews_key")
        self.assertEqual(api.session.headers.get("x-api-key"), "env_freenews_key")
        api.close()

    @patch.object(NewsAPI, "_getsource", return_value={"data": []})
    @patch.dict("os.environ", {"NEWS_API_KEY": "env_general_news_key"}, clear=True)
    def test_init_resolves_key_from_news_api_key_env_var(self, mock_getsource):
        """Verify NewsAPI resolves API key from NEWS_API_KEY fallback env var."""
        api = NewsAPI()
        self.assertEqual(api.api_key, "env_general_news_key")
        self.assertEqual(api.session.headers.get("x-api-key"), "env_general_news_key")
        api.close()

    @patch.object(NewsAPI, "_getsource", return_value={"data": []})
    @patch.dict("os.environ", {"FREENEWSAPI_API_KEY": "env_key"}, clear=True)
    def test_init_explicit_key_overrides_env_var(self, mock_getsource):
        """Verify explicit apiKey argument overrides environment variable."""
        api = NewsAPI(apiKey="explicit_override_key")
        self.assertEqual(api.api_key, "explicit_override_key")
        self.assertEqual(api.session.headers.get("x-api-key"), "explicit_override_key")
        api.close()

    @patch.object(NewsAPI, "_make_request")
    def test_get_topics(self, mock_make_request):
        """Verify get_topics calls /topics endpoint."""
        mock_make_request.return_value = {"data": ["technology", "business"]}
        result = self.news_api.get_topics()
        mock_make_request.assert_called_once_with("/topics")
        self.assertEqual(result, {"data": ["technology", "business"]})

    @patch.object(NewsAPI, "_make_request")
    def test_get_publishers(self, mock_make_request):
        """Verify get_publishers calls /publishers endpoint."""
        mock_make_request.return_value = {"data": [{"uuid": "pub-1", "name": "Reuters"}]}
        result = self.news_api.get_publishers()
        mock_make_request.assert_called_once_with("/publishers")
        self.assertEqual(result, {"data": [{"uuid": "pub-1", "name": "Reuters"}]})

    @patch.object(NewsAPI, "_make_request")
    def test_get_news_with_filters(self, mock_make_request):
        """Verify get_news passes filters to /news endpoint."""
        mock_make_request.return_value = {"data": [{"title": "AI Breakthrough"}]}
        result = self.news_api.get_news(
            in_title="autonomous",
            topic="technology",
            language="en",
            country="us",
        )
        mock_make_request.assert_called_once_with(
            "/news",
            params={
                "in_title": "autonomous",
                "topic": "technology",
                "publisher_uuid": None,
                "author_uuid": None,
                "language": "en",
                "country": "us",
                "published_after": None,
                "published_before": None,
                "offset": None,
                "cursor": None,
            },
        )
        self.assertEqual(result, {"data": [{"title": "AI Breakthrough"}]})

    @patch.object(NewsAPI, "_make_request")
    def test_get_article_details(self, mock_make_request):
        """Verify get_article_details queries /details with article uuid."""
        mock_make_request.return_value = {"data": {"uuid": "123-uuid", "content": "Full article..."}}
        result = self.news_api.get_article_details(uuid="123-uuid")
        mock_make_request.assert_called_once_with("/details", params={"uuid": "123-uuid"})
        self.assertEqual(result, {"data": {"uuid": "123-uuid", "content": "Full article..."}})

    @patch.object(NewsAPI, "get_news")
    def test_get_news_by_keyword(self, mock_get_news):
        """Verify get_news_by_keyword delegates to get_news with in_title."""
        mock_get_news.return_value = {"data": [{"title": "Agentic AI"}]}
        result = self.news_api.get_news_by_keyword(keyword="agentic", topic="technology")
        mock_get_news.assert_called_once_with(in_title="agentic", topic="technology")
        self.assertEqual(result, {"data": [{"title": "Agentic AI"}]})

    @patch.object(NewsAPI, "get_news")
    def test_get_news_by_topic_default_category(self, mock_get_news):
        """Verify get_news_by_topic defaults to configured Category."""
        mock_get_news.return_value = {"data": []}
        result = self.news_api.get_news_by_topic()
        mock_get_news.assert_called_once_with(topic=Category, in_title=None)
        self.assertEqual(result, {"data": []})

    @patch.object(NewsAPI, "get_news")
    def test_get_news_by_publisher(self, mock_get_news):
        """Verify get_news_by_publisher sets publisher_uuid."""
        mock_get_news.return_value = {"data": []}
        result = self.news_api.get_news_by_publisher(publisher_uuid="pub-99")
        mock_get_news.assert_called_once_with(publisher_uuid="pub-99", in_title=None)
        self.assertEqual(result, {"data": []})

    @patch.object(NewsAPI, "get_news")
    def test_fetch_recent_and_search_base_contract(self, mock_get_news):
        """Verify BaseProvider abstract contracts fetch_recent and search."""
        mock_get_news.return_value = {"data": [{"title": "Recent news"}]}

        res_recent = self.news_api.fetch_recent(limit=5)
        mock_get_news.assert_called_with(topic=Category)
        self.assertEqual(res_recent, {"data": [{"title": "Recent news"}]})

        res_search = self.news_api.search(query="robotics")
        mock_get_news.assert_called_with(in_title="robotics")
        self.assertEqual(res_search, {"data": [{"title": "Recent news"}]})

    @patch.object(NewsAPI, "get_news_by_keyword")
    @patch.object(NewsAPI, "get_news_by_topic")
    def test_legacy_news_query_by_keyword(self, mock_by_topic, mock_by_kw):
        """Verify legacy _NewsQueryByKeyword returns list with search results."""
        mock_by_kw.return_value = {"data": [{"title": "K1"}]}
        mock_by_topic.return_value = {"data": [{"title": "K2"}]}

        result = self.news_api._NewsQueryByKeyword(topic="deep learning")
        self.assertEqual(result, [{"data": [{"title": "K1"}]}, {"data": [{"title": "K2"}]}])

    @patch.object(NewsAPI, "get_news_by_topic")
    @patch.object(NewsAPI, "get_news")
    def test_legacy_news_query_by_category(self, mock_get_news, mock_by_topic):
        """Verify legacy _NewsQueryByCategory returns category and keyword matches."""
        mock_by_topic.return_value = {"data": [{"title": "C1"}]}
        mock_get_news.return_value = {"data": [{"title": "C2"}]}

        result = self.news_api._NewsQueryByCategory(topic="chips", category="science")
        mock_by_topic.assert_called_once_with(topic="science", in_title="chips")
        mock_get_news.assert_called_once_with(in_title="chips")
        self.assertEqual(result, [{"data": [{"title": "C1"}]}, {"data": [{"title": "C2"}]}])

    @patch.object(NewsAPI, "get_news_by_publisher")
    @patch.object(NewsAPI, "get_news")
    def test_legacy_news_query_by_source(self, mock_get_news, mock_by_pub):
        """Verify legacy _NewsQueryBySource filters by publisher and keyword."""
        mock_by_pub.return_value = {"data": [{"title": "S1"}]}
        mock_get_news.return_value = {"data": [{"title": "S2"}]}

        result = self.news_api._NewsQueryBySource(source="bbc-uuid", topic="ai")
        mock_by_pub.assert_called_once_with(publisher_uuid="bbc-uuid", in_title="ai")
        self.assertEqual(result, [{"data": [{"title": "S1"}]}, {"data": [{"title": "S2"}]}])

    @patch.object(NewsAPI, "get_news")
    def test_legacy_news_query_everything_and_headlines(self, mock_get_news):
        """Verify legacy _NewsQueryEverything and _NewsQueryHeadlines."""
        mock_get_news.return_value = {"data": []}

        res_ev = self.news_api._NewsQueryEverything(source="pub-1", topic="test")
        self.assertEqual(res_ev, [{"data": []}, {"data": []}])

        res_hl = self.news_api._NewsQueryHeadlines(source="pub-1", topic="test")
        self.assertEqual(res_hl, [{"data": []}, {"data": []}])

    @patch.object(NewsAPI, "get_news_by_keyword")
    @patch.object(NewsAPI, "get_news_by_topic")
    def test_legacy_handles_none_failures(self, mock_by_topic, mock_by_kw):
        """Verify legacy queries return None when all requests fail."""
        mock_by_kw.return_value = None
        mock_by_topic.return_value = None

        result = self.news_api._NewsQueryByKeyword(topic="error")
        self.assertIsNone(result)

    def tearDown(self):
        self.news_api.close()


if __name__ == "__main__":
    unittest.main()
