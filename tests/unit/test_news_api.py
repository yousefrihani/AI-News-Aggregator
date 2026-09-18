import os
import sys
import unittest
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta

# Ensure project paths are available for import
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ARTICLES_DIR = os.path.join(PROJECT_ROOT, "Articles")
for path in (PROJECT_ROOT, ARTICLES_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from Articles.NewsApi import NewsAPI
from Articles.Vars import Category


class TestNewsAPI(unittest.TestCase):
    """Unit tests for the updated NewsAPI class in Articles.NewsApi."""

    @patch("Articles.NewsApi.NewsApiClient")
    def setUp(self, mock_newsapi_client_cls):
        self.api_key = "test_dummy_api_key"
        self.mock_client_instance = MagicMock()
        mock_newsapi_client_cls.return_value = self.mock_client_instance

        # Mock get_sources return value for initial automatic fetch
        self.default_sources = {"status": "ok", "sources": [{"id": "wired", "name": "Wired"}]}
        self.mock_client_instance.get_sources.return_value = self.default_sources

        self.news_api = NewsAPI(apiKey=self.api_key)
        self.mock_newsapi_client_cls = mock_newsapi_client_cls

    def test_init_calls_source_automatically(self):
        """Test NewsAPI initializes client and automatically populates _sources."""
        self.mock_newsapi_client_cls.assert_called_once_with(api_key=self.api_key)
        self.assertEqual(self.news_api.client, self.mock_client_instance)
        # Verify get_sources was called during __init__
        self.mock_client_instance.get_sources.assert_called_with(category=Category)
        self.assertEqual(self.news_api._sources, self.default_sources)

    def test_getsource_raises_exception_without_failsafe(self):
        """Test _getsource and _source raise exception directly when failsafe check is removed."""
        self.mock_client_instance.get_sources.side_effect = Exception("Sources API down")
        with self.assertRaises(Exception):
            self.news_api._getsource()
        with self.assertRaises(Exception):
            self.news_api._source()

    @patch("Articles.NewsApi.datetime")
    def test_news_query_by_keyword(self, mock_datetime):
        """Test _NewsQueryByKeyword calls _getsource, searches everything and top headlines, returning list."""
        fixed_now = datetime(2026, 9, 16, 12, 0, 0)
        mock_datetime.now.return_value = fixed_now
        expected_date = (fixed_now - timedelta(days=7)).strftime("%Y-%m-%d")

        expected_everything = {"status": "ok", "totalResults": 5, "articles": [{"title": "New AI Model"}]}
        expected_headlines = {"status": "ok", "articles": [{"title": "AI Headline"}]}
        self.mock_client_instance.get_everything.return_value = expected_everything
        self.mock_client_instance.get_top_headlines.return_value = expected_headlines

        result = self.news_api._NewsQueryByKeyword(topic="generative ai")

        self.mock_client_instance.get_sources.assert_called_with(category=Category)
        self.mock_client_instance.get_everything.assert_called_once_with(
            q="generative ai",
            from_param=expected_date,
            sort_by="publishedAt",
        )
        self.mock_client_instance.get_top_headlines.assert_called_once_with(
            q="generative ai",
        )
        self.assertEqual(result, [expected_everything, expected_headlines])

    def test_news_query_by_keyword_handles_exception(self):
        """Test _NewsQueryByKeyword returns None on failure."""
        self.mock_client_instance.get_everything.side_effect = Exception("Keyword search failure")
        result = self.news_api._NewsQueryByKeyword(topic="test")
        self.assertIsNone(result)

    def test_news_query_by_category_default(self):
        """Test _NewsQueryByCategory calls _getsource, queries with Category and only on topic, returns list."""
        expected_cat = {"status": "ok", "articles": [{"title": "Tech Headline"}]}
        expected_topic = {"status": "ok", "articles": [{"title": "Robot Headline"}]}
        self.mock_client_instance.get_top_headlines.side_effect = [expected_cat, expected_topic]

        result = self.news_api._NewsQueryByCategory(topic="robots")

        self.mock_client_instance.get_sources.assert_called_with(category=Category)
        self.assertEqual(self.mock_client_instance.get_top_headlines.call_count, 2)
        self.mock_client_instance.get_top_headlines.assert_any_call(
            q="robots",
            category=Category,
        )
        self.mock_client_instance.get_top_headlines.assert_any_call(
            q="robots",
        )
        self.assertEqual(result, [expected_cat, expected_topic])

    def test_news_query_by_category_custom(self):
        """Test _NewsQueryByCategory with custom category."""
        expected_cat = {"status": "ok", "articles": [{"title": "Science Headline"}]}
        expected_topic = {"status": "ok", "articles": [{"title": "Space Headline"}]}
        self.mock_client_instance.get_top_headlines.side_effect = [expected_cat, expected_topic]

        result = self.news_api._NewsQueryByCategory(topic="space", category="science")

        self.mock_client_instance.get_sources.assert_called_with(category=Category)
        self.assertEqual(self.mock_client_instance.get_top_headlines.call_count, 2)
        self.mock_client_instance.get_top_headlines.assert_any_call(
            q="space",
            category="science",
        )
        self.mock_client_instance.get_top_headlines.assert_any_call(
            q="space",
        )
        self.assertEqual(result, [expected_cat, expected_topic])

    def test_news_query_by_category_handles_exception(self):
        """Test _NewsQueryByCategory returns None on failure."""
        self.mock_client_instance.get_top_headlines.side_effect = Exception("Headlines error")
        result = self.news_api._NewsQueryByCategory(topic="error")
        self.assertIsNone(result)

    @patch("Articles.NewsApi.datetime")
    def test_news_query_by_source(self, mock_datetime):
        """Test _NewsQueryBySource searches by source and only by topic, returns list."""
        fixed_now = datetime(2026, 9, 16, 12, 0, 0)
        mock_datetime.now.return_value = fixed_now
        expected_date = (fixed_now - timedelta(days=7)).strftime("%Y-%m-%d")

        expected_source = {"status": "ok", "articles": [{"title": "TechCrunch Article"}]}
        expected_topic = {"status": "ok", "articles": [{"title": "Global Startups Article"}]}
        self.mock_client_instance.get_everything.side_effect = [expected_source, expected_topic]

        result = self.news_api._NewsQueryBySource(source="techcrunch", topic="startups")

        self.mock_client_instance.get_sources.assert_called_with(category=Category)
        self.assertEqual(self.mock_client_instance.get_everything.call_count, 2)
        self.mock_client_instance.get_everything.assert_any_call(
            q="startups",
            sources="techcrunch",
            from_param=expected_date,
            sort_by="publishedAt",
        )
        self.mock_client_instance.get_everything.assert_any_call(
            q="startups",
            from_param=expected_date,
            sort_by="publishedAt",
        )
        self.assertEqual(result, [expected_source, expected_topic])

    @patch("Articles.NewsApi.datetime")
    def test_news_query_everything_without_source(self, mock_datetime):
        """Test _NewsQueryEverything without source searches all domains and calls only on topic."""
        fixed_now = datetime(2026, 9, 16, 12, 0, 0)
        mock_datetime.now.return_value = fixed_now
        expected_date = (fixed_now - timedelta(days=7)).strftime("%Y-%m-%d")

        expected_response1 = {"status": "ok", "articles": [{"title": "AI Global 1"}]}
        expected_response2 = {"status": "ok", "articles": [{"title": "AI Global 2"}]}
        self.mock_client_instance.get_everything.side_effect = [expected_response1, expected_response2]

        result = self.news_api._NewsQueryEverything(topic="ai agents")

        self.mock_client_instance.get_sources.assert_called_with(category=Category)
        self.assertEqual(self.mock_client_instance.get_everything.call_count, 2)
        self.mock_client_instance.get_everything.assert_any_call(
            q="ai agents",
            from_param=expected_date,
            sort_by="publishedAt",
        )
        self.assertEqual(result, [expected_response1, expected_response2])

    @patch("Articles.NewsApi.datetime")
    def test_news_query_everything_with_source(self, mock_datetime):
        """Test _NewsQueryEverything with source filters by sources, plus call without source."""
        fixed_now = datetime(2026, 9, 16, 12, 0, 0)
        mock_datetime.now.return_value = fixed_now
        expected_date = (fixed_now - timedelta(days=7)).strftime("%Y-%m-%d")

        expected_source = {"status": "ok", "articles": [{"title": "Ars Technica"}]}
        expected_topic = {"status": "ok", "articles": [{"title": "Hardware Topic"}]}
        self.mock_client_instance.get_everything.side_effect = [expected_source, expected_topic]

        result = self.news_api._NewsQueryEverything(source="ars-technica", topic="hardware")

        self.mock_client_instance.get_sources.assert_called_with(category=Category)
        self.assertEqual(self.mock_client_instance.get_everything.call_count, 2)
        self.mock_client_instance.get_everything.assert_any_call(
            q="hardware",
            sources="ars-technica",
            from_param=expected_date,
            sort_by="publishedAt",
        )
        self.mock_client_instance.get_everything.assert_any_call(
            q="hardware",
            from_param=expected_date,
            sort_by="publishedAt",
        )
        self.assertEqual(result, [expected_source, expected_topic])

    def test_news_query_everything_handles_exception(self):
        """Test _NewsQueryEverything catches exceptions and returns None."""
        self.mock_client_instance.get_everything.side_effect = Exception("Everything failed")
        result = self.news_api._NewsQueryEverything(topic="fail")
        self.assertIsNone(result)

    def test_news_query_headlines_with_source(self):
        """Test _NewsQueryHeadlines with specific source and second call on topic."""
        expected_source = {"status": "ok", "articles": [{"title": "BBC Headline"}]}
        expected_topic = {"status": "ok", "articles": [{"title": "Tech Headline"}]}
        self.mock_client_instance.get_top_headlines.side_effect = [expected_source, expected_topic]

        result = self.news_api._NewsQueryHeadlines(source="bbc-news", topic="tech")

        self.mock_client_instance.get_sources.assert_called_with(category=Category)
        self.assertEqual(self.mock_client_instance.get_top_headlines.call_count, 2)
        self.mock_client_instance.get_top_headlines.assert_any_call(
            q="tech",
            sources="bbc-news",
        )
        self.mock_client_instance.get_top_headlines.assert_any_call(
            q="tech",
        )
        self.assertEqual(result, [expected_source, expected_topic])

    def test_news_query_headlines_without_source(self):
        """Test _NewsQueryHeadlines without source does not use failsafe Category check."""
        expected_res1 = {"status": "ok", "articles": [{"title": "General Headline 1"}]}
        expected_res2 = {"status": "ok", "articles": [{"title": "General Headline 2"}]}
        self.mock_client_instance.get_top_headlines.side_effect = [expected_res1, expected_res2]

        result = self.news_api._NewsQueryHeadlines(source=None, topic="ai")

        self.mock_client_instance.get_sources.assert_called_with(category=Category)
        self.assertEqual(self.mock_client_instance.get_top_headlines.call_count, 2)
        for call_item in self.mock_client_instance.get_top_headlines.call_args_list:
            self.assertNotIn("category", call_item.kwargs)
        self.assertEqual(result, [expected_res1, expected_res2])


if __name__ == "__main__":
    unittest.main()
