"""Tests for the web search fallback chain and result dedup.

search_web cascades through four providers: the ddgs library, the legacy
duckduckgo-search library, DDG HTML scraping, and the Instant Answer API.
The library imports are stubbed out so the offline fallbacks are exercised
deterministically without network access.
"""

import sys
import unittest
from unittest import mock

from agent.web_search import fetch_url_content, multi_search, search_web

from tests.helpers import FakeResponse

HTML_WITH_RESULTS = """
<div class="result">
  <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Farticle&amp;rut=abc">Example <b>Article</b></a>
  <a class="result__snippet">Some snippet text</a>
</div>
"""


def _force_offline_libraries():
    """Make ddgs imports fail so search cascades to the offline fallbacks."""
    saved = {name: sys.modules.get(name) for name in ("ddgs", "duckduckgo_search")}
    for name in saved:
        sys.modules[name] = None

    def restore():
        for name, mod in saved.items():
            if mod is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = mod

    return restore


def _fake_http_client(post_response, get_response):
    """Build a stand-in for httpx.Client with canned post/get responses."""

    class FakeHttpClient:
        def __init__(self, timeout=15, follow_redirects=True):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def post(self, url, headers=None, data=None):
            return post_response

        def get(self, url, params=None, headers=None):
            return get_response

    return FakeHttpClient


class SearchWebTest(unittest.TestCase):
    def setUp(self):
        self._restore = _force_offline_libraries()
        self.addCleanup(self._restore)

    def test_html_fallback_parses_results_and_decodes_uddg(self):
        fake = _fake_http_client(FakeResponse(200, text=HTML_WITH_RESULTS), None)
        with mock.patch("agent.web_search.httpx.Client", fake):
            results = search_web("email api providers")

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["title"], "Example Article")
        self.assertEqual(results[0]["snippet"], "Some snippet text")
        self.assertEqual(results[0]["url"], "https://example.com/article")

    def test_instant_answer_api_is_last_resort(self):
        json_data = {
            "Heading": "Test Topic",
            "Abstract": "The abstract text",
            "AbstractURL": "https://example.org/abstract",
            "RelatedTopics": [
                {"Text": "Related one", "FirstURL": "https://example.org/r1"},
                {"Text": "Related two", "FirstURL": "https://example.org/r2"},
                {"Name": "Category only — no Text field"},
            ],
        }
        fake = _fake_http_client(
            FakeResponse(200, text="<html>no results</html>"),
            FakeResponse(200, json_data=json_data),
        )
        with mock.patch("agent.web_search.httpx.Client", fake):
            results = search_web("anything")

        urls = [r["url"] for r in results]
        self.assertIn("https://example.org/abstract", urls)
        self.assertIn("https://example.org/r1", urls)
        self.assertIn("https://example.org/r2", urls)
        # Category entries without a Text field are dropped.
        self.assertEqual(len(results), 3)

    def test_all_fallbacks_fail_returns_empty_list(self):
        fake = _fake_http_client(FakeResponse(500, text="broken"), None)
        with mock.patch("agent.web_search.httpx.Client", fake):
            results = search_web("anything")

        self.assertEqual(results, [])


class MultiSearchTest(unittest.TestCase):
    def test_deduplicates_results_by_url(self):
        self._restore = _force_offline_libraries()
        self.addCleanup(self._restore)

        first_query = [
            {"title": "A", "url": "https://same.example.org", "snippet": "1"},
            {"title": "B", "url": "https://b.example.org", "snippet": "2"},
        ]
        second_query = [
            {"title": "A again", "url": "https://same.example.org", "snippet": "dup"},
            {"title": "C", "url": "https://c.example.org", "snippet": "3"},
        ]
        with mock.patch(
            "agent.web_search.search_web",
            side_effect=[first_query, second_query],
        ), mock.patch("time.sleep") as sleep:
            results = multi_search(["query one", "query two"], max_results_per_query=5)

        self.assertEqual([r["url"] for r in results], [
            "https://same.example.org",
            "https://b.example.org",
            "https://c.example.org",
        ])
        sleep.assert_called_with(0.5)


class FetchUrlContentTest(unittest.TestCase):
    def test_http_error_returns_status_message(self):
        fake = _fake_http_client(None, FakeResponse(404, text="not found"))
        with mock.patch("agent.web_search.httpx.Client", fake):
            content = fetch_url_content("https://example.org/missing")

        self.assertIn("[HTTP 404]", content)

    def test_strips_html_when_no_extractor_available(self):
        html = "<html><body><script>var x=1;</script><p>Hello <b>world</b></p></body></html>"
        fake = _fake_http_client(None, FakeResponse(200, text=html))
        with mock.patch("agent.web_search.httpx.Client", fake), mock.patch.dict(
            sys.modules, {"trafilatura": None}
        ):
            content = fetch_url_content("https://example.org/page")

        self.assertIn("Hello", content)
        self.assertNotIn("<", content)
        self.assertNotIn("script", content)


if __name__ == "__main__":
    unittest.main()
