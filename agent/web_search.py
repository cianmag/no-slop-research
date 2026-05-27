"""
No-Slop Research — Web Search

Provides web search capabilities for research agents.
Uses duckduckgo-search library with fallback to DuckDuckGo API.
"""

import re
import os
from typing import Optional

import httpx


def search_web(query: str, max_results: int = 8) -> list:
    """
    Search the web. Returns list of {title, snippet, url} dicts.

    Uses duckduckgo-search library (primary) with fallback to API.
    """
    results = []

    # Method 1: ddgs library (most reliable)
    try:
        from ddgs import DDGS
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append({
                    "title": r.get("title", ""),
                    "snippet": r.get("body", ""),
                    "url": r.get("href", "")
                })
        if results:
            return results
    except Exception:
        pass

    # Method 1b: duckduckgo-search library (legacy fallback)
    try:
        from duckduckgo_search import DDGS as DDGS_Legacy
        with DDGS_Legacy() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append({
                    "title": r.get("title", ""),
                    "snippet": r.get("body", ""),
                    "url": r.get("href", "")
                })
        if results:
            return results
    except Exception:
        pass

    # Method 2: DuckDuckGo HTML scraping (fallback)
    try:
        url = "https://html.duckduckgo.com/html/"
        headers = {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"
        }
        data = {"q": query, "b": ""}

        with httpx.Client(timeout=15, follow_redirects=True) as client:
            resp = client.post(url, headers=headers, data=data)

        if resp.status_code == 200:
            html = resp.text
            result_blocks = re.findall(
                r'<a[^>]+class="result__a"[^>]+href="([^"]*)"[^>]*>(.*?)</a>.*?'
                r'<a[^>]+class="result__snippet"[^>]*>(.*?)</a>',
                html, re.DOTALL
            )

            for link, title, snippet in result_blocks[:max_results]:
                title = re.sub(r'<[^>]+>', '', title).strip()
                snippet = re.sub(r'<[^>]+>', '', snippet).strip()
                if "uddg=" in link:
                    match = re.search(r'uddg=([^&]+)', link)
                    if match:
                        from urllib.parse import unquote
                        link = unquote(match.group(1))
                if title and link:
                    results.append({
                        "title": title,
                        "snippet": snippet,
                        "url": link
                    })
    except Exception:
        pass

    # Method 3: DuckDuckGo Instant Answer API (last resort)
    if not results:
        try:
            url = "https://api.duckduckgo.com/"
            params = {"q": query, "format": "json", "no_redirect": "1"}

            with httpx.Client(timeout=15) as client:
                resp = client.get(url, params=params)

            if resp.status_code == 200:
                data = resp.json()
                if data.get("Abstract"):
                    results.append({
                        "title": data.get("Heading", query),
                        "snippet": data["Abstract"],
                        "url": data.get("AbstractURL", "")
                    })
                for topic in data.get("RelatedTopics", [])[:max_results]:
                    if isinstance(topic, dict) and topic.get("Text"):
                        results.append({
                            "title": topic.get("Text", "")[:80],
                            "snippet": topic.get("Text", ""),
                            "url": topic.get("FirstURL", "")
                        })
        except Exception:
            pass

    return results


def fetch_url_content(url: str, max_chars: int = 8000) -> str:
    """
    Fetch and extract text content from a URL.
    Returns cleaned text, truncated to max_chars.
    """
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"
        }

        with httpx.Client(timeout=20, follow_redirects=True) as client:
            resp = client.get(url, headers=headers)

        if resp.status_code != 200:
            return f"[HTTP {resp.status_code}] Could not fetch {url}"

        html = resp.text

        # Try trafilatura first (better extraction)
        try:
            import trafilatura
            text = trafilatura.extract(html, include_links=False, include_tables=True)
            if text:
                return text[:max_chars]
        except ImportError:
            pass

        # Fallback: basic HTML stripping
        html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL)
        html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL)
        text = re.sub(r'<[^>]+>', ' ', html)
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'\n{3,}', '\n\n', text)

        return text[:max_chars]

    except Exception as e:
        return f"[Error fetching {url}: {str(e)}]"


def multi_search(queries: list, max_results_per_query: int = 5) -> list:
    """
    Run multiple search queries and deduplicate results by URL.
    """
    import time
    seen_urls = set()
    all_results = []

    for query in queries:
        results = search_web(query, max_results=max_results_per_query)
        for r in results:
            url = r.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                all_results.append(r)
        time.sleep(0.5)  # Rate limit between queries

    return all_results
