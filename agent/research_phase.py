"""
Phase 1: Deep Research — Spawns multiple research subagents to gather
comprehensive data on the topic from diverse angles.

UPDATED: Now makes direct LLM API calls + web search.
No longer returns raw prompts — actually executes research.
"""

import json
import time
from typing import Optional, Callable

from .llm_client import LLMClient, estimate_tokens
from .web_search import search_web, fetch_url_content


# Research angles — each subagent focuses on a different perspective
RESEARCH_ANGLES = [
    {
        "name": "primary",
        "focus": "Core facts, definitions, key players, market data, and primary sources",
        "instruction": "Gather the most comprehensive primary research possible. Focus on facts, data points, statistics, key entities, and foundational knowledge.",
        "search_queries": lambda topic: [
            topic,
            f"{topic} market size data 2024 2025",
            f"{topic} key players companies",
            f"{topic} statistics facts"
        ]
    },
    {
        "name": "critical",
        "focus": "Criticisms, counter-arguments, failures, risks, and negative outcomes",
        "instruction": "Focus specifically on the NEGATIVE side: criticisms, failures, risks, counter-arguments, things that went wrong, and reasons this might fail or be wrong.",
        "search_queries": lambda topic: [
            f"{topic} criticism problems",
            f"{topic} failures risks downsides",
            f"{topic} why it fails",
            f"{topic} complaints issues"
        ]
    },
    {
        "name": "comparative",
        "focus": "Alternatives, competitors, adjacent approaches, and market context",
        "instruction": "Focus on the COMPETITIVE LANDSCAPE: alternatives, competitors, adjacent approaches, how others solve the same problem, and market positioning.",
        "search_queries": lambda topic: [
            f"{topic} alternatives competitors",
            f"{topic} vs comparison",
            f"{topic} market landscape",
            f"{topic} competitive analysis"
        ]
    },
    {
        "name": "emerging",
        "focus": "Recent developments, trends, future outlook, and emerging signals",
        "instruction": "Focus on the FUTURE and EMERGING TRENDS: recent developments, where this is heading, new research, upcoming changes, and forward-looking signals.",
        "search_queries": lambda topic: [
            f"{topic} trends 2025 2026",
            f"{topic} future outlook",
            f"{topic} new developments",
            f"{topic} emerging innovations"
        ]
    }
]


class ResearchPhase:
    """Spawns research subagents to gather comprehensive data on a topic."""

    def __init__(self, config: dict = None):
        self.config = config or {}
        self.subagent_timeout = self.config.get("subagent_timeout", 300)

    def execute(self, topic: str, num_agents: int = 4, run_id: str = "",
                log_fn: Optional[Callable] = None,
                llm_client: Optional[LLMClient] = None) -> str:
        """
        Execute deep research on a topic.

        Uses web search to gather raw data, then LLM to synthesize findings.
        Returns concatenated research data from all angles.
        """
        angles = RESEARCH_ANGLES[:num_agents]
        results = []

        for i, angle in enumerate(angles):
            angle_name = angle["name"]
            if log_fn:
                log_fn("research", 0, f"researcher_{angle_name}", i, "running")

            start = time.time()
            try:
                data = self._research_angle(topic, angle, llm_client)
                duration = int((time.time() - start) * 1000)
                results.append({"angle": angle_name, "data": data})
                if log_fn:
                    log_fn("research", 0, f"researcher_{angle_name}", i,
                           "completed", data[:1000], duration_ms=duration)
            except Exception as e:
                duration = int((time.time() - start) * 1000)
                results.append({"angle": angle_name, "data": f"[Error: {e}]"})
                if log_fn:
                    log_fn("research", 0, f"researcher_{angle_name}", i,
                           "error", str(e)[:500], duration_ms=duration)

        combined = self._combine_results(topic, results)
        return combined

    def _research_angle(self, topic: str, angle: dict,
                        llm_client: Optional[LLMClient] = None) -> str:
        """Research a single angle: web search + LLM synthesis."""

        # Step 1: Web search for this angle
        query_fn = angle.get("search_queries")
        if query_fn:
            queries = query_fn(topic)
        else:
            queries = [f"{topic} {angle['focus']}"]

        search_results = []
        for query in queries[:3]:  # Max 3 queries per angle
            try:
                results = search_web(query, max_results=5)
                search_results.extend(results)
            except Exception:
                pass

        # Step 2: Fetch content from top results
        fetched_content = []
        for result in search_results[:5]:
            url = result.get("url", "")
            if url:
                try:
                    content = fetch_url_content(url, max_chars=3000)
                    fetched_content.append({
                        "title": result.get("title", ""),
                        "url": url,
                        "content": content
                    })
                except Exception:
                    pass

        # Step 3: If we have an LLM client, synthesize with it
        if llm_client and (search_results or fetched_content):
            return self._synthesize_with_llm(topic, angle, search_results,
                                              fetched_content, llm_client)
        elif search_results:
            # No LLM available — return raw search results
            return self._format_raw_results(search_results, fetched_content)
        else:
            return f"[No search results found for angle: {angle['name']}]"

    def _synthesize_with_llm(self, topic: str, angle: dict,
                              search_results: list, fetched_content: list,
                              llm_client: LLMClient) -> str:
        """Use LLM to synthesize search results into structured research."""

        # Build context from search results
        context_parts = []
        for i, result in enumerate(search_results[:8]):
            context_parts.append(
                f"[{i+1}] {result.get('title', 'No title')}\n"
                f"    URL: {result.get('url', 'N/A')}\n"
                f"    Snippet: {result.get('snippet', 'N/A')}"
            )

        for item in fetched_content[:5]:
            context_parts.append(
                f"\n--- Full content from: {item['title']} ---\n"
                f"{item['content'][:2000]}\n"
            )

        context = "\n".join(context_parts)

        system = "You are a rigorous research analyst. Extract facts, cite sources, never fabricate."
        user_msg = f"""TOPIC: {topic}
RESEARCH ANGLE: {angle['name'].upper()} — {angle['focus']}

INSTRUCTIONS: {angle['instruction']}

SEARCH RESULTS:
{context}

TASK:
Analyze the above search results and extract structured research findings.

OUTPUT FORMAT:
## KEY FACTS
- [fact] (Source: [URL or title])

## DATA POINTS
- [specific number/statistic] (Source: [URL])

## CONTRADICTIONS
- [any conflicting information between sources]

## GAPS
- [what you couldn't find or verify]

## SOURCES CONSULTED
[numbered list of all sources with URLs]

Be thorough. Be specific. Cite everything. Do NOT fabricate information."""

        result = llm_client.chat(
            messages=[{"role": "user", "content": user_msg}],
            system=system,
            temperature=0.3,
            max_tokens=4096
        )

        if result["success"]:
            return result["content"]
        else:
            # Fallback to raw results if LLM fails
            return self._format_raw_results(search_results, fetched_content)

    def _format_raw_results(self, search_results: list,
                             fetched_content: list) -> str:
        """Format raw search results as fallback when no LLM available."""
        parts = ["## Raw Search Results\n"]
        for i, r in enumerate(search_results[:10]):
            parts.append(f"{i+1}. **{r.get('title', 'No title')}**")
            parts.append(f"   URL: {r.get('url', 'N/A')}")
            parts.append(f"   {r.get('snippet', 'No snippet')}\n")

        if fetched_content:
            parts.append("\n## Fetched Content\n")
            for item in fetched_content[:5]:
                parts.append(f"### {item['title']}")
                parts.append(f"URL: {item['url']}")
                parts.append(f"{item['content'][:1500]}\n")

        return "\n".join(parts)

    def _combine_results(self, topic: str, results: list) -> str:
        """Combine research results from all angles into one document."""
        sections = [f"# Research Data: {topic}\n"]
        for r in results:
            angle = r.get("angle", "unknown")
            data = r.get("data", "")
            sections.append(f"\n## Research Angle: {angle.upper()}\n\n{data}\n")
            sections.append("---\n")
        return "\n".join(sections)
