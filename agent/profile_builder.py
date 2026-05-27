"""
Phase 2: Profile Builder — Synthesizes raw research data into a
structured Research Profile.

UPDATED: Now makes direct LLM API calls.
"""

from .llm_client import LLMClient, estimate_tokens


class ProfileBuilder:
    """Synthesizes research data into a structured Research Profile."""

    def build(self, topic: str, research_data: str,
              llm_client: LLMClient = None) -> str:
        """
        Build a structured Research Profile from raw research data.

        Args:
            topic: The research topic
            research_data: Combined output from all research agents
            llm_client: LLM client for API calls

        Returns:
            Structured Research Profile text
        """
        if llm_client:
            return self._build_with_llm(topic, research_data, llm_client)
        else:
            return self._build_fallback(topic, research_data)

    def _build_with_llm(self, topic: str, research_data: str,
                         llm_client: LLMClient) -> str:
        """Use LLM to synthesize research into a profile."""
        system = (
            "You are a research synthesizer. Your job is to take raw research "
            "data from multiple sources and angles, and compile it into a "
            "structured, balanced Research Profile. Be rigorous — note where "
            "evidence is strong vs weak, where sources agree vs disagree."
        )

        user_msg = f"""TOPIC: {topic}

RAW RESEARCH DATA FROM {research_data.count('## Research Angle:')} RESEARCH ANGLES:
{research_data[:15000]}

TASK: Synthesize this into a structured Research Profile.

OUTPUT FORMAT:

# Research Profile: {topic}

## Executive Summary
2-3 paragraph overview of what we know, what's uncertain, and key findings.

## Key Findings
For each major finding:
- **Finding:** [clear statement]
- **Evidence Strength:** [Strong/Moderate/Weak]
- **Sources:** [which sources support this]
- **Confidence:** [High/Medium/Low]

## Data & Statistics
Key numbers, metrics, and data points with sources.

## Stakeholder Map
Who are the key players, beneficiaries, and affected parties?

## Contradictions & Tensions
Where do sources disagree? What are the competing narratives?

## Knowledge Gaps
What couldn't be verified? What's missing from the research?

## Sources
Complete list of all sources consulted with URLs.

## Overall Confidence Assessment
Rate the overall research quality: HIGH / MODERATE / LOW
Justify the rating based on source quality, corroboration, and gaps."""

        result = llm_client.chat(
            messages=[{"role": "user", "content": user_msg}],
            system=system,
            temperature=0.3,
            max_tokens=6000
        )

        if result["success"]:
            return result["content"]
        else:
            return self._build_fallback(topic, research_data)

    def _build_fallback(self, topic: str, research_data: str) -> str:
        """Basic profile without LLM — just restructure the raw data."""
        return f"""# Research Profile: {topic}

## Executive Summary
This profile was generated from raw research data without LLM synthesis.
Quality may be lower than when an LLM client is configured.

## Raw Research Data
{research_data[:10000]}

## Overall Confidence Assessment: LOW
No LLM client configured — raw data only. Configure an API key for full synthesis.
"""
