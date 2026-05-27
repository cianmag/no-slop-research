---
name: no-slop-research
description: "Adversarial research & validation agent — eliminates bias, gaps, and unverified claims through multi-phase interrogation with Team A (validators) vs Team B (challengers)."
version: 2.0.0
author: No-Slop Research
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [research, validation, adversarial, market-research, fact-checking]
    related_skills: [firecrawl-search, web-search, mirofish-marketing]
---

# No-Slop Research — Adversarial Research Agent

When given a topic, this skill runs a multi-phase adversarial research pipeline that eliminates three common LLM failures:
1. **Bias** toward telling the user what they want to hear
2. **Incomplete research** that misses key angles
3. **No verification** that the answer is actually correct

## Quick Start

```
/no-slop-research <topic or question>
```

## How It Works

### Phase 1: Deep Research
4 research agents search the web from different angles:
- **Primary**: Core facts, definitions, key players, market data
- **Critical**: Criticisms, failures, risks, counter-arguments
- **Comparative**: Alternatives, competitors, market context
- **Emerging**: Recent developments, trends, future outlook

Each agent uses DuckDuckGo web search + LLM synthesis.

### Phase 2: Profile Building
All gathered data is synthesized into a structured Research Profile.

### Phase 3: Adversarial Interrogation
Two opposing teams interrogate the profile:
- **Team A (Validators)**: Argues WHY findings are correct, rates confidence
- **Team B (Challengers)**: Tries to BREAK the research, finds flaws

Team B's output is noise-filtered to remove vague/duplicate points.

### Phase 4: Synthesis & Re-test
Improvement points fed back into the profile. Loop repeats until Team B finds no flaws (max 3 rounds).

### Phase 5: Final Report
Clean report with evidence-based confidence ratings, sources, and caveats.

## Key Components

- `agent/llm_client.py` — Direct LLM API calls (any OpenAI-compatible endpoint)
- `agent/web_search.py` — DuckDuckGo web search (no API key needed)
- `agent/noise_filter.py` — Filters Team B noise (dedup, severity, relevance)
- `agent/confidence_scorer.py` — Evidence-based scoring (not keyword matching)

## Execution Instructions

When the user provides a topic, execute the pipeline using `delegate_task` for research agents, or use the standalone orchestrator:

```python
from agent.orchestrator import ResearchPipeline

pipeline = ResearchPipeline(topic="...", config={"max_rounds": 2})
result = pipeline.run()
```

Requires: LLM_API_KEY, LLM_BASE_URL, LLM_MODEL_NAME in env or .env

## Dashboard

```bash
cd no-slop-research
pip install -r requirements.txt
python -m dashboard.app
# http://localhost:5060
```

## Pitfalls

- **API key required**: Without an LLM key, runs in degraded mode (raw search only)
- **Model speed**: Slow models make runs take 5-10+ minutes
- **Cost**: Each round uses 4-6 LLM calls. Budget accordingly.
- **Web search**: DuckDuckGo may rate-limit. Falls back gracefully.
