# No-Slop Research

An adversarial research agent that eliminates LLM bias, incomplete research, and unverified claims through multi-phase interrogation with Team A (validators) vs Team B (challengers).

## What Problem Does This Solve?

LLMs have three well-known failure modes:
1. **Bias toward telling you what you want to hear** — they'll confirm your hypothesis instead of challenging it
2. **Incomplete research** — they miss key angles, counter-arguments, and emerging trends
3. **No verification** — they present claims as facts without evidence chains or confidence ratings

No-Slop Research solves all three by running every research output through an adversarial gauntlet before presenting it.

## How It Works

```
User Topic
    │
    ▼
┌─────────────────────────────────────────────────────┐
│ PHASE 1: DEEP RESEARCH                              │
│ 4 research agents search the web from different     │
│ angles: primary, critical, comparative, emerging    │
│ Uses DuckDuckGo + LLM synthesis                     │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│ PHASE 2: PROFILE BUILDING                           │
│ Raw research synthesized into structured Research   │
│ Profile with findings, evidence ratings, sources    │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│ PHASE 3: ADVERSARIAL LOOP (up to 3 rounds)         │
│                                                     │
│  ┌──────────────┐    ┌──────────────┐              │
│  │  TEAM A      │    │  TEAM B      │              │
│  │  Validators  │    │  Challengers │              │
│  │              │    │              │              │
│  │ "Why is this │    │ "Why is this │              │
│  │  research    │    │  research    │              │
│  │  CORRECT?"   │    │  WRONG?"     │              │
│  └──────┬───────┘    └──────┬───────┘              │
│         │                   │                       │
│         └───────┬───────────┘                       │
│                 │                                   │
│                 ▼                                   │
│    ┌────────────────────────┐                      │
│    │  IMPROVEMENT POINTS    │                      │
│    │  (noise-filtered)      │                      │
│    └────────────┬───────────┘                      │
│                 │                                   │
│    ┌────────────▼───────────┐                      │
│    │  SYNTHESIZER           │                      │
│    │  Merges fixes back     │                      │
│    │  into profile          │                      │
│    └────────────┬───────────┘                      │
│                 │                                   │
│         Loop until Team B finds no flaws            │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│ PHASE 5: FINAL REPORT                               │
│ Clean report with confidence ratings, sources,      │
│ what survived adversarial testing, and caveats      │
└─────────────────────────────────────────────────────┘
```

## Architecture

### Direct LLM Calls
Every agent (research, profile builder, validator, challenger, synthesizer, report generator) makes **direct API calls** to any OpenAI-compatible endpoint. No external agent framework dependency.

### Supported LLM Providers
- OpenAI (GPT-4o, GPT-4o-mini, etc.)
- OpenRouter (Claude, Gemini, Llama, etc.)
- Anthropic (Claude direct API)
- Groq, Together AI, DeepSeek
- Any OpenAI-compatible endpoint (custom)

### Web Search
Uses [duckduckgo-search](https://github.com/deedy5/duckduckgo_search) for web research — no API key required.

### Noise Filtering
Team B's improvement points are automatically filtered for:
- Generic noise ("more research is needed")
- Vague/non-actionable suggestions
- Duplicate points (similarity detection)
- Minimum substance requirements

### Confidence Scoring
Evidence-based scoring using weighted composite:
- Source quality (30%) — number and credibility of sources
- Corroboration (25%) — multiple sources agreeing
- Evidence strength (30%) — strength of evidence chains
- Weakness penalty (15%) — deductions for known flaws

### Cost Tracking
Every LLM call is tracked with token counts and estimated cost per run.

## Quick Start

### 1. Install

```bash
git clone https://github.com/YOUR_USERNAME/no-slop-research.git
cd no-slop-research
pip install -r requirements.txt
```

### 2. Configure API Key

Option A: Environment variables
```bash
export LLM_API_KEY="your-api-key"
export LLM_BASE_URL="https://api.openai.com/v1"  # or any OpenAI-compatible
export LLM_MODEL_NAME="gpt-4o-mini"
```

Option B: Dashboard (recommended)
```bash
python -m dashboard.app
# Open http://localhost:5060 → API Keys tab → Add your key
```

Option C: .env file
```bash
cp .env.example .env
# Edit .env with your keys
```

### 3. Run Research

Via the dashboard:
```bash
python -m dashboard.app
# Open http://localhost:5060 → Research tab → Enter topic
```

Via Python:
```python
from agent.orchestrator import ResearchPipeline

pipeline = ResearchPipeline(
    topic="What are the best email API providers in 2025?",
    config={"max_rounds": 2, "max_research_agents": 4}
)
result = pipeline.run()

print(result["final_report"])
print(f"Confidence: {result['confidence']['confidence_label']}")
print(f"Cost: ${result['cost']['total_cost_usd']:.4f}")
```

## Dashboard

```bash
python -m dashboard.app
# http://localhost:5060
```

Features:
- **Research tab** — Start new adversarial research from the UI
- **Active tab** — Monitor running pipelines in real-time
- **History tab** — View all completed research runs
- **API Keys tab** — Manage keys for 7 providers (OpenAI, OpenRouter, Anthropic, Groq, Together, DeepSeek, Custom)
- **Settings tab** — Configure pipeline defaults

## File Structure

```
no-slop-research/
├── README.md
├── requirements.txt
├── .env.example
├── agent/
│   ├── __init__.py
│   ├── orchestrator.py          # Main pipeline controller
│   ├── llm_client.py            # Direct LLM API client (requests-based)
│   ├── web_search.py            # DuckDuckGo web search
│   ├── research_phase.py        # Web research + LLM synthesis
│   ├── profile_builder.py       # Synthesizes research into profile
│   ├── validator_team.py        # Team A — validates findings
│   ├── challenger_team.py       # Team B — finds flaws
│   ├── synthesizer.py           # Merges improvement points back
│   ├── report_generator.py      # Final output
│   ├── noise_filter.py          # Filters Team B noise
│   └── confidence_scorer.py     # Evidence-based scoring
├── dashboard/
│   ├── app.py                   # Flask dashboard (port 5060)
│   └── templates/
│       └── index.html
└── examples/
    └── sample_run.md
```

## Cost Estimation

Approximate cost per run (varies by topic complexity):

| Model | 1 Round | 2 Rounds | 3 Rounds |
|-------|---------|----------|----------|
| GPT-4o-mini | ~$0.05 | ~$0.10 | ~$0.15 |
| GPT-4o | ~$0.50 | ~$1.00 | ~$1.50 |
| Claude Sonnet | ~$0.40 | ~$0.80 | ~$1.20 |
| Llama 3.1 70B (Groq) | ~$0.03 | ~$0.06 | ~$0.09 |

Each round uses ~4-6 LLM calls (research agents + profile + validator + challenger + synthesizer).

## Limitations

- **Web search quality depends on DuckDuckGo** — rate limits may apply
- **LLM quality affects output** — garbage in, garbage out
- **Cost scales with rounds** — complex topics may need 3 rounds
- **Not real-time** — a full run takes 2-10 minutes depending on model speed
- **Confidence scores are estimates** — not statistical confidence intervals

## License

MIT
