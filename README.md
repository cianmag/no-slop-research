# No-Slop Research

> Adversarial research that survives interrogation before it reaches you.

## What Is This?

No-Slop Research is an adversarial research and validation agent system built on [Hermes Agent](https://hermes-agent.nousresearch.com). When you give it a topic, it doesn't just search the web and hand you results. Instead, it runs a **multi-phase adversarial pipeline** where two opposing teams — Validators (Team A) and Challengers (Team B) — interrogate the research until it's bulletproof.

The system eliminates three critical LLM failures that plague every AI research tool:

1. **Bias toward telling you what you want to hear** — Team B actively tries to break every finding, looking for cherry-picked data, missing perspectives, and overconfident claims.
2. **Incomplete research that misses key angles** — Four specialized research agents attack the topic from different angles (primary facts, critical analysis, competitive landscape, emerging trends).
3. **No verification that the answer is actually correct** — Every claim gets a confidence rating, source citation, and survives adversarial rounds before appearing in the final report.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    USER PROVIDES TOPIC                       │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              PHASE 1: DEEP RESEARCH                          │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐      │
│  │ Primary   │ │ Critical │ │Comparative│ │ Emerging │      │
│  │ Agent     │ │ Agent    │ │ Agent     │ │ Agent    │      │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘      │
│       └─────────────┴─────────────┴─────────────┘           │
│                           │                                  │
│                    Combined Research Data                    │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              PHASE 2: PROFILE BUILDING                       │
│         Synthesize into Structured Research Profile          │
│    (findings, data points, gaps, contradictions, sources)    │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│         PHASE 3: ADVERSARIAL INTERROGATION (loop)            │
│                                                              │
│   ┌─────────────────┐     ┌─────────────────┐              │
│   │  TEAM A          │     │  TEAM B          │              │
│   │  Validators      │     │  Challengers     │              │
│   │  ✅ Why it's     │     │  🔴 Why it's    │              │
│   │     correct      │     │     WRONG        │              │
│   │  Rate confidence │     │  Find flaws      │              │
│   └────────┬────────┘     └────────┬────────┘              │
│            └──────────┬────────────┘                        │
│                       │                                      │
│              Improvement Points                              │
│                       │                                      │
│                       ▼                                      │
│   ┌─────────────────────────────────────┐                   │
│   │  PHASE 4: SYNTHESIS & RE-RESEARCH   │                   │
│   │  Address weaknesses → Re-interrogate│                   │
│   │  Loop until Team B finds no flaws   │                   │
│   │  (max 3 rounds default)             │                   │
│   └─────────────────────────────────────┘                   │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              PHASE 5: FINAL REPORT                           │
│   -Executive Summary                                       │
│   -Key Findings with Confidence Ratings                    │
│   -What Was Challenged & How It Was Addressed              │
│   -Remaining Caveats                                       │
│   - Full Source Citations                                    │
└─────────────────────────────────────────────────────────────┘
```

## Installation

### Prerequisites

- Python 3.10+
- [Hermes Agent](https://hermes-agent.nousresearch.com/docs) (for subagent orchestration)
- An LLM API key (OpenAI, OpenRouter, Anthropic, Groq, etc.)

### Clone & Install

```bash
git clone https://github.com/YOUR_USERNAME/no-slop-research.git
cd no-slop-research
pip install -r requirements.txt
```

### Configure Environment

```bash
cp .env.example .env
```

Edit `.env` and add your API keys:

```env
# Required: Any OpenAI-compatible LLM API
LLM_API_KEY=your_api_key_here
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL_NAME=gpt-4o

# Optional: TinyFish for enhanced web research
TINYFISH_API_KEY=your_tinyfish_key
```

Or add keys through the dashboard (recommended — see below).

## How to Run the Agent

### Option 1: Via Hermes Agent (Recommended)

If you have Hermes Agent installed, the skill is automatically available:

```
/no-slop-research What is the state of quantum computing in 2025?
```

Hermes will orchestrate the full pipeline using delegate_task for parallel subagent execution.

### Option 2: Standalone Python

```python
from agent.orchestrator import ResearchPipeline

pipeline = ResearchPipeline(
    topic="What is the state of quantum computing in 2025?",
    config={"max_rounds": 3}
)

result = pipeline.run()

print(result["final_report"])
print(f"Completed in {result['rounds']} adversarial rounds")
```

### Option 3: CLI

```bash
cd no-slop-research
python -c "
from agent.orchestrator import ResearchPipeline
import sys

topic = ' '.join(sys.argv[1:])
if not topic:
    print('Usage: python run.py <topic>')
    sys.exit(1)

pipeline = ResearchPipeline(topic=topic, config={'max_rounds': 3})
result = pipeline.run()

if result['success']:
    print(result['final_report'])
else:
    print(f'Error: {result[\"error\"]}')
" "Your research topic here"
```

## How to Start the Dashboard

```bash
cd no-slop-research
pip install -r requirements.txt
python -m dashboard.app
```

The dashboard starts on **http://localhost:5060**

### Dashboard Features

#### Research Tab
- Enter any research topic
- Configure rounds, agents, depth, and output format
- One-click adversarial research execution
- Live stats (total runs, completed, running, avg rounds)
- Recent results with progress indicators

#### Active Tab
- Real-time view of running research pipelines
- Phase progress bars
- Current round and status

#### History Tab
- Full history of all research runs
- Status, phase, round tracking
- Click any run to see full details

#### API Keys Tab
- Add keys for any supported provider:
  - **OpenAI** (GPT-4o, GPT-4 Turbo)
  - **OpenRouter** (Claude, Gemini, Llama, etc.)
  - **Anthropic** (Claude 3.5 Sonnet, Haiku)
  - **Groq** (Llama 3.1, Mixtral)
  - **Together AI** (Llama, Mistral)
  - **DeepSeek** (DeepSeek Chat, Coder)
  - **Custom** (any OpenAI-compatible endpoint)
- Toggle keys active/inactive
- Keys stored locally in SQLite (never sent anywhere except the provider)
- Masked display for security

#### Settings Tab
- Default pipeline settings
- System health info
- Database status

### Run Detail View
Click any research run to see:
- Phase timeline (research → profile → adversarial → synthesis → report)
- Team A vs Team B comparison (validators vs challengers)
- Improvement points tracker (found vs resolved)
- Subagent activity log (role, status, duration)
- Full final report with confidence ratings

## How the Adversarial Validation Loop Works

```
Round 1:
  Team A validates → "These 5 findings are solid, 2 are weak"
  Team B challenges → "3 critical flaws found, 5 improvement points"
  → Synthesize improvements back into profile

Round 2:
  Team A validates → "7 findings now solid, improved from 62% → 78% confidence"
  Team B challenges → "2 minor gaps remaining"
  → Synthesize improvements back into profile

Round 3:
  Team A validates → "83% confidence, all major claims verified"
  Team B challenges → "No significant flaws found"
  → Research validated! Exit loop.

Final Report: Clean output with confidence ratings, source citations,
              and transparency about what was challenged.
```

The loop exits when:
- Team B finds **no significant improvement points** (research is bulletproof), OR
- **Max rounds reached** (default: 3, configurable)

## How to Add API Tokens

### Via Dashboard (Recommended)

1. Open http://localhost:5060
2. Click "API Keys" in the navigation
3. Select your provider from the dropdown
4. Enter your API key name and value
5. Click "Save API Key"
6. The key is stored locally in SQLite and marked as Active

### Via Environment Variables

Add to your `.env` file:

```env
# OpenAI
LLM_API_KEY=sk-...
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL_NAME=gpt-4o

# Or OpenRouter (access to many models)
LLM_API_KEY=sk-or-...
LLM_BASE_URL=https://openrouter.ai/api/v1
LLM_MODEL_NAME=anthropic/claude-sonnet-4

# Or Groq (fast inference)
LLM_API_KEY=gsk_...
LLM_BASE_URL=https://api.groq.com/openai/v1
LLM_MODEL_NAME=llama-3.1-70b-versatile
```

### Supported Provider Quick Reference

| Provider | Base URL | Key Prefix | Best Models |
|----------|----------|------------|-------------|
| OpenAI | `https://api.openai.com/v1` | `sk-` | gpt-4o, gpt-4o-mini |
| OpenRouter | `https://openrouter.ai/api/v1` | `sk-or-` | Any model via routing |
| Anthropic | `https://api.anthropic.com/v1` | `sk-ant-` | claude-sonnet-4 |
| Groq | `https://api.groq.com/openai/v1` | `gsk_` | llama-3.1-70b |
| Together AI | `https://api.together.xyz/v1` | — | Llama, Mistral |
| DeepSeek | `https://api.deepseek.com/v1` | `sk-` | deepseek-chat |
| Custom | Your URL | Any | Any OpenAI-compatible |

## Example Usage

### Research a Market

```bash
/no-slop-research What is the market opportunity for an email API SaaS targeting mid-market companies? Analyze competitors like Resend, SendGrid, and Customer.io.
```

### Validate a Business Idea

```bash
/no-slop-research Is there a viable market for an AI-powered onboarding email automation tool? What would the pricing look like? Who are the target customers?
```

### Competitive Intelligence

```bash
/no-slop-research Compare the developer experience and pricing of Resend vs Postmark vs Mailgun in 2025. Which has the best API design?
```

### Technical Research

```bash
/no-slop-research What are the best approaches for building a multi-tenant email sending infrastructure? What are the deliverability challenges and solutions?
```

## File Structure

```
no-slop-research/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── .env.example                       # Environment template
├── agent/                             # Core pipeline modules
│   ├── __init__.py
│   ├── orchestrator.py                # Main pipeline controller
│   ├── research_phase.py              # Phase 1: Research subagents
│   ├── profile_builder.py             # Phase 2: Profile synthesis
│   ├── validator_team.py              # Phase 3a: Team A validators
│   ├── challenger_team.py             # Phase 3b: Team B challengers
│   ├── synthesizer.py                 # Phase 4: Improvement merger
│   └── report_generator.py            # Phase 5: Final report
├── dashboard/                         # Web dashboard
│   ├── app.py                         # Flask server (port 5060)
│   └── templates/
│       └── index.html                 # Dark-themed dashboard UI
├── skills/
│   └── no-slop-research/
│       └── SKILL.md                   # Hermes Agent skill definition
└── examples/
    └── sample_run.md                  # Example research pipeline run
```

## How It's Different

| Feature | Regular AI Search | No-Slop Research |
|---------|-------------------|------------------|
| Research depth | Single query, top results | 4 specialized agents, 5-10 sources each |
| Bias check | None | Team B actively attacks framing |
| Verification | Trust the output | Claims rated 0-100% confidence |
| Source quality | Whatever comes up | Cross-referenced, quality-assessed |
| Missing perspectives | You won't know | Explicitly identified and addressed |
| Iterative improvement | One-shot | Up to 3 adversarial rounds |
| Transparency | Black box | Full audit trail of what was challenged |

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with a real research topic
5. Submit a pull request

## License

MIT License — see [LICENSE](LICENSE) for details.
