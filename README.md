# Acelab Material Recommendation Agent

An AI-powered agent that helps architects find the right building materials. Given a natural-language project description, it decomposes the request, searches the Acelab catalog across multiple dimensions, and returns ranked recommendations with clear reasoning.

## How to Run

```bash
# 1. Install dependencies
uv sync

# 2. Copy env and fill in your keys
cp .env.example .env
# Edit .env with your ACELAB_API_KEY, ACELAB_BASE_URL, OPENROUTER_API_KEY

# 3. Run the CLI
uv run python main.py "high-traffic hospital corridor, LEED Silver, mid-range budget"

# 4. Or run the web UI
uv run uvicorn server:app --reload
# Open http://localhost:8000

# 5. Run tests (no API keys needed)
uv run pytest tests/ -v
```

## Example Output

```
Query: "high-traffic hospital corridor, LEED Silver, mid-range budget"

→ Analyzer: 9 searches across products, materials, certifications, companies, taxonomy
→ Executor: 16 products, 16 materials, 13 certs, 16 companies (parallel, ~3s)
→ Synthesizer: 5 ranked recommendations

#1  Symphonia — Altro USA (92% fit)
    "Altro is the gold standard for healthcare flooring with proven
    antimicrobial properties and slip-resistance specifically engineered
    for hospital environments."
    Certifications: FloorScore
    Materials: Solid Vinyl

#2  Castle Gate 595A — Mohawk Flooring (88% fit)
    "Sheet vinyl offers excellent value for healthcare applications with
    proven durability in high-traffic environments."
    Certifications: FloorScore, LEED Possible Points
    Materials: Solid Vinyl

Recommended next steps: request product samples and EPDs from Altro,
Mohawk, and Mannington to verify LEED credit contributions.
```

## Approach & Key Design Decisions

### Three-Stage Pipeline with Conditional Retry

```
User Query → [Analyzer] → [Executor] → [Quality Check] → [Synthesizer] → Recommendations
               LLM #1     Acelab SDK    sparse results?     LLM #2
                                              ↓
                                        [Re-Analyzer] → [Executor] (retry with broader queries)
```

**Why this pattern over a ReAct loop:**
- Predictable latency (~30 seconds) and cost (~$0.035 per query)
- Exactly 2 LLM calls (3 if retry triggers), not 6-10 in a ReAct loop
- Every decision is explicit and debuggable — no hidden tool-calling abstractions

**Why LangGraph:**
- Makes the agent architecture visible as a graph, not hidden in function chains
- Conditional retry edge was a one-line addition, not a refactor
- Shared typed state (`AgentState` TypedDict) ensures clean data flow between nodes

**Why AsyncAcelab with parallel execution:**
- 6-10 SDK calls fire concurrently via `asyncio.gather()` — ~8x faster than sequential
- `return_exceptions=True` means one failed search doesn't block the others

**Why structured output via Pydantic:**
- Both LLM calls produce JSON parsed into typed models (`SearchPlan`, `AgentResponse`)
- Fallback plans handle parse failures — the pipeline always completes

### Prompt Engineering Highlights

- **Consultant framing** — "senior architectural materials consultant" produces expert recommendations, not search result summaries
- **Specific cross-referencing** — certifications assigned per product only when they logically match the material type (FloorScore for flooring, not wall panels)
- **Anti-hallucination** — "Only recommend products from actual search results. NEVER invent products."
- **Actionable next steps** — gaps section gives architect concrete actions, not a list of failures

## Project Structure

```
├── acelab/                  # SDK (unmodified)
├── agent/
│   ├── config.py            # Env-based settings (pydantic-settings)
│   ├── llm.py               # OpenRouter client via ChatOpenAI
│   ├── models.py            # SearchPlan, Recommendation, AgentResponse
│   ├── prompts.py           # All LLM prompt templates (centralized)
│   ├── state.py             # LangGraph AgentState TypedDict
│   ├── graph.py             # Graph definition with conditional retry edge
│   └── nodes/
│       ├── analyzer.py      # LLM #1: Query → SearchPlan
│       ├── executor.py      # Parallel SDK calls → CollectedResults
│       ├── re_analyzer.py   # Broadens search on sparse results
│       └── synthesizer.py   # LLM #2: Results → Ranked Recommendations
├── templates/index.html     # Web UI (FastAPI + Jinja2)
├── main.py                  # CLI entry point
├── server.py                # FastAPI web server
├── tests/                   # Model + parser tests (21 passing)
├── DESIGN.md                # Detailed architecture document
└── .env.example             # Configuration template
```

## What I'd Improve With More Time

- **Evaluation harness** — benchmark queries with expected material types, auto-scored to measure recommendation quality across prompt iterations
- **Token/cost tracking** — log token usage per LLM call via OpenRouter response headers for cost monitoring
- **Richer cross-referencing** — use `client.deduplicate()` to verify product uniqueness and `client.taxonomy.search()` to classify each recommendation into Acelab's taxonomy
- **Caching layer** — cache SDK responses for repeated or similar queries to reduce latency and API load
- **More comprehensive tests** — integration tests with mocked SDK responses covering the full pipeline, and LLM output quality assertions
- **Streaming responses** — use LangGraph's streaming support + SSE to show progress stage-by-stage ("Analyzing... → Searching... → Recommending...") instead of a 30-second wait