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

## Evaluation

After the take-home submission, I built a deterministic evaluation framework on the `eval-framework` branch to measure pipeline quality without relying on LLM-as-judge opinion. The goal: separate verifiable structural correctness from harder-to-measure semantic quality.

### Methodology

The framework runs against typed Pydantic state from each pipeline pass, computing three independent checks:

1. **Synthesis grounding rate** — every `Recommendation.product_id` returned by the synthesizer must appear in the executor's `CollectedResults.products` for that run. A recommended UUID that does not exist in collected results is a fabricated recommendation. This is the strongest deterministic anti-hallucination check available given the system's structured outputs.

2. **Constraint adherence** — when the benchmark query carries `required_certifications` (FDA, LEED, ADA, FloorScore, etc.), the recommendations must surface those certifications in `relevant_certifications`. Case-insensitive substring match. Only applicable to queries with explicit cert requirements (4 of 25 in the current benchmark).

3. **Category recall@5** — when the benchmark query specifies `preferred_materials` (vinyl, rubber, wood, terrazzo, etc.), at least one expected material must appear in the top-5 recommendations' `relevant_materials`. Case-insensitive substring match.

The benchmark suite (`eval/benchmarks/queries.jsonl`) contains 25 architect project descriptions across healthcare, commercial, hospitality, residential, educational, institutional, and adversarial categories. Adversarial queries (underwater spaceship cafeteria, FAKE-MATERIAL-X compliance) test whether the system fabricates product UUIDs when the input is nonsensical.

### Results from the latest full run (25 queries)

Run artifact: `eval/runs/20260503_143718/results.json`

| Metric | Value | Notes |
|---|---|---|
| Runs completed | 25 / 25 | No pipeline failures |
| Synthesis grounding rate (avg) | **1.00** | No UUID fabrication on any query, including adversarial |
| Constraint adherence rate (avg) | 0.625 | 4 applicable queries; FDA-required query failed (0 / 1), LEED/ADA queries passed |
| Category recall@5 (avg) | 0.513 | 13 applicable queries; half the time the expected material appears in top-5 |
| Adversarial fabrication count | 0 / 2 | Pipeline did not invent UUIDs for nonsense queries |
| Average latency | 26.4 s / query | |
| Cost (full benchmark run) | ~$0.90 | OpenRouter, Claude Sonnet |

### Notable findings

- **The pipeline never fabricates product UUIDs**, even on adversarial inputs. The structured-output design (synthesizer is constrained to a Pydantic schema, recommendations must reference `product_id` strings, prompts forbid invention) is doing its job at the structural level.
- **Constraint adherence is materially worse than grounding.** The FDA-required restaurant kitchen query (q010) returned five recommendations all with empty `relevant_certifications` lists. The synthesizer either could not find FDA-certified products in the catalog or did not prioritize surfacing the cert. This is a real production-quality issue the deterministic check exposes.
- **UUID grounding does not catch semantic fabrication.** Both adversarial queries (q024 underwater spaceship, q025 fake material/cert) returned 5 recommendations of REAL Acelab products with confident project summaries treating the absurd inputs as legitimate. The synthesizer pattern-matched "underwater + spaceship" to "marine + futuristic" and generated 0.78–0.92 confidence scores. This motivates a future semantic-relevance check (LLM-as-judge or embedding-based) layered on top of the deterministic checks.

### Design pivot worth noting

An earlier version of the framework on the BioReason sister project tried to compare cited references against a static ground-truth set populated by direct API search on cleaned queries. That approach failed because the agent's LLM dynamically reformulates queries before calling tools, so the ground-truth set diverged from what the pipeline actually saw. The pivot — comparing cited UUIDs against the run's actual `raw_tool_data` — gave a direct hallucination signal without external ground truth dependencies. This pattern is reused here: grounding is computed against the run's own `CollectedResults`, not an external oracle.

### How to run

```bash
uv run python -m eval.run_suite                  # full 25-query benchmark
uv run python -m eval.run_suite --limit 3        # smoke test on first 3
uv run python -m eval.run_suite --ids q001,q010  # specific queries
```

Each run writes a timestamped artifact to `eval/runs/{timestamp}/results.json` containing per-query scores, full `AgentResponse` payloads, and aggregate metrics for run-to-run comparison.

## What I'd Improve With More Time

* **Semantic-relevance check** — extend the eval framework with an LLM-as-judge or embedding-based check that scores whether recommendations are semantically appropriate for the query, not just structurally grounded. Adversarial benchmark queries showed UUID grounding cannot catch this class of failure.
* **Token/cost tracking** — log token usage per LLM call via OpenRouter response headers for cost monitoring
* **Richer cross-referencing** — use `client.deduplicate()` to verify product uniqueness and `client.taxonomy.search()` to classify each recommendation into Acelab's taxonomy
* **Caching layer** — cache SDK responses for repeated or similar queries to reduce latency and API load
* **More comprehensive tests** — integration tests with mocked SDK responses covering the full pipeline, and LLM output quality assertions
* **Streaming responses** — use LangGraph's streaming support + SSE to show progress stage-by-stage ("Analyzing... → Searching... → Recommending...") instead of a 30-second wait