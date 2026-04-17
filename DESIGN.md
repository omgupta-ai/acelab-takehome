# Design Document: Acelab Material Recommendation Agent

## Architecture Overview

The agent uses a three-stage pipeline built with LangGraph, enhanced with a conditional retry loop for result quality assurance. Each stage has a single responsibility, and state flows through a shared `AgentState` TypedDict.

```
┌─────────────────────────────────────────────────────────┐
│                     User Query                          │
│  "Hospital corridor, LEED Silver, mid-range budget"     │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────┐
│                  ANALYZER NODE                           │
│                  (LLM Call #1)                            │
│                                                          │
│  Decomposes the request into a SearchPlan:               │
│  ┌────────────────────────────────────────────┐          │
│  │ product:  "hospital sheet vinyl flooring"  │          │
│  │ material: "antimicrobial vinyl"            │          │
│  │ cert:     "LEED Silver"                    │          │
│  │ cert:     "FloorScore"                     │          │
│  │ company:  "Altro healthcare flooring"      │          │
│  │ taxonomy: "commercial vinyl flooring"      │          │
│  └────────────────────────────────────────────┘          │
└──────────────────────┬───────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────┐
│                  EXECUTOR NODE                           │
│              (Parallel SDK Calls)                         │
│                                                          │
│  asyncio.gather() fires ALL searches concurrently:       │
│                                                          │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐       │
│  │Products │ │Materials│ │  Certs  │ │Companies│       │
│  │ 16 hits │ │ 16 hits │ │ 15 hits │ │  8 hits │       │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘       │
│                                                          │
│  → Filter by similarity threshold (≥0.4)                 │
│  → Deduplicate by product_id / resource id               │
│  → Log each search result count                          │
└──────────────────────┬───────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────┐
│              QUALITY CHECK (Conditional Edge)             │
│                                                          │
│  if products < 2 AND retry_count < 1:                    │
│    → Route to RE-ANALYZER (broaden search strategy)      │
│    → Then back to EXECUTOR                               │
│  else:                                                   │
│    → Route to SYNTHESIZER                                │
└──────────┬──────────────────────┬────────────────────────┘
           │                      │
     [sparse results]       [sufficient results]
           │                      │
           ▼                      ▼
┌─────────────────────┐  ┌────────────────────────────────┐
│  RE-ANALYZER NODE   │  │       SYNTHESIZER NODE         │
│  (LLM Call #1b)     │  │        (LLM Call #2)           │
│                     │  │                                │
│  Generates broader  │  │  Cross-references products     │
│  search queries     │  │  with certifications and       │
│  with different     │  │  materials. Produces:          │
│  terms + missing    │  │                                │
│  search types       │  │  ┌───────────────────────┐     │
│         │           │  │  │ #1 Symphonia (92%)    │     │
│         └───────┐   │  │  │ #2 Castle Gate (88%)  │     │
│            back │   │  │  │ #3 Assurance III (85%)│     │
│          to     │   │  │  └───────────────────────┘     │
│        executor │   │  │                                │
└─────────────────┘   │  │  + Search strategy summary     │
                      │  │  + Actionable next steps       │
                      │  └────────────────────────────────┘
                      │
                      ▼
                    [END]
```

## Why This Architecture

### Two LLM Calls, Not a ReAct Loop

A ReAct agent (think → act → observe → repeat) would call the LLM 6-10 times per query. This creates:
- Unpredictable latency (each loop adds 3-5 seconds)
- Higher cost ($0.15+ per query vs our ~$0.035)
- More failure points (any loop iteration can produce bad output)

Our "Plan → Execute → Synthesize" pattern delivers the same multi-step reasoning the challenge requires while keeping latency predictable (~30 seconds) and cost fixed at exactly 2 LLM calls.

### Conditional Retry for Quality Assurance

Rather than accepting sparse results, the agent can detect when a search strategy underperformed (fewer than 2 products found) and automatically broaden its approach. This adds:
- **Resilience** — vague or niche queries that initially return few results get a second chance with broader terms
- **Coverage** — the re-analyzer adds missing search types (e.g., if no certifications were searched initially)
- **Bounded cost** — max 1 retry (3 LLM calls worst case), preventing infinite loops

### LangGraph Over Plain Functions

A simple `analyze() → execute() → synthesize()` chain would work, but LangGraph provides:
- **Visible graph structure** — the architecture is inspectable, not hidden in function calls
- **Shared typed state** — `AgentState` TypedDict ensures data flows cleanly between nodes
- **Conditional routing** — the retry edge is a one-line graph change, not a refactor
- **Extensibility** — adding more quality checks or feedback loops is straightforward

We chose LangGraph specifically over LangChain's `AgentExecutor` because the latter hides tool-calling logic behind abstractions. In our pipeline, every prompt, every SDK call, and every filtering decision is explicit in the node functions.

### AsyncAcelab with Parallel Execution

The executor fires 6-10 SDK calls concurrently via `asyncio.gather()`. For a typical query:
- Sequential: ~8 calls × 1.5s = 12 seconds
- Parallel: ~1.5 seconds (bounded by slowest call)

This is a ~8x improvement. We use `return_exceptions=True` so one failed search doesn't block the others.

### Structured Output via Pydantic

Both LLM calls produce JSON parsed into typed Pydantic models (`SearchPlan`, `AgentResponse`). This gives us:
- Validation at every stage (malformed LLM output caught immediately)
- Type safety throughout the pipeline
- Self-documenting data flow (read the models, understand the system)

Fallback plans handle parse failures so the pipeline always completes.

## Key Design Decisions

### Prompt Engineering as Product Design

The synthesizer prompt is the most carefully designed component. Key choices:
- **Consultant framing** — "You are a senior architectural materials consultant" produces recommendations that sound like expert advice, not search result summaries
- **Specific cross-referencing** — certifications are only assigned to products when they logically apply to that product's material type and application. A rubber tile and sheet vinyl may qualify for different standards.
- **Variety instruction** — the agent recommends across different material types and suppliers when available, not 5 versions of the same thing
- **Actionable next steps** — instead of listing failures, the gaps section provides what an architect would actually do next ("request EPDs from top suppliers")
- **Anti-hallucination guard** — "Only recommend products from actual search results. NEVER invent products."

### Similarity Threshold Filtering

Results below 0.4 similarity are filtered out before synthesis. This prevents the LLM from recommending irrelevant products that happened to appear in broad semantic searches. The threshold is configurable via `SIMILARITY_THRESHOLD` env var.

### Product Deduplication

Multiple searches often return the same product (e.g., "hospital vinyl flooring" and "commercial sheet vinyl" may overlap). The executor deduplicates by `product_id` for products and by `id` for materials, certifications, and companies before passing to the synthesizer.

## Tradeoffs

| Decision | Benefit | Cost |
|---|---|---|
| 2 LLM calls + optional retry | Predictable, fast, cheap | Can't adapt search strategy based on intermediate results (except via retry) |
| JSON output (not tool calling) | Simple, works with any model | Must handle JSON parse failures manually |
| Parallel SDK calls | ~8x faster execution | All searches planned upfront — can't do sequential dependent searches |
| Single synthesizer call | One coherent recommendation set | Limited by context window for very large result sets |
| Max 1 retry | Bounds cost at 3 LLM calls | Very niche queries may still return sparse results |

## Technology Choices

| Component | Choice | Rationale |
|---|---|---|
| **Orchestration** | LangGraph | Typed state graph, conditional routing, extensible |
| **LLM** | Claude Sonnet 4 via OpenRouter | Strong structured output, cost-effective |
| **SDK client** | AsyncAcelab | Required for parallel execution |
| **Models** | Pydantic v2 | Validation, serialization, type safety |
| **Web UI** | FastAPI + Jinja2 | Minimal, async-native, no JS framework needed |
| **Config** | pydantic-settings | Type-safe env loading with defaults |