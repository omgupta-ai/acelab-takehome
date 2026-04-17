"""LangGraph agent graph — the core pipeline definition.

Graph:
  analyze → execute → [quality check] → synthesize     (happy path)
                                       → re_analyze → execute  (retry path)

The conditional edge after executor checks if results are too sparse.
If fewer than 2 products found AND we haven't retried yet, the agent
broadens its search strategy and tries again. Max 1 retry to bound cost.
"""

import logging

from langgraph.graph import END, START, StateGraph

from agent.nodes.analyzer import analyze_node
from agent.nodes.executor import execute_node
from agent.nodes.re_analyzer import re_analyze_node
from agent.nodes.synthesizer import synthesize_node
from agent.state import AgentState

logger = logging.getLogger(__name__)

# Thresholds for retry decision
MIN_PRODUCTS_FOR_SYNTHESIS = 2
MAX_RETRIES = 1


def should_retry(state: AgentState) -> str:
    """Decide whether to synthesize or retry with broader searches.

    Routes to 're_analyzer' if:
      - Fewer than MIN_PRODUCTS_FOR_SYNTHESIS products found
      - AND retry_count < MAX_RETRIES (prevents infinite loops)

    Otherwise routes to 'synthesizer'.
    """
    collected = state.get("collected_results")
    retry_count = state.get("retry_count", 0)

    if collected is None:
        logger.warning("No collected results — routing to synthesizer")
        return "synthesizer"

    product_count = len(collected.products)

    if product_count < MIN_PRODUCTS_FOR_SYNTHESIS and retry_count < MAX_RETRIES:
        logger.info(
            "Sparse results (%d products) — routing to re-analyzer (retry %d/%d)",
            product_count,
            retry_count + 1,
            MAX_RETRIES,
        )
        return "re_analyzer"

    logger.info(
        "Results sufficient (%d products) — routing to synthesizer",
        product_count,
    )
    return "synthesizer"


def build_graph() -> StateGraph:
    """Build and compile the material recommendation agent graph."""
    graph = StateGraph(AgentState)

    # Register nodes
    graph.add_node("analyzer", analyze_node)
    graph.add_node("executor", execute_node)
    graph.add_node("re_analyzer", re_analyze_node)
    graph.add_node("synthesizer", synthesize_node)

    # Edges
    graph.add_edge(START, "analyzer")
    graph.add_edge("analyzer", "executor")

    # Conditional: after executor, check result quality
    graph.add_conditional_edges(
        "executor",
        should_retry,
        {
            "synthesizer": "synthesizer",
            "re_analyzer": "re_analyzer",
        },
    )

    # Re-analyzer feeds back into executor
    graph.add_edge("re_analyzer", "executor")

    # Synthesizer ends the graph
    graph.add_edge("synthesizer", END)

    return graph.compile()


# Singleton compiled graph — import and use directly
agent = build_graph()