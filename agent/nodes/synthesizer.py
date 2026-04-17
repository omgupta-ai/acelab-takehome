"""Synthesizer node — ranks and explains material recommendations.

This is LLM Call #2. It receives the raw search results from the executor
and produces a ranked list of recommendations with per-product reasoning
tied to the architect's specific project requirements.
"""

from __future__ import annotations

import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage

from agent.llm import get_llm
from agent.models import AgentResponse
from agent.prompts import SYNTHESIZER_SYSTEM_PROMPT, SYNTHESIZER_USER_TEMPLATE
from agent.state import AgentState

logger = logging.getLogger(__name__)

# Cap items sent to LLM to avoid token overflow
_MAX_ITEMS_PER_CATEGORY = 15


def _fmt(items: list[dict], max_items: int = _MAX_ITEMS_PER_CATEGORY) -> str:
    """Format result dicts into readable JSON for the LLM context."""
    if not items:
        return "No results found."
    # Sort by similarity_score descending so the best results are seen first
    sorted_items = sorted(items, key=lambda x: x.get("similarity_score", 0), reverse=True)
    return json.dumps(sorted_items[:max_items], indent=2, default=str)


def _parse_response(raw_text: str) -> AgentResponse:
    """Parse LLM response into AgentResponse."""
    text = raw_text.strip()
    if "```" in text:
        if "```json" in text:
            text = text.split("```json", 1)[1]
        else:
            text = text.split("```", 1)[1]
        text = text.split("```", 1)[0]
    text = text.strip()
    return AgentResponse.model_validate(json.loads(text))


async def synthesize_node(state: AgentState) -> dict:
    """LLM Call #2: synthesize collected results into ranked recommendations.

    Returns:
        Dict with 'response' key to be merged into AgentState.
    """
    collected = state["collected_results"]
    plan = state["search_plan"]
    user_query = state["user_query"]

    logger.info("Synthesizing %d products into recommendations", len(collected.products))

    llm = get_llm(temperature=0.3)
    messages = [
        SystemMessage(content=SYNTHESIZER_SYSTEM_PROMPT),
        HumanMessage(
            content=SYNTHESIZER_USER_TEMPLATE.format(
                user_query=user_query,
                requirements=json.dumps(plan.requirements, indent=2),
                products=_fmt(collected.products),
                materials=_fmt(collected.materials),
                certifications=_fmt(collected.certifications),
                companies=_fmt(collected.companies),
                taxonomies=_fmt(collected.taxonomies),
            )
        ),
    ]

    try:
        response = await llm.ainvoke(messages)
        agent_response = _parse_response(response.content)
        logger.info(
            "Generated %d recommendations", len(agent_response.recommendations)
        )
    except Exception:
        logger.exception("Failed to parse synthesizer response — returning partial result")
        agent_response = AgentResponse(
            project_summary=plan.project_understanding,
            recommendations=[],
            search_strategy_summary="\n".join(collected.search_log),
            gaps_and_caveats=(
                "The agent collected search results successfully but failed to "
                "synthesize them into recommendations. The raw search log above "
                "shows what was found."
            ),
        )

    return {"response": agent_response}
