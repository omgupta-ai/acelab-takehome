"""Analyzer node — decomposes an architect's request into a structured search plan.

This is LLM Call #1 in the pipeline. It takes a natural-language project
description and produces a SearchPlan with multiple targeted queries across
different Acelab API endpoints.
"""

from __future__ import annotations

import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage

from agent.llm import get_llm
from agent.models import SearchPlan, SearchQuery, SearchType
from agent.prompts import ANALYZER_SYSTEM_PROMPT, ANALYZER_USER_TEMPLATE
from agent.state import AgentState

logger = logging.getLogger(__name__)


def _parse_plan(raw_text: str) -> SearchPlan:
    """Parse LLM response into a SearchPlan, handling common formatting issues."""
    text = raw_text.strip()

    # Strip markdown code fences if the model wraps its output
    if "```" in text:
        # Try ```json ... ``` first, then generic ``` ... ```
        if "```json" in text:
            text = text.split("```json", 1)[1]
        else:
            text = text.split("```", 1)[1]
        text = text.split("```", 1)[0]
    text = text.strip()

    return SearchPlan.model_validate(json.loads(text))


def _fallback_plan(user_query: str) -> SearchPlan:
    """Create a basic plan when LLM parsing fails — ensures the pipeline continues."""
    short_query = user_query[:80]
    return SearchPlan(
        project_understanding=f"Direct search for: {short_query}",
        requirements=[user_query],
        searches=[
            SearchQuery(
                search_type=SearchType.PRODUCT,
                query=short_query,
                reasoning="Fallback: direct product search",
            ),
            SearchQuery(
                search_type=SearchType.MATERIAL,
                query=short_query,
                reasoning="Fallback: direct material search",
            ),
        ],
    )


async def analyze_node(state: AgentState) -> dict:
    """LLM Call #1: decompose the architect's query into a SearchPlan.

    Returns:
        Dict with 'search_plan' key to be merged into AgentState.
    """
    user_query = state["user_query"]
    logger.info("Analyzing query: %s", user_query[:120])

    llm = get_llm()
    messages = [
        SystemMessage(content=ANALYZER_SYSTEM_PROMPT),
        HumanMessage(content=ANALYZER_USER_TEMPLATE.format(user_query=user_query)),
    ]

    try:
        response = await llm.ainvoke(messages)
        plan = _parse_plan(response.content)
        logger.info(
            "Search plan generated: %d searches across types %s",
            len(plan.searches),
            sorted({s.search_type.value for s in plan.searches}),
        )
    except Exception:
        logger.exception("Failed to parse analyzer response — using fallback plan")
        plan = _fallback_plan(user_query)

    return {"search_plan": plan}
