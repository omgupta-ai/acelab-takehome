"""Re-analyzer node — broadens search strategy when initial results are sparse.

This node is triggered by a conditional edge when the executor finds too few
products or is missing entire search categories. It generates a new, broader
SearchPlan with different queries and any missing search types.
"""

from __future__ import annotations

import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage

from agent.llm import get_llm
from agent.models import SearchPlan, SearchQuery, SearchType
from agent.prompts import RE_ANALYZER_SYSTEM_PROMPT, RE_ANALYZER_USER_TEMPLATE
from agent.state import AgentState

logger = logging.getLogger(__name__)


def _parse_plan(raw_text: str) -> SearchPlan:
    """Parse LLM response into a SearchPlan."""
    text = raw_text.strip()
    if "```" in text:
        if "```json" in text:
            text = text.split("```json", 1)[1]
        else:
            text = text.split("```", 1)[1]
        text = text.split("```", 1)[0]
    text = text.strip()
    return SearchPlan.model_validate(json.loads(text))


def _fallback_broad_plan(user_query: str) -> SearchPlan:
    """Fallback broad plan when re-analysis LLM call fails."""
    short = user_query.split(",")[0].strip()[:50]
    return SearchPlan(
        project_understanding=f"Broadened search for: {short}",
        requirements=[user_query],
        searches=[
            SearchQuery(
                search_type=SearchType.PRODUCT,
                query=short,
                reasoning="Broadened: general product search",
            ),
            SearchQuery(
                search_type=SearchType.MATERIAL,
                query="flooring",
                reasoning="Broadened: general material search",
            ),
            SearchQuery(
                search_type=SearchType.CERTIFICATION,
                query="LEED",
                reasoning="Broadened: common certification",
            ),
            SearchQuery(
                search_type=SearchType.COMPANY,
                query="flooring manufacturer",
                reasoning="Broadened: general company search",
            ),
        ],
    )


async def re_analyze_node(state: AgentState) -> dict:
    """Generate a broader search plan based on sparse initial results.

    Returns:
        Dict with updated 'search_plan' and incremented 'retry_count'.
    """
    user_query = state["user_query"]
    collected = state["collected_results"]
    previous_plan = state["search_plan"]
    retry_count = state.get("retry_count", 0)

    logger.info(
        "Re-analyzing: previous results had %d products, %d materials, %d certs. Retry #%d",
        len(collected.products),
        len(collected.materials),
        len(collected.certifications),
        retry_count + 1,
    )

    # Format previous searches for context
    previous_searches = "\n".join(
        f"  - [{s.search_type.value}] '{s.query}'" for s in previous_plan.searches
    )
    search_log = "\n".join(f"  - {log}" for log in collected.search_log)

    llm = get_llm()
    messages = [
        SystemMessage(content=RE_ANALYZER_SYSTEM_PROMPT),
        HumanMessage(
            content=RE_ANALYZER_USER_TEMPLATE.format(
                user_query=user_query,
                search_log=search_log,
                previous_searches=previous_searches,
            )
        ),
    ]

    try:
        response = await llm.ainvoke(messages)
        plan = _parse_plan(response.content)
        logger.info(
            "Broadened plan: %d searches across types %s",
            len(plan.searches),
            sorted({s.search_type.value for s in plan.searches}),
        )
    except Exception:
        logger.exception("Re-analyzer failed — using fallback broad plan")
        plan = _fallback_broad_plan(user_query)

    return {"search_plan": plan, "retry_count": retry_count + 1}