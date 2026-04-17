"""LangGraph state definition for the material recommendation agent."""

from typing import TypedDict

from agent.models import AgentResponse, CollectedResults, SearchPlan


class AgentState(TypedDict, total=False):
    """State passed between graph nodes.

    Fields are populated progressively as nodes execute:
      analyze → search_plan
      execute → [check quality] → synthesize OR re_analyze → execute
    """

    user_query: str
    search_plan: SearchPlan
    collected_results: CollectedResults
    response: AgentResponse
    error: str
    retry_count: int

