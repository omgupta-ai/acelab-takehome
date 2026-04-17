"""Acelab Material Recommendation Agent.

Usage::

    from agent import agent

    result = await agent.ainvoke({"user_query": "hospital corridor flooring"})
    response = result["response"]  # AgentResponse
"""

from agent.graph import agent
from agent.models import AgentResponse, CollectedResults, Recommendation, SearchPlan

__all__ = [
    "agent",
    "AgentResponse",
    "CollectedResults",
    "Recommendation",
    "SearchPlan",
]
