"""Typed models for the agent pipeline.

Covers every stage: analysis → execution → synthesis.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Search Plan (output of Analyzer node)
# ---------------------------------------------------------------------------


class SearchType(str, Enum):
    """The Acelab SDK endpoint to target."""

    PRODUCT = "product"
    MATERIAL = "material"
    CERTIFICATION = "certification"
    COMPANY = "company"
    TAXONOMY = "taxonomy"


class SearchQuery(BaseModel):
    """A single search the agent wants to execute."""

    search_type: SearchType
    query: str = Field(..., description="Short search query (1-6 words work best)")
    reasoning: str = Field(..., description="Why this search is needed for the project")


class SearchPlan(BaseModel):
    """The agent's decomposed search strategy for an architect's request."""

    project_understanding: str = Field(
        ..., description="2-3 sentence summary of what the architect needs"
    )
    requirements: list[str] = Field(
        ..., description="Extracted requirements: durability, certifications, aesthetics, budget, etc."
    )
    searches: list[SearchQuery] = Field(
        ..., description="4-10 planned searches across multiple endpoint types"
    )


# ---------------------------------------------------------------------------
# Collected Results (output of Executor node)
# ---------------------------------------------------------------------------


class CollectedResults(BaseModel):
    """All results gathered from parallel SDK calls."""

    products: list[dict] = Field(default_factory=list)
    materials: list[dict] = Field(default_factory=list)
    certifications: list[dict] = Field(default_factory=list)
    companies: list[dict] = Field(default_factory=list)
    taxonomies: list[dict] = Field(default_factory=list)
    search_log: list[str] = Field(
        default_factory=list,
        description="Human-readable log of what was searched and what was found",
    )


# ---------------------------------------------------------------------------
# Final Recommendation (output of Synthesizer node)
# ---------------------------------------------------------------------------


class Recommendation(BaseModel):
    """A single ranked product recommendation with reasoning."""

    rank: int = Field(..., ge=1)
    product_name: str
    supplier: str | None = None
    product_id: str | None = None
    score: float = Field(..., ge=0.0, le=1.0, description="Agent confidence score")
    why_recommended: str = Field(
        ..., description="Reasoning tied to specific project requirements"
    )
    relevant_certifications: list[str] = Field(default_factory=list)
    relevant_materials: list[str] = Field(default_factory=list)
    potential_concerns: str | None = Field(
        default=None, description="Honest caveats or limitations"
    )


class AgentResponse(BaseModel):
    """Complete agent response returned to the user."""

    project_summary: str
    recommendations: list[Recommendation] = Field(default_factory=list)
    search_strategy_summary: str = Field(
        ..., description="What the agent searched and key findings"
    )
    gaps_and_caveats: str = Field(
        ..., description="What wasn't found, limitations, honest assessment"
    )
