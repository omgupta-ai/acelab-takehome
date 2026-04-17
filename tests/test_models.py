"""Tests for agent Pydantic models."""

import json

import pytest

from agent.models import (
    AgentResponse,
    CollectedResults,
    Recommendation,
    SearchPlan,
    SearchQuery,
    SearchType,
)


# ---------------------------------------------------------------------------
# SearchPlan
# ---------------------------------------------------------------------------


class TestSearchPlan:
    def test_valid_plan(self):
        plan = SearchPlan(
            project_understanding="Hospital corridor needing durable, hygienic flooring.",
            requirements=["durability", "infection control", "LEED Silver"],
            searches=[
                SearchQuery(
                    search_type=SearchType.PRODUCT,
                    query="commercial vinyl flooring",
                    reasoning="Primary flooring product search",
                ),
                SearchQuery(
                    search_type=SearchType.CERTIFICATION,
                    query="LEED",
                    reasoning="Verify LEED certifications",
                ),
            ],
        )
        assert len(plan.searches) == 2
        assert plan.searches[0].search_type == SearchType.PRODUCT

    def test_plan_from_json(self):
        raw = json.dumps(
            {
                "project_understanding": "Test project",
                "requirements": ["req1"],
                "searches": [
                    {
                        "search_type": "material",
                        "query": "vinyl tile",
                        "reasoning": "test",
                    }
                ],
            }
        )
        plan = SearchPlan.model_validate_json(raw)
        assert plan.searches[0].search_type == SearchType.MATERIAL

    def test_search_types_enum(self):
        for st in ["product", "material", "certification", "company", "taxonomy"]:
            assert SearchType(st).value == st

    def test_plan_rejects_empty_searches(self):
        """Plan should still be valid with empty searches list (analyzer fallback)."""
        plan = SearchPlan(
            project_understanding="Test",
            requirements=[],
            searches=[],
        )
        assert plan.searches == []


# ---------------------------------------------------------------------------
# Recommendation
# ---------------------------------------------------------------------------


class TestRecommendation:
    def test_valid_recommendation(self):
        rec = Recommendation(
            rank=1,
            product_name="Acme Vinyl Tile",
            supplier="Acme Corp",
            product_id="prod_123",
            score=0.85,
            why_recommended="Excellent durability for high-traffic areas.",
            relevant_certifications=["LEED", "FloorScore"],
            relevant_materials=["luxury vinyl tile"],
            potential_concerns="May require professional installation.",
        )
        assert rec.score == 0.85
        assert len(rec.relevant_certifications) == 2

    def test_score_bounds(self):
        with pytest.raises(Exception):
            Recommendation(
                rank=1,
                product_name="Test",
                score=1.5,
                why_recommended="test",
            )

    def test_minimal_recommendation(self):
        rec = Recommendation(
            rank=1,
            product_name="Test Product",
            score=0.7,
            why_recommended="It works.",
        )
        assert rec.supplier is None
        assert rec.potential_concerns is None
        assert rec.relevant_certifications == []


# ---------------------------------------------------------------------------
# AgentResponse
# ---------------------------------------------------------------------------


class TestAgentResponse:
    def test_full_response(self):
        resp = AgentResponse(
            project_summary="Hospital corridor project.",
            recommendations=[
                Recommendation(
                    rank=1,
                    product_name="Product A",
                    score=0.9,
                    why_recommended="Best fit.",
                )
            ],
            search_strategy_summary="Searched products, materials, certifications.",
            gaps_and_caveats="No exact LEED match found.",
        )
        assert len(resp.recommendations) == 1

    def test_empty_recommendations(self):
        resp = AgentResponse(
            project_summary="Test",
            recommendations=[],
            search_strategy_summary="Tried but found nothing.",
            gaps_and_caveats="Sparse catalog for this query.",
        )
        assert resp.recommendations == []

    def test_serialization_roundtrip(self):
        resp = AgentResponse(
            project_summary="Test",
            recommendations=[],
            search_strategy_summary="Summary",
            gaps_and_caveats="Gaps",
        )
        dumped = resp.model_dump_json()
        restored = AgentResponse.model_validate_json(dumped)
        assert restored.project_summary == "Test"


# ---------------------------------------------------------------------------
# CollectedResults
# ---------------------------------------------------------------------------


class TestCollectedResults:
    def test_empty_results(self):
        cr = CollectedResults()
        assert cr.products == []
        assert cr.search_log == []

    def test_with_data(self):
        cr = CollectedResults(
            products=[{"product_id": "p1", "manufacturer_product_name": "Tile A"}],
            materials=[{"id": "m1", "name": "Vinyl"}],
            search_log=["product 'tile' → 1 results"],
        )
        assert len(cr.products) == 1
        assert len(cr.search_log) == 1
