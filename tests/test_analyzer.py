"""Tests for the analyzer node's parsing logic.

These test the _parse_plan and _fallback_plan functions without needing
real LLM calls or API keys.
"""

import json

import pytest

from agent.nodes.analyzer import _fallback_plan, _parse_plan
from agent.models import SearchType


class TestParsePlan:
    def test_clean_json(self):
        raw = json.dumps(
            {
                "project_understanding": "A hospital needs flooring.",
                "requirements": ["durability", "LEED"],
                "searches": [
                    {
                        "search_type": "product",
                        "query": "hospital flooring",
                        "reasoning": "Main product search",
                    },
                    {
                        "search_type": "certification",
                        "query": "LEED Silver",
                        "reasoning": "Check LEED certs",
                    },
                ],
            }
        )
        plan = _parse_plan(raw)
        assert len(plan.searches) == 2
        assert plan.searches[0].search_type == SearchType.PRODUCT
        assert plan.searches[1].search_type == SearchType.CERTIFICATION

    def test_json_with_code_fences(self):
        raw = '```json\n{"project_understanding":"test","requirements":[],"searches":[{"search_type":"material","query":"vinyl","reasoning":"test"}]}\n```'
        plan = _parse_plan(raw)
        assert len(plan.searches) == 1

    def test_json_with_generic_fences(self):
        raw = '```\n{"project_understanding":"test","requirements":[],"searches":[{"search_type":"product","query":"tile","reasoning":"test"}]}\n```'
        plan = _parse_plan(raw)
        assert plan.searches[0].query == "tile"

    def test_json_with_surrounding_text(self):
        raw = 'Here is the plan:\n```json\n{"project_understanding":"test","requirements":["a"],"searches":[{"search_type":"company","query":"Armstrong","reasoning":"test"}]}\n```\nLet me know if you need changes.'
        plan = _parse_plan(raw)
        assert plan.searches[0].search_type == SearchType.COMPANY

    def test_invalid_json_raises(self):
        with pytest.raises(Exception):
            _parse_plan("This is not JSON at all.")

    def test_missing_fields_raises(self):
        with pytest.raises(Exception):
            _parse_plan('{"project_understanding": "test"}')


class TestFallbackPlan:
    def test_fallback_produces_valid_plan(self):
        plan = _fallback_plan("hospital corridor flooring")
        assert len(plan.searches) >= 2
        types = {s.search_type for s in plan.searches}
        assert SearchType.PRODUCT in types
        assert SearchType.MATERIAL in types

    def test_fallback_truncates_long_query(self):
        long_query = "x" * 200
        plan = _fallback_plan(long_query)
        assert len(plan.searches[0].query) <= 80

    def test_fallback_preserves_original_in_requirements(self):
        plan = _fallback_plan("my specific query")
        assert "my specific query" in plan.requirements
