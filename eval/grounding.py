"""
Deterministic grounding checks for the Acelab material recommendation pipeline.

Three checks, each run against typed Pydantic state from a single pipeline run:

  1. synthesis_grounding_rate: every recommended product_id must appear in the
     executor's collected_results.products. A product_id in the response that
     does not exist in collected results is a fabricated recommendation.

  2. constraint_adherence: explicit query constraints (required_certifications,
     required_features) are checked against the recommendations. We can only
     verify constraints that have a structural representation in the response;
     this is intentionally a soft check.

  3. category_recall_at_k: of the materials/aesthetic categories we expected
     for this query, what fraction appear in the top-K recommendations.

These run alongside any LLM-based scoring to validate it, and produce defensible
metrics that don't depend on judge opinion.
"""
from __future__ import annotations

from typing import Any


def synthesis_grounding_rate(
    response: Any,
    collected_results: Any,
) -> dict:
    """How many recommended product_ids exist in the executor's collected products.

    Args:
      response: AgentResponse object with .recommendations (list of Recommendation).
      collected_results: CollectedResults object with .products (list[dict]).
        Each product dict has a 'product_id' key (per executor.py dedupe logic).

    Returns:
      dict with cited count, grounded count, rate, and list of ungrounded ids.
    """
    cited_ids = {
        r.product_id for r in response.recommendations if r.product_id
    }
    available_ids = {
        p.get("product_id") for p in collected_results.products if p.get("product_id")
    }

    if not cited_ids:
        return {
            "cited": 0,
            "grounded": 0,
            "rate": 1.0,
            "ungrounded": [],
        }

    ungrounded = sorted(cited_ids - available_ids)
    grounded = len(cited_ids) - len(ungrounded)
    return {
        "cited": len(cited_ids),
        "grounded": grounded,
        "rate": grounded / len(cited_ids),
        "ungrounded": ungrounded,
    }


def constraint_adherence(
    response: Any,
    expected_attributes: dict,
) -> dict:
    """Check whether recommendations satisfy required constraints from the query.

    Currently checks two constraint types:
      - required_certifications: at least one recommendation must list each
        required certification (case-insensitive substring match).
      - required_features: not directly verifiable from response shape, so
        this returns 'unverifiable' for now and we rely on synthesis grounding
        plus category recall as primary signals.

    Args:
      response: AgentResponse with .recommendations.
      expected_attributes: dict from queries.jsonl. May be empty for adversarial
        queries (we just return 'unverifiable' in that case).

    Returns:
      dict with per-constraint pass/fail and overall rate.
    """
    if expected_attributes.get("is_adversarial"):
        return {
            "applicable": False,
            "reason": "adversarial query — constraint check skipped",
        }

    required_certs = [
        c.lower() for c in expected_attributes.get("required_certifications", [])
    ]
    if not required_certs:
        return {
            "applicable": False,
            "reason": "no required_certifications on this query",
        }

    # Flatten all certs mentioned across recommendations
    all_recommended_certs_text = " ".join(
        c.lower()
        for r in response.recommendations
        for c in (r.relevant_certifications or [])
    )

    cert_results = []
    for required_cert in required_certs:
        matched = required_cert in all_recommended_certs_text
        cert_results.append({"cert": required_cert, "found": matched})

    matched_count = sum(1 for c in cert_results if c["found"])
    return {
        "applicable": True,
        "required_certs": len(required_certs),
        "matched_certs": matched_count,
        "rate": matched_count / len(required_certs) if required_certs else 1.0,
        "details": cert_results,
    }


def category_recall_at_k(
    response: Any,
    expected_attributes: dict,
    k: int = 5,
) -> dict:
    """Of the materials we expected, how many appear in top-K recommendations.

    Uses preferred_materials (or required_materials if present) as the expected
    set. Match is case-insensitive substring against each recommendation's
    relevant_materials list.

    Args:
      response: AgentResponse with .recommendations.
      expected_attributes: dict from queries.jsonl.
      k: how many top recommendations to consider (default 5).

    Returns:
      dict with expected count, recalled count, rate, and breakdown.
    """
    if expected_attributes.get("is_adversarial"):
        return {
            "applicable": False,
            "reason": "adversarial query — category recall skipped",
        }

    expected = expected_attributes.get(
        "required_materials"
    ) or expected_attributes.get("preferred_materials") or []

    if not expected:
        return {
            "applicable": False,
            "reason": "no material expectations on this query",
        }

    expected_lower = [m.lower().replace("_", " ") for m in expected]

    top_k = response.recommendations[:k]
    recommended_text = " ".join(
        m.lower()
        for r in top_k
        for m in (r.relevant_materials or [])
    )

    matched = []
    for exp_mat in expected_lower:
        if exp_mat in recommended_text:
            matched.append(exp_mat)

    return {
        "applicable": True,
        "k": k,
        "expected_count": len(expected),
        "recalled_count": len(matched),
        "rate": len(matched) / len(expected),
        "matched": matched,
        "missed": [m for m in expected_lower if m not in matched],
    }