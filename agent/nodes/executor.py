"""Executor node — runs all planned searches concurrently via the Acelab SDK.

This is the data-gathering stage. It takes the SearchPlan from the analyzer,
fires every query in parallel using AsyncAcelab + asyncio.gather, filters
results by similarity threshold, and deduplicates products.
"""

from __future__ import annotations

import asyncio
import logging

from acelab import AsyncAcelab
from agent.config import settings
from agent.models import CollectedResults, SearchType
from agent.state import AgentState

logger = logging.getLogger(__name__)


async def execute_node(state: AgentState) -> dict:
    """Execute all planned searches in parallel.

    Returns:
        Dict with 'collected_results' key to be merged into AgentState.
    """
    plan = state["search_plan"]
    threshold = settings.similarity_threshold
    limit = settings.max_results_per_search

    products: list[dict] = []
    materials: list[dict] = []
    certifications: list[dict] = []
    companies: list[dict] = []
    taxonomies: list[dict] = []
    search_log: list[str] = []

    async with AsyncAcelab(
        api_key=settings.acelab_api_key,
        base_url=settings.acelab_base_url,
    ) as client:

        # Build coroutines and track metadata for each
        tasks: list[asyncio.Task] = []
        task_meta: list[tuple[str, str]] = []

        for search in plan.searches:
            stype = search.search_type
            query = search.query

            if stype == SearchType.PRODUCT:
                tasks.append(client.search(query, limit=limit))
                task_meta.append(("product", query))

            elif stype == SearchType.MATERIAL:
                tasks.append(client.materials.search(query, limit=limit))
                task_meta.append(("material", query))

            elif stype == SearchType.CERTIFICATION:
                tasks.append(client.certifications.search(query, limit=limit))
                task_meta.append(("certification", query))

            elif stype == SearchType.COMPANY:
                tasks.append(client.companies.search(query, limit=limit))
                task_meta.append(("company", query))

            elif stype == SearchType.TAXONOMY:
                tasks.append(
                    client.taxonomy.search(
                        product_category_scraped=query,
                        product_description="",
                    )
                )
                task_meta.append(("taxonomy", query))

        # Fire ALL searches concurrently — exceptions captured, not raised
        logger.info("Executing %d searches in parallel", len(tasks))
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process each result
        for i, result in enumerate(results):
            search_type, query = task_meta[i]

            if isinstance(result, Exception):
                logger.warning("Search failed [%s] '%s': %s", search_type, query, result)
                search_log.append(f"FAILED: {search_type} '{query}' — {result}")
                continue

            if search_type == "product":
                filtered = [
                    r.model_dump() for r in result.results if r.similarity_score >= threshold
                ]
                products.extend(filtered)
                search_log.append(
                    f"product '{query}' → {len(filtered)} results (of {result.total_results})"
                )

            elif search_type == "material":
                filtered = [
                    r.model_dump() for r in result.results if r.similarity_score >= threshold
                ]
                materials.extend(filtered)
                search_log.append(f"material '{query}' → {len(filtered)} results")

            elif search_type == "certification":
                filtered = [
                    r.model_dump() for r in result.results if r.similarity_score >= threshold
                ]
                certifications.extend(filtered)
                search_log.append(f"certification '{query}' → {len(filtered)} results")

            elif search_type == "company":
                filtered = [
                    r.model_dump() for r in result.results if r.similarity_score >= threshold
                ]
                companies.extend(filtered)
                search_log.append(f"company '{query}' → {len(filtered)} results")

            elif search_type == "taxonomy":
                tax_items: list[dict] = []
                # Taxonomy has dual old/new structure
                for tax_result in (result.old_taxonomy, result.new_taxonomy):
                    if tax_result and tax_result.matched_taxonomy:
                        tax_items.append(tax_result.matched_taxonomy.model_dump())
                    elif tax_result and tax_result.top_candidates:
                        for candidate in tax_result.top_candidates[:3]:
                            if candidate.similarity_score >= threshold:
                                tax_items.append(candidate.model_dump())
                taxonomies.extend(tax_items)
                search_log.append(f"taxonomy '{query}' → {len(tax_items)} matches")

    # Deduplicate products by product_id
    seen_product_ids: set[str] = set()
    deduped_products: list[dict] = []
    for p in products:
        pid = p.get("product_id", "")
        if pid and pid not in seen_product_ids:
            seen_product_ids.add(pid)
            deduped_products.append(p)

    # Deduplicate materials/certifications/companies by id
    materials = _dedupe_by_key(materials, "id")
    certifications = _dedupe_by_key(certifications, "id")
    companies = _dedupe_by_key(companies, "id")
    taxonomies = _dedupe_by_key(taxonomies, "id")

    collected = CollectedResults(
        products=deduped_products,
        materials=materials,
        certifications=certifications,
        companies=companies,
        taxonomies=taxonomies,
        search_log=search_log,
    )

    logger.info(
        "Collected: %d products, %d materials, %d certs, %d companies, %d taxonomies",
        len(collected.products),
        len(collected.materials),
        len(collected.certifications),
        len(collected.companies),
        len(collected.taxonomies),
    )

    return {"collected_results": collected}


def _dedupe_by_key(items: list[dict], key: str) -> list[dict]:
    """Remove duplicate dicts based on a key field."""
    seen: set[str] = set()
    deduped: list[dict] = []
    for item in items:
        val = item.get(key, "")
        if val and val not in seen:
            seen.add(val)
            deduped.append(item)
    return deduped
