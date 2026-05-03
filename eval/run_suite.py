"""
Run the Acelab material recommendation pipeline against the benchmark suite,
compute deterministic grounding metrics, and persist a timestamped run artifact.

Usage:
    uv run python -m eval.run_suite              # run all queries
    uv run python -m eval.run_suite --limit 3    # smoke test on first 3
    uv run python -m eval.run_suite --ids q001,q010,q024
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Path setup so we can import project root modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.graph import build_graph
from eval.grounding import (
    synthesis_grounding_rate,
    constraint_adherence,
    category_recall_at_k,
)

QUERIES_PATH = Path(__file__).parent / "benchmarks" / "queries.jsonl"
RUNS_DIR = Path(__file__).parent / "runs"


def load_queries() -> list[dict]:
    queries = []
    with open(QUERIES_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                queries.append(json.loads(line))
    return queries


async def run_pipeline(query_text: str):
    """Invoke the agent graph and return the final state."""
    graph = build_graph()
    initial_state = {"user_query": query_text, "retry_count": 0}
    final_state = await graph.ainvoke(initial_state)
    return final_state


def serialize_response(response) -> dict:
    """Convert AgentResponse Pydantic to a JSON-serializable dict."""
    return response.model_dump() if response else None


def serialize_collected(collected) -> dict:
    """Compact summary of CollectedResults for the run artifact."""
    if not collected:
        return None
    return {
        "products_count": len(collected.products),
        "materials_count": len(collected.materials),
        "certifications_count": len(collected.certifications),
        "companies_count": len(collected.companies),
        "taxonomies_count": len(collected.taxonomies),
        "product_ids": [p.get("product_id") for p in collected.products if p.get("product_id")],
    }


async def run_one_query(q: dict) -> dict:
    qid = q["id"]
    query_text = q["query"]
    expected_attributes = q.get("expected_attributes", {})

    print(f"\n[{qid}] {query_text[:70]}...")
    t0 = time.time()

    try:
        final_state = await run_pipeline(query_text)
    except Exception as e:
        print(f"  PIPELINE FAILED: {e}")
        return {
            "id": qid,
            "query": query_text,
            "category": q.get("category"),
            "status": "pipeline_failed",
            "error": str(e),
            "elapsed_s": round(time.time() - t0, 2),
        }

    elapsed = round(time.time() - t0, 2)

    response = final_state.get("response")
    collected = final_state.get("collected_results")

    if not response or not collected:
        return {
            "id": qid,
            "query": query_text,
            "category": q.get("category"),
            "status": "no_output",
            "error": final_state.get("error"),
            "elapsed_s": elapsed,
        }

    # Run all three grounding checks
    synth_check = synthesis_grounding_rate(response, collected)
    constraint_check = constraint_adherence(response, expected_attributes)
    recall_check = category_recall_at_k(response, expected_attributes, k=5)

    result = {
        "id": qid,
        "query": query_text,
        "category": q.get("category"),
        "status": "ok",
        "elapsed_s": elapsed,
        "expected_attributes": expected_attributes,
        "response": serialize_response(response),
        "collected_summary": serialize_collected(collected),
        "synthesis_grounding": synth_check,
        "constraint_adherence": constraint_check,
        "category_recall": recall_check,
    }

    # Compact one-line console summary
    cr = recall_check
    cr_str = (
        f"{cr['recalled_count']}/{cr['expected_count']} (rate {cr['rate']:.2f})"
        if cr.get("applicable")
        else "n/a"
    )
    ca = constraint_check
    ca_str = (
        f"{ca['matched_certs']}/{ca['required_certs']} (rate {ca['rate']:.2f})"
        if ca.get("applicable")
        else "n/a"
    )
    print(
        f"  Synth grounding: {synth_check['grounded']}/{synth_check['cited']} "
        f"(rate {synth_check['rate']:.2f})  |  "
        f"Constraint: {ca_str}  |  "
        f"Recall@5: {cr_str}  |  "
        f"{elapsed}s"
    )
    return result


def aggregate(results: list[dict]) -> dict:
    """Compute summary stats across all successful results."""
    ok = [r for r in results if r.get("status") == "ok"]
    if not ok:
        return {"runs": len(results), "ok": 0}

    def avg(vals):
        return round(sum(vals) / len(vals), 3) if vals else None

    synth_rates = [r["synthesis_grounding"]["rate"] for r in ok]
    elapsed = [r["elapsed_s"] for r in ok]

    constraint_rates = [
        r["constraint_adherence"]["rate"]
        for r in ok
        if r["constraint_adherence"].get("applicable")
    ]
    recall_rates = [
        r["category_recall"]["rate"]
        for r in ok
        if r["category_recall"].get("applicable")
    ]

    # Adversarial-specific check: did pipeline fabricate when query was nonsense?
    adversarial_results = [r for r in ok if r.get("category") == "adversarial"]
    adversarial_fabrications = [
        r for r in adversarial_results if r["synthesis_grounding"]["rate"] < 1.0
    ]

    return {
        "runs": len(results),
        "ok": len(ok),
        "failed": len(results) - len(ok),
        "avg_synthesis_grounding_rate": avg(synth_rates),
        "avg_constraint_adherence_rate": avg(constraint_rates),
        "avg_category_recall_at_5": avg(recall_rates),
        "constraint_applicable_count": len(constraint_rates),
        "recall_applicable_count": len(recall_rates),
        "adversarial_runs": len(adversarial_results),
        "adversarial_fabrication_count": len(adversarial_fabrications),
        "avg_elapsed_s": avg(elapsed),
    }


async def main_async():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="Run only the first N queries.")
    parser.add_argument("--ids", type=str, default=None,
                        help="Comma-separated query IDs to run (e.g. q001,q010).")
    args = parser.parse_args()

    queries = load_queries()
    if args.ids:
        wanted = set(args.ids.split(","))
        queries = [q for q in queries if q["id"] in wanted]
    elif args.limit:
        queries = queries[: args.limit]

    print(f"Running {len(queries)} queries.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RUNS_DIR / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for q in queries:
        results.append(await run_one_query(q))

    summary = aggregate(results)

    with open(run_dir / "results.json", "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)

    print("\n" + "=" * 60)
    print("AGGREGATE")
    print("=" * 60)
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"\nResults written to {run_dir}/results.json")


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()