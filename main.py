"""CLI entry point for the Acelab Material Recommendation Agent.

Usage:
    uv run python main.py
    uv run python main.py "high-traffic hospital corridor, LEED Silver, mid-range budget"
"""

from __future__ import annotations

import asyncio
import logging
import sys

from agent.graph import agent
from agent.models import AgentResponse

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
# Quiet noisy libraries
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

logger = logging.getLogger("acelab-agent")

# ---------------------------------------------------------------------------
# Pretty-print helpers
# ---------------------------------------------------------------------------

DIVIDER = "=" * 64


def _print_response(resp: AgentResponse) -> None:
    """Pretty-print the agent's recommendations to stdout."""

    print(f"\n{DIVIDER}")
    print("  PROJECT SUMMARY")
    print(DIVIDER)
    print(f"\n{resp.project_summary}\n")

    print(DIVIDER)
    print("  RECOMMENDATIONS")
    print(DIVIDER)

    if not resp.recommendations:
        print("\n  No recommendations could be generated.\n")
    else:
        for rec in resp.recommendations:
            print(f"\n  #{rec.rank}  {rec.product_name}")
            if rec.supplier:
                print(f"      Supplier:       {rec.supplier}")
            if rec.product_id:
                print(f"      Product ID:     {rec.product_id}")
            print(f"      Confidence:     {rec.score:.0%}")
            print(f"      Why:            {rec.why_recommended}")
            if rec.relevant_certifications:
                print(f"      Certifications: {', '.join(rec.relevant_certifications)}")
            if rec.relevant_materials:
                print(f"      Materials:      {', '.join(rec.relevant_materials)}")
            if rec.potential_concerns:
                print(f"      ⚠ Concerns:    {rec.potential_concerns}")

    print(f"\n{DIVIDER}")
    print("  SEARCH STRATEGY")
    print(DIVIDER)
    print(f"\n{resp.search_strategy_summary}\n")

    if resp.gaps_and_caveats:
        print(DIVIDER)
        print("  RECOMMENDED NEXT STEPS")
        print(DIVIDER)
        print(f"\n{resp.gaps_and_caveats}\n")



# Main

async def run(query: str) -> AgentResponse:
    """Run the agent pipeline and return the response."""
    logger.info("Starting agent for query: %s", query[:120])

    result = await agent.ainvoke({"user_query": query})

    if result.get("error"):
        logger.error("Agent error: %s", result["error"])

    return result.get("response")


async def main() -> None:
    """CLI entrypoint — accepts query as arg or via interactive prompt."""

    print("\n🏗️  Acelab Material Recommendation Agent")
    print(DIVIDER)

    # Accept query from command-line arg or interactive prompt
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = input("\nDescribe your project:\n> ").strip()

    if not query:
        print("No query provided. Exiting.")
        return

    print("\n⏳ Analyzing your request and searching the Acelab catalog...\n")

    response = await run(query)

    if response:
        _print_response(response)
    else:
        print("\n❌ Agent failed to produce recommendations. Check logs above for details.")


if __name__ == "__main__":
    asyncio.run(main())
