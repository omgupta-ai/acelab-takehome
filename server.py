"""FastAPI web interface for the Acelab Material Recommendation Agent.

Usage:
    uv run uvicorn server:app --reload
    # Then open http://localhost:8000
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from agent.graph import agent
from agent.models import AgentResponse

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

logger = logging.getLogger("acelab-server")

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Acelab Material Recommendation Agent",
    description="AI-powered material recommendations for architects",
    version="0.1.0",
)

templates = Jinja2Templates(directory=Path(__file__).parent / "templates")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    """Render the main search form."""
    return templates.TemplateResponse(request=request, name="index.html")


@app.post("/recommend", response_class=HTMLResponse)
async def recommend(request: Request) -> HTMLResponse:
    """Run the agent and render recommendations."""
    form = await request.form()
    query = form.get("query", "").strip()

    if not query:
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={"error": "Please describe your project."},
        )

    logger.info("Web request: %s", query[:120])

    try:
        result = await agent.ainvoke({"user_query": query})
        response: AgentResponse | None = result.get("response")
    except Exception:
        logger.exception("Agent failed")
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={"query": query, "error": "Agent encountered an error. Please try again."},
        )

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "query": query,
            "response": response,
        },
    )


@app.post("/api/recommend")
async def api_recommend(request: Request) -> dict:
    """JSON API endpoint for programmatic access."""
    body = await request.json()
    query = body.get("query", "").strip()

    if not query:
        return {"error": "query is required"}

    result = await agent.ainvoke({"user_query": query})
    response: AgentResponse | None = result.get("response")

    if response:
        return response.model_dump()
    return {"error": "Agent failed to produce recommendations"}