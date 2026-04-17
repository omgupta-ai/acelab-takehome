"""OpenRouter LLM client — uses ChatOpenAI with base_url override."""

from langchain_openai import ChatOpenAI

from agent.config import settings


def get_llm(temperature: float | None = None) -> ChatOpenAI:
    """Return a configured ChatOpenAI instance pointing to OpenRouter.

    Args:
        temperature: Override default temperature. Lower = more deterministic.

    Returns:
        ChatOpenAI instance ready for .invoke() or .ainvoke().
    """
    return ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=settings.openrouter_api_key,
        model=settings.llm_model,
        temperature=temperature if temperature is not None else settings.llm_temperature,
        max_tokens=4096,
        default_headers={
            "HTTP-Referer": "https://acelab-takehome.dev",
            "X-Title": "Acelab Material Recommendation Agent",
        },
    )
