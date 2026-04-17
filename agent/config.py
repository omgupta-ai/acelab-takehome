"""Agent configuration — all settings loaded from environment variables."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with env-based configuration."""

    # Acelab SDK
    acelab_api_key: str
    acelab_base_url: str

    # OpenRouter LLM
    openrouter_api_key: str

    # Agent tuning
    llm_model: str = "anthropic/claude-sonnet-4"
    llm_temperature: float = 0.2
    similarity_threshold: float = 0.4
    max_results_per_search: int = 8

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
