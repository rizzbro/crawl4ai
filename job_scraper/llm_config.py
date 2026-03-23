"""LLM provider configuration with support for local (Ollama) and cloud providers."""
import os
from crawl4ai import LLMConfig


PROVIDERS = {
    "ollama": {
        "provider": "ollama/llama3.2",
        "api_token": "no-token-needed",
        "description": "Local Ollama (no API key required)",
    },
    "ollama-mistral": {
        "provider": "ollama/mistral",
        "api_token": "no-token-needed",
        "description": "Local Ollama with Mistral",
    },
    "anthropic": {
        "provider": "anthropic/claude-sonnet-4-5",
        "api_token_env": "ANTHROPIC_API_KEY",
        "description": "Anthropic Claude Sonnet 4.5",
    },
    "anthropic-haiku": {
        "provider": "anthropic/claude-haiku-4-5-20251001",
        "api_token_env": "ANTHROPIC_API_KEY",
        "description": "Anthropic Claude Haiku 4.5 (faster, cheaper)",
    },
    "openai": {
        "provider": "openai/gpt-4o-mini",
        "api_token_env": "OPENAI_API_KEY",
        "description": "OpenAI GPT-4o Mini",
    },
    "gemini": {
        "provider": "gemini/gemini-2.0-flash",
        "api_token_env": "GEMINI_API_KEY",
        "description": "Google Gemini 2.0 Flash",
    },
}


def get_llm_config(provider_name: str, model_override: str = None) -> LLMConfig:
    """
    Build an LLMConfig for the given provider name.

    Args:
        provider_name: One of the keys in PROVIDERS (e.g. 'ollama', 'anthropic').
        model_override: Optional full model string like 'ollama/llama3.1'.

    Returns:
        Configured LLMConfig instance.
    """
    if model_override:
        # Determine token from override prefix
        prefix = model_override.split("/")[0]
        env_map = {
            "anthropic": "ANTHROPIC_API_KEY",
            "openai": "OPENAI_API_KEY",
            "gemini": "GEMINI_API_KEY",
        }
        token = os.getenv(env_map.get(prefix, ""), "no-token-needed")
        return LLMConfig(provider=model_override, api_token=token or "no-token-needed")

    if provider_name not in PROVIDERS:
        available = ", ".join(PROVIDERS.keys())
        raise ValueError(
            f"Unknown provider '{provider_name}'. Available: {available}"
        )

    cfg = PROVIDERS[provider_name]
    token = cfg.get("api_token", None)

    if token is None:
        env_var = cfg["api_token_env"]
        token = os.getenv(env_var)
        if not token:
            raise EnvironmentError(
                f"Provider '{provider_name}' requires the environment variable "
                f"'{env_var}' to be set."
            )

    return LLMConfig(provider=cfg["provider"], api_token=token)


def list_providers() -> str:
    """Return a formatted list of available providers."""
    lines = []
    for key, val in PROVIDERS.items():
        lines.append(f"  {key:<20} – {val['description']}")
    return "\n".join(lines)
