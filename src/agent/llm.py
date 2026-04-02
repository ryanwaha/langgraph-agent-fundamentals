"""Model routing layer.

Provides get_compress_model() / get_fallback_compress_model() as the single point
for compression model instantiation.
Model selection is config-driven via environment variables — graph code never names a model.

Supported providers (via COMPRESS_MODEL / COMPRESS_FALLBACK_MODEL env vars):
  google_genai/<model> → init_chat_model with thinking_budget (no include_thoughts)
  ollama/<model>       → ChatOllama (local Ollama server; interface preserved for re-enablement)
  <other>/<model>      → init_chat_model generic fallback

To switch compression to a different model, change COMPRESS_MODEL in .env only.
To set thinking depth, change COMPRESS_THINKING_BUDGET (default 512 = minimum).
"""

import os

from langchain_core.language_models import BaseChatModel

COMPRESS_MODEL: str = os.getenv("COMPRESS_MODEL", "google_genai/gemini-3.1-flash-lite-preview")
COMPRESS_FALLBACK_MODEL: str = os.getenv("COMPRESS_FALLBACK_MODEL", "")
COMPRESS_THINKING_BUDGET: int = int(os.getenv("COMPRESS_THINKING_BUDGET", "512"))
OLLAMA_URL: str = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")


def get_compress_model() -> BaseChatModel:
    """Return a BaseChatModel instance configured for compression tasks.

    Called once per node invocation (not cached at module level), so each
    invocation gets a fresh connection without holding sockets across idle periods.
    """
    return _instantiate_model(COMPRESS_MODEL)


def get_fallback_compress_model() -> BaseChatModel | None:
    """Return the fallback BaseChatModel, or None if COMPRESS_FALLBACK_MODEL is unset."""
    if not COMPRESS_FALLBACK_MODEL:
        return None
    return _instantiate_model(COMPRESS_FALLBACK_MODEL)


def _instantiate_model(model_str: str) -> BaseChatModel:
    """Parse 'provider/model-name' and return a configured BaseChatModel."""
    provider, name = model_str.split("/", 1)

    if provider == "google_genai":
        from langchain.chat_models import init_chat_model

        # include_thoughts intentionally omitted (defaults False): thinking tokens
        # must not appear in structured-output responses — they break JSON parsing.
        # thinking_budget gives the model a small reasoning budget for edge cases
        # without the overhead of full chain-of-thought.
        return init_chat_model(
            name,
            model_provider=provider,
            thinking_budget=COMPRESS_THINKING_BUDGET,
        )

    if provider == "ollama":
        # Interface preserved for re-enablement via COMPRESS_MODEL=ollama/<model>.
        # reasoning=False disables chain-of-thought at the constructor level
        # (langchain-ollama v1.0+ maps this to the Ollama API's think=False).
        from langchain_ollama import ChatOllama  # deferred: only needed when Ollama is configured

        return ChatOllama(model=name, base_url=OLLAMA_URL, reasoning=False)

    # Generic LangChain provider (e.g. openai).
    from langchain.chat_models import init_chat_model

    return init_chat_model(name, model_provider=provider)
