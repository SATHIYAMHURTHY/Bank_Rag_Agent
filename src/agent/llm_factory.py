"""
llm_factory.py
Returns a LangChain chat model based on a string key.
All models must support tool-calling (bind_tools / tool use).
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── Gemini models ─────────────────────────────────────────────────────────────
# Free tier: 15 RPM, 1M TPM on gemini-2.5-flash
# gemini-2.5-flash   — best free model for agents, fast + tool-calling
# gemini-2.5-pro     — smarter, lower free quota
# gemini-2.5-flash-lite — fastest, lowest cost
GEMINI_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-2.5-flash-lite",
]

# ── Groq models (free tier, very fast inference) ──────────────────────────────
# All previous models deprecated as of mid-2025.
# These are the current working models on Groq free tier:
# openai/ prefix = OpenAI-compatible models hosted on Groq
# qwen/  prefix = Alibaba Qwen models hosted on Groq
GROQ_MODELS = [
    "openai/gpt-oss-120b",   # largest, best quality
    "openai/gpt-oss-20b",    # faster, still strong
    "qwen/qwen3-32b",        # good multilingual + reasoning
]

# ── Ollama models (local, no API key needed) ──────────────────────────────────
# Only list models you have pulled: `ollama list` to check
# Tool-calling support varies — llama3.2+ and mistral-nemo are reliable
OLLAMA_MODELS = [
    "llama3.2",
    "llama3.1",
    "mistral",
    "phi3",
]


# Combined registry for the UI dropdown
ALL_MODELS: dict[str, list[str]] = {
    "Gemini (Google)": GEMINI_MODELS,
    "Groq (Free)":     GROQ_MODELS,
    "Ollama (Local)":  OLLAMA_MODELS,
}


def get_llm(provider: str, model: str, temperature: float = 0.0):
    """
    Returns a LangChain chat LLM instance.

    Args:
        provider:    one of 'gemini', 'groq', 'ollama', 'anthropic'
        model:       model string for that provider
        temperature: 0.0 = deterministic (good for tool-calling agents)
                     higher = more creative (good for summarization)

    Returns:
        A LangChain BaseChatModel instance.
    """
    provider = provider.lower()

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set in .env")
        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=api_key,
            temperature=temperature,
        )

    elif provider == "groq":
        from langchain_groq import ChatGroq
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not set in .env")
        return ChatGroq(
            model=model,
            groq_api_key=api_key,
            temperature=temperature,
        )

    elif provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=model,
            temperature=temperature,
        )

    else:
        raise ValueError(
            f"Unknown provider '{provider}'. "
            f"Choose from: gemini, groq, ollama, anthropic"
        )