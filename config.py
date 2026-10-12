"""Configuration for the LLM provider."""

import os
import ollama

LLM_MODEL = os.getenv("COCO_MODEL", "gpt-oss:20b-cloud")

# Seconds an LLM request may wait for the server before failing with
# httpx.TimeoutException. Without it the ollama client waits forever, and one
# stalled call holds the workflow lock for every other request. 60 s is well
# above a normal call (~5-10 s) and, with speech-to-text's own 30 s limit,
# still answers before the frontend gives up (~2 min). Override with
# LLM_TIMEOUT_SECONDS; an invalid or non-positive value falls back to 60.
_DEFAULT_LLM_TIMEOUT_SECONDS = 60.0


def _llm_timeout_seconds() -> float:
    try:
        value = float(os.getenv("LLM_TIMEOUT_SECONDS", _DEFAULT_LLM_TIMEOUT_SECONDS))
    except ValueError:
        return _DEFAULT_LLM_TIMEOUT_SECONDS
    return value if value > 0 else _DEFAULT_LLM_TIMEOUT_SECONDS


LLM_TIMEOUT_SECONDS = _llm_timeout_seconds()


def get_client() -> ollama.Client:
    api_key = os.environ["OLLAMA_API_KEY"]

    return ollama.Client(
        host="https://ollama.com",
        headers={
            "Authorization": f"Bearer {api_key}"
        },
        timeout=LLM_TIMEOUT_SECONDS,
    )
