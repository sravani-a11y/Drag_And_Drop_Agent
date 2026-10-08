"""Configuration for the LLM provider."""

import os
import ollama

LLM_MODEL = os.getenv("COCO_MODEL", "gpt-oss:20b-cloud")


def get_client() -> ollama.Client:
    api_key = os.environ["OLLAMA_API_KEY"]

    return ollama.Client(
        host="https://ollama.com",
        headers={
            "Authorization": f"Bearer {api_key}"
        },
    )
