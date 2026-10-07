"""Configuration for the LLM provider used by the assistant."""

import ollama

LLM_MODEL = "qwen2.5:7b"


def get_client() -> ollama.Client:
    return ollama.Client()
