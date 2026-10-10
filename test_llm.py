"""Tests for llm.analyze(): response parsing and normalization, with a fake LLM client.

Plain script (matches the other test_*.py files, not pytest) - run with
`python test_llm.py`. No network and no OLLAMA_API_KEY needed: get_client()
is replaced by a fake that returns a fixed reply. To also run one live call
against the real configured LLM, set RUN_LIVE_LLM_TESTS=1 and OLLAMA_API_KEY.
"""

import os

import llm

passed = 0
failed = 0


def check(label, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"PASS: {label}")
    else:
        failed += 1
        print(f"FAIL: {label} {detail}")


class _Message:
    def __init__(self, content):
        self.content = content


class _Response:
    def __init__(self, content):
        self.message = _Message(content)


class FakeClient:
    """Stands in for ollama.Client: records each chat() call and returns `reply`."""

    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        return _Response(self.reply)


_real_get_client = llm.get_client


def analyze_with(reply, user_request="Add ESP32"):
    """Run llm.analyze() against a FakeClient returning `reply`; returns (result or exception, client)."""
    client = FakeClient(reply)
    llm.get_client = lambda: client
    try:
        return llm.analyze("PROMPT-BODY", user_request), client
    except Exception as exc:
        return exc, client
    finally:
        llm.get_client = _real_get_client


# --- 1. A clean single-intent reply is parsed and returned ---
result, client = analyze_with('{"intent": "add_component", "component": "", "components": ["ESP32"]}')
check("1. parsed into a dict", isinstance(result, dict), result)
check("1. intent kept", result.get("intent") == "add_component", result)
check("1. components kept", result.get("components") == ["ESP32"], result)
check("1. sent as JSON-format chat to the configured model",
      client.calls and client.calls[0].get("format") == "json" and client.calls[0].get("model") == llm.LLM_MODEL, client.calls)
check("1. prompt body is in the message sent", "PROMPT-BODY" in client.calls[0]["messages"][0]["content"], client.calls)

# --- 2. Intent spelling is normalized to the canonical vocabulary ---
result, _ = analyze_with('{"intent": "showCanvas", "component": ""}')
check("2. 'showCanvas' -> get_canvas_state", result.get("intent") == "get_canvas_state", result)
result, _ = analyze_with('{"intent": "Connect Components", "source_component": "ESP32", "target_component": "LED"}')
check("2. 'Connect Components' -> connect_components", result.get("intent") == "connect_components", result)

# --- 3. Markdown code fences around the JSON are accepted ---
result, _ = analyze_with('```json\n{"intent": "remove_component", "component": "ESP32"}\n```')
check("3. fenced JSON parsed", isinstance(result, dict) and result.get("intent") == "remove_component", result)

# --- 4. Multi-action shapes ---
result, _ = analyze_with('[{"intent": "add_component", "components": ["ESP32"]}, {"intent": "add_component", "components": ["LED"]}]')
check("4. bare list -> {'actions': [...]}", isinstance(result, dict) and len(result.get("actions", [])) == 2, result)
result, _ = analyze_with('{"intents": [{"intent": "add_component", "components": ["ESP32"]}]}')
check("4. 'intents' adopted as 'actions'", isinstance(result, dict) and len(result.get("actions", [])) == 1, result)

# --- 5. Bad replies raise LLMResponseError (the workflow turns that into a safe message) ---
result, _ = analyze_with("")
check("5. empty reply -> LLMResponseError", isinstance(result, llm.LLMResponseError), result)
result, _ = analyze_with("this is not json")
check("5. non-JSON reply -> LLMResponseError", isinstance(result, llm.LLMResponseError), result)

# --- Optional: one real call, only when explicitly requested ---
if os.environ.get("RUN_LIVE_LLM_TESTS") == "1" and os.environ.get("OLLAMA_API_KEY"):
    from knowledge_loader import KnowledgeLoader
    from prompt_builder import build_prompt

    live = llm.analyze(build_prompt(KnowledgeLoader().load_all(), "Add ESP32"), "Add ESP32")
    check("live. real LLM returns a dict for 'Add ESP32'", isinstance(live, dict), live)
else:
    print("SKIP: live LLM check (set RUN_LIVE_LLM_TESTS=1 and OLLAMA_API_KEY to run it)")

print(f"\n{passed} passed, {failed} failed")
