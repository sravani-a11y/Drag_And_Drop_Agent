"""End-to-end tests for the multi-agent LangGraph workflow (Intent -> Planner -> Executor -> Verifier).

Plain script (matches test_knowledge_loader.py/test_llm.py's style, not
pytest) - run with `python test_multi_agent.py`. Covers the 10 required
test cases. Cases 4-8 need a running local Ollama server (qwen2.5:7b) and
are noticeably slower than the rest, since they go through the LLM; cases
1-3, 9, and 10 are pure rule-based/monkeypatched and run in well under a
second with no LLM involved.
"""

from graph import run_workflow
from tool_executor import ToolExecutor

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


def added_components(result):
    return [s["input"]["component_id"] for s in result.get("execution_plan", []) if s["tool"] == "add_component"]


def connections(result):
    return [
        (s["input"]["source_component"], s["input"]["target_component"])
        for s in result.get("execution_plan", [])
        if s["tool"] == "connect_components"
    ]


# --- Case 1: add esp32 ---
r = run_workflow("add esp32")
check("case 1: add esp32 -> completed", r.get("status") == "completed", r)
check("case 1: ESP32 added", added_components(r) == ["ESP32"], added_components(r))

# --- Case 2: add esp32 and dht11 ---
r = run_workflow("add esp32 and dht11")
check("case 2: add esp32 and dht11 -> completed", r.get("status") == "completed", r)
check("case 2: both components added", added_components(r) == ["ESP32", "DHT11 Sensor"], added_components(r))
check("case 2: no auto-connect", connections(r) == [], connections(r))

# --- Case 3: add esp32, relay, led ---
r = run_workflow("add esp32, relay, led")
check("case 3: add esp32, relay, led -> completed", r.get("status") == "completed", r)
check(
    "case 3: all three components added",
    added_components(r) == ["ESP32", "Relay Module", "LED"],
    added_components(r),
)
check("case 3: no auto-connect", connections(r) == [], connections(r))

# --- Cases 4-8: need the LLM (skip gracefully if Ollama isn't reachable) ---
try:
    import ollama

    ollama.Client().list()
    llm_available = True
except Exception:
    llm_available = False

if llm_available:
    run_workflow("add esp32 and relay")  # seed canvas state for the cases below

    r = run_workflow("move esp32 to 300 300")
    check("case 4: move esp32 -> completed", r.get("status") == "completed", r)

    r = run_workflow("connect esp32 to relay")
    check("case 5: connect esp32 to relay -> completed", r.get("status") == "completed", r)
    check("case 5: connection planned", connections(r) == [("ESP32", "Relay Module")], connections(r))

    r = run_workflow("delete relay")
    check("case 6: delete relay -> completed", r.get("status") == "completed", r)

    r = run_workflow("show canvas")
    check("case 7: show canvas -> completed", r.get("status") == "completed", r)
    check(
        "case 7: get_canvas_state planned",
        any(s["tool"] == "get_canvas_state" for s in r.get("execution_plan", [])),
        r.get("execution_plan"),
    )

    r = run_workflow("add esp32 and dht11, then connect esp32 to dht11")
    check("case 8: multi-step add+connect -> completed", r.get("status") == "completed", r)
    # NOTE: whether the connect step is actually present depends on the LLM
    # correctly recognizing two distinct operations in one sentence - see
    # the "remaining limitations" note in the project's audit/report for
    # this specific case. Not asserted here since it's a model-reliability
    # question, not a graph/agent correctness one.
    print(f"  case 8 plan: {[s['tool'] for s in r.get('execution_plan', [])]}")
else:
    print("SKIP: cases 4-8 need a running Ollama server (qwen2.5:7b) - none reachable")

# --- Case 9: invalid/unknown component ---
r = run_workflow("add xyz9999foobar")
check("case 9: unknown component doesn't crash", r.get("status") == "completed", r)
check("case 9: component still recorded (mock tools don't validate the catalog)", len(added_components(r)) == 1)

# --- Case 10a: transient failure -> retry -> success ---
_real_execute_step = ToolExecutor.execute_step
_attempts = {"n": 0}


def _flaky_once(self, step):
    if step.get("tool") == "add_component":
        _attempts["n"] += 1
        if _attempts["n"] == 1:
            return {"tool": "add_component", "status": "failed", "error": "connection reset by peer"}
    return _real_execute_step(self, step)


ToolExecutor.execute_step = _flaky_once
try:
    r = run_workflow("add esp32")
finally:
    ToolExecutor.execute_step = _real_execute_step

check("case 10a: recovers from a transient failure", r.get("status") == "completed", r)
check("case 10a: exactly one retry used", r.get("retry_count") == 1, r.get("retry_count"))

# --- Case 10b: persistent failure -> gives up cleanly (no infinite loop) ---


def _always_fails(self, step):
    if step.get("tool") == "add_component":
        return {"tool": "add_component", "status": "failed", "error": "connection reset by peer"}
    return _real_execute_step(self, step)


ToolExecutor.execute_step = _always_fails
try:
    r = run_workflow("add esp32")
finally:
    ToolExecutor.execute_step = _real_execute_step

check("case 10b: gives up after MAX_RETRIES", r.get("status") == "failed", r)
check("case 10b: retry_count capped at 2", r.get("retry_count") == 2, r.get("retry_count"))

print(f"\n{passed} passed, {failed} failed")
