"""Tests for the generalized add+connect intent/planning fix (planner.py's _detect_add_and_connect()).

Plain script (matches the other test_*.py files, not pytest) - run with
`python test_planner_intent.py`. Runs the varied natural-language commands
through the real LangGraph workflow (graph.run_workflow()).

The requests that aren't a plain "add" go through the LLM. These tests
check the Planner, not the model, so by default the LLM's answer is fixed
per request (_LLM_ANSWERS, a realistic reply for each) - no network or
OLLAMA_API_KEY needed. Set RUN_LIVE_LLM_TESTS=1 (with OLLAMA_API_KEY) to
run them against the real LLM instead.
"""

import os

import graph.workflow as workflow
from graph import run_workflow

passed = 0
failed = 0

# The answer a well-behaved LLM gives for each non-"add" request below.
_LLM_ANSWERS = {
    "connect ESP32 to accelerometer": {"intent": "connect_components", "source_component": "ESP32", "target_component": "Accelerometer"},
    "link ESP32 with accelerometer": {"intent": "connect_components", "source_component": "ESP32", "target_component": "Accelerometer"},
    "add ESP32 and connect it to accelerometer": {"intent": "connect_components", "source_component": "ESP32", "target_component": "Accelerometer"},
    "create ESP32 connected to accelerometer": {"intent": "connect_components", "source_component": "ESP32", "target_component": "Accelerometer"},
    "move ESP32 to a new position": {"intent": "move_component", "component": "ESP32"},
    "remove ESP32": {"intent": "remove_component", "component": "ESP32"},
}


def _fixed_llm_answer(prompt, user_request):
    if user_request not in _LLM_ANSWERS:
        raise AssertionError(f"test_planner_intent: no fixed LLM answer for {user_request!r} - add one to _LLM_ANSWERS")
    return {"component": "", "components": [], "source_component": "", "target_component": "",
            "confidence": "high", "reasoning": "fixed test answer", **_LLM_ANSWERS[user_request]}


if os.environ.get("RUN_LIVE_LLM_TESTS") == "1" and os.environ.get("OLLAMA_API_KEY"):
    print("Using the live LLM (RUN_LIVE_LLM_TESTS=1)")
else:
    workflow.analyze = _fixed_llm_answer


def added_components(result):
    return [s["input"]["component_id"] for s in result.get("execution_plan", []) if s["tool"] == "add_component"]


def connections(result):
    return [
        (s["input"]["source_component"], s["input"]["target_component"])
        for s in result.get("execution_plan", [])
        if s["tool"] == "connect_components"
    ]


def check(label, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"PASS: {label}")
    else:
        failed += 1
        print(f"FAIL: {label} {detail}")


# --- Varied natural-language commands (the actual required test list) ---

r = run_workflow("add ESP32")
check("1. add ESP32 -> completed", r.get("status") == "completed", r)
check("1. ESP32 added", added_components(r) == ["ESP32"], added_components(r))

r = run_workflow("create an ESP32")
check("2. create an ESP32 -> completed", r.get("status") == "completed", r)
check("2. ESP32 added", "ESP32" in added_components(r), added_components(r))

r = run_workflow("add ESP32 and DHT11")
check("3. add ESP32 and DHT11 -> completed", r.get("status") == "completed", r)
check("3. both components added", added_components(r) == ["ESP32", "DHT11 Sensor"], added_components(r))

# Connecting requires both components on the canvas (connecting to a missing
# one is correctly an error), so put the Accelerometer there first.
r = run_workflow("add accelerometer")
check("4. setup: accelerometer added", r.get("status") == "completed", r.get("error"))

r = run_workflow("connect ESP32 to accelerometer")
check("4. connect ESP32 to accelerometer -> completed", r.get("status") == "completed", r)
check("4. connection planned (ESP32 -> Accelerometer)", connections(r) == [("ESP32", "Accelerometer")], r.get("execution_plan"))

r = run_workflow("link ESP32 with accelerometer")
check("5. link ESP32 with accelerometer -> completed", r.get("status") == "completed", r)
check("5. connection planned (ESP32 -> Accelerometer)", connections(r) == [("ESP32", "Accelerometer")], r.get("execution_plan"))

r = run_workflow("add ESP32 and connect it to accelerometer")
check("6. add+connect -> completed", r.get("status") == "completed", r)
check("6. both components added", added_components(r) == ["ESP32", "Accelerometer"], added_components(r))
check("6. connection planned", connections(r) == [("ESP32", "Accelerometer")], r.get("execution_plan"))

r = run_workflow("create ESP32 connected to accelerometer")
check("7. create+connected -> completed", r.get("status") == "completed", r)
check("7. both components added", added_components(r) == ["ESP32", "Accelerometer"], added_components(r))
check("7. connection planned", connections(r) == [("ESP32", "Accelerometer")], r.get("execution_plan"))

r = run_workflow("move ESP32 to a new position")
check("8. move ESP32 -> completed", r.get("status") == "completed", r)
check("8. move_component planned", any(s["tool"] == "move_component" for s in r.get("execution_plan", [])), r.get("execution_plan"))

r = run_workflow("remove ESP32")
check("9. remove ESP32 -> completed", r.get("status") == "completed", r)
check(
    "9. delete/remove_component planned",
    any(s["tool"] in ("delete_component", "remove_component") for s in r.get("execution_plan", [])),
    r.get("execution_plan"),
)

# --- Regression: must still work exactly as before ---

r = run_workflow("add ESP32 and Relay")
check("regression: add ESP32 and Relay -> completed", r.get("status") == "completed", r)
check("regression: both added, no auto-connect", added_components(r) == ["ESP32", "Relay Module"] and connections(r) == [], r)

r = run_workflow("add ESP32, relay, led")
check("regression: add ESP32, relay, led -> completed", r.get("status") == "completed", r)
check("regression: all three added", added_components(r) == ["ESP32", "Relay Module", "LED"], added_components(r))

print(f"\n{passed} passed, {failed} failed")
