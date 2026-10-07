"""Tests for the generalized add+connect intent/planning fix (planner.py's _detect_add_and_connect()).

Plain script (matches test_knowledge_loader.py/test_llm.py's style, not
pytest) - run with `python test_planner_intent.py`. Runs the varied
natural-language commands through the real, unmodified LangGraph workflow
(graph.run_workflow()) - some of these need a running local Ollama server
(qwen2.5:7b) since they aren't pure-add requests, so this can take a few
minutes end to end.
"""

from graph import run_workflow

passed = 0
failed = 0


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
