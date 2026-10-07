"""Exercises multi-component add requests end to end.

Plain script (matches test_knowledge_loader.py/test_llm.py's style, not
pytest) - run with `python test_multi_component.py`. Every case here is a
pure "add" request, so the Intent Agent's rule-based shortcut
(graph/workflow.py) handles it without ever calling the LLM - this can run
with no Ollama server up.

No connections are expected here: components are never auto-connected to
each other (e.g. to ESP32) unless the user explicitly asks for a
connection - see test_multi_agent.py for that case.
"""

from graph import run_workflow

CASES = [
    "add esp32 and dht11",
    "add esp32, relay, led",
    "add esp32 and relay and dht11",
    "add led, buzzer, motor",
]

for request_text in CASES:
    result = run_workflow(request_text)

    # tool_results only carries {"tool", "status", "result"} - the
    # component id/connection pair planned for each step lives on
    # execution_plan instead.
    plan = result.get("execution_plan", [])
    added = [step["input"]["component_id"] for step in plan if step["tool"] == "add_component"]
    connected = [
        (step["input"]["source_component"], step["input"]["target_component"])
        for step in plan
        if step["tool"] == "connect_components"
    ]

    print(f"Request: {request_text!r}")
    print(f"  status:    {result.get('status')}")
    print(f"  added:     {added}")
    print(f"  connected: {connected}")
    print()
