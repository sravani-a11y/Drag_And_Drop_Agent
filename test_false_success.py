"""Tests for the "false success" fix: failed canvas operations must be reported as errors.

Plain script (matches the other test_*.py files, not pytest) - run with
`python test_false_success.py`. Requests go through the real FastAPI app,
Planner, Executor, Verifier and Context Engine; only the LLM's *answer* is
fixed per case (no Ollama key needed). Each failure case also checks that
no command for the failed operation is sent to the frontend.
"""

from fastapi.testclient import TestClient

import graph.workflow as workflow
from api.server import app
from context import context_engine

client = TestClient(app)
passed = 0
failed = 0

BUZZER_ID = 1775642938124.512
RELAY_ID = 1775642941100.234
TEMP_A_ID = 1775642950000.111
TEMP_B_ID = 1775642960000.222
ESP32_ID = 1775642970000.333


def check(label, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"PASS: {label}")
    else:
        failed += 1
        print(f"FAIL: {label} {detail}")


def component(instance_id, name, label=None, x=200, y=200):
    return {"id": instance_id, "name": name, "label": label, "x": x, "y": y, "width": 100, "height": 100, "rotation": 0}


def canvas(components, connections=()):
    return {"tabId": 1, "tabName": "Simulation 1", "components": list(components), "connections": list(connections)}


BUZZER_AND_RELAY = canvas([component(BUZZER_ID, "Buzzer"), component(RELAY_ID, "Relay Module", x=400)])

_real_analyze = workflow.analyze


def chat(command, llm_answer=None, canvas_state=None):
    """POST /api/agent/chat; llm_answer (if given) replaces only the LLM's answer for this request."""
    if llm_answer is not None:
        answer = {"component": "", "components": [], "source_component": "", "target_component": "",
                  "confidence": "high", "reasoning": "test", **llm_answer}
        workflow.analyze = lambda prompt, user_request: dict(answer)
    try:
        body = {"command": command}
        if canvas_state is not None:
            body["canvas_state"] = canvas_state
        return client.post("/api/agent/chat", json=body)
    finally:
        workflow.analyze = _real_analyze


def methods(body):
    return [c["method"] for c in body.get("commands", [])]


# --- 1. Connect a missing component -> error ---
context_engine.clear()
r = chat("Connect LED to Buzzer", {"intent": "connect_components", "source_component": "LED", "target_component": "Buzzer"}, BUZZER_AND_RELAY)
body = r.json()
check("1. connect missing LED -> status error", r.status_code == 200 and body["status"] == "error", body)
check("1. clear message", body.get("message") == "Cannot connect LED to Buzzer because LED was not found.", body.get("message"))
check("1. no connect command sent to the frontend", "canvas.connectComponents" not in methods(body), methods(body))
check("1. not marked as a clarification", "needs_clarification" not in body, body)
check("1. no connection recorded", context_engine.get_context()["connections"] == {})

# --- 1b. Same, without canvas_state (agent memory only) ---
context_engine.clear()
chat("Add Buzzer")
r = chat("Connect LED to Buzzer", {"intent": "connect_components", "source_component": "LED", "target_component": "Buzzer"})
body = r.json()
check("1b. without canvas_state: connect missing LED -> error", body["status"] == "error" and "LED was not found" in body["message"], body)

# --- 1c. Both components missing ---
context_engine.clear()
body = chat("Connect LED to Fan", {"intent": "connect_components", "source_component": "LED", "target_component": "Fan"}, BUZZER_AND_RELAY).json()
check("1c. both missing -> names both", body["message"] == "Cannot connect LED to Fan because LED and Fan were not found.", body["message"])

# --- 2. Remove a missing component -> error ---
context_engine.clear()
body = chat("Remove LED", {"intent": "remove_component", "components": ["LED"]}, BUZZER_AND_RELAY).json()
check("2. remove missing LED -> status error", body["status"] == "error", body)
check("2. clear message", body["message"] == "Cannot remove LED because LED was not found.", body["message"])
check("2. no remove command sent", "canvas.removeComponent" not in methods(body), methods(body))

# --- 3. Move a missing component -> error ---
context_engine.clear()
body = chat("Move LED", {"intent": "move_component", "components": ["LED"]}, BUZZER_AND_RELAY).json()
check("3. move missing LED -> status error", body["status"] == "error", body)
check("3. clear message", body["message"] == "Cannot move LED because LED was not found.", body["message"])
check("3. no move command sent", "canvas.moveComponent" not in methods(body), methods(body))

# --- 4. Disconnect a connection that does not exist -> error ---
context_engine.clear()
body = chat("Disconnect Buzzer from Relay Module",
            {"intent": "disconnect_components", "source_component": "Buzzer", "target_component": "Relay Module"}, BUZZER_AND_RELAY).json()
check("4. disconnect a pair that isn't connected -> status error", body["status"] == "error", body)
check("4. clear message", body["message"] == "Cannot disconnect Buzzer from Relay Module because they are not connected.", body["message"])
check("4. no disconnect command sent", "canvas.disconnectComponents" not in methods(body), methods(body))
body = chat("Disconnect LED from Buzzer",
            {"intent": "disconnect_components", "source_component": "LED", "target_component": "Buzzer"}, BUZZER_AND_RELAY).json()
check("4b. disconnect with a missing component -> not-found error",
      body["status"] == "error" and body["message"] == "Cannot disconnect LED from Buzzer because LED was not found.", body["message"])

# --- 5. Successful operations are still success (with Phase 1 instance IDs) ---
connected = canvas([component(BUZZER_ID, "Buzzer"), component(RELAY_ID, "Relay Module", x=400)],
                   [{"connectionKey": f"{BUZZER_ID}-{RELAY_ID}", "sourceId": BUZZER_ID, "targetId": RELAY_ID}])
context_engine.clear()
body = chat("Connect Buzzer to Relay Module",
            {"intent": "connect_components", "source_component": "Buzzer", "target_component": "Relay Module"}, BUZZER_AND_RELAY).json()
p = body["commands"][0]["params"] if body.get("commands") else {}
check("5. connect existing -> success", body["status"] == "success" and methods(body) == ["canvas.connectComponents"], body)
check("5. connect still carries sourceId/targetId", p.get("sourceId") == BUZZER_ID and p.get("targetId") == RELAY_ID, p)
body = chat("Disconnect Buzzer from Relay Module",
            {"intent": "disconnect_components", "source_component": "Buzzer", "target_component": "Relay Module"}, connected).json()
check("5. disconnect an existing connection -> success", body["status"] == "success" and methods(body) == ["canvas.disconnectComponents"], body)
body = chat("Move Buzzer", {"intent": "move_component", "components": ["Buzzer"]}, BUZZER_AND_RELAY).json()
check("5. move existing -> success with id", body["status"] == "success" and body["commands"][0]["params"].get("id") == BUZZER_ID, body)
body = chat("Remove Buzzer", {"intent": "remove_component", "components": ["Buzzer"]}, BUZZER_AND_RELAY).json()
check("5. remove existing -> success with id", body["status"] == "success" and body["commands"][0]["params"].get("id") == BUZZER_ID, body)
context_engine.clear()
body = chat("Add ESP32").json()
check("5. add -> success, response shape unchanged", body["status"] == "success" and set(body) == {"status", "message", "commands", "results"}, body)

# --- 6. Ambiguous component -> error + needs_clarification, nothing chosen ---
two_sensors = canvas([component(TEMP_A_ID, "Temperature Sensor", x=100), component(TEMP_B_ID, "Temperature Sensor", x=300),
                      component(ESP32_ID, "Microcontroller", "ESP32", x=500)])
context_engine.clear()
body = chat("Connect Temperature Sensor to ESP32",
            {"intent": "connect_components", "source_component": "Temperature Sensor", "target_component": "ESP32"}, two_sensors).json()
check("6. ambiguous -> status error", body["status"] == "error", body)
check("6. needs_clarification true", body.get("needs_clarification") is True, body)
check("6. message names both instance ids", str(TEMP_A_ID) in body["message"] and str(TEMP_B_ID) in body["message"], body["message"])
check("6. no connect command sent (no random pick)", "canvas.connectComponents" not in methods(body), methods(body))

# --- 7. Multi-step: later step fails -> error, earlier successful command kept ---
context_engine.clear()
body = chat("Add Relay Module and connect it to ESP32",
            {"actions": [{"intent": "add_component", "components": ["Relay Module"]},
                         {"intent": "connect_components", "source": "ESP32", "target": "Relay Module"}]}).json()
check("7. second step fails -> overall status error", body["status"] == "error" and "ESP32 was not found" in body["message"], body)
check("7. only the add that really happened is sent", methods(body) == ["canvas.addComponent"], methods(body))

# --- 8. A tool that reports success: false is not turned into success ---
context_engine.clear()
# The workflow's executor holds its tool methods in a dispatch table built at
# import time, so the tool is replaced there (not on the ComponentTool class).
dispatch = workflow._tool_executor._dispatch
_real_add = dispatch["add_component"]
dispatch["add_component"] = lambda component_id, x, y: {"success": False, "error": "Frontend rejected the component"}
try:
    body = chat("Add ESP32").json()
finally:
    dispatch["add_component"] = _real_add
check("8. tool reports success:false -> status error", body["status"] == "error", body)
check("8. tool's own error message passed through", body["message"] == "Frontend rejected the component", body["message"])
check("8. failed add not recorded in context", context_engine.get_context()["components"] == {}, context_engine.get_context())

context_engine.clear()
print(f"\n{passed} passed, {failed} failed")
