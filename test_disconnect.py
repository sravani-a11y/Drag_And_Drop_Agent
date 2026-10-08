"""Focused tests for disconnect: the real connection is removed, and failures are never reported as success.

Plain script (matches the other test_*.py files, not pytest) - run with
`python test_disconnect.py`. Requests go through the real FastAPI app,
Planner, Executor, Verifier and Context Engine; only the LLM's *answer* is
fixed per case (no Ollama key needed).
"""

from fastapi.testclient import TestClient

import graph.workflow as workflow
from api.server import app
from context import context_engine

client = TestClient(app)
passed = 0
failed = 0

ESP32_ID = 1775642938124.512
RELAY_ID = 1775642941100.234
TEMP_A_ID = 1775642950000.111
TEMP_B_ID = 1775642960000.222
# Exactly as the frontend builds it: the two instance IDs joined with "-".
KEY = f"{ESP32_ID}-{RELAY_ID}"


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


PARTS = [component(ESP32_ID, "Microcontroller", "ESP32"), component(RELAY_ID, "Relay Module", x=400)]


def canvas(connections=()):
    return {"tabId": 1, "tabName": "Simulation 1", "components": PARTS, "connections": list(connections)}


CONNECTED = canvas([{"connectionKey": KEY, "sourceId": ESP32_ID, "targetId": RELAY_ID}])
NOT_CONNECTED = canvas()

_real_analyze = workflow.analyze


def chat(command, intent, source, target, canvas_state=None):
    """POST /api/agent/chat with the LLM's answer fixed to `intent source -> target`."""
    answer = {"intent": intent, "component": "", "components": [], "source_component": source,
              "target_component": target, "confidence": "high", "reasoning": "test"}
    workflow.analyze = lambda prompt, user_request: dict(answer)
    try:
        body = {"command": command}
        if canvas_state is not None:
            body["canvas_state"] = canvas_state
        return client.post("/api/agent/chat", json=body).json()
    finally:
        workflow.analyze = _real_analyze


def disconnect_commands(body):
    return [c for c in body.get("commands", []) if c["method"] == "canvas.disconnectComponents"]


def agent_connections():
    return list(context_engine.get_context()["connections"].values())


# --- A. Connect ESP32 to Relay Module -> connection created, success ---
context_engine.clear()
body = chat("Connect ESP32 to Relay Module", "connect_components", "ESP32", "Relay Module", NOT_CONNECTED)
check("A. connect -> success", body["status"] == "success" and [c["method"] for c in body["commands"]] == ["canvas.connectComponents"], body)
conns = agent_connections()
check("A. connection recorded with the real instance ids", len(conns) == 1 and (conns[0]["source_id"], conns[0]["target_id"]) == (ESP32_ID, RELAY_ID), conns)

# --- B. Disconnect ESP32 from Relay Module -> connection removed, success ---
context_engine.clear()
body = chat("Disconnect ESP32 from Relay Module", "disconnect_components", "ESP32", "Relay Module", CONNECTED)
cmds = disconnect_commands(body)
params = cmds[0]["params"] if cmds else {}
check("B. disconnect -> success", body["status"] == "success" and len(cmds) == 1, body)
check("B. command carries the frontend's exact connectionKey", params.get("connectionKey") == KEY, params)
check("B. command carries sourceId/targetId", params.get("sourceId") == ESP32_ID and params.get("targetId") == RELAY_ID, params)
check("B. command keeps the name fields", params.get("sourceComponentId") == "ESP32" and params.get("targetComponentId") == "Relay Module", params)
check("B. connection removed from the agent's state", agent_connections() == [], agent_connections())

# --- B2. Reversed wording finds the same connection ---
context_engine.clear()
body = chat("Disconnect Relay Module from ESP32", "disconnect_components", "Relay Module", "ESP32", CONNECTED)
cmds = disconnect_commands(body)
check("B2. reversed order -> success with the same connectionKey",
      body["status"] == "success" and cmds and cmds[0]["params"].get("connectionKey") == KEY, body)
check("B2. connection removed", agent_connections() == [], agent_connections())

# --- B3. Without canvas_state: a connection the agent made itself is still removed ---
context_engine.clear()
client.post("/api/agent/chat", json={"command": "Add ESP32 and Relay Module"})
chat("Connect ESP32 to Relay Module", "connect_components", "ESP32", "Relay Module")
body = chat("Disconnect ESP32 from Relay Module", "disconnect_components", "ESP32", "Relay Module")
cmds = disconnect_commands(body)
check("B3. no canvas_state: disconnect -> success", body["status"] == "success" and len(cmds) == 1, body)
check("B3. no canvas_state: no connectionKey invented", cmds and "connectionKey" not in cmds[0]["params"], cmds)
check("B3. no canvas_state: connection removed", agent_connections() == [], agent_connections())

# --- C. Disconnect again -> error, the connection no longer exists ---
body = chat("Disconnect ESP32 from Relay Module", "disconnect_components", "ESP32", "Relay Module", NOT_CONNECTED)
check("C. disconnect again -> status error", body["status"] == "error", body)
check("C. clear message", body["message"] == "Cannot disconnect ESP32 from Relay Module because they are not connected.", body["message"])
check("C. no disconnect command sent", disconnect_commands(body) == [], body.get("commands"))

# --- D. Component that does not exist -> error ---
body = chat("Disconnect LED from Relay Module", "disconnect_components", "LED", "Relay Module", CONNECTED)
check("D. missing component -> status error", body["status"] == "error", body)
check("D. clear message", body["message"] == "Cannot disconnect LED from Relay Module because LED was not found.", body["message"])
check("D. no disconnect command sent", disconnect_commands(body) == [], body.get("commands"))
check("D. existing ESP32-Relay connection untouched", len(agent_connections()) == 1, agent_connections())

# --- Ambiguous component -> error + needs_clarification, nothing removed ---
two_sensors = {"tabId": 1, "tabName": "Simulation 1",
               "components": PARTS + [component(TEMP_A_ID, "Temperature Sensor", x=600), component(TEMP_B_ID, "Temperature Sensor", x=800)],
               "connections": [{"connectionKey": f"{ESP32_ID}-{TEMP_A_ID}", "sourceId": ESP32_ID, "targetId": TEMP_A_ID}]}
body = chat("Disconnect ESP32 from Temperature Sensor", "disconnect_components", "ESP32", "Temperature Sensor", two_sensors)
check("ambiguous -> status error with needs_clarification", body["status"] == "error" and body.get("needs_clarification") is True, body)
check("ambiguous -> no disconnect command sent", disconnect_commands(body) == [], body.get("commands"))
check("ambiguous -> connection not removed", len(agent_connections()) == 1, agent_connections())

context_engine.clear()
print(f"\n{passed} passed, {failed} failed")
