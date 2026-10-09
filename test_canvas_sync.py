"""Phase 1 tests: frontend canvas_state -> Context Engine synchronization.

Plain script (matches the other test_*.py files, not pytest) - run with
`python test_canvas_sync.py`. Requests go through the real FastAPI app
(/api/agent/chat via TestClient) and the real Planner/Executor/Context
Engine/MCP layers. Only for intents the rule-based path doesn't cover
(connect/move/remove/show) is the LLM's *answer* replaced with a fixed one,
so no Ollama key is needed; the rule-based "add" path calls no LLM at all.
"""

import json

from fastapi.testclient import TestClient

import graph.workflow as workflow
from api.server import app
from context import context_engine
from context.context_builder import ContextBuilder
from tools.canvas_tool import CanvasTool

client = TestClient(app)
passed = 0
failed = 0

ESP32_ID = 1775642938124.512
RELAY_ID = 1775642941100.234
TEMP_A_ID = 1775642950000.111
TEMP_B_ID = 1775642960000.222


def check(label, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"PASS: {label}")
    else:
        failed += 1
        print(f"FAIL: {label} {detail}")


def canvas(components, connections=()):
    return {"tabId": 1, "tabName": "Simulation 1", "components": list(components), "connections": list(connections)}


def component(instance_id, name, label=None, x=200, y=200):
    return {"id": instance_id, "name": name, "label": label, "x": x, "y": y, "width": 100, "height": 100, "rotation": 0}


def chat(command, canvas_state=None):
    body = {"command": command}
    if canvas_state is not None:
        body["canvas_state"] = canvas_state
    return client.post("/api/agent/chat", json=body)


_real_analyze = workflow.analyze


def llm_returns(answer):
    """Replace only the LLM's answer for the next requests (planner/executor/context run for real)."""
    workflow.analyze = lambda prompt, user_request: dict(answer)


def restore_llm():
    workflow.analyze = _real_analyze


def symbols():
    return CanvasTool().get_canvas_state()["symbols"]


# --- 1. Empty canvas + "add ESP32" ---
context_engine.clear()
r = chat("add ESP32", canvas([]))
body = r.json()
check("1. empty canvas + add ESP32 -> HTTP 200 success", r.status_code == 200 and body["status"] == "success", body)
check(
    "1. one canvas.addComponent for ESP32",
    [c["method"] for c in body["commands"]] == ["canvas.addComponent"] and body["commands"][0]["params"]["componentId"] == "ESP32",
    body["commands"],
)
check("1. context now holds exactly ESP32", symbols() == ["ESP32"], symbols())

# --- 2. Canvas already containing ESP32 + "add ESP32" ---
context_engine.clear()
r = chat("add ESP32", canvas([component(ESP32_ID, "Microcontroller", "ESP32")]))
body = r.json()
check("2. canvas has ESP32 + add ESP32 -> success", body["status"] == "success", body)
check("2. one new canvas.addComponent", [c["method"] for c in body["commands"]] == ["canvas.addComponent"], body["commands"])
check("2. both ESP32s tracked (synced one not overwritten)", symbols() == ["ESP32", "ESP32"], symbols())
check(
    "2. synced ESP32 keeps its instance id",
    context_engine.get_context()["components"]["ESP32"]["instance_id"] == ESP32_ID,
    context_engine.get_context()["components"],
)

# --- 3. Two Temperature Sensors with different IDs ---
context_engine.clear()
summary = context_engine.sync_from_canvas(
    canvas([component(TEMP_A_ID, "Temperature Sensor", x=100), component(TEMP_B_ID, "Temperature Sensor", x=400)])
)
ctx = context_engine.get_context()["components"]
ids = sorted(m["instance_id"] for m in ctx.values())
check("3. two temperature sensors -> two tracked components", summary["components"] == 2 and len(ctx) == 2, ctx)
check("3. instance ids preserved exactly", ids == sorted([TEMP_A_ID, TEMP_B_ID]), ids)
check("3. symbols list both", symbols() == ["Temperature Sensor", "Temperature Sensor"], symbols())
prompt_text = ContextBuilder().build_context()
check(
    "3. LLM context shows both instance ids",
    f"[id {TEMP_A_ID}]" in prompt_text and f"[id {TEMP_B_ID}]" in prompt_text,
    prompt_text,
)
check("3. name alone is ambiguous -> no instance id guessed", context_engine.find_instance_id("Temperature Sensor") is None)
check("3. instance id resolves to that sensor", context_engine.find_instance_id(TEMP_B_ID) == TEMP_B_ID)
try:
    context_engine.connect_components("Temperature Sensor", "Temperature Sensor")
    check("3. connecting an ambiguous name is rejected", False, "no error raised")
except ValueError as exc:
    check("3. connecting an ambiguous name is rejected", "matches 2 components" in str(exc), str(exc))
context_engine.connect_components(str(TEMP_A_ID), TEMP_B_ID)
conn = list(context_engine.get_context()["connections"].values())[0]
check(
    "3. connecting by instance ids records the real ids",
    {conn["source_id"], conn["target_id"]} == {TEMP_A_ID, TEMP_B_ID},
    conn,
)

# --- 4. Connecting two existing components using their instance IDs ---
context_engine.clear()
two_parts = canvas([component(ESP32_ID, "Microcontroller", "ESP32"), component(RELAY_ID, "Relay Module", x=400)])
llm_returns({"intent": "connect_components", "component": "", "components": [],
             "source_component": "ESP32", "target_component": "Relay Module", "confidence": "high", "reasoning": "test"})
r = chat("connect ESP32 to Relay Module", two_parts)
restore_llm()
body = r.json()
params = body["commands"][0]["params"] if body.get("commands") else {}
check("4. connect existing components -> success", body["status"] == "success", body)
check("4. command keeps name fields", params.get("sourceComponentId") == "ESP32" and params.get("targetComponentId") == "Relay Module", params)
check("4. command carries real instance ids", params.get("sourceId") == ESP32_ID and params.get("targetId") == RELAY_ID, params)
check("4. ids exact in raw JSON", f'"sourceId":{ESP32_ID}' in r.text and f'"targetId":{RELAY_ID}' in r.text, r.text[:400])
conn = list(context_engine.get_context()["connections"].values())
check(
    "4. context records the connection with sourceId/targetId",
    len(conn) == 1 and conn[0]["source_id"] == ESP32_ID and conn[0]["target_id"] == RELAY_ID,
    conn,
)
check("4. frontend name 'Microcontroller' resolves to the ESP32 instance", context_engine.find_instance_id("Microcontroller") == ESP32_ID)

# --- 4b. Move and remove carry the instance id too ---
context_engine.clear()
llm_returns({"intent": "move_component", "component": "", "components": ["ESP32"],
             "source_component": "", "target_component": "", "confidence": "high", "reasoning": "test"})
body = chat("move ESP32", two_parts).json()
restore_llm()
check("4b. move -> command carries instance id", body["status"] == "success" and body["commands"][0]["params"].get("id") == ESP32_ID, body)
llm_returns({"intent": "remove_component", "component": "", "components": ["Relay Module"],
             "source_component": "", "target_component": "", "confidence": "high", "reasoning": "test"})
body = chat("remove Relay Module", two_parts).json()
restore_llm()
check("4b. remove -> command carries instance id", body["status"] == "success" and body["commands"][0]["params"].get("id") == RELAY_ID, body)
check("4b. removed component gone from context", symbols() == ["ESP32"], symbols())

# --- 5. Synced connections are represented with sourceId/targetId ---
context_engine.clear()
summary = context_engine.sync_from_canvas(
    canvas(
        [component(ESP32_ID, "Microcontroller", "ESP32"), component(RELAY_ID, "Relay Module")],
        [{"connectionKey": f"{ESP32_ID}-{RELAY_ID}", "sourceId": ESP32_ID, "targetId": RELAY_ID},
         {"connectionKey": "1-2", "sourceId": 1, "targetId": 2}],
    )
)
conns = CanvasTool().get_canvas_state()["connections"]
check("5. one valid connection synced, unknown-id connection skipped", summary["connections"] == 1 and summary["skipped_connections"] == 1, summary)
check(
    "5. synced connection keeps sourceId/targetId/connectionKey",
    len(conns) == 1 and conns[0]["source_id"] == ESP32_ID and conns[0]["target_id"] == RELAY_ID
    and conns[0]["connection_key"] == f"{ESP32_ID}-{RELAY_ID}",
    conns,
)

# --- 6. Agent no longer relies on its old memory when canvas_state is supplied ---
context_engine.clear()
chat("add Buzzer")  # old behaviour: goes into the agent's own memory
check("6. without canvas_state, agent memory holds Buzzer", symbols() == ["Buzzer"], symbols())
llm_returns({"intent": "get_canvas_state", "component": "", "components": [],
             "source_component": "", "target_component": "", "confidence": "high", "reasoning": "test"})
body = chat("show canvas", canvas([component(RELAY_ID, "Relay Module")])).json()
restore_llm()
state = body["results"][0]["result"] if body.get("results") else {}
check("6. with canvas_state, the frontend canvas replaces old memory", state.get("symbols") == ["Relay Module"], state)
llm_returns({"intent": "get_canvas_state", "component": "", "components": [],
             "source_component": "", "target_component": "", "confidence": "high", "reasoning": "test"})
body = chat("show canvas", canvas([])).json()
restore_llm()
check("6. empty canvas_state -> empty context", body["results"][0]["result"]["symbols"] == [], body["results"])

# --- 7. Existing requests without canvas_state still work exactly as before ---
context_engine.clear()
baseline = {
    "status": "success",
    "message": "Command generated successfully",
    "commands": [{"method": "canvas.addComponent", "params": {"componentId": "ESP32", "position": {"x": 200, "y": 200}}}],
    "results": [
        {"tool": "search_component", "status": "success", "result": {"success": True, "component": {"id": "esp32", "name": "ESP32"}}},
        {"tool": "add_component", "status": "success", "result": {"success": True, "instanceId": 12345}},
    ],
}
r = chat("Add ESP32")
check("7. no canvas_state: 'Add ESP32' response identical to Phase 0 baseline", r.json() == baseline, json.dumps(r.json()))
chat("Add Relay Module")
llm_returns({"intent": "connect_components", "component": "", "components": [],
             "source_component": "ESP32", "target_component": "Relay Module", "confidence": "high", "reasoning": "test"})
body = chat("connect ESP32 to Relay Module").json()
restore_llm()
check(
    "7. no canvas_state: connect by name works, command unchanged (no id fields)",
    body["status"] == "success"
    and body["commands"] == [{"method": "canvas.connectComponents", "params": {"sourceComponentId": "ESP32", "targetComponentId": "Relay Module"}}],
    body,
)
r = chat("   ")
check("7. blank command still HTTP 400", r.status_code == 400 and r.json()["message"] == "Command is required", r.text)

# --- 8. Invalid canvas_state is rejected cleanly ---
r = chat("add ESP32", {"tabId": 1, "components": [{"name": "Microcontroller", "x": 1, "y": 1}], "connections": []})
check("8. component without id -> HTTP 400 (no crash)", r.status_code == 400 and r.json()["status"] == "error", r.text)

context_engine.clear()
print(f"\n{passed} passed, {failed} failed")
