"""Focused tests for reply types, clarifying questions, duplicate instances, safe errors and free-spot placement.

Plain script (matches the other test_*.py files, not pytest) - run with
`python test_smart_replies.py`. Requests go through the real FastAPI app,
Planner, Clarifier, Executor and Context Engine; only the LLM's *answer* is
fixed per case (no LLM key needed).
"""

from fastapi.testclient import TestClient

import graph.workflow as workflow
from api.server import app
from context import context_engine
from llm import LLMResponseError

client = TestClient(app)
passed = 0
failed = 0
all_bodies = []

ESP32_ID, RELAY_ID, LED_ID, TEMP_A_ID, TEMP_B_ID = 101, 102, 103, 111, 112
DEFAULT_SIZE, GAP = 100, 50


def check(label, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"PASS: {label}")
    else:
        failed += 1
        print(f"FAIL: {label} {detail}")


def component(instance_id, label, x, y=200, width=None, height=None):
    c = {"id": instance_id, "name": label, "label": label, "x": x, "y": y}
    if width is not None:
        c.update(width=width, height=height)
    return c


def canvas(components, connections=()):
    return {"tabId": 1, "tabName": "Simulation 1", "components": components, "connections": list(connections)}


def wire(a, b):
    return {"connectionKey": f"{a}-{b}", "sourceId": a, "targetId": b}


_real_analyze = workflow.analyze


def chat(command, llm_answer=None, canvas_state=None):
    """POST /api/agent/chat with the LLM's answer fixed to `llm_answer` (all other fields empty)."""
    answer = {"intent": "", "component": "", "components": [], "source_component": "",
              "target_component": "", "confidence": "low", "reasoning": "test", **(llm_answer or {})}
    workflow.analyze = lambda prompt, user_request: dict(answer)
    try:
        body = {"command": command}
        if canvas_state is not None:
            body["canvas_state"] = canvas_state
        result = client.post("/api/agent/chat", json=body).json()
        all_bodies.append(result)
        return result
    finally:
        workflow.analyze = _real_analyze


def methods(body):
    return [c["method"] for c in body.get("commands", [])]


def add_positions(body):
    return [c["params"]["position"] for c in body.get("commands", []) if c["method"] == "canvas.addComponent"]


def overlaps(position, other):
    """Whether a new DEFAULT_SIZE component at `position` comes within GAP of `other` (a canvas component or position)."""
    ow, oh = other.get("width") or DEFAULT_SIZE, other.get("height") or DEFAULT_SIZE
    x, y = position["x"], position["y"]
    return x < other["x"] + ow + GAP and other["x"] < x + DEFAULT_SIZE + GAP and y < other["y"] + oh + GAP and other["y"] < y + DEFAULT_SIZE + GAP


def no_overlap(positions, existing):
    """No new position overlaps an existing component or another new position."""
    for i, position in enumerate(positions):
        if any(overlaps(position, other) for other in existing):
            return False
        if any(overlaps(position, other) for other in positions[:i]):
            return False
    return True


ESP_AND_RELAY = [component(ESP32_ID, "ESP32", 200), component(RELAY_ID, "Relay Module", 400)]
ESP_RELAY_LED = ESP_AND_RELAY + [component(LED_ID, "LED", 600)]
TWO_TEMPS = [component(ESP32_ID, "ESP32", 200), component(TEMP_A_ID, "Temperature Sensor", 400), component(TEMP_B_ID, "Temperature Sensor", 600)]

# --- 1. Not a canvas command -> reply_type "info", status success, nothing done ---
context_engine.clear()
body = chat("Hi Hema, how are you today?")
check("1. greeting -> status success, reply_type info", body["status"] == "success" and body["reply_type"] == "info", body)
check("1. greeting -> friendly reply with examples",
      body["message"].startswith("Hi! I can help you build your circuit") and "Add ESP32" in body["message"], body["message"])
check("1. greeting -> no commands, not a question", body["commands"] == [] and body["needs_clarification"] is False, body)

body = chat("What is the weather like?", {"intent": "add_component"})
check("1b. small talk with an LLM guess -> info reply, no 'Hi!'",
      body["reply_type"] == "info" and body["message"].startswith("I can help you build your circuit"), body)

body = chat("I can't connect to the Wi-Fi", {"intent": "connect_components"})
check("1c. verb mid-sentence ('I can't connect to the Wi-Fi') -> info, not a canvas question",
      body["reply_type"] == "info" and body["needs_clarification"] is False and body["commands"] == [], body)

# --- 2. Incomplete commands -> reply_type "question", needs_clarification, no commands ---
body = chat("connect esp32", {"intent": "connect_components", "source_component": "ESP32"}, canvas(ESP_AND_RELAY))
check("2a. connect with one component -> asks which one",
      body["message"] == "Connect ESP32 to which component?" and body["reply_type"] == "question" and body["needs_clarification"] is True, body)
check("2a. status error and no command sent", body["status"] == "error" and body["commands"] == [], body)

body = chat("connect", {"intent": "connect_components"}, canvas(ESP_AND_RELAY))
check("2b. connect with no components -> asks for two", body["message"] == "Which two components do you want to connect?" and body["reply_type"] == "question", body)

body = chat("Could you please connect them?", {"intent": "connect_components"}, canvas(ESP_AND_RELAY))
check("2c. polite command with no components -> still a question", body["reply_type"] == "question", body)

body = chat("add", {"intent": "add_component"}, canvas([]))
check("2d. add with no component -> asks which one", body["message"] == "Which component do you want to add?" and body["reply_type"] == "question", body)

# --- 3. Disconnect with one component named: the canvas decides ---
body = chat("disconnect esp32", {"intent": "disconnect_components", "source_component": "ESP32"},
            canvas(ESP_AND_RELAY, [wire(ESP32_ID, RELAY_ID)]))
cmds = [c for c in body["commands"] if c["method"] == "canvas.disconnectComponents"]
check("3a. only one wire -> disconnect command generated", body["status"] == "success" and body["reply_type"] == "info" and len(cmds) == 1, body)
check("3a. reply names both components, without claiming the IDE did it",
      body["message"] == "Generated a command to disconnect ESP32 from Relay Module (its only connection).", body["message"])
check("3a. uses that wire's exact key and ids",
      cmds and cmds[0]["params"] == {"sourceComponentId": "ESP32", "targetComponentId": "Relay Module",
                                     "sourceId": ESP32_ID, "targetId": RELAY_ID, "connectionKey": f"{ESP32_ID}-{RELAY_ID}"}, cmds)

body = chat("disconnect esp32", {"intent": "disconnect_components", "source_component": "ESP32"},
            canvas(ESP_RELAY_LED, [wire(ESP32_ID, RELAY_ID), wire(ESP32_ID, LED_ID)]))
check("3b. two wires -> asks which one, naming both",
      body["reply_type"] == "question" and body["message"] == "ESP32 is connected to Relay Module and LED. Which one do you want to disconnect it from?", body)
check("3b. no command sent, both wires kept", body["commands"] == [] and len(context_engine.get_context()["connections"]) == 2, body)

body = chat("disconnect esp32", {"intent": "disconnect_components", "source_component": "ESP32"}, canvas(ESP_AND_RELAY))
check("3c. no wires -> error, not a question", body["message"] == "ESP32 isn't connected to anything." and body["reply_type"] == "error", body)

body = chat("disconnect esp32", {"intent": "disconnect_components", "source_component": "ESP32"}, canvas([]))
check("3d. component not on canvas -> not-found error",
      body["message"] == "Cannot disconnect ESP32 because ESP32 was not found." and body["reply_type"] == "error" and body["commands"] == [], body)

# One wire, but no canvas_state: the connection key is unknown -> fails safely, never success.
context_engine.clear()
chat("Add ESP32 and Relay Module")
chat("Connect ESP32 to Relay Module", {"intent": "connect_components", "source_component": "ESP32", "target_component": "Relay Module"})
body = chat("disconnect esp32", {"intent": "disconnect_components", "source_component": "ESP32"})
check("3e. only wire but no connection key -> error, no command",
      body["status"] == "error" and body["reply_type"] == "error" and body["commands"] == [] and "cannot be identified" in body["message"], body)
check("3e. connection kept in agent memory", len(context_engine.get_context()["connections"]) == 1, context_engine.get_context()["connections"])

# --- 4. Duplicate instances: readable labels, ask which one ---
body = chat("disconnect temperature sensor", {"intent": "disconnect_components", "source_component": "Temperature Sensor"},
            canvas(TWO_TEMPS, [wire(ESP32_ID, TEMP_A_ID)]))
check("4a. ambiguous disconnect -> question with readable labels",
      body["reply_type"] == "question"
      and f"Temperature Sensor #1 (id {TEMP_A_ID}, at x=400, y=200)" in body["message"]
      and f"Temperature Sensor #2 (id {TEMP_B_ID}, at x=600, y=200)" in body["message"], body["message"])

body = chat("disconnect esp32", {"intent": "disconnect_components", "source_component": "ESP32"},
            canvas(TWO_TEMPS, [wire(ESP32_ID, TEMP_A_ID)]))
cmds = [c for c in body["commands"] if c["method"] == "canvas.disconnectComponents"]
check("4b. only wire goes to a duplicate -> reply says which instance",
      body["message"] == f"Generated a command to disconnect ESP32 from Temperature Sensor #1 (id {TEMP_A_ID}, at x=400, y=200) (its only connection).", body["message"])
check("4b. command uses the real name, never an internal key",
      cmds and cmds[0]["params"]["targetComponentId"] == "Temperature Sensor" and cmds[0]["params"]["targetId"] == TEMP_A_ID, cmds)

body = chat("disconnect esp32", {"intent": "disconnect_components", "source_component": "ESP32"},
            canvas(TWO_TEMPS, [wire(ESP32_ID, TEMP_A_ID), wire(ESP32_ID, TEMP_B_ID)]))
check("4c. wires to both duplicates -> question telling them apart",
      body["reply_type"] == "question" and "Temperature Sensor #1" in body["message"] and "Temperature Sensor #2" in body["message"], body["message"])

two_esp = [component(ESP32_ID, "ESP32", 200), component(ESP32_ID + 50, "ESP32", 400), component(RELAY_ID, "Relay Module", 600)]
body = chat("connect esp32", {"intent": "connect_components", "source_component": "ESP32"}, canvas(two_esp))
check("4d. connect with an ambiguous component -> asks which instance first",
      body["reply_type"] == "question" and body["message"].startswith("More than one ESP32 is on the canvas: ESP32 #1"), body["message"])

for verb, intent, method in (("move", "move_component", "canvas.moveComponent"), ("remove", "remove_component", "canvas.removeComponent")):
    body = chat(f"{verb} temperature sensor", {"intent": intent, "component": "Temperature Sensor"}, canvas(TWO_TEMPS))
    check(f"4e. {verb} an ambiguous component -> question with labels, nothing sent",
          body["reply_type"] == "question" and body["needs_clarification"] is True and "Temperature Sensor #2" in body["message"]
          and method not in methods(body), body)

# --- 5. Placement: real sizes, never overlapping, always finds a spot ---
context_engine.clear()
body = chat("Add ESP32", canvas_state=canvas([]))
check("5a. empty canvas -> default spot", add_positions(body) == [{"x": 200, "y": 200}], body)

existing = [component(ESP32_ID, "ESP32", 200)]
body = chat("Add Relay Module", canvas_state=canvas(existing))
check("5b. default spot taken -> next spot to the right", add_positions(body) == [{"x": 400, "y": 200}], body)

existing = [component(ESP32_ID, "ESP32", 200, width=300, height=100)]
body = chat("Add Relay Module", canvas_state=canvas(existing))
check("5c. 300px-wide component -> placed clear of its real width", add_positions(body) == [{"x": 600, "y": 200}] and no_overlap(add_positions(body), existing), body)

existing = [component(ESP32_ID, "ESP32", 200, width=100, height=300)] + [component(300 + i, "LED", 400 + 200 * i) for i in range(4)]
body = chat("Add Relay Module", canvas_state=canvas(existing))
check("5d. 300px-tall component + full row -> next row, clear of its real height",
      add_positions(body) == [{"x": 400, "y": 400}] and no_overlap(add_positions(body), existing), body)

existing = [component(400 + r * 10 + c, "LED", 200 + c * 200, 200 + r * 200) for r in range(5) for c in range(5)]
body = chat("Add Relay Module", canvas_state=canvas(existing))
check("5e. whole 5x5 area taken -> keeps searching to the first free row", add_positions(body) == [{"x": 200, "y": 1200}] and no_overlap(add_positions(body), existing), body)

existing = [component(ESP32_ID, "ESP32", 0, 0, width=2000, height=600)]
body = chat("Add Relay Module", canvas_state=canvas(existing))
check("5f. one huge component covering the search area -> placed below it", add_positions(body) == [{"x": 200, "y": 800}] and no_overlap(add_positions(body), existing), body)

existing = [component(ESP32_ID, "ESP32", 200, width=300, height=100), component(LED_ID, "LED", 800, 200, width=150, height=250)]
body = chat("Add ESP32, Relay Module and LED", canvas_state=canvas(existing))
positions = add_positions(body)
check("5g. three adds in one request on a busy canvas -> three distinct, non-overlapping spots",
      len(positions) == 3 and no_overlap(positions, existing), positions)

context_engine.clear()
chat("Add ESP32")
body = chat("Add Relay Module")
check("5h. no canvas_state: avoids the component the agent just added", add_positions(body) == [{"x": 400, "y": 200}], body)
memory = {key: c["position"] for key, c in context_engine.get_context()["components"].items()}
check("5h. agent memory records the moved position", memory.get("Relay Module") == {"x": 400, "y": 200}, memory)

# --- 6. Unexpected errors -> safe messages, no internals ---
def raising(exc):
    def analyze(prompt, user_request):
        raise exc
    return analyze


workflow.analyze = raising(RuntimeError("401 Unauthorized: OLLAMA_API_KEY=sk-SECRET-123"))
try:
    r = client.post("/api/agent/chat", json={"command": "connect esp32 to relay module please"})
finally:
    workflow.analyze = _real_analyze
body = r.json()
all_bodies.append(body)
check("6a. LLM failure -> HTTP 200, error reply", r.status_code == 200 and body["status"] == "error" and body["reply_type"] == "error", body)
check("6a. no secret or internal text in the response", "SECRET" not in r.text and "OLLAMA_API_KEY" not in r.text and "401" not in r.text, r.text)

workflow.analyze = raising(LLMResponseError("LLM response is not valid JSON: Expecting value: line 1 column 1"))
try:
    body = client.post("/api/agent/chat", json={"command": "connect esp32 to relay module"}).json()
finally:
    workflow.analyze = _real_analyze
all_bodies.append(body)
check("6b. unparseable LLM reply -> asks to rephrase, no parser details",
      body["message"] == "I couldn't understand that request. Please rephrase it." and "Expecting value" not in str(body), body)

# The executor holds bound tool methods, so the crash is injected into its dispatch table.
_dispatch = workflow._tool_executor._dispatch
_real_add = _dispatch["add_component"]
_dispatch["add_component"] = lambda **kwargs: (_ for _ in ()).throw(KeyError("INTERNAL_SETTING"))
try:
    context_engine.clear()
    body = chat("Add ESP32")
finally:
    _dispatch["add_component"] = _real_add
check("6c. tool crash -> generic step error, no exception text",
      body["status"] == "error" and body["message"] == "add_component failed because of an internal error." and "INTERNAL_SETTING" not in str(body), body)

_real_resolve = workflow.resolve_unplanned
workflow.resolve_unplanned = lambda *args: (_ for _ in ()).throw(RuntimeError("clarifier internals"))
try:
    body = chat("Hi there")
finally:
    workflow.resolve_unplanned = _real_resolve
check("6d. clarifier crash -> safe error reply, HTTP 200 body", body["reply_type"] == "error" and "internals" not in body["message"], body)

r = client.post("/api/agent/chat", json={"command": "   "})
check("6e. validation error -> 400 with reply fields",
      r.status_code == 400 and r.json() == {"status": "error", "message": "Command is required", "reply_type": "error", "needs_clarification": False}, r.text)

# --- 7. Every response: reply_type is valid and needs_clarification always agrees with it ---
check("7. reply_type always present and valid", all(b.get("reply_type") in ("info", "question", "error") for b in all_bodies),
      [b for b in all_bodies if b.get("reply_type") not in ("info", "question", "error")])
check("7. needs_clarification == (reply_type == 'question') in every response",
      all(b.get("needs_clarification") is (b.get("reply_type") == "question") for b in all_bodies),
      [b for b in all_bodies if b.get("needs_clarification") is not (b.get("reply_type") == "question")])
check("7. status is success exactly when reply_type is info",
      all((b["status"] == "success") == (b["reply_type"] == "info") for b in all_bodies),
      [b for b in all_bodies if (b["status"] == "success") != (b["reply_type"] == "info")])

context_engine.clear()
print(f"\n{passed} passed, {failed} failed")
