"""Tests for POST /api/agent/voice: reply fields, optional canvas_state, and consistency with chat.

Plain script (matches the other test_*.py files, not pytest) - run with
`python test_voice.py`. The speech-to-text call (voice_input.transcribe_audio_bytes)
and the LLM's answer are fixed per case, so no audio, Groq key or LLM key is
needed; everything after transcription runs for real (normalizer, workflow,
Context Engine).
"""

import io
import json

from fastapi.testclient import TestClient

import graph.workflow as workflow
import voice_input
from api.server import app
from context import context_engine

client = TestClient(app)
passed = 0
failed = 0

ESP32_ID, RELAY_ID = 101, 102
REPLY_KEYS = {"status", "message", "commands", "results", "reply_type", "needs_clarification"}


def check(label, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"PASS: {label}")
    else:
        failed += 1
        print(f"FAIL: {label} {detail}")


def component(instance_id, label, x, y=200):
    return {"id": instance_id, "name": label, "label": label, "x": x, "y": y}


def canvas(components, connections=()):
    return {"tabId": 1, "tabName": "Simulation 1", "components": components, "connections": list(connections)}


ESP_AND_RELAY = [component(ESP32_ID, "ESP32", 200), component(RELAY_ID, "Relay Module", 400)]
CONNECTED = canvas(ESP_AND_RELAY, [{"connectionKey": f"{ESP32_ID}-{RELAY_ID}", "sourceId": ESP32_ID, "targetId": RELAY_ID}])

_real_transcribe = voice_input.transcribe_audio_bytes
_real_analyze = workflow.analyze
transcribe_calls = []


def voice(transcript, llm_answer=None, canvas_state=None, raw_canvas_state=None):
    """POST /api/agent/voice with STT fixed to `transcript` and the LLM's answer fixed to `llm_answer`."""
    def fake_transcribe(audio_bytes):
        transcribe_calls.append(len(audio_bytes))
        return transcript

    answer = {"intent": "", "component": "", "components": [], "source_component": "",
              "target_component": "", "confidence": "high", "reasoning": "test", **(llm_answer or {})}
    voice_input.transcribe_audio_bytes = fake_transcribe
    workflow.analyze = lambda prompt, user_request: dict(answer)
    data = {}
    if canvas_state is not None:
        data["canvas_state"] = json.dumps(canvas_state)
    if raw_canvas_state is not None:
        data["canvas_state"] = raw_canvas_state
    try:
        return client.post("/api/agent/voice", files={"audio": ("clip.webm", io.BytesIO(b"\x1a\x45\xdf\xa3audio"), "audio/webm")}, data=data)
    finally:
        voice_input.transcribe_audio_bytes = _real_transcribe
        workflow.analyze = _real_analyze


def add_positions(body):
    return [c["params"]["position"] for c in body.get("commands", []) if c["method"] == "canvas.addComponent"]


# --- 1. Without canvas_state: works as before, now with reply fields ---
context_engine.clear()
r = voice("Add ESP32")
body = r.json()
check("1. no canvas_state -> success", r.status_code == 200 and body["status"] == "success", body)
check("1. reply_type info, needs_clarification false", body["reply_type"] == "info" and body["needs_clarification"] is False, body)
check("1. transcript and normalized command returned", body["transcript"] == "Add ESP32" and body["normalized_command"], body)
check("1. add command generated", [c["method"] for c in body["commands"]] == ["canvas.addComponent"], body)

# --- 2. With canvas_state: synced before planning ---
context_engine.clear()
body = voice("Add Relay Module", canvas_state=canvas([component(ESP32_ID, "ESP32", 200)])).json()
check("2a. canvas_state used for placement (ESP32 at 200 -> Relay at 400)", add_positions(body) == [{"x": 400, "y": 200}], body)
check("2a. agent memory now matches the sent canvas", "ESP32" in context_engine.get_context()["components"], context_engine.get_context()["components"])

body = voice("Disconnect ESP32 from Relay Module",
             {"intent": "disconnect_components", "source_component": "ESP32", "target_component": "Relay Module"}, CONNECTED).json()
cmds = [c for c in body["commands"] if c["method"] == "canvas.disconnectComponents"]
check("2b. voice disconnect with canvas_state -> exact key and ids",
      body["status"] == "success" and cmds and cmds[0]["params"].get("connectionKey") == f"{ESP32_ID}-{RELAY_ID}"
      and cmds[0]["params"].get("sourceId") == ESP32_ID, body)

body = voice("Disconnect ESP32 from Relay Module",
             {"intent": "disconnect_components", "source_component": "ESP32", "target_component": "Relay Module"}, canvas(ESP_AND_RELAY)).json()
check("2c. canvas_state says not connected -> error, no command",
      body["status"] == "error" and body["reply_type"] == "error" and body["commands"] == [] and "not connected" in body["message"], body)

# --- 3. Questions and info replies through voice ---
body = voice("connect ESP32", {"intent": "connect_components", "source_component": "ESP32"}, canvas(ESP_AND_RELAY)).json()
check("3a. incomplete voice command -> question with needs_clarification",
      body["reply_type"] == "question" and body["needs_clarification"] is True and body["message"] == "Connect ESP32 to which component?", body)

body = voice("Hi Hema, how are you today?").json()
check("3b. small talk by voice -> info reply, status success", body["reply_type"] == "info" and body["status"] == "success" and body["commands"] == [], body)

# --- 4. Bad canvas_state -> 400 before any transcription ---
transcribe_calls.clear()
r = voice("Add ESP32", raw_canvas_state="{not json")
check("4a. malformed canvas_state JSON -> 400 error reply", r.status_code == 400 and r.json()["reply_type"] == "error" and r.json()["needs_clarification"] is False, r.text)
check("4a. audio not transcribed for a rejected request", transcribe_calls == [], transcribe_calls)
r = voice("Add ESP32", raw_canvas_state=json.dumps({"components": [{"name": "ESP32"}]}))
check("4b. canvas_state missing required fields -> 400", r.status_code == 400 and "Invalid canvas_state" in r.json()["message"], r.text)
r = voice("Add ESP32", raw_canvas_state="   ")
check("4c. blank canvas_state treated as absent -> success", r.status_code == 200 and r.json()["status"] == "success", r.text)

# --- 5. Voice-only early failures carry the reply fields too ---
body = voice(None).json()
check("5a. transcription failed -> error reply fields", body["message"] == "Unable to transcribe audio" and body["reply_type"] == "error" and body["needs_clarification"] is False, body)
body = voice("add the ZQ99 board").json()
check("5b. unresolved component -> error reply fields with transcript",
      body["message"] == "Could not confidently identify component. Please repeat." and body["reply_type"] == "error" and body["transcript"] == "add the ZQ99 board", body)

import command_normalizer

_real_normalize = command_normalizer.normalize_command
command_normalizer.normalize_command = lambda transcript: None
try:
    body = voice("esp32 esp32 esp32 esp32 esp32").json()
finally:
    command_normalizer.normalize_command = _real_normalize
check("5c. speech not understood -> error reply fields with transcript",
      body["message"] == "Speech not understood" and body["reply_type"] == "error" and body["needs_clarification"] is False
      and body["transcript"] == "esp32 esp32 esp32 esp32 esp32", body)

# --- 6. Same command, chat vs voice: same reply fields ---
context_engine.clear()
voice_body = voice("Add ESP32", canvas_state=canvas([])).json()
context_engine.clear()
chat_body = client.post("/api/agent/chat", json={"command": "Add ESP32", "canvas_state": canvas([])}).json()
check("6. voice = chat + transcript/normalized_command",
      set(chat_body) == REPLY_KEYS and set(voice_body) == REPLY_KEYS | {"transcript", "normalized_command"}, (sorted(chat_body), sorted(voice_body)))
check("6. same status, message, reply_type and commands",
      all(voice_body[k] == chat_body[k] for k in ("status", "message", "reply_type", "needs_clarification", "commands")), (chat_body, voice_body))

context_engine.clear()
print(f"\n{passed} passed, {failed} failed")
