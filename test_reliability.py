"""Reliability regression tests: voice work stays off the event loop, LLM requests time out, timeouts fail cleanly.

Plain script (matches the other test_*.py files, not pytest) - run with
`python test_reliability.py`. Starts the real app with uvicorn on a free
local port, because a blocked event loop only shows with real concurrent
requests. Slow speech-to-text and a stalling LLM server are faked locally -
no network, no API keys.
"""

import http.server
import io
import os
import socket
import threading
import time

import httpx
import ollama
import uvicorn

import api.server as server
import config
import graph.workflow as workflow
import llm
import voice_input
from context import context_engine

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


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _Stalling(http.server.BaseHTTPRequestHandler):
    """An LLM server that never answers in time."""

    def do_POST(self):
        time.sleep(5)
        try:
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"{}")
        except Exception:
            pass

    def log_message(self, *args):
        pass


stalling = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Stalling)
threading.Thread(target=stalling.serve_forever, daemon=True).start()
STALLING_HOST = f"http://127.0.0.1:{stalling.server_address[1]}"

app_server = uvicorn.Server(uvicorn.Config(server.app, host="127.0.0.1", port=free_port(), log_level="error"))
threading.Thread(target=app_server.run, daemon=True).start()
while not app_server.started:
    time.sleep(0.05)
BASE = f"http://127.0.0.1:{app_server.config.port}"

_real_transcribe = voice_input.transcribe_audio_bytes
_real_analyze = workflow.analyze
_real_get_client = llm.get_client


def voice_request(**data):
    return httpx.post(f"{BASE}/api/agent/voice", files={"audio": ("clip.webm", io.BytesIO(b"\x1a\x45\xdf\xa3audio"), "audio/webm")},
                      data=data, timeout=60)


# The slow fakes take SLOW seconds; a /api/health answer counts as "not blocked"
# under HEALTH_LIMIT. Wide margins both ways keep this stable on a busy machine:
# unblocked, health answers in milliseconds; blocked, it waits most of SLOW.
SLOW = 4
HEALTH_LIMIT = 2.0


def health_seconds_during(request_fn, wait=1.0):
    """Start request_fn in the background, then time /api/health while it runs. Returns (health seconds, request response)."""
    holder = {}
    worker = threading.Thread(target=lambda: holder.setdefault("response", request_fn()))
    worker.start()
    time.sleep(wait)
    start = time.time()
    health = httpx.get(f"{BASE}/api/health", timeout=60)
    elapsed = time.time() - start
    worker.join()
    return elapsed, health, holder.get("response")


# --- 1. LLM timeout configuration ---
check("1a. default LLM timeout is 60 s", config.LLM_TIMEOUT_SECONDS == 60.0, config.LLM_TIMEOUT_SECONDS)
_saved_env = os.environ.get("LLM_TIMEOUT_SECONDS")
for raw, expected in (("5", 5.0), ("12.5", 12.5), ("abc", 60.0), ("0", 60.0), ("-3", 60.0)):
    os.environ["LLM_TIMEOUT_SECONDS"] = raw
    check(f"1b. LLM_TIMEOUT_SECONDS={raw!r} -> {expected}", config._llm_timeout_seconds() == expected, config._llm_timeout_seconds())
if _saved_env is None:
    os.environ.pop("LLM_TIMEOUT_SECONDS", None)
else:
    os.environ["LLM_TIMEOUT_SECONDS"] = _saved_env

_saved_key = os.environ.get("OLLAMA_API_KEY")
os.environ["OLLAMA_API_KEY"] = "test-key-not-real"
try:
    client = config.get_client()
    check("1c. get_client() passes the timeout to the HTTP client",
          client._client.timeout == httpx.Timeout(config.LLM_TIMEOUT_SECONDS), client._client.timeout)
finally:
    if _saved_key is None:
        os.environ.pop("OLLAMA_API_KEY", None)
    else:
        os.environ["OLLAMA_API_KEY"] = _saved_key

# --- 2. A stalling LLM server: the real client gives up at the timeout ---
llm.get_client = lambda: ollama.Client(host=STALLING_HOST, timeout=1)
try:
    start = time.time()
    try:
        llm.analyze("PROMPT", "connect esp32 to relay module")
        outcome = None
    except Exception as exc:
        outcome = exc
    elapsed = time.time() - start
    check("2a. stalled LLM call raises httpx.TimeoutException", isinstance(outcome, httpx.TimeoutException), repr(outcome))
    check("2a. ...after about the timeout, not the stall", elapsed < 3, f"{elapsed:.1f}s")

    # --- 3. Timeouts reach the API as clean errors, never as success ---
    context_engine.clear()
    start = time.time()
    r = httpx.post(f"{BASE}/api/agent/chat", json={"command": "connect esp32 to relay module"}, timeout=60)
    body, elapsed = r.json(), time.time() - start
    check("3a. chat with a timed-out LLM -> HTTP 200 error reply",
          r.status_code == 200 and body["status"] == "error" and body["reply_type"] == "error" and body["needs_clarification"] is False, body)
    check("3a. says it timed out", body["message"] == "The language service took too long to respond. Please try again in a moment.", body["message"])
    check("3a. no commands generated", body["commands"] == [], body["commands"])
    check("3a. answered within the timeout window", elapsed < 4, f"{elapsed:.1f}s")

    voice_input.transcribe_audio_bytes = lambda audio_bytes: "connect esp32 to relay module"
    body = voice_request().json()
    check("3b. voice with a timed-out LLM -> same clean error, no commands",
          body["status"] == "error" and body["reply_type"] == "error" and body["commands"] == []
          and body["message"].startswith("The language service took too long") and body["transcript"] == "connect esp32 to relay module", body)

    r = httpx.post(f"{BASE}/api/agent/chat", json={"command": "connect esp32 to relay module"}, timeout=60)
    check("3c. no internal details in the timeout response", "ReadTimeout" not in r.text and "127.0.0.1" not in r.text, r.text)
finally:
    llm.get_client = _real_get_client
    voice_input.transcribe_audio_bytes = _real_transcribe

# --- 4. Slow speech-to-text no longer blocks other requests ---
def slow_transcribe(audio_bytes):
    time.sleep(SLOW)
    return "Add ESP32"


voice_input.transcribe_audio_bytes = slow_transcribe
try:
    context_engine.clear()
    elapsed, health, response = health_seconds_during(voice_request)
    check("4a. /api/health answers during slow speech-to-text", health.status_code == 200 and elapsed < HEALTH_LIMIT, f"{elapsed:.1f}s")
    body = response.json()
    check("4b. the slow voice request still completes correctly",
          body["status"] == "success" and body["transcript"] == "Add ESP32"
          and [c["method"] for c in body["commands"]] == ["canvas.addComponent"], body)
finally:
    voice_input.transcribe_audio_bytes = _real_transcribe


# --- 5. A slow LLM inside a voice request no longer blocks other requests ---
def slow_analyze(prompt, user_request):
    time.sleep(SLOW)
    return {"intent": "get_canvas_state", "component": "", "components": [], "source_component": "",
            "target_component": "", "confidence": "high", "reasoning": "test"}


voice_input.transcribe_audio_bytes = lambda audio_bytes: "show canvas"
workflow.analyze = slow_analyze
try:
    elapsed, health, response = health_seconds_during(voice_request)
    check("5a. /api/health answers while a voice request waits on the LLM", health.status_code == 200 and elapsed < HEALTH_LIMIT, f"{elapsed:.1f}s")
    body = response.json()
    check("5b. that voice request still completes correctly",
          body["status"] == "success" and [c["method"] for c in body["commands"]] == ["canvas.getState"], body)

    # A chat holding the workflow lock, then a voice request queues behind it: health must stay fast.
    chat_thread = threading.Thread(target=lambda: httpx.post(f"{BASE}/api/agent/chat", json={"command": "show canvas"}, timeout=60))
    chat_thread.start()
    time.sleep(0.5)
    elapsed, health, response = health_seconds_during(voice_request, wait=0.5)
    chat_thread.join()
    check("5c. health stays fast while a voice request waits for the workflow lock", elapsed < HEALTH_LIMIT, f"{elapsed:.1f}s")
    check("5c. the queued voice request still completes", response.json()["status"] == "success", response.json())
finally:
    workflow.analyze = _real_analyze
    voice_input.transcribe_audio_bytes = _real_transcribe

app_server.should_exit = True
stalling.shutdown()
context_engine.clear()
print(f"\n{passed} passed, {failed} failed")
