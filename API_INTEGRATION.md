# InnoIDE AI Agent — FastAPI Integration Guide

This is a thin HTTP layer around the existing Python AI agent (LangGraph: Intent → Planner → Executor → Verifier). It does not touch the canvas directly — every response contains a list of **MCP commands** that your React/Redux code maps to existing canvas operations.

## Frontend responsibility

The FastAPI backend performs all AI interpretation (speech-to-text, natural-language understanding, planning, execution, verification) and hands back a list of MCP commands. The React/InnoIDE frontend is responsible for:

1. Sending typed text commands to `POST /api/agent/chat`.
2. Capturing microphone audio and sending it to `POST /api/agent/voice`.
3. Reading the returned `commands` array.
4. Mapping MCP command methods (`canvas.addComponent`, etc.) to the existing InnoIDE canvas operations.
5. Updating the canvas/UI based on those commands.
6. Showing the transcript / normalized command / status to the user where appropriate.
7. Showing a loading state while the AI agent is processing (see "Response time" under Remaining limitations — this can take a while).

The React team does **not** need to implement any of the following — they're entirely on the backend:

- Whisper (speech-to-text)
- LangGraph
- Planner
- Tool Executor
- Verifier
- MCP Bridge

## Starting the server

```bash
cd Agentt
venv\Scripts\activate          # Windows
python -m uvicorn api.server:app --host 0.0.0.0 --port 8001 --reload
```

This is the current recommended development command — it runs the API on **port 8001**. `--host 0.0.0.0` means it listens on every network interface, not just `localhost`, which is what lets a frontend on a different machine reach it (see "Base URL" below).

> **Note:** `python api/server.py` also works and now starts on port 8001 too (matching the command above), via that file's own startup block.

Requires the same local Ollama server (`qwen2.5:7b`) the CLI already needs for anything beyond a plain "add X, Y and Z" request. Starting this server does **not** replace or interfere with the existing CLI (`python main.py`) — both can run at the same time, independently.

Nothing about the API is hard-coded to port 8001 specifically — it's just the port this environment currently runs on. Deploy it on whatever host/port makes sense for you; just update the URLs below (and `INNOIDE_FRONTEND_ORIGIN`, see CORS) to match.

## Base URL

```
http://localhost:8001
```

If React and FastAPI are running **on the same machine** (typical local development), use `http://localhost:8001`.

If React and FastAPI are running **on different machines** (e.g. testing from a phone/tablet on the same network, or a separate frontend dev machine), `localhost` won't work from the frontend's side — point it at the backend machine's actual IP address instead:

```
http://<BACKEND-PC-IP>:8001
```

Replace `<BACKEND-PC-IP>` with the IP address of the machine running the FastAPI server (e.g. `http://192.168.1.42:8001`). This is also why the recommended start command above uses `--host 0.0.0.0` — binding to `localhost` only would refuse connections from other machines entirely.

## Interactive API docs

FastAPI auto-generates these — useful for the React team to poke around without reading this doc line by line:

- `http://localhost:8001/docs` (Swagger UI)
- `http://localhost:8001/redoc`

---

## `GET /api/health`

Health check — no auth, no body.

**Response 200:**
```json
{ "status": "ok", "service": "InnoIDE AI Agent" }
```

---

## `POST /api/agent/chat`

**Purpose:** send a user's typed natural-language InnoIDE command to the AI agent.

**Example request:**
```json
{
  "command": "Disconnect ESP32 from Relay Module",
  "canvas_state": {
    "tabId": 1,
    "tabName": "Simulation 1",
    "components": [
      { "id": 1775642938124.512, "name": "Microcontroller", "label": "ESP32", "x": 200, "y": 200, "width": 100, "height": 100 },
      { "id": 1775642941100.234, "name": "Relay Module", "label": "Relay Module", "x": 400, "y": 200, "width": 100, "height": 100 }
    ],
    "connections": [
      { "connectionKey": "1775642938124.512-1775642941100.234", "sourceId": 1775642938124.512, "targetId": 1775642941100.234 }
    ]
  }
}
```

| Field | Required | Meaning |
|---|---|---|
| `command` | yes | The user's text. |
| `canvas_state` | no, but **send it with every request** | The active tab's real canvas. The agent replaces its own memory with it before planning. `connectionKey` must be exactly the key stored in Redux for that wire; disconnect cannot work without it. `width`/`height` are optional (100×100 is assumed) and are used to place new components without overlap. Without `canvas_state` the agent uses its own memory, which may not match the canvas. |

Behind this endpoint, the backend runs:

```
Text
→ LangGraph
→ Planner
→ Tool Executor
→ Verifier
→ MCP Bridge
```

The frontend receives the resulting MCP commands in the response — it never needs to know any of the above happened.

### Response fields

Every response from `/api/agent/chat` and `/api/agent/voice` (including 400/500 errors) has these fields:

| Field | Meaning |
|---|---|
| `status` | `"success"` or `"error"`. |
| `message` | Text to show the user. |
| `commands` | MCP commands to apply to the canvas (see "Supported MCP command methods"). Empty when nothing should change. |
| `results` | Raw tool results, for debugging/logging only. |
| `reply_type` | What kind of reply `message` is — see below. |
| `needs_clarification` | `true` exactly when `reply_type` is `"question"`, otherwise `false`. |

| `reply_type` | `status` | When | What the frontend should do |
|---|---|---|---|
| `"info"` | `"success"` | Commands were generated, **or** the message wasn't a canvas command (e.g. "Hi") and there was nothing to do. | Apply `commands` (if any); show `message` normally. |
| `"question"` | `"error"` | The agent needs more information (missing component, or two components share a name). Nothing was done. | Show `message` as a question; let the user answer with a new command. |
| `"error"` | `"error"` | The request couldn't be done (component not found, not connected, service problem). Nothing was done. | Show `message` as an error. |

`"success"` means the backend **generated** the commands — not that the canvas changed. The frontend applies them.

**Response 200 — commands generated:**
```json
{
  "status": "success",
  "message": "Command generated successfully",
  "commands": [
    { "method": "canvas.addComponent", "params": { "componentId": "ESP32", "position": { "x": 200, "y": 200 } } },
    { "method": "canvas.addComponent", "params": { "componentId": "DHT11 Sensor", "position": { "x": 400, "y": 200 } } }
  ],
  "results": [ /* raw tool-execution results, mostly useful for debugging/logging */ ],
  "reply_type": "info",
  "needs_clarification": false
}
```

**Response 200 — agent chose for the user** (e.g. "disconnect ESP32" when ESP32 has exactly one wire): `message` says what was chosen.
```json
{
  "status": "success",
  "message": "Generated a command to disconnect ESP32 from Relay Module (its only connection).",
  "commands": [
    { "method": "canvas.disconnectComponents", "params": { "sourceComponentId": "ESP32", "targetComponentId": "Relay Module", "sourceId": 1775642938124.512, "targetId": 1775642941100.234, "connectionKey": "1775642938124.512-1775642941100.234" } }
  ],
  "results": [ /* ... */ ],
  "reply_type": "info",
  "needs_clarification": false
}
```

**Response 200 — not a canvas command** ("Hi Hema, how are you today?"):
```json
{
  "status": "success",
  "message": "Hi! I can help you build your circuit on the canvas. Try \"Add ESP32\", \"Connect ESP32 to Relay Module\" or \"Show canvas\".",
  "commands": [],
  "results": [],
  "reply_type": "info",
  "needs_clarification": false
}
```

**Response 200 — question** ("connect ESP32"):
```json
{
  "status": "error",
  "message": "Connect ESP32 to which component?",
  "commands": [],
  "results": [],
  "reply_type": "question",
  "needs_clarification": true
}
```
When two components share a name, the question lists each one so they can be told apart, e.g. `"More than one Temperature Sensor is on the canvas: Temperature Sensor #1 (id 1775642950000.111, at x=400, y=200) and Temperature Sensor #2 (id 1775642960000.222, at x=600, y=200). Which one do you want to disconnect, and from which component?"`

**Response 200 — error** (the agent couldn't complete the request):
```json
{
  "status": "error",
  "message": "Cannot disconnect ESP32 from Relay Module because they are not connected.",
  "commands": [],
  "results": [ /* whatever steps did run before the failure */ ],
  "reply_type": "error",
  "needs_clarification": false
}
```
Note this is still HTTP 200 — the request itself was valid, the *agent* just couldn't finish. Always check the `status` field, not just the HTTP status code (see "Error handling" below).

**Response 400 (bad request):**
```json
{ "status": "error", "message": "Command is required", "reply_type": "error", "needs_clarification": false }
```
Returned for a missing/blank `"command"`, malformed JSON, or an invalid `canvas_state`.

**Response 500 (unexpected server error):** `{ "status": "error", "message": "Internal server error", "reply_type": "error", "needs_clarification": false }` — the real exception is logged server-side only, never sent to the client. Agent-level failures (e.g. the language service being unavailable) return HTTP 200 with a fixed, safe `message`; exception details are never included.

---

## `POST /api/agent/voice`

**Important:** the end user does **not** manually upload a WAV file. That's only how this doc's `curl` examples exercise the endpoint for testing. The real frontend flow is:

```
User presses microphone
→ React captures microphone audio
→ React creates/encodes an audio blob
→ React sends multipart/form-data
→ POST /api/agent/voice
→ Speech-to-Text
→ Command Normalization
→ SAME LangGraph workflow
→ Planner
→ Tool Executor
→ Verifier
→ MCP Bridge
→ React/InnoIDE canvas
```

**There is no separate voice planner.** Voice and text both end up running through the exact same LangGraph/Planner/Executor/Verifier/MCP Bridge pipeline — the only voice-specific steps are speech-to-text and command normalization, both of which happen *before* that shared pipeline ever runs. A command typed as text and the same command spoken and transcribed correctly are handled identically from that point on.

### Audio format requirement

Speech-to-text uses Groq's hosted Whisper API. The backend accepts what browsers record — **WebM, OGG, MP4** — plus **WAV, FLAC and MP3**, up to **25 MB**, English only. The format is detected from the file's own bytes, so browser `MediaRecorder` output can be sent as-is; no conversion is needed.

Do not tell the end user to manually upload a file — recording and sending happens automatically in the React app.

### Request

`multipart/form-data` with these fields:

| Field | Required | Meaning |
|---|---|---|
| `audio` | yes | The recording. |
| `canvas_state` | no, but **send it with every request** | The same JSON object as chat's `canvas_state`, sent as a **JSON string**. Synced into the agent before planning, exactly as for chat. An invalid value returns HTTP 400 before the audio is transcribed; an empty value is treated as absent. |

**curl example (for testing):**
```bash
curl -X POST http://localhost:8001/api/agent/voice \
  -F "audio=@recording.webm;type=audio/webm" \
  -F 'canvas_state={"tabId":1,"tabName":"Simulation 1","components":[{"id":101,"name":"Microcontroller","label":"ESP32","x":200,"y":200}],"connections":[]}'
```

### Response fields

The same fields as chat (`status`, `message`, `commands`, `results`, `reply_type`, `needs_clarification` — see "Response fields" under `/api/agent/chat`), plus:

| Field | Meaning |
|---|---|
| `transcript` | The raw speech-to-text result, before any correction (`null` if transcription failed). |
| `normalized_command` | The transcript after component-name normalization (e.g. "ESP thirty two" → "ESP32") — this is the text actually handed to the LangGraph agent. |

**Response 200 (commands generated):**
```json
{
  "status": "success",
  "message": "Command generated successfully",
  "commands": [
    { "method": "canvas.addComponent", "params": { "componentId": "Relay Module", "position": { "x": 400, "y": 200 } } }
  ],
  "results": [ /* ... */ ],
  "reply_type": "info",
  "needs_clarification": false,
  "transcript": "Add Relay Module",
  "normalized_command": "Add Relay Module"
}
```

**Response 200 (question):**
```json
{
  "status": "error",
  "message": "Connect ESP32 to which component?",
  "commands": [],
  "results": [],
  "reply_type": "question",
  "needs_clarification": true,
  "transcript": "connect esp32",
  "normalized_command": "connect ESP32"
}
```

**Response 200 (transcription failed)** — silence, too short, empty or over 25 MB, or the speech service failed:
```json
{ "status": "error", "message": "Unable to transcribe audio", "commands": [], "results": [], "reply_type": "error", "needs_clarification": false, "transcript": null, "normalized_command": null }
```

**Response 200 (speech recognized but not understood)** — e.g. a severely corrupted/repetitive transcription:
```json
{ "status": "error", "message": "Speech not understood", "commands": [], "results": [], "reply_type": "error", "needs_clarification": false, "transcript": "...", "normalized_command": null }
```

**Response 200 (component name unclear)** — the transcript looked like it named a component, but nothing in the catalog was a confident match:
```json
{ "status": "error", "message": "Could not confidently identify component. Please repeat.", "commands": [], "results": [], "reply_type": "error", "needs_clarification": false, "transcript": "...", "normalized_command": null }
```

**Response 400 (invalid `canvas_state`):**
```json
{ "status": "error", "message": "Invalid canvas_state: send the same JSON object as the chat API's canvas_state.", "reply_type": "error", "needs_clarification": false }
```

### Real example: multi-component + relationship understanding

**User voice command:** *"Add ESP32 and connect it to the accelerometer."*

**Actual normalized command:** `"Add ESP32 and connect it to the Accelerometer."`

**The API generated three MCP commands, in order:**
```json
{ "method": "canvas.addComponent", "params": { "componentId": "ESP32", "position": { "x": 200, "y": 200 } } }
```
```json
{ "method": "canvas.addComponent", "params": { "componentId": "Accelerometer", "position": { "x": 400, "y": 200 } } }
```
```json
{ "method": "canvas.connectComponents", "params": { "sourceComponentId": "ESP32", "targetComponentId": "Accelerometer" } }
```

This one example demonstrates the full round trip: **voice → multiple-component understanding → component addition → relationship understanding → connection → MCP commands** — all from a single spoken sentence, with no special-casing on the frontend side.

### Frontend JavaScript example

Sending a recorded audio blob together with the current canvas:

```js
const formData = new FormData();
formData.append("audio", audioBlob, "command.webm");
formData.append("canvas_state", JSON.stringify(currentCanvasState)); // same object as chat's canvas_state

const response = await fetch(
  "http://localhost:8001/api/agent/voice",
  {
    method: "POST",
    body: formData
  }
);

const data = await response.json();
```

Do **not** manually set a `Content-Type` header when sending `FormData` — the browser sets the correct `multipart/form-data` boundary automatically, and overriding it will break the upload.

Then inspect the response:

```js
data.status               // "success" | "error"
data.reply_type            // "info" | "question" | "error"
data.needs_clarification   // true when reply_type is "question"
data.transcript            // raw speech-to-text result
data.normalized_command    // command after normalization
data.commands               // MCP commands to apply to the canvas
data.results                 // raw tool results (debugging only)
```

The React team is responsible for connecting their own microphone-recording implementation (e.g. `MediaRecorder`, capturing a `Blob`) to this endpoint — that recording logic isn't part of this backend and isn't detailed here beyond the format requirement above.

---

## Supported MCP command methods

These come from the existing `mcp/command_builder.py` — this list is exhaustive, nothing else will ever appear in `"commands"`. **The frontend must use these exact method names:**

| Method | params |
|---|---|
| **`canvas.addComponent`** | `{ "componentId": str, "position": { "x": int, "y": int } }` — `position` is a free spot (no overlap with existing components) |
| **`canvas.removeComponent`** | `{ "componentId": str, "id"?: instanceId }` |
| **`canvas.moveComponent`** | `{ "componentId": str, "position": { "x": int, "y": int }, "id"?: instanceId }` |
| **`canvas.connectComponents`** | `{ "sourceComponentId": str, "targetComponentId": str, "sourceId"?: instanceId, "targetId"?: instanceId }` |
| **`canvas.disconnectComponents`** | `{ "sourceComponentId": str, "targetComponentId": str, "connectionKey": str, "sourceId"?: instanceId, "targetId"?: instanceId }` — remove the wire by `connectionKey` |
| **`canvas.getState`** | `{}` |

Fields marked `?` are included when the component came from `canvas_state` (so its instance ID is known). Name fields are always the component's real name, never an internal key. A disconnect command is only ever sent with the exact `connectionKey` from `canvas_state`; without it the request fails instead.

**Important — exact names matter:**
- It's **`canvas.removeComponent`**, **not** `canvas.deleteComponent`.
- It's **`canvas.getState`**, **not** `canvas.getCanvasState`.

These are the real, current method names produced by the backend and are not being renamed for this integration. If your existing Redux/canvas code expects the other names, please reconcile the naming on the frontend side (or raise it with the backend team if you'd rather standardize the other way) — but for now, match what's actually sent.

---

## CORS

Configured via the `INNOIDE_FRONTEND_ORIGIN` environment variable (comma-separated for multiple origins). Defaults to `http://localhost:3000` if unset — there is **no wildcard/unsafe default**.

**The value must match your actual React dev server origin exactly**, including the port. Don't assume it's `3000` — e.g. Vite's default is `5173`. If your React app runs on `http://localhost:5173`:

```cmd
:: Windows CMD
set INNOIDE_FRONTEND_ORIGIN=http://localhost:5173
python -m uvicorn api.server:app --host 0.0.0.0 --port 8001 --reload
```

```powershell
# Windows PowerShell
$env:INNOIDE_FRONTEND_ORIGIN = "http://localhost:5173"
python -m uvicorn api.server:app --host 0.0.0.0 --port 8001 --reload
```

If it's not set correctly, the browser will block requests with a CORS error even though `curl`/Postman work fine (they don't enforce CORS, so this class of problem is easy to miss when testing outside the browser).

**The React team should provide their actual development frontend URL** so CORS can be configured correctly on the backend — please don't assume a default port.

---

## Error handling

**The frontend must check `data.status`, not just `response.ok`.** Many agent-level failures are returned as HTTP 200 with `"status": "error"` in the body, because the *request* was valid — the AI agent just couldn't complete it. `response.ok` alone will not catch these.

| Case | HTTP status | `status` | `reply_type` | Example `message` |
|---|---|---|---|---|
| Commands generated | 200 | `"success"` | `"info"` | `"Command generated successfully"` |
| Not a canvas command (greeting, small talk) | 200 | `"success"` | `"info"` | `"Hi! I can help you build your circuit on the canvas. Try ..."` |
| Missing information / two components share a name | 200 | `"error"` | `"question"` | `"Connect ESP32 to which component?"` |
| Agent couldn't complete the request | 200 | `"error"` | `"error"` | `"Cannot disconnect ESP32 from Relay Module because they are not connected."` |
| Disconnect without the canvas connection key | 200 | `"error"` | `"error"` | `"Cannot disconnect ESP32 from Relay Module because the connection cannot be identified: the current canvas connection key is unavailable."` |
| Language service unavailable | 200 | `"error"` | `"error"` | `"The language service is unavailable right now. Please try again in a moment."` |
| Bad request (missing/blank command, malformed JSON, invalid `canvas_state`) | 400 | `"error"` | `"error"` | `"Command is required"` |
| Voice: transcription failed | 200 | `"error"` | `"error"` | `"Unable to transcribe audio"` |
| Voice: speech recognized but not understood | 200 | `"error"` | `"error"` | `"Speech not understood"` |
| Voice: component name unclear | 200 | `"error"` | `"error"` | `"Could not confidently identify component. Please repeat."` |
| Unexpected server error | 500 | `"error"` | `"error"` | `"Internal server error"` (the real exception is logged server-side only) |

These are the actual messages the API returns today — this table won't be extended with hypothetical messages the backend doesn't produce.

---

## Remaining limitations (be aware of these)

1. **Voice audio formats** — WebM, OGG, MP4, WAV, FLAC or MP3, up to 25 MB, English only.
2. **Send `canvas_state` with every chat and voice request.** Without it the agent works from its own memory, which can differ from the canvas, and **disconnect always fails** (it needs the canvas `connectionKey`).
3. **Placement assumes 100×100 components positioned by their top-left corner** unless `width`/`height` are sent. `canvas_state` has no canvas width/height, so the backend doesn't know the canvas edges: a new component goes up to 4 spots (800 px) to the right of the default spot, then on the next row down, until it fits — on a very full canvas that can be below the visible area.
4. **"Success" means commands were generated, not applied.** The backend has no confirmation that the canvas executed them.
5. **Answering a question about duplicates:** the question identifies each instance (number, id, position), but the agent can only act on an answer that names the instance unambiguously.
6. **Agent-level errors may return HTTP 200** with `"status": "error"` in the body — always check `data.status`, not `response.ok`.
7. **MCP method names must match exactly** — `canvas.removeComponent`/`canvas.getState`, not `canvas.deleteComponent`/`canvas.getCanvasState`.
8. **Some backend/tool behavior is still mock behavior** where applicable — e.g. an unrecognized component name may still produce a `canvas.addComponent` command under a fallback name rather than being rejected outright, since no real canvas is wired up on the backend yet.
9. **LLM-based commands can take several seconds or longer**, depending on local Ollama/CPU performance — plan your frontend's loading state accordingly. A plain multi-component "add" is typically answered in well under a second (handled by rule-based extraction, no LLM call at all); anything more complex (e.g. involving "connect") may take noticeably longer.

---

## Frontend Integration Checklist

- [ ] Start FastAPI backend on port 8001
- [ ] Verify `/api/health`
- [ ] Open `/docs`
- [ ] Configure CORS for the React frontend origin
- [ ] Test `POST /api/agent/chat`
- [ ] Test `POST /api/agent/voice`
- [ ] Send `canvas_state` (with each wire's exact Redux `connectionKey`) on every chat request
- [ ] Capture microphone audio in React
- [ ] Send `multipart/form-data` with field named `"audio"`
- [ ] Send `canvas_state` as a JSON string field on every voice request
- [ ] Read `data.status`
- [ ] Read `data.reply_type` — show `"question"` replies as questions and `"info"` replies normally, not as errors
- [ ] Read `transcript`
- [ ] Read `normalized_command`
- [ ] Process `data.commands`
- [ ] Map `canvas.addComponent`
- [ ] Map `canvas.removeComponent`
- [ ] Map `canvas.moveComponent`
- [ ] Map `canvas.connectComponents`
- [ ] Map `canvas.disconnectComponents` (remove the wire by `params.connectionKey`)
- [ ] Map `canvas.getState`
- [ ] Handle errors/loading state
