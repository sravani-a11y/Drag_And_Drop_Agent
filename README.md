# Agentt — AI Drag-and-Drop Canvas Agent

An AI agent that turns natural-language requests ("Add an ESP32", "connect it to the relay module, then move it right") into structured actions on a hardware/component canvas. Requests are orchestrated end-to-end by a [LangGraph](https://github.com/langchain-ai/langgraph) pipeline: a local LLM (via [Ollama](https://ollama.com/)) interprets the request against a small knowledge base, a planner turns the interpretation into an ordered list of tool calls, and a tool executor applies them to an in-memory **Context Engine** that is the canvas's single source of truth.

The Python backend is fully wired and runnable today **as a CLI**. A FastAPI/WebSocket bridge, an MCP command layer, and a React frontend exist as scaffolding for a future live-canvas integration, but are not yet connected to anything runnable — see [Current Limitations](#current-limitations).

## Table of Contents

- [Architecture](#architecture)
- [Project Layout](#project-layout)
- [Requirements](#requirements)
- [Setup](#setup)
- [Running](#running)
- [Request/Response Shapes](#requestresponse-shapes)
- [Module Reference](#module-reference)
- [Testing](#testing)
- [Current Limitations](#current-limitations)

## Architecture

Every request flows through one linear LangGraph pipeline (`graph/workflow.py`), fail-fast with no branching:

```
User request
    │
    ▼
Knowledge Loader   (knowledge_loader.py)      reads registries/*.json
    │
    ▼
Prompt Builder     (prompt_builder.py)        builds LLM prompt + schema + examples
    │
    ▼
LLM                (llm.py)                   Ollama (qwen2.5:7b), returns structured intent(s)
    │
    ▼
Planner            (planner.py)               intent(s) -> ordered [{tool, input}, ...]
    │
    ▼
Tool Executor       (tool_executor.py)         runs each step via CanvasTool / ComponentTool
    │                                          (mirrors mutations into the Context Engine)
    ▼
Verification        (verification.py)         classifies each step's result (continue/retry/recover/stop)
    │
    ▼
Context Update                                 fresh snapshot from the Context Engine
    │
    ▼
Response ({"status", "results"/"steps", "state"/"final_canvas"})
```

The **Context Engine** (`context/`) is an in-process, in-memory singleton holding canvas state (components, positions, connections, selection). It is the authoritative source for `get_canvas_state()` — the canvas state is never read back from a frontend. State does not persist across process restarts.

Tool calls are routed through `bridge.py`'s `FrontendBridge`, which today runs in `"mock"` mode: no real frontend is required for the pipeline to work, because mutations are always mirrored into the Context Engine regardless of what the bridge returns.

## Project Layout

```
main.py                    CLI entry point: handle_request() / run_multi_intent()
bridge.py                  FrontendBridge (mock/mcp modes) + unmounted WebSocket router
config.py                  Ollama model config (LLM_MODEL, get_client())
knowledge_loader.py        Loads registries/*.json (components, features, apis)
prompt_builder.py          Assembles the LLM prompt (schema + few-shot examples + knowledge)
llm.py                     Calls Ollama, normalizes the structured intent(s) response
planner.py                 Converts intent(s) into an ordered tool-call plan
tool_executor.py           Executes a plan against CanvasTool/ComponentTool
simple_tool_executor.py    Standalone demo/teaching reimplementation (not used by the real pipeline)
verification.py            Classifies each tool result (continue/retry/recover/stop)
capability_resolver.py     Resolves intent/component against features & APIs (not wired into the graph)
state.py                   AgentState — response "state" field shape
tool_selector.py           Empty (unused)

graph/
  workflow.py               LangGraph StateGraph definition and run_workflow()

context/
  context_engine.py         Canvas state mutations + get_context() (the source of truth)
  context_builder.py        Renders context/prompt as text for the LLM
  session_state.py          @dataclass backing store (in-memory, not persistent)

tools/
  canvas_tool.py             CanvasTool: search_component, get_canvas_state, calculate_snap_position
  component_tool.py          ComponentTool: add/remove/move/connect/disconnect component
  add_component.py           Older/alternate add-component tool (not wired into the executor)
  search_component.py        Standalone catalog search demo
  get_canvas_state.py         Empty (unused — real logic lives in canvas_tool.py)

mcp/
  client.py                  MCPClient — placeholder, no real wire protocol yet
  command_builder.py         Translates bridge actions into {"method","params"} MCP-style commands

registries/
  component_catalog.json     Component definitions (ESP32, STM32, sensors, etc.)
  feature_registry.json      Frontend canvas features
  api_registry.json          Backend API definitions

frontend/src/mcp/
  useMcpCanvas.js             Reducer + hook maintaining canvas state from MCP commands
  useMcpListener.js            Transport-agnostic command receiver (WebSocket wiring not yet done)
  CanvasDemo.jsx               Demo component replaying a hardcoded command sequence

test_knowledge_loader.py     Plain script (not pytest) exercising KnowledgeLoader
test_llm.py                   Plain script (not pytest); currently broken against llm.analyze()'s signature
```

## Requirements

- **Python 3.11** (the committed `venv` was built with 3.11.15)
- **[Ollama](https://ollama.com/)** running locally with the `qwen2.5:7b` model pulled:
  ```
  ollama pull qwen2.5:7b
  ```
  `config.py` connects to Ollama's default local endpoint (`http://localhost:11434`) — no API key or `.env` file is needed.
- Key Python packages (already installed in the committed `venv`; there is no `requirements.txt` yet): `ollama`, `langgraph`, `langchain-core`, `pydantic`. `fastapi`, `starlette`, `Flask`, `anthropic`, and `websockets` are also installed but not currently wired into a runnable server — see [Current Limitations](#current-limitations).

## Setup

```powershell
# From c:\Users\Sravani\Desktop\Agentt
venv\Scripts\Activate.ps1

# Make sure Ollama is running and has the model pulled
ollama pull qwen2.5:7b
```

If you need to rebuild the environment from scratch instead of using the committed `venv`, install the packages listed above with `pip install ollama langgraph langchain-core pydantic` (add `fastapi`, `websockets`, etc. if you pick up the frontend bridge work).

## Running

The backend runs as a **CLI**, not a server — there is currently no HTTP/WebSocket process to start.

**Interactive loop** (keeps the Context Engine's canvas state alive across commands, so pronouns like "it"/"them" resolve across requests):

```powershell
python main.py
# Enter request: Add ESP32
# Enter request: connect it to a relay module
# Enter request: exit
```

**One-shot** (single request, prints JSON, exits):

```powershell
python main.py "Add an ESP32 and connect it to the relay module"
```

**Programmatic use** — import the two entry points directly:

```python
from main import handle_request, run_multi_intent

handle_request("Add an ESP32")
run_multi_intent("Add an ESP32, then connect it to the relay module")
```

## Request/Response Shapes

`handle_request(user_request: str) -> dict`

```json
{
  "status": "completed" | "failed",
  "results": [ /* tool_results from the workflow */ ],
  "state": { /* AgentState.to_dict() */ },
  "error": "..."  // present only when status == "failed"
}
```

`run_multi_intent(user_request: str) -> dict` — a more compact shape for callers that want the resulting canvas directly rather than the full `AgentState`:

```json
{
  "status": "completed" | "failed",
  "steps": [ /* tool_results from the workflow */ ],
  "final_canvas": { "symbols": [...], "connections": [...] },
  "error": "..."  // present only when status == "failed"
}
```

Both entry points run the *same* LangGraph pipeline and never raise — failures are reported in the response, not propagated as exceptions. `final_canvas` is always fetched fresh from the Context Engine after the workflow finishes (`CanvasTool().get_canvas_state()`), so it reflects reality even for requests whose plan never explicitly asked to see the canvas.

## Module Reference

| Module | Responsibility |
|---|---|
| `knowledge_loader.py` | Reads `registries/component_catalog.json`, `feature_registry.json`, `api_registry.json`; tolerant of missing/invalid files (logs and returns `[]`). |
| `prompt_builder.py` | Builds the LLM prompt: system instructions, a fixed output JSON schema (`intent`, `component`, `source_component`, `target_component`, `confidence`, `reasoning`), an intent enum (`add_component`, `remove_component`, `move_component`, `connect_components`, `disconnect_components`, `get_canvas_state`), 10 few-shot examples, then the registries and the raw request. |
| `llm.py` | `analyze(prompt, user_request)` prepends the live canvas context, appends multi-action instructions, calls Ollama with `format="json"`, then normalizes intent casing/synonyms. Returns either a single intent object or `{"actions": [...]}` for multi-step requests. |
| `planner.py` | `create_plan(llm_output, capability=None, user_request="", knowledge=None)` converts LLM output into an ordered `[{"tool", "input"}, ...]` plan. Handles multi-intent, deterministic text segmentation ("then"/"and"/","/";" with pronoun resolution), and single-intent dispatch. Never calls the LLM or executes tools. |
| `tool_executor.py` | `ToolExecutor.execute_plan()` dispatches each step to `CanvasTool`/`ComponentTool`, then mirrors successful mutations into the Context Engine. Stops at the first failed step. |
| `verification.py` | `verify(tool_result)` classifies a result into `continue`/`retry`/`recover`/`stop` by inspecting the error string. Currently informational only — no retry loop consumes the decision yet. |
| `context/context_engine.py` | The canvas's source of truth: `add_component`, `remove_component`, `move_component`, `connect_components`, `disconnect_components`, `select_component`, `get_context()`, `clear()`. In-memory singleton, not persistent. |
| `capability_resolver.py` | Resolves an intent/component against `feature_registry.json`/`api_registry.json`. Implemented but not currently invoked from `graph/workflow.py`. |
| `mcp/` | Scaffolding for a future MCP integration: `MCPClient` is a placeholder with no real wire protocol; `MCPCommandBuilder` translates bridge actions into `{"method", "params"}` commands. |
| `bridge.py` | `FrontendBridge` in `"mock"` (default) or `"mcp"` mode, plus a FastAPI `APIRouter` with a `/ws/frontend` WebSocket route that is defined but not mounted into any running app. |

## Testing

There is no pytest suite yet — `test_knowledge_loader.py` and `test_llm.py` are plain scripts that call the module under test and `print()` the result:

```powershell
python test_knowledge_loader.py
python test_llm.py   # requires Ollama running with qwen2.5:7b pulled
```

Note: `test_llm.py` currently calls `llm.analyze()` with a single argument, but `analyze()` requires `(prompt, user_request)` — this script needs updating before it will run successfully.

## Current Limitations

These are gaps to be aware of before building on top of this project, not bugs in the parts that do run:

- **No runnable server.** `bridge.py` defines a FastAPI router (`/ws/frontend`), but nothing in the repo creates a `FastAPI()` app, mounts the router, or calls `uvicorn.run()`; `uvicorn` isn't even installed. The only real, runnable entry point today is the `main.py` CLI.
- **No frontend build.** `frontend/src/mcp/*.jsx`/`*.js` are reference/demo files with no `package.json`, no bundler config, and no dependency installation anywhere in the repo. `CanvasDemo.jsx` replays a hardcoded mock command sequence rather than connecting to a live backend.
- **MCP is scaffolding only.** `mcp/client.py` is an explicit placeholder (it just echoes back whatever command it "sent") — no real MCP wire protocol is implemented yet.
- **`capability_resolver.py` is dead code** in the current pipeline: it's fully implemented but `graph/workflow.py`'s planner node never passes a `capability` argument to `planner.create_plan()`.
- **Verification results aren't acted on.** `verification.verify()` classifies outcomes as `retry`/`recover`/`stop`, but the graph doesn't branch on this yet — it's recorded, not enforced.
- **Empty files**: `tool_selector.py` and `tools/get_canvas_state.py` are placeholders with no content.
- **No `requirements.txt`/`pyproject.toml`.** Dependencies are only captured in the committed `venv`; see [Requirements](#requirements) for the package list if you need to recreate it.
