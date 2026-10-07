"""Thin entry point: hands every request off to the LangGraph workflow.

Architecture (Phase 3.2):
    User -> main.py -> LangGraph Workflow -> Knowledge Loader -> Prompt Builder
          -> LLM -> Planner -> Tool Executor -> Verification -> Context Update
          -> Bridge

LangGraph (graph/workflow.py, built in Phase 3.1) is now the sole runtime
orchestrator - it, not this file, sequences Knowledge Loader, Prompt
Builder, LLM, Planner, Tool Executor, and Verification. main.py only:
    1. Calls graph.run_workflow(user_request).
    2. Adapts the workflow's result into the same {"status", "results",
       "state", ...} response shape callers already depend on.
No planning, execution, or verification decisions are made here - that
business logic still lives entirely inside the existing modules; this file
does not duplicate any of it.
"""

import logging

from graph import run_workflow
from state import AgentState
from tools.canvas_tool import CanvasTool

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def _describe_intent(llm_output: dict) -> str:
    """Render llm_output's intent(s) as one string for AgentState.intent.

    AgentState.intent (state.py, out of scope for this phase) is a single
    string field - it was always just llm_output["intent"] verbatim. Since
    Phase 3.4, llm_output may instead be the multi-intent
    {"actions": [...]} shape (see llm.py), which has no top-level
    "intent" of its own; without this, AgentState.intent would silently go
    blank for a multi-intent request. This only formats what the workflow
    already decided (each action's own "intent", already resolved by
    planner_node) - it makes no planning or intent decision itself.
    """
    actions = llm_output.get("actions")
    if isinstance(actions, list) and actions:
        intents = [action.get("intent", "") for action in actions if isinstance(action, dict)]
        return ",".join(intents)

    return llm_output.get("intent", "")


def _build_state(workflow_result: dict) -> AgentState:
    """Translate a WorkflowState result into an AgentState.

    This exists purely to keep the response's "state" field in the same
    shape callers already depend on - it makes no planning, execution, or
    verification decisions itself; it only mirrors what the workflow
    already decided.
    """
    state = AgentState()
    state.set_user_request(workflow_result.get("user_request", ""))

    llm_output = workflow_result.get("llm_output") or {}
    state.set_intent(_describe_intent(llm_output))

    state.set_plan(workflow_result.get("execution_plan") or [])

    for result in workflow_result.get("tool_results") or []:
        if result.get("status") == "success":
            state.mark_step_completed(result)
        else:
            state.mark_step_failed(result)
        state.add_tool_result(result)

    state.set_status(workflow_result.get("status", "failed"))
    return state


def handle_request(user_request: str) -> dict:
    """Run one user request through the LangGraph workflow and return the outcome.

    Returns:
        {"status": ..., "results": [...], "state": {...}}, with an "error"
        key added whenever status is "failed". This is the same response
        shape callers already depend on - only the orchestration underneath
        has moved into LangGraph (graph.workflow.run_workflow()).
        This function never raises - a failure anywhere in the workflow is
        reported here, not propagated.
    """
    logger.info("Received user request: %r", user_request)

    try:
        workflow_result = run_workflow(user_request)
    except Exception as exc:
        logger.error("Workflow execution failed: %s", exc)
        state = AgentState()
        state.set_user_request(user_request)
        state.set_status("failed")
        return {
            "status": "failed",
            "results": [],
            "error": f"Workflow execution failed: {exc}",
            "state": state.to_dict(),
        }

    state = _build_state(workflow_result)
    logger.info("Request handled with status=%r", workflow_result.get("status"))

    response = {
        "status": workflow_result.get("status", "failed"),
        "results": workflow_result.get("tool_results", []),
        "state": state.to_dict(),
    }

    error = workflow_result.get("error")
    if error:
        response["error"] = error

    return response


def run_multi_intent(user_request: str) -> dict:
    """Run one (possibly multi-action) request, in a {"status", "steps",
    "final_canvas"} response shape.

    Purely additive alongside handle_request(): it runs the exact same
    LangGraph-orchestrated pipeline (run_workflow() - Knowledge Loader ->
    Prompt Builder -> LLM -> Planner -> Tool Executor -> Verification ->
    Context Update, entirely unchanged) and only repackages the result
    differently, for callers that want this compact shape instead of
    handle_request()'s {"status", "results", "state"}. Nothing about the
    pipeline itself is duplicated or altered - both entry points share one
    implementation.

    "final_canvas" always reflects the Context Engine's current state via
    CanvasTool.get_canvas_state() (the same accessor tool_executor.py's own
    get_canvas_state step uses), fetched fresh after the workflow finishes -
    not merely whatever the plan's own steps happened to return - so it's
    populated even for a request whose plan never explicitly asked to see
    the canvas.

    Returns:
        {"status": "completed", "steps": [...], "final_canvas": {...}} on
        success, or {"status": "failed", "steps": [...], "error": str,
        "final_canvas": {...}} if any step failed or the workflow itself
        raised. This function never raises.
    """
    logger.info("Received multi-intent request: %r", user_request)

    try:
        workflow_result = run_workflow(user_request)
    except Exception as exc:
        logger.error("Multi-intent workflow execution failed: %s", exc)
        return {
            "status": "failed",
            "steps": [],
            "error": f"Workflow execution failed: {exc}",
            "final_canvas": CanvasTool().get_canvas_state(),
        }

    status = workflow_result.get("status", "failed")
    response = {
        "status": "completed" if status == "completed" else "failed",
        "steps": workflow_result.get("tool_results", []),
        "final_canvas": CanvasTool().get_canvas_state(),
    }

    error = workflow_result.get("error")
    if error:
        response["error"] = error

    logger.info("Multi-intent request handled with status=%r", response["status"])
    return response


def _run_and_print(request_text: str) -> None:
    """Run one request through handle_request() and print its JSON result."""
    import json

    outcome = handle_request(request_text)
    print(json.dumps(outcome, indent=2))


if __name__ == "__main__":
    import sys

    print("AI Drag-and-Drop Agent Ready")

    # A single CLI argument still runs exactly one request and exits -
    # this preserves the existing non-interactive invocation
    # (`python main.py "Add ESP32"`), e.g. for scripts/tests. Only the
    # no-argument case becomes the new interactive loop below.
    if len(sys.argv) > 1:
        _run_and_print(" ".join(sys.argv[1:]))
    else:
        # The process (and with it, the module-level Context Engine
        # singleton in context/context_engine.py) stays alive for as long
        # as this loop runs, so the canvas session persists across
        # commands within one interactive run - "Move it" after "Add
        # ESP32" sees ESP32, exactly like calling handle_request() twice
        # in the same script already does.
        while True:
            try:
                request_text = input("Enter request (or 'voice' to speak it): ").strip()
            except (KeyboardInterrupt, EOFError):
                print()
                print("Goodbye!")
                break

            if not request_text:
                continue

            if request_text.lower() in ("exit", "quit"):
                print("Goodbye!")
                break

            if request_text.lower() == "voice":
                # Imported lazily so a normal text-only run never needs
                # speech_recognition installed at all - only typing "voice"
                # pulls it in, and get_voice_input() itself falls back to
                # input() if that import (or anything else about voice
                # capture) fails.
                from voice_input import get_voice_input

                request_text = get_voice_input()
                if not request_text:
                    continue

            _run_and_print(request_text)
