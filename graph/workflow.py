"""LangGraph multi-agent workflow: Intent -> Planner -> Executor -> Verifier, with retry/replan.

Architecture:
    START -> Intent Agent -> Planner Agent -> Executor Agent -> Verifier Agent
                                  ^                                   |
                                  |----------------- retry -----------|
                                                                       |
                                                          success/failure -> Context Update -> END

Each agent has one real responsibility, not just a renamed pipeline step:
    - Intent Agent: understand the request (rule-based component extraction
      first; the LLM, via llm.analyze(), only when that isn't enough) and
      hand the Planner a structured intent.
    - Planner Agent: turn that intent into an ordered execution plan
      (planner.create_plan(), unchanged) on the first pass; on a retry, it
      decides whether to retry the failed step or skip it as already-safe -
      it never re-calls the Intent Agent/LLM, and it never executes tools.
    - Executor Agent: run the plan step by step via the existing
      ToolExecutor, through the existing Bridge/MCP command builder,
      resuming from wherever the last attempt stopped.
    - Verifier Agent: classify a failure (verification.verify(), unchanged)
      into continue/retry/recover/stop and decide the route: success,
      retry (bounded by MAX_RETRIES so the loop can never run forever), or
      failure.

Every module this graph calls (KnowledgeLoader, prompt_builder.build_prompt,
llm.analyze, planner.create_plan, tool_executor.ToolExecutor,
verification.verify, context_engine.get_context, bridge's MCP command log)
already existed and is reused unchanged - this file only adds the
orchestration and the retry/replan routing around them.
"""

import logging
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from bridge import clear_recorded_commands, get_recorded_commands
from context import context_engine
from knowledge_loader import KnowledgeLoader
from llm import LLMResponseError, analyze
from planner import create_plan, extract_and_resolve_components, is_pure_add_request
from prompt_builder import build_prompt
from tool_executor import ToolExecutor
from verification import verify

logger = logging.getLogger(__name__)

# Caps how many times the Verifier can route a failure back to the Planner
# for a retry/replan, regardless of what verification.verify() decides - the
# one thing standing between a persistently-failing step and an infinite
# loop. 2 retries (3 attempts total for that step) is enough for a
# transient/recoverable error to resolve without letting a genuinely broken
# step spin forever.
MAX_RETRIES = 2


class WorkflowState(TypedDict, total=False):
    """Shared state threaded through every agent.

    Attributes:
        user_request: The raw request text (the only required input field).
        knowledge: KnowledgeLoader().load_all()'s result.
        prompt: prompt_builder.build_prompt()'s result (only set when the
            Intent Agent actually had to call the LLM).
        llm_output: The Intent Agent's structured result, in the same shape
            create_plan() has always accepted - kept under this name too
            (alongside "intent") so main.py's existing response-building
            code keeps working unchanged.
        intent: Same value as llm_output - the structured information handed
            to the Planner Agent.
        components: Flat list of every component name the Intent Agent
            identified, for state visibility/logging (the Planner still
            consumes the richer `intent`/`llm_output`, not this list).
        execution_plan: planner.create_plan()'s result. Set once, on the
            first planning pass, and never rebuilt on retry.
        current_step_index: How many steps of execution_plan have been
            attempted so far - lets the Executor Agent resume mid-plan
            after a retry instead of re-running completed steps.
        current_step: The most recently attempted step's result.
        completed_steps: Every step result that succeeded, across all
            attempts.
        tool_results: Every step result from every attempt, including
            failures that were later retried - the full execution history.
        failed_step: The step that just failed ({"index", "tool", "status",
            "error"}), or None once resolved.
        verification_result: The Verifier Agent's latest decision:
            {"outcome": "success"|"retry"|"failure", "decision": ...,
            "failed_step": ...}.
        retry_count: How many retries have been used so far this request.
        mcp_commands: Every MCP command generated this request (mirrors
            bridge.py's ui_commands.json for this request only).
        context: A context_engine.get_context() snapshot, taken at the end.
        status: "completed" or "failed" once the workflow finishes.
        error: A human-readable error message, set only when status == "failed".
    """

    user_request: str
    knowledge: Dict[str, Any]
    prompt: str
    llm_output: Dict[str, Any]
    intent: Dict[str, Any]
    components: List[str]
    execution_plan: List[Dict[str, Any]]
    current_step_index: int
    current_step: Optional[Dict[str, Any]]
    completed_steps: List[Dict[str, Any]]
    tool_results: List[Dict[str, Any]]
    failed_step: Optional[Dict[str, Any]]
    verification_result: Dict[str, Any]
    retry_count: int
    mcp_commands: List[Dict[str, Any]]
    context: Dict[str, Any]
    status: str
    error: Optional[str]


# One shared ToolExecutor instance for this graph, mirroring main.py's own
# module-level `tool_executor = ToolExecutor()`. ToolExecutor is a stateless
# dispatcher, so reusing one instance across requests is safe.
_tool_executor = ToolExecutor()


def _components_from_intent(intent: Dict[str, Any]) -> List[str]:
    """Best-effort flat component list from a structured intent, for state visibility only.

    Never consulted by the Planner Agent (which still reads the richer
    `intent`/`llm_output` dict via create_plan()) - this only feeds the
    "components" state field and log lines.
    """
    if not isinstance(intent, dict):
        return []

    if isinstance(intent.get("components"), list):
        return [c for c in intent["components"] if isinstance(c, str) and c]

    if intent.get("component"):
        return [intent["component"]]

    names: List[str] = []
    for action in intent.get("actions") or []:
        if not isinstance(action, dict):
            continue
        if isinstance(action.get("components"), list):
            names.extend(c for c in action["components"] if isinstance(c, str) and c)
        elif action.get("component"):
            names.append(action["component"])
    return names


def intent_agent(state: WorkflowState) -> Dict[str, Any]:
    """Understand the request: load knowledge, then extract intent/components.

    Rule-based extraction (planner.extract_and_resolve_components(), the fix
    for the multi-component parsing issue) is tried first for a plain "add
    X, Y and Z" request - if it applies, the LLM is never called at all.
    Everything else still goes through llm.analyze() exactly as before. This
    is the only agent that calls the LLM; Planner/Executor/Verifier never do.
    """
    if state.get("status") == "failed":
        return {}

    user_request = state.get("user_request", "")
    logger.info("[Intent Agent] understanding request: %r", user_request)
    clear_recorded_commands()

    try:
        knowledge = KnowledgeLoader().load_all()
    except Exception as exc:
        logger.error("[Intent Agent] failed to load knowledge: %s", exc)
        return {"status": "failed", "error": f"Failed to load knowledge: {exc}"}

    if is_pure_add_request(user_request):
        components = extract_and_resolve_components(user_request, knowledge)
        if components:
            intent = {"intent": "add_component", "components": components}
            logger.info("[Intent Agent] rule-based extraction handled this request (no LLM call): %s", components)
            return {
                "knowledge": knowledge,
                "intent": intent,
                "llm_output": intent,
                "components": components,
            }

    try:
        prompt = build_prompt(knowledge, user_request)
        llm_output = analyze(prompt, user_request)
    except LLMResponseError as exc:
        logger.error("[Intent Agent] LLM response error: %s", exc)
        return {"knowledge": knowledge, "status": "failed", "error": f"LLM response error: {exc}"}
    except Exception as exc:
        logger.error("[Intent Agent] unexpected error calling the LLM: %s", exc)
        return {"knowledge": knowledge, "status": "failed", "error": f"Unexpected error while calling the LLM: {exc}"}

    components = _components_from_intent(llm_output)
    logger.info("[Intent Agent] understood via LLM: intent=%s components=%s", llm_output, components)
    return {
        "knowledge": knowledge,
        "prompt": prompt,
        "intent": llm_output,
        "llm_output": llm_output,
        "components": components,
    }


def _extract_actions(llm_output: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
    """Return llm_output["actions"] if it's the multi-intent shape, else None."""
    actions = llm_output.get("actions")
    if isinstance(actions, list) and actions:
        return actions
    return None


def _create_plan_for_action(action: Any, index: int) -> List[Dict[str, Any]]:
    """Run one multi-intent action through the existing, unmodified planner.create_plan()."""
    if not isinstance(action, dict):
        logger.warning("Multi-intent action %d is not a JSON object, skipping: %r", index, action)
        return []
    return create_plan(action)


def _plan_multi_intent(actions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Planner(Action 1) -> Planner(Action 2) -> ... -> Merge Plans, order-preserving."""
    merged: List[Dict[str, Any]] = []
    for index, action in enumerate(actions):
        plan = _create_plan_for_action(action, index)
        if not plan:
            logger.warning("Multi-intent action %d produced no plan, skipping: %s", index, action)
            continue
        merged.extend(plan)
    return merged


def planner_agent(state: WorkflowState) -> Dict[str, Any]:
    """Build the plan on the first pass; decide retry-vs-skip on a replan.

    First pass: converts the Intent Agent's structured output into an
    ordered plan via planner.create_plan() (multi-intent actions merged via
    _plan_multi_intent(), or a single create_plan() call otherwise) -
    unchanged planning logic, just invoked from here now.

    Replan (only reached when the Verifier routed here with a "retry"
    outcome): the plan itself is never rebuilt and the LLM is never
    re-consulted - it only adjusts current_step_index, using the Verifier's
    classification of *why* the step failed: "recover" (e.g. "already
    exists") means the desired end state was already reached, so the failed
    step is skipped; "retry" (e.g. a transient timeout) means the same step
    should simply be attempted again.
    """
    if state.get("status") == "failed":
        return {}

    verification_result = state.get("verification_result")
    if verification_result and verification_result.get("outcome") == "retry":
        decision = verification_result.get("decision")
        failed_step = verification_result.get("failed_step") or {}
        current_index = state.get("current_step_index", 0)

        if decision == "recover":
            logger.info(
                "[Planner Agent] replanning: %r's failure is recoverable - skipping it and continuing",
                failed_step.get("tool"),
            )
            return {"current_step_index": current_index + 1, "failed_step": None}

        logger.info("[Planner Agent] replanning: retrying %r after a transient failure", failed_step.get("tool"))
        return {"current_step_index": current_index, "failed_step": None}

    intent = state.get("intent") or {}
    actions = _extract_actions(intent)

    try:
        if actions is not None:
            plan = _plan_multi_intent(actions)
        else:
            plan = create_plan(intent, user_request=state.get("user_request", ""), knowledge=state.get("knowledge"))
    except Exception as exc:
        logger.error("[Planner Agent] failed to build a plan: %s", exc)
        return {"status": "failed", "error": f"Failed to create execution plan: {exc}"}

    if not plan:
        logger.warning("[Planner Agent] no execution plan could be created for: %s", intent)
        return {
            "execution_plan": [],
            "current_step_index": 0,
            "status": "failed",
            "error": "No execution plan could be created for this request.",
        }

    logger.info("[Planner Agent] built plan with %d step(s): %s", len(plan), plan)
    return {"execution_plan": plan, "current_step_index": 0}


def executor_agent(state: WorkflowState) -> Dict[str, Any]:
    """Run execution_plan from current_step_index onward via the existing ToolExecutor.

    Stops at the first failure in this pass (ToolExecutor.execute_plan()'s
    existing behavior, unchanged) - it doesn't decide what to do about a
    failure, that's the Verifier/Planner's job. Every MCP command generated
    while these steps ran (via the existing Bridge/MCPCommandBuilder) is
    captured into state["mcp_commands"].
    """
    if state.get("status") == "failed":
        return {}

    plan = state.get("execution_plan") or []
    start_index = state.get("current_step_index", 0)
    remaining = plan[start_index:]

    logger.info("[Executor Agent] executing %d step(s) starting at plan index %d", len(remaining), start_index)

    commands_before = len(get_recorded_commands())
    try:
        results = _tool_executor.execute_plan(remaining)
    except Exception as exc:
        logger.error("[Executor Agent] failed to execute plan: %s", exc)
        return {"status": "failed", "error": f"Failed to execute plan: {exc}"}
    new_commands = get_recorded_commands()[commands_before:]

    failed_step = None
    successes = results
    if results and results[-1]["status"] == "failed":
        failed_step = {"index": start_index + len(results) - 1, **results[-1]}
        successes = results[:-1]

    logger.info(
        "[Executor Agent] this pass: %d succeeded, %s",
        len(successes),
        f"failed at index {failed_step['index']} ({failed_step.get('tool')})" if failed_step else "no failures",
    )

    return {
        "tool_results": list(state.get("tool_results") or []) + results,
        "completed_steps": list(state.get("completed_steps") or []) + successes,
        "current_step": (results[-1] if results else state.get("current_step")),
        "current_step_index": start_index + len(successes),
        "failed_step": failed_step,
        "mcp_commands": list(state.get("mcp_commands") or []) + new_commands,
    }


def verifier_agent(state: WorkflowState) -> Dict[str, Any]:
    """Classify the Executor's outcome and decide: success, retry, or failure.

    No failed_step -> success. A failed_step is classified via the existing
    verification.verify() ("continue"/"retry"/"recover"/"stop"); "retry" or
    "recover" routes back to the Planner as long as retry_count hasn't hit
    MAX_RETRIES, otherwise (or on "stop") this is a hard failure.
    """
    if state.get("status") == "failed":
        return {}

    failed_step = state.get("failed_step")
    retry_count = state.get("retry_count", 0)

    if failed_step is None:
        logger.info("[Verifier Agent] all steps in this pass succeeded")
        return {"verification_result": {"outcome": "success"}, "status": "completed"}

    decision = verify(failed_step)["decision"]

    if decision in ("retry", "recover") and retry_count < MAX_RETRIES:
        logger.warning(
            "[Verifier Agent] step %r failed (decision=%s) - routing to Planner for retry %d/%d",
            failed_step.get("tool"),
            decision,
            retry_count + 1,
            MAX_RETRIES,
        )
        return {
            "verification_result": {"outcome": "retry", "decision": decision, "failed_step": failed_step},
            "retry_count": retry_count + 1,
        }

    logger.error(
        "[Verifier Agent] step %r failed (decision=%s) - giving up after %d retr%s",
        failed_step.get("tool"),
        decision,
        retry_count,
        "y" if retry_count == 1 else "ies",
    )
    return {
        "verification_result": {"outcome": "failure", "decision": decision, "failed_step": failed_step},
        "status": "failed",
        "error": failed_step.get("error") or f"Step {failed_step.get('tool')!r} failed",
    }


def _route_after_verifier(state: WorkflowState) -> str:
    """Conditional-edge function: where the Verifier Agent sends the graph next."""
    if state.get("status") == "failed":
        return "failure"
    return (state.get("verification_result") or {}).get("outcome", "failure")


def context_update_node(state: WorkflowState) -> Dict[str, Any]:
    """Read the current Context Engine snapshot into the state.

    Bookkeeping only - not one of the four agents - since it has no
    decision to make: the Context Engine is already updated as a side
    effect of ToolExecutor.execute_step(), this just captures the resulting
    snapshot via the existing context_engine.get_context().
    """
    return {"context": context_engine.get_context()}


def build_workflow() -> StateGraph:
    """Assemble the multi-agent LangGraph workflow.

    Returns:
        An uncompiled StateGraph. Call .compile() to get a runnable graph
        (see `compiled_workflow` below for a ready-made one).
    """
    graph = StateGraph(WorkflowState)

    graph.add_node("intent_agent", intent_agent)
    graph.add_node("planner_agent", planner_agent)
    graph.add_node("executor_agent", executor_agent)
    graph.add_node("verifier_agent", verifier_agent)
    graph.add_node("context_update", context_update_node)

    graph.add_edge(START, "intent_agent")
    graph.add_edge("intent_agent", "planner_agent")
    graph.add_edge("planner_agent", "executor_agent")
    graph.add_edge("executor_agent", "verifier_agent")

    graph.add_conditional_edges(
        "verifier_agent",
        _route_after_verifier,
        {
            "success": "context_update",
            "retry": "planner_agent",
            "failure": "context_update",
        },
    )

    graph.add_edge("context_update", END)

    return graph


# A ready-to-use compiled graph, mirroring how tool_executor.py exposes a
# shared default instance - callers can `.invoke(...)` this directly.
compiled_workflow = build_workflow().compile()


def run_workflow(user_request: str) -> Dict[str, Any]:
    """Run one user request through the LangGraph-orchestrated multi-agent pipeline.

    This is the LangGraph equivalent of main.handle_request(): same
    response-relevant fields, same fail-fast-on-unrecoverable-error
    behavior - only the sequencing (and now retry/replan routing) is done
    by LangGraph instead of a hand-written function.

    Args:
        user_request: The user's natural-language request.

    Returns:
        The final WorkflowState, as a plain dict.
    """
    initial_state: WorkflowState = {"user_request": user_request}
    final_state = compiled_workflow.invoke(initial_state)
    return dict(final_state)
