"""Dispatches planned tool-execution steps to CanvasTool and ComponentTool.

Architecture:
    Planner -> Tool Executor -> Canvas Tool -> Component Tool -> Bridge
             -> Context Engine (mirrors successful operations only)

ToolExecutor dispatches: given a planned step, it calls the correct tool
method and returns that tool's result unchanged. It holds no business
logic of its own (no default values, no validation rules beyond "does this
tool exist") and never calls frontend/backend APIs directly - only the
tools it dispatches to do that.

Phase 2.4: after a step that mutates canvas state succeeds, ToolExecutor
mirrors that change into the Canvas Context Engine (context/context_engine.py)
so later requests (via ContextBuilder) are aware of it. The Bridge remains
the sole source of truth for whether an operation actually happened - the
Context Engine only mirrors what already succeeded, and is never updated
for a failed step.
"""

import logging
from typing import Any, Callable, Dict, List, Optional

from context import AmbiguousComponentError, ComponentNotFoundError, context_engine
from tools.canvas_tool import CanvasTool
from tools.component_tool import ComponentTool
from tools.connect_tool import ConnectTool

logger = logging.getLogger(__name__)

StepResult = Dict[str, Any]


class ToolExecutor:
    """Dispatches plan steps to the tool method that implements them."""

    def __init__(self) -> None:
        self.canvas_tool = CanvasTool()
        self.component_tool = ComponentTool()
        self.connect_tool = ConnectTool()

        # Maps a plan step's "tool" name to the method that implements it.
        # To register a new tool, add one entry here - no other code needs to change.
        self._dispatch: Dict[str, Callable[..., Dict[str, Any]]] = {
            "search_component": self.canvas_tool.search_component,
            "get_canvas_state": self.canvas_tool.get_canvas_state,
            "calculate_snap_position": self.canvas_tool.calculate_snap_position,
            "add_component": self.component_tool.add_component,
            "delete_component": self.component_tool.delete_component,
            "remove_component": self.component_tool.remove_component,
            "move_component": self.component_tool.move_component,
            "connect_components": self.connect_tool.connect_components,
            "disconnect_components": self.connect_tool.disconnect_components,
        }

        # Maps a tool name to the Context Engine call that mirrors its
        # effect, run only after that tool succeeds. Tools with no lasting
        # canvas-state effect (search_component, get_canvas_state,
        # calculate_snap_position, remove_component) have no entry here and
        # are simply not synced. To mirror a new tool, add one entry here -
        # no other code needs to change.
        self._context_sync: Dict[str, Callable[[Any], None]] = {
            "add_component": self._sync_add_component,
            "delete_component": self._sync_remove_component,
            "move_component": self._sync_move_component,
            "connect_components": self._sync_connect_components,
            "disconnect_components": self._sync_disconnect_components,
        }

    @staticmethod
    def _sync_add_component(input_value: Dict[str, Any]) -> None:
        context_engine.add_component(input_value["component_id"], input_value["x"], input_value["y"])

    @staticmethod
    def _sync_remove_component(input_value: str) -> None:
        context_engine.remove_component(input_value)

    @staticmethod
    def _sync_move_component(input_value: Dict[str, Any]) -> None:
        context_engine.move_component(input_value["component_id"], input_value["x"], input_value["y"])

    @staticmethod
    def _sync_connect_components(input_value: Dict[str, Any]) -> None:
        context_engine.connect_components(input_value["source_component"], input_value["target_component"])

    @staticmethod
    def _sync_disconnect_components(input_value: Dict[str, Any]) -> None:
        context_engine.disconnect_components(input_value["source_component"], input_value["target_component"])

    @staticmethod
    def _join(names: List[Any]) -> str:
        return " and ".join(str(name) for name in names)

    @staticmethod
    def _join_options(labels: List[str]) -> str:
        return labels[0] if len(labels) == 1 else f"{', '.join(labels[:-1])} or {labels[-1]}"

    @staticmethod
    def _name(reference: Any) -> Any:
        """A readable name for a step's component reference: its label when it names exactly one component."""
        labels = context_engine.matching_labels(reference)
        return labels[0] if len(labels) == 1 else reference

    @staticmethod
    def _safe_error(tool_name: str, exc: Exception) -> str:
        """The message for a tool/sync exception, safe to show the user.

        A ValueError keeps its own text (the Context Engine's user-facing
        rejections). Timeouts and connection errors get fixed wording that
        verification.py still classifies as retryable. Anything else gets a
        generic message - unexpected exception text can expose internals and
        is only logged.
        """
        if isinstance(exc, ValueError):
            return str(exc)
        if isinstance(exc, TimeoutError):
            return f"{tool_name} timed out."
        if isinstance(exc, ConnectionError):
            return f"{tool_name} failed: the service is temporarily unavailable."
        return f"{tool_name} failed because of an internal error."

    def _precheck(self, tool_name: str, input_value: Any) -> Optional[StepResult]:
        """Check, before the tool runs, that the components (and connection) a step needs exist.

        A step that would act on a missing or ambiguous component fails here
        - so no command is ever sent to the frontend for an operation that
        can't be performed. Returns the failed StepResult, or None to run
        the step. Read-only: never changes the Context Engine.
        """
        if tool_name == "connect_components":
            source, target = input_value["source_component"], input_value["target_component"]
            action, references = f"connect {self._name(source)} to {self._name(target)}", [source, target]
        elif tool_name == "disconnect_components":
            source, target = input_value["source_component"], input_value["target_component"]
            action, references = f"disconnect {self._name(source)} from {self._name(target)}", [source, target]
        elif tool_name == "move_component":
            action, references = f"move {self._name(input_value['component_id'])}", [input_value["component_id"]]
        elif tool_name == "delete_component":
            action, references = f"remove {self._name(input_value)}", [input_value]
        else:
            return None

        missing = []
        for reference in references:
            try:
                context_engine.resolve_component(reference)
            except AmbiguousComponentError:
                labels = context_engine.matching_labels(reference)
                message = (
                    f"Cannot {action} because {len(labels)} components match {reference!r} on the canvas: "
                    f"{self._join_options(labels)}. Which one do you mean?"
                )
                logger.warning("Tool %s not run: %s", tool_name, message)
                return {"tool": tool_name, "status": "failed", "error": message, "needs_clarification": True}
            except ComponentNotFoundError:
                missing.append(reference)

        if missing:
            verb = "was" if len(missing) == 1 else "were"
            message = f"Cannot {action} because {self._join(missing)} {verb} not found."
        elif tool_name == "disconnect_components" and not context_engine.connection_exists(source, target):
            message = f"Cannot {action} because they are not connected."
        else:
            return None

        logger.warning("Tool %s not run: %s", tool_name, message)
        return {"tool": tool_name, "status": "failed", "error": message}

    def _sync_context(self, tool_name: str, input_value: Any, result: Any) -> Optional[str]:
        """Mirror a successful tool's effect into the Context Engine.

        A tool can complete without raising an exception yet still report
        failure in its own result (e.g. {"success": False, "error": ...},
        which the Bridge returns when a real frontend declines a command).
        The Context Engine must never mirror that, so this checks the
        tool's own reported success in addition to "no exception was
        raised" - the Bridge remains the sole source of truth for whether
        an operation actually happened.

        Returns None on success, or the error message when the Context
        Engine refused the change - the caller reports that step as failed
        instead of claiming a success the agent's state doesn't reflect.
        """
        if isinstance(result, dict) and not result.get("success", True):
            logger.debug("Skipping Context Engine sync for %s - tool reported failure", tool_name)
            return None

        sync = self._context_sync.get(tool_name)
        if sync is None:
            return None

        try:
            sync(input_value)
        except Exception as exc:
            logger.exception("Context Engine sync failed for %s", tool_name)
            return self._safe_error(tool_name, exc)
        return None

    @staticmethod
    def _invoke(handler: Callable[..., Dict[str, Any]], input_value: Any) -> Dict[str, Any]:
        """Call `handler` with whatever shape `input_value` is.

        A dict is passed as keyword arguments (for tools that take multiple
        parameters, e.g. add_component); any other non-None value is passed
        as the tool's single argument; None calls the tool with no arguments
        (e.g. get_canvas_state).
        """
        if isinstance(input_value, dict):
            return handler(**input_value)
        if input_value is None:
            return handler()
        return handler(input_value)

    def execute_step(self, step: Dict[str, Any]) -> StepResult:
        """Execute a single plan step and return its result.

        Args:
            step: {"tool": str, "input": Any}, as produced by planner.create_plan().

        Returns:
            {"tool": str, "status": "success", "result": ...} or
            {"tool": str, "status": "failed", "error": str}.
        """
        tool_name = step.get("tool")
        input_value = step.get("input")

        logger.info("Tool execution started: %s (input=%r)", tool_name, input_value)

        handler = self._dispatch.get(tool_name)
        if handler is None:
            error = f"Unknown tool: {tool_name}"
            logger.error("Tool execution failed: %s - %s", tool_name, error)
            return {"tool": tool_name, "status": "failed", "error": error}

        failed = self._precheck(tool_name, input_value)
        if failed is not None:
            return failed

        try:
            if tool_name == "add_component" and isinstance(input_value, dict):
                # The Planner only knows a default spot; move it to a free one so a
                # new component never lands on top of one already on the canvas.
                # Both the command and the Context Engine sync use the new position.
                position = context_engine.find_free_position(input_value["x"], input_value["y"])
                if (position["x"], position["y"]) != (input_value["x"], input_value["y"]):
                    logger.info("add_component %r moved to free position %s", input_value.get("component_id"), position)
                    input_value = {**input_value, **position}

            result = self._invoke(handler, input_value)
        except Exception as exc:
            logger.exception("Tool execution failed: %s", tool_name)
            return {"tool": tool_name, "status": "failed", "error": self._safe_error(tool_name, exc)}

        if isinstance(result, dict) and result.get("success") is False:
            error = result.get("error") or f"{tool_name} failed"
            logger.error("Tool execution failed: %s - tool reported failure: %s", tool_name, error)
            return {"tool": tool_name, "status": "failed", "error": error, "result": result}

        logger.info("Tool execution completed: %s", tool_name)
        sync_error = self._sync_context(tool_name, input_value, result)
        if sync_error is not None:
            return {"tool": tool_name, "status": "failed", "error": sync_error, "result": result}
        return {"tool": tool_name, "status": "success", "result": result}

    def execute_plan(self, plan: List[Dict[str, Any]]) -> List[StepResult]:
        """Execute a plan sequentially, stopping at the first failed step.

        Args:
            plan: A list of steps, each {"tool": str, "input": Any}.

        Returns:
            A list of step results, one per step actually executed. Execution
            stops immediately after the first failure - remaining steps are
            not attempted and do not appear in the returned list.
        """
        results: List[StepResult] = []

        for step in plan:
            step_result = self.execute_step(step)
            results.append(step_result)

            if step_result["status"] == "failed":
                logger.warning("Stopping execution after failed step: %s", step_result)
                break

        return results


# Module-level default instance so existing callers (e.g. main.py) can keep
# using `from tool_executor import execute_plan` without instantiating
# ToolExecutor themselves.
_default_executor = ToolExecutor()


def execute_plan(plan: List[Dict[str, Any]]) -> List[StepResult]:
    """Execute `plan` using a shared default ToolExecutor instance."""
    return _default_executor.execute_plan(plan)
