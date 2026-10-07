"""Tracks the state of a single request as it flows through the pipeline.

Architecture:
    User -> Main Orchestrator -> Knowledge Loader -> Prompt Builder -> LLM Engine
          -> Planner -> Agent State -> Tool Executor

AgentState is a pure data container: it only stores and updates workflow
information. It never calls the LLM, the planner, tools, or any backend or
frontend API.
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class AgentState:
    """Holds the workflow state for a single user request."""

    def __init__(self) -> None:
        self.user_request: Optional[str] = None
        self.intent: Optional[str] = None
        self.execution_plan: List[Dict[str, Any]] = []
        self.current_step: Optional[Dict[str, Any]] = None
        self.completed_steps: List[Dict[str, Any]] = []
        self.failed_step: Optional[Dict[str, Any]] = None
        self.tool_results: List[Dict[str, Any]] = []
        self.status: str = "pending"

    def set_user_request(self, user_request: str) -> None:
        """Record the raw request text this state is tracking."""
        logger.debug("Setting user_request: %r", user_request)
        self.user_request = user_request

    def set_intent(self, intent: str) -> None:
        """Record the intent extracted from the LLM's structured output."""
        logger.debug("Setting intent: %r", intent)
        self.intent = intent

    def set_plan(self, execution_plan: List[Dict[str, Any]]) -> None:
        """Record a new execution plan and reset per-run step tracking."""
        logger.debug("Setting execution_plan with %d step(s): %s", len(execution_plan), execution_plan)
        self.execution_plan = execution_plan
        self.current_step = None
        self.completed_steps = []
        self.failed_step = None

    def mark_step_completed(self, step: Dict[str, Any]) -> None:
        """Record that `step` finished successfully."""
        logger.info("Step completed: %s", step)
        self.current_step = step
        self.completed_steps.append(step)

    def mark_step_failed(self, step: Dict[str, Any]) -> None:
        """Record that `step` failed."""
        logger.error("Step failed: %s", step)
        self.current_step = step
        self.failed_step = step

    def add_tool_result(self, result: Dict[str, Any]) -> None:
        """Append a tool execution result to the recorded history."""
        logger.debug("Adding tool result: %s", result)
        self.tool_results.append(result)

    def set_status(self, status: str) -> None:
        """Update the overall workflow status (e.g. pending/running/completed/failed)."""
        logger.info("Status changed: %r -> %r", self.status, status)
        self.status = status

    def to_dict(self) -> Dict[str, Any]:
        """Return the complete state as a plain, JSON-serializable dict."""
        return {
            "user_request": self.user_request,
            "intent": self.intent,
            "execution_plan": self.execution_plan,
            "current_step": self.current_step,
            "completed_steps": self.completed_steps,
            "failed_step": self.failed_step,
            "tool_results": self.tool_results,
            "status": self.status,
        }
