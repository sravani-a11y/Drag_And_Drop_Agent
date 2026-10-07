"""Evaluates a single tool execution result and decides what happens next.

Architecture:
    ... -> Tool Executor -> Verification

Verification only evaluates tool execution results and returns a decision.
It never executes tools, calls the LLM, builds prompts, creates execution
plans, or calls backend/frontend APIs.
"""

import logging
from typing import Dict

logger = logging.getLogger(__name__)

# Error substrings (case-insensitive) that mean the desired end state was
# already reached, or the failure is a known, harmless condition - no retry
# is needed, the plan can just move on to the next step.
_RECOVERABLE_ERROR_SUBSTRINGS = (
    "already exists",
    "already added",
    "duplicate",
)

# Error substrings (case-insensitive) that indicate a likely transient
# failure - the same step may succeed if attempted again.
_RETRYABLE_ERROR_SUBSTRINGS = (
    "timed out",
    "timeout",
    "did not respond in time",
    "temporarily unavailable",
    "connection reset",
)


def _classify_error(error_message: str) -> str:
    """Classify a failed step's error as recoverable, retryable, or critical.

    This is the single place decision logic lives - callers never inspect
    error text themselves.
    """
    normalized = (error_message or "").lower()

    if any(substring in normalized for substring in _RECOVERABLE_ERROR_SUBSTRINGS):
        return "recover"

    if any(substring in normalized for substring in _RETRYABLE_ERROR_SUBSTRINGS):
        return "retry"

    return "stop"


def verify(tool_result: Dict) -> Dict[str, str]:
    """Evaluate a single tool execution result and decide what happens next.

    Args:
        tool_result: One result from tool_executor.execute_plan(), e.g.
            {"tool": "search_component", "status": "success", "result": {...}}
            {"tool": "add_component", "status": "failed", "error": "..."}

    Returns:
        {"decision": "continue" | "retry" | "recover" | "stop"}
    """
    tool_result = tool_result or {}
    tool_name = tool_result.get("tool", "<unknown>")
    status = tool_result.get("status")

    if status == "success":
        logger.info("Tool %r succeeded - decision: continue", tool_name)
        return {"decision": "continue"}

    error_message = tool_result.get("error", "")
    decision = _classify_error(error_message)

    if decision == "recover":
        logger.info("Tool %r failed with a recoverable error (%r) - decision: recover", tool_name, error_message)
    elif decision == "retry":
        logger.warning("Tool %r failed with a retryable error (%r) - decision: retry", tool_name, error_message)
    else:
        logger.error("Tool %r failed with a critical error (%r) - decision: stop", tool_name, error_message)

    return {"decision": decision}
