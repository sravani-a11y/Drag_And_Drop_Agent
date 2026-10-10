"""Canvas Context Engine package (Phase 2.1).

Tracks the current canvas editing session - which components exist, their
positions, their connections, the current selection, and the last action
taken - completely independently of the LLM / Planner / Tool Executor /
Bridge pipeline.

This is NOT chatbot memory and NOT long-term memory: it only reflects the
current canvas state for future phases (LangGraph, MCP, Context
Engineering) to read from and write to. Nothing in the existing pipeline
calls this package yet.
"""

from .context_engine import (
    AmbiguousComponentError,
    ComponentNotFoundError,
    add_component,
    clear,
    component_label,
    connect_components,
    connected_components,
    connection_exists,
    find_connection_key,
    disconnect_components,
    display_name,
    find_free_position,
    find_instance_id,
    get_context,
    matching_labels,
    move_component,
    remove_component,
    resolve_component,
    select_component,
    sync_from_canvas,
)
from .session_state import SessionState

__all__ = [
    "SessionState",
    "add_component",
    "remove_component",
    "move_component",
    "connect_components",
    "disconnect_components",
    "select_component",
    "get_context",
    "clear",
    "sync_from_canvas",
    "find_instance_id",
    "resolve_component",
    "connection_exists",
    "find_connection_key",
    "connected_components",
    "find_free_position",
    "display_name",
    "component_label",
    "matching_labels",
    "ComponentNotFoundError",
    "AmbiguousComponentError",
]
