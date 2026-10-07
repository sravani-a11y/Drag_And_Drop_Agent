"""Canvas Context Engine (Phase 2.1; richer data model as of Phase 3.2.1).

Tracks the current canvas editing session so that future phases
(LangGraph, MCP, Context Engineering) have a ready-made place to read and
update "what does the canvas look like right now."

This is NOT chatbot memory and NOT long-term memory - it only reflects the
current canvas: which components exist, where they are, how they're
connected, and what's selected. Calling clear(), or restarting the
process, discards it entirely.

Phase 3.2.1: each tracked component now carries structured metadata (id,
display_name, category, position, status, properties) instead of just an
id, and each connection carries a status alongside its source/target.
Category is looked up from the existing Component Registry (via
KnowledgeLoader) on a best-effort basis, since add_component() is called
with only (component_id, x, y) by tool_executor.py, which is out of scope
for this change.

SessionState's own field layout (session_state.py) is unchanged - it is
out of scope for this phase. The richer per-component/per-connection
metadata is built and stored inside SessionState's existing
Dict[str, Dict[str, Any]]-shaped fields, and get_context() composes each
component's stored position into its metadata (as a nested "position" key)
only at the point it's read out - SessionState still stores positions in
its own separate field internally, exactly as before.

Context fix (multi-intent hardening): move_component/remove_component/
connect_components used to only log a warning when given a component the
session never saw add_component() for, then proceed anyway - leaving a
position with no matching component, or a connection naming one, in the
session. They now raise ValueError instead (see _require_tracked()), so an
untracked component can never enter the session's state in the first
place. This matters more now that a single multi-intent request can plan
several add/connect/move steps together: if an earlier step in that batch
never actually landed (e.g. add_component failed, or the LLM referenced a
component under a slightly different name), a later step in the same
batch must fail loudly here rather than quietly corrupting the canvas.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from knowledge_loader import KnowledgeLoader

from .session_state import SessionState

logger = logging.getLogger(__name__)

# A single, module-level session. The Context Engine tracks one canvas
# editing session at a time - there is no per-user or per-request scoping
# yet, which matches this being a Phase 2.1 building block rather than a
# multi-tenant service.
_session = SessionState()

# Default category assigned when a component can't be matched against the
# Component Registry (e.g. it was never cataloged, or the registry failed
# to load) - "Unknown" rather than None, so every component record always
# has a usable, non-null category value.
_DEFAULT_CATEGORY = "Unknown"


def _find_catalog_entry(component_name: str) -> Optional[Dict[str, Any]]:
    """Look up a component's catalog metadata by name (case-insensitive).

    Best-effort only: add_component() is called with just a name/id and an
    (x, y) position - tool_executor.py, out of scope for this change,
    calls it with exactly those three arguments. Looking up the Component
    Registry lets the richer component record include catalog metadata
    (category, canonical id) without needing a new parameter on
    add_component() itself. Returns None if there's no match or the
    registry can't be loaded - callers fall back to sensible defaults.
    """
    try:
        components = KnowledgeLoader().load_components()
    except Exception as exc:
        logger.warning("Context: could not load component catalog for enrichment: %s", exc)
        return None

    target = component_name.strip().lower()
    for entry in components:
        if str(entry.get("name", "")).strip().lower() == target:
            return entry

    return None


def _slugify(component_name: str) -> str:
    """Fall back to a simple lowercase, underscore-separated id."""
    return component_name.strip().lower().replace(" ", "_")


def _connection_key(source_component: str, target_component: str) -> str:
    """Build a stable, order-independent key identifying a connection.

    A connection between A and B is the same connection regardless of
    which component is named "source" vs "target", so both orderings must
    map to the same key.
    """
    return "::".join(sorted((source_component, target_component)))


def _record_action(action: str, **details: Any) -> None:
    """Update last_action with what just happened and when."""
    _session.last_action = {
        "action": action,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **details,
    }


def add_component(component_id: str, x: int, y: int) -> None:
    """Record that a component was added to the canvas at (x, y).

    Stores structured metadata for the component - id, display_name,
    category, status, properties - not just its identifier. category and
    the canonical id are looked up from the Component Registry by name
    (best-effort; falls back to a slugified id and "Unknown" category if
    there's no match). Position is stored separately, in
    component_positions, exactly as before - get_context() composes it
    into each component's metadata only when the snapshot is read out.
    """
    catalog_entry = _find_catalog_entry(component_id)

    _session.components[component_id] = {
        "id": catalog_entry["id"] if catalog_entry else _slugify(component_id),
        "display_name": component_id,
        "category": catalog_entry.get("category", _DEFAULT_CATEGORY) if catalog_entry else _DEFAULT_CATEGORY,
        "status": "active",
        "properties": {},
    }
    _session.component_positions[component_id] = {"x": x, "y": y}
    _record_action("add_component", component_id=component_id, x=x, y=y)
    logger.info("Context: added component %r at (%d, %d)", component_id, x, y)


def _require_tracked(component_id: str, action: str) -> None:
    """Raise if `component_id` isn't currently tracked on the canvas.

    Turns what used to be "log a warning but proceed anyway" for
    move_component/remove_component/connect_components into a hard stop:
    an operation referencing a component the Context Engine never saw
    added must never be allowed to leave the session in an inconsistent
    state (a position with no matching component, or a connection naming
    a component that doesn't exist) - exactly the "untracked component"
    issue that would otherwise surface downstream, in get_canvas_state().
    """
    if component_id not in _session.components:
        logger.error("Context: %s rejected - untracked component_id=%r", action, component_id)
        raise ValueError(f"Cannot {action}: component {component_id!r} is not on the canvas")


def remove_component(component_id: str) -> None:
    """Record that a component was removed from the canvas.

    Also drops any connections involving it and clears the selection if the
    removed component was selected, so the session never references a
    component that no longer exists.

    Raises:
        ValueError: `component_id` isn't currently tracked.
    """
    _require_tracked(component_id, "remove_component")

    _session.components.pop(component_id, None)
    _session.component_positions.pop(component_id, None)

    stale_keys = [
        key
        for key, connection in _session.connections.items()
        if component_id in (connection["source"], connection["target"])
    ]
    for key in stale_keys:
        del _session.connections[key]

    if _session.selected_component == component_id:
        _session.selected_component = None

    _record_action("remove_component", component_id=component_id)
    logger.info("Context: removed component %r (and %d stale connection(s))", component_id, len(stale_keys))


def move_component(component_id: str, x: int, y: int) -> None:
    """Record a component's new position on the canvas.

    Raises:
        ValueError: `component_id` isn't currently tracked.
    """
    _require_tracked(component_id, "move_component")

    _session.component_positions[component_id] = {"x": x, "y": y}
    _record_action("move_component", component_id=component_id, x=x, y=y)
    logger.info("Context: moved component %r to (%d, %d)", component_id, x, y)


def connect_components(source_component: str, target_component: str) -> None:
    """Record a new connection between two components, as a structured object.

    Raises:
        ValueError: either `source_component` or `target_component` isn't
            currently tracked.
    """
    _require_tracked(source_component, "connect_components")
    _require_tracked(target_component, "connect_components")

    key = _connection_key(source_component, target_component)
    _session.connections[key] = {
        "source": source_component,
        "target": target_component,
        "status": "connected",
    }
    _record_action("connect_components", source_component=source_component, target_component=target_component)
    logger.info("Context: connected %r <-> %r", source_component, target_component)


def disconnect_components(source_component: str, target_component: str) -> None:
    """Record that the connection between two components was removed."""
    key = _connection_key(source_component, target_component)
    if key not in _session.connections:
        logger.warning(
            "Context: disconnect_components called for an untracked connection (%r <-> %r)",
            source_component,
            target_component,
        )

    _session.connections.pop(key, None)
    _record_action("disconnect_components", source_component=source_component, target_component=target_component)
    logger.info("Context: disconnected %r <-> %r", source_component, target_component)


def select_component(component_id: Optional[str]) -> None:
    """Record which component is currently selected.

    Pass None to clear the selection.
    """
    _session.selected_component = component_id
    _record_action("select_component", component_id=component_id)
    logger.info("Context: selected component %r", component_id)


def get_context() -> Dict[str, Any]:
    """Return a snapshot of the current canvas session state.

    Each component's structured metadata (id, display_name, category,
    status, properties) is merged with its current position - stored
    separately in SessionState, unchanged - as a nested "position" key, so
    each component in the returned "components" dict looks like:
        {"id": ..., "display_name": ..., "category": ..., "position":
         {"x": ..., "y": ...}, "status": ..., "properties": {...}}

    "connections" stays a dict keyed by connection key (not a bare list):
    both ContextBuilder._connections_section() and
    CanvasTool.get_canvas_state() already call .values() on this field, so
    changing its container type here would break every existing command.
    Each connection's value is still a full structured object -
    {"source": ..., "target": ..., "status": ...} - it's only the outer
    container that stays a dict for backward compatibility.
    """
    snapshot = _session.to_dict()

    components: Dict[str, Any] = {}
    for component_id, metadata in snapshot["components"].items():
        position = snapshot["component_positions"].get(component_id, {})
        components[component_id] = {**metadata, "position": dict(position)}

    return {
        "components": components,
        "connections": snapshot["connections"],
        "selected_component": snapshot["selected_component"],
        "last_action": snapshot["last_action"],
    }


def clear() -> None:
    """Reset the canvas session back to empty."""
    _session.reset()
    logger.info("Context: session cleared")
