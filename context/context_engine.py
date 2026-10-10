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
from typing import Any, Dict, List, Optional

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

    If the name is already taken by a component synced from the frontend
    canvas (one with an instance ID), the new one gets its own key
    ("ESP32 #2") instead of overwriting it - the canvas really does hold
    both until the next sync replaces this record with the frontend's truth.
    """
    catalog_entry = _find_catalog_entry(component_id)

    key = component_id
    if key in _session.components and _session.components[key].get("instance_id") is not None:
        suffix = 2
        while f"{component_id} #{suffix}" in _session.components:
            suffix += 1
        key = f"{component_id} #{suffix}"

    _session.components[key] = {
        "id": catalog_entry["id"] if catalog_entry else _slugify(component_id),
        "display_name": component_id,
        "category": catalog_entry.get("category", _DEFAULT_CATEGORY) if catalog_entry else _DEFAULT_CATEGORY,
        "status": "active",
        "properties": {},
    }
    _session.component_positions[key] = {"x": x, "y": y}
    _record_action("add_component", component_id=component_id, x=x, y=y)
    logger.info("Context: added component %r at (%d, %d)", component_id, x, y)


def _instance_id_key(instance_id: Any) -> str:
    """Normalize an instance ID for comparison (1775642938124.512, "1775642938124.512" and 5 vs 5.0 all compare equal)."""
    try:
        number = float(instance_id)
    except (TypeError, ValueError):
        return str(instance_id).strip()
    return repr(int(number)) if number.is_integer() else repr(number)


def _matching_keys(reference: Any) -> List[str]:
    """Every tracked component key that `reference` names.

    `reference` may be the internal key itself, a frontend instance ID, or a
    name (case-insensitive) matching the component's display name, its
    frontend label (e.g. "ESP32") or its frontend component name (e.g.
    "Microcontroller"). More than one key back means the reference is
    ambiguous (e.g. "Temperature Sensor" when two are placed).
    """
    if reference in _session.components:
        return [reference]

    reference_id = _instance_id_key(reference)
    by_id = [
        key
        for key, metadata in _session.components.items()
        if metadata.get("instance_id") is not None and _instance_id_key(metadata["instance_id"]) == reference_id
    ]
    if by_id:
        return by_id

    wanted = str(reference).strip().lower()
    by_name = []
    for key, metadata in _session.components.items():
        names = {key, metadata.get("display_name"), metadata.get("label"), metadata.get("component_name")}
        if wanted in {str(name).strip().lower() for name in names if name}:
            by_name.append(key)
    return by_name


class ComponentNotFoundError(ValueError):
    """The referenced component isn't on the canvas."""

    def __init__(self, message: str, reference: Any) -> None:
        super().__init__(message)
        self.reference = reference


class AmbiguousComponentError(ValueError):
    """The reference matches more than one component on the canvas."""

    def __init__(self, message: str, reference: Any, instance_ids: List[Any]) -> None:
        super().__init__(message)
        self.reference = reference
        self.instance_ids = instance_ids


def _require_tracked(component_id: str, action: str) -> str:
    """Return the tracked key for `component_id`, or raise if it isn't on the canvas (or is ambiguous).

    Turns what used to be "log a warning but proceed anyway" for
    move_component/remove_component/connect_components into a hard stop:
    an operation referencing a component the Context Engine never saw
    added must never be allowed to leave the session in an inconsistent
    state (a position with no matching component, or a connection naming
    a component that doesn't exist) - exactly the "untracked component"
    issue that would otherwise surface downstream, in get_canvas_state().
    """
    matches = _matching_keys(component_id)
    if len(matches) == 1:
        return matches[0]

    if matches:
        instance_ids = [_session.components[key].get("instance_id") for key in matches]
        logger.error("Context: %s rejected - %r matches %d components %s", action, component_id, len(matches), instance_ids)
        raise AmbiguousComponentError(
            f"Cannot {action}: {component_id!r} matches {len(matches)} components on the canvas "
            f"(instance ids {instance_ids})",
            component_id,
            instance_ids,
        )

    logger.error("Context: %s rejected - untracked component_id=%r", action, component_id)
    raise ComponentNotFoundError(f"Cannot {action}: component {component_id!r} is not on the canvas", component_id)


def resolve_component(component_reference: Any) -> str:
    """The tracked key `component_reference` names; raises ComponentNotFoundError / AmbiguousComponentError.

    Read-only - used to check an operation's components exist before it runs.
    """
    return _require_tracked(component_reference, "resolve_component")


def connection_exists(source_component: Any, target_component: Any) -> bool:
    """Whether the two components (resolved as resolve_component() does) are connected. Read-only."""
    source_key = resolve_component(source_component)
    target_key = resolve_component(target_component)
    return _connection_key(source_key, target_key) in _session.connections


def find_connection_key(source_component: Any, target_component: Any) -> Optional[str]:
    """The frontend's own connectionKey for the connection between the two components, else None.

    Taken exactly as the frontend sent it in canvas_state (never rebuilt
    here). None when either component doesn't resolve to exactly one, they
    aren't connected, or the connection wasn't synced from the frontend.
    Never raises - callers use this to enrich a command when possible.
    """
    source_matches, target_matches = _matching_keys(source_component), _matching_keys(target_component)
    if len(source_matches) != 1 or len(target_matches) != 1:
        return None
    connection = _session.connections.get(_connection_key(source_matches[0], target_matches[0]))
    return connection.get("connection_key") if connection else None


def find_instance_id(component_reference: Any) -> Optional[Any]:
    """The frontend instance ID of the one component `component_reference` names, else None.

    None when it isn't on the canvas, is ambiguous, or was added by the
    agent and not yet synced back from the frontend (so it has no ID yet).
    Never raises - callers use this to enrich a command when possible.
    """
    matches = _matching_keys(component_reference)
    if len(matches) != 1:
        return None
    return _session.components[matches[0]].get("instance_id")


def connected_components(component_reference: Any) -> List[str]:
    """Tracked keys of every component connected to the one `component_reference` names.

    Keys rather than display names, so each one still names exactly one
    component when two share a name ("Temperature Sensor [<id>]"). Raises
    ComponentNotFoundError / AmbiguousComponentError as resolve_component()
    does. Read-only.
    """
    key = resolve_component(component_reference)
    partners = []
    for connection in _session.connections.values():
        if connection["source"] == key:
            partners.append(connection["target"])
        elif connection["target"] == key:
            partners.append(connection["source"])
    return partners


def display_name(component_reference: Any) -> Any:
    """The display name of the one component `component_reference` names, else the reference unchanged.

    Turns an internal key ("Temperature Sensor [<id>]") back into the name
    the frontend knows ("Temperature Sensor"). Never raises.
    """
    matches = _matching_keys(component_reference)
    if len(matches) != 1:
        return component_reference
    return _session.components[matches[0]].get("display_name", matches[0])


def component_label(key: str) -> str:
    """A readable label for one tracked component, for messages shown to the user.

    Just the display name when it's the only component with that name;
    otherwise "<name> #<n> (id <instance id>, at x=<x>, y=<y>)", numbered
    in canvas order, so two instances can be told apart.
    """
    metadata = _session.components.get(key, {})
    name = metadata.get("display_name", key)
    same_name = [k for k, m in _session.components.items() if str(m.get("display_name", k)).lower() == str(name).lower()]
    if len(same_name) <= 1:
        return name

    details = []
    if metadata.get("instance_id") is not None:
        details.append(f"id {metadata['instance_id']}")
    position = _session.component_positions.get(key) or {}
    if position.get("x") is not None and position.get("y") is not None:
        details.append(f"at x={position['x']}, y={position['y']}")
    suffix = f" ({', '.join(details)})" if details else ""
    return f"{name} #{same_name.index(key) + 1}{suffix}"


def matching_labels(component_reference: Any) -> List[str]:
    """component_label() of every tracked component `component_reference` names (several when ambiguous)."""
    return [component_label(key) for key in _matching_keys(component_reference)]


# The frontend creates a new component 100x100 and positions it by its
# top-left corner (x, y); the same size is assumed for any component whose
# size isn't known (e.g. one the agent added and the frontend hasn't synced).
_DEFAULT_COMPONENT_SIZE = 100

# Minimum empty space (px) kept between a new component and any other.
_PLACEMENT_GAP = 50

# Where find_free_position() looks when the requested spot is taken: up to
# _PLACEMENT_COLUMNS spots to the right on the same row, then the same
# spots on each row below, _PLACEMENT_STEP px apart. Rows continue until a
# free spot is found - the canvas size isn't known, so there's no edge to
# stop at, and below every existing component is always free.
_PLACEMENT_STEP = 200
_PLACEMENT_COLUMNS = 5


def _size_or_default(value: Any) -> float:
    return value if isinstance(value, (int, float)) and value > 0 else _DEFAULT_COMPONENT_SIZE


def _occupied_boxes() -> List[tuple]:
    """(x, y, width, height) of every tracked component with a known position."""
    boxes = []
    for key, position in _session.component_positions.items():
        x, y = position.get("x"), position.get("y")
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            continue
        size = _session.components.get(key, {}).get("size") or {}
        boxes.append((x, y, _size_or_default(size.get("width")), _size_or_default(size.get("height"))))
    return boxes


def _overlaps(x: float, y: float, width: float, height: float, box: tuple) -> bool:
    """Whether the rectangle at (x, y) comes within _PLACEMENT_GAP of `box`."""
    box_x, box_y, box_width, box_height = box
    return (
        x < box_x + box_width + _PLACEMENT_GAP
        and box_x < x + width + _PLACEMENT_GAP
        and y < box_y + box_height + _PLACEMENT_GAP
        and box_y < y + height + _PLACEMENT_GAP
    )


def find_free_position(
    x: int, y: int, width: float = _DEFAULT_COMPONENT_SIZE, height: float = _DEFAULT_COMPONENT_SIZE
) -> Dict[str, int]:
    """(x, y) itself when a width x height component fits there, else the first free spot after it.

    Every tracked component counts with its real size (from canvas_state)
    or the default size. Always finds a spot: rows continue downward, and
    a row below every existing component is free. Read-only.
    """
    boxes = _occupied_boxes()
    lowest_edge = max((box_y + box_height for _, box_y, _, box_height in boxes), default=y)

    row = 0
    while True:
        candidate_y = y + row * _PLACEMENT_STEP
        for column in range(_PLACEMENT_COLUMNS):
            candidate_x = x + column * _PLACEMENT_STEP
            if not any(_overlaps(candidate_x, candidate_y, width, height, box) for box in boxes):
                return {"x": candidate_x, "y": candidate_y}
        if candidate_y > lowest_edge + _PLACEMENT_GAP:
            # Unreachable: nothing extends below lowest_edge. Kept so a bug
            # here can never loop forever.
            logger.error("Context: placement search passed every component without a free spot at (%s, %s)", x, y)
            return {"x": x, "y": candidate_y}
        row += 1


def sync_from_canvas(canvas_state: Dict[str, Any]) -> Dict[str, Any]:
    """Replace the tracked canvas with the frontend's current canvas (its active tab).

    The frontend is the source of truth: components, positions and
    connections from `canvas_state` replace what the agent had recorded,
    so components the user dragged in by hand are known and ones they
    deleted are gone. Each component is tracked under its label (or name)
    when that is unique on the canvas, else under "<label> [<instance id>]",
    so duplicates stay distinct; its instance ID, frontend name, label,
    size and rotation are kept in its metadata. Connections are stored with
    their real sourceId/targetId. The selection is kept only if that
    component is still present; last_action (the agent's memory of what it
    last did) is kept as-is.

    Args:
        canvas_state: {"tabId", "tabName", "components": [{"id", "name",
            "label", "x", "y", "width", "height", "rotation"}],
            "connections": [{"connectionKey", "sourceId", "targetId"}]}.

    Returns:
        {"components": int, "connections": int, "skipped_connections": int}
    """
    canvas_components = canvas_state.get("components") or []
    canvas_connections = canvas_state.get("connections") or []

    def display_name(component: Dict[str, Any]) -> str:
        label = component.get("label")
        return str(label).strip() if label and str(label).strip() else str(component.get("name", "")).strip()

    name_counts: Dict[str, int] = {}
    for component in canvas_components:
        lowered = display_name(component).lower()
        name_counts[lowered] = name_counts.get(lowered, 0) + 1

    components: Dict[str, Dict[str, Any]] = {}
    positions: Dict[str, Dict[str, Any]] = {}
    key_by_instance: Dict[str, str] = {}
    for component in canvas_components:
        name = display_name(component)
        instance_id = component.get("id")
        key = name if name_counts[name.lower()] == 1 else f"{name} [{instance_id}]"

        catalog_entry = _find_catalog_entry(name) or _find_catalog_entry(str(component.get("name", "")))
        components[key] = {
            "id": catalog_entry["id"] if catalog_entry else _slugify(name),
            "display_name": name,
            "category": catalog_entry.get("category", _DEFAULT_CATEGORY) if catalog_entry else _DEFAULT_CATEGORY,
            "status": "active",
            "properties": {},
            "instance_id": instance_id,
            "component_name": component.get("name"),
            "label": component.get("label"),
            "size": {"width": component.get("width"), "height": component.get("height")},
            "rotation": component.get("rotation"),
            "source": "canvas",
        }
        positions[key] = {"x": component.get("x"), "y": component.get("y")}
        key_by_instance[_instance_id_key(instance_id)] = key

    connections: Dict[str, Dict[str, Any]] = {}
    skipped = 0
    for connection in canvas_connections:
        source_id, target_id = connection.get("sourceId"), connection.get("targetId")
        source = key_by_instance.get(_instance_id_key(source_id))
        target = key_by_instance.get(_instance_id_key(target_id))
        if source is None or target is None:
            skipped += 1
            logger.warning(
                "Context: canvas connection %r skipped - sourceId=%r/targetId=%r not among the canvas components",
                connection.get("connectionKey"),
                source_id,
                target_id,
            )
            continue
        connections[_connection_key(source, target)] = {
            "source": source,
            "target": target,
            "status": "connected",
            "source_id": source_id,
            "target_id": target_id,
            "connection_key": connection.get("connectionKey"),
        }

    _session.components.clear()
    _session.components.update(components)
    _session.component_positions.clear()
    _session.component_positions.update(positions)
    _session.connections.clear()
    _session.connections.update(connections)
    if _session.selected_component not in _session.components:
        _session.selected_component = None

    summary = {"components": len(components), "connections": len(connections), "skipped_connections": skipped}
    logger.info(
        "Context: synced from frontend canvas (tabId=%r, tabName=%r): %s",
        canvas_state.get("tabId"),
        canvas_state.get("tabName"),
        summary,
    )
    return summary


def remove_component(component_id: str) -> None:
    """Record that a component was removed from the canvas.

    Also drops any connections involving it and clears the selection if the
    removed component was selected, so the session never references a
    component that no longer exists.

    Raises:
        ValueError: `component_id` isn't currently tracked.
    """
    component_id = _require_tracked(component_id, "remove_component")

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
    component_id = _require_tracked(component_id, "move_component")

    _session.component_positions[component_id] = {"x": x, "y": y}
    _record_action("move_component", component_id=component_id, x=x, y=y)
    logger.info("Context: moved component %r to (%d, %d)", component_id, x, y)


def connect_components(source_component: str, target_component: str) -> None:
    """Record a new connection between two components, as a structured object.

    Raises:
        ValueError: either `source_component` or `target_component` isn't
            currently tracked.
    """
    source_component = _require_tracked(source_component, "connect_components")
    target_component = _require_tracked(target_component, "connect_components")

    key = _connection_key(source_component, target_component)
    connection = {
        "source": source_component,
        "target": target_component,
        "status": "connected",
    }
    # Components synced from the frontend carry instance IDs - record the
    # connection with them too, the way the frontend identifies it.
    source_id = _session.components[source_component].get("instance_id")
    target_id = _session.components[target_component].get("instance_id")
    if source_id is not None and target_id is not None:
        connection["source_id"] = source_id
        connection["target_id"] = target_id
    _session.connections[key] = connection
    _record_action("connect_components", source_component=source_component, target_component=target_component)
    logger.info("Context: connected %r <-> %r", source_component, target_component)


def disconnect_components(source_component: str, target_component: str) -> None:
    """Record that the connection between two components was removed."""
    source_matches, target_matches = _matching_keys(source_component), _matching_keys(target_component)
    if len(source_matches) == 1:
        source_component = source_matches[0]
    if len(target_matches) == 1:
        target_component = target_matches[0]
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
