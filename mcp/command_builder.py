"""MCPCommandBuilder: translates Bridge-native commands into standardized MCP commands.

Phase 4.2 - MCP Command Translation Layer:
    This module ONLY reshapes the {"action": str, "payload": dict} commands
    bridge.py already builds (see FrontendBridge._send_via_mock() /
    _send_via_mcp()) into the {"method": str, "params": dict} shape a real
    MCP server is expected to speak. It is pure translation - one dict in,
    one MCPCommand out - and has no knowledge of the Bridge, MCPClient, the
    WebSocket frontend, or anything upstream of them (Planner, Tool
    Executor, Verification, etc.). Nothing here executes a command, sends
    anything over a wire, or talks to a mock or real frontend.

    _ACTION_TO_METHOD and _PAYLOAD_TRANSLATORS are the only two places this
    module encodes the wire format. To support a new Bridge action, add one
    entry to _ACTION_TO_METHOD - and, only if its payload needs reshaping
    beyond a straight passthrough (as x/y -> "position" does for
    addComponentToCanvas/moveComponent), one entry to _PAYLOAD_TRANSLATORS.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MCPCommand:
    """A standardized MCP command: {"method": str, "params": dict}.

    Immutable (frozen) since a translated command is a finished value, not
    something a caller should mutate in place after the fact.
    """

    method: str
    params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Return this command as a plain, JSON-serializable dict."""
        return {"method": self.method, "params": self.params}


def _nest_position(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Reshape a payload's flat x/y keys into a nested "position" object.

    Bridge's addComponentToCanvas/moveComponent payloads carry x/y as
    siblings of componentId; the standardized MCP schema nests them under
    a "position" key instead. Every other key (e.g. componentId) passes
    through unchanged; x/y themselves are removed from the top level once
    nested under "position".
    """
    params = {key: value for key, value in payload.items() if key not in ("x", "y")}
    if "x" in payload or "y" in payload:
        params["position"] = {"x": payload.get("x"), "y": payload.get("y")}
    return params


def _passthrough(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Carry a payload over as MCP "params" unchanged, as its own copy."""
    return dict(payload)


# Maps a Bridge action name to its standardized MCP method name. One entry
# per action the Bridge currently knows how to send (see
# ComponentTool/CanvasTool in tools/) - every action they can produce today
# is registered here, so build() never has to guess at an unregistered one
# in normal operation.
_ACTION_TO_METHOD: Dict[str, str] = {
    "addComponentToCanvas": "canvas.addComponent",
    "removeComponentFromCanvas": "canvas.removeComponent",
    "moveComponent": "canvas.moveComponent",
    "connectComponents": "canvas.connectComponents",
    "disconnectComponents": "canvas.disconnectComponents",
    "getCanvasState": "canvas.getState",
}

# Per-action payload reshaping, for actions whose MCP "params" isn't a
# straight passthrough of the Bridge "payload". An action with no entry
# here (e.g. connectComponents, removeComponentFromCanvas, getCanvasState)
# falls back to _passthrough - its payload becomes "params" verbatim.
_PAYLOAD_TRANSLATORS: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]] = {
    "addComponentToCanvas": _nest_position,
    "moveComponent": _nest_position,
}


class MCPCommandBuilder:
    """Converts a Bridge-native {"action", "payload"} command into an MCPCommand.

    Stateless: build() is a pure function of its arguments, so one shared
    instance (as FrontendBridge keeps) and a fresh instance per call behave
    identically.
    """

    def build(self, action: str, payload: Dict[str, Any]) -> MCPCommand:
        """Translate one Bridge command into its standardized MCP form.

        Args:
            action: A Bridge action name, e.g. "addComponentToCanvas".
            payload: That action's Bridge-native payload.

        Returns:
            The equivalent MCPCommand ({"method": ..., "params": ...}).

        Raises:
            ValueError: `action` has no registered MCP method. Every
                action the Bridge can currently send is registered in
                _ACTION_TO_METHOD, so this only fires for a genuinely new,
                not-yet-registered action - callers decide how to handle
                that (see FrontendBridge, which logs and degrades
                gracefully rather than letting this propagate).
        """
        method = _ACTION_TO_METHOD.get(action)
        if method is None:
            raise ValueError(f"No MCP method registered for Bridge action {action!r}")

        translate_payload = _PAYLOAD_TRANSLATORS.get(action, _passthrough)
        params = translate_payload(payload or {})

        command = MCPCommand(method=method, params=params)
        logger.debug(
            "Translated Bridge command action=%r payload=%r -> %s",
            action,
            payload,
            command.to_dict(),
        )
        return command
