"""MCPClient: placeholder client for the future MCP (Model Context Protocol) server.

Phase 4.1 - MCP Foundation:
    This defines the interface bridge.py calls when running in "mcp" mode
    (see bridge.py's FrontendBridge._send_via_mcp()). It implements no real
    MCP wire protocol - no sockets, no WebSocket, no HTTP, no JSON-RPC, no
    retries, no authentication. Every method here only tracks minimal
    in-memory state and logs what it would have done, so a real MCP
    transport can be dropped in behind this exact interface later without
    bridge.py (or anything upstream of it) needing to change.

This module has no knowledge of the AI pipeline (LangGraph, Planner, Tool
Executor, etc.) and never will - it only speaks the connect/disconnect/
send/receive vocabulary bridge.py already expects from a transport.
"""

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class MCPClient:
    """Placeholder client for a future MCP server connection.

    No real connection, protocol, or transport is implemented yet.
    connect()/disconnect()/is_connected() track simple in-memory state;
    send()/receive() log what would have been transmitted and echo the
    last command back, so callers (bridge.py) can already be written
    against the shape a real implementation will eventually return.
    """

    def __init__(self) -> None:
        self._connected: bool = False
        self._last_command: Optional[Dict[str, Any]] = None

    def connect(self) -> None:
        """Establish a connection to the MCP server.

        Placeholder only: no real MCP server exists yet, so this just
        marks the client as connected in-memory.
        """
        logger.info("MCPClient: connect() called (placeholder - no real MCP server yet)")
        self._connected = True

    def disconnect(self) -> None:
        """Tear down the connection to the MCP server.

        Placeholder only: no real connection exists yet, so this just
        clears the client's in-memory state.
        """
        logger.info("MCPClient: disconnect() called (placeholder)")
        self._connected = False
        self._last_command = None

    def is_connected(self) -> bool:
        """Return whether the client currently considers itself connected."""
        return self._connected

    def send(self, command: Dict[str, Any]) -> None:
        """Send a command to the MCP server.

        Placeholder only: nothing is actually transmitted. The command is
        recorded in-memory so receive() has something to echo back.

        Raises:
            ConnectionError: if called before connect().
        """
        if not self._connected:
            raise ConnectionError("MCPClient.send() called before connect()")

        logger.info("MCPClient: send() called with command=%r (placeholder - not transmitted)", command)
        self._last_command = command

    def receive(self) -> Dict[str, Any]:
        """Receive the MCP server's response to the last command sent.

        Placeholder only: no real MCP server exists yet, so this echoes
        back the last command sent via send(), wrapped in the
        {"success", "result"} shape a real response will eventually have.

        Raises:
            ConnectionError: if called before connect().
        """
        if not self._connected:
            raise ConnectionError("MCPClient.receive() called before connect()")

        logger.info("MCPClient: receive() called (placeholder - no real MCP server yet)")
        return {"success": True, "result": None, "command": self._last_command}
