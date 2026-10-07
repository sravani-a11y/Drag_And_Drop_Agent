"""MCP (Model Context Protocol) package.

Phase 4.1 (MCP Foundation) added MCPClient - a placeholder client with no
real MCP wire protocol implementation yet (see client.py's module
docstring). Phase 4.2 (MCP Command Translation Layer) added
MCPCommandBuilder/MCPCommand - pure translation from bridge.py's native
{"action", "payload"} commands into the standardized {"method", "params"}
shape a real MCP server is expected to speak (see command_builder.py's
module docstring). A future phase will add a real MCP server/transport
behind these same interfaces without requiring changes to bridge.py or
anything upstream of it.
"""

from .client import MCPClient
from .command_builder import MCPCommand, MCPCommandBuilder

__all__ = ["MCPClient", "MCPCommand", "MCPCommandBuilder"]
