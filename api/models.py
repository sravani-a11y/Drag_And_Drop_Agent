"""Pydantic request/response shapes for the FastAPI integration layer.

Pure data contracts - no LangGraph/agent logic lives here. MCP commands and
tool results are passed through as plain dicts (List[Dict[str, Any]])
rather than modeled field-by-field, so this layer never has to be updated
just because the existing MCP Bridge/ToolExecutor add a new field to what
they already return.
"""

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, field_validator

# Instance IDs are JavaScript numbers created by the frontend
# (e.g. 1775642938124.512). int is tried first so a whole-number ID keeps its
# exact form instead of becoming a float.
InstanceId = Union[int, float]


class CanvasComponent(BaseModel):
    """One placed component in the frontend's active tab, as sent by the frontend."""

    id: InstanceId
    name: str
    label: Optional[str] = None
    x: Union[int, float]
    y: Union[int, float]
    width: Optional[Union[int, float]] = None
    height: Optional[Union[int, float]] = None
    rotation: Optional[Union[int, float]] = None


class CanvasConnection(BaseModel):
    """One connection between two placed components, by their instance IDs."""

    connectionKey: Optional[str] = None
    sourceId: InstanceId
    targetId: InstanceId


class CanvasState(BaseModel):
    """The frontend's current canvas (active tab), sent with a chat request."""

    tabId: Optional[int] = None
    tabName: Optional[str] = None
    components: List[CanvasComponent] = []
    connections: List[CanvasConnection] = []


class ExecuteRequest(BaseModel):
    """POST /api/agent/execute request body.

    canvas_state is optional: when present, the Context Engine is synced to
    it before the request runs, so the agent works from the frontend's real
    canvas; when absent, the agent's own memory is used exactly as before.
    """

    command: str
    canvas_state: Optional[CanvasState] = None

    @field_validator("command")
    @classmethod
    def command_must_not_be_blank(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("Command is required")
        return value.strip()


class AgentResponse(BaseModel):
    """Response shape shared by /api/agent/execute and (extended) /api/agent/voice."""

    status: str
    message: Optional[str] = None
    commands: List[Dict[str, Any]] = []
    results: List[Dict[str, Any]] = []


class VoiceResponse(AgentResponse):
    """POST /api/agent/voice response - AgentResponse plus the transcript and its normalized form.

    Both are returned (not just the normalized command actually run) so
    speech-recognition accuracy can be debugged separately from command
    understanding - see command_normalizer.py.
    """

    transcript: Optional[str] = None
    normalized_command: Optional[str] = None


class HealthResponse(BaseModel):
    """GET /api/health response."""

    status: str
    service: str


class ErrorResponse(BaseModel):
    """Shape used by every error path (validation, agent failure, unhandled exception)."""

    status: str = "error"
    message: str
    results: List[Dict[str, Any]] = []
