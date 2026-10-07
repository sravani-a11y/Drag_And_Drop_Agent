"""Pydantic request/response shapes for the FastAPI integration layer.

Pure data contracts - no LangGraph/agent logic lives here. MCP commands and
tool results are passed through as plain dicts (List[Dict[str, Any]])
rather than modeled field-by-field, so this layer never has to be updated
just because the existing MCP Bridge/ToolExecutor add a new field to what
they already return.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, field_validator


class ExecuteRequest(BaseModel):
    """POST /api/agent/execute request body."""

    command: str

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
