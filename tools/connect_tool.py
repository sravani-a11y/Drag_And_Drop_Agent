"""ConnectTool: wiring operations between components on the design canvas.

Architecture:
    Planner -> Tool Executor -> Connect Tool -> Bridge -> Frontend

Split out of ComponentTool so connecting/disconnecting components has its
own dedicated module, separate from placement (add/remove/move). Both
still go through the same Bridge dispatch pattern - this module never
calls React, or any other frontend code, directly.
"""

import logging
from typing import Any, Dict

from bridge import bridge as default_bridge

logger = logging.getLogger(__name__)


class ConnectTool:
    """Connect/disconnect operations used by the Tool Executor."""

    name = "connect_tool"
    description = "Connect and disconnect components on the design canvas."

    def __init__(self, bridge=default_bridge) -> None:
        self.bridge = bridge

    def _dispatch(self, action: str, payload: Dict[str, Any], mock_result: Dict[str, Any]) -> Dict[str, Any]:
        """Send a command through the Bridge and normalize its response.

        In mock mode (no frontend connected) the Bridge only echoes the
        command back, so `mock_result` supplies this method's documented
        placeholder result. In real mode, the frontend's actual response is
        used instead.
        """
        response = self.bridge.send(action, payload)

        if response.get("mode") != "real":
            return {"success": True, **mock_result}

        if not response.get("success"):
            return {"success": False, "error": response.get("error", f"{action} failed")}

        result = response.get("result") or {}
        return {"success": True, **result}

    def connect_components(self, source_component: str, target_component: str) -> Dict[str, Any]:
        """Create a connection between two components on the canvas.

        Returns:
            {"success": bool, "connectionId": str}
        """
        logger.info(
            "connect_components called with source_component=%r, target_component=%r",
            source_component,
            target_component,
        )
        return self._dispatch(
            "connectComponents",
            {"sourceComponentId": source_component, "targetComponentId": target_component},
            mock_result={"connectionId": "123-456"},
        )

    def disconnect_components(self, source_component: str, target_component: str) -> Dict[str, bool]:
        """Remove the connection between two components on the canvas.

        Returns:
            {"success": bool}
        """
        logger.info(
            "disconnect_components called with source_component=%r, target_component=%r",
            source_component,
            target_component,
        )
        return self._dispatch(
            "disconnectComponents",
            {"sourceComponentId": source_component, "targetComponentId": target_component},
            mock_result={},
        )
