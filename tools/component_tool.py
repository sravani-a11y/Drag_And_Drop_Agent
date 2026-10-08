"""ComponentTool: component placement and wiring operations exposed to the Tool Executor.

Architecture:
    Planner -> Tool Executor -> Component Tool -> Bridge -> Frontend

Every method dispatches its command through the Bridge (bridge.py). The
Bridge itself decides whether to mock the response (no frontend connected -
e.g. local development) or forward it to a real, connected frontend;
ComponentTool only supplies the mock result to fall back on when nothing
real is connected. This module never calls React, or any other frontend
code, directly.
"""

import logging
from typing import Any, Dict

from bridge import bridge as default_bridge
from context import context_engine

logger = logging.getLogger(__name__)


def _with_instance_id(payload: Dict[str, Any], component_id: str) -> Dict[str, Any]:
    """Add the frontend's instance "id" to `payload` when `component_id` is exactly one synced canvas component.

    componentId stays unchanged, so a frontend that ignores "id" behaves
    exactly as before.
    """
    instance_id = context_engine.find_instance_id(component_id)
    if instance_id is not None:
        payload = {**payload, "id": instance_id}
    return payload


class ComponentTool:
    """Component placement and connection operations used by the Tool Executor.

    Each method is independent: none depends on another having been called
    first, and none share mutable state beyond the shared Bridge instance.
    """

    name = "component_tool"
    description = "Add, remove, move, and connect/disconnect components on the design canvas."

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

    def add_component(self, component_id: str, x: int, y: int) -> Dict[str, Any]:
        """Add a component instance to the canvas at (x, y).

        Returns:
            {"success": bool, "instanceId": int}
        """
        logger.info("add_component called with component_id=%r, x=%d, y=%d", component_id, x, y)
        return self._dispatch(
            "addComponentToCanvas",
            {"componentId": component_id, "x": x, "y": y},
            mock_result={"instanceId": 12345},
        )

    def delete_component(self, component_id: str) -> Dict[str, bool]:
        """Delete a component from the canvas by component id (not instance id).

        Follows the same dispatch pattern as add_component(): builds the
        Bridge command and lets the Bridge decide mock vs real.

        Returns:
            {"success": bool}
        """
        logger.info("delete_component called with component_id=%r", component_id)
        return self._dispatch(
            "removeComponentFromCanvas",
            _with_instance_id({"componentId": component_id}, component_id),
            mock_result={},
        )

    def remove_component(self, instance_id: int) -> Dict[str, bool]:
        """Remove a component instance from the canvas.

        Returns:
            {"success": bool}
        """
        logger.info("remove_component called with instance_id=%r", instance_id)
        return self._dispatch(
            "removeComponentFromCanvas",
            {"instanceId": instance_id},
            mock_result={},
        )

    def move_component(self, component_id: str, x: int, y: int) -> Dict[str, bool]:
        """Move a component to a new position on the canvas.

        Follows the same dispatch pattern as add_component()/delete_component():
        builds the Bridge command and lets the Bridge decide mock vs real.

        Returns:
            {"success": bool}
        """
        logger.info("move_component called with component_id=%r, x=%d, y=%d", component_id, x, y)
        return self._dispatch(
            "moveComponent",
            _with_instance_id({"componentId": component_id, "x": x, "y": y}, component_id),
            mock_result={},
        )

    # connect_components()/disconnect_components() moved to
    # tools/connect_tool.py's ConnectTool - see tool_executor.py's dispatch
    # table for the wiring.
