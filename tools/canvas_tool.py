"""CanvasTool: canvas-related operations exposed to the Tool Executor.

Architecture:
    Planner -> Tool Executor -> Canvas Tool -> Context Engine (source of truth)
                              -> Bridge -> Frontend (notified, not queried)

get_canvas_state() always builds its result from the Context Engine
(context/context_engine.py) - the one place every successful
add_component/move_component/connect_components/disconnect_components/
delete_component is already mirrored into. It still notifies the Bridge
(bridge.py) of the read, but never uses the Bridge/frontend's own response
to build the returned symbols/connections - a connected frontend's
bookkeeping can lag behind a real mutation (e.g. right after a delete), and
surfacing that stale response instead of the Context Engine's is what
previously let a removed component's connections reappear in "Show
Canvas". search_component and calculate_snap_position are still plain
mocks pending the same treatment. This module never calls React, or any
other frontend code, directly.
"""

import logging
from typing import Any, Dict

from bridge import bridge as default_bridge
from context import context_engine

logger = logging.getLogger(__name__)


class CanvasTool:
    """Canvas operations used by the Tool Executor.

    Each method is independent: none depends on another having been called
    first, and none share mutable state beyond the shared Bridge instance.
    """

    name = "canvas_tool"
    description = "Search components, read canvas state, and compute snap positions on the design canvas."

    def __init__(self, bridge=default_bridge) -> None:
        self.bridge = bridge

    def search_component(self, query: str) -> Dict[str, Any]:
        """Search for a component on the canvas by name.

        Returns:
            {"success": bool, "component": {"id": str, "name": str}}
        """
        logger.info("search_component called with query=%r", query)

        # TODO: replace this mock with a call to the frontend's
        # searchComponents() function through the frontend integration layer.
        component_id = query.strip().lower().replace(" ", "-")
        return {
            "success": True,
            "component": {
                "id": component_id,
                "name": query,
            },
        }

    def get_canvas_state(self) -> Dict[str, Any]:
        """Return the current canvas state.

        Unlike ComponentTool's write operations, this never defers to the
        Bridge/frontend's own response for its result. The Context Engine
        is the single place every successful add_component/move_component/
        connect_components/disconnect_components/delete_component is
        already mirrored into (see ToolExecutor._sync_context) - it, not
        whatever a connected frontend happens to report back for
        "getCanvasState", is the authoritative source for which symbols and
        connections currently exist. The Bridge is still notified (so a
        connected frontend can log or react to the read), but that
        response is never used to build the returned symbols/connections -
        otherwise a frontend whose own bookkeeping lags behind (e.g. after
        a delete) could leak a stale symbol or connection that the Context
        Engine already knows is gone.

        Returns:
            {"symbols": list[str], "connections": list[dict]}
        """
        logger.info("get_canvas_state called")

        self.bridge.send("getCanvasState", {})

        context = context_engine.get_context()
        return {
            "symbols": list(context.get("components", {}).keys()),
            "connections": list(context.get("connections", {}).values()),
        }

    def calculate_snap_position(self, component_id: str) -> Dict[str, int]:
        """Compute the canvas position a component should snap to.

        Args:
            component_id: The id of the component being placed.

        Returns:
            {"x": int, "y": int}
        """
        logger.info("calculate_snap_position called with component_id=%r", component_id)

        # TODO: replace this mock with a call to the frontend's
        # calculateSnapPosition() function through the frontend integration layer.
        return {
            "x": 200,
            "y": 200,
        }
