/**
 * Canvas state driven entirely by incoming MCP commands.
 *
 * Mirrors the shape the Python side already works with (see
 * simple_tool_executor.py's canvas_state): {"components": [...], "connections": [...]},
 * with components as {id, x, y} and connections as {source, target}.
 *
 * Handles exactly the MCP methods this project's MCPCommandBuilder
 * (mcp/command_builder.py) actually emits:
 *   - canvas.addComponent      { componentId, position: { x, y } }
 *   - canvas.moveComponent     { componentId, position: { x, y } }
 *   - canvas.connectComponents { sourceComponentId, targetComponentId }
 *   - canvas.removeComponent   { componentId }
 * An unrecognized method is logged and ignored - it never throws, since a
 * malformed or not-yet-supported command shouldn't take down the canvas.
 */
import { useReducer } from "react";

const initialCanvasState = {
  components: [],
  connections: [],
};

function findComponent(components, componentId) {
  return components.find((component) => component.id === componentId);
}

function addComponent(state, params) {
  const { componentId, position } = params;

  if (findComponent(state.components, componentId)) {
    console.warn(`[MCP] canvas.addComponent ignored - "${componentId}" already exists`);
    return state;
  }

  const component = { id: componentId, x: position?.x ?? 0, y: position?.y ?? 0 };
  return { ...state, components: [...state.components, component] };
}

function moveComponent(state, params) {
  const { componentId, position } = params;

  if (!findComponent(state.components, componentId)) {
    console.warn(`[MCP] canvas.moveComponent ignored - "${componentId}" not found`);
    return state;
  }

  return {
    ...state,
    components: state.components.map((component) =>
      component.id === componentId
        ? { ...component, x: position?.x ?? component.x, y: position?.y ?? component.y }
        : component
    ),
  };
}

function connectComponents(state, params) {
  const { sourceComponentId, targetComponentId } = params;

  if (!findComponent(state.components, sourceComponentId) || !findComponent(state.components, targetComponentId)) {
    console.warn("[MCP] canvas.connectComponents ignored - both components must exist on the canvas", params);
    return state;
  }

  const alreadyConnected = state.connections.some(
    (connection) => connection.source === sourceComponentId && connection.target === targetComponentId
  );
  if (alreadyConnected) {
    return state;
  }

  return {
    ...state,
    connections: [...state.connections, { source: sourceComponentId, target: targetComponentId }],
  };
}

function removeComponent(state, params) {
  const { componentId } = params;

  return {
    components: state.components.filter((component) => component.id !== componentId),
    // Drop any connection that referenced the removed component too - a
    // deleted component must never leave a stale connection on the canvas
    // (the exact bug this project's get_canvas_state() was fixed for).
    connections: state.connections.filter(
      (connection) => connection.source !== componentId && connection.target !== componentId
    ),
  };
}

const MCP_HANDLERS = {
  "canvas.addComponent": addComponent,
  "canvas.moveComponent": moveComponent,
  "canvas.connectComponents": connectComponents,
  "canvas.removeComponent": removeComponent,
};

/** Pure reducer: (current canvas state, one MCP command) -> next canvas state. */
export function canvasReducer(state, command) {
  const handler = MCP_HANDLERS[command?.method];
  if (!handler) {
    console.warn(`[MCP] Unknown method, ignored: ${command?.method}`);
    return state;
  }
  return handler(state, command.params || {});
}

/** Canvas state plus a dispatcher that applies one MCP command at a time. */
export function useMcpCanvas() {
  const [canvasState, dispatchMcpCommand] = useReducer(canvasReducer, initialCanvasState);
  return { canvasState, dispatchMcpCommand };
}
