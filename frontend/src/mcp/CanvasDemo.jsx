/**
 * Example usage: wires useMcpCanvas + useMcpListener together and drives
 * them with a mock sequence of MCP commands (the "Simulate receiving
 * commands" requirement) - no backend connection needed to see it work.
 *
 * The same sequence a real add -> connect -> move -> delete workflow would
 * produce, in the exact {"method", "params"} shape MCPCommandBuilder
 * (mcp/command_builder.py) already emits.
 */
import React from "react";
import { useMcpCanvas } from "./useMcpCanvas";
import { useMcpListener } from "./useMcpListener";

const MOCK_COMMANDS = [
  { method: "canvas.addComponent", params: { componentId: "ESP32", position: { x: 200, y: 200 } } },
  { method: "canvas.addComponent", params: { componentId: "Relay Module", position: { x: 400, y: 200 } } },
  {
    method: "canvas.connectComponents",
    params: { sourceComponentId: "ESP32", targetComponentId: "Relay Module" },
  },
  { method: "canvas.moveComponent", params: { componentId: "ESP32", position: { x: 300, y: 300 } } },
  { method: "canvas.removeComponent", params: { componentId: "Relay Module" } },
];

export default function CanvasDemo() {
  const { canvasState, dispatchMcpCommand } = useMcpCanvas();
  const { receive } = useMcpListener(dispatchMcpCommand);

  // Simulates commands arriving one at a time, exactly like a real
  // WebSocket message handler would deliver them.
  const runMockSequence = () => {
    MOCK_COMMANDS.forEach((command) => receive(command));
  };

  return (
    <div style={{ fontFamily: "sans-serif", padding: 16 }}>
      <button onClick={runMockSequence}>Simulate MCP Commands</button>

      <h3>Components</h3>
      <ul>
        {canvasState.components.map((component) => (
          <li key={component.id}>
            {component.id} @ ({component.x}, {component.y})
          </li>
        ))}
      </ul>

      <h3>Connections</h3>
      <ul>
        {canvasState.connections.map((connection) => (
          <li key={`${connection.source}->${connection.target}`}>
            {connection.source} → {connection.target}
          </li>
        ))}
      </ul>

      <h3>Raw canvas state</h3>
      <pre>{JSON.stringify(canvasState, null, 2)}</pre>
    </div>
  );
}

/**
 * Real-transport wiring (not used by the demo above, shown for reference):
 * once bridge.py's mode is switched to a real MCP server, replace the
 * simulated `runMockSequence()` call with a WebSocket subscription -
 * useMcpCanvas/useMcpListener need no changes either way.
 *
 *   useEffect(() => {
 *     const socket = new WebSocket("ws://localhost:8000/ws/frontend");
 *     socket.onmessage = (event) => receive(JSON.parse(event.data));
 *     return () => socket.close();
 *   }, [receive]);
 */
