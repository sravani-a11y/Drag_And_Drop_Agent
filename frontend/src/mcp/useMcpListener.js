/**
 * Transport-agnostic MCP command listener.
 *
 * This hook only defines *how* one incoming command reaches the canvas
 * reducer (useMcpCanvas) - not *where* it comes from. Today, tests and
 * demos call `receive(command)` directly to simulate an incoming command
 * (mock JSON). Wiring this to bridge.py's real /ws/frontend WebSocket
 * later is a one-line change - a socket's onmessage handler just calls
 * receive(JSON.parse(event.data)) - nothing in useMcpCanvas changes.
 */
import { useCallback, useRef } from "react";

export function useMcpListener(onCommand) {
  // Keeps `receive` referentially stable across renders (a real WebSocket
  // effect can depend on it without resubscribing every render) while
  // still always calling the latest onCommand.
  const onCommandRef = useRef(onCommand);
  onCommandRef.current = onCommand;

  const receive = useCallback((command) => {
    console.log("[MCP] received:", command);
    onCommandRef.current(command);
  }, []);

  return { receive };
}
