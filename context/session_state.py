"""SessionState: the data container for a single canvas editing session.

This holds the current state of the canvas only - which components exist,
where they are positioned, how they're connected, what's currently
selected, and the most recent action taken. It is NOT chatbot memory and
NOT long-term memory: there is no persistence, no history beyond the last
action, and no notion of past sessions. Restarting the process or calling
`reset()` discards it completely.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class SessionState:
    """Holds the current canvas editing session.

    Attributes:
        components: Maps a component id to its metadata (e.g. the id itself,
            and any other descriptive fields future phases choose to store).
        component_positions: Maps a component id to its current
            {"x": int, "y": int} position on the canvas.
        connections: Maps a connection key (see
            context_engine._connection_key) to {"source": str, "target": str}.
        selected_component: The component id currently selected, or None if
            nothing is selected.
        last_action: A record of the most recently applied action
            ({"action": str, "timestamp": str, ...details}), or None if no
            action has been applied yet this session.
    """

    components: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    component_positions: Dict[str, Dict[str, int]] = field(default_factory=dict)
    connections: Dict[str, Dict[str, str]] = field(default_factory=dict)
    selected_component: Optional[str] = None
    last_action: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Return the full session state as a plain, JSON-serializable dict.

        Returns shallow copies of the mutable fields so callers can't
        accidentally mutate the live session through the returned dict.
        """
        return {
            "components": dict(self.components),
            "component_positions": dict(self.component_positions),
            "connections": dict(self.connections),
            "selected_component": self.selected_component,
            "last_action": self.last_action,
        }

    def reset(self) -> None:
        """Clear the session back to its initial, empty state."""
        self.components.clear()
        self.component_positions.clear()
        self.connections.clear()
        self.selected_component = None
        self.last_action = None
