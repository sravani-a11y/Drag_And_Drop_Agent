"""Context Builder (Phase 2.2): Context Engineering for the Canvas Context Engine.

Converts the current canvas session (as tracked by context_engine.py /
session_state.py) into a human-readable string suitable for inclusion in
an LLM prompt.

This module ONLY formats. It does not plan, does not execute tools, and
does not update state - it reads the current session via
context_engine.get_context() (a read-only call) and turns it into text.
It is not called from anywhere else in the pipeline yet.
"""

from typing import Any, Dict, List, Optional

from .context_engine import get_context

# Placeholder text used whenever a section of the canvas has nothing to show.
_EMPTY_PLACEHOLDER = "(none)"


class ContextBuilder:
    """Builds a readable text summary of the current canvas session.

    By default, each build call reads the live session from the Context
    Engine, so the text always reflects the canvas as it stands right now.
    A fixed context dict can be supplied instead (e.g. for testing) so the
    builder doesn't depend on the live singleton.
    """

    def __init__(self, context: Optional[Dict[str, Any]] = None) -> None:
        """
        Args:
            context: An explicit context dict to render (the same shape
                returned by context_engine.get_context()). When omitted,
                the current live session is read fresh on every call.
        """
        self._context = context

    def _current_context(self) -> Dict[str, Any]:
        """Return the context dict to render: the fixed one if given, else live."""
        return self._context if self._context is not None else get_context()

    @staticmethod
    def _section(title: str, lines: List[str]) -> str:
        """Format one titled section: a heading, a matching underline, then its lines."""
        underline = "-" * len(title)
        body = "\n".join(lines) if lines else _EMPTY_PLACEHOLDER
        return f"{title}\n{underline}\n{body}"

    @staticmethod
    def _component_label(component_id: str, metadata: Any) -> str:
        """Return the display label for a component: its "name" if present, else its id."""
        if isinstance(metadata, dict) and metadata.get("name"):
            return str(metadata["name"])
        return component_id

    def _components_section(self, components: Dict[str, Any]) -> str:
        lines = [self._component_label(component_id, metadata) for component_id, metadata in components.items()]
        return self._section("Components", lines)

    def _connections_section(self, connections: Dict[str, Dict[str, str]]) -> str:
        lines = [f"{conn['source']} → {conn['target']}" for conn in connections.values()]
        return self._section("Connections", lines)

    def _selected_component_section(self, selected_component: Optional[str]) -> str:
        lines = [selected_component] if selected_component else []
        return self._section("Selected Component", lines)

    def _last_action_section(self, last_action: Optional[Dict[str, Any]]) -> str:
        lines = [self._humanize_action(last_action["action"])] if last_action else []
        return self._section("Last Action", lines)

    @staticmethod
    def _humanize_action(action: str) -> str:
        """Turn a canonical action name (e.g. "move_component") into readable
        text (e.g. "Move Component")."""
        return action.replace("_", " ").title()

    def build_context(self) -> str:
        """Build a readable summary of the current canvas session.

        Returns:
            A plain-text string with a "Current Canvas" header followed by
            Components, Connections, Selected Component, and Last Action
            sections. Any section with nothing to show renders "(none)"
            instead of being omitted, so the shape of the output is always
            the same regardless of how empty the canvas is.
        """
        context = self._current_context()

        sections = [
            "Current Canvas",
            self._components_section(context.get("components", {})),
            self._connections_section(context.get("connections", {})),
            self._selected_component_section(context.get("selected_component")),
            self._last_action_section(context.get("last_action")),
        ]

        return "\n\n".join(sections)

    def build_prompt(self, user_request: str) -> str:
        """Build the canvas context plus the user's request, ready for an LLM prompt.

        Args:
            user_request: The user's natural-language request.

        Returns:
            build_context()'s output, followed by a "User Request" section
            containing `user_request` verbatim.
        """
        request_section = self._section("User Request", [user_request] if user_request else [])
        return f"{self.build_context()}\n\n{request_section}"
