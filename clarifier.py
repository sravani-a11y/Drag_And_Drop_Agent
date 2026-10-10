"""Turns a request the Planner couldn't plan into a helpful reply - or, when the canvas makes the answer obvious, a plan.

Architecture:
    Planner Agent (empty plan) -> Clarifier -> plan (+ reply) | message + reply_type

Called by graph/workflow.py's planner_agent only after create_plan() came
back empty, so every request that already plans is unaffected. Cases:
    - Not a canvas command ("Hi Hema, how are you?", "I can't connect to
      the Wi-Fi"): reply_type "info" - a short friendly reply with examples.
    - A command with something missing ("disconnect ESP32", "add") or an
      ambiguous component (two ESP32s): reply_type "question" - except when
      the canvas leaves only one possible answer (ESP32 has exactly one
      wire), in which case that plan is returned with a reply naming it.
    - A command that can't be done ("disconnect ESP32" when it has no
      wires, or isn't on the canvas): reply_type "error".
    - A command whose components are named in the text but were dropped by
      the LLM: the plan is built from the text.

Read-only: never executes tools or changes the Context Engine.
"""

import re
from typing import Any, Dict, List, Optional

from context import AmbiguousComponentError, ComponentNotFoundError, context_engine
from planner import (
    _ADD_ACTION_WORDS,
    _CONNECT_ACTION_WORDS,
    _detect_clause_intent,
    _find_components_in_text,
    _known_component_names,
    create_plan,
)

_EXAMPLES = 'Try "Add ESP32", "Connect ESP32 to Relay Module" or "Show canvas".'
_GREETING = re.compile(r"^\W*(?:hi|hello|hey|hii+|good (?:morning|afternoon|evening))\b", re.IGNORECASE)

# Greeting and politeness words that can come before a command ("Hi, please
# connect ...", "Could you add ..."), stripped before looking for a leading verb.
_LEADING_FILLER = re.compile(
    r"^(?:\W*(?:hi|hello|hey|hii+|please|pls|kindly|can you|could you|would you|will you|now|then|ok|okay)\b)*\W*",
    re.IGNORECASE,
)

# A request that starts with one of these (after _LEADING_FILLER) is a command.
_COMMAND_VERB = re.compile(
    r"(?:add|create|place|insert|put|attach|connect|link|wire|disconnect|unwire|detach|remove|delete|move|drag)\b",
    re.IGNORECASE,
)

# The word used for each intent in a reply ("Which component do you want to add?").
_VERBS = {
    "add_component": "add",
    "remove_component": "remove",
    "move_component": "move",
    "connect_components": "connect",
    "disconnect_components": "disconnect",
}


def _is_command(user_request: str, knowledge: Dict[str, Any]) -> bool:
    """Whether the request is addressed to the canvas: it names a catalog component, or starts with a command verb.

    A verb in the middle of a sentence ("I can't connect to the Wi-Fi") is
    not enough on its own - that is small talk, not a canvas action.
    """
    if _find_components_in_text(user_request, _known_component_names(knowledge)):
        return True
    return bool(_COMMAND_VERB.match(_LEADING_FILLER.sub("", user_request)))


def _requested_intent(user_request: str) -> Optional[str]:
    """The canvas intent the request's own verbs ask for, or None.

    Deliberately ignores the LLM's guess: for small talk the model can
    still return some intent, and that must not turn "Hi" into a question
    about which component to add.
    """
    intent = _detect_clause_intent(user_request)
    if intent is None and _CONNECT_ACTION_WORDS.search(user_request):
        intent = "connect_components"
    if intent is None and _ADD_ACTION_WORDS.search(user_request):
        intent = "add_component"
    return intent


def _mentioned_components(user_request: str, llm_output: Dict[str, Any], knowledge: Dict[str, Any]) -> List[str]:
    """Every component the request names - catalog names found in the text, then the LLM's own fields - deduplicated."""
    names = _find_components_in_text(user_request, _known_component_names(knowledge))
    if isinstance(llm_output, dict):
        for field in ("source_component", "source", "target_component", "target", "component"):
            value = llm_output.get(field)
            if isinstance(value, str) and value.strip():
                names.append(value.strip())
        components = llm_output.get("components")
        if isinstance(components, list):
            names.extend(c.strip() for c in components if isinstance(c, str) and c.strip())

    distinct: Dict[str, str] = {}
    for name in names:
        distinct.setdefault(name.lower(), name)
    return list(distinct.values())


def _join(labels: List[str], word: str) -> str:
    return labels[0] if len(labels) == 1 else f"{', '.join(labels[:-1])} {word} {labels[-1]}"


def _reply(message: str, reply_type: str) -> Dict[str, Any]:
    return {"message": message, "reply_type": reply_type}


def _ambiguity_question(component: str, verb: str, follow_up: str) -> Optional[Dict[str, Any]]:
    """A question listing every instance `component` could mean, or None when it isn't ambiguous."""
    labels = context_engine.matching_labels(component)
    if len(labels) < 2:
        return None
    return _reply(f"More than one {component} is on the canvas: {_join(labels, 'and')}. "
                  f"Which one do you want to {verb}{follow_up}?", "question")


def _resolve_disconnect(component: str) -> Dict[str, Any]:
    """Disconnect with one component named: use its only wire, or ask which one."""
    try:
        partners = context_engine.connected_components(component)
    except ComponentNotFoundError:
        return _reply(f"Cannot disconnect {component} because {component} was not found.", "error")
    except AmbiguousComponentError:
        return _ambiguity_question(component, "disconnect", ", and from which component")

    label = context_engine.matching_labels(component)[0]
    partner_labels = [context_engine.component_label(partner) for partner in partners]
    if len(partners) == 1:
        plan = create_plan({"intent": "disconnect_components", "source_component": component, "target_component": partners[0]})
        return {"plan": plan, "reply": f"Generated a command to disconnect {label} from {partner_labels[0]} (its only connection)."}
    if not partners:
        return _reply(f"{label} isn't connected to anything.", "error")
    return _reply(f"{label} is connected to {_join(partner_labels, 'and')}. Which one do you want to disconnect it from?", "question")


def resolve_unplanned(user_request: str, llm_output: Dict[str, Any], knowledge: Dict[str, Any]) -> Dict[str, Any]:
    """Decide what to do with a request create_plan() returned no plan for.

    Returns:
        {"plan": [...], "reply": str | None} when the request can be
        completed after all (reply, when set, describes what was chosen),
        else {"message": str, "reply_type": "info" | "question" | "error"}.
    """
    user_request = user_request or ""
    intent = _requested_intent(user_request) if _is_command(user_request, knowledge) else None
    verb = _VERBS.get(intent)
    if verb is None:
        greeting = "Hi! " if _GREETING.search(user_request) else ""
        return _reply(f"{greeting}I can help you build your circuit on the canvas. {_EXAMPLES}", "info")

    components = _mentioned_components(user_request, llm_output, knowledge)

    if intent in ("connect_components", "disconnect_components"):
        if len(components) >= 2:
            plan = create_plan({"intent": intent, "source_component": components[0], "target_component": components[1]})
            if plan:
                return {"plan": plan, "reply": None}
        if not components:
            return _reply(f"Which two components do you want to {verb}?", "question")
        if intent == "disconnect_components":
            return _resolve_disconnect(components[0])
        return _ambiguity_question(components[0], "connect", ", and to which component") or _reply(
            f"Connect {components[0]} to which component?", "question"
        )

    if not components:
        return _reply(f"Which component do you want to {verb}?", "question")

    plan = create_plan({"intent": intent, "components": components})
    if plan:
        return {"plan": plan, "reply": None}
    return _reply(f"I couldn't work out how to {verb} {', '.join(components)}. {_EXAMPLES}", "error")
