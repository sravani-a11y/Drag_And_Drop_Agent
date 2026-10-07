"""LLM invocation for the configured provider.

Accepts a fully prepared prompt (built by prompt_builder.py) plus the raw
user request, and returns the LLM's parsed JSON response as a dictionary.

Phase 2.3: before every LLM request, the current canvas context (from the
Canvas Context Engine, via ContextBuilder.build_context()) is prepended to
the existing prompt, so the LLM is aware of the live canvas state
(components, connections, selection, last action) in addition to the
registries and output-format instructions prompt_builder.py already
supplies. This module still has no knowledge of prompt_builder's internal
structure or schema - it only prepends canvas context ahead of whatever
prompt it is given. Changes to prompt_builder.py's output schema or
instructions never require changes here.

Phase 3.3: uses ContextBuilder.build_context() (the canvas snapshot alone)
rather than build_prompt() (which also bundled its own copy of the user
request). prompt_builder.py's own output now places the real user request
at the very end of the prompt it builds (see prompt_builder.build_prompt()),
so the request appears exactly once, as the most recent thing the model
reads before answering, instead of twice at two different points in the
combined prompt.

Phase 3.4 - True Multi-Intent Planning:
    prompt_builder.py is out of scope for this phase, so its single-object
    schema/instructions are untouched. Instead, _MULTI_ACTION_INSTRUCTIONS
    is appended after prompt_builder's output (i.e. as the very last thing
    the model reads before answering) to additionally offer a second,
    equally valid response shape - {"actions": [ {...one action...}, ... ]}
    - for requests that describe more than one operation, with each list
    entry using exactly the same per-action fields as the existing single
    object. The model still returns the plain single-object shape for a
    single-operation request; nothing about that path changes.

    Parsing grew to match: _normalize_actions_field() applies the exact
    same per-field normalization already used for a single response
    (_normalize_intent_field, _normalize_component_fields) to every item in
    "actions", when present. analyze() still returns the JSON exactly as
    shaped by the model (either the legacy flat object or the new
    {"actions": [...]} list) - this module does not collapse the two
    shapes into one, so graph/workflow.py's planner_node can do the
    "detect actions, else fall back to the legacy path" branching the
    architecture calls for, using the existing planner.create_plan()
    unchanged either way.
"""

import json
import logging
import re
from typing import Any

from ollama import ChatResponse

from config import LLM_MODEL, get_client
from context.context_builder import ContextBuilder

logger = logging.getLogger(__name__)

_CODE_FENCE_PATTERN = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)
_DEBUG_BANNER = "=" * 50

# Matches the boundary just before an uppercase letter that follows a
# lowercase letter or digit, e.g. the "C" in "showCanvas" - used to split
# camelCase into words before the whole string is lowercased.
_CAMEL_CASE_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")

# The fixed, canonical intent vocabulary (mirrors prompt_builder.py's
# _VALID_INTENTS, which instructs the LLM to always choose one of these).
# Kept here too - not imported from prompt_builder.py - so this module's
# guarantee ("intent is canonical whenever it's recognizable at all") does
# not depend on prompt_builder.py's prompt text actually working; it's a
# deterministic backstop, not a duplicate of the prompt instruction.
_CANONICAL_INTENTS = frozenset(
    {
        "add_component",
        "remove_component",
        "move_component",
        "connect_components",
        "disconnect_components",
        "get_canvas_state",
    }
)

# Synonyms actually observed for the canvas-state intent: the LLM has
# returned "show", "show_canvas", and "get_canvas_state" for the same
# "show canvas" request, sometimes ignoring the prompt's explicit enum
# instruction. This is a deterministic safety net on top of that
# instruction (see prompt_builder.py's _build_output_format_section), not
# a replacement for it - the prompt is what shapes the model's behavior;
# this is what guarantees the contract even when the model doesn't comply.
_GET_CANVAS_STATE_SYNONYMS = frozenset(
    {"show", "show_canvas", "display", "display_canvas", "view", "view_canvas", "canvas", "canvas_state"}
)


def _canonicalize_intent(intent: str) -> str:
    """Coerce an already-snake_cased intent to the fixed canonical vocabulary.

    A known synonym for "get_canvas_state" is mapped to it. Anything else
    is left unchanged - this backstop only corrects the specific variance
    that has actually been observed; it does not invent a mapping for an
    intent that isn't recognizable at all (an unrecognized intent is
    logged and passed through, so it fails visibly downstream in
    planner.py rather than being silently guessed at here).
    """
    if intent in _GET_CANVAS_STATE_SYNONYMS:
        logger.info("Canonicalized intent %r -> 'get_canvas_state'", intent)
        return "get_canvas_state"

    if intent not in _CANONICAL_INTENTS:
        logger.warning(
            "Intent %r is not in the canonical vocabulary %s - passing it through unchanged",
            intent,
            sorted(_CANONICAL_INTENTS),
        )

    return intent


class LLMResponseError(Exception):
    """Raised when the LLM response is empty or is not valid JSON."""


def _extract_response_text(response: ChatResponse) -> str:
    """Extract the text content from an Ollama chat response."""
    message = getattr(response, "message", None)
    if message is None or not message.content:
        return ""
    return message.content.strip()


def _strip_code_fences(text: str) -> str:
    """Remove a leading/trailing markdown code fence, if the LLM added one."""
    return _CODE_FENCE_PATTERN.sub("", text).strip()


def _to_snake_case(value: str) -> str:
    """Canonicalize a string to snake_case.

    The LLM is not consistent about how it cases or separates a multi-word
    intent across calls - "show_canvas", "show canvas", "showCanvas", and
    "ShowCanvas" have all been observed for the same request. Splitting
    camelCase boundaries must happen *before* lowercasing - lowercasing
    first destroys the capital-letter signal a camelCase split depends on,
    collapsing e.g. "showCanvas" into the unsplittable "showcanvas".
    """
    split_camel_case = _CAMEL_CASE_BOUNDARY.sub("_", value.strip())
    return re.sub(r"[\s\-]+", "_", split_camel_case).lower()


def _normalize_intent_field(result: Any) -> Any:
    """Canonicalize result["intent"], if present.

    This is the single point where every intent value the LLM might
    produce - regardless of how it happened to case, separate, or phrase
    it - is reduced to one consistent, canonical form before any other
    module (e.g. planner.py) ever sees it: first to snake_case (handles
    casing/separator variance), then coerced to the fixed canonical
    vocabulary (handles the known "get_canvas_state" synonym variance).
    Only the "intent" field is touched; other fields (e.g. component
    names) keep their original casing.
    """
    if isinstance(result, dict) and isinstance(result.get("intent"), str):
        snake_case_intent = _to_snake_case(result["intent"])
        result["intent"] = _canonicalize_intent(snake_case_intent)
    return result


# Matches a leading English article ("a ", "an ", "the ") on a component
# name field - e.g. the prompt asks for "Relay Module" but a small model
# occasionally echoes back "a relay module" or "the relay module" instead.
_LEADING_ARTICLE_PATTERN = re.compile(r"^(?:a|an|the)\s+", re.IGNORECASE)


def _clean_component_value(value: Any) -> Any:
    """Trim whitespace, surrounding quotes, and a leading article from one field.

    Pure text hygiene - it does not decide or infer anything about intent,
    it only tidies whatever string the model already chose. Non-string
    values (including the field being absent, i.e. None) pass through
    unchanged.
    """
    if not isinstance(value, str):
        return value

    cleaned = value.strip().strip("\"'").strip()
    cleaned = _LEADING_ARTICLE_PATTERN.sub("", cleaned)
    return cleaned.strip()


def _normalize_components_list(value: Any) -> Any:
    """Clean every entry of a "components" list the same way a single component field is cleaned.

    Non-list values (including the field being absent) pass through
    unchanged. Entries that clean down to an empty string are dropped,
    since planner.py treats "components" as the set of components to plan
    for - a blank entry there would otherwise become a bogus step.
    """
    if not isinstance(value, list):
        return value
    cleaned = [_clean_component_value(item) for item in value]
    return [item for item in cleaned if item]


def _normalize_component_fields(result: Any) -> Any:
    """Clean every component-name field a single action might carry.

    Covers "component", "source_component"/"target_component" (and their
    "source"/"target" shorthand, used in the multi-intent action shape - see
    _MULTI_ACTION_INSTRUCTIONS), and "components" (plural - a list, used
    when one operation names more than one component). This is a
    deterministic backstop for the prompt's "return the exact, real
    component name" instruction - it does not change which component the
    model chose, only strips incidental wrapping (whitespace, quotes, a
    leading "a"/"an"/"the") the model added around it, so a well-classified
    response like {"component": "a relay module"} reaches planner.py as
    "relay module" instead of a phrase planner.py's intent planners would
    otherwise pass straight through as a literal component id. Each entry
    of "components" gets the same per-item cleanup via
    _normalize_components_list().
    """
    if not isinstance(result, dict):
        return result

    for field in ("component", "source_component", "target_component", "source", "target"):
        if field in result:
            result[field] = _clean_component_value(result[field])

    if "components" in result:
        result["components"] = _normalize_components_list(result["components"])

    return result


def _normalize_actions_field(result: Any) -> Any:
    """Apply the same per-field normalization to every entry in result["actions"].

    Phase 3.4: when the model uses the multi-intent shape
    ({"actions": [...]}) instead of a single flat object, each entry in
    that list carries the same "intent"/"component"/"source_component"/
    "target_component" fields a flat response would - so it gets exactly
    the same cleanup (_normalize_intent_field, _normalize_component_fields)
    a flat response already gets, just once per action instead of once for
    the whole result. A non-dict entry (a malformed action) is left as-is;
    graph/workflow.py's planner_node is what decides whether to skip it.
    """
    if not isinstance(result, dict):
        return result

    actions = result.get("actions")
    if not isinstance(actions, list):
        return result

    normalized_actions = []
    for action in actions:
        if isinstance(action, dict):
            action = _normalize_intent_field(action)
            action = _normalize_component_fields(action)
        normalized_actions.append(action)

    result["actions"] = normalized_actions
    return result


# Phase 3.4: appended after prompt_builder.py's own output (untouched,
# since prompt_builder.py is out of scope this phase) so the model reads it
# last, right before it has to answer. Offers the {"actions": [...]} shape
# as an alternative to the single-object schema prompt_builder.py already
# describes, strictly for requests that describe more than one operation -
# the single-object path is unchanged and still expected for everything
# else. Every action's fields mirror the single-object schema exactly, so
# nothing new is introduced for the model to learn beyond "put a list of
# these under 'actions' when there's more than one."
_MULTI_ACTION_INSTRUCTIONS = """One more thing: some requests describe more than one operation in a single sentence - not just one operation naming several components (that still uses the single object above, with "components"), but genuinely different operations chained together: "Add ESP32 and Relay Module, then connect them", "Add ESP32 and connect it to Relay Module", "Move ESP32 then show canvas". Watch for a second verb (add/remove/move/connect/disconnect/show) later in the same sentence - that is what signals a second, different operation rather than another component of the same one. For a request like that, respond with a list of actions instead of one object:

{
    "actions": [
        {"intent": "...", "components": ["..."], "source": "", "target": "", "confidence": "...", "reasoning": "..."},
        {"intent": "...", "components": [], "source": "...", "target": "...", "confidence": "...", "reasoning": "..."}
    ]
}                                                                                                                                           

Each entry in "actions" uses the same fixed intent vocabulary already described above, plus these fields:
- "components": for "add_component", "remove_component", or "move_component" - a list of every component name that single action applies to (one entry for a single component, e.g. ["ESP32"]).
- "source" / "target": for "connect_components" or "disconnect_components" - the two component names being wired (equivalent to "source_component"/"target_component" above, just shorter).
- "confidence" and "reasoning": exactly as described above.

List one entry per individual operation, in the order the user described them. Resolve a pronoun or reference such as "them"/"it"/"these" against the other actions in this same response (an action that adds a component makes that component's exact name available to a later action in the list) as well as against the current canvas.

If the request only describes one operation - even if that operation names several components - respond with the single object exactly as already described; do not wrap it in "actions".

Example 1:
Request: "Add ESP32 and Relay Module, then connect them."
Response:
{
    "actions": [
        {"intent": "add_component", "components": ["ESP32", "Relay Module"], "source": "", "target": "", "confidence": "high", "reasoning": "Both components are added before anything is wired."},
        {"intent": "connect_components", "components": [], "source": "ESP32", "target": "Relay Module", "confidence": "high", "reasoning": "\\"Them\\" resolves to the two components just added above."}
    ]
}

Example 2:
Request: "Add ESP32 and connect it to Relay Module."
Response:
{
    "actions": [
        {"intent": "add_component", "components": ["ESP32"], "source": "", "target": "", "confidence": "high", "reasoning": "The first verb, add, applies to ESP32."},
        {"intent": "connect_components", "components": [], "source": "ESP32", "target": "Relay Module", "confidence": "high", "reasoning": "The second verb, connect, is a different operation; \\"it\\" resolves to the ESP32 just added above."}
    ]
}

Respond with ONLY valid JSON - either the single object, the "actions" object above, or (equivalently) just the JSON array of actions on its own with no surrounding "actions" object - never any text before or after it."""


def _print_debug_context(full_prompt: str) -> None:
    """Print the exact prompt sent to the LLM, for manual debugging only.

    Gated behind DEBUG-level logging so it stays silent in normal
    operation - this is a plain `print()` (not a log record) so the banner
    is easy to spot in a terminal while testing.
    """
    if not logger.isEnabledFor(logging.DEBUG):
        return
    print(f"\n{_DEBUG_BANNER}")
    print("CONTEXT SENT TO LLM")
    print(_DEBUG_BANNER)
    print(f"\n{full_prompt}\n")
    print(_DEBUG_BANNER)


def _build_context_aware_prompt(prompt: str, user_request: str) -> str:
    """Prepend the current canvas context ahead of the given prompt.

    Uses ContextBuilder().build_context() (Phase 2.2) to render the live
    canvas session - components, connections, selection, last action - then
    places `prompt` (the existing, unmodified output of
    prompt_builder.build_prompt()) after it. This keeps prompt_builder's
    schema/output-format instructions intact while adding canvas awareness
    the LLM didn't have before.

    Phase 3.3: build_context() (not build_prompt(user_request)) is used
    deliberately - prompt_builder.build_prompt() now places the real user
    request at the very end of `prompt` itself (see prompt_builder.py), so
    adding a second copy of it here would duplicate it and push the actual
    question earlier in the combined prompt instead of leaving it last.
    `user_request` is still accepted (unused) so this function's signature
    doesn't need to change if a future phase needs it again.

    Phase 3.4: _MULTI_ACTION_INSTRUCTIONS is appended after `prompt`, so it
    is the very last thing the model reads - see that constant's comment
    for why (prompt_builder.py is out of scope this phase, so the
    multi-intent schema is layered on here instead of edited into it).

    Args:
        prompt: The existing fully-built prompt (system instructions,
            output format, examples, registries, and the user's request)
            from prompt_builder.py.
        user_request: The raw user request driving this call.

    Returns:
        The canvas context, `prompt`, and the multi-action instructions,
        as one string, in that order.
    """
    del user_request  # kept in the signature; see docstring.
    canvas_context = ContextBuilder().build_context()
    return f"{canvas_context}\n\n{prompt}\n\n{_MULTI_ACTION_INSTRUCTIONS}"


def analyze(prompt: str, user_request: str) -> dict:
    """Send a context-aware prompt to the configured LLM and return its parsed JSON response.

    Args:
        prompt: A complete prompt string, typically produced by
            `prompt_builder.build_prompt(knowledge, user_request)`.
        user_request: The raw user request. Used to build the current
            canvas context (via ContextBuilder), which is prepended to
            `prompt` before the request is sent to the LLM.

    Returns:
        The LLM's response, parsed as a Python dictionary, in whichever
        shape the model chose (Phase 3.4): the legacy single-object shape
        ({"intent": ..., "component": ..., ...}) for a single operation, or
        {"actions": [ {...}, {...} ]} - a list of entries in that same
        per-action shape - for a request describing more than one. A bare
        JSON array (no surrounding "actions" object) is also accepted from
        the model and normalized into the {"actions": [...]} shape before
        it is returned. This function does not collapse the single-object
        and multi-action shapes into one common shape; whichever the model
        chose is returned as-is (after field normalization), so callers can
        branch on whether "actions" is present.

    Raises:
        LLMResponseError: if the LLM response is empty or is not valid JSON.
    """
    client = get_client()

    full_prompt = _build_context_aware_prompt(prompt, user_request)

    logger.debug("Sending prompt to LLM (model=%s, prompt_length=%d)", LLM_MODEL, len(full_prompt))
    _print_debug_context(full_prompt)

    response = client.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": full_prompt}],
        format="json",
    )

    raw_text = _extract_response_text(response)
    if not raw_text:
        logger.error("LLM returned an empty response")
        raise LLMResponseError("LLM returned an empty response")

    cleaned_text = _strip_code_fences(raw_text)

    try:
        result = json.loads(cleaned_text)
    except json.JSONDecodeError as exc:
        logger.error("LLM response is not valid JSON: %s", raw_text)
        raise LLMResponseError(f"LLM response is not valid JSON: {exc}") from exc

    if isinstance(result, list):
        # _MULTI_ACTION_INSTRUCTIONS accepts a bare JSON array as equivalent
        # to {"actions": [...]} - normalize it to that shape here so every
        # downstream consumer (this module's own normalization below,
        # planner.py, graph/workflow.py) only ever has to look for
        # "actions" on a dict, exactly as before.
        result = {"actions": result}

    if isinstance(result, dict) and "actions" not in result and isinstance(result.get("intents"), list):
        # Some callers use "intents" instead of "actions" for the same
        # per-item shape (e.g. {"intents": [{"intent": ..., "component":
        # ...}, ...]}) - adopt it under "actions" so nothing downstream
        # needs to know this key exists.
        result["actions"] = result["intents"]

    print("\n========================")
    print("RAW LLM RESPONSE")
    print("========================")
    print(json.dumps(result, indent=2))
    print("========================\n")

    result = _normalize_intent_field(result)
    result = _normalize_component_fields(result)
    result = _normalize_actions_field(result)

    logger.debug("LLM response parsed successfully")
    return result
