"""Converts structured LLM output into an execution plan.

Architecture:
    User -> Knowledge Loader -> Prompt Builder -> LLM -> Planner -> Tool Executor

This module only builds an execution plan (a list of {"tool", "input"} steps).
It never executes tools, calls backend APIs, or talks to the frontend.

Phase 3.2 - Multi-Step Intelligent Planner:
    The LLM (llm.py) and its prompt (prompt_builder.py) are fixed contracts -
    each call still returns exactly one {"intent", "component", ...} guess for
    the *whole* request, and neither module is touched by this phase. So when
    one request actually describes several operations (e.g. "Add ESP32 and
    Relay Module", "Move ESP32 then show canvas"), detecting that is done
    here, deterministically, from the raw request text - not by asking the
    LLM again. create_plan() now accepts the raw user_request and the
    Knowledge Loader's registries (both already computed upstream, by
    graph/workflow.py) purely to attempt this segmentation; if it finds at
    most one operation (i.e. this is a genuinely single-action request, or
    the caller didn't pass user_request/knowledge at all), it falls through
    unchanged to the original single-intent path below, so every existing
    single-action command keeps working exactly as before. See
    _build_multi_step_plan() for the segmentation itself.
"""

import logging
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Matches the boundary just before an uppercase letter that follows a
# lowercase letter or digit, e.g. the "C" in "showCanvas" - used to split
# camelCase into words before the whole string is lowercased. llm.py
# already normalizes intent casing at the source, but _normalize_intent()
# applies the same rule independently so planner.py is not solely
# dependent on that upstream step to recognize a known intent.
_CAMEL_CASE_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")

PlanStep = Dict[str, Any]
Plan = List[PlanStep]

# Temporary default canvas position for a newly added component. Every
# single-op add_component step (i.e. one add per request) places at the
# same spot until a real canvas positioning engine can compute where it
# should actually go.
_DEFAULT_ADD_X = 200
_DEFAULT_ADD_Y = 200

# Horizontal spacing applied between components added by the *same*
# multi-step request (e.g. "Add ESP32 and Relay Module") - see
# _build_multi_step_plan(). Without this, every add_component step in one
# request would still target _DEFAULT_ADD_X/_DEFAULT_ADD_Y and every added
# component would land on top of the previous one. This only applies
# within _build_multi_step_plan()'s own plan assembly - the single-op path
# (_plan_add_component(), still used by _INTENT_PLANNERS and by the
# per-action calls graph/workflow.py's own multi-intent merge makes) is
# unchanged and always places at _DEFAULT_ADD_X/_DEFAULT_ADD_Y.
_ADD_X_STEP = 200

# Temporary default target position for a moved component. Every
# move_component step moves to the same spot until a real canvas
# positioning engine can compute where it should actually go.
_DEFAULT_MOVE_X = 300
_DEFAULT_MOVE_Y = 300


def _plan_step(tool: str, input_value: Any) -> PlanStep:
    return {"tool": tool, "input": input_value}


def _plan_add_component_at(component: str, x: int, y: int) -> Plan:
    """Build add_component's two steps at an explicit (x, y).

    Factored out of _plan_add_component() so _build_multi_step_plan() can
    place several components added by the same request at distinct
    positions (see _ADD_X_STEP) while the single-op path below keeps using
    one fixed default.
    """
    return [
        _plan_step("search_component", component),
        _plan_step("add_component", {"component_id": component, "x": x, "y": y}),
    ]


def _plan_add_component(component: str) -> Plan:
    return _plan_add_component_at(component, _DEFAULT_ADD_X, _DEFAULT_ADD_Y)


def _plan_delete_component(component: str) -> Plan:
    return [
        _plan_step("search_component", component),
        _plan_step("delete_component", component),
    ]


def _plan_move_component(component: str) -> Plan:
    return [
        _plan_step("search_component", component),
        _plan_step(
            "move_component",
            {"component_id": component, "x": _DEFAULT_MOVE_X, "y": _DEFAULT_MOVE_Y},
        ),
    ]


def _plan_get_canvas_state() -> Plan:
    """get_canvas_state needs no component - it's a single, argument-free step."""
    return [_plan_step("get_canvas_state", None)]


def _plan_connect_components(source_component: str, target_component: str) -> Plan:
    # Keys match ComponentTool.connect_components(source_component, target_component).
    return [
        _plan_step("search_component", source_component),
        _plan_step("search_component", target_component),
        _plan_step(
            "connect_components",
            {"source_component": source_component, "target_component": target_component},
        ),
    ]


def _plan_disconnect_components(source_component: str, target_component: str) -> Plan:
    # Keys match ComponentTool.disconnect_components(source_component, target_component).
    return [
        _plan_step("search_component", source_component),
        _plan_step("search_component", target_component),
        _plan_step(
            "disconnect_components",
            {"source_component": source_component, "target_component": target_component},
        ),
    ]


def _plan_backend_operation(backend_api: str, component: str) -> Plan:
    """Prepare a backend execution step for a resolved backend API.

    This only builds the step - it never calls the backend API.
    """
    return [_plan_step("backend_api", {"api": backend_api, "component": component})]


# ---------------------------------------------------------------------------
# Phase 3.2: Multi-Step Intelligent Planner
#
# Splits one raw request into ordered (intent, payload) operations using the
# component registry (from KnowledgeLoader) as ground truth for what counts
# as a component mention, then reuses the exact single-op plan builders above
# (_plan_add_component, _plan_connect_components, etc.) for each operation -
# no tool-building or business logic is duplicated, only the segmentation of
# the request text is new.
# ---------------------------------------------------------------------------

# Candidate boundaries between two clauses: a sequencing word ("then",
# "and") or punctuation (",", ";"). This is only a *candidate* list, not a
# guaranteed split point - "and" and "," just as often join two components
# within the same action ("Add ESP32 and Relay Module") as they join two
# separate actions ("Add ESP32, then connect it to Relay Module"). Which
# case applies is decided by _split_into_clauses(), which only starts a new
# clause at a boundary whose following fragment has its own detected verb -
# see that function's docstring.
_SEQUENCE_SEPARATOR_PATTERN = re.compile(r"\b(?:then|and)\b|[,;]", re.IGNORECASE)

# Ordered scan for a clause's intent verb. Checked in this order (not
# alphabetical) because "disconnect_components" must win over
# "connect_components" - "disconnect" also contains a boundary-safe
# "connect", so the more specific pattern is tried first. Anything checked
# here should be a keyword whose presence unambiguously implies one intent
# for the whole clause - the same tradeoff prompt_builder.py's fixed
# instruction set makes, just applied to a clause instead of a full request.
_CLAUSE_INTENT_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\bdisconnect\b", re.IGNORECASE), "disconnect_components"),
    (re.compile(r"\bconnect\b", re.IGNORECASE), "connect_components"),
    (re.compile(r"\b(remove|delete)\b", re.IGNORECASE), "remove_component"),
    (re.compile(r"\bmove\b", re.IGNORECASE), "move_component"),
    (re.compile(r"\badd\b", re.IGNORECASE), "add_component"),
    (re.compile(r"\bcanvas\b", re.IGNORECASE), "get_canvas_state"),
]

Operation = Tuple[str, Any]


def _known_component_names(knowledge: Dict[str, Any]) -> List[str]:
    """Return every component name in the Knowledge Loader's registry, as-is.

    Preserves the registry's own casing (e.g. "ESP32", "Relay Module") so a
    multi-step match reuses the same canonical spelling a well-formed
    llm_output["component"] would already carry - tools/context sync never
    see a differently-cased component id depending on which path built the
    plan.
    """
    return [entry.get("name", "") for entry in (knowledge or {}).get("components", []) if entry.get("name")]


def _find_components_in_text(text: str, known_names: List[str]) -> List[str]:
    """Find every known component name mentioned in `text`, in reading order.

    Longer names are matched first and their matched span is excluded from
    further matches, so a multi-word name (e.g. "Relay Module") is never
    also double-counted as containing a shorter, unrelated name. Returns
    names in the order they appear in the text (not registry order), since
    that order becomes source/target for connect/disconnect and step order
    for add/remove/move.
    """
    if not text or not known_names:
        return []

    consumed = bytearray(len(text))
    found: List[Tuple[int, str]] = []

    for name in sorted(set(known_names), key=len, reverse=True):
        pattern = re.compile(r"\b" + re.escape(name) + r"\b", re.IGNORECASE)
        for match in pattern.finditer(text):
            start, end = match.span()
            if any(consumed[start:end]):
                continue
            found.append((start, name))
            consumed[start:end] = bytes(end - start)

    found.sort(key=lambda pair: pair[0])
    return [name for _, name in found]


def _detect_clause_intent(clause: str) -> Optional[str]:
    """Return the canonical intent implied by a clause's keywords, or None."""
    for pattern, intent in _CLAUSE_INTENT_PATTERNS:
        if pattern.search(clause):
            return intent
    return None


def _split_into_clauses(user_request: str) -> List[str]:
    """Split a request into ordered clauses, one per described operation.

    First splits `user_request` at every candidate boundary
    (_SEQUENCE_SEPARATOR_PATTERN: "then", "and", ",", ";"), then walks the
    resulting fragments and merges each one that has no detected verb of
    its own (_detect_clause_intent()) back into the clause being built -
    it's read as a continuation of the current action's component list,
    not a new action. A fragment that does have its own verb starts a new
    clause instead.

    This is what makes "Add ESP32 and Relay Module, then connect them."
    split into exactly two clauses ("Add ESP32 Relay Module", "connect
    them") instead of three or four: "and" and "," both fire as candidate
    boundaries, but "Relay Module" has no verb of its own, so it merges
    back into the "Add ESP32" clause instead of starting a bogus one.
    Likewise, "Add ESP32, Relay Module, and DHT11 Sensor" (a plain
    comma-separated list, no second action at all) stays a single clause.
    """
    fragments = [fragment.strip() for fragment in _SEQUENCE_SEPARATOR_PATTERN.split(user_request) if fragment.strip()]

    clauses: List[str] = []
    current: Optional[str] = None
    for fragment in fragments:
        if current is None or _detect_clause_intent(fragment) is not None:
            if current is not None:
                clauses.append(current)
            current = fragment
        else:
            current = f"{current} {fragment}"

    if current is not None:
        clauses.append(current)

    return clauses


# Pronouns _clause_operations() falls back to when a clause's own text
# doesn't name enough components for its intent (e.g. "connect them" names
# none) - resolved against components mentioned earlier in the same
# request instead. Plural forms resolve to the last two distinct
# components mentioned so far, singular forms to the single most recent
# one. This mirrors how llm.py's _MULTI_ACTION_INSTRUCTIONS already asks
# the LLM to resolve the same pronouns against "the other actions in this
# same response" - just done deterministically here, as a backstop for
# when the model doesn't reliably decompose the request itself.
_PLURAL_PRONOUNS = frozenset({"them", "these", "those"})
_SINGULAR_PRONOUNS = frozenset({"it", "this", "that"})


def _clause_has_pronoun(clause: str, pronouns: frozenset) -> bool:
    """Return whether `clause` contains any of `pronouns` as a whole word."""
    pattern = re.compile(r"\b(?:" + "|".join(pronouns) + r")\b", re.IGNORECASE)
    return pattern.search(clause) is not None


def _resolve_pronoun_pair(clause: str, mentioned: List[str]) -> Optional[Tuple[str, str]]:
    """Resolve a plural pronoun ("them"/"these"/"those") to the last two distinct components mentioned so far.

    Returned in the order they were originally mentioned, so "Add ESP32
    and Relay Module, then connect them" connects ESP32 -> Relay Module,
    not the reverse - which component ends up "source" vs "target" doesn't
    affect the resulting connection (context_engine._connection_key() is
    order-independent), but keeping the order stable makes the plan
    predictable to read.
    """
    if not _clause_has_pronoun(clause, _PLURAL_PRONOUNS):
        return None

    distinct = list(dict.fromkeys(mentioned))
    if len(distinct) < 2:
        return None
    return distinct[-2], distinct[-1]


def _resolve_pronoun_single(clause: str, mentioned: List[str]) -> Optional[str]:
    """Resolve a singular pronoun ("it"/"this"/"that") to the last component mentioned so far."""
    if not _clause_has_pronoun(clause, _SINGULAR_PRONOUNS):
        return None
    if not mentioned:
        return None
    return mentioned[-1]


def _resolve_connect_pair(clause: str, mentioned: List[str], found: List[str]) -> Optional[Tuple[str, str]]:
    """Resolve source/target for a connect_components/disconnect_components clause.

    Three cases, in order:
      - Two or more components already named explicitly ("connect ESP32
        to Relay Module"): use them directly, unchanged from before
        pronoun resolution existed.
      - None named at all ("connect them"): needs a plural pronoun,
        resolved to the last two distinct components mentioned so far
        (see _resolve_pronoun_pair()).
      - Exactly one named ("connect it to Relay Module" / "connect Relay
        Module to it"): needs a singular pronoun, resolved to the most
        recently mentioned component other than the one already named.
        Which one ends up "source" vs "target" follows whichever of the
        name/pronoun appears first in the clause's own text, so "connect
        it to Relay Module" and "connect Relay Module to it" resolve in
        opposite orders, matching how they read.
    """
    if len(found) >= 2:
        return found[0], found[1]

    if not found:
        return _resolve_pronoun_pair(clause, mentioned)

    if not _clause_has_pronoun(clause, _SINGULAR_PRONOUNS):
        return None

    other = next((name for name in reversed(mentioned) if name != found[0]), None)
    if other is None:
        return None

    named_match = re.search(r"\b" + re.escape(found[0]) + r"\b", clause, re.IGNORECASE)
    pronoun_match = re.search(r"\b(?:" + "|".join(_SINGULAR_PRONOUNS) + r")\b", clause, re.IGNORECASE)
    if named_match and pronoun_match and pronoun_match.start() < named_match.start():
        return other, found[0]
    return found[0], other


def _operation_component_names(intent: str, payload: Any) -> List[str]:
    """Return every component name one resolved operation refers to.

    Used by _build_multi_step_plan() to build up the running "mentioned"
    list that later clauses in the same request can resolve a pronoun
    against (see _resolve_pronoun_pair()/_resolve_pronoun_single()).
    """
    if intent in ("connect_components", "disconnect_components"):
        return list(payload)
    if intent == "get_canvas_state":
        return []
    return [payload]


def _clause_operations(clause: str, known_names: List[str], mentioned: List[str]) -> Optional[List[Operation]]:
    """Resolve one clause into its ordered (intent, payload) operation(s).

    `mentioned` is every component name resolved by earlier clauses in
    this same request, in the order they were introduced - consulted only
    as a fallback, when the clause's own text doesn't name enough
    components for its intent (e.g. "connect them" names none; "connect it
    to Relay Module" names only one). It is read-only here: the caller
    (_build_multi_step_plan()) appends this clause's own resolved
    components to it once this function returns successfully.

    Returns None whenever the clause's intent, or the component(s) it
    needs - named explicitly, or resolved via a pronoun - can't be cleanly
    resolved. The caller then abandons the whole multi-step attempt (see
    _build_multi_step_plan()) rather than guess at a partial plan.
    """
    intent = _detect_clause_intent(clause)
    if intent is None:
        return None

    if intent == "get_canvas_state":
        return [(intent, None)]

    found = _find_components_in_text(clause, known_names)

    if intent in ("connect_components", "disconnect_components"):
        pair = _resolve_connect_pair(clause, mentioned, found)
        if pair is None:
            return None
        return [(intent, pair)]

    if not found:
        resolved = _resolve_pronoun_single(clause, mentioned)
        if resolved is None:
            return None
        found = [resolved]
    return [(intent, component) for component in found]


def _plan_for_operation(intent: str, payload: Any) -> Plan:
    """Build one operation's plan steps by reusing the existing single-op builders.

    Looks up _INTENT_PLANNERS (defined further below, alongside the
    single-intent path it also serves) at call time rather than at import
    time, purely so this section doesn't have to be reordered above that
    table's definition.
    """
    if intent == "get_canvas_state":
        return _plan_get_canvas_state()
    if intent == "connect_components":
        return _plan_connect_components(*payload)
    if intent == "disconnect_components":
        return _plan_disconnect_components(*payload)
    return _INTENT_PLANNERS[intent](payload)


def _build_multi_step_plan(user_request: str, knowledge: Dict[str, Any]) -> Optional[Plan]:
    """Detect and build a combined plan when one request describes several operations.

    Returns None (never []) whenever this request should instead go through
    the original single-intent path just below - either because no
    user_request/knowledge was supplied (older callers), because the
    registry has no components to match against, because a clause's intent
    or components (named explicitly, or via a pronoun resolved against
    components mentioned earlier in the same request - see
    _resolve_pronoun_pair()/_resolve_pronoun_single()) can't be cleanly
    resolved, or because the request only ever resolves to a single
    operation. In every one of those cases, create_plan() falls back to
    using llm_output exactly as it did before this phase - this function
    only ever *adds* a new path, it never replaces the fallback's
    behavior.

    Every add_component operation in the resulting plan is placed at its
    own position (_DEFAULT_ADD_X + its index among this request's adds *
    _ADD_X_STEP), so two components added by the same request never land
    on top of each other - unlike the single-op path (_plan_add_component),
    which always places at the one fixed default since it only ever adds
    one component at a time.
    """
    if not user_request:
        return None

    known_names = _known_component_names(knowledge)
    if not known_names:
        return None

    clauses = _split_into_clauses(user_request)
    if not clauses:
        return None

    operations: List[Operation] = []
    mentioned: List[str] = []
    for clause in clauses:
        clause_ops = _clause_operations(clause, known_names, mentioned)
        if clause_ops is None:
            logger.debug(
                "Multi-step detection abandoned: clause %r did not cleanly resolve to an intent+component(s)",
                clause,
            )
            return None
        operations.extend(clause_ops)
        for op_intent, op_payload in clause_ops:
            mentioned.extend(_operation_component_names(op_intent, op_payload))

    if len(operations) <= 1:
        return None

    plan: Plan = []
    add_count = 0
    for intent, payload in operations:
        if intent == "add_component":
            plan.extend(_plan_add_component_at(payload, _DEFAULT_ADD_X + add_count * _ADD_X_STEP, _DEFAULT_ADD_Y))
            add_count += 1
        else:
            plan.extend(_plan_for_operation(intent, payload))

    logger.info(
        "Detected %d operations in one request %r; built combined plan: %s",
        len(operations),
        user_request,
        plan,
    )
    return plan


# Maps a known intent to the function that builds its execution plan.
# To support a new intent, register it here - no other code needs to change.
_INTENT_PLANNERS: Dict[str, Callable[[str], Plan]] = {
    "add_component": _plan_add_component,
    "remove_component": _plan_delete_component,
    "move_component": _plan_move_component,
}

# Maps loosely-phrased or shorthand intents (as an LLM might return them) to
# the canonical intent names used as keys in _INTENT_PLANNERS. Normalization
# happens once, in _normalize_intent(), before any planner lookup - the
# planner functions above never see a raw alias.
#
# Keys are written in the post-normalization form (underscores only, no
# spaces/hyphens/camelCase) since _normalize_intent() always converts a raw
# intent to that form before consulting this table. An entry is needed only
# when the normalized alias differs from the canonical intent it should
# resolve to (e.g. the singular "connect_component" vs. the registered
# plural "connect_components", or a genuinely different phrase like
# "canvas_state" vs. "get_canvas_state") - forms that already normalize to
# their own canonical name (e.g. "add_component") need no entry, since an
# unmatched key simply passes through unchanged.
_INTENT_ALIASES: Dict[str, str] = {
    "add": "add_component",
    "remove": "remove_component",
    "delete": "remove_component",
    "move": "move_component",
    "connect": "connect_components",
    "connect_component": "connect_components",
    "disconnect": "disconnect_components",
    "disconnect_component": "disconnect_components",
    "getcanvasstate": "get_canvas_state",
    "canvas_state": "get_canvas_state",
    "show_canvas": "get_canvas_state",
    "view_canvas": "get_canvas_state",
}


def _normalize_intent(intent: str) -> str:
    """Map a raw intent string to its canonical form.

    Camel-case boundaries are split - and separators (spaces, hyphens)
    collapsed to underscores - *before* lowercasing, so "showCanvas",
    "show canvas", "show-canvas", and "show_canvas" all normalize to the
    same "show_canvas" prior to the alias lookup below. Lowercasing first
    would destroy the capital-letter signal a camel-case split depends on
    (collapsing "showCanvas" into the unsplittable "showcanvas").

    Unrecognized intents - including ones already in canonical form, like
    "add_component" - pass through unchanged. The alias table only rewrites
    known shorthand/synonym forms; it never invents a new canonical intent.
    """
    split_camel_case = _CAMEL_CASE_BOUNDARY.sub("_", intent.strip())
    normalized = re.sub(r"[\s\-]+", "_", split_camel_case).lower()
    return _INTENT_ALIASES.get(normalized, normalized)


# Maps a resolved Frontend Feature Registry name (capability_resolver's
# output) to the canonical intent used to pick a planner below. This is how
# a resolved frontend feature "chooses the tool" instead of the raw intent.
_FRONTEND_FEATURE_TO_INTENT: Dict[str, str] = {
    "Drag Component": "move_component",
    "Move Component": "move_component",
    "Delete Component": "remove_component",
    "Connect Components": "connect_components",
}


# ---------------------------------------------------------------------------
# Multi-intent LLM output: {"actions": [{"intent": ..., ...}, ...]}.
#
# llm.py's own {"actions": [...]} shape (its _MULTI_ACTION_INSTRUCTIONS) is
# already merged into one plan today, but only inside graph/workflow.py's
# planner_node, which extracts "actions" and calls create_plan() once per
# action itself before create_plan() ever runs - so create_plan() in
# isolation (e.g. called directly, outside the graph) has never actually
# understood an {"actions": [...]} llm_output; it would just see no
# top-level "intent" and report an empty plan. The functions below give
# create_plan() that same capability natively, so it's self-sufficient
# without needing a caller to pre-process "actions" for it first.
# ---------------------------------------------------------------------------


def _extract_actions(llm_output: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
    """Return llm_output["actions"] (or its "intents" alias) if non-empty, else None.

    llm.py's analyze() already renames "intents" to "actions" before this
    is ever called through the normal pipeline - the fallback here just
    means create_plan() also handles it correctly for a caller that builds
    llm_output itself and skips analyze() (e.g. a test, or {"intents":
    [{"intent": "add_component", "component": "ESP32"}, ...]} passed
    directly). None covers every case that should fall through to the
    existing text-segmentation/single-intent paths unchanged: neither key
    is present (the legacy shape), present but not a list, or an empty list.
    """
    actions = llm_output.get("actions")
    if isinstance(actions, list) and actions:
        return actions

    intents = llm_output.get("intents")
    if isinstance(intents, list) and intents:
        return intents

    return None


def _plan_multi_intent(actions: List[Dict[str, Any]]) -> Plan:
    """Convert a multi-intent {"actions": [...]} response into one combined plan.

    Loops through `actions` in order and runs each one back through
    create_plan() itself (single-op path only - each action already
    carries its own resolved intent, so this never re-enters
    _build_multi_step_plan()'s text segmentation), concatenating the
    resulting per-action plans so action N's steps always precede action
    N+1's. An action that yields no plan (unrecognized intent, missing
    component, not a JSON object) is logged and skipped rather than
    aborting every other action. Field names ("source"/"target" as well as
    "source_component"/"target_component") are resolved by create_plan()
    itself, not here - see its "source"/"target" alias handling - so every
    caller of create_plan() accepts both spellings identically, including
    graph/workflow.py's own separate per-action create_plan() calls.

    Every add_component action is placed at its own position
    (_DEFAULT_ADD_X + its index among this whole batch's adds * _ADD_X_STEP),
    the same way _build_multi_step_plan() already spaces out several adds
    in one request - otherwise two components added by the same actions
    list would land on top of each other. `add_count` is shared across
    every action in the batch regardless of whether a given add_component
    action names its component via the singular "component" or the plural
    "components" (one or more names) - both are collected into `names`
    before a single call into _plan_for_components() advances the same
    counter, so e.g. one "component": "ESP32" action followed by another
    "components": ["Relay Module", "DHT11 Sensor"] action still places all
    three at distinct positions instead of the second action's names
    restarting at index 0 and overlapping the first.
    """
    plan: Plan = []
    add_count = 0
    for index, action in enumerate(actions):
        if not isinstance(action, dict):
            logger.warning("Multi-intent action %d is not a JSON object, skipping: %r", index, action)
            continue

        intent = _normalize_intent(action.get("intent", ""))
        component = action.get("component", "")
        components = action.get("components")
        names = components if isinstance(components, list) and components else ([component] if component else None)

        if intent == "add_component" and names:
            action_plan = _plan_for_components(intent, names, start_index=add_count)
            add_count += len(names)
        else:
            action_plan = create_plan(action)

        if not action_plan:
            logger.warning("Multi-intent action %d produced no plan, skipping: %s", index, action)
            continue

        plan.extend(action_plan)

    return plan


# ---------------------------------------------------------------------------
# Rule-based component extraction: an LLM-free backstop for a pure "add
# these components" request. Guarantees every named component reaches the
# plan, regardless of how reliably the LLM itself enumerated them.
# ---------------------------------------------------------------------------

_ADD_ACTION_WORDS = re.compile(
    r"\b(?:add|adds|added|adding|"
    r"create|creates|created|creating|"
    r"place|places|placed|placing|"
    r"insert|inserts|inserted|inserting|"
    r"attach|attaches|attached|attaching|"
    r"put|puts|putting|"
    r"need|needs|needed)\b",
    re.IGNORECASE,
)
_AND_SEPARATOR = re.compile(r"\band\b", re.IGNORECASE)
_TRAILING_PUNCTUATION = re.compile(r"[.,!?]+$")

# Recognizes "connect"/"link"/"attach"/"wire" and their common inflections
# (matters for e.g. "connected", which _detect_add_and_connect() below and
# is_pure_add_request() both need to recognize as the same verb as
# "connect" - a plain `\bconnect\b` never matches "connected").
_CONNECT_ACTION_WORDS = re.compile(
    r"\b(?:connect|connects|connected|connecting|"
    r"link|links|linked|linking|"
    r"attach|attaches|attached|attaching|"
    r"wire|wires|wired|wiring)\b",
    re.IGNORECASE,
)

# Any of these appearing alongside an add word means the request mixes in a
# different operation (e.g. "Add ESP32 and connect it to Relay Module") -
# extract_components()'s plain delimiter splitting has no notion of intent,
# so a clause like "connect it to Relay Module" would otherwise be sliced
# up and misread as more component names. Inflected forms matter here too:
# without "connected"/"connecting" etc., is_pure_add_request() would
# wrongly call a request like "create ESP32 connected to the Accelerometer"
# a pure add (a plain `\bconnect\b` doesn't match "connected").
_OTHER_OPERATION_WORDS = re.compile(
    r"\b(?:remove|removes|removed|removing|"
    r"delete|deletes|deleted|deleting|"
    r"move|moves|moved|moving|"
    r"connect|connects|connected|connecting|"
    r"disconnect|disconnects|disconnected|disconnecting|"
    r"show|shows|showing|"
    r"display|displays|displayed|displaying|"
    r"view|views|viewed|viewing)\b",
    re.IGNORECASE,
)


def extract_components(user_input: str) -> List[str]:
    """Rule-based, LLM-free extraction of every component name in `user_input`.

    Lowercases the text, treats "and" the same as "," (both just separate
    one component name from the next), strips add-type action words, then
    splits on "," and cleans each piece. No catalog lookup and no LLM call -
    a request naming N components always yields N names.
    """
    if not user_input:
        return []

    text = _AND_SEPARATOR.sub(",", user_input.lower())
    text = _ADD_ACTION_WORDS.sub("", text)

    components = []
    for piece in text.split(","):
        name = _TRAILING_PUNCTUATION.sub("", piece).strip()
        if name:
            components.append(name)

    print("DEBUG Components:", components)
    return components


def is_pure_add_request(user_request: str) -> bool:
    """Whether `user_request` only ever asks to add one or more components.

    True only when an add-type word is present and no other operation verb
    is - this keeps a genuinely mixed request on the existing LLM/actions
    path (see create_plan()) instead of being misparsed by
    extract_components()'s plain delimiter splitting. Public because
    graph/workflow.py's llm_node also calls this, to skip the LLM call
    entirely for a pure add request (rule-based extraction *before* the
    LLM, not just as a backstop after it).
    """
    if not user_request:
        return False
    return bool(_ADD_ACTION_WORDS.search(user_request)) and not _OTHER_OPERATION_WORDS.search(user_request)


def _detect_add_and_connect(user_request: str, knowledge: Optional[Dict[str, Any]]) -> Optional["Plan"]:
    """Deterministically catch any phrasing of "connect A to B" - optionally combined with adding A/B first.

    This is the generalized fix for a request like "create ESP32
    connected to the Accelerometer" or "add ESP32 and connect it to the
    accelerometer" being misread as a plain single add_component: it
    doesn't depend on the exact verb ("connect"/"link"/"attach"/"wire"/
    "connected to"/"make ... connect to ..." all qualify - see
    _CONNECT_ACTION_WORDS) or on whether the request also asks to add the
    components ("add"/"create"/"place"/... - see _ADD_ACTION_WORDS) versus
    assuming they already exist ("connect ESP32 to accelerometer" alone).

    The only requirement is structural, not textual: a connect-type verb
    appears anywhere in the request, and exactly two distinct catalog
    components are named in it, in the order named (so "connect A to B"
    and "connect B to A" resolve to opposite source/target - see
    _find_components_in_text()). Zero, one, or three-or-more components
    with a connect verb is genuinely ambiguous and is left to the existing
    LLM/multi-step-segmentation path rather than guessed at here.

    Returns None whenever this doesn't confidently apply, so create_plan()
    falls through to its existing behavior unchanged.
    """
    if not user_request or not _CONNECT_ACTION_WORDS.search(user_request):
        return None

    known_names = _known_component_names(knowledge)
    distinct = list(dict.fromkeys(_find_components_in_text(user_request, known_names)))
    if len(distinct) != 2:
        return None

    source, target = distinct
    plan: Plan = []

    if _ADD_ACTION_WORDS.search(user_request):
        plan.extend(_plan_for_components("add_component", [source, target]))

    plan.extend(_plan_connect_components(source, target))

    logger.info(
        "Detected a connect relationship in %r (source=%r, target=%r)%s: %s",
        user_request,
        source,
        target,
        " with add steps" if _ADD_ACTION_WORDS.search(user_request) else "",
        plan,
    )
    return plan


def _resolve_component_name(raw_name: str, known_names: List[str]) -> str:
    """Map a raw name from extract_components() to its exact catalog name, when possible.

    Tries an exact case-insensitive match first (covers "esp32" -> "ESP32"),
    then a substring match in either direction (covers shorthand like
    "relay" -> "Relay Module"). Before giving up, checks whether any
    letters+digits token *within* the phrase (e.g. "th11" inside "the th11
    sensor") confidently resolves to a catalog component's code via
    command_normalizer's shared, digit-aware fuzzy matcher - the same rule
    that fixes "TH11" -> "DHT11" for voice input applies here too, so a
    garbled code doesn't get title-cased into an invented name like "The
    Th11 Sensor". Only once none of that applies does this fall back to
    `raw_name` itself, title-cased - extract_components() still never drops
    the component, it just can't always give it a canonical name.
    """
    lowered = raw_name.lower()

    for name in known_names:
        if name.lower() == lowered:
            return name

    for name in known_names:
        if lowered in name.lower() or name.lower() in lowered:
            return name

    from command_normalizer import _catalog_codes, _fuzzy_match_alnum_code

    codes = _catalog_codes(known_names)
    for token in re.findall(r"[A-Za-z]+\d+", raw_name):
        resolved_code = _fuzzy_match_alnum_code(token, codes)
        if resolved_code is None:
            continue
        for name in known_names:
            if resolved_code.lower() in name.lower():
                return name

    return raw_name.title()


def extract_and_resolve_components(user_request: str, knowledge: Dict[str, Any] = None) -> List[str]:
    """extract_components() + catalog-name resolution, in one call.

    The single entry point for turning a raw "add X, Y and Z" request into
    a clean, catalog-resolved component list - used both by
    graph/workflow.py's llm_node (to skip the LLM call for a pure add
    request) and by create_plan() below, so the two can never drift apart.
    """
    raw_components = extract_components(user_request)
    known_names = _known_component_names(knowledge)
    return [_resolve_component_name(name, known_names) for name in raw_components]


def _plan_for_components(intent: str, components: List[str], start_index: int = 0) -> Plan:
    """Build a combined plan by running each name in `components` through intent's single-op planner.

    Backs create_plan()'s "components" (plural) list field - the
    structured-output counterpart to _build_multi_step_plan()'s text
    segmentation, for when the LLM itself names several components for one
    "add_component"/"remove_component"/"move_component" operation (e.g.
    {"intent": "add_component", "components": ["ESP32", "DHT11 Sensor"]}).
    No per-component planning logic is duplicated: add_component reuses
    _plan_add_component_at with the same spacing _build_multi_step_plan()
    already uses so several added components never land on top of each
    other; every other intent reuses its existing single-op planner from
    _INTENT_PLANNERS unchanged.

    `start_index` offsets add_component's spacing so _plan_multi_intent()
    can give every add across a whole multi-intent batch - whether from
    this action's "components" list or another action's singular
    "component" - its own position, instead of every call restarting at
    index 0 and colliding with an add already placed by a different
    action.
    """
    if intent == "add_component":
        plan: Plan = []
        for offset, component in enumerate(components):
            index = start_index + offset
            plan.extend(_plan_add_component_at(component, _DEFAULT_ADD_X + index * _ADD_X_STEP, _DEFAULT_ADD_Y))
        return plan

    planner = _INTENT_PLANNERS.get(intent)
    if planner is None:
        return []

    plan = []
    for component in components:
        plan.extend(planner(component))
    return plan


def create_plan(
    llm_output: dict,
    capability: Dict[str, Any] = None,
    user_request: str = "",
    knowledge: Dict[str, Any] = None,
) -> Plan:
    """Convert structured LLM output into an execution plan.

    Args:
        llm_output: dict with "intent" and "component" keys, e.g.
            {"intent": "add_component", "component": "ESP32"}. "intent" may
            be a raw alias (e.g. "add", "delete") - it is normalized before
            the planner lookup. When one "add_component"/"remove_component"/
            "move_component" operation names more than one component, use
            "components" (a list) instead of "component" - e.g.
            {"intent": "add_component", "components": ["ESP32", "DHT11
            Sensor"]} - and every named component gets its own step (see
            _plan_for_components()); "component" is still honored unchanged
            when "components" is absent or empty. May instead be the
            multi-intent shape {"actions": [{"intent": ..., ...}, ...]} -
            see _plan_multi_intent() - in which case every other argument is
            ignored and each action (itself optionally using "components")
            is planned and concatenated in order.
        capability: optional resolved capability, as returned by
            capability_resolver.resolve_capability():
            {"component": ..., "frontend_feature": ..., "backend_api": ...}.
            When a "backend_api" is resolved, this produces a backend
            execution step directly (never executed here). Otherwise, a
            resolved "frontend_feature" picks the tool in place of the raw
            intent, and a resolved "component" takes precedence over
            llm_output's component.
        user_request: optional raw request text (Phase 3.2). When given
            together with `knowledge`, it is checked first for multiple
            operations (e.g. "Add ESP32 and Relay Module", "Move ESP32 then
            show canvas") - see _build_multi_step_plan(). Omitting it (the
            default) always skips straight to the single-intent path below,
            so existing callers are unaffected. Never consulted when
            llm_output is already the multi-intent {"actions": [...]}
            shape - the LLM has already decomposed the request in that
            case, so there's nothing left for text segmentation to find.
        knowledge: optional {"components": [...], ...} from
            KnowledgeLoader.load_all(), used only to recognize component
            names while segmenting `user_request` into multiple operations.

    Returns:
        A list of steps, each a dict of {"tool": str, "input": ...}.
        Returns an empty list if the intent is unrecognized, or if the
        input it needs is missing (a component for most intents; both
        source_component and target_component for connect/disconnect;
        nothing at all for get_canvas_state) - it never raises for bad
        input, and it never calls the LLM or executes a tool itself.
    """
    if isinstance(llm_output, list):
        # llm.py's analyze() already normalizes a bare JSON array into
        # {"actions": [...]} before returning, so this only matters for a
        # caller that hands create_plan() a raw actions list directly.
        llm_output = {"actions": llm_output}

    llm_output = llm_output or {}

    actions = _extract_actions(llm_output)
    if actions is not None:
        plan = _plan_multi_intent(actions)
        logger.info("Created combined plan for %d multi-intent action(s): %s", len(actions), plan)
        return plan

    connect_plan = _detect_add_and_connect(user_request, knowledge)
    if connect_plan is not None:
        return connect_plan

    if is_pure_add_request(user_request):
        # Rule-based extraction, not the LLM's own "component"/"components"
        # fields: guarantees every component named in a plain "add X, Y and
        # Z" request reaches the plan, ahead of _build_multi_step_plan()
        # below (whose catalog-exact-name matching can otherwise miss a
        # shorthand mention and silently drop it). graph/workflow.py's
        # llm_node already handles the common case before the LLM is even
        # called; this is a second check so create_plan() is still correct
        # on its own (e.g. called directly, or if llm_node's own check was
        # bypassed).
        components = extract_and_resolve_components(user_request, knowledge)
        if components:
            plan = _plan_for_components("add_component", components)
            logger.info("Rule-based extraction found %d component(s) in %r: %s", len(components), user_request, plan)
            return plan

    multi_step_plan = _build_multi_step_plan(user_request, knowledge)
    if multi_step_plan is not None:
        return multi_step_plan

    capability = capability or {}

    raw_intent = llm_output.get("intent", "")
    component = capability.get("component") or llm_output.get("component", "")
    # "source"/"target" accepted as aliases for "source_component"/
    # "target_component" - this is what makes a single multi-intent action
    # shaped either way (see _normalize_action_fields()'s docstring) work
    # not just through _plan_multi_intent() below, but also through
    # graph/workflow.py's own separate per-action create_plan() calls
    # (planner_node's _create_plan_for_action()), which hands a single
    # action dict here directly with no normalization of its own.
    source_component = llm_output.get("source_component") or llm_output.get("source", "")
    target_component = llm_output.get("target_component") or llm_output.get("target", "")

    backend_api = capability.get("backend_api")
    if backend_api:
        plan = _plan_backend_operation(backend_api, component)
        logger.info("Prepared backend execution step for backend_api=%r: %s", backend_api, plan)
        return plan

    frontend_feature = capability.get("frontend_feature")
    if frontend_feature in _FRONTEND_FEATURE_TO_INTENT:
        intent = _FRONTEND_FEATURE_TO_INTENT[frontend_feature]
        logger.debug("Frontend feature %r resolved to intent %r", frontend_feature, intent)
    else:
        intent = _normalize_intent(raw_intent)

    logger.debug(
        "Creating plan for raw_intent=%r resolved_intent=%r component=%r "
        "source_component=%r target_component=%r",
        raw_intent,
        intent,
        component,
        source_component,
        target_component,
    )

    if not intent:
        logger.warning("No intent provided in LLM output: %s", llm_output)
        return []

    if intent == "get_canvas_state":
        plan = _plan_get_canvas_state()
        logger.info("Created plan for intent %r: %s", intent, plan)
        return plan

    if intent in ("connect_components", "disconnect_components"):
        if not source_component or not target_component:
            logger.warning(
                "Both source_component and target_component are required for intent %r "
                "(got source_component=%r, target_component=%r)",
                intent,
                source_component,
                target_component,
            )
            return []

        plan_builder = _plan_connect_components if intent == "connect_components" else _plan_disconnect_components
        plan = plan_builder(source_component, target_component)
        logger.info("Created plan for intent %r: %s", intent, plan)
        return plan

    components = llm_output.get("components")
    if isinstance(components, list) and components and intent in _INTENT_PLANNERS:
        plan = _plan_for_components(intent, components)
        logger.info("Created plan for intent %r over %d components: %s", intent, len(components), plan)
        return plan

    if not component:
        logger.warning("No component provided for intent %r", intent)
        return []

    planner = _INTENT_PLANNERS.get(intent)
    if planner is None:
        logger.warning("No planner registered for intent %r", intent)
        return []

    plan = planner(component)
    logger.info("Created plan for intent %r: %s", intent, plan)
    return plan
