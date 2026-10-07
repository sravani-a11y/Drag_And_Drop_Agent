"""Builds the structured prompt sent to the LLM.

Pure string construction from known knowledge (components, features, APIs)
and the user's request. Does not call the LLM, tools, or backend APIs.

Phase 3.3 - Natural Language Understanding:
    The JSON schema and canonical intent vocabulary below are unchanged -
    planner.py still receives the exact same fields it always has. What
    changed is how hard this prompt works to get the LLM to fill them in
    correctly for natural, indirect phrasing ("Could you place a relay
    module?", "Delete everything.", "Connect these two.") instead of only
    literal keyword-shaped requests ("Add ESP32."). Three additions do
    this:
      1. _SYSTEM_INSTRUCTIONS now explicitly tells the model to reason
         about meaning rather than pattern-match words, and to resolve
         indirect component references (pronouns, articles, "both"/"them")
         using the live canvas context and component registry rather than
         inventing or echoing back vague text.
      2. _build_output_format_section() adds an explicit reference-
         resolution rule and an explicit "don't guess" rule alongside the
         existing intent-mapping table.
      3. _build_examples_section() gives the model several worked
         request -> JSON examples spanning polite/indirect phrasing,
         pronoun resolution, and honest abstention - concrete
         demonstrations a small local model benefits from far more than
         another paragraph of instructions.
    None of this is rule-based keyword matching in code: every mapping
    happens inside the LLM's own reasoning over this prompt. This module
    still only builds a string; it never inspects the user's request
    itself or decides an intent.
"""

import json

_SYSTEM_INSTRUCTIONS = """You are an Enterprise AI Agent for InnoIDE, a hardware/embedded design canvas.

Engineers describe what they want the way they would to a human teammate: politely, indirectly, with pronouns and shorthand - not as rigid commands. "Could you place a relay module?" and "Add ESP32" both mean the same thing, even though only one contains the word "add". Your job is to read past the specific wording, identify what the engineer actually wants done, and return exactly that as structured JSON.

Reason about meaning, not literal keywords - do not require a specific trigger word to appear before recognizing an intent.

When a request names a component indirectly ("the relay", "it", "these two", "them", "both") or not at all ("delete everything", "connect these two"), resolve the reference using the current canvas state and the available component list provided below, and return the component's exact, real name - never a pronoun, article, or description. Never invent a component that isn't listed or already on the canvas; if a reference can't be resolved to one specific real component, leave that field empty rather than guessing."""


_OUTPUT_SCHEMA = {
    "intent": "",
    "component": "",
    "components": [],
    "source_component": "",
    "target_component": "",
    "confidence": "",
    "reasoning": ""
}

# The fixed, canonical intent vocabulary. The LLM must choose exactly one
# of these for "intent" - never a free-form word/phrase/synonym - so that
# every request maps to a value the Planner already recognizes.
_VALID_INTENTS = (
    "add_component",
    "remove_component",
    "move_component",
    "connect_components",
    "disconnect_components",
    "get_canvas_state",
)

# Few-shot examples (Phase 3.3): each pairs a natural-language request with
# the exact JSON this prompt asks for, covering polite/indirect phrasing,
# pronoun/reference resolution against canvas context, and honest
# abstention when a reference can't be resolved to one real component.
# These are illustrative only - the model's actual answer must be about the
# real "User request" given later in the prompt, not about these.
_EXAMPLES = [
    {
        "request": "Please add an ESP32.",
        "response": {
            "intent": "add_component", "component": "", "components": ["ESP32"],
            "source_component": "", "target_component": "",
            "confidence": "high", "reasoning": "Polite phrasing for adding a component already in the catalog.",
        },
    },
    {
        "request": "Could you place a relay module?",
        "response": {
            "intent": "add_component", "component": "", "components": ["Relay Module"],
            "source_component": "", "target_component": "",
            "confidence": "high", "reasoning": "\"Place\" describes adding a component, not a literal keyword match.",
        },
    },
    {
        "request": "Add ESP32 and DHT11 Sensor.",
        "response": {
            "intent": "add_component", "component": "", "components": ["ESP32", "DHT11 Sensor"],
            "source_component": "", "target_component": "",
            "confidence": "high", "reasoning": "Two components are named for the same add operation, so both go in \"components\".",
        },
    },
    {
        "request": "add esp32, relay, led",
        "response": {
            "intent": "add_component", "component": "", "components": ["ESP32", "Relay Module", "LED"],
            "source_component": "", "target_component": "",
            "confidence": "high",
            "reasoning": "A comma-separated list (no \"and\" needed) naming three components; each shorthand mention (\"esp32\", \"relay\", \"led\") is resolved to its exact catalog name, and none are dropped.",
        },
    },
    {
        "request": "I need an ESP32 connected to a relay.",
        "response": {
            "intent": "connect_components", "component": "", "components": [],
            "source_component": "ESP32", "target_component": "Relay Module",
            "confidence": "high", "reasoning": "The engineer wants a wiring relationship between two named components.",
        },
    },
    {
        "request": "Remove the relay.",
        "response": {
            "intent": "remove_component", "component": "", "components": ["Relay Module"],
            "source_component": "", "target_component": "",
            "confidence": "high", "reasoning": "\"The relay\" resolves to the Relay Module already on the canvas.",
        },
    },
    {
        "request": "Delete everything.",
        "response": {
            "intent": "remove_component", "component": "", "components": [],
            "source_component": "", "target_component": "",
            "confidence": "low",
            "reasoning": "Refers to every component collectively rather than one named component, so none is guessed.",
        },
    },
    {
        "request": "Move ESP32 near the relay.",
        "response": {
            "intent": "move_component", "component": "", "components": ["ESP32"],
            "source_component": "", "target_component": "",
            "confidence": "high", "reasoning": "A move request; the relay only describes the intended direction, not a second target.",
        },
    },
    {
        "request": "Connect these two.",
        "response": {
            "intent": "connect_components", "component": "", "components": [],
            "source_component": "ESP32", "target_component": "Relay Module",
            "confidence": "medium",
            "reasoning": "\"These two\" resolves to the only two components shown in the current canvas state.",
        },
    },
    {
        "request": "Disconnect them.",
        "response": {
            "intent": "disconnect_components", "component": "", "components": [],
            "source_component": "ESP32", "target_component": "Relay Module",
            "confidence": "medium",
            "reasoning": "\"Them\" resolves to the pair of components shown as connected in the current canvas state.",
        },
    },
    {
        "request": "Show me the canvas.",
        "response": {
            "intent": "get_canvas_state", "component": "", "components": [],
            "source_component": "", "target_component": "",
            "confidence": "high", "reasoning": "A request to view canvas state, regardless of exact wording.",
        },
    },
    {
        "request": "What's currently on the canvas?",
        "response": {
            "intent": "get_canvas_state", "component": "", "components": [],
            "source_component": "", "target_component": "",
            "confidence": "high", "reasoning": "A question about canvas contents is still a get_canvas_state request.",
        },
    },
]


def _format_items(items: list) -> str:
    if not items:
        return "(none available)"

    lines = []
    for item in items:
        if isinstance(item, dict) and "name" in item:
            description = item.get("description")
            if description:
                lines.append(f"- {item['name']}: {description}")
            else:
                lines.append(f"- {item['name']}")
        else:
            lines.append(f"- {json.dumps(item)}")
    return "\n".join(lines)


def _build_system_section() -> str:
    return _SYSTEM_INSTRUCTIONS


def _build_components_section(components: list) -> str:
    return "Available components:\n" + _format_items(components)


def _build_features_section(features: list) -> str:
    return "Available frontend features:\n" + _format_items(features)


def _build_apis_section(apis: list) -> str:
    return "Available backend APIs:\n" + _format_items(apis)


def _build_user_request_section(user_request: str) -> str:
    return "User request:\n" + user_request


def _build_output_format_section() -> str:
    valid_intents = ", ".join(f'"{intent}"' for intent in _VALID_INTENTS)
    return (
        "Respond with ONLY valid JSON — no explanation, no markdown fences, "
        "and no text before or after it. Use exactly this format:\n"
        + json.dumps(_OUTPUT_SCHEMA, indent=4)
        + "\n\nThe \"intent\" field MUST be exactly one of these values - "
        "never any other word, phrase, or synonym: " + valid_intents + ".\n"
        "Map the underlying meaning of the request to exactly one of them "
        "(the exact wording varies far more than the underlying meaning "
        "does - judge by the latter):\n"
        '- Adding a component to the canvas -> "add_component"\n'
        '- Deleting or removing a component -> "remove_component"\n'
        '- Moving or repositioning a component -> "move_component"\n'
        '- Connecting or wiring two components -> "connect_components"\n'
        '- Disconnecting or unwiring two components -> "disconnect_components"\n'
        '- Showing, displaying, or viewing the canvas (e.g. "show canvas", '
        '"display canvas", "view canvas", "what\'s on the canvas") -> '
        '"get_canvas_state"\n\n'
        "For \"connect_components\" or \"disconnect_components\" requests "
        "involving two components, leave \"component\" and \"components\" "
        "empty and instead fill in \"source_component\" (the first "
        "component named or implied) and \"target_component\" (the "
        "second). For \"get_canvas_state\", leave every component field "
        "empty - it needs none.\n\n"
        "Components are always a list: for \"add_component\", "
        "\"remove_component\", or \"move_component\", always put every "
        "named component, in the order named, under \"components\" - even "
        "when there is only one (e.g. \"components\": [\"ESP32\"]) - and "
        "always leave \"component\" empty. Never answer with just "
        "\"component\" by itself for these three intents, no matter how "
        "many components the request names.\n\n"
        "Reference resolution: when a component is referred to indirectly "
        "(\"it\", \"the relay\", \"these two\", \"them\", \"both\") rather "
        "than named outright, resolve it using the current canvas state and "
        "the available components above, and write in its exact real name. "
        "Never write back a pronoun, article, or descriptive phrase as a "
        "component value.\n\n"
        "Honesty over guessing: if a request's component reference can't be "
        "resolved this way to one specific, real component (e.g. it refers "
        "to every component collectively, or to something not present "
        "anywhere), leave that field as an empty string rather than "
        "inventing or guessing a name."
    )


def _build_examples_section() -> str:
    lines = [
        "Examples (illustrative only - answer about the real \"User request\" "
        "given further below, not about these):",
    ]
    for example in _EXAMPLES:
        lines.append("")
        lines.append(f'Request: "{example["request"]}"')
        lines.append("Response:")
        lines.append(json.dumps(example["response"], indent=4))
    return "\n".join(lines)


def build_prompt(knowledge: dict, user_request: str) -> str:
    components = knowledge.get("components", [])
    features = knowledge.get("features", [])
    apis = knowledge.get("apis", [])

    # Ordered so the model sees, in this order: who it is and how it should
    # think, the exact schema/rules it must follow, worked examples of
    # applying those rules, the registries it may reference by name, and
    # finally the real request - kept last so it's the most recent thing
    # the model reads before it has to answer.
    sections = [
        _build_system_section(),
        _build_output_format_section(),
        _build_examples_section(),
        _build_components_section(components),
        _build_features_section(features),
        _build_apis_section(apis),
        _build_user_request_section(user_request),
    ]

    return "\n\n".join(sections)
