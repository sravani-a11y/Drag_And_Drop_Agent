"""Speech-command normalization: the layer between Speech-to-Text and LangGraph.

    Audio -> Speech-to-Text -> Command Normalization -> LangGraph

Speech recognition regularly mangles hardware part names - "ESP32" comes
back as "ESP 32", "ESP thirty two", or even a wholesale mishearing like
"NSP". None of that is something the Planner/Executor/Verifier/MCP Bridge
should have to deal with, and none of it is safe to fix by guessing - so
this module only ever corrects a token into an *actual, registered*
component name (via knowledge_loader.KnowledgeLoader, the same catalog
prompt_builder.py and planner.py already use), and only when it's confident
the correction is unambiguous. It never touches LangGraph, the Planner, the
Executor, the Verifier, or the MCP Bridge - normalize_command() just
returns a corrected string for whatever already calls run_workflow().

Before any of that, two safety checks run first, since neither is safe to
paper over with correction:
    0a. Severe-repetition detection: if the transcript is dominated by one
        word repeated many times (a known STT/Whisper hallucination
        failure mode - e.g. "SPI, SPI, SPI, SPI, SPI..."), normalize_command()
        refuses to normalize it at all and returns None, so the caller can
        stop before ever reaching the Planner instead of "correcting" a
        garbage transcript into a differently-garbage command. This is a
        general statistical check (a word making up too large a fraction
        of all words), not a check for any specific corrupted phrase.
    0b. Known technical terms (see _KNOWN_TECHNICAL_TERMS) are protected
        from fuzzy correction outright - "SPI", "UART", "I2C", "WiFi",
        etc. are valid vocabulary in their own right and must never be
        "corrected" into a different component just because they happen to
        share letters with one (e.g. "SPI" superficially resembles "ESP"
        by plain character overlap).

Then, four passes, each strictly narrower/safer than a blind find-and-replace:
    1. Spelled-letter collapsing: "E S P" (each letter spoken/transcribed
       separately) collapses to "ESP" only when the concatenated letters
       match a real catalog code prefix - never an arbitrary run of
       single-letter words.
    2. Catalog-anchored number expansion: "<code letters> <number>" is
       only ever rewritten when <code letters> matches a real catalog
       component's alphanumeric-code prefix (e.g. "ESP" from "ESP32") -
       never a bare number word anywhere else in the sentence ("connect
       two components" is left alone).
    3. Exact, case-insensitive catalog-name matching: fixes casing only
       (e.g. "accelerometer" -> "Accelerometer") when the word(s) already
       spell out a real component name.
    4. Fuzzy single-token correction: only for a token that (a) isn't a
       command/structure word (see _STRUCTURE_WORDS) or a protected
       technical term (see _KNOWN_TECHNICAL_TERMS), (b) doesn't already
       match a catalog name, and (c) is unambiguously close to exactly one
       catalog component's short code - either a purely alphabetic token
       (e.g. "NSP" -> "ESP32") or a letters+digits token where the digits
       are used as corroborating evidence (e.g. "TH11" -> "DHT11": a
       dropped leading consonant with the number intact is a far more
       plausible STT error than a coincidence). Never compares against a
       full, multi-word descriptive name (e.g. "Relay Module", "Temperature
       Sensor"), since a generic English word like "relay" or "sensor"
       would otherwise false-positive against those.

Safety net: if a letters+digits token looks like an attempted technical
code (e.g. "TH55") but nothing in the catalog is confidently close to it,
normalize_command() raises UnresolvedComponentError instead of guessing -
this is what stops "TH11 sensor" from ever becoming an invented "The Th11
Sensor" component. The caller is expected to stop before ever reaching the
Planner (see api/server.py's voice endpoint) and ask the user to repeat,
exactly like the severe-repetition case above.
"""

import difflib
import logging
import re
from collections import Counter
from typing import Dict, List, Optional, Tuple

from knowledge_loader import KnowledgeLoader

logger = logging.getLogger(__name__)

_ONES = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
}
_TEENS = {
    "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13", "fourteen": "14",
    "fifteen": "15", "sixteen": "16", "seventeen": "17", "eighteen": "18", "nineteen": "19",
}
_TENS = {
    "twenty": "20", "thirty": "30", "forty": "40", "fifty": "50",
    "sixty": "60", "seventy": "70", "eighty": "80", "ninety": "90",
}

# Command/structure words a garbled component name should never be
# confused with - the fuzzy pass (step 3) skips any token in this set
# before it ever considers a catalog match for it.
_STRUCTURE_WORDS = frozenset(
    {
        "create", "add", "place", "insert", "attach", "make",
        "connect", "connected", "disconnect", "disconnected",
        "remove", "delete", "move", "moved", "show", "display", "view",
        "the", "a", "an", "to", "and", "with", "from", "on", "at", "of",
        "is", "it", "this", "that", "these", "those", "then", "please",
        "canvas", "component", "components",
    }
)

# Below this fuzzy-match score, or without a clear margin over the next
# best catalog code, a token is left alone rather than guessed at.
_FUZZY_MATCH_THRESHOLD = 0.6
_FUZZY_MATCH_MARGIN = 0.15

# Confidence floor for a letters+digits token (e.g. "TH11"). Lower than the
# pure-letters threshold above because an exact digit match is treated as
# strong independent corroborating evidence on top of letter similarity -
# see _fuzzy_match_alnum_code().
_ALNUM_FUZZY_THRESHOLD = 0.7


class UnresolvedComponentError(Exception):
    """Raised by normalize_command() when a technical-code-shaped token can't be confidently resolved.

    E.g. "TH55" - letters+digits, clearly meant to name *some* component -
    where nothing in the catalog is a confident match. The caller must
    never invent a component for this; it should stop before the Planner
    and ask the user to repeat, exactly like a severely repetitive
    transcript (see _is_severely_repetitive()).
    """

    def __init__(self, unresolved_terms: List[str]):
        self.unresolved_terms = unresolved_terms
        super().__init__(f"Could not confidently resolve: {unresolved_terms!r}")

# Real, valid domain vocabulary that must never be fuzzy-corrected into a
# component name, no matter how similar the characters look to a fuzzy
# matcher ("SPI" superficially resembles "ESP" - shares "s" and "p" in the
# same order - but they are unrelated terms). Checked *before* fuzzy
# matching is even attempted (see _fuzzy_correct_remaining_tokens()), so
# exact known terms always take precedence over any fuzzy guess. Lowercase,
# single-token entries only - multi-word terms (e.g. "relay module") are
# already handled safely because the fuzzy pass only ever compares against
# short alphanumeric *codes* (see _catalog_codes()/_fuzzy_match_code()),
# never a full descriptive name, so a generic word like "relay" was never
# at risk in the first place.
_KNOWN_TECHNICAL_TERMS = frozenset(
    {
        "esp32", "stm32", "dht11", "accelerometer", "gyroscope",
        "led", "uart", "i2c", "spi", "wifi", "bluetooth",
    }
)

# Repetition-corruption detection (a known STT/Whisper hallucination
# failure mode - see module docstring). Both conditions must hold before a
# transcript is rejected: the repeated word must occur at least this many
# times, *and* make up at least this fraction of all words - so a
# legitimate short repeat (e.g. two identical numbers in "move ESP32 to
# 300 300") never trips it, but "SPI, SPI, SPI, SPI, SPI, SPI, SPI" does.
_MIN_WORDS_FOR_REPETITION_CHECK = 5
_REPETITION_COUNT_THRESHOLD = 4
_REPETITION_RATIO_THRESHOLD = 0.4


def _is_severely_repetitive(transcript: str) -> bool:
    """Whether `transcript` is dominated by one word repeated an implausible number of times.

    A general statistical check, not a check for any specific corrupted
    phrase - it would catch "SPI, SPI, SPI, ..." exactly the same way it
    would catch any other word stuck in a repetition loop, whatever that
    word happens to be.
    """
    words = [w for w in re.split(r"[^A-Za-z0-9]+", transcript) if w]
    if len(words) < _MIN_WORDS_FOR_REPETITION_CHECK:
        return False

    _, most_common_count = Counter(w.lower() for w in words).most_common(1)[0]
    return most_common_count >= _REPETITION_COUNT_THRESHOLD and (most_common_count / len(words)) >= _REPETITION_RATIO_THRESHOLD


def _known_component_names(knowledge: Optional[Dict]) -> List[str]:
    return [entry.get("name", "") for entry in (knowledge or {}).get("components", []) if entry.get("name")]


def _catalog_codes(known_names: List[str]) -> Dict[str, Tuple[str, str]]:
    """{lowercased letters: (canonical letters, digits)} for every catalog name shaped like '<letters><digits>...'.

    E.g. "ESP32" -> {"esp": ("ESP", "32")}, "DHT11 Sensor" -> {"dht": ("DHT", "11")}.
    Deliberately only the short alphanumeric *code* portion, never the full
    (possibly multi-word) descriptive name - that's what keeps both the
    number-expansion pass and the fuzzy pass from matching a generic word
    like "relay" or "sensor" against "Relay Module"/"Temperature Sensor".
    """
    codes: Dict[str, Tuple[str, str]] = {}
    for name in known_names:
        match = re.match(r"([A-Za-z]+)(\d+)", name)
        if match:
            codes[match.group(1).lower()] = (match.group(1), match.group(2))
    return codes


def _number_phrase_to_digits(phrase: str) -> str:
    """Convert '32', 'thirty two', or 'eleven' to its digit string."""
    phrase = phrase.strip().lower()
    if phrase.isdigit():
        return phrase

    parts = re.split(r"[\s-]+", phrase)
    if len(parts) == 2 and parts[0] in _TENS and parts[1] in _ONES:
        return str(int(_TENS[parts[0]]) + int(_ONES[parts[1]]))
    if phrase in _TENS:
        return _TENS[phrase]
    if phrase in _TEENS:
        return _TEENS[phrase]
    if phrase in _ONES:
        return _ONES[phrase]
    return phrase


def _collapse_spelled_letters(text: str, codes: Dict[str, Tuple[str, str]]) -> str:
    """Collapse a spelled-out acronym ("E S P") into its catalog form ("ESP").

    Only ever collapses a run of single-letter words whose concatenation
    matches a real catalog code's letters - an arbitrary run of unrelated
    single letters ("a b c") is left alone, since the concatenation simply
    won't be a key in `codes`. Runs longest-first so "E S P" collapses as
    one 3-letter unit rather than leaving a stray "P" behind from a
    shorter, also-matching 2-letter attempt.
    """
    if not codes:
        return text

    max_len = max(len(letters) for letters in codes)

    def _replace(match: "re.Match") -> str:
        letters = "".join(match.group(0).split()).lower()
        canonical = codes.get(letters)
        return canonical[0] if canonical else match.group(0)

    for length in range(max_len, 1, -1):
        pattern = re.compile(r"\b" + r"\s+".join(["[A-Za-z]"] * length) + r"\b", re.IGNORECASE)
        text = pattern.sub(_replace, text)

    return text


def _expand_catalog_anchored_numbers(text: str, codes: Dict[str, Tuple[str, str]]) -> str:
    """Rewrite '<catalog code letters> <number, spoken or digits>' into '<Letters><digits>'.

    Handles "ESP 32", "ESP thirty two", and "DHT eleven" - never a bare
    number word elsewhere in the sentence, since the pattern only matches
    right after a real catalog component's code prefix.
    """
    if not codes:
        return text

    tens_names = "|".join(re.escape(w) for w in _TENS)
    ones_names = "|".join(re.escape(w) for w in _ONES)
    teens_names = "|".join(re.escape(w) for w in _TEENS)
    number_phrase = rf"\d{{1,3}}|(?:{tens_names})(?:[\s-]+(?:{ones_names}))?|{teens_names}|{ones_names}"

    prefix_alternation = "|".join(re.escape(letters) for letters in codes)
    pattern = re.compile(rf"\b({prefix_alternation})[\s-]+({number_phrase})\b", re.IGNORECASE)

    def _replace(match: "re.Match") -> str:
        canonical_letters, _ = codes[match.group(1).lower()]
        digits = _number_phrase_to_digits(match.group(2))
        return f"{canonical_letters}{digits}"

    return pattern.sub(_replace, text)


def _normalize_known_name_casing(text: str, known_names: List[str]) -> str:
    """Fix casing for any full catalog name already spelled out correctly (e.g. 'accelerometer' -> 'Accelerometer')."""
    for name in sorted(known_names, key=len, reverse=True):
        pattern = re.compile(r"\b" + re.escape(name) + r"\b", re.IGNORECASE)
        text = pattern.sub(name, text)
    return text


def _fuzzy_match_code(token: str, codes: Dict[str, Tuple[str, str]]) -> Optional[str]:
    """Return the one component code `token` unambiguously sounds like, or None.

    Only ever compares against short code prefixes (e.g. "esp", "dht"),
    never a full multi-word descriptive name - see module docstring for why.
    """
    if not codes:
        return None

    token_lower = token.lower()
    scored = [(difflib.SequenceMatcher(None, token_lower, prefix).ratio(), letters, digits) for prefix, (letters, digits) in codes.items()]
    scored.sort(key=lambda item: item[0], reverse=True)

    best_score, best_letters, best_digits = scored[0]
    second_score = scored[1][0] if len(scored) > 1 else 0.0

    if best_score >= _FUZZY_MATCH_THRESHOLD and (best_score - second_score) >= _FUZZY_MATCH_MARGIN:
        return f"{best_letters}{best_digits}"
    return None


_ALNUM_TOKEN_PATTERN = re.compile(r"^([A-Za-z]+)(\d+)$")


def _fuzzy_match_alnum_code(token: str, codes: Dict[str, Tuple[str, str]]) -> Optional[str]:
    """Fuzzy-resolve a letters+digits token (e.g. "TH11") against known codes, or return None.

    An exact digit match is strong corroborating evidence - a dropped
    leading consonant with the number intact ("DHT11" heard as "TH11") is a
    far more plausible STT error than an unrelated code happening to share
    the same number by coincidence - so it can outweigh a middling letter
    score. A *mismatched* digit heavily discounts the letter-only score
    instead, since a code with the wrong number is arguably a different,
    equally valid component (e.g. "STM32" is not "ESP32" just because both
    are microcontrollers) - never returns anything for it.
    """
    match = _ALNUM_TOKEN_PATTERN.match(token)
    if not match or not codes:
        return None

    letters, digits = match.group(1).lower(), match.group(2)

    best_score, best_code = 0.0, None
    for prefix, (canon_letters, canon_digits) in codes.items():
        letter_score = difflib.SequenceMatcher(None, letters, prefix).ratio()
        score = max(letter_score, 0.75) if digits == canon_digits else letter_score * 0.5
        if score > best_score:
            best_score, best_code = score, f"{canon_letters}{canon_digits}"

    if best_score >= _ALNUM_FUZZY_THRESHOLD:
        return best_code
    return None


def _fuzzy_correct_remaining_tokens(
    text: str,
    known_names: List[str],
    codes: Dict[str, Tuple[str, str]],
    unresolved: List[str],
) -> str:
    """Apply the fuzzy matchers to whichever words survived the earlier, safer passes.

    Tokenizes on whole alphanumeric runs (e.g. "ESP32" is one token), not
    letters alone - matching only the letters would split an
    already-correct "ESP32" into "ESP" + "32", and "ESP" alone fuzzy-matches
    "ESP32"'s own prefix, duplicating the digits (e.g. "ESP3232"). A purely
    alphabetic, unresolved token (like "NSP") is matched against catalog
    codes by letters alone; a letters+digits token (like "TH11") is matched
    using the digit-aware matcher instead, which never re-touches an
    already-correct code (see _fuzzy_match_alnum_code()'s own exact-digit
    handling) or double-appends digits.

    Appends to `unresolved` (in place) every letters+digits token that
    looks like an attempted technical code but didn't confidently resolve -
    normalize_command() uses this to decide whether to raise
    UnresolvedComponentError rather than silently leaving it as-is.
    """
    exact_lower_names = {name.lower() for name in known_names}

    def _replace(match: "re.Match") -> str:
        word = match.group(0)
        lowered = word.lower()

        if lowered in exact_lower_names or lowered in _STRUCTURE_WORDS or lowered in _KNOWN_TECHNICAL_TERMS:
            return word
        if len(word) < 2:
            return word

        looks_like_code = bool(_ALNUM_TOKEN_PATTERN.match(word))
        if word.isalpha():
            corrected = _fuzzy_match_code(word, codes)
        elif looks_like_code:
            corrected = _fuzzy_match_alnum_code(word, codes)
        else:
            corrected = None

        if corrected is None:
            if looks_like_code:
                unresolved.append(word)
            return word

        logger.info("Command normalization: fuzzy-corrected %r -> %r", word, corrected)
        return corrected

    return re.sub(r"[A-Za-z0-9]+", _replace, text)


def normalize_command(transcript: str, knowledge: Optional[Dict] = None) -> Optional[str]:
    """Correct speech-recognition artifacts in `transcript` against the real component catalog.

    Args:
        transcript: Raw Speech-to-Text output.
        knowledge: Optional pre-loaded KnowledgeLoader().load_all() result;
            loaded fresh if omitted, so this is usable standalone (e.g. from
            a test) without a caller having to load it first.

    Returns:
        The corrected command text, structurally unchanged (same word
        order, same connecting words) - only component-name tokens are
        ever rewritten, and only when confidently resolved against the
        catalog. Returns `transcript` unchanged if there's no catalog to
        check against. Returns None - never a "best effort" guess - when
        `transcript` is dominated by one word repeated an implausible
        number of times (see _is_severely_repetitive()): that is a sign
        the underlying speech recognition itself failed, not something
        normalization can safely paper over, so the caller is expected to
        stop before ever reaching the Planner (see api/server.py's voice
        endpoint) rather than run a "corrected" garbage command.

    Raises:
        UnresolvedComponentError: a letters+digits token looks like an
            attempted technical component code but nothing in the catalog
            is confidently close to it - the caller is expected to stop
            before the Planner here too, and ask the user to repeat,
            rather than invent a component from a guess.
    """
    logger.info("RAW TRANSCRIPT:\n%s", transcript)

    if not transcript:
        logger.info("NORMALIZED COMMAND:\n%s", transcript)
        return transcript

    if _is_severely_repetitive(transcript):
        logger.warning(
            "NORMALIZED COMMAND:\nrejected - transcript is dominated by one repeated word "
            "(likely a speech-recognition failure, not a real command): %r",
            transcript,
        )
        return None

    known_names = _known_component_names(knowledge if knowledge is not None else KnowledgeLoader().load_all())
    if not known_names:
        logger.info("NORMALIZED COMMAND:\n%s", transcript)
        return transcript

    codes = _catalog_codes(known_names)
    unresolved: List[str] = []

    normalized = transcript
    normalized = _collapse_spelled_letters(normalized, codes)
    normalized = _expand_catalog_anchored_numbers(normalized, codes)
    normalized = _normalize_known_name_casing(normalized, known_names)
    normalized = _fuzzy_correct_remaining_tokens(normalized, known_names, codes, unresolved)

    resolved_components = [name for name in known_names if re.search(r"\b" + re.escape(name) + r"\b", normalized, re.IGNORECASE)]
    confidence = "high" if not unresolved else "low"

    if normalized != transcript:
        logger.info("Command normalization: %r -> %r", transcript, normalized)

    logger.info("NORMALIZED COMMAND:\n%s", normalized)
    logger.info("RESOLVED COMPONENTS:\n%s", resolved_components or "(none)")
    logger.info("UNRESOLVED TERMS:\n%s", unresolved or "(none)")
    logger.info("CONFIDENCE:\n%s", confidence)

    if unresolved:
        logger.warning("Refusing to guess at unresolved technical term(s) %r - not executing", unresolved)
        raise UnresolvedComponentError(unresolved)

    return normalized
