"""Tests for command_normalizer.py - the Speech-to-Text -> LangGraph normalization layer.

Plain script (matches test_knowledge_loader.py/test_llm.py's style, not
pytest) - run with `python test_command_normalizer.py`. No LLM/Ollama or
LangGraph involved - normalize_command() only needs the component catalog.
"""

from command_normalizer import UnresolvedComponentError, normalize_command

passed = 0
failed = 0


def check(transcript, expected):
    global passed, failed
    got = normalize_command(transcript)
    if got == expected:
        passed += 1
        print(f"PASS: {transcript!r} -> {got!r}")
    else:
        failed += 1
        print(f"FAIL: {transcript!r} -> {got!r} (expected {expected!r})")


# --- Required cases ---
check("create NSP connected to the accelerometer", "create ESP32 connected to the Accelerometer")
check("add ESP 32", "add ESP32")
check("add DHT eleven", "add DHT11")
check("add ESP32 and accelerometer", "add ESP32 and Accelerometer")

# --- Case-insensitivity ---
check("add esp 32", "add ESP32")
check("ADD ESP 32", "ADD ESP32")

# --- Spoken compound numbers ---
check("add esp thirty two and dht eleven", "add ESP32 and DHT11")

# --- Must NOT blindly replace arbitrary/structure words ---
check("connect two components together", "connect two components together")
check("show canvas", "show canvas")

# --- Must NOT fuzzy-match a generic word against a multi-word catalog name ---
# ("relay"/"sensor"/"module" are real English words and part of real catalog
# names, but aren't themselves garbled STT output - the existing
# planner.py already resolves shorthand like "relay" downstream; this layer
# only fixes STT artifacts in component *codes* (ESP32/DHT11/STM32-style).
check("delete relay", "delete relay")
check("add esp32 and relay module", "add ESP32 and Relay Module")
check("remove the temperature sensor", "remove the Temperature Sensor")

# --- Known technical terms must never be fuzzy-corrected into a component,
# no matter how similar the characters look (e.g. "SPI" superficially
# resembles "ESP" by plain character overlap - they are unrelated) ---
check("connect ESP32 to SPI sensor", "connect ESP32 to SPI sensor")
check("add ESP32, relay module and LED", "add ESP32, Relay Module and LED")
check("connect the UART to the ESP32", "connect the UART to the ESP32")
check("I2C and SPI and Bluetooth and WiFi", "I2C and SPI and Bluetooth and WiFi")

# --- Severely repetitive/corrupted transcripts must be rejected (None),
# not "corrected" into a repeated wrong component - a general statistical
# check, not a check for this exact phrase ---
global_repetitive_cases = [
    "LED, UART, I2C, SPI, Bluetooth, WiFi, ESP32, ESP32, ESP32, ESP32, ESP32",
    "ESP32 ESP32 ESP32 ESP32 ESP32 ESP32 ESP32",
    "connect connect connect connect connect connect connect",
]
for corrupted in global_repetitive_cases:
    result = normalize_command(corrupted)
    if result is None:
        passed += 1
        print(f"PASS: {corrupted!r} -> rejected (None), as expected")
    else:
        failed += 1
        print(f"FAIL: {corrupted!r} -> {result!r} (expected None - severe repetition)")

# --- A legitimate short repeat (e.g. two identical coordinates) must NOT
# be mistaken for corruption ---
check("move ESP32 to 300 300", "move ESP32 to 300 300")

# --- General technical component resolution (registry-driven, not
# sentence-specific) ---
check("add ESP-32", "add ESP32")
check("add DHT-11", "add DHT11")
check("add E S P thirty two", "add ESP32")
check("add D H T eleven", "add DHT11")

# --- Must resolve a garbled code to the closest real catalog component
# with sufficient confidence, never invent a title-cased guess like
# "The Th11 Sensor" ---
check("Add the TH11 sensor", "Add the DHT11 sensor")

# --- Must NOT invent a component when confidence is insufficient - reject
# instead (never sentence-specific: any sufficiently garbled letters+digits
# code triggers this, not just "TH55") ---
for unresolvable in ["add the TH55 sensor", "add the ZQ99 board"]:
    try:
        result = normalize_command(unresolvable)
        failed += 1
        print(f"FAIL: {unresolvable!r} -> {result!r} (expected UnresolvedComponentError)")
    except UnresolvedComponentError:
        passed += 1
        print(f"PASS: {unresolvable!r} -> correctly rejected (UnresolvedComponentError)")

print(f"\n{passed} passed, {failed} failed")
