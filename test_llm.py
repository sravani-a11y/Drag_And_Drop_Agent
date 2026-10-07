from llm import analyze

prompt = """
Return ONLY valid JSON.

{
    "intent": "",
    "component": ""
}

User Request:
Add ESP32
"""

result = analyze(prompt)
print(result)