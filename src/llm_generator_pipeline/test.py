import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from llm_generator_pipeline.llm_client import call_ollama, parse_json_response

response = call_ollama(
    prompt='Return JSON: {"hello": "world"}',
    model="llama3.1",
    base_url="http://localhost:11434"
)

print("RAW RESPONSE:")
print(response)

print("\nPARSED JSON:")
print(parse_json_response(response))