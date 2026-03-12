import ollama
import time
import json

def call_ollama(
    prompt: str,
    model: str,
    base_url: str,
    max_retries: int = 3,
) -> str:
    """Send a prompt to Ollama, return raw string response."""
    client = ollama.Client(host=base_url)
    for attempt in range(max_retries):
        try:
            response = client.generate(
                model=model,
                prompt=prompt,
                stream=False,
            )
            return response["response"]
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)

def parse_json_response(response: str) -> dict:
    """
    Parse LLM response as JSON. Strips markdown code fences if present.
    Raises ValueError on malformed JSON after stripping.
    """
    response = response.strip()
    if response.startswith("```"):
        parts = response.split("```")
        if len(parts) >= 2:
            response = parts[1]
            if response.startswith("json"):
                response = response[4:]
            response = response.strip()

    return json.loads(response)

