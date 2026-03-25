import json
import re
import os
import concurrent.futures
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

_client = None
_BASE_URL = "https://api.tulip.tudelft.nl/code/v1/"


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.environ["TUDELFT_TULIP_API_KEY"]
        _client = OpenAI(base_url=_BASE_URL, api_key=api_key)
    return _client


def call_hf_batch(prompts: list[str], model_id: str, max_new_tokens: int = 512) -> list[str]:
    """Call TULIP API in parallel threads. model_id is ignored (uses TULIP code model)."""
    client = _get_client()

    def _call(prompt):
        r = client.chat.completions.create(
            model="code",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_new_tokens,
        )
        return r.choices[0].message.content or ""

    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as ex:
        return list(ex.map(_call, prompts))


def parse_json_response(response: str):
    """Parse LLM response as JSON. Strips markdown code fences if present."""
    response = response.strip()
    if "```" in response:
        parts = response.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("{") and part.endswith("}"):
                return json.loads(part)
            if part.startswith("[") and part.endswith("]"):
                return json.loads(part)
    match = re.search(r"[\[{].*[\]}]", response, re.DOTALL)
    if match:
        return json.loads(match.group())
    raise ValueError("No valid JSON found in LLM response")
