import json
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

_tokenizer = None
_model = None
_loaded_model_id = None


def _load_model(model_id: str):
    global _tokenizer, _model, _loaded_model_id
    if _model is not None and _loaded_model_id == model_id:
        return
    print(f"Loading model: {model_id}", flush=True)
    _tokenizer = AutoTokenizer.from_pretrained(model_id)
    _tokenizer.padding_side = "left"
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token
    _model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    _model.eval()
    _loaded_model_id = model_id
    print("Model loaded.", flush=True)


def call_hf_batch(prompts: list[str], model_id: str, max_new_tokens: int = 512) -> list[str]:
    """Run a batch of prompts through the HF model. Returns list of response strings."""
    _load_model(model_id)

    chats = [[{"role": "user", "content": p}] for p in prompts]
    formatted = [
        _tokenizer.apply_chat_template(c, tokenize=False, add_generation_prompt=True)
        for c in chats
    ]

    inputs = _tokenizer(
        formatted,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to(_model.device)

    with torch.no_grad():
        outputs = _model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=_tokenizer.pad_token_id,
        )

    input_len = inputs["input_ids"].shape[1]
    results = []
    for out in outputs:
        new_tokens = out[input_len:]
        text = _tokenizer.decode(new_tokens, skip_special_tokens=True)
        results.append(text.strip())
    return results


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
