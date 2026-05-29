import json
import sys
from pathlib import Path
from tqdm.auto import tqdm
import yaml

sys.path.append(str(Path(__file__).resolve().parents[1]))
from llm_generator_pipeline.llm_client import call_hf_batch, parse_json_response

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def load_processed_ids(output_path):
    processed = set()
    try:
        with open(output_path, encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                processed.add(record["entity_id"])
    except FileNotFoundError:
        pass
    return processed

def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)

def build_prompt(name, n_perturbations):
    return f"""Generate {n_perturbations} DISTINCT phonetic spelling variants of this name as it sounds when spoken: "{name}"

Rules:
- Each variant must be spelled differently from all others and from the original
- Simulate how different people might mishear or misspell the name phonetically
- Keep the same number of words as the original
- Do NOT repeat variants
- Do NOT use nicknames, abbreviations, or shortened forms
- Do NOT change language (stay in Latin script)

Example: "Catherine" → ["Kathryn", "Katherin", "Cathryn", "Katheryne"]

Return a JSON array of exactly {n_perturbations} strings, no explanation:
["variant1", "variant2", ...]"""

def parse_variants(raw: str, name: str = "") -> list:
    try:
        if raw.strip().startswith("["):
            return json.loads(raw.strip())
        result = parse_json_response(raw)
        if isinstance(result, list):
            return result
        if isinstance(result, dict):
            # Model returned {"variant1": "...", "variant2": "..."} — extract values
            values = [v for v in result.values() if isinstance(v, str) and v.strip()]
            if values:
                return values
    except Exception:
        pass
    print(f"parse_variants failed for '{name}': {repr(raw[:200])}", file=sys.stderr)
    return []

def main(config_path, input_path, output_path, shard, n_shards):
    config = load_config(config_path)
    batch_size = config.get("batch_size", 20)
    model_id = config["hf_model_id"]
    n_perturbations = config["n_perturbations"]

    processed_ids = load_processed_ids(output_path)
    records = [r for r in read_jsonl(input_path) if r["entity_id"] not in processed_ids]
    records = records[shard::n_shards]

    with open(output_path, "a", encoding="utf-8") as out:
        for i in tqdm(range(0, len(records), batch_size), desc="Generating Latin variants"):
            batch = records[i : i + batch_size]
            prompts = [build_prompt(r["name_en"], n_perturbations) for r in batch]
            responses = call_hf_batch(prompts, model_id)
            for record, raw in zip(batch, responses):
                variants = parse_variants(raw, record["name_en"])
                output = {
                    "entity_id": record["entity_id"],
                    "name_en": record["name_en"],
                    "wikidata": record["wikidata"],
                    "latin_variants": variants,
                }
                out.write(json.dumps(output, ensure_ascii=False) + "\n")
            out.flush()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",    default="config.yaml")
    parser.add_argument("--input",     default="data/pipeline/01_sampled.jsonl")
    parser.add_argument("--output",    default="data/pipeline/02_perturbed_latin.jsonl")
    parser.add_argument("--shard",     type=int, default=0)
    parser.add_argument("--n_shards",  type=int, default=1)
    args = parser.parse_args()
    main(args.config, args.input, args.output, args.shard, args.n_shards)
