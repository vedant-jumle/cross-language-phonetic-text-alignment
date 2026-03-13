import json
import os
import sys
from pathlib import Path
from tqdm import tqdm
import yaml
sys.path.append(str(Path(__file__).resolve().parents[1]))
from llm_generator_pipeline.llm_client import call_ollama, parse_json_response

INPUT_PATH = "data/pipeline/01_sampled.jsonl"
OUTPUT_PATH = "data/pipeline/02_perturbed_latin.jsonl"


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)
    

def load_processed_ids():
    processed = set()
    try:
        with open(OUTPUT_PATH) as f:
            for line in f:
                record = json.loads(line)
                processed.add(record["entity_id"])
    except FileNotFoundError:
        pass
    return processed

def read_jsonl(path):
    with open(path) as f:
        for line in f:
            yield json.loads(line)

def build_prompt(names, n_perturbations):
    return f"""
Given these English names, generate {n_perturbations} realistic phonetic
spelling variants for each. Variants should sound similar when spoken aloud
but differ in spelling. Do NOT generate: nicknames, abbreviations, or
shortened forms.

Names: {json.dumps(names)}

Return JSON only, no explanation:
"""

def process_batch(batch, config):
    names = [r["name_en"] for r in batch]
    prompt = build_prompt(names, config["n_perturbations"])
    for attempt in range(3):
        try:
            raw = call_ollama(
                prompt,
                config["llm_model"],
                config["llm_base_url"]
            )
            data = parse_json_response(raw)
            return data
        except Exception:
            if attempt == 2:
                print("Failed batch:", names, file=sys.stderr)
                return None
            
def main(config_path):
    config = load_config(config_path)
    processed_ids = load_processed_ids()
    records = list(read_jsonl(INPUT_PATH))
    remaining = [r for r in records if r["entity_id"] not in processed_ids]
    batch_size = config["batch_size"]
    batch = []
    with open(OUTPUT_PATH, "a") as out:
        for record in tqdm(remaining):
            batch.append(record)
            if len(batch) == batch_size:
                data = process_batch(batch, config)
                if data:
                    for r in batch:
                        variants = data.get(r["name_en"], [])
                        if r["name_en"] not in data:
                            print("Missing key:", r["name_en"], file=sys.stderr)
                        output = {
                            "entity_id": r["entity_id"],
                            "name_en": r["name_en"],
                            "wikidata": r["wikidata"],
                            "latin_variants": variants
                        }
                        out.write(json.dumps(output, ensure_ascii=False) + "\n")
                out.flush()
                batch = []
            if batch:
                data = process_batch(batch, config)
                if data:
                    for r in batch:
                        variants = data.get(r["name_en"], [])
                    output = {
                        "entity_id": r["entity_id"],
                        "name_en": r["name_en"],
                        "wikidata": r["wikidata"],
                        "latin_variants": variants
                    }
                    out.write(json.dumps(output, ensure_ascii=False) + "\n")
                out.flush()

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")

    args = parser.parse_args()

    main(args.config)