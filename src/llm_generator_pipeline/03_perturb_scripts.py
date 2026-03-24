import json
import sys
from pathlib import Path
from tqdm.auto import tqdm
import yaml

sys.path.append(str(Path(__file__).resolve().parents[1]))
from llm_generator_pipeline.llm_client import call_hf_batch, parse_json_response

INPUT_PATH = "data/pipeline/02_perturbed_latin.jsonl"
OUTPUT_PATH = "data/pipeline/03_perturbed_scripts.jsonl"


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def load_processed_ids():
    processed = set()
    try:
        with open(OUTPUT_PATH, encoding="utf-8") as f:
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

def build_prompt(name, target_scripts):
    return f"""Transliterate this name into the following scripts: {target_scripts}.
Use phonetic transliteration only — how the name sounds, not its meaning.
Use only the target script characters, no Latin letters.

Name: "{name}"

Return a JSON object only, no explanation:
{{"ar": "...", "ru": "...", ...}}"""

def parse_scripts(raw: str, target_scripts: list) -> dict:
    try:
        result = parse_json_response(raw)
        if isinstance(result, dict):
            return result
    except Exception:
        pass
    return {s: "" for s in target_scripts}

def main(config_path):
    config = load_config(config_path)
    batch_size = config.get("batch_size", 20)
    model_id = config["hf_model_id"]
    target_scripts = config["target_scripts"]

    processed_ids = load_processed_ids()
    records = [r for r in read_jsonl(INPUT_PATH) if r["entity_id"] not in processed_ids]

    with open(OUTPUT_PATH, "a", encoding="utf-8") as out:
        for i in tqdm(range(0, len(records), batch_size), desc="Transliterating scripts"):
            batch = records[i : i + batch_size]

            # Collect all (entity_idx, name) pairs across the batch
            pairs = []
            for rec_idx, record in enumerate(batch):
                all_names = [record["name_en"]] + record.get("latin_variants", [])
                for name in all_names:
                    pairs.append((rec_idx, name))

            # Send names in sub-batches to balance throughput vs GPU memory
            script_variants = [{} for _ in batch]
            prompt_batch_size = 8
            for j in range(0, len(pairs), prompt_batch_size):
                sub_pairs = pairs[j : j + prompt_batch_size]
                prompts = [build_prompt(name, target_scripts) for _, name in sub_pairs]
                responses = call_hf_batch(prompts, model_id, max_new_tokens=150)
                for (rec_idx, name), raw in zip(sub_pairs, responses):
                    script_variants[rec_idx][name] = parse_scripts(raw, target_scripts)

            for record, sv in zip(batch, script_variants):
                output = {
                    "entity_id": record["entity_id"],
                    "name_en": record["name_en"],
                    "wikidata": record["wikidata"],
                    "latin_variants": record.get("latin_variants", []),
                    "script_variants": sv,
                }
                out.write(json.dumps(output, ensure_ascii=False).encode('utf-8', errors='replace').decode('utf-8') + "\n")
            out.flush()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    main(args.config)
