import json
import sys
import yaml
from pathlib import Path
from tqdm import tqdm
sys.path.append(str(Path(__file__).resolve().parents[1]))

from llm_generator_pipeline.llm_client import call_ollama, parse_json_response


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

def build_prompt(entity, target_scripts):
    all_names = [entity["name_en"]] + entity.get("latin_variants", [])
    return f"""
Transliterate each of these names into the following scripts: {target_scripts}.
Use phonetic transliteration in the actual script only — do not write Latin approximations. 
Each Latin variant must have its own distinct transliteration. Do not copy from other names. Use only the target script, no Latin letters.
Do not translate meaning.

Names: {json.dumps(all_names)}

Return JSON only, with each name as a separate key and the script variants in strictly the target script as the values, like this example:

{{
  "Catherine": {{
    "ar": "كاثرين",
    "ru": "Екатерина",
    "zh": "凯瑟琳",
    "ja": "カテリン",
    "he": "קתרין",
    "hi": "कैथरीन",
    "el": "Καθερίν",
    "ko": "캐서린"
  }},
  ...
}}

Return JSON only, no explanation.
"""

def process_entity(entity, config):
    target_scripts = config["target_scripts"]
    prompt = build_prompt(entity, target_scripts)
    for attempt in range(3):
        try:
            raw = call_ollama(
                prompt,
                config["llm_model"],
                config["llm_base_url"]
            )
            data = parse_json_response(raw)
            return data
        except Exception as e:
            if attempt == 2:
                print(f"Failed entity {entity['entity_id']}: {e}", file=sys.stderr)
                return None
            
def main(config_path):
    config = load_config(config_path)
    processed_ids = load_processed_ids()
    records = list(read_jsonl(INPUT_PATH))
    remaining = [r for r in records if r["entity_id"] not in processed_ids]

    with open(OUTPUT_PATH, "a", encoding="utf-8") as out:
        for record in tqdm(remaining):
            data = process_entity(record, config)
            if data:
                output = {
                    "entity_id": record["entity_id"],
                    "name_en": record["name_en"],
                    "wikidata": record["wikidata"],
                    "latin_variants": record.get("latin_variants", []),
                    "script_variants": {}
                }
                # Fill in script_variants for each name
                all_names = [record["name_en"]] + record.get("latin_variants", [])
                output["script_variants"] = {}
                for name in all_names:
                    if isinstance(data, dict):
                        variants = data.get(name)
                        if not isinstance(variants, dict):
                            variants = {s: "" for s in config["target_scripts"]}
                            print(f"Missing or invalid key for {name}", file=sys.stderr)
                        output["script_variants"][name] = variants
                    else:
                        output["script_variants"][name] = {s: "" for s in config["target_scripts"]}
                out.write(json.dumps(output, ensure_ascii=False) + "\n")
                out.flush()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    main(args.config)