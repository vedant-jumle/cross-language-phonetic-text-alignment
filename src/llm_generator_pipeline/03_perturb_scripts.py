import json
import sys
import os
import concurrent.futures
from pathlib import Path
from tqdm.auto import tqdm
import yaml
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

sys.path.append(str(Path(__file__).resolve().parents[1]))
from llm_generator_pipeline.llm_client import parse_json_response

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


def _make_json_schema(target_scripts: list) -> dict:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "transliteration",
            "schema": {
                "type": "object",
                "properties": {s: {"type": "string"} for s in target_scripts},
                "required": target_scripts,
                "additionalProperties": False,
            },
        },
    }


def transliterate_name(name: str, target_scripts: list, client: OpenAI, json_schema: dict) -> dict:
    try:
        r = client.chat.completions.create(
            model="code",
            messages=[{"role": "user", "content": f'Transliterate "{name}" into {target_scripts} phonetically.'}],
            max_tokens=200,
            response_format=json_schema,
        )
        content = r.choices[0].message.content
        if content:
            return json.loads(content)
    except Exception:
        pass
    return {s: "" for s in target_scripts}


def main(config_path):
    config = load_config(config_path)
    target_scripts = config["target_scripts"]

    client = OpenAI(
        base_url="https://api.tulip.tudelft.nl/code/v1/",
        api_key=os.environ["TUDELFT_TULIP_API_KEY"],
    )
    json_schema = _make_json_schema(target_scripts)

    processed_ids = load_processed_ids()
    records = [r for r in read_jsonl(INPUT_PATH) if r["entity_id"] not in processed_ids]

    # Build flat list of (record_idx, name) pairs across all records
    pairs = []
    for rec_idx, record in enumerate(records):
        for name in [record["name_en"]] + record.get("latin_variants", []):
            pairs.append((rec_idx, name))

    script_variants = [{} for _ in records]

    with concurrent.futures.ThreadPoolExecutor(max_workers=50) as ex:
        future_to_pair = {
            ex.submit(transliterate_name, name, target_scripts, client, json_schema): (rec_idx, name)
            for rec_idx, name in pairs
        }
        with open(OUTPUT_PATH, "a", encoding="utf-8") as out:
            completed = 0
            pending_records = {}  # rec_idx -> number of names still in flight
            for rec_idx, name in pairs:
                pending_records[rec_idx] = pending_records.get(rec_idx, 0) + 1

            for fut in tqdm(concurrent.futures.as_completed(future_to_pair), total=len(pairs), desc="Transliterating names"):
                rec_idx, name = future_to_pair[fut]
                script_variants[rec_idx][name] = fut.result()
                pending_records[rec_idx] -= 1

                # Write record as soon as all its names are done
                if pending_records[rec_idx] == 0:
                    record = records[rec_idx]
                    output = {
                        "entity_id": record["entity_id"],
                        "name_en": record["name_en"],
                        "wikidata": record["wikidata"],
                        "latin_variants": record.get("latin_variants", []),
                        "script_variants": script_variants[rec_idx],
                    }
                    out.write(json.dumps(output, ensure_ascii=False) + "\n")
                    out.flush()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    main(args.config)
