"""
Stage 02 (combined): Generate Latin phonetic variants + script transliterations
using the Anthropic Message Batches API.

Replaces 02_perturb_latin.py and 03_perturb_scripts.py.
Output is compatible with 04_merge_wikidata.py.
"""

import json
import sys
import time
from pathlib import Path

import anthropic
import yaml

INPUT_PATH = "data/pipeline/01_sampled.jsonl"
OUTPUT_PATH = "data/pipeline/02_perturbed_combined.jsonl"
BATCH_STATE_PATH = "data/pipeline/02_batch_state.json"


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


def build_prompt(name, n_perturbations, target_scripts):
    return f'''For the name "{name}", do two things:

1. Generate {n_perturbations} DISTINCT phonetic spelling variants (how different people might mishear or misspell it):
   - Stay in Latin script
   - Keep the same number of words as the original
   - Each variant must differ from the original and from each other

2. Phonetically transliterate the original name into these scripts: {target_scripts}
   - Use only target script characters, no Latin letters

Return a single JSON object only, no explanation:
{{
  "latin_variants": ["variant1", "variant2", "variant3", "variant4"],
  "scripts": {{"ar": "...", "ru": "...", "zh": "...", "ja": "...", "he": "...", "hi": "...", "el": "...", "ko": "..."}}
}}'''


def parse_response(raw: str, target_scripts: list):
    raw = raw.strip()
    if "```" in raw:
        parts = raw.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("{"):
                raw = part
                break
    data = json.loads(raw)
    latin_variants = data.get("latin_variants", [])
    if isinstance(latin_variants, dict):
        latin_variants = [v for v in latin_variants.values() if isinstance(v, str) and v.strip()]
    scripts = data.get("scripts", {s: "" for s in target_scripts})
    return latin_variants, scripts


def load_batch_state():
    try:
        with open(BATCH_STATE_PATH) as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def save_batch_state(state):
    Path(BATCH_STATE_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(BATCH_STATE_PATH, "w") as f:
        json.dump(state, f)


def main(config_path):
    config = load_config(config_path)
    n_perturbations = config["n_perturbations"]
    target_scripts = config["target_scripts"]
    max_names = config.get("max_names", None)
    api_key = config.get("anthropic_api_key", None)

    client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)

    # Check if we have a pending batch to resume
    state = load_batch_state()
    if state and state.get("status") != "complete":
        print(f"Resuming batch {state['batch_id']} (status: {state['status']})")
        batch_id = state["batch_id"]
        id_to_record = {r["entity_id"]: r for r in state["records"]}
        poll_and_write(client, batch_id, id_to_record, target_scripts, config)
        return

    # Load records to process
    processed_ids = load_processed_ids()
    records = [
        r for r in read_jsonl(INPUT_PATH)
        if r["entity_id"] not in processed_ids
    ]
    if max_names is not None:
        records = records[:max_names]

    if not records:
        print("Nothing to process.")
        return

    print(f"Submitting {len(records):,} names to Anthropic Batch API...")

    # Build batch requests
    requests = [
        {
            "custom_id": record["entity_id"],
            "params": {
                "model": "claude-haiku-4-5-20251001",
                "max_tokens": 512,
                "messages": [
                    {
                        "role": "user",
                        "content": build_prompt(record["name_en"], n_perturbations, target_scripts)
                    }
                ]
            }
        }
        for record in records
    ]

    # Submit batch (Anthropic API limit: 100k requests per batch)
    BATCH_LIMIT = 100_000
    if len(requests) > BATCH_LIMIT:
        print(f"Warning: truncating to {BATCH_LIMIT} requests (API limit)")
        requests = requests[:BATCH_LIMIT]
        records = records[:BATCH_LIMIT]

    batch = client.messages.batches.create(requests=requests)
    print(f"Batch submitted: {batch.id}")
    print(f"Status: {batch.processing_status}")

    # Save state so we can resume if interrupted
    save_batch_state({
        "batch_id": batch.id,
        "status": batch.processing_status,
        "records": records,
    })

    poll_and_write(client, batch.id, {r["entity_id"]: r for r in records}, target_scripts, config)


def poll_and_write(client, batch_id, id_to_record, target_scripts, config):
    print(f"Polling batch {batch_id}...")
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        counts = batch.request_counts
        print(
            f"  Status: {batch.processing_status} | "
            f"processing={counts.processing} succeeded={counts.succeeded} "
            f"errored={counts.errored} canceled={counts.canceled} expired={counts.expired}",
            flush=True,
        )
        if batch.processing_status == "ended":
            break
        time.sleep(60)

    # Write results
    succeeded = 0
    failed = 0
    with open(OUTPUT_PATH, "a", encoding="utf-8") as out:
        for result in client.messages.batches.results(batch_id):
            entity_id = result.custom_id
            record = id_to_record[entity_id]

            if result.result.type == "succeeded":
                raw = result.result.message.content[0].text
                try:
                    latin_variants, scripts = parse_response(raw, target_scripts)
                except Exception as e:
                    print(f"  Parse error for {entity_id} ({record['name_en']}): {e}", file=sys.stderr)
                    latin_variants, scripts = [], {s: "" for s in target_scripts}
                    failed += 1
                else:
                    succeeded += 1
            else:
                print(f"  API error for {entity_id}: {result.result.type}", file=sys.stderr)
                latin_variants, scripts = [], {s: "" for s in target_scripts}
                failed += 1

            output = {
                "entity_id": entity_id,
                "name_en": record["name_en"],
                "wikidata": record["wikidata"],
                "latin_variants": latin_variants,
                # stage 04 expects script_variants[name][script] = text
                "script_variants": {record["name_en"]: scripts},
            }
            out.write(json.dumps(output, ensure_ascii=False) + "\n")

    print(f"\nDone. succeeded={succeeded} failed={failed}")
    print(f"Output written to {OUTPUT_PATH}")

    # Mark state complete
    state = load_batch_state()
    if state:
        state["status"] = "complete"
        save_batch_state(state)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    main(args.config)
