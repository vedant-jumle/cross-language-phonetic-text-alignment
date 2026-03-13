import json
import hashlib
from pathlib import Path

INPUT_PATH = "data/pipeline/03_perturbed_scripts.jsonl"
OUTPUT_PATH = "data/pipeline/04_dataset.jsonl"

def to_bytes(text: str) -> list[int]:
    return list(text.encode("utf-8"))

def assign_split(entity_id: str, train: float, val: float) -> str:
    bucket = int(hashlib.md5(entity_id.encode()).hexdigest(), 16) % 1000
    if bucket < train * 1000:
        return "train"
    elif bucket < (train + val) * 1000:
        return "val"
    return "test"

def merge_wikidata_record(record: dict, config: dict) -> dict:
    anchor_text = record["name_en"]
    anchor_bytes = to_bytes(anchor_text)

    positives = []

    for name in record.get("latin_variants", []):
        if name:
            positives.append({
                "text": name,
                "bytes": to_bytes(name),
                "type": "phonetic",
                "source": "llm"
            })

    for name, scripts in record.get("script_variants", {}).items():
        if name == record["name_en"]:
            typ = "script"
        else:
            typ = "combined"
        for text in scripts.values():
            if text:
                positives.append({
                    "text": text,
                    "bytes": to_bytes(text),
                    "type": typ,
                    "source": "llm"
                })

    for value in record.get("wikidata", {}).values():
        if value:
            positives.append({
                "text": value,
                "bytes": to_bytes(value),
                "type": "script",
                "source": "wikidata"
            })

    seen_texts = {}
    deduped = []
    for p in positives:
        key = p["text"].strip().lower()
        if key not in seen_texts:
            deduped.append(p)
            seen_texts[key] = len(deduped) - 1
        else:
            if p["source"] == "wikidata":
                idx = seen_texts[key]
                deduped[idx] = p

    positives = deduped
    split = assign_split(record["entity_id"], config["split_train"], config["split_val"])
    return {
        "entity_id": record["entity_id"],
        "split": split,
        "anchor": {"text": anchor_text, "bytes": anchor_bytes},
        "positives": positives
    }

def main(config):
    output_path = Path(OUTPUT_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(INPUT_PATH, encoding="utf-8") as fin, open(OUTPUT_PATH, "w", encoding="utf-8") as fout:
        for line in fin:
            record = json.loads(line)
            merged = merge_wikidata_record(record, config)
            fout.write(json.dumps(merged, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    import yaml
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    main(config)