"""
Merge new 1M-entity stage-3 shards with original 04_dataset.jsonl.

Reads:
  --original     04_dataset.jsonl           (119K already-processed entities)
  --new_stage3   glob pattern or single file of 03_shard_*.jsonl files
                 (1M new entities, output of 03_perturb_scripts.py shards)
  --config       config.yaml (for split ratios)

Writes:
  --output       04_dataset.jsonl (overwritten with ~1.12M combined entities)

Split assignment uses deterministic MD5 hash on entity_id — existing entities
keep their original split automatically.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
from pathlib import Path

from tqdm.auto import tqdm


def to_bytes(text: str) -> list[int]:
    return list(text.encode("utf-8"))


def assign_split(entity_id: str, train: float, val: float) -> str:
    bucket = int(hashlib.md5(entity_id.encode()).hexdigest(), 16) % 1000
    if bucket < train * 1000:
        return "train"
    elif bucket < (train + val) * 1000:
        return "val"
    return "test"


def merge_stage3_record(record: dict, split_train: float, split_val: float) -> dict:
    """Convert a 03_perturbed_scripts.jsonl record into 04_dataset.jsonl format."""
    anchor_text = record["name_en"]
    anchor_bytes = to_bytes(anchor_text)

    positives = []

    for name in record.get("latin_variants", []):
        if name:
            positives.append({
                "text": name,
                "bytes": to_bytes(name),
                "type": "phonetic",
                "source": "llm",
            })

    for name, scripts in record.get("script_variants", {}).items():
        typ = "script" if name == record["name_en"] else "combined"
        for script_key, text in scripts.items():
            if text:
                positives.append({
                    "text": text,
                    "bytes": to_bytes(text),
                    "type": typ,
                    "script": script_key,
                    "source": "llm",
                })

    for key, value in record.get("wikidata", {}).items():
        if value:
            positives.append({
                "text": value,
                "bytes": to_bytes(value),
                "type": "script",
                "script": key.replace("name_", ""),
                "source": "wikidata",
            })

    # deduplicate, wikidata wins on collision
    seen: dict[str, int] = {}
    deduped = []
    for p in positives:
        key = p["text"].strip().lower()
        if key not in seen:
            deduped.append(p)
            seen[key] = len(deduped) - 1
        else:
            if p["source"] == "wikidata":
                deduped[seen[key]] = p

    split = assign_split(record["entity_id"], split_train, split_val)
    return {
        "entity_id": record["entity_id"],
        "split": split,
        "anchor": {"text": anchor_text, "bytes": anchor_bytes},
        "positives": deduped,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original",   default="data/pipeline/04_dataset.jsonl",
                        help="existing 04_dataset.jsonl (119K entities)")
    parser.add_argument("--new_stage3", default="data/pipeline/03_shard_*.jsonl",
                        help="glob for new stage-3 shard files")
    parser.add_argument("--output",     default="data/pipeline/04_dataset.jsonl",
                        help="output path (will overwrite --original if same)")
    parser.add_argument("--config",     default="src/llm_generator_pipeline/config.yaml")
    args = parser.parse_args()

    import yaml
    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    split_train = config["split_train"]
    split_val   = config["split_val"]

    # Collect new shard files
    shard_files = sorted(glob.glob(args.new_stage3))
    if not shard_files:
        raise FileNotFoundError(f"No files matched: {args.new_stage3}")
    print(f"New shard files: {shard_files}")

    # Count for progress
    original_count = sum(1 for _ in open(args.original, encoding="utf-8"))
    new_count = sum(
        sum(1 for _ in open(f, encoding="utf-8")) for f in shard_files
    )
    print(f"Original entities: {original_count:,} | New entities: {new_count:,} | Total: {original_count + new_count:,}")

    # Write to a temp path first if output == original to avoid clobbering mid-write
    out_path = Path(args.output)
    tmp_path = out_path.with_suffix(".tmp")

    with open(tmp_path, "w", encoding="utf-8") as out:
        # Pass-through original records unchanged
        with open(args.original, encoding="utf-8") as f:
            for line in tqdm(f, total=original_count, desc="Copying original"):
                out.write(line)

        # Convert and write new stage-3 records
        for shard_file in shard_files:
            shard_count = sum(1 for _ in open(shard_file, encoding="utf-8"))
            with open(shard_file, encoding="utf-8") as f:
                for line in tqdm(f, total=shard_count, desc=f"Merging {Path(shard_file).name}"):
                    record = json.loads(line)
                    merged = merge_stage3_record(record, split_train, split_val)
                    out.write(json.dumps(merged, ensure_ascii=False) + "\n")

    tmp_path.replace(out_path)
    total = original_count + new_count
    print(f"Done. {total:,} entities written to {out_path}")


if __name__ == "__main__":
    main()
