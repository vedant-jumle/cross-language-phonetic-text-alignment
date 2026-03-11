"""
Combine all batch CSVs from data/raw/ into a single data/names.csv,
deduplicating on name_en (keep first occurrence).

Usage:
    python src/combine_and_dedup.py                  # up to 2M rows
    python src/combine_and_dedup.py --limit 500000   # custom row limit
    python src/combine_and_dedup.py --no-limit       # all batches
"""

import argparse
import csv
from pathlib import Path

from tqdm import tqdm

RAW_DIR = Path("data/raw")
OUT_FILE = Path("data/names.csv")
DEFAULT_LIMIT = 2_000_000

COLUMNS = ["entity_id", "name_en", "name_ru", "name_ar", "name_zh",
           "name_ja", "name_he", "name_hi", "name_el", "name_ko"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--no-limit", action="store_true")
    args = parser.parse_args()

    row_limit = None if args.no_limit else args.limit

    batch_files = sorted(RAW_DIR.glob("batch_*.csv"))
    print(f"Found {len(batch_files)} batch files")

    seen_en = set()
    rows_written = 0
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(OUT_FILE, "w", newline="", encoding="utf-8") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=COLUMNS)
        writer.writeheader()

        for batch_path in tqdm(batch_files, desc="Processing batches"):
            if row_limit is not None and rows_written >= row_limit:
                break

            with open(batch_path, newline="", encoding="utf-8") as in_f:
                reader = csv.DictReader(in_f)
                for row in reader:
                    if row_limit is not None and rows_written >= row_limit:
                        break
                    name_en = row.get("name_en", "").strip()
                    if not name_en or name_en in seen_en:
                        continue
                    seen_en.add(name_en)
                    writer.writerow({col: row.get(col, "") for col in COLUMNS})
                    rows_written += 1

    print(f"Done. {rows_written:,} unique rows written to {OUT_FILE}")


if __name__ == "__main__":
    main()
