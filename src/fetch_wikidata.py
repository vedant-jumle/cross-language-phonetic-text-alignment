"""
Fetch multilingual name labels for human entities from Wikidata via QLever.
QLever (https://qlever.cs.uni-freiburg.de/wikidata) is a fast SPARQL engine
that handles full Q5 scans without timeouts.

Dumps one CSV per batch to data/raw/. Resumable — skips already-fetched batches.

Usage:
    python src/fetch_wikidata.py           # full fetch
    python src/fetch_wikidata.py --test    # 2 pages only
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import requests
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ENDPOINT    = "https://qlever.cs.uni-freiburg.de/api/wikidata"
BATCH_SIZE  = 9000
DELAY       = 0.5       # QLever is self-hosted — less need for long delays
MAX_RETRIES = 5
OUT_DIR     = Path("data/raw")

HEADERS = {
    "Accept": "application/sparql-results+json",
    "User-Agent": "IR-NameProject/1.0 (student research, TU Delft; contact via GitHub)",
}

# QLever requires explicit prefixes (no Wikidata shorthand)
QUERY_TEMPLATE = """
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT DISTINCT ?person
  ?personLabel_en ?personLabel_ru ?personLabel_ar ?personLabel_zh
  ?personLabel_ja ?personLabel_he ?personLabel_hi ?personLabel_el ?personLabel_ko
WHERE {{
  ?person wdt:P31 wd:Q5 .
  ?person rdfs:label ?personLabel_en . FILTER(LANG(?personLabel_en) = "en")
  OPTIONAL {{ ?person rdfs:label ?personLabel_ru . FILTER(LANG(?personLabel_ru) = "ru") }}
  OPTIONAL {{ ?person rdfs:label ?personLabel_ar . FILTER(LANG(?personLabel_ar) = "ar") }}
  OPTIONAL {{ ?person rdfs:label ?personLabel_zh . FILTER(LANG(?personLabel_zh) = "zh") }}
  OPTIONAL {{ ?person rdfs:label ?personLabel_ja . FILTER(LANG(?personLabel_ja) = "ja") }}
  OPTIONAL {{ ?person rdfs:label ?personLabel_he . FILTER(LANG(?personLabel_he) = "he") }}
  OPTIONAL {{ ?person rdfs:label ?personLabel_hi . FILTER(LANG(?personLabel_hi) = "hi") }}
  OPTIONAL {{ ?person rdfs:label ?personLabel_el . FILTER(LANG(?personLabel_el) = "el") }}
  OPTIONAL {{ ?person rdfs:label ?personLabel_ko . FILTER(LANG(?personLabel_ko) = "ko") }}
}}
ORDER BY ?person
LIMIT {limit}
OFFSET {offset}
"""

# en=Latin, ru=Cyrillic, ar=Arabic, zh=Han, ja=Kana/Kanji, he=Hebrew,
# hi=Devanagari, el=Greek, ko=Hangul
CSV_COLUMNS = ["entity_id", "name_en", "name_ru", "name_ar", "name_zh",
               "name_ja", "name_he", "name_hi", "name_el", "name_ko"]


# ---------------------------------------------------------------------------
# SPARQL helpers
# ---------------------------------------------------------------------------

def execute_query(session: requests.Session, query: str) -> list[dict]:
    """POST a SPARQL query and return the list of binding dicts."""
    for attempt in range(MAX_RETRIES):
        try:
            resp = session.post(
                ENDPOINT,
                data={"query": query},
                timeout=120,
            )
        except requests.exceptions.RequestException as e:
            wait = 2 ** attempt
            print(f"\nRequest error: {e}. Retrying in {wait}s...")
            time.sleep(wait)
            continue

        if resp.status_code == 200:
            return resp.json()["results"]["bindings"]

        if resp.status_code == 429:
            wait = int(resp.headers.get("Retry-After", 60))
            print(f"\nRate limited (429). Sleeping {wait}s...")
            time.sleep(wait)
            continue

        if resp.status_code >= 500:
            wait = 2 ** attempt
            print(f"\nServer error {resp.status_code}. Retrying in {wait}s...")
            time.sleep(wait)
            continue

        resp.raise_for_status()

    raise RuntimeError(f"SPARQL query failed after {MAX_RETRIES} retries.")


def parse_bindings(bindings: list[dict]) -> list[dict]:
    """Convert SPARQL JSON bindings to flat row dicts."""
    rows = []
    for b in bindings:
        rows.append({
            "entity_id": b["person"]["value"].split("/")[-1],
            "name_en":   b.get("personLabel_en", {}).get("value", ""),
            "name_ru":   b.get("personLabel_ru", {}).get("value", ""),
            "name_ar":   b.get("personLabel_ar", {}).get("value", ""),
            "name_zh":   b.get("personLabel_zh", {}).get("value", ""),
            "name_ja":   b.get("personLabel_ja", {}).get("value", ""),
            "name_he":   b.get("personLabel_he", {}).get("value", ""),
            "name_hi":   b.get("personLabel_hi", {}).get("value", ""),
            "name_el":   b.get("personLabel_el", {}).get("value", ""),
            "name_ko":   b.get("personLabel_ko", {}).get("value", ""),
        })
    return rows


# ---------------------------------------------------------------------------
# Resume logic
# ---------------------------------------------------------------------------

def get_resume_offset(out_dir: Path) -> int:
    """Return the next offset to fetch based on existing batch files."""
    existing = sorted(out_dir.glob("batch_*.csv"))
    if not existing:
        return 0
    # Parse offset from filename: batch_0000009000.csv -> 9000
    last_offset = int(existing[-1].stem.split("_")[1])
    return last_offset + BATCH_SIZE


def batch_path(out_dir: Path, offset: int) -> Path:
    return out_dir / f"batch_{offset:010d}.csv"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true",
                        help="Fetch only 2 pages then stop")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    start_offset = get_resume_offset(OUT_DIR)
    if start_offset > 0:
        print(f"Resuming from offset {start_offset:,} "
              f"({start_offset // BATCH_SIZE} batches already done)")
    else:
        print("Starting fresh fetch from QLever (Wikidata)...")

    session = requests.Session()
    session.headers.update(HEADERS)

    offset = start_offset
    max_pages = 2 if args.test else None
    pages_fetched = 0

    pbar = tqdm(desc="Batches fetched", unit="batch")

    while True:
        if max_pages is not None and pages_fetched >= max_pages:
            print(f"\n--test mode: stopping after {pages_fetched} pages.")
            break

        query = QUERY_TEMPLATE.format(limit=BATCH_SIZE, offset=offset)
        bindings = execute_query(session, query)

        if not bindings:
            print("\nEmpty response — fetch complete.")
            break

        rows = parse_bindings(bindings)
        out_path = batch_path(OUT_DIR, offset)

        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)

        pages_fetched += 1
        pbar.update(1)
        pbar.set_postfix({"offset": offset, "rows": len(rows), "file": out_path.name})

        if len(bindings) < BATCH_SIZE:
            print(f"\nLast page ({len(bindings)} rows). Fetch complete.")
            break

        offset += BATCH_SIZE
        time.sleep(DELAY)

    pbar.close()
    total_batches = len(sorted(OUT_DIR.glob("batch_*.csv")))
    print(f"\nDone. {total_batches} batch files in {OUT_DIR}/")


if __name__ == "__main__":
    main()
