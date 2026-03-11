# Cross-Language Phonetic Text Alignment

**DSAIT4050 — Information Retrieval, Q3 2025/26 | TU Delft | Group 10**

This project investigates cross-script phonetic name retrieval using byte-level contrastive embeddings. The goal is to retrieve transliterated or phonetically similar names across scripts (e.g., matching "Mohamed" in Latin script to "محمد" in Arabic) without relying on script-specific preprocessing or transliteration rules.

## Approach

- Byte-level transformer encoder trained with contrastive loss (no tokenizer, script-agnostic)
- Training data: multilingual name pairs generated via LLM-based phonetic perturbation
- Evaluation: cross-script retrieval benchmarks, ablations on byte vs subword encodings

## Repository Structure

```
.
├── proposal.md / proposal.pdf   # Project proposal
├── requirements.txt
├── src/
│   ├── fetch_wikidata.py        # Fetch multilingual human name labels from Wikidata via QLever
│   └── combine_and_dedup.py     # Combine batch CSVs, deduplicate on English name
└── data/
    └── raw/                     # Batch CSVs from fetch_wikidata.py (batch_*.csv)
```

## Dataset Pipeline

### 1. Fetch raw names from Wikidata

Fetches labels for ~2M human entities across 9 scripts via the [QLever SPARQL engine](https://qlever.cs.uni-freiburg.de/wikidata). Output is batched CSVs in `data/raw/`.

```bash
# Test run (2 batches)
python src/fetch_wikidata.py --test

# Full fetch
python src/fetch_wikidata.py
```

**Scripts covered:** Latin (`en`), Cyrillic (`ru`), Arabic (`ar`), Han (`zh`), Kana/Kanji (`ja`), Hebrew (`he`), Devanagari (`hi`), Greek (`el`), Hangul (`ko`)

### 2. Combine and deduplicate

Merges all batch CSVs into a single `data/names.csv`, deduplicating on the English name column.

```bash
python src/combine_and_dedup.py             # cap at 2M rows (default)
python src/combine_and_dedup.py --no-limit  # all available data
```

Output: `data/names.csv` with columns: `entity_id, name_en, name_ru, name_ar, name_zh, name_ja, name_he, name_hi, name_el, name_ko`

## Setup

```bash
pip install -r requirements.txt
```

Requires Python 3.9+. No GPU needed for data collection.
