# LLM Generator Pipeline — Design Spec

**Date:** 2026-03-11
**Project:** Cross-Script Phonetic Name Retrieval (DSAIT4050/Q3, Group 10)

---

## Overview

A staged, resumable pipeline that takes the raw 2M-row Wikidata name dataset and produces a fully preprocessed training dataset of name identities with their positive variants (phonetic + cross-script), ready for byte-level contrastive training. No on-the-fly preprocessing during training.

---

## Pipeline Stages

```
data/names.csv
  → 01_sample.py          → data/pipeline/01_sampled.jsonl
  → 02_perturb_latin.py   → data/pipeline/02_perturbed_latin.jsonl
  → 03_perturb_scripts.py → data/pipeline/03_perturbed_scripts.jsonl
  → 04_merge_wikidata.py  → data/pipeline/04_dataset.jsonl
```

Each stage is independently resumable — it checks which `entity_id`s are already in the output file and skips them.

---

## File Structure

```
src/
  llm_generator_pipeline/
    01_sample.py
    02_perturb_latin.py
    03_perturb_scripts.py
    04_merge_wikidata.py
    config.yaml

data/
  names.csv
  pipeline/
    01_sampled.jsonl
    02_perturbed_latin.jsonl
    03_perturbed_scripts.jsonl
    04_dataset.jsonl

docs/
  2026-03-11-llm-generator-pipeline-design.md
```

---

## Configuration (`config.yaml`)

```yaml
# Sampling
sample_size: 50000
sample_seed: 42

# LLM
llm_model: "llama3.1"           # any Ollama model name
llm_base_url: "http://localhost:11434"
batch_size: 20                  # names per LLM prompt

# Perturbation
n_perturbations: 4              # phonetic Latin variants per name
target_scripts:                 # scripts for phase 2 transliteration
  - ar
  - ru
  - zh
  - ja
  - he
  - hi
  - el
  - ko

# Dataset split ratios
split_train: 0.8
split_val: 0.1
split_test: 0.1

# Hard negative mining (used by training, not pipeline)
hard_negative_warmup_steps: 1000
hard_negative_mix_ratio: 0.5    # fraction of hard vs easy negatives per batch
```

---

## Stage 1 — Sampler (`01_sample.py`)

**Input:** `data/names.csv`
**Output:** `data/pipeline/01_sampled.jsonl`

### Sampling Strategy

Stratified by script coverage bucket:
- Bucket 0: no non-English labels
- Bucket 1–2: 1–2 non-English labels
- Bucket 3–4: 3–4 non-English labels
- Bucket 5+: 5 or more non-English labels

Sample proportionally within each bucket so the sample is not dominated by English-only names. Controlled by `--seed`.

### Output Record

```json
{
  "entity_id": "Q12345",
  "name_en": "Catherine",
  "wikidata": {
    "name_ru": "Екатерина",
    "name_ar": "كاثرين",
    "name_zh": "",
    "name_ja": "",
    "name_he": "",
    "name_hi": "",
    "name_el": "",
    "name_ko": ""
  }
}
```

---

## Stage 2 — Latin Perturbations (`02_perturb_latin.py`)

**Input:** `data/pipeline/01_sampled.jsonl`
**Output:** `data/pipeline/02_perturbed_latin.jsonl`

### LLM Prompt

```
Given these English names, generate {n_perturbations} realistic phonetic
spelling variants for each. Variants should sound similar when spoken aloud
but differ in spelling. Do NOT generate: nicknames, abbreviations, or
shortened forms.

Names: ["Catherine", "John", "Mohammed", ...]

Return JSON only: {"Catherine": ["Kathryn", "Katerin", ...], "John": [...], ...}
```

Batches `batch_size` names per prompt. Parses JSON response, retries on malformed output (up to 3 attempts).

### Checkpointing

Tracks processed `entity_id`s in output file. On restart, skips already-processed names. Flushes to disk after each batch.

### Output Record

```json
{
  "entity_id": "Q12345",
  "name_en": "Catherine",
  "wikidata": { "name_ru": "Екатерина", ... },
  "latin_variants": ["Kathryn", "Katerin", "Kathrin", "Katharine"]
}
```

---

## Stage 3 — Script Transliteration (`03_perturb_scripts.py`)

**Input:** `data/pipeline/02_perturbed_latin.jsonl`
**Output:** `data/pipeline/03_perturbed_scripts.jsonl`

### LLM Prompt

```
Transliterate each of these names into the following scripts: {scripts}.
Use phonetic transliteration only — how the name sounds, not its meaning.

Names: ["Catherine", "Kathryn", "Katerin", "Kathrin"]

Return JSON only:
{
  "Catherine": {"ar": "كاثرين", "ru": "Катрин", ...},
  "Kathryn":   {"ar": "...", "ru": "...", ...},
  ...
}
```

Processes `name_en` + all `latin_variants` for each identity. Same checkpoint/resume logic as Stage 2.

### Output Record

```json
{
  "entity_id": "Q12345",
  "name_en": "Catherine",
  "wikidata": { "name_ru": "Екатерина", ... },
  "latin_variants": ["Kathryn", "Katerin", "Kathrin", "Katharine"],
  "script_variants": {
    "Catherine": {"ar": "كاثرين", "ru": "Катрин", "he": "קתרין", ...},
    "Kathryn":   {"ar": "كاثرين", "ru": "Катрин", ...},
    "Katerin":   {"ar": "...", "ru": "...", ...},
    "Kathrin":   {"ar": "...", "ru": "...", ...},
    "Katharine": {"ar": "...", "ru": "...", ...}
  }
}
```

---

## Stage 4 — Merge & Preprocess (`04_merge_wikidata.py`)

**Input:** `data/pipeline/03_perturbed_scripts.jsonl`
**Output:** `data/pipeline/04_dataset.jsonl`

### Processing Steps

1. **Merge positives** — combine Wikidata ground truth labels + LLM-generated variants into one unified positive set per identity
2. **Deduplicate** — remove duplicate strings across sources (case-insensitive, strip whitespace)
3. **Tag each positive** with:
   - `source`: `"wikidata"` or `"llm"`
   - `type`: `"phonetic"` (Latin variant), `"script"` (cross-script), or `"combined"` (phonetic variant that was then transliterated)
4. **Precompute UTF-8 byte sequences** — store as list of ints for every string
5. **Assign split** — train/val/test strictly at identity level; all variants of a name go to one split only. Split is deterministic (hash of `entity_id` mod 1000)

### Final Record

```json
{
  "entity_id": "Q12345",
  "split": "train",
  "anchor": {
    "text": "Catherine",
    "bytes": [67, 97, 116, 104, 101, 114, 105, 110, 101]
  },
  "positives": [
    {"text": "Kathryn",   "bytes": [...], "type": "phonetic",  "source": "llm"},
    {"text": "Kathrine",  "bytes": [...], "type": "phonetic",  "source": "llm"},
    {"text": "كاثرين",    "bytes": [...], "type": "script",    "source": "llm"},
    {"text": "Екатерина", "bytes": [...], "type": "script",    "source": "wikidata"},
    {"text": "كاثرين",    "bytes": [...], "type": "combined",  "source": "llm"}
  ]
}
```

Negatives are **not stored** — mined dynamically during training (random for first `hard_negative_warmup_steps`, then async ANN refresh with `hard_negative_mix_ratio` hard/easy mix).

---

## Hard Negative Mining Strategy (Training Time)

Not part of this pipeline, but documented here for completeness:

- **Phase 1 (steps 0 → warmup):** Random negatives sampled from the full dataset
- **Phase 2 (steps > warmup):** Async FAISS index refresh — model embeddings used to find nearest neighbors as hard negatives; index refreshed periodically in background
- **Within each batch:** Mix of hard and easy negatives controlled by `hard_negative_mix_ratio` to prevent loss collapse

False negatives (same identity appearing as a negative) are filtered using `entity_id` ground truth.

---

## Resumability Contract

Every stage that calls the LLM:
1. Reads existing output file on startup, builds a set of already-processed `entity_id`s
2. Skips those IDs in the input
3. Appends to output file (never overwrites)
4. Flushes after every batch

This means a crash loses at most one in-flight batch of LLM calls.
