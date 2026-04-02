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
    02_perturb_combined.py   # v0.1 POC only — Anthropic Batch API, not used in v2
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
# Sampling — set large; if >= population, returns full dataset
sample_size: 1000000
sample_seed: 42

# LLM (Stage 2 — HF inference on DelftBlue HPC)
hf_model_id: "meta-llama/Llama-3.1-8B-Instruct"
batch_size: 2                   # names per model.generate() call (HF inference)
                                # 8B on 1x V100S 32GB: batch_size=64 on HPC

# Anthropic Batch API (02_perturb_combined.py — v0.1 POC only, not used in v2)
anthropic_api_key: ""           # falls back to ANTHROPIC_API_KEY env var
max_names: 100000

# Perturbation
n_perturbations: 4              # phonetic Latin variants per name (Stage 2)
target_scripts:                 # scripts for Stage 3 transliteration
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
hard_negative_warmup_steps: 200
hard_negative_mix_ratio: 0.7    # fraction of hard vs easy negatives per batch
hard_negative_mix_ramp_steps: 500
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
**Infrastructure:** TU Delft DelftBlue HPC (A100 80GB / V100S 32GB)
**Model:** `meta-llama/Llama-3.1-8B-Instruct` via HuggingFace `call_hf_batch`

### LLM Prompt

One name per prompt call (not batched at prompt level — batched at inference level via `call_hf_batch`):

```
Generate {n_perturbations} DISTINCT phonetic spelling variants of this name
as it sounds when spoken: "{name}"

Rules:
- Each variant must be spelled differently from all others and from the original
- Simulate how different people might mishear or misspell the name phonetically
- Keep the same number of words as the original
- Do NOT repeat variants
- Do NOT use nicknames, abbreviations, or shortened forms
- Do NOT change language (stay in Latin script)

Example: "Catherine" → ["Kathryn", "Katherin", "Cathryn", "Katheryne"]

Return a JSON array of exactly {n_perturbations} strings, no explanation:
["variant1", "variant2", ...]
```

Processes `batch_size` names per `model.generate()` call. Parses JSON array response; falls back to dict-value extraction on malformed output. Checkpointed — skips already-processed `entity_id`s on restart, flushes after each batch.

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
**Infrastructure:** TU Delft TULIP API (`https://api.tulip.tudelft.nl/code/v1/`)
**Model:** `Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8` (accessed as model `"code"` via OpenAI-compatible client)
**Concurrency:** 50 threads via `ThreadPoolExecutor`

### LLM Prompt

One name per API call; structured JSON output enforced via `json_schema` response format:

```
Transliterate "{name}" into {target_scripts} phonetically.
```

Response format schema enforces a JSON object with one key per target script (e.g. `{"ar": "...", "ru": "...", ...}`). No free-text parsing needed — model returns validated JSON directly.

Processes `name_en` + all `latin_variants` for each identity as independent concurrent requests. Writes a record only after all names for that entity complete. Checkpointed — skips already-processed `entity_id`s on restart.

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
