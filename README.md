# Cross-Script Phonetic Name Retrieval via Byte-Level Contrastive Embeddings

**DSAIT4050 — Information Retrieval, Q3 2025/26 | TU Delft | Group 10** ![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

Retrieving person names across writing systems is a fundamental challenge in multilingual IR: a query like "Schwarzenegger" should match "שוורצנגר" (Hebrew) or "Владимир" (Cyrillic), despite sharing no characters. This project trains a byte-level transformer encoder from scratch — no tokenizer, no pretrained backbone, no script detection — to solve this problem.

**Results:** 0.775 MRR, 0.897 R@10 overall. Script gap (R@10 difference between Latin and non-Latin queries) reduced from 0.93 (edit-distance baselines) to 0.096 — a 10× reduction. Outperforms the strongest baseline (ICU Transliterate) by 37%.

---

## Repository Structure

```
project/
├── src/
│   ├── fetch_wikidata.py                  # SPARQL fetch of 2M person names from Wikidata via QLever
│   ├── combine_and_dedup.py               # Merge batch CSVs into data/names.csv
│   ├── llm_generator_pipeline/
│   │   ├── 01_sample.py                   # Stratified sampling from names.csv
│   │   ├── 02_perturb_latin.py            # LLM phonetic variant generation (Llama-3.1-8B)
│   │   ├── 03_perturb_scripts.py          # Cross-script transliteration (Qwen3-30B via TULIP API)
│   │   ├── 04_merge_wikidata.py           # Merge, deduplicate, tag, precompute bytes, split
│   │   ├── dataloader.py                  # ANCE batch sampler + InfoNCE dataloader
│   │   ├── dataset.py                     # NameDataset with precomputed UTF-8 bytes
│   │   ├── llm_client.py                  # TULIP API client (ThreadPoolExecutor)
│   │   └── config.py                      # Config loader
│   ├── model/
│   │   ├── encoder.py                     # ByteLevelEncoder (TransformerEncoder backbone)
│   │   ├── train.py                       # Training loop with hard-negative mining
│   │   ├── loss.py                        # InfoNCE loss
│   │   ├── encode_all.py                  # Batch inference for FAISS indexing
│   │   └── config.py                      # ByteEncoderConfig (HF PretrainedConfig)
│   └── eval/
│       ├── evaluation.py                  # Main IR evaluation pipeline
│       ├── model_retriever.py             # Model inference + FAISS integration
│       ├── faiss_ablation.py              # 4-index comparison (FlatIP, IVF-Flat, HNSW, IVF-PQ)
│       └── baselines/
│           ├── levenshtein.py
│           ├── soundex.py                 # Double Metaphone implementation
│           ├── bm25.py
│           └── transliterate.py           # ICU Any-Latin; Latin-ASCII
├── data/
│   ├── names.csv                          # 2M raw entities from Wikidata (not in git)
│   └── pipeline/
│       ├── 01_sampled.jsonl               # 119,040 stratified entities
│       ├── 02_perturbed_latin.jsonl       # + phonetic Latin variants
│       ├── 03_perturbed_scripts.jsonl     # + cross-script transliterations
│       └── 04_dataset.jsonl               # Final training dataset (1.1GB, not in git)
├── checkpoints/
│   └── best_v2/                           # Trained model checkpoint (not in git)
├── results/                               # JSON evaluation results
├── notebooks/
│   └── eda.ipynb                          # Exploratory data analysis
├── docs/                                  # Design specs and analysis
├── slurm_data_pipeline.sh                 # HPC job: stages 2–4
├── slurm_train.sh                         # HPC job: model training
└── requirements.txt
```

---

## Quick Start — Reproduce Results

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

Requires Python 3.10+.

### 2. Download the trained checkpoint

```bash
python scripts/download_model.py
```

This downloads `checkpoints/best_v2/` (~112MB).

### 3. Run evaluation

```bash
python src/eval/evaluation.py \
    --checkpoint checkpoints/best_v2 \
    --dataset data/pipeline/04_dataset.jsonl \
    --split test
```

Results are written to `results/`.

### 4. Run FAISS index ablation

```bash
python src/eval/faiss_ablation.py \
    --checkpoint checkpoints/best_v2 \
    --dataset data/pipeline/04_dataset.jsonl
```

---

## Reproducing the Dataset

The full dataset pipeline runs in 4 stages. Stages 2 and 3 require HPC access; see [`docs/HPC.md`](docs/HPC.md) for setup.

### Stage 1 — Wikidata fetch + sampling (local, CPU)

```bash
# Fetch 2M names from Wikidata (takes ~1h)
python src/fetch_wikidata.py

# Merge batch CSVs
python src/combine_and_dedup.py

# Stratified sample (produces data/pipeline/01_sampled.jsonl)
python src/llm_generator_pipeline/01_sample.py
```

### Stage 2 — Latin phonetic variants (HPC, GPU)

Requires DelftBlue access. Runs `Llama-3.1-8B-Instruct` locally on a V100S 32GB GPU.

```bash
sbatch slurm_data_pipeline.sh
```

Or locally (slow):

```bash
python src/llm_generator_pipeline/02_perturb_latin.py
```

### Stage 3 — Cross-script transliteration (TULIP API)

Requires a TU Delft TULIP API key. Set the environment variable:

```bash
export TULIP_API_KEY=your_key_here
python src/llm_generator_pipeline/03_perturb_scripts.py
```

Uses `Qwen3-Coder-30B-A3B-Instruct-FP8` with 50 concurrent threads.

### Stage 4 — Merge, tag, split (local, CPU)

```bash
python src/llm_generator_pipeline/04_merge_wikidata.py
```

Produces `data/pipeline/04_dataset.jsonl` — the final training dataset (119,040 entities, 4.67M positive pairs).

---

## Reproducing Training

Training runs on DelftBlue with a single V100S 32GB GPU (24h wall time).

```bash
sbatch slurm_train.sh
```

Or locally:

```bash
python src/model/train.py --config src/llm_generator_pipeline/config.yaml
```

**Architecture:** 6-layer transformer encoder, 8 attention heads, hidden dim 256, FFN dim 1024, max length 256 bytes, dropout 0.1.

**Training:** InfoNCE loss, AdamW optimizer, cosine decay scheduler, ANCE-style hard negative mining (warmup 200 steps, then async FAISS index refresh with 70% hard / 30% easy mix).

Checkpoints saved to `checkpoints/` after each validation epoch; best checkpoint by validation MRR is kept as `checkpoints/best_v2/`.

---

## Key Results

| Retriever | MRR | R@1 | R@10 | NDCG@10 |
|---|---|---|---|---|
| Levenshtein | 0.094 | 0.089 | 0.105 | 0.097 |
| Double Metaphone | 0.096 | 0.092 | 0.106 | 0.098 |
| BM25 (char trigrams) | 0.083 | 0.077 | 0.097 | 0.086 |
| Transliterate (ICU) | 0.565 | 0.511 | 0.681 | 0.592 |
| **Byte encoder (ours)** | **0.775** | **0.711** | **0.897** | **0.804** |

**By query type (MRR):**

| Query type | Transliterate | Ours |
|---|---|---|
| Phonetic (Latin variants) | 0.894 | **0.937** |
| Script (cross-script) | 0.684 | **0.827** |
| Combined (hardest) | 0.485 | **0.738** |

**Script gap** (R@10 difference, Latin vs. non-Latin): reduced from **0.931** (Levenshtein) to **0.096** (ours).

Per-script R@10: Arabic 0.967, Russian 0.974, Hebrew 0.954, Hindi 0.973, Greek 0.967, Japanese 0.870, Korean 0.728, Chinese 0.666.

---

## Dataset

- **119,040** person-name entities, stratified from 2M Wikidata entities
- **4.67M** positive name pairs
- **8 non-Latin scripts:** Arabic, Russian, Chinese, Japanese, Hebrew, Hindi, Greek, Korean
- **Pair types:** phonetic (Latin spelling variants), script (cross-script transliterations), combined (phonetic variants then transliterated)
- **0.5%** Wikidata ground truth, **99.5%** LLM-generated
- **Split:** 80/10/10 at entity level (deterministic MD5 hash), no leakage

---

## Paper

See [`report-latex/report.tex`](report-latex/report.tex) for the full ACM SIGCONF paper.

---

## Authors

- Vedant Vivek Jumle (5296196) — v.v.jumle@student.tudelft.nl
- Carolyn Alcaraz (5680581) — C.Alcaraz-1@student.tudelft.nl
- Gönenç Turanlı (5794765) — G.Turanli-1@student.tudelft.nl
- Ceylin Ece (5716950) — C.Ece-1@student.tudelft.nl
