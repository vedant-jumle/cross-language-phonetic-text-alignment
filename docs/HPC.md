# Running the Pipeline on DelftBlue HPC

The data pipeline (stages 2–4) and model training run on TU Delft's DelftBlue supercomputer.

- **Stage 2** — Latin phonetic variants: `Llama-3.1-8B-Instruct` on 1× V100S 32GB
- **Stage 3** — Cross-script transliteration: `Qwen3-Coder-30B-A3B-Instruct-FP8` via TU Delft TULIP API (no GPU needed, 50 concurrent threads)
- **Stage 4** — Merge & preprocess: CPU only, runs locally or on HPC
- **Training** — ByteLevelEncoder: 1× V100S 32GB, 24h wall time

---

## Prerequisites

- Access to DelftBlue (`ssh <netid>@login.delftblue.tudelft.nl`)
- HuggingFace account with [Llama 3.1 access](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) approved + HF token
- TU Delft TULIP API key (request via the TU Delft DHPC portal or your supervisor)

---

## One-Time Setup (on the login node)

### 1. Clone the repo to scratch

```bash
git clone https://github.com/vedant-jumle/cross-language-phonetic-text-alignment /scratch/<netid>/ir-pipeline
cd /scratch/<netid>/ir-pipeline
```

### 2. Copy input data (not in git — too large)

From your **local machine**, after running stage 1 locally:

```bash
scp data/pipeline/01_sampled.jsonl \
    <netid>@login.delftblue.tudelft.nl:/scratch/<netid>/ir-pipeline/data/pipeline/
```

### 3. Set up the conda environment

```bash
module load 2025
module load miniconda3
conda env create -f environment.yml
conda activate ir-pipeline
```

### 4. Configure credentials

```bash
# HuggingFace token (for Llama download in stage 2)
huggingface-cli login

# TULIP API key (for stage 3)
export TULIP_API_KEY=your_key_here
# Add to ~/.bashrc or pass via the Slurm script environment
```

### 5. Download the Llama model (stage 2 only)

```bash
python scripts/download_model.py
```

Downloads ~16GB to `/scratch/<netid>/models/Llama-3.1-8B-Instruct/`. Takes ~10 minutes.

---

## Running the Data Pipeline (Stages 2–4)

```bash
cd /scratch/<netid>/ir-pipeline
sbatch slurm_data_pipeline.sh
```

This runs stages 2, 3, and 4 sequentially. Each stage is **resumable** — if the job hits the wall time limit, resubmit and it will skip already-processed entity IDs.

### Monitor

```bash
squeue -u <netid>
tail -f /scratch/<netid>/logs/ir_pipeline_<jobid>.out
```

### Output

| Stage | Output file |
|---|---|
| Stage 2 | `data/pipeline/02_perturbed_latin.jsonl` |
| Stage 3 | `data/pipeline/03_perturbed_scripts.jsonl` |
| Stage 4 | `data/pipeline/04_dataset.jsonl` (final training dataset) |

---

## Running Training

```bash
sbatch slurm_train.sh
```

**Resources:** 1× V100S 32GB GPU, 24h wall time.

Checkpoint saved to `checkpoints/best_v2/` after training completes. Copy back to local machine:

```bash
scp -r <netid>@login.delftblue.tudelft.nl:/scratch/<netid>/ir-pipeline/checkpoints/best_v2 ./checkpoints/
```

---

## Stage-Specific Notes

### Stage 2 — Latin perturbations (Llama-3.1-8B-Instruct)

- Runs on 1× V100S 32GB. `batch_size: 2` in `config.yaml` is correct for this GPU.
- If OOM, reduce `batch_size` to 1 in `config.yaml`.
- Processes ~960K entities; estimated wall time ~18h.

### Stage 3 — Script transliteration (TULIP API)

- No GPU needed. Runs on a CPU node or the login node.
- Uses 50 concurrent threads (`ThreadPoolExecutor`) against the TULIP API.
- Model: `Qwen3-Coder-30B-A3B-Instruct-FP8`, accessed as model `"code"` via OpenAI-compatible endpoint at `https://api.tulip.tudelft.nl/code/v1/`.
- If the API is rate-limited, reduce concurrency in `llm_client.py`.
- Estimated wall time: ~8h for 119K entities × 5 names each.

### Stage 4 — Merge & preprocess

- CPU only, finishes in minutes.
- Can run locally after copying stage 3 output back.

---

## Troubleshooting

**Model not found (stage 2):** Verify the path in `config.yaml` matches where the model was downloaded:
```bash
ls /scratch/<netid>/models/Llama-3.1-8B-Instruct/
```

**TULIP API errors (stage 3):** Check your API key is exported in the Slurm script environment. The TULIP endpoint requires TU Delft network access (VPN or on-campus).

**Conda env broken:** Remove and recreate:
```bash
conda env remove -n ir-pipeline
conda env create -f environment.yml
```

**Job hits wall time:** Stages 2 and 3 are resumable. Just resubmit:
```bash
sbatch slurm_data_pipeline.sh
```
