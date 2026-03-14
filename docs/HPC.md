# Running the Pipeline on DelftBlue HPC

Runs stages 02→03→04 using Llama 3.1 8B on a V100 GPU for 100k names.

## Prerequisites

- Access to DelftBlue (`ssh vvjumle@login.delftblue.tudelft.nl`)
- HuggingFace account with [Llama 3.1 access](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) approved
- HF token available at https://huggingface.co/settings/tokens

---

## One-Time Setup (on the login node)

### 1. Clone the repo to scratch

```bash
git clone <repo-url> /scratch/vvjumle/ir-pipeline
cd /scratch/vvjumle/ir-pipeline
```

### 2. Copy the input data (not in git — too large)

From your **local machine**:
```bash
scp data/pipeline/01_sampled.jsonl \
    vvjumle@login.delftblue.tudelft.nl:/scratch/vvjumle/ir-pipeline/data/pipeline/
```

### 3. Download the model

On the **login node** (not a compute node):
```bash
module load 2025
module load miniconda3
conda env create -f /scratch/vvjumle/ir-pipeline/environment.yml
conda activate ir-pipeline

huggingface-cli login   # paste your HF token when prompted

python /scratch/vvjumle/ir-pipeline/scripts/download_model.py
```

This downloads ~16GB to `/scratch/vvjumle/models/Llama-3.1-8B-Instruct/`. Takes ~10 minutes on the login node.

---

## Running the Job

```bash
cd /scratch/vvjumle/ir-pipeline
sbatch slurm_pipeline.sh
```

### Monitor

```bash
squeue -u vvjumle
tail -f /scratch/vvjumle/logs/ir_pipeline_<jobid>.out
```

### Output

`/scratch/vvjumle/ir-pipeline/data/pipeline/04_dataset.jsonl`

---

## Troubleshooting

**OOM on V100 (16GB):** Reduce `batch_size` in `config.yaml` from 32 → 16.

**Model not found:** Verify the path in `config.yaml` matches where `download_model.py` saved it:
```bash
ls /scratch/vvjumle/models/Llama-3.1-8B-Instruct/
```

**Conda env already exists but broken:** Remove and recreate:
```bash
conda env remove -n ir-pipeline
conda env create -f environment.yml
```

**Job hits 24h wall time before finishing:** Stages 02 and 03 are resumable — they skip already-processed entity IDs on restart. Just resubmit with `sbatch slurm_pipeline.sh`.
