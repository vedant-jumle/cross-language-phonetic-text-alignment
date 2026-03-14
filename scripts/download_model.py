"""
Run on the cluster LOGIN NODE before submitting the Slurm job:

    module load 2025
    module load miniconda3
    conda activate ir-pipeline
    huggingface-cli login   # enter HF token with Llama 3.1 access
    python scripts/download_model.py

Downloads Llama-3.1-8B-Instruct to /scratch/vvjumle/models/.
"""
import os
from huggingface_hub import snapshot_download

MODEL_ID = "meta-llama/Llama-3.1-70B-Instruct"
LOCAL_DIR = "/scratch/vvjumle/models/Llama-3.1-70B-Instruct"

os.makedirs(LOCAL_DIR, exist_ok=True)
print(f"Downloading {MODEL_ID} → {LOCAL_DIR}")
snapshot_download(
    repo_id=MODEL_ID,
    local_dir=LOCAL_DIR,
    ignore_patterns=["*.pt", "original/"],  # keep safetensors, skip legacy weights
)
print("Done.")
