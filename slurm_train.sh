#!/bin/bash
#SBATCH --job-name=ir_train
#SBATCH --partition=gpu-v100
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=5G
#SBATCH --account=Education-EEMCS-MSc-DSAIT
#SBATCH --output=/scratch/vvjumle/logs/ir_train_%j.out
#SBATCH --error=/scratch/vvjumle/logs/ir_train_%j.err

set -e

REPO=/scratch/vvjumle/ir-pipeline
SCRATCH_DATA=/scratch/vvjumle/ir-pipeline/data/pipeline

mkdir -p /scratch/vvjumle/logs
mkdir -p /scratch/vvjumle/ir-pipeline/checkpoints

module load 2025
module load miniconda3
conda activate ir-pipeline
module load cuda/12.9

# Ensure data/pipeline symlink exists
ln -sfn $SCRATCH_DATA $REPO/data/pipeline

cd $REPO

echo "=== Starting training ==="
PYTHONPATH=$REPO python src/model/train.py \
    --config src/llm_generator_pipeline/config_hpc.yaml

echo "=== Training complete. Checkpoints at /scratch/vvjumle/ir-pipeline/checkpoints/ ==="
