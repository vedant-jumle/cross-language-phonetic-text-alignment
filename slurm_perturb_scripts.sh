#!/bin/bash
#SBATCH --job-name=ir_pipeline
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=5G
#SBATCH --account=Education-EEMCS-MSc-DSAIT
#SBATCH --output=/scratch/vvjumle/logs/ir_pipeline_%j.out
#SBATCH --error=/scratch/vvjumle/logs/ir_pipeline_%j.err

set -e

REPO=/scratch/vvjumle/ir-pipeline
SCRATCH_DATA=/scratch/vvjumle/ir-pipeline/data/pipeline

mkdir -p /scratch/vvjumle/logs
mkdir -p $SCRATCH_DATA

module load 2025
module load miniconda3
conda activate ir-pipeline
module load cuda/12.9

# Make sure data/pipeline points to scratch (symlink)
ln -sfn $SCRATCH_DATA $REPO/data/pipeline

cd $REPO

echo "=== Stage 03: Script transliterations ==="
python src/llm_generator_pipeline/03_perturb_scripts.py \
    --config src/llm_generator_pipeline/config.yaml

