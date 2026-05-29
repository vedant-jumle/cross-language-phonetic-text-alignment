#!/bin/bash
#SBATCH --job-name=ir_regen_test
#SBATCH --array=0-9
#SBATCH --partition=compute-p1
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2G
#SBATCH --account=Education-EEMCS-MSc-DSAIT
#SBATCH --output=/scratch/vvjumle/logs/ir_regen_test_%A_%a.out
#SBATCH --error=/scratch/vvjumle/logs/ir_regen_test_%A_%a.err

set -e

REPO=/scratch/vvjumle/ByteTransformers

mkdir -p /scratch/vvjumle/logs
mkdir -p $REPO/data/pipeline

module load 2025
module load miniconda3
conda activate ir-pipeline

cd $REPO

echo "=== Regen test (chat endpoint, shard ${SLURM_ARRAY_TASK_ID}/10) ==="
python src/llm_generator_pipeline/05_regen_test_independent.py \
    --dataset  data/pipeline/04_dataset.jsonl \
    --output   data/pipeline/05_shard_${SLURM_ARRAY_TASK_ID}.jsonl \
    --shard    ${SLURM_ARRAY_TASK_ID} \
    --n_shards 10

echo "=== Shard ${SLURM_ARRAY_TASK_ID} complete ==="
