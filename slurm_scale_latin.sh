#!/bin/bash
#SBATCH --job-name=ir_scale_latin
#SBATCH --array=0-9
#SBATCH --partition=gpu-v100
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=5G
#SBATCH --account=Education-EEMCS-MSc-DSAIT
#SBATCH --output=/scratch/vvjumle/logs/ir_scale_latin_%A_%a.out
#SBATCH --error=/scratch/vvjumle/logs/ir_scale_latin_%A_%a.err

set -e

REPO=/scratch/vvjumle/ByteTransformers

mkdir -p /scratch/vvjumle/logs
mkdir -p $REPO/data/pipeline

module load 2025
module load miniconda3
conda activate ir-pipeline
module load cuda/12.9

cd $REPO

echo "=== Stage 02: Latin phonetic variants (shard ${SLURM_ARRAY_TASK_ID}/10) ==="
python src/llm_generator_pipeline/02_perturb_latin.py \
    --config   src/llm_generator_pipeline/config_hpc.yaml \
    --input    data/pipeline/01_sampled.jsonl \
    --output   data/pipeline/02_shard_${SLURM_ARRAY_TASK_ID}.jsonl \
    --shard    ${SLURM_ARRAY_TASK_ID} \
    --n_shards 10

echo "=== Shard ${SLURM_ARRAY_TASK_ID} complete ==="
