#!/bin/bash
#SBATCH --job-name=ir_pipeline
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=8G
#SBATCH --account=Education-EEMCS-Courses-DSAIT4095
#SBATCH --output=/scratch/vvjumle/logs/ir_pipeline_%j.out
#SBATCH --error=/scratch/vvjumle/logs/ir_pipeline_%j.err

set -e

REPO=/scratch/vvjumle/ir-pipeline
SCRATCH_DATA=/scratch/vvjumle/ir-pipeline/data/pipeline

mkdir -p /scratch/vvjumle/logs
mkdir -p $SCRATCH_DATA

module load 2025
module load miniconda3
module load cuda/12.9

# Create conda env from environment.yml if it doesn't exist yet
if ! conda env list | grep -q "^ir-pipeline "; then
    echo "=== Creating conda environment ir-pipeline ==="
    conda env create -f $REPO/environment.yml
fi

source activate ir-pipeline

# Make sure data/pipeline points to scratch (symlink)
ln -sfn $SCRATCH_DATA $REPO/data/pipeline

cd $REPO

echo "=== Stage 02: Latin phonetic variants ==="
python src/llm_generator_pipeline/02_perturb_latin.py \
    --config src/llm_generator_pipeline/config.yaml

echo "=== Stage 03: Script transliterations ==="
python src/llm_generator_pipeline/03_perturb_scripts.py \
    --config src/llm_generator_pipeline/config.yaml

echo "=== Stage 04: Merge + dataset split ==="
python src/llm_generator_pipeline/04_merge_wikidata.py \
    --config src/llm_generator_pipeline/config.yaml

echo "=== Pipeline complete. Output: $SCRATCH_DATA/04_dataset.jsonl ==="
