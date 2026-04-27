#!/bin/bash
#SBATCH --job-name=gen_csv
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --gres=gpumem:20000m
#SBATCH --time=01:00:00
#SBATCH --mem-per-cpu=8192
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

cd /cluster/home/tunguyen1/llmgap/reasoning-efficiency/experiments/proof_search

eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate llmgap

python generate_csv.py \
  --out-path out/full_data.csv \
  --nr-problems 2000 \
  --min-depth 3 \
  --max-depth 4 \
  --num-irrelevant-trees 4 \
  --profile full \
  --seed 42