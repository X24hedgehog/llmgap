#!/bin/bash
#SBATCH --job-name=gen_next_sq
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --mem-per-cpu=8192
#SBATCH --output=slurm-next-subquestion-%j.out
#SBATCH --error=slurm-next-subquestion-%j.err

set -euo pipefail

cd /cluster/home/tunguyen1/llmgap/reasoning-efficiency/experiments/proof_search

eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate llmgap

python generate_next_subquestion.py \
  --input-path out/full_data.csv \
  --compact-out out/next_subquestion_compact.csv \
  --pairs-out out/next_subquestion_pairs.csv
