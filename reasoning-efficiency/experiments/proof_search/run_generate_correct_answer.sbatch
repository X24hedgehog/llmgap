#!/bin/bash
#SBATCH --job-name=gen_correct_ans
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --mem-per-cpu=8192
#SBATCH --output=slurm-correct-answer-%j.out
#SBATCH --error=slurm-correct-answer-%j.err

set -euo pipefail

cd /cluster/home/tunguyen1/llmgap/reasoning-efficiency/experiments/proof_search

eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate llmgap

python generate_correct_answer.py \
  --input-path out/full_data.csv \
  --out-path out/correct_answer_pairs.csv
