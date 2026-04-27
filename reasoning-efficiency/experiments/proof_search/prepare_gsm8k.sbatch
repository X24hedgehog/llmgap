#!/bin/bash
#SBATCH --job-name=prepare_gsm8k
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --gres=gpumem:20000m
#SBATCH --time=20:00:00
#SBATCH --mem-per-cpu=8192
#SBATCH --output=logs/gsm8k/slurm-%j.out
#SBATCH --error=logs/gsm8k/slurm-%j.err

set -euo pipefail
cd /cluster/home/tunguyen1/llmgap/reasoning-efficiency/experiments/proof_search
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate llmgap

python prepare_gsm8k.py
