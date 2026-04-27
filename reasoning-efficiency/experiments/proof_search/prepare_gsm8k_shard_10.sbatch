#!/bin/bash
#SBATCH --job-name=gsm8k_shard_10
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --gres=gpumem:20000m
#SBATCH --time=20:00:00
#SBATCH --mem-per-cpu=8192
#SBATCH --output=logs/shard_10-%j.out
#SBATCH --error=logs/shard_10-%j.err

set -euo pipefail
cd /cluster/home/tunguyen1/llmgap/reasoning-efficiency/experiments/proof_search
mkdir -p logs
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate llmgap

# No --end: processes from 7911 to whatever the actual dataset size is
python prepare_gsm8k.py --start 7911
