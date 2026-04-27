#!/bin/bash
#SBATCH --job-name=ft_llama3b_ca
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --gres=gpumem:20000m
#SBATCH --time=10:00:00
#SBATCH --mem-per-cpu=8192
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail
cd /cluster/home/tunguyen1/llmgap/reasoning-efficiency/experiments/proof_search
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate llmgap

python finetune.py \
    --model-name "meta-llama/Llama-3.2-3B-Instruct" \
    --task correct_answer \
    --out-dir out/checkpoints/Llama-3.2-3B-Instruct_correct_answer/final
